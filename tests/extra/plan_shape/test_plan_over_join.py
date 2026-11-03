"""``h.plan()`` and ``gh.plan()`` over a join give each output the same as ``join_plan``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import graphed
import hist
import hist.graphed as hg
import numpy as np
import pytest
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet
from graphed.core.execution import SequentialRunner
from graphed.errors import GraphedError
from graphed_executors.submit import SubmitRunner, ThreadBackend

import graphed_histogram as gh


def _submit(plan: Any) -> Any:
    with SubmitRunner(ThreadBackend()) as runner:
        return runner.run(plan)


RUNNERS: dict[str, Any] = {"sequential": SequentialRunner().run, "submit": _submit}


def _source(path: Path, session: Session, rows: dict[str, Any]) -> Any:
    ak.to_parquet(ak.Array(rows), path)
    return from_parquet(path.stem, [str(path)], session=session, steps_per_file=2)


def _sides(tmp_path: Path) -> tuple[Any, Any]:
    session = Session(AwkwardBackend())
    event: Any = np.arange(10) + 2**33
    left = _source(tmp_path / "l.parquet", session, {"event": event, "x": np.arange(10.0)})
    right = _source(tmp_path / "r.parquet", session, {"event": event, "y": np.arange(10.0) * 2})
    return left, right


def _joined(tmp_path: Path, how: str) -> Any:
    left, right = _sides(tmp_path)
    if how == "positional":
        return graphed.join(left[["x"]], right[["y"]], how=how)
    return graphed.join(left, right, on=["event"], how=how)


def _book(kind: Any) -> Any:
    return kind.new.Reg(30, 0, 30, name="v").Weight()


def _hist(values: Any) -> Any:
    return _book(hg.Hist).fill(values)


def _assert_same_as_join_plan(value: Any, output: Any) -> None:
    rows = SequentialRunner().run(graphed.join_plan(output)).value
    alone = _book(hist.Hist).fill(ak.to_numpy(ak.concatenate(rows)))
    assert alone.sum().value == 10
    assert np.array_equal(value.values(flow=True), alone.values(flow=True))
    assert np.array_equal(value.variances(flow=True), alone.variances(flow=True))


@pytest.mark.parametrize("how", ["positional", "inner"])
@pytest.mark.parametrize("runner", RUNNERS)
def test_a_histogram_over_a_join(tmp_path: Path, how: str, runner: str) -> None:
    joined = _joined(tmp_path, how)
    total = joined.x + joined.y
    _assert_same_as_join_plan(RUNNERS[runner](_hist(total).plan()).value, total)


@pytest.mark.parametrize("how", ["positional", "inner"])
@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("shape", ["flat", "nested"])
def test_several_histograms_over_a_join(tmp_path: Path, how: str, runner: str, shape: str) -> None:
    joined = _joined(tmp_path, how)
    x, total = joined.x, joined.x + joined.y
    if shape == "flat":
        plan = gh.plan({"x": _hist(x), "total": _hist(total)})
        value = gh.unpack(RUNNERS[runner](plan).value)
    else:
        plan = gh.plan({"x": _hist(x), "more": {"total": _hist(total)}})
        value = RUNNERS[runner](plan).value
        value["total"] = value.pop("more")["total"]
    _assert_same_as_join_plan(value["x"], x)
    _assert_same_as_join_plan(value["total"], total)


def test_histserv_over_a_join_is_refused_with_the_way_through(tmp_path: Path) -> None:
    joined = _joined(tmp_path, "positional")
    backed = _hist(joined.x)
    backed.hold_partials(histserv=object())
    with pytest.raises(GraphedError, match="without histserv"):
        gh.plan({"x": backed})


def test_a_bare_histogram_over_a_join(tmp_path: Path) -> None:
    joined = _joined(tmp_path, "inner")
    _assert_same_as_join_plan(SequentialRunner().run(gh.plan(_hist(joined.x))).value, joined.x)


def test_a_join_beside_another_session_is_refused(tmp_path: Path) -> None:
    joined = _joined(tmp_path, "inner")
    other = _source(tmp_path / "o.parquet", Session(AwkwardBackend()), {"x": np.arange(3.0)})
    with pytest.raises(TypeError, match="runs in stages"):
        gh.plan({"joined": {"x": _hist(joined.x)}, "other": {"x": _hist(other.x)}})


def test_an_output_short_of_the_join_is_refused(tmp_path: Path) -> None:
    left, right = _sides(tmp_path)
    joined = graphed.join(left, right, on=["event"], how="inner")
    with pytest.raises(TypeError, match="reads a source directly"):
        gh.plan({"left": _hist(left.x), "joined": _hist(joined.y)})
