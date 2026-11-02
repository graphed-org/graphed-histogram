"""``gh.plan`` gives back the nesting it was given, over any number of Sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import hist
import hist.graphed as hg
import pytest
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet
from graphed.checkpoint import resumable
from graphed.core.execution import SequentialRunner
from graphed_exec_local import ProcessPoolExecutor, ThreadExecutor

import graphed_histogram as gh
from graphed_histogram import _combine

RUNNERS: dict[str, Any] = {
    "sequential": SequentialRunner,
    "thread": lambda: ThreadExecutor(max_workers=2),
    "process": lambda: ProcessPoolExecutor(max_workers=2),
}


def _events(path: Path, offset: float) -> Any:
    ak.to_parquet(ak.Array({"x": [offset + i for i in range(8)]}), path)
    return from_parquet(
        path.stem, [str(path)], session=Session(AwkwardBackend()), steps_per_file=2
    )


def _hist(x: Any, weight: Any = None, *, variation_axis: bool = False) -> Any:
    h = hg.Hist.new.Reg(20, 0, 20, name="x").Weight()
    return h.fill(x, weight=weight, variation_axis=variation_axis)


def _varied(x: Any) -> Any:
    return graphed.vary(x * 0.5, "s", up=x * 0.75, down=x * 0.25)


@pytest.fixture
def fileset(tmp_path: Path) -> dict[str, Any]:
    dy, tt = _events(tmp_path / "dy.parquet", 0.0), _events(tmp_path / "tt.parquet", 10.0)
    return {
        "DY": {"zmass": _hist(dy.x, _varied(dy.x)), "n": _hist(dy.x)},
        "TT": {"zmass": _hist(tt.x, _varied(tt.x)), "n": _hist(tt.x)},
    }


def _by_hand(hists: dict[str, Any]) -> dict[str, Any]:
    plan = graphed.collate({name: gh.plan(h) for name, h in hists.items()})
    value = SequentialRunner().run(plan).value
    return {name: gh.unpack(v) for name, v in value.items()}


@pytest.mark.parametrize("runner", RUNNERS)
def test_a_fileset_comes_back_nested_and_equal_to_the_collate_form(
    fileset: dict[str, Any], runner: str
) -> None:
    plan = gh.plan(fileset)
    out = RUNNERS[runner]().run(plan).value
    want = _by_hand(fileset)
    assert len(plan.tasks) == 4
    assert list(out) == ["DY", "TT"] and list(out["DY"]) == ["zmass", "n"]
    assert list(out["TT"]["zmass"]) == ["nominal", "s_up", "s_down"]
    assert type(out["TT"]["n"]) is hist.Hist
    for name, outputs in want.items():
        assert out[name]["n"] == outputs["n"]
        for label, h in outputs["zmass"].items():
            assert out[name]["zmass"][label] == h


def test_one_session_nests_without_collating(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet", 0.0)
    out = SequentialRunner().run(gh.plan({"sr": {"x": _hist(ev.x)}, "cr": _hist(ev.x * 2)})).value
    assert out["sr"]["x"].sum().value == 8
    assert out["cr"].sum().value == 8


def test_an_axis_mode_output_stays_one_histogram(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet", 0.0)
    h = _hist(ev.x, _varied(ev.x), variation_axis=True)
    out = SequentialRunner().run(gh.plan({"sr": {"h": h}})).value
    assert sorted(out["sr"]["h"].axes[-1]) == ["nominal", "s_down", "s_up"]


def test_a_bare_histogram_is_its_histogram(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet", 0.0)
    assert type(SequentialRunner().run(gh.plan(_hist(ev.x))).value) is hist.Hist


def test_a_flat_mapping_comes_back_by_output_and_label(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet", 0.0)
    plan = gh.plan({"a": _hist(ev.x, _varied(ev.x)), "b": _hist(ev.x)})
    value = RUNNERS["process"]().run(plan).value
    assert list(value["a"]) == ["nominal", "s_up", "s_down"]
    assert type(value["b"]) is hist.Hist
    assert gh.unpack(value)["a"]["s_up"] is value["a"]["s_up"]


def test_a_mapping_keeps_keys_that_are_not_strings(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet", 0.0)
    assert list(SequentialRunner().run(gh.plan({2024: _hist(ev.x)})).value) == [2024]


def test_a_sequence_keeps_its_names(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet", 0.0)
    value = SequentialRunner().run(gh.plan([_hist(ev.x), _hist(ev.x * 2)])).value
    assert list(value) == ["0", "1"]


def test_unpack_is_idempotent_and_recurses() -> None:
    h = bh.Histogram(bh.axis.Regular(1, 0, 1))
    once = gh.unpack({"DY": {("zmass", "nominal"): h, ("zmass", "s_up"): h, "n": h}})
    assert once == {"DY": {"zmass": {"nominal": h, "s_up": h}, "n": h}}
    assert gh.unpack(once) == once


def test_label_listing_takes_the_nesting(fileset: dict[str, Any]) -> None:
    assert gh.label_listing(fileset)["TT"] == {
        "zmass": ["nominal", "s_up", "s_down"],
        "n": ["nominal"],
    }


def test_a_resumable_plan_returns_the_same_shape(fileset: dict[str, Any], tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    plan = gh.plan(fileset)
    first = SequentialRunner().run(resumable(plan, store)).value
    second = SequentialRunner().run(resumable(plan, store)).value
    assert first["DY"]["n"] == second["DY"]["n"] == _by_hand(fileset)["DY"]["n"]


def test_partitions_refuse_several_sessions(fileset: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="partitions="):
        gh.plan(fileset, partitions=[])


def test_two_paths_to_one_name_refuse(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet", 0.0)
    with pytest.raises(ValueError, match="same '/'-joined name"):
        gh.plan({"a/b": _hist(ev.x), "a": {"b": _hist(ev.x)}})


def test_a_histserv_output_refuses_nesting(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet", 0.0)
    h = _hist(ev.x)
    h.hold_partials(histserv=object())
    with pytest.raises(TypeError, match="histserv-backed output 'a/b'"):
        gh.plan({"a": {"b": h}})


def test_any_iterable_keeps_its_names(tmp_path: Path) -> None:
    h = _hist(_events(tmp_path / "a.parquet", 0.0).x)
    assert SequentialRunner().run(gh.plan({"h": h}.values())).value.keys() == {"0"}


def test_something_else_refuses_by_name() -> None:
    with pytest.raises(TypeError, match="a/b is int"):
        gh.plan({"a": {"b": 1}})
    with pytest.raises(TypeError, match="the argument is int"):
        gh.plan(1)


def test_a_dataset_with_no_tasks_is_absent(fileset: dict[str, Any]) -> None:
    nested: Any = gh.plan(fileset).finalize
    h = bh.Histogram(bh.axis.Regular(1, 0, 1))
    assert nested({"DY/zmass": {"DY/zmass": h}}) == {"DY": {"zmass": h}}


def test_an_unfilled_histogram_keeps_its_refusal() -> None:
    with pytest.raises(ValueError, match="at least one staged fill"):
        gh.plan({"a": hg.Hist.new.Reg(2, 0, 1, name="x").Weight()})


def test_nothing_keeps_its_refusal() -> None:
    with pytest.raises(ValueError, match="at least one histogram"):
        gh.plan({})


def test_the_reduce_describes_its_outputs(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet", 0.0)
    plan: Any = gh.plan({"v": _hist(ev.x, _varied(ev.x)), "n": _hist(ev.x)})
    assert plan.process.reduce.output_summaries() == [
        "v (Hist, 20 bins, 3 labels)",
        "n (Hist, 20 bins)",
    ]
    one: Any = _hist(ev.x).plan()
    assert one.process.reduce.output_summaries() == ["Hist, 20 bins"]


def test_each_session_describes_its_outputs_by_path(fileset: dict[str, Any]) -> None:
    plan: Any = gh.plan(fileset)
    plans = plan.process.processes.values()
    assert [line for p in plans for line in p.reduce.output_summaries()] == [
        "DY/zmass (Hist, 20 bins, 3 labels)",
        "DY/n (Hist, 20 bins)",
        "TT/zmass (Hist, 20 bins, 3 labels)",
        "TT/n (Hist, 20 bins)",
    ]


def test_a_summary_counts_without_building_the_histogram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ev = _events(tmp_path / "a.parquet", 0.0)
    grown = hg.Hist.new.StrCat([], growth=True).Reg(10, 0, 1).Double().fill("a", ev.x)
    plan: Any = gh.plan({"g": grown})
    reduce = plan.process.reduce
    assert reduce.output_summaries() == ["g (Hist, 10+ bins)"]
    monkeypatch.setattr(_combine, "zero_of", None)
    assert reduce.output_summaries() == ["g (Hist, 10+ bins)"]
