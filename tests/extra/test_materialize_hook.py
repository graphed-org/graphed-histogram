"""`graphed.materialize` gives a deferred histogram's result, as a runner does."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import pytest
from graphed import GraphedError, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward
from graphed.awkward.io import parquet_write
from graphed.core import Partition
from graphed.core.execution import SequentialRunner
from graphed.errors import StageError

import graphed_histogram as gh

hist = pytest.importorskip("hist")
hist_graphed = pytest.importorskip("hist.graphed")


@dataclass
class _Chunks:
    """A partitioned source that counts its reads."""

    chunks: tuple[ak.Array, ...]
    reads: list[Partition] = field(default_factory=list)

    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return tuple(Partition.blind(f"f{i}", "t", 0, 1) for i in range(len(self.chunks)))

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> ak.Array:
        self.reads.append(partition)
        return self.chunks[int(partition.uri[1:])]

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must stay unused")


def _source(session: Session) -> tuple[Any, _Chunks]:
    one = ak.Array(
        {"x": [1.0, 2.0], "y": [1.0, 1.0], "j": [[1.0], [2.0]], "k": [[1.0, 2.0], [1.0]]}
    )
    two = ak.Array({"x": [3.0], "y": [1.0], "j": [[1.0]], "k": [[1.0, 2.0]]})
    data = _Chunks((one, two))
    form = AwkwardForm(ak.Array(data.chunks[0].layout.to_typetracer(forget_length=True)))
    return session.source("chunks", form=form, data=data), data


def _axis() -> bh.axis.Regular:
    return bh.axis.Regular(4, 0.0, 4.0)


def test_a_hist_comes_back_filled_as_a_runner_fills_it() -> None:
    source, _ = _source(Session(AwkwardBackend()))
    h = hist_graphed.Hist.new.Reg(4, 0.0, 4.0, name="x").Double().fill(source.x)
    filled = graphed.materialize(h)
    expected = SequentialRunner().run(h.plan()).value
    assert type(filled) is type(expected)
    assert not isinstance(filled, gh.Histogram)
    assert h.staged_fills()
    assert filled == expected
    assert filled.sum() == 3.0


def test_a_histogram_with_no_fills_comes_back_empty_without_a_warning() -> None:
    h = hist_graphed.Hist.new.Reg(4, 0.0, 4.0, name="x").Double()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filled = graphed.materialize(h)
    assert type(filled) is hist.Hist
    assert filled.axes == h.axes
    assert filled.sum() == 0.0
    assert not isinstance(filled, gh.Histogram)


def test_materialize_reads_no_storage_of_the_deferred_histogram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _source(Session(AwkwardBackend()))
    h = hist_graphed.Hist.new.Reg(4, 0.0, 4.0, name="x").Double().fill(source.x)
    unfilled = hist_graphed.Hist.new.Reg(4, 0.0, 4.0, name="x").Double()
    real_view, real_init = bh.Histogram.view, hist.Hist.__init__

    def view(self: Any, *args: Any, **kwargs: Any) -> Any:
        assert not isinstance(self, gh.Histogram)
        return real_view(self, *args, **kwargs)

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        assert not (args and isinstance(args[0], gh.Histogram))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(bh.Histogram, "view", view)
    monkeypatch.setattr(hist.Hist, "__init__", init)
    filled, empty = graphed.materialize([h, unfilled])
    assert (filled.sum(), empty.sum()) == (3.0, 0.0)


def test_a_failing_fill_raises_at_the_users_line() -> None:
    source, _ = _source(Session(AwkwardBackend()))
    bad = source.j + source.k
    with pytest.raises(StageError) as caught:
        graphed.materialize(gh.Histogram(_axis()).fill(bad))
    assert caught.value.user_frame.filename == __file__


def test_a_boost_histogram_keeps_its_class() -> None:
    source, _ = _source(Session(AwkwardBackend()))
    filled = graphed.materialize(gh.Histogram(_axis()).fill(source.x))
    assert type(filled) is bh.Histogram
    assert filled.values().tolist() == [0.0, 1.0, 1.0, 1.0]


def test_a_varied_histogram_comes_back_per_label() -> None:
    source, _ = _source(Session(AwkwardBackend()))
    varied = graphed.vary(source.x, "s", up=source.x + 1.0, down=source.x - 1.0)
    h = gh.Histogram(_axis()).fill(varied)
    filled = graphed.materialize(h)
    assert filled == gh.unpack(SequentialRunner().run(gh.plan({"h": h})).value)["h"]
    assert len(filled) == 3


def test_histograms_arrays_and_writes_share_one_read_in_their_nesting(tmp_path: Path) -> None:
    source, data = _source(Session(AwkwardBackend()))
    first = gh.Histogram(_axis()).fill(source.x)
    second = gh.Histogram(_axis()).fill(source.y)
    write = parquet_write(source.x, str(tmp_path), name=lambda p: f"{p.uri}.parquet")
    done = graphed.materialize({"h": (first, [second]), "x": source.x, "w": write, "n": 1})
    assert done["h"][0].values().tolist() == [0.0, 1.0, 1.0, 1.0]
    assert done["h"][1][0].values().tolist() == [0.0, 3.0, 0.0, 0.0]
    assert ak.to_list(done["x"]) == [1.0, 2.0, 3.0]
    assert done["w"] == [str(tmp_path / "f0.parquet"), str(tmp_path / "f1.parquet")]
    assert done["n"] == 1
    assert len(data.reads) == 2


def test_an_in_memory_source_fills_though_it_has_no_plan() -> None:
    session = Session(AwkwardBackend())
    source = from_awkward("m", ak.Array({"x": [1.0, 2.0]}), session=session)
    h = gh.Histogram(_axis()).fill(source.x)
    with pytest.raises(TypeError, match="exactly one partitioned source"):
        h.plan()
    assert graphed.materialize(h).sum() == 2.0


def test_histserv_backed_is_refused_naming_the_plan() -> None:
    source, _ = _source(Session(AwkwardBackend()))
    h = gh.Histogram(_axis()).fill(source.x)
    h.hold_partials(histserv=object())
    with pytest.raises(GraphedError, match="runs no servers"):
        graphed.materialize(h)


def test_a_filled_in_memory_histogram_passes_through() -> None:
    plain = bh.Histogram(_axis())
    assert graphed.materialize([plain])[0] is plain
