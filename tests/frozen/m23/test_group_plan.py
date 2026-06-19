"""M23: ``graphed_histogram.plan(histograms)`` aggregates SEVERAL histograms sharing a source in ONE
pass — their fills compile into one IR, so a sub-graph feeding multiple histograms is read+evaluated
ONCE, not once per histogram. A NEW frozen file (test-authoring deliverable); the existing m23 suite
is unchanged.

The bug this guards: building a plan PER histogram recomputes a shared sub-graph N times for an
N-output query (e.g. the ADL trijet query's pT + b-tag histograms), an N-fold read+compute blowup
vs dask's ``compute(dict_of_hists)`` which evaluates the shared graph once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import boost_histogram as bh
import numpy as np
import pytest
from graphed import Array, Session
from graphed_core import Partition
from graphed_core.execution import SequentialRunner
from graphed_numpy import NumpyBackend
from graphed_numpy.forms import NumpyForm

import graphed_histogram as gh

DATA = np.random.default_rng(0).normal(5.0, 2.0, 1000)


@dataclass
class ChunkedSource:
    """A PartitionedSource over an in-memory array, counting partition reads (the efficiency witness)."""

    data: np.ndarray
    part_reads: list = field(default_factory=list)

    def __call__(self) -> np.ndarray:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://chunks", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition, columns, resources) -> np.ndarray:  # type: ignore[no-untyped-def]
        part = partition.resolve(len(self.data))
        self.part_reads.append((part.entry_start, part.entry_stop))
        return self.data[part.entry_start : part.entry_stop]


def _source(s: Session) -> tuple[Array, ChunkedSource]:
    src = ChunkedSource(DATA)
    return s.source("x", form=NumpyForm(DATA.dtype, shape=(None,)), data=src), src


def _two_hists(x: Array) -> tuple[gh.Histogram, gh.Histogram]:
    # two histograms sharing the source read x (and the x*0.5+1 sub-expression)
    hi = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 10.0), storage=bh.storage.Int64()).fill(x)
    lo = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 10.0), storage=bh.storage.Int64()).fill(x * 0.5 + 1.0)
    return hi, lo


def _eager(hi_data: np.ndarray, lo_data: np.ndarray) -> tuple[bh.Histogram, bh.Histogram]:
    eh = bh.Histogram(bh.axis.Regular(20, 0.0, 10.0), storage=bh.storage.Int64())
    el = bh.Histogram(bh.axis.Regular(20, 0.0, 10.0), storage=bh.storage.Int64())
    eh.fill(hi_data)
    el.fill(lo_data)
    return eh, el


def test_group_plan_reads_the_source_once_not_once_per_histogram() -> None:
    s = Session(NumpyBackend())
    x, src = _source(s)
    hi, lo = _two_hists(x)
    out = SequentialRunner().run(gh.plan({"hi": hi, "lo": lo}, steps_per_file=4)).value
    eh, el = _eager(DATA, DATA * 0.5 + 1.0)
    assert np.array_equal(out["hi"].values(), eh.values())
    assert np.array_equal(out["lo"].values(), el.values())
    # the witness: 4 partitions read ONCE each (8 would mean a per-histogram re-read)
    assert len(src.part_reads) == 4


def test_group_plan_matches_per_histogram_plans() -> None:
    s = Session(NumpyBackend())
    x, _ = _source(s)
    hi, lo = _two_hists(x)
    grouped = SequentialRunner().run(gh.plan([hi, lo], steps_per_file=3)).value  # sequence -> "0","1"
    assert np.array_equal(
        grouped["0"].values(), SequentialRunner().run(hi.plan(steps_per_file=3)).value.values()
    )
    assert np.array_equal(
        grouped["1"].values(), SequentialRunner().run(lo.plan(steps_per_file=3)).value.values()
    )


def test_group_plan_empty_and_validation() -> None:
    s = Session(NumpyBackend())
    x, _ = _source(s)
    hi, _lo = _two_hists(x)
    # empty histogram (no axes filled) zero state is well-formed: a 0-task plan returns the zeros
    zero = SequentialRunner().run(gh.plan({"hi": hi}, steps_per_file=0)).value
    assert zero["hi"].sum() == 0
    with pytest.raises(ValueError, match="at least one histogram"):
        gh.plan({})
    blank = gh.boost.Histogram(bh.axis.Regular(5, 0, 1))  # nothing staged
    with pytest.raises(ValueError, match="at least one staged fill"):
        gh.plan([blank])
