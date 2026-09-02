"""Shared fixtures for the m48 fill-lowering anchors.

One synthetic dataset for the whole tree (the vendored corpus generator, so the events carry the
jagged Jet/Photon/Muon collections and the per-event MET the §6.1d per-object anchors need), one
read-counting `graphed.write.PartitionedSource` over it, and the storage-pinned histogram
factories the `sample=` anchors require: boost-histogram 1.8.0 rejects `sample=` on `Double()` AND
on `Weight()` while the evaluator passes `sample` straight to `h.fill`, so a default-storage
fixture records cleanly and dies at EVALUATION.

The manual reference builders live here too: §6.1d's assertions are elementwise against a value
computed eagerly from the same array, never against the fill's own machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import awkward as ak
import boost_histogram as bh
import numpy as np
from graphed import Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward
from graphed.core import Partition
from graphed.core.execution import WorkerResources
from graphed_corpus import make_events

import graphed_histogram as gh

#: small enough to keep the suite fast, large enough that every selection below is non-empty
EVENTS = make_events(n_events=400, seed=1948)


@dataclass
class ChunkedEvents:
    """A `PartitionedSource` over an in-memory awkward array, counting partition reads."""

    data: ak.Array
    part_reads: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://events", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: WorkerResources) -> ak.Array:
        part = partition.resolve(len(self.data))
        self.part_reads.append((part.entry_start, part.entry_stop))
        return self.data[part.entry_start : part.entry_stop]


def partitioned_events() -> tuple[Session, Any, ChunkedEvents]:
    """A session whose single source is partitioned — what `aggregate_plan` binds a plan to."""
    session = Session(AwkwardBackend())
    data = ChunkedEvents(EVENTS)
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    return session, session.source("events", form=form, data=data), data


def in_memory_events() -> tuple[Session, Any]:
    """The record-time and `Session.materialize` fixture: no plan, so no partitioned source."""
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "events", EVENTS)


# ---- storage-pinned histogram factories -----------------------------------------------------
def weighted(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> gh.boost.Histogram:
    return gh.boost.Histogram(bh.axis.Regular(bins, lo, hi), storage=bh.storage.Weight())


def counts(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> gh.boost.Histogram:
    return gh.boost.Histogram(bh.axis.Regular(bins, lo, hi), storage=bh.storage.Int64())


def sampled(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> gh.boost.Histogram:
    """`sample=` needs a Mean/WeightedMean storage: bh 1.8.0 raises
    `TypeError: Keyword(s) sample not expected` on Double() and on Weight()."""
    return gh.boost.Histogram(bh.axis.Regular(bins, lo, hi), storage=bh.storage.WeightedMean())


def sampled_2d(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> gh.boost.Histogram:
    axes = (bh.axis.Regular(bins, lo, hi), bh.axis.Regular(bins, lo, hi))
    return gh.boost.Histogram(*axes, storage=bh.storage.WeightedMean())


def weighted_2d(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> gh.boost.Histogram:
    axes = (bh.axis.Regular(bins, lo, hi), bh.axis.Regular(bins, lo, hi))
    return gh.boost.Histogram(*axes, storage=bh.storage.Weight())


# ---- eager reference builders ----------------------------------------------------------------
def eager_weighted(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> bh.Histogram:
    """A plain boost histogram matching `weighted()` — the oracle side of every comparison."""
    return bh.Histogram(bh.axis.Regular(bins, lo, hi), storage=bh.storage.Weight())


def eager_counts(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> bh.Histogram:
    return bh.Histogram(bh.axis.Regular(bins, lo, hi), storage=bh.storage.Int64())


def eager_sampled(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> bh.Histogram:
    return bh.Histogram(bh.axis.Regular(bins, lo, hi), storage=bh.storage.WeightedMean())


def eager_sampled_2d(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> bh.Histogram:
    axes = (bh.axis.Regular(bins, lo, hi), bh.axis.Regular(bins, lo, hi))
    return bh.Histogram(*axes, storage=bh.storage.WeightedMean())


def eager_weighted_2d(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> bh.Histogram:
    axes = (bh.axis.Regular(bins, lo, hi), bh.axis.Regular(bins, lo, hi))
    return bh.Histogram(*axes, storage=bh.storage.Weight())


def flat(values: Any) -> np.ndarray:
    """The evaluator's own per-input flatten (`boost.py::_flat`), for manual references."""
    return ak.to_numpy(ak.flatten(values, axis=None))


def broadcast_reference(value: Any, factor: Any) -> np.ndarray:
    """§6.1d's manual-broadcast reference: the per-event factor broadcast to the per-object value's
    structure BEFORE either is flattened — what `graphed.broadcast_like` records."""
    broadcast, _ = ak.broadcast_arrays(factor, value)
    return flat(broadcast)


def reindex_reference(value: Any, mask: Any) -> Any:
    """§6.1d link kind (1): the ancestor value re-indexed by THAT label's own mask."""
    return value[mask]
