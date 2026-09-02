"""Shared fixtures for the m50 variation-axis (§6.2) anchors.

Self-contained on purpose: the sibling m48/m49 helper dirs are NOT on `pythonpath`, and this
tree's helper basename is `m50_`-prefixed so the several milestone dirs collected in ONE pytest
process cannot silently bind each other's module under prepend import mode. Everything here is
built from `graphed`, `graphed.awkward`, `graphed_corpus` (vendored, on `pythonpath`) and
`boost_histogram` — never an `importorskip`, which would skip the milestone's headline gate.

The corpus program is per-OBJECT (jagged `Jet.pt`) with a per-EVENT weight factor (§6.2/AX-1): the
broadcast seam (§6.1d) is a no-op on flat per-event data, so a per-event fixture would admit an
implementation that hands the loop node raw columns and length-mismatches only on the per-object
analyses axis mode exists for.

The oracle for every axis-vs-sibling comparison is the SIBLING-mode decomposition — the frozen,
eager-referenced m48/m49 lowering — sliced label-by-label; axis mode reproduces it or reds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
from graphed import Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak, gnano
from graphed.core import Partition
from graphed.core.execution import SequentialRunner, WorkerResources
from graphed_corpus import make_events

import graphed_histogram as gh

#: small enough to keep the suite fast, large enough that every jet selection below is non-empty
EVENTS = make_events(n_events=300, seed=5050)

#: the per-fill axis-mode OPT-IN spelling, chosen at m50 freeze (§6.2 leaves it to the test-author,
#: binding only that it is expressed per-`fill()` and remembered by the histogram). The implementer
#: inherits it.
AXIS_MODE_KW = "variation_axis"

#: the plan-level `{output: [labels]}` listing verb, chosen at m50 freeze (§9.1)
LABEL_LISTING = "label_listing"


@dataclass
class CountingSource:
    """A `graphed.write.PartitionedSource` over in-memory events, counting partition reads."""

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


def partitioned() -> tuple[Session, Any, CountingSource]:
    """A session whose single source is partitioned — what `aggregate_plan` binds a plan to."""
    session = Session(AwkwardBackend())
    data = CountingSource(EVENTS)
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    root = session.source("events", form=form, data=data)
    return session, gnano.events(root), data


def in_memory() -> tuple[Session, Any]:
    """The record-time fixture: no plan, so no partitioned source."""
    session = Session(AwkwardBackend())
    return session, gnano.events(from_awkward(session, "events", EVENTS))


# ---- storage-pinned histogram factories -----------------------------------------------------
def weighted(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> gh.boost.Histogram:
    return gh.boost.Histogram(bh.axis.Regular(bins, lo, hi), storage=bh.storage.Weight())


def weighted_mean(bins: int = 20, lo: float = 0.0, hi: float = 200.0) -> gh.boost.Histogram:
    """`sample=` needs a Mean/WeightedMean storage: bh rejects it on Double()/Weight()."""
    return gh.boost.Histogram(bh.axis.Regular(bins, lo, hi), storage=bh.storage.WeightedMean())


# ---- program builders ------------------------------------------------------------------------
def shift_jets(source: Any, scale: float) -> Any:
    """§5.1's shift: the JES varies the jets record so each universe re-derives its own value."""
    jets = source.Jet
    return gak.with_field(jets, jets.pt * scale, "pt")


def weight_family(source: Any) -> Any:
    """A per-EVENT weight factor varied up/down (the `W` labels — collapsible into the loop)."""
    factor = source.MET.pt * 0.01
    return graphed.vary(factor, "wgt", up=factor * 1.2, down=factor * 0.8)


def fill_weight_program(mode_axis: bool, *, source: Any) -> gh.boost.Histogram:
    """Per-object value (`Jet.pt`), one per-event varied weight factor — weight labels ONLY."""
    h = weighted()
    h.fill(source.Jet.pt, weight=[weight_family(source)], **({AXIS_MODE_KW: True} if mode_axis else {}))
    return h


def fill_mixed_program(mode_axis: bool, *, source: Any) -> gh.boost.Histogram:
    """A shift (`jes`, in `S`), a weight (`wgt`, in `W`) and a sample-only variation (`smp`, in `S`)
    on a WeightedMean storage with per-label sample values that DIFFER (§6.1b): the one program
    that witnesses the `S`/`W` split and the `1 + |S|` axis arity."""
    shifted = graphed.vary(
        source, "jes", Jet={"up": shift_jets(source, 1.05), "down": shift_jets(source, 0.95)}
    )
    btag = shifted.Jet.btag
    h = weighted_mean()
    h.fill(
        shifted.Jet.pt,
        weight=[weight_family(shifted)],
        sample=graphed.vary(btag, "smp", up=btag * 1.1, down=btag * 0.9),
        **({AXIS_MODE_KW: True} if mode_axis else {}),
    )
    return h


# ---- execution + slot inspection -------------------------------------------------------------
def execute(histograms: dict[str, gh.boost.Histogram], steps: int = 3) -> dict[Any, bh.Histogram]:
    """Run a group plan and return the flat slot-keyed combine payload (§6.1c)."""
    return dict(SequentialRunner().run(gh.plan(histograms, steps_per_file=steps)).value)


def node_chash(session: Session, node: Any) -> str:
    """The External `content_hash` a recorded fill node carries (§6.2's per-fill carrier)."""
    stored = next(n for n in session._store.nodes() if n["id"] == node.node_id)
    return str(stored["descriptor"]["content_hash"])


# ---- bin-for-bin oracle helpers --------------------------------------------------------------
def var_index(hist: bh.Histogram) -> int:
    """The variation axis's position, read per-axis from `__dict__` — never `h.axes.name`, which
    raises `AttributeError` unless EVERY axis carries a name (§6.2 i-bis)."""
    return next(i for i, a in enumerate(hist.axes) if a.__dict__.get("name") == "variation")


def slice_label(hist: bh.Histogram, label: str) -> bh.Histogram:
    """A pure-bh positional slice of the variation axis — the accessor-independent oracle for the
    equality/scaling anchors. The named-dict form `h[{"variation": label}]` is a `TypeError` on a
    bare `bh.Histogram` (§6.2 i-bis); only the positional index works."""
    return hist[{var_index(hist): bh.loc(label)}]


def views_equal(a: bh.Histogram, b: bh.Histogram) -> bool:
    """Bin-for-bin equality including flow, NaN-tolerant per storage field (empty WeightedMean bins
    carry NaN means)."""
    va, vb = a.view(flow=True), b.view(flow=True)
    if va.dtype.names is None:
        return bool(np.allclose(np.asarray(va), np.asarray(vb), rtol=1e-12, equal_nan=True))
    return all(
        np.allclose(np.asarray(va[name]), np.asarray(vb[name]), rtol=1e-12, equal_nan=True)
        for name in va.dtype.names
    )
