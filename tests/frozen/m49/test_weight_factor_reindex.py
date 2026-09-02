"""m49/F7 — §6.1d: a WEIGHT FACTOR from an ancestor context is re-indexed, per label, like a value.

m48 froze the ancestor re-indexing over axis values and over `sample=`; the weight factors go
through the same `graphed.reindex_to` call and nothing frozen witnesses it, so dropping it there
leaves m48's suite green while every varied weighted fill silently weights the wrong rows.

The fixture puts the offending operand where only a factor can sit: axis 0 is at the SELECTION's
row space, the factor at the PARENT's. The mask is varied, so each label selects its own row set
and re-indexing by nominal's mask is a different answer for every non-nominal label.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import graphed
import numpy as np
from graphed.awkward import gak, gnano
from graphed.core.execution import SequentialRunner
from m49_hist_fixtures import JES, TOY_EVENTS, eager_jes_mask, eager_weighted, in_memory, partitioned

import graphed_histogram as gh

N_JETS = 2
STEPS = 3


def _selected(events_ctx: Any) -> Any:
    """A VARIED mask derivation: each label selects its own rows."""
    shifted = graphed.vary(
        events_ctx,
        "jes",
        Jet={
            "up": gak.with_field(events_ctx.Jet, events_ctx.Jet.pt * JES["jes_up"], "pt"),
            "down": gak.with_field(events_ctx.Jet, events_ctx.Jet.pt * JES["jes_down"], "pt"),
        },
    )
    return shifted[gak.num(shifted.Jet[shifted.Jet.pt > 25.0], axis=1) >= N_JETS]


def _hist() -> gh.boost.Histogram:
    return gh.boost.Histogram(*eager_weighted().axes, storage=eager_weighted().storage_type())


def _weighted_fill(events_ctx: Any) -> gh.boost.Histogram:
    """Axis value at the SELECTION; weight factor at the PARENT — an ancestor-context FACTOR."""
    selection = _selected(events_ctx)
    hist = _hist()
    hist.fill(selection.MET.pt, weight=[events_ctx.MET.pt * 0.01])
    return hist


def _reference(label: str) -> Any:
    mask = eager_jes_mask(TOY_EVENTS, label, N_JETS)
    rows = ak.to_numpy(TOY_EVENTS.MET.pt[mask])
    want = eager_weighted()
    want.fill(rows, weight=rows * 0.01)
    return want


def test_the_row_sets_really_differ_so_the_re_indexing_is_observable() -> None:
    """The instrument: were every label's mask the same rows, weighting by nominal's slice would
    be indistinguishable from weighting per label."""
    rows = {label: int(ak.sum(eager_jes_mask(TOY_EVENTS, label, N_JETS))) for label in JES}
    assert len(set(rows.values())) == len(rows), rows
    assert all(0 < n < len(TOY_EVENTS) for n in rows.values())


def test_each_labels_weight_factor_is_re_indexed_by_that_labels_own_mask() -> None:
    session, root = in_memory(TOY_EVENTS)
    per_label = gh.fill_nodes_by_label(_weighted_fill(gnano.events(root)))
    assert set(per_label) == set(JES)
    for label, node in per_label.items():
        want = _reference(label)
        got = session.materialize(node)
        for field in ("value", "variance"):
            assert np.allclose(got.view(flow=True)[field], want.view(flow=True)[field], rtol=1e-12), (
                f"{label}: the weight factor was not re-indexed to this label's rows"
            )


def test_the_plan_path_agrees_with_the_per_label_reference_too() -> None:
    """The factor is re-indexed at RECORD time, so both evaluation routes must show it."""
    _session, source, _data = partitioned(TOY_EVENTS)
    hist = _weighted_fill(gnano.events(source))
    result = gh.unpack(SequentialRunner().run(gh.plan({"met": hist}, steps_per_file=STEPS)).value)
    assert sorted(result["met"]) == sorted(JES)
    for label in JES:
        want = _reference(label)
        assert np.allclose(
            result["met"][label].view(flow=True)["value"],
            want.view(flow=True)["value"],
            rtol=1e-12,
        )
        assert want.view(flow=True)["value"].sum() > 0
