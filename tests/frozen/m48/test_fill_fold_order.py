"""m48/H10 — §6.1d: the fill folds its operands LEFT in a bound order.

§2.4 binds only a binary combination; a fill is multi-way, and an unbound order would let two
conforming implementations produce different label orders for one program — a determinism-gate
difference (§3.2) and a different `_GroupReduce` layout (§6.1c). The bound order is: axis values in
ARGUMENT order, then the AMBIENT weight, then explicit `weight=[...]` factors in LIST order, then
`sample=` LAST.

`sample=` is a first-class fourth label source. Today's `fill` appends it to the input list with no
type check, so a `Varied` sample falls straight into `record_external` and dies on `.node_id`;
§6.1d makes it ACCEPTED and expanded like any other operand.

**This anchor is written as a RECORD-TIME assertion**, as §10/m48 requires it to say: the fold
order is read off the per-label fill-node accessor and the plan is never run. One evaluation rides
along so the storage pin stays load-bearing rather than decorative — bh 1.8.0 rejects `sample=` on
`Double()` AND `Weight()`, so a default-storage fixture would record cleanly and die at evaluation.

Both varied axis values are PER-EVENT: the evaluator flattens each axis independently and bh
requires equal lengths across axes.
"""

from __future__ import annotations

from typing import Any

import graphed
import numpy as np
from graphed.awkward import gnano
from vary_hist_fixtures import in_memory_events, sampled_2d

import graphed_histogram as gh

#: one variation NAME per operand slot, so the fold order is readable off the label order alone
FOLD_ORDER = ("nominal", "ax0_up", "ax1_up", "pu_up", "f0_up", "f1_up", "smp_up")


def _four_way() -> tuple[Any, gh.boost.Histogram]:
    session, root = in_memory_events()
    events = gnano.events(root)
    ambient = events.MET.pt * 0.01
    ctx = graphed.vary(events, "pu", ambient, is_weight=True, up=ambient * 1.1)

    axis0 = graphed.vary(ctx.MET.pt, "ax0", up=ctx.MET.pt * 1.01)
    axis1 = graphed.vary(ctx.MET.pt * 0.5, "ax1", up=ctx.MET.pt * 0.55)
    factor0 = graphed.vary(ctx.MET.pt * 0.0 + 1.0, "f0", up=ctx.MET.pt * 0.0 + 1.1)
    factor1 = graphed.vary(ctx.MET.pt * 0.0 + 2.0, "f1", up=ctx.MET.pt * 0.0 + 2.2)
    sample = graphed.vary(ctx.MET.phi, "smp", up=ctx.MET.phi * 1.5)

    h = sampled_2d()
    h.fill(axis0, axis1, weight=[factor0, factor1], sample=sample)
    return session, h


def test_the_four_operand_kinds_fold_in_the_bound_order() -> None:
    _session, h = _four_way()
    assert tuple(gh.fill_nodes_by_label(h)) == FOLD_ORDER


def test_a_varied_sample_is_accepted_and_expanded_rather_than_raising() -> None:
    """Its label reaches the fill, which is only possible if `sample=` went through the same
    expansion as the axis values instead of being appended unchecked."""
    _session, h = _four_way()
    labels = gh.fill_nodes_by_label(h)
    assert "smp_up" in labels
    assert labels["smp_up"].node_id != labels["nominal"].node_id


def test_each_operand_slot_contributes_its_own_label_exactly_once() -> None:
    """The instrument for the order assertion: an implementation that dropped one operand kind
    would still produce an ordered list, just a shorter one."""
    _session, h = _four_way()
    labels = list(gh.fill_nodes_by_label(h))
    assert len(labels) == len(set(labels)) == len(FOLD_ORDER)
    assert h.staged_fills() == len(FOLD_ORDER)


def test_the_storage_pin_is_load_bearing_and_the_fixture_evaluates() -> None:
    """The one evaluation. A `Double()` or `Weight()` fixture records identically and raises
    `TypeError: Keyword(s) sample not expected` here, so the record-time assertions above would be
    frozen over a program that can never run."""
    session, h = _four_way()
    filled = session.materialize(gh.fill_nodes_by_label(h)["nominal"])
    assert np.nansum(filled.view(flow=True)["sum_of_weights"]) > 0
