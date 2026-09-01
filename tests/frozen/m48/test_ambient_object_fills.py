"""m48/H6 — §6.1d: register-then-forget, completed at the fill.

`fill` reads its inputs' context handle and auto-applies that context's ambient weight, so a plain
Jet-pT fill yields the pileup universes with zero per-fill bookkeeping — the owner's simultaneity
requirement. The value is passed UNFLATTENED (`h.fill(sel.Jet.pt)`, never `gak.flatten(...)`,
which destroys the structure there is nothing left to broadcast against), and the frontend records
a broadcast-to-value-structure seam for EVERY weight factor it applies: the ambient one and each
explicit `weight=[...]` entry alike, since the evaluator flattens each input independently and
multiplies after flattening.

The trigger for that seam is a DISJUNCTION (§6.3(2)) — a context handle OR any `Varied` input — so
the contexted-but-unvaried fill is a case of its own, asserted here with a witness that the seam
node was really recorded rather than the reference agreeing by accident.

This file also carries the three fill-shaped label-set assertions that §10/m48's §2.6 bullet
routes to `graphed-histogram`: asserting a fill's label set needs `Histogram.fill`, which no
`graphed` tree may import.
"""

from __future__ import annotations

from typing import Any

import graphed
import numpy as np
from graphed.awkward import gak, gnano
from vary_hist_fixtures import (
    EVENTS,
    broadcast_reference,
    eager_weighted,
    flat,
    in_memory_events,
    weighted,
)

import graphed_histogram as gh

PU = {"nominal": 1.0, "pu_up": 1.1, "pu_down": 0.9}
JES = {"nominal": 1.0, "jes_up": 1.05, "jes_down": 0.95}


def _ambient(source: Any, scale: float) -> Any:
    """A per-EVENT pileup-style weight — deliberately not constant, so a reference re-indexed or
    broadcast the wrong way compares unequal."""
    return source.MET.pt * 0.01 * scale


def _explicit(source: Any) -> Any:
    """The user's own per-EVENT factor, handed to `weight=[...]`."""
    return source.MET.pt * 0.5 + 1.0


def _shifted_jets(source: Any, scale: float) -> Any:
    jets = source.Jet
    return gak.with_field(jets, jets.pt * scale, "pt")


def _program() -> tuple[Any, Any, gh.boost.Histogram]:
    """Ambient pileup weight registered first, then a JES shift on the Jet collection: the fill's
    label set is the §2.4 union of the VALUE's labels and the AMBIENT weight's."""
    session, root = in_memory_events()
    events = gnano.events(root)
    ambient = _ambient(events, 1.0)
    weighted_ctx = graphed.vary(
        events, "pu", ambient, is_weight=True, up=_ambient(events, 1.1), down=_ambient(events, 0.9)
    )
    shifted = graphed.vary(
        weighted_ctx,
        "jes",
        Jet={"up": _shifted_jets(weighted_ctx, 1.05), "down": _shifted_jets(weighted_ctx, 0.95)},
    )
    h = weighted(bins=20, lo=0.0, hi=400.0)
    h.fill(shifted.Jet.pt, weight=[_explicit(shifted)])  # the value stays UNFLATTENED
    return session, shifted, h


def _reference(label: str) -> Any:
    """The manual reference for one label: the per-object value that label sees, weighted by the
    ambient factor AND the explicit factor, each broadcast to the value's structure before the
    evaluator flattens them."""
    jet_pt = EVENTS.Jet.pt * JES.get(label, 1.0)
    ambient = EVENTS.MET.pt * 0.01 * PU.get(label, 1.0)
    explicit = EVENTS.MET.pt * 0.5 + 1.0
    weights = broadcast_reference(jet_pt, ambient) * broadcast_reference(jet_pt, explicit)
    reference = eager_weighted(bins=20, lo=0.0, hi=400.0)
    reference.fill(flat(jet_pt), weight=weights)
    return reference


def test_a_per_object_fill_carries_the_value_labels_UNION_the_ambient_labels() -> None:
    _session, shifted, h = _program()
    assert set(gh.fill_nodes_by_label(h)) == {"nominal", "pu_up", "pu_down", "jes_up", "jes_down"}
    assert set(graphed.labels(shifted)) == set(gh.fill_nodes_by_label(h))


def test_every_labels_contents_equal_the_manual_broadcast_reference() -> None:
    """Both factors are per-EVENT against a per-OBJECT value, so an implementation that forgets to
    broadcast either one length-mismatches, and one that broadcasts only the ambient factor gets
    the wrong contents rather than an error."""
    session, _shifted, h = _program()
    per_label = gh.fill_nodes_by_label(h)
    for label in per_label:
        got = session.materialize(per_label[label])
        want = _reference(label)
        assert np.allclose(got.view(flow=True)["value"], want.view(flow=True)["value"], rtol=1e-12)
        assert np.allclose(got.view(flow=True)["variance"], want.view(flow=True)["variance"], rtol=1e-12)


def test_the_labels_are_not_all_the_same_histogram() -> None:
    """The instrument for the reference comparison: if every label produced nominal's contents the
    assertions above would pass under an implementation that never applied a variation."""
    session, _shifted, h = _program()
    per_label = gh.fill_nodes_by_label(h)
    nominal = np.asarray(session.materialize(per_label["nominal"]).view(flow=True)["value"])
    for label in ("pu_up", "pu_down", "jes_up", "jes_down"):
        assert not np.array_equal(
            np.asarray(session.materialize(per_label[label]).view(flow=True)["value"]), nominal
        )


def _unvaried_contexted() -> tuple[Any, gh.boost.Histogram, int]:
    """§6.3(2)'s context-handle-only case: a context with NO registrations, no `Varied` input, a
    per-object value and an explicit per-event factor."""
    session, root = in_memory_events()
    events = gnano.events(root)
    value, factor = events.Jet.pt, _explicit(events)
    assert not isinstance(value, graphed.Varied) and not isinstance(factor, graphed.Varied)
    assert graphed.weight(events) is None
    h = weighted(bins=20, lo=0.0, hi=400.0)
    before = session.node_count()
    h.fill(value, weight=[factor])
    return session, h, session.node_count() - before


def test_a_contexted_but_unvaried_fill_broadcasts_and_records_the_seam() -> None:
    session, h, contexted_delta = _unvaried_contexted()

    plain_session, plain_root = in_memory_events()
    value, factor = plain_root.Jet.pt, _explicit(plain_root)
    plain = weighted(bins=20, lo=0.0, hi=400.0)
    before = plain_session.node_count()
    plain.fill(value, weight=[factor])
    plain_delta = plain_session.node_count() - before

    assert plain_delta == 1, "a fill with neither a handle nor a Varied input records as today"
    assert contexted_delta > plain_delta, "the broadcast seam node was not recorded"

    jet_pt = EVENTS.Jet.pt
    explicit = EVENTS.MET.pt * 0.5 + 1.0
    reference = eager_weighted(bins=20, lo=0.0, hi=400.0)
    reference.fill(flat(jet_pt), weight=broadcast_reference(jet_pt, explicit))
    got = session.materialize(h.fill_nodes()[0])
    assert np.allclose(got.view(flow=True)["value"], reference.view(flow=True)["value"], rtol=1e-12)


# --- the three label-set assertions §10/m48 routes here from the §2.6 bullet -------------------
def test_the_contexts_labels_are_the_CONTEXT_BORNE_half_of_a_fills_label_set() -> None:
    """§2.2's superset clause, on the shift-varied-collection program: `graphed.labels(ctx)` is a
    superset of the context-borne half, NOT of the whole label set, which also carries the value's
    own labels — here a loose `lumi` variation the context never saw."""
    _session, root = in_memory_events()
    events = gnano.events(root)
    shifted = graphed.vary(
        events, "jes", Jet={"up": _shifted_jets(events, 1.05), "down": _shifted_jets(events, 0.95)}
    )
    unvaried_mask = shifted.MET.pt > 20.0
    assert not isinstance(unvaried_mask, graphed.Varied)
    sel = shifted[unvaried_mask]

    lumi = _explicit(sel)
    h = weighted(bins=20, lo=0.0, hi=400.0)
    h.fill(sel.Jet.pt, weight=[graphed.vary(lumi, "lumi", up=lumi * 1.02)])

    context_labels = set(graphed.labels(sel))
    fill_labels = set(gh.fill_nodes_by_label(h))
    assert context_labels == {"nominal", "jes_up", "jes_down"}
    assert context_labels < fill_labels
    assert fill_labels - context_labels == {"lumi_up"}


def test_a_fill_from_the_pre_vary_context_carries_no_new_label() -> None:
    """§2.6b: `vary` returns a NEW context and leaves its input alone, which is observable at the
    sink — the pre-`vary` fill must not acquire the registration made after it."""
    session, root = in_memory_events()
    events = gnano.events(root)
    before = weighted()
    before.fill(events.MET.pt)

    weighted_ctx = graphed.vary(events, "pu", _ambient(events, 1.0), is_weight=True, up=_ambient(events, 1.1))
    after = weighted()
    after.fill(weighted_ctx.MET.pt)

    assert set(gh.fill_nodes_by_label(before)) == {"nominal"}
    assert set(gh.fill_nodes_by_label(after)) == {"nominal", "pu_up"}
    assert session.materialize(gh.fill_nodes_by_label(before)["nominal"]).sum(flow=True).value == len(EVENTS)


def test_the_origination_pair_has_one_node_id_and_two_fill_label_sets() -> None:
    """§2.3e's origination rule at the sink: the merge-from-inputs rule alone gets this wrong, and
    the two reads are indistinguishable by node id — only the handle separates them."""
    _session, root = in_memory_events()
    events = gnano.events(root)
    weighted_ctx = graphed.vary(events, "pu", _ambient(events, 1.0), is_weight=True, up=_ambient(events, 1.1))

    through_parent = events.MET.pt
    through_child = weighted_ctx.MET.pt
    assert through_parent.node_id == through_child.node_id
    assert graphed.context_of(through_parent) is not graphed.context_of(through_child)

    parent_h, child_h = weighted(), weighted()
    parent_h.fill(through_parent)
    child_h.fill(through_child)
    assert set(gh.fill_nodes_by_label(parent_h)) == {"nominal"}
    assert set(gh.fill_nodes_by_label(child_h)) == {"nominal", "pu_up"}
