"""m48/H2 — §4.3: a weight variation MUST NOT change the selection, asserted STRUCTURALLY.

Equal counts is a tautology under §3's record-time expansion — the selection nodes are the same
interned ids by construction — so the binding predicate is the per-label one §4.3 quotes: the
selection cone's node ids are identical across all weight labels. The bound extraction is the
fill node's recorded input PREFIX, `store.nodes()[fill_id]["inputs"][:n_axes]` with `n_axes` read
from the node's own `params`; identical node ids imply identical cones by interning.

Two wordings §4.3 explicitly rejects are NOT used here: the impact-set-subset one (false for a
correct implementation — a label's sibling fill node always lands in the impact set) and the
containment form `reachable(mask) subset of reachable(fill[L])`, which holds in any program that
fills selected data and passes a `mask_L = mask & g_L` implementation.

The per-label fill nodes come from §9.1's accessor, `graphed_histogram.fill_nodes_by_label(h)`:
`Histogram.fill_nodes()` is a bare ordered list with no label attribution and no private route
reaches a correspondence that exists nowhere else.
"""

from __future__ import annotations

from typing import Any

import boost_histogram as bh
import graphed
import numpy as np
from graphed import Array, Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed_corpus import make_events

import graphed_histogram as gh

EVENTS = make_events(n_events=2000, seed=48)

LABELS = ("nominal", "btag_up", "btag_down")


def _program() -> tuple[Session, gh.boost.Histogram]:
    """The 4j1b region with a b-tag weight variation — selection identical across all three
    labels, weights different in all three."""
    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", EVENTS)
    good = events.Jet[events.Jet.pt > 25]
    selected = (gak.num(good, axis=1) >= 4) & (gak.sum(good.btag > 0.7, axis=1) == 1)
    sel_jets = good[selected]
    observable = gak.sum(sel_jets.pt, axis=1)

    def sf(scale: float) -> Array:
        return gak.prod((0.95 + 0.10 * sel_jets.btag) * scale, axis=1)

    weight = graphed.vary(sf(1.0), "btag", up=sf(1.03), down=sf(0.97))
    h = gh.boost.Histogram(bh.axis.Regular(40, 0, 800), storage=bh.storage.Double())
    h.fill(observable, weight=[weight])
    return session, h


def _nodes(session: Session) -> dict[int, Any]:
    return {n["id"]: n for n in session._store.nodes()}


def test_every_label_has_its_own_fill_node_under_the_per_label_accessor() -> None:
    session, h = _program()
    by_label = gh.fill_nodes_by_label(h)
    assert list(by_label) == list(LABELS)
    ids = [by_label[label].node_id for label in LABELS]
    assert len(set(ids)) == len(LABELS), "three distinct weights must record three distinct fills"
    assert set(ids) == {n.node_id for n in h.fill_nodes()}
    assert all(n.session is session for n in by_label.values())


def test_the_non_weight_input_prefix_is_identical_across_every_weight_label() -> None:
    """§4.3's binding form, quoted: the selection cone's node ids are identical across all weight
    labels. The axis inputs ARE the selection cone's terminal nodes, and interning makes equal ids
    equal cones."""
    session, h = _program()
    nodes = _nodes(session)
    by_label = gh.fill_nodes_by_label(h)

    nominal = nodes[by_label["nominal"].node_id]
    n_axes = int(nominal["params"]["n_axes"])
    expected = list(nominal["inputs"][:n_axes])
    assert len(expected) == 1

    for label in LABELS:
        node = nodes[by_label[label].node_id]
        assert int(node["params"]["n_axes"]) == n_axes
        assert list(node["inputs"][:n_axes]) == expected, f"{label} fills a different selection"


def test_the_weight_input_is_what_differs_between_the_labels() -> None:
    """The instrument for the assertion above: if the weight inputs agreed too, that test would
    pass under an implementation that recorded one fill and replayed it under three names."""
    session, h = _program()
    nodes = _nodes(session)
    by_label = gh.fill_nodes_by_label(h)
    weights = []
    for label in LABELS:
        node = nodes[by_label[label].node_id]
        n_axes = int(node["params"]["n_axes"])
        weights.append(tuple(node["inputs"][n_axes:]))
    assert len(set(weights)) == len(LABELS)


def test_the_labels_occupy_the_same_bins_and_carry_different_contents() -> None:
    """m05's equal-counts check as sanity, in the form a weighted histogram can carry: the b-tag SF
    is strictly positive, so an unchanged selection means an unchanged set of occupied bins, while
    a re-run selection (`mask_L = mask & g_L`) drops events and empties bins."""
    session, h = _program()
    by_label = gh.fill_nodes_by_label(h)
    filled = {label: session.materialize(by_label[label]) for label in LABELS}
    nominal_occupied = np.asarray(filled["nominal"].values()) != 0
    assert nominal_occupied.any()
    for label in LABELS:
        occupied = np.asarray(filled[label].values()) != 0
        assert np.array_equal(occupied, nominal_occupied), f"{label} changed which bins are filled"
    for label in ("btag_up", "btag_down"):
        assert not np.array_equal(np.asarray(filled[label].values()), np.asarray(filled["nominal"].values()))
