"""§6.3(2): the broadcast seam follows the recorded row spaces, not the context handle.

A fill records the seam only where a factor sits at a different row space than its value and the
backend can broadcast it there (`0 < depth(factor) < depth(value)`, with `broadcast_like`
supplied). At execution the guard passes a factor already at the value's leaf row space through,
broadcasts one at its outer row space, and blames the operand by name otherwise; a value carrying
missing entries stands the seam down at both ends, because the evaluator's flatten drops them.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
import pytest
from graphed import GraphedError
from graphed.awkward import from_awkward, gak
from graphed.numpy import NumpyBackend, NumpyForm, from_array
from vary_hist_fixtures import (
    EVENTS,
    broadcast_reference,
    eager_weighted,
    flat,
    in_memory_events,
    weighted,
)

import graphed_histogram as gh


def _blames(session: Any) -> set[str]:
    """The blame coordinates of the row-space guards this session recorded — the seam's witness,
    since a guard is recorded only where the fill decided a factor needs broadcasting.
    """
    return {
        node["params"]["blame"]
        for node in session._store.nodes()
        if node.get("params", {}).get("blame") is not None
    }


def test_a_plain_fill_at_one_row_space_records_a_single_node() -> None:
    """The control on the seam trigger. A plain per-EVENT fill weighted per EVENT needs no
    broadcast, so it must record exactly the one fill node it always did — the pre-m48 golden blob
    is that same shape. Triggering the seam on the mere presence of a weight (or unconditionally)
    reds here and moves that golden; the mismatched twin next to it shows the rule still fires.
    """
    session, root = in_memory_events()
    value, factor = root.MET.pt, root.MET.pt * 0.5 + 1.0
    h = weighted(bins=20, lo=0.0, hi=400.0)
    before = session.node_count()
    h.fill(value, weight=[factor])
    assert session.node_count() - before == 1

    mixed_session, mixed_root = in_memory_events()
    mixed_value, mixed_factor = mixed_root.Jet.pt, mixed_root.MET.pt * 0.5 + 1.0
    mixed = weighted(bins=20, lo=0.0, hi=400.0)
    before = mixed_session.node_count()
    mixed.fill(mixed_value, weight=[mixed_factor])
    assert mixed_session.node_count() - before > 1


def test_an_awkward_scalar_weight_is_no_row_space_and_takes_no_seam() -> None:
    """A scalar broadcasts against anything, so it is at NO row space — the same contract `_rows`
    already holds at execution time. Reading a scalar's depth as 0 and calling it "shallower than
    1" fires the seam on a fill that works, moving its graph to record a guard that provably cannot
    trip.
    """
    session, root = in_memory_events()
    value, factor = root.MET.pt, gak.sum(root.MET.pt, axis=None)
    h = weighted(bins=20, lo=0.0, hi=400.0)
    before = session.node_count()
    h.fill(value, weight=[factor])
    assert session.node_count() - before == 1

    reference = eager_weighted(bins=20, lo=0.0, hi=400.0)
    reference.fill(flat(EVENTS.MET.pt), weight=ak.sum(EVENTS.MET.pt, axis=None))
    got = session.materialize(h.fill_nodes()[0])
    assert np.allclose(
        got.view(flow=True)["value"], reference.view(flow=True)["value"], rtol=1e-12
    )


def _ravelled_value(session: Any) -> Any:
    """A 2-D value the evaluator's `_flat` ravels — its ndim disagrees with a weight that is
    nonetheless at the fill's row space.
    """
    return session.source(
        "x2",
        form=NumpyForm(np.dtype("float64"), kind="vector", shape=(None, 2)),
        data=np.arange(8.0).reshape(4, 2),
    )


def _scalar_weight(session: Any) -> Any:
    return session.source(
        "s", form=NumpyForm(np.dtype("float64"), kind="scalar", shape=()), data=np.float64(2.0)
    )


@pytest.mark.parametrize(
    ("value_of", "factor_of"),
    [
        (_ravelled_value, lambda s: from_array(s, "w", np.full(8, 2.0))),
        (lambda s: from_array(s, "x", np.arange(8.0)), _scalar_weight),
    ],
    ids=["2-D value / 1-D weight", "1-D value / scalar weight"],
)
def test_the_numpy_idiom_records_no_seam_at_any_row_space(value_of: Any, factor_of: Any) -> None:
    """A rectilinear backend supplies no `broadcast_like`, so `graphed.broadcast_like` takes the
    bound no-op: firing the row-space rule here can only turn a working fill into a guard error,
    never repair one. Both ends of the range are driven, each keeping the single fill node it
    recorded before the rule existed and still evaluating to 16.0.
    """
    session = graphed.Session(NumpyBackend())
    value, factor = value_of(session), factor_of(session)
    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight())
    before = session.node_count()
    h.fill(value, weight=[factor])
    assert session.node_count() - before == 1
    assert _blames(session) == set()
    got = session.materialize(h.fill_nodes()[0])
    assert float(got.view(flow=True)["value"].sum()) == 16.0


def _at_the_jet_row_space(session: Any, name: str, weights: np.ndarray) -> Any:
    """A weight already at the JET row space — one entry per object across the whole fixture, the
    shape a record-time form cannot tell from a per-event column.
    """
    return from_awkward(session, name, ak.Array(weights))


def test_an_already_flat_per_object_weight_fills_while_its_per_event_twin_takes_the_seam() -> None:
    """The two 1-D factors a per-object fill can carry, which no form tells apart: one already at
    the value's LEAF row space (nothing to broadcast — the evaluator would multiply it elementwise
    unaided), one at the value's OUTER row space (the seam's whole purpose). Deciding either at
    record time, or refusing the leaf one in the guard, turns a working fill into a hard error.

    The leaf factor is asserted twice: flat 2.0s pin the total, and a factor that varies per jet
    pins the ORDER, which re-nesting the factor into the value's structure could otherwise scramble
    without moving any total.
    """
    n_jets = int(ak.count(EVENTS.Jet.pt))
    session, root = in_memory_events()
    h = weighted(bins=20, lo=0.0, hi=400.0)
    h.fill(root.Jet.pt, weight=[_at_the_jet_row_space(session, "flat_sf", np.full(n_jets, 2.0))])
    got = session.materialize(h.fill_nodes()[0])
    reference = eager_weighted(bins=20, lo=0.0, hi=400.0)
    reference.fill(flat(EVENTS.Jet.pt), weight=np.full(n_jets, 2.0))
    assert np.array_equal(got.view(flow=True)["value"], reference.view(flow=True)["value"])
    assert float(got.view(flow=True)["value"].sum()) == 2.0 * n_jets

    varying = 1.0 + np.arange(n_jets, dtype=float) / n_jets
    ordered = weighted(bins=20, lo=0.0, hi=400.0)
    ordered.fill(root.Jet.pt, weight=[_at_the_jet_row_space(session, "varying_sf", varying)])
    ordered_got = session.materialize(ordered.fill_nodes()[0])
    ordered_ref = eager_weighted(bins=20, lo=0.0, hi=400.0)
    ordered_ref.fill(flat(EVENTS.Jet.pt), weight=varying)
    assert np.allclose(
        ordered_got.view(flow=True)["value"], ordered_ref.view(flow=True)["value"], rtol=1e-12
    )

    event_session, event_root = in_memory_events()
    per_event = weighted(bins=20, lo=0.0, hi=400.0)
    per_event.fill(event_root.Jet.pt, weight=[event_root.MET.pt * 0.5 + 1.0])
    assert _blames(event_session) == {"weight[0]"}
    event_got = event_session.materialize(per_event.fill_nodes()[0])
    broadcast_ref = eager_weighted(bins=20, lo=0.0, hi=400.0)
    broadcast_ref.fill(
        flat(EVENTS.Jet.pt), weight=broadcast_reference(EVENTS.Jet.pt, EVENTS.MET.pt * 0.5 + 1.0)
    )
    assert np.allclose(
        event_got.view(flow=True)["value"], broadcast_ref.view(flow=True)["value"], rtol=1e-12
    )


def _outer_option(root: Any) -> tuple[Any, ak.Array]:
    """`option[var * float64]`: whole events masked away, their jets still in the layout."""
    keep = ak.to_numpy(EVENTS.MET.pt) > 20.0
    return gak.mask(root.Jet.pt, root.MET.pt > 20.0), ak.mask(EVENTS.Jet.pt, keep)


def _inner_option(root: Any) -> tuple[Any, ak.Array]:
    """`var * ?float64`: every event padded out, the padding missing at the leaf."""
    return gak.pad_none(root.Jet.pt, 8), ak.pad_none(EVENTS.Jet.pt, 8)


@pytest.mark.parametrize("value_of", [_outer_option, _inner_option], ids=["outer", "inner"])
def test_a_value_with_missing_entries_keeps_its_pre_seam_fill_and_still_blames(
    value_of: Any,
) -> None:
    """The evaluator's flatten DROPS missing entries, so such a value's leaf row space is not the
    one its structure counts out: a factor already flat at it cannot be re-nested into the value,
    and at record time it cannot be told from a per-event factor either. Both ends of the seam
    stand down — the plain fill keeps the one node and the elementwise semantics it had before the
    seam existed, and the guard, wherever a broadcast is forced anyway, blames the factor by name
    instead of dying inside awkward with a message naming neither the fill nor the factor.
    """
    session, root = in_memory_events()
    value, eager = value_of(root)
    leaf = 1.0 + np.arange(len(flat(eager)), dtype=float) / 10.0
    at_the_leaf = _at_the_jet_row_space(session, "leaf_sf", leaf)

    h = weighted(bins=20, lo=0.0, hi=400.0)
    before = session.node_count()
    h.fill(value, weight=[at_the_leaf])
    assert session.node_count() - before == 1
    assert _blames(session) == set()
    got = session.materialize(h.fill_nodes()[0])
    reference = eager_weighted(bins=20, lo=0.0, hi=400.0)
    reference.fill(flat(eager), weight=leaf)
    assert np.allclose(
        got.view(flow=True)["value"], reference.view(flow=True)["value"], rtol=1e-12
    )

    varied_session, varied_root = in_memory_events()
    varied_value, _ = value_of(varied_root)
    factor = _at_the_jet_row_space(varied_session, "leaf_sf", leaf)
    forced = weighted(bins=20, lo=0.0, hi=400.0)
    forced.fill(varied_value, weight=[graphed.vary(factor, "sf", up=factor * 1.1)])
    with pytest.raises(GraphedError) as excinfo:
        varied_session.materialize(forced.fill_nodes()[0])
    assert "weight[0]" in str(excinfo.value)


def test_a_factor_deeper_than_the_value_records_no_seam_but_a_shallower_one_does() -> None:
    """`broadcast_like` lines a SHALLOWER factor up with a deeper value and does nothing else, so a
    per-object weight on a per-event value must record exactly the graph it recorded before the
    rule existed — one fill node, no guard — while the shallower twin still takes the seam. A rule
    keyed on "the depths differ" records a step here that cannot repair anything, moving the graph
    of a fill (this one, over a one-object-per-event collection) that evaluates correctly.
    """
    session, root = in_memory_events()
    singleton = ak.Array([[float(pt)] for pt in ak.to_numpy(EVENTS.MET.pt)])
    value, factor = root.MET.pt, from_awkward(session, "singleton", singleton)
    h = weighted(bins=20, lo=0.0, hi=400.0)
    before = session.node_count()
    h.fill(value, weight=[factor])
    assert session.node_count() - before == 1
    assert _blames(session) == set()
    got = session.materialize(h.fill_nodes()[0])
    reference = eager_weighted(bins=20, lo=0.0, hi=400.0)
    reference.fill(flat(EVENTS.MET.pt), weight=flat(singleton))
    assert np.allclose(
        got.view(flow=True)["value"], reference.view(flow=True)["value"], rtol=1e-12
    )

    shallow_session, shallow_root = in_memory_events()
    shallow = weighted(bins=20, lo=0.0, hi=400.0)
    before = shallow_session.node_count()
    shallow.fill(shallow_root.Jet.pt, weight=[shallow_root.MET.pt * 0.5 + 1.0])
    assert shallow_session.node_count() - before > 1
    assert _blames(shallow_session) == {"weight[0]"}
