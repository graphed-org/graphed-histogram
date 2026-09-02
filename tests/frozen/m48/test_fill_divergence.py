"""m48/H8 — §6.1d: the fill is a combining point, so it raises the divergence error itself.

§2.3e's op-level rule is EARLY detection, but no op precedes a fill: it is the first place where
independent axis, weight and `sample=` handles meet. So the fill runs the same most-derived
unification and divergence check across ALL of them, and `sample=` is included explicitly because
nothing upstream checks it — today's `fill` type-checks `args` and `weights` and appends `sample`
straight onto the input list.

The raise is RECORD-time, so this fixture carries no storage constraint: the histogram is never
evaluated.
"""

from __future__ import annotations

from typing import Any

import graphed
import pytest
from graphed import GraphedError
from graphed.awkward import gnano
from vary_hist_fixtures import in_memory_events, weighted, weighted_2d


def _divergent() -> tuple[Any, Any, Any]:
    """Two SIBLING contexts: neither is an ancestor of the other, and a mask has no inverse, so no
    re-indexing exists in either direction."""
    _session, root = in_memory_events()
    events = gnano.events(root)
    left = events[events.MET.pt > 20.0]
    right = events[events.MET.pt > 40.0]
    return events, left, right


def test_two_divergent_axis_values_are_refused_naming_both_contexts() -> None:
    _events, left, right = _divergent()
    h = weighted_2d()
    with pytest.raises(GraphedError) as excinfo:
        h.fill(left.MET.pt, right.MET.pt)
    message = str(excinfo.value)
    assert repr(left) in message and repr(right) in message


def test_a_divergent_sample_is_refused_by_the_same_check() -> None:
    """`sample=` is a first-class operand of the unification, not an unchecked appendix: here both
    axis and weight sit on ONE context and only the sample diverges."""
    _events, left, right = _divergent()
    h = weighted()
    with pytest.raises(GraphedError) as excinfo:
        h.fill(left.MET.pt, weight=[left.MET.pt * 0.01], sample=right.MET.phi)
    message = str(excinfo.value)
    assert repr(left) in message and repr(right) in message


def test_a_divergent_weight_factor_is_refused_too() -> None:
    _events, left, right = _divergent()
    h = weighted()
    with pytest.raises(GraphedError) as excinfo:
        h.fill(left.MET.pt, weight=[right.MET.pt * 0.01])
    message = str(excinfo.value)
    assert repr(left) in message and repr(right) in message


def test_no_fill_node_is_staged_by_the_refused_fill() -> None:
    """A refusal that staged half a fill leaves the histogram carrying a node no label maps to,
    which `plan()` then slices against. Scoped to the histogram's own staging: `record_external`
    adds its node to the arena before the handle merge runs, so the session's node count is not the
    histogram's to keep flat."""
    _events, left, right = _divergent()
    h = weighted_2d()
    with pytest.raises(GraphedError):
        h.fill(left.MET.pt, right.MET.pt)
    assert h.staged_fills() == 0
    assert h.fill_nodes() == []


def test_an_ancestor_chain_fill_unifies_instead_of_raising() -> None:
    """The positive control. Inputs whose contexts sit on ONE ancestry chain unify to the
    most-derived context — the divergence check must not fire on a legal program."""
    events, left, _right = _divergent()
    h = weighted_2d()
    assert h.fill(events.MET.pt, left.MET.pt) is h
    assert h.staged_fills() > 0


def test_a_loose_input_alongside_a_contexted_one_does_not_diverge() -> None:
    """§6.1d's adopt rule: a context-free value has no handle to diverge with, so it is ignored by
    the unification rather than treated as a third branch."""
    _session, root = in_memory_events()
    events = gnano.events(root)
    sel = events[events.MET.pt > 20.0]
    assert graphed.context_of(root.MET.phi) is None
    h = weighted()
    assert h.fill(sel.MET.pt, weight=[sel.MET.pt * 0.01], sample=root.MET.phi) is h
