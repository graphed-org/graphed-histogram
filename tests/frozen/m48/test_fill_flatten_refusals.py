"""m48/H7 — §6.1d: the length refusal is an EXECUTION-time message contract.

There is no record-time discriminator. A legitimately per-event value (`gak.firsts`, `gak.num`,
`MET.pt`) and a flattened per-object value have identical 1-D forms and differ only in runtime
length, and the record-time alternative — a flatten-hunting cone walk — false-positives on
`gak.flatten(x, axis=2)`. So this file freezes WHAT THE USER READS, not who raises it: the
broadcast seam is a recorded node upstream of the fill and executes first, so freezing
`FillEvaluator` as the raiser would red a correct implementation.

Three causes, three distinguishable messages (§6.1d):

* an offending AMBIENT factor — names the ambient weight, points at "pass the value unflattened";
* an offending EXPLICIT factor — names that entry BY POSITION, same pointer;
* an offending loose VALUE — names the VALUE, and NOT "pass the value unflattened": nothing was
  passed flattened, the value simply carries no handle and so no re-indexing is possible.
"""

from __future__ import annotations

from typing import Any

import graphed
import pytest
from graphed import GraphedError
from graphed.awkward import gak, gnano
from vary_hist_fixtures import in_memory_events, weighted

UNFLATTEN = "pass the value unflattened"


def _ambient(source: Any, scale: float = 1.0) -> Any:
    return source.MET.pt * 0.01 * scale


def _weighted_context() -> tuple[Any, Any]:
    session, root = in_memory_events()
    events = gnano.events(root)
    return session, graphed.vary(events, "pu", _ambient(events), is_weight=True, up=_ambient(events, 1.1))


def test_an_already_flattened_value_against_the_ambient_weight_names_the_ambient_factor() -> None:
    session, ctx = _weighted_context()
    h = weighted(bins=20, lo=0.0, hi=400.0)
    h.fill(gak.flatten(ctx.Jet.pt))  # the mistake: nothing is left to broadcast against
    with pytest.raises(GraphedError) as excinfo:
        session.materialize(h.fill_nodes()[0])
    message = str(excinfo.value)
    assert "ambient" in message
    assert UNFLATTEN in message


def test_the_refusal_is_not_raised_at_record_time() -> None:
    """The recording must succeed: the two 1-D forms are indistinguishable until the rows exist,
    and a record-time guard would have to false-positive on `gak.flatten(x, axis=2)`."""
    _session, ctx = _weighted_context()
    h = weighted(bins=20, lo=0.0, hi=400.0)
    assert h.fill(gak.flatten(ctx.Jet.pt)) is h
    assert h.staged_fills() > 0


def test_an_offending_explicit_factor_is_named_by_its_position_in_the_weight_list() -> None:
    """Two entries, one of them fine: a message that names the whole list, or the wrong entry,
    sends the reader to the factor that is not the problem."""
    session, root = in_memory_events()
    events = gnano.events(root)
    value = gak.flatten(events.Jet.pt)
    per_object = gak.flatten(events.Jet.pt * 0.0 + 1.0)  # already at the VALUE's row space
    per_event = events.MET.pt * 0.01  # the offender: per-EVENT against a flattened value
    h = weighted(bins=20, lo=0.0, hi=400.0)
    h.fill(value, weight=[per_object, per_event])
    with pytest.raises(GraphedError) as excinfo:
        session.materialize(h.fill_nodes()[0])
    message = str(excinfo.value)
    assert "weight[1]" in message
    assert "weight[0]" not in message
    assert UNFLATTEN in message


def test_a_loose_VALUE_at_the_wrong_row_space_gets_its_own_message() -> None:
    """§6.1d binds a DISTINCT message here: a loose input adopts the unified context for LABEL
    ALIGNMENT only — its row space is NOT adjusted, because no intervening mask is known — so the
    cause is the value, not a factor, and "pass the value unflattened" would be wrong advice."""
    session, root = in_memory_events()
    events = gnano.events(root)
    sel = events[events.MET.pt > 20.0]
    loose = root.MET.pt  # read from the SOURCE: context-free, and at the PARENT's row count
    assert graphed.context_of(loose) is None
    h = weighted()
    h.fill(loose, weight=[sel.MET.pt * 0.01])
    with pytest.raises(GraphedError) as excinfo:
        session.materialize(h.fill_nodes()[0])
    message = str(excinfo.value)
    assert "value[0]" in message
    assert UNFLATTEN not in message
    assert "ambient" not in message


def test_a_correctly_unflattened_per_object_fill_raises_nothing() -> None:
    """The positive control: the same program with the value left unflattened runs, so the three
    refusals above are reporting a real length fault and not a broken fill path."""
    session, ctx = _weighted_context()
    h = weighted(bins=20, lo=0.0, hi=400.0)
    h.fill(ctx.Jet.pt, weight=[ctx.MET.pt * 0.5 + 1.0])
    assert session.materialize(h.fill_nodes()[0]).sum(flow=True).value > 0
