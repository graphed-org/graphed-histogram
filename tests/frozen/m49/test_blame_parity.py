"""m49/F8 — §6.1d: the plan/executor path blames the SAME operand `Session.materialize` does.

The row-space guard is an External whose evaluator is resolved by CONTENT HASH. `materialize`
evaluates the node the session holds; a plan ships a registry keyed by that hash and merged across
every histogram, so a guard identity that does not separate the two coordinates leaves one
evaluator answering for both and the plan path names the wrong factor. m48 anchors the guard only
through `materialize`, so that collapse survives its frozen suite.

The fixture is VARIED: sibling lowering records one guard per label per factor, which is where a
merged registry has the most room to answer with the wrong message.
"""

from __future__ import annotations

from typing import Any

import graphed
import pytest
from graphed import GraphedError
from graphed.awkward import gak, gnano
from graphed.core.execution import SequentialRunner
from m49_hist_fixtures import JES, TOY_EVENTS, eager_weighted, in_memory, partitioned

import graphed_histogram as gh

STEPS = 3
LABELS = tuple(JES)


def _shifted(events_ctx: Any) -> Any:
    return graphed.vary(
        events_ctx,
        "jes",
        Jet={
            "up": gak.with_field(events_ctx.Jet, events_ctx.Jet.pt * JES["jes_up"], "pt"),
            "down": gak.with_field(events_ctx.Jet, events_ctx.Jet.pt * JES["jes_down"], "pt"),
        },
    )


def _hist() -> gh.boost.Histogram:
    return gh.boost.Histogram(*eager_weighted().axes, storage=eager_weighted().storage_type())


def _offender_at(root: Any, index: int) -> gh.boost.Histogram:
    """Two explicit factors over a flattened per-OBJECT value; one of them is per-EVENT."""
    ctx = _shifted(gnano.events(root))
    value = gak.flatten(ctx.Jet.pt)
    fine = gak.flatten(ctx.Jet.pt * 0.0 + 1.0)
    offender = ctx.MET.pt * 0.01
    factors = [offender, fine] if index == 0 else [fine, offender]
    hist = _hist()
    hist.fill(value, weight=factors)
    return hist


def _matching(root: Any) -> gh.boost.Histogram:
    ctx = _shifted(gnano.events(root))
    hist = _hist()
    hist.fill(gak.flatten(ctx.Jet.pt), weight=[gak.flatten(ctx.Jet.pt * 0.0 + 1.0)])
    return hist


@pytest.mark.parametrize("index", [0, 1])
def test_the_plan_path_reports_the_same_message_materialize_reports(index: int) -> None:
    session, root = in_memory(TOY_EVENTS)
    per_label = gh.fill_nodes_by_label(_offender_at(root, index))
    assert set(per_label) == set(LABELS)
    from_materialize = {}
    for label, node in per_label.items():
        with pytest.raises(GraphedError) as excinfo:
            session.materialize(node)
        from_materialize[label] = str(excinfo.value)
    assert len(set(from_materialize.values())) == 1, "the siblings must blame the same operand"

    _plan_session, source, _data = partitioned(TOY_EVENTS)
    with pytest.raises(GraphedError) as excinfo:
        SequentialRunner().run(gh.plan({"jets": _offender_at(source, index)}, steps_per_file=STEPS))
    assert str(excinfo.value) == from_materialize["nominal"]


@pytest.mark.parametrize("index", [0, 1])
def test_the_plan_path_names_the_offending_factor_and_not_its_neighbour(index: int) -> None:
    """Message equality alone is satisfied by two paths that agree on the WRONG factor."""
    _session, source, _data = partitioned(TOY_EVENTS)
    with pytest.raises(GraphedError) as excinfo:
        SequentialRunner().run(gh.plan({"jets": _offender_at(source, index)}, steps_per_file=STEPS))
    message = str(excinfo.value)
    assert f"weight[{index}]" in message
    assert f"weight[{1 - index}]" not in message


def test_a_matching_factor_raises_on_neither_path() -> None:
    """The positive control: the same varied program with the factor at the fill's row space runs
    to a non-empty result on both routes, so the refusals above report a real length fault."""
    session, root = in_memory(TOY_EVENTS)
    assert session.materialize(_matching(root).fill_nodes()[0]).sum(flow=True).value > 0

    _plan_session, source, _data = partitioned(TOY_EVENTS)
    result = gh.unpack(
        SequentialRunner().run(gh.plan({"jets": _matching(source)}, steps_per_file=STEPS)).value
    )
    assert sorted(result["jets"]) == sorted(LABELS)
    assert all(result["jets"][label].sum(flow=True).value > 0 for label in LABELS)
