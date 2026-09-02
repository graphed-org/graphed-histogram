"""m48/H4 — §6.1c: the single-histogram `.plan()` path refuses a merge hazard.

`_SumFills` sums ALL staged fill nodes into one histogram, so a varied histogram routed through
`Histogram.plan()` would silently merge universes into a plausible-looking, physically wrong
result. §6.1c makes the refusal a DISJUNCTION — VARIED *or* in §6.2's axis mode — and m48
exercises the varied arm; the axis-mode arm is m50's, which is why nothing here is worded as "the
refusal fires because the histogram is varied".

Three wordings §6.1c rejects are NOT used: the fill-node COUNT (a single varied fill must refuse
too), the spec comparison (that decides only the axis-mode arm and has no m48-constructible
fixture), and any scoping to sibling mode (§6.1c: the refusal covers both merge hazards).
"""

from __future__ import annotations

import graphed
import numpy as np
import pytest
from graphed import GraphedError
from graphed.core.execution import SequentialRunner
from vary_hist_fixtures import EVENTS, in_memory_events, partitioned_events, weighted

import graphed_histogram as gh


def _varied(events: object, *, labels: int = 3) -> gh.boost.Histogram:
    factor = events.MET.pt * 0.01
    knobs = {"up": factor * 1.2} if labels == 2 else {"up": factor * 1.2, "down": factor * 0.8}
    h = weighted()
    h.fill(events.MET.pt, weight=[graphed.vary(factor, "sig", **knobs)])
    return h


def test_plan_on_a_varied_histogram_refuses_and_points_at_the_group_api() -> None:
    _session, events, _source = partitioned_events()
    with pytest.raises(GraphedError) as excinfo:
        _varied(events).plan()
    assert "graphed_histogram.plan" in str(excinfo.value)


def test_the_refusal_is_not_keyed_on_the_number_of_staged_fills() -> None:
    """A histogram carrying a SINGLE varied fill still refuses: the trigger is the merge hazard,
    not a count, and a count-based guard would let a one-fill varied histogram through."""
    _session, events, _source = partitioned_events()
    h = _varied(events, labels=2)
    assert h.staged_fills() == 2
    with pytest.raises(GraphedError):
        h.plan()
    single = weighted()
    factor = events.MET.pt * 0.01
    single.fill(events.MET.pt, weight=[graphed.vary(factor, "sig", up=factor * 1.2)])
    with pytest.raises(GraphedError):
        single.plan()


def test_plan_on_an_unvaried_sibling_mode_histogram_still_works() -> None:
    """§6.1c's positive control. Two staged fills, no variation: the sum IS the intended answer,
    and today's path must keep producing it."""
    _session, events, source = partitioned_events()
    h = weighted()
    h.fill(events.MET.pt)
    h.fill(events.MET.pt * 0.5)  # a DISTINCT observable: identical fills would intern to one node
    result = SequentialRunner().run(h.plan(steps_per_file=3)).value
    assert result.sum(flow=True).value == 2 * len(EVENTS)
    assert len(source.part_reads) == 3


def test_the_varied_route_to_a_plan_is_the_group_api_the_refusal_names() -> None:
    """The refusal is a redirect, not a dead end: the same histogram plans through
    `graphed_histogram.plan` and its per-label results come back."""
    _session, events, _source = partitioned_events()
    h = _varied(events)
    value = SequentialRunner().run(gh.plan({"met": h}, steps_per_file=3)).value
    result = gh.unpack(value)
    assert list(result["met"]) == ["nominal", "sig_up", "sig_down"]
    nominal = np.asarray(result["met"]["nominal"].view(flow=True)["value"])
    assert not np.array_equal(np.asarray(result["met"]["sig_up"].view(flow=True)["value"]), nominal)


def test_materialize_of_one_fill_node_is_untouched_by_the_refusal() -> None:
    """`Histogram.plan` is the refused surface; the reference eager path per fill node is not, and
    §4.3/§7.2's per-label reads depend on it staying open."""
    session, events = in_memory_events()
    h = _varied(events)
    per_label = gh.fill_nodes_by_label(h)
    assert session.materialize(per_label["sig_up"]).sum(flow=True).value > 0
