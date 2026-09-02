"""m49/F3 — §7.2's merge-shortfall refusal, widened to its CLASS: both consumers, every program.

m48 refused only on the group-plan builder and only for VARIED programs. The premise under that
scoping — that unvaried programs are unaffected — is false on both members, and they fail
differently:

* the GROUP builder mis-slices an unvaried merged program and dies in the worker with an opaque
  `IndexError`, one partition deep;
* `Histogram.plan()` raises nothing at all. `_SumFills` ITERATES the evaluated fills rather than
  indexing them, so a compile that dropped one simply sums fewer and returns a plausible,
  physically wrong histogram — the silent miscompilation §5.4 itself calls worse than refusal,
  reachable from public API.

The trigger is the M4 reducer merging DISTINCT record ids: `weight=[w]` against `weight=[w * 1.0]`
records two fill nodes and compiles to one. Both refusals are BUILDER-side, so they are asserted
around the builder call — a refusal that waits for the run is the `IndexError` again.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import boost_histogram as bh
import numpy as np
import pytest
from graphed import GraphedError, Session
from graphed.awkward import AwkwardBackend, AwkwardForm
from graphed.core.execution import SequentialRunner
from m49_hist_fixtures import CountingSource, eager_weighted

import graphed_histogram as gh

STEPS = 2

#: `w` and `w2` carry the SAME values through DIFFERENT source fields: two fills over them do not
#: merge, and their correct total is exactly twice a single fill's — the shape a shortfall halves
EVENTS = ak.Array(
    {
        "x": [1.0, 4.0, 7.0, 2.5, 6.0] * 8,
        "w": [0.5, 1.0, 2.0, 1.5, 0.2] * 8,
        "w2": [0.5, 1.0, 2.0, 1.5, 0.2] * 8,
    }
)


def _events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    return session, session.source("events", form=form, data=CountingSource(EVENTS))


def _hist() -> gh.boost.Histogram:
    return gh.boost.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight())


def _merging() -> gh.boost.Histogram:
    """Two UNVARIED fills the M4 identity rules merge: `w` against `w * 1.0`."""
    _session, events = _events()
    h = _hist()
    h.fill(events.x, weight=[events.w])
    h.fill(events.x, weight=[events.w * 1.0])
    return h


def _merge_free() -> gh.boost.Histogram:
    """The positive control: same two-fill shape, same VALUES, distinct expressions."""
    _session, events = _events()
    h = _hist()
    h.fill(events.x, weight=[events.w])
    h.fill(events.x, weight=[events.w2])
    return h


def _single() -> gh.boost.Histogram:
    _session, events = _events()
    h = _hist()
    h.fill(events.x, weight=[events.w])
    return h


def _values(hist: bh.Histogram) -> np.ndarray:
    return np.asarray(hist.view(flow=True)["value"])


def test_the_group_builder_refuses_a_merged_UNVARIED_program() -> None:
    """m48's refusal was scoped to varied programs, so this one reaches the worker `IndexError`."""
    with pytest.raises(GraphedError) as excinfo:
        gh.plan({"h": _merging()}, steps_per_file=STEPS)
    message = str(excinfo.value)
    assert "h" in message
    assert "nominal" not in message, "an unvaried program has no labels to name"


def test_the_single_histogram_plan_refuses_a_merged_program() -> None:
    """New work, not a widening: `Histogram.plan()` has never carried a shortfall check."""
    with pytest.raises(GraphedError) as excinfo:
        _merging().plan(steps_per_file=STEPS)
    assert "merged" in str(excinfo.value)


def test_neither_refusal_waits_for_the_run() -> None:
    """Both are builder-side. A check installed in the worker still ships a plan whose slots
    cannot be told apart, which is the failure mode the refusal exists to replace."""
    for build in (lambda: gh.plan({"h": _merging()}, steps_per_file=STEPS), lambda: _merging().plan()):
        with pytest.raises(GraphedError):
            build()


def test_the_merge_free_pair_still_plans_and_sums_BOTH_fills_on_both_consumers() -> None:
    """The positive control. A refusal that fires on any two-fill program is red here, and the
    factor-two comparison is what names the wrong answer's shape: the merged program's silent
    answer drops a whole fill."""
    want = eager_weighted(bins=4, lo=0.0, hi=8.0)
    for _ in range(2):
        want.fill(ak.to_numpy(EVENTS.x), weight=ak.to_numpy(EVENTS.w))

    grouped = gh.unpack(SequentialRunner().run(gh.plan({"h": _merge_free()}, steps_per_file=STEPS)).value)
    single = SequentialRunner().run(_merge_free().plan(steps_per_file=STEPS)).value
    for got in (grouped["h"], single):
        assert np.allclose(_values(got), _values(want), rtol=1e-12)

    one = SequentialRunner().run(_single().plan(steps_per_file=STEPS)).value
    assert np.allclose(_values(want), 2.0 * _values(one), rtol=1e-12)
    assert _values(one).sum() > 0


def test_the_merged_pair_really_does_record_two_nodes_that_compile_to_one() -> None:
    """The instrument: if the identity token stopped merging, every refusal above would be
    asserting a refusal of a program that has nothing wrong with it."""
    from graphed import compile_ir  # noqa: PLC0415  (a diagnostic import, not a fixture surface)
    from graphed.core import GraphStore  # noqa: PLC0415

    merged, free = _merging(), _merge_free()
    for hist, shrinks in ((merged, True), (free, False)):
        marked = len(dict.fromkeys(node.node_id for node in hist.fill_nodes()))
        compiled = compile_ir(hist.fill_nodes()[0].session, *hist.fill_nodes())
        outputs = len(GraphStore.deserialize(compiled.ir).outputs())
        assert marked == 2
        assert (outputs < marked) is shrinks
