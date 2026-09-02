"""m48/H3 — §6.1a result shapes in SIBLING mode, plus §2.3d's *accepting* representative.

Two shapes are frozen in ONE anchor because freezing the nested user-facing shape alone would red
an implementation conforming to §6.1c's flat keying: the executed plan's value is the flat
slot-keyed `{(output, label) -> bh.Histogram}` mapping, and `graphed_histogram.unpack` is what
turns it into `{output_name: hist | {label: hist}}`.

Scoped to sibling mode deliberately — §6.2's axis mode is m50 and has a third result shape, so
nothing here may freeze a general rule m50 must contradict.

§2.3d's *accepting* class means the verb consumes a `Varied` operand and handles it INTERNALLY
without returning per-label results to the caller; `Histogram.fill` is that class's behaviourally
real representative, so its disposition is asserted here rather than in a table.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import boost_histogram as bh
import graphed
import numpy as np
from graphed.core.execution import SequentialRunner
from vary_hist_fixtures import EVENTS, partitioned_events, weighted

import graphed_histogram as gh

LABELS = ("nominal", "sig_up", "sig_down")


def _mixed() -> tuple[dict[Any, bh.Histogram], gh.boost.Histogram]:
    """A MIXED plan: one output a variation reaches, one it does not."""
    _session, events, _source = partitioned_events()
    factor = events.MET.pt * 0.01
    varied = weighted()
    varied.fill(events.MET.pt, weight=[graphed.vary(factor, "sig", up=factor * 1.2, down=factor * 0.8)])
    plain = weighted()
    plain.fill(events.MET.pt, weight=factor)
    value = SequentialRunner().run(gh.plan({"varied": varied, "plain": plain}, steps_per_file=3)).value
    return value, varied


def test_the_plans_combined_value_is_the_flat_slot_keyed_mapping() -> None:
    """§6.1c: `(output, label)` slots for a varied output, today's BARE `output_name` key for an
    output no variation reaches — the rule that keeps the already-frozen m23 suite green."""
    value, _varied = _mixed()
    assert set(value) == {("varied", label) for label in LABELS} | {"plain"}
    assert all(isinstance(hist, bh.Histogram) for hist in value.values())


def test_unpacking_gives_a_label_mapping_for_the_varied_output_and_a_bare_hist_for_the_other() -> None:
    value, _varied = _mixed()
    result = gh.unpack(value)
    assert set(result) == {"varied", "plain"}

    assert isinstance(result["varied"], Mapping)
    assert list(result["varied"]) == list(LABELS)

    assert isinstance(result["plain"], bh.Histogram)
    assert not isinstance(result["plain"], Mapping), "an unreached output is a BARE hist"


def test_absent_labels_are_absent_and_never_duplicated_from_nominal() -> None:
    """The `plain` output carries no `sig_*` entry at all — not a copy of nominal under each
    label — and the varied output's members genuinely differ, so neither shape could be produced
    by broadcasting one histogram over the plan's whole label set."""
    value, _varied = _mixed()
    assert not any(isinstance(key, tuple) and key[0] == "plain" for key in value)
    result = gh.unpack(value)
    nominal = np.asarray(result["varied"]["nominal"].view(flow=True)["value"])
    for label in ("sig_up", "sig_down"):
        assert not np.array_equal(np.asarray(result["varied"][label].view(flow=True)["value"]), nominal)


def test_the_narrowing_helpers_answer_uniformly_on_both_shapes() -> None:
    """§2.2 binds `labels`/`universe`/`nominal` on both result shapes, `graphed.nominal` INCLUDED
    — the assertion that rules out an implementation returning its argument unchanged for every
    histogram. A bare `hist` reads as the single label `"nominal"`."""
    value, _varied = _mixed()
    result = gh.unpack(value)

    assert tuple(graphed.labels(result["varied"])) == LABELS
    assert tuple(graphed.labels(result["plain"])) == ("nominal",)

    for label in LABELS:
        assert graphed.universe(result["varied"], label) is result["varied"][label]
    assert graphed.nominal(result["varied"]) is result["varied"]["nominal"]

    assert graphed.universe(result["plain"], "nominal") is result["plain"]
    assert graphed.nominal(result["plain"]) is result["plain"]


def test_a_wholly_unvaried_group_plan_keeps_todays_value_verbatim() -> None:
    """The positive control for `plan()`'s widened return type: every key is a bare output name,
    so the m23 indexing idiom still works."""
    _session, events, _source = partitioned_events()
    hi = weighted()
    hi.fill(events.MET.pt)
    lo = weighted()
    lo.fill(events.MET.pt * 0.5)
    value = SequentialRunner().run(gh.plan({"hi": hi, "lo": lo}, steps_per_file=3)).value
    assert set(value) == {"hi", "lo"}
    assert value["hi"].sum(flow=True).value == len(EVENTS)
    assert gh.unpack(value)["hi"] is value["hi"]


def test_fill_ACCEPTS_a_varied_operand_and_returns_the_histogram_itself() -> None:
    """§2.3d's *accepting* disposition: `fill` consumes the container internally, so the caller
    gets its histogram back — never a per-label mapping, which is what *expanding* would do."""
    _session, events, _source = partitioned_events()
    factor = events.MET.pt * 0.01
    h = weighted()
    returned = h.fill(events.MET.pt, weight=[graphed.vary(factor, "sig", up=factor * 1.2)])
    assert returned is h
    assert not isinstance(returned, Mapping)
    assert h.staged_fills() == 2  # the container expanded into SIBLING fill nodes, not into a result
