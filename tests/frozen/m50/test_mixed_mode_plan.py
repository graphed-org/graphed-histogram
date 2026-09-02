"""m50 — §6.1c mixed-mode plan: axis, sibling and bare outputs in ONE pass.

A single `plan(...)` carrying an axis-mode output, a sibling-mode varied output and an output no
variation reaches is the only program that reaches the mixed unpack. It witnesses the unpacking and
the per-slot spec, NOT a per-output MODE field — §6.1c records none; the three key forms (bare
`output`, `(output, label)`, `(output, None)`) are disjoint and per output, so `unpack` reads the
shape off the key. A FOURTH output — axis-mode opt-in with NO variations — pins that the MODE, not
the variation count, decides the `(output, None)` key and the appended 1-bin axis, and that
`.plan()` on that same histogram RAISES §6.1c's refusal (m48's varied-only trigger would let an
unvaried axis-mode histogram through into an opaque bh `ValueError`).

Exercises the axis-mode slot/opt-in, absent at the m50 baseline: FAILS today at the opt-in keyword.
"""

from __future__ import annotations

from collections.abc import Mapping

import graphed
import pytest
from graphed import GraphedError
from m50_axis_fixtures import (
    execute,
    fill_weight_program,
    partitioned,
    slice_label,
    var_index,
    views_equal,
    weighted,
)

import graphed_histogram as gh

WEIGHT_LABELS = ("nominal", "wgt_up", "wgt_down")


def _four_output_plan() -> tuple[dict, dict]:
    """One plan: axis-mode varied, sibling-mode varied, unvaried, and axis-mode with NO variations."""
    _s, events, _src = partitioned()
    hists = {
        "ax": fill_weight_program(True, source=events),
        "sib": fill_weight_program(False, source=events),
        "plain": weighted().fill(events.MET.pt),
        "axnovar": weighted().fill(events.MET.pt, variation_axis=True),
    }
    return execute(hists), hists


def test_the_four_slot_key_forms_are_disjoint_and_per_output() -> None:
    value, _hists = _four_output_plan()
    assert ("ax", None) in value  # axis-mode varied → (output, None)
    assert {("sib", label) for label in WEIGHT_LABELS} <= set(value)  # sibling → (output, label)
    assert "plain" in value  # unvaried → today's bare output key
    assert ("axnovar", None) in value  # the MODE, not the variation count, decides the key


def test_unpack_reads_each_shape_off_its_key() -> None:
    value, _hists = _four_output_plan()
    result = gh.unpack(value)
    assert set(result) == {"ax", "sib", "plain", "axnovar"}

    assert not isinstance(result["ax"], Mapping) and var_index(result["ax"]) >= 0
    assert isinstance(result["sib"], Mapping) and set(result["sib"]) == set(WEIGHT_LABELS)
    assert not isinstance(result["plain"], Mapping)
    assert not any(a.__dict__.get("name") == "variation" for a in result["plain"].axes)
    assert not isinstance(result["axnovar"], Mapping) and var_index(result["axnovar"]) >= 0


def test_the_axis_and_sibling_outputs_agree_bin_for_bin() -> None:
    """Correctness of the mixed unpack: the axis output sliced per label equals the sibling output's
    mapping entry — the two outputs run the SAME weight program in different modes."""
    value, _hists = _four_output_plan()
    result = gh.unpack(value)
    for label in WEIGHT_LABELS:
        assert views_equal(slice_label(result["ax"], label), result["sib"][label]), label


def test_the_narrowing_helpers_answer_over_all_four_shapes() -> None:
    """§2.2/§6.2(i-bis): `labels`/`universe`/`nominal` read the axis-mode histogram, the
    `{label: hist}` mapping, the bare unvaried hist and the axis-mode-no-variation hist uniformly."""
    value, _hists = _four_output_plan()
    result = gh.unpack(value)

    assert set(graphed.labels(result["ax"])) == set(WEIGHT_LABELS)
    assert graphed.labels(result["ax"])[0] == "nominal"
    assert set(graphed.labels(result["sib"])) == set(WEIGHT_LABELS)
    assert graphed.labels(result["plain"]) == ("nominal",)
    assert graphed.labels(result["axnovar"]) == ("nominal",)

    assert graphed.universe(result["ax"], "wgt_up") == slice_label(result["ax"], "wgt_up")
    assert graphed.nominal(result["ax"]) == slice_label(result["ax"], "nominal")
    assert graphed.universe(result["sib"], "wgt_up") is result["sib"]["wgt_up"]
    assert graphed.universe(result["plain"], "nominal") is result["plain"]


def test_the_axis_mode_no_variation_output_carries_a_one_bin_axis() -> None:
    """§6.2(ii): an axis-mode fill ALWAYS declares the variation axis, so a program with no
    variations lands a bare histogram with a 1-bin `{"nominal"}` axis — not a today's bare hist."""
    value, _hists = _four_output_plan()
    axnovar = gh.unpack(value)["axnovar"]
    axis = axnovar.axes[var_index(axnovar)]
    assert len(axis) == 1
    assert list(axis) == ["nominal"]


def test_plan_refuses_an_unvaried_axis_mode_histogram_pointing_at_the_group_api() -> None:
    """§6.1c: `Histogram.plan` starts from `self._spec` (fixed in `__init__`, no variation axis), so
    it cannot render an axis-mode result and refuses — GENERAL over the MODE, so the unvaried
    axis-mode histogram refuses too rather than falling through into an opaque bh error."""
    _s, events, _src = partitioned()
    h = weighted().fill(events.MET.pt, variation_axis=True)
    with pytest.raises(GraphedError, match=r"group|graphed_histogram\.plan"):
        h.plan()
