"""m50 — §6.2 declaration contract: the frontend declares the variation axis from the inferred set.

The declared bin set equals the §6.1d inferred label set EXACTLY. That needs TWO assertions: the
flow check `h.sum(flow=True) == h.sum()` (projected onto the variation axis, so the value axis's
own flow does not confound it) catches UNDER-declaration — a label silently landing in the
non-growth StrCategory's overflow — while an OVER-declaration passes it unchanged, so the axis's bin
tuple is asserted against a LITERALLY spelled list, never one read back from the histogram (circular
per §6.2 i-bis). The remaining refusals: a cross-fill label-set disagreement, a later fill in the
OTHER mode (the opt-in is per-`fill()` and remembered, so this is buildable), and a user-constructed
histogram already carrying a `"variation"` axis — recognised by `axis.__dict__.get("name")`, so a
user axis under ANY OTHER name is untouched.

Exercises the axis-mode opt-in, absent at the m50 baseline: each FAILS today at the opt-in keyword.
"""

from __future__ import annotations

import graphed
import pytest
from graphed import GraphedError
from m50_axis_fixtures import (
    execute,
    fill_weight_program,
    in_memory,
    partitioned,
    var_index,
    weight_family,
    weighted,
)

import graphed_histogram as gh

#: the inferred weight-family label set, sorted lexicographically (the frontend's non-growth axis
#: order); spelled literally so the over-declaration half is not read back from the histogram
EXPECTED_BINS = ["nominal", "wgt_down", "wgt_up"]


def test_the_declared_axis_matches_the_inferred_label_set_exactly() -> None:
    _s, events, _src = partitioned()
    z = gh.unpack(execute({"h": fill_weight_program(True, source=events)}))["h"]
    vi = var_index(z)
    (value_axis,) = (i for i in range(len(z.axes)) if i != vi)
    projected = z[{value_axis: sum}]  # fold the value axis (with its flow) onto the variation axis
    assert projected.sum(flow=True).value == projected.sum().value  # nothing in the variation overflow
    assert list(z.axes[vi]) == EXPECTED_BINS  # literal: an extra declared bin would show here


def test_a_cross_fill_label_disagreement_is_refused_naming_the_mismatch() -> None:
    _s, events = in_memory()
    h = weighted()
    h.fill(events.Jet.pt, weight=[weight_family(events)], variation_axis=True)  # {nominal, wgt_*}
    other = events.MET.pt * 0.02
    scale = graphed.vary(other, "scale", up=other * 1.3, down=other * 0.7)  # {nominal, scale_*}
    with pytest.raises(GraphedError) as exc:
        h.fill(events.Jet.pt, weight=[scale], variation_axis=True)
    assert "scale" in str(exc.value) or "wgt" in str(exc.value)  # the error names the mismatch


def test_a_later_fill_in_the_other_mode_is_refused_naming_both() -> None:
    _s, events = in_memory()
    h = weighted()
    h.fill(events.Jet.pt, weight=[weight_family(events)], variation_axis=True)  # first fill fixes axis mode
    with pytest.raises(GraphedError) as exc:
        h.fill(events.Jet.pt, weight=[weight_family(events)])  # sibling-mode fill into an axis-mode hist
    assert "axis" in str(exc.value).lower()  # the MODE is a property of the histogram; names it


def test_a_user_supplied_variation_axis_is_refused_pointing_at_the_opt_in() -> None:
    """Recognition is `axis.__dict__.get("name") == "variation"` (the kwarg form
    `StrCategory(..., name="variation")` is itself a `TypeError`); a pre-existing one is refused."""
    _s, events = in_memory()
    h = weighted()
    h.axes[0].__dict__["name"] = "variation"
    with pytest.raises(GraphedError) as exc:
        h.fill(events.MET.pt, variation_axis=True)
    assert "variation" in str(exc.value).lower()


def test_a_user_axis_under_another_name_is_untouched() -> None:
    """The recognition keys on the NAME, not on "any named axis" nor "any StrCategory": a value axis
    named `"process"` is not the variation axis, so the frontend proceeds and appends its own."""
    _s, events = in_memory()
    h = weighted()
    h.axes[0].__dict__["name"] = "process"
    returned = h.fill(events.MET.pt, variation_axis=True)
    assert returned is h  # not refused
    assert h.staged_fills() >= 1  # it proceeded past the recognition check and recorded a fill
