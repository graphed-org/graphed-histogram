"""m50 — §6.2(i-bis) axis-mode result shape and the narrowing helpers over it.

An axis-mode varied output is a BARE `bh.Histogram` carrying the `"variation"` axis — type-identical
to an unvaried one — so `graphed.labels`/`universe`/`nominal` recognise it explicitly. The histogram
is HAND-CONSTRUCTED here (a StrCategory whose `__dict__["name"]` is `"variation"`, since the kwarg
form is a `TypeError`), so these anchors do not depend on the fill machinery — only on the spec
codec round-trip and on `graphed`'s §6.2(i-bis) arm.

`graphed.labels` re-orders to `"nominal"`-first-then-axis-order while the STORED bin order stays
lexicographic — the discriminating family has a lexicographic-first bin that is NOT `"nominal"`. The
round-trip anchor already ships (the codec preserves axis metadata); the `graphed` narrowing anchors
do NOT ship at the m50 baseline (the duck-typed histogram arm reads every histogram as unvaried), so
they FAIL today, and that failure is their non-vacuity.
"""

from __future__ import annotations

import boost_histogram as bh
import graphed
import pytest
from m50_axis_fixtures import slice_label, var_index, views_equal

from graphed_histogram._spec import spec_of, zero_of

#: a family whose lexicographic-first bin is NOT "nominal", so re-ordering is observable
STORED_ORDER = ["jes_down", "jes_up", "nominal"]  # StrCategory bin order (frontend sort)
LABELS_ORDER = ("nominal", "jes_down", "jes_up")  # graphed.labels: nominal first, then axis order


def _axis_histogram() -> bh.Histogram:
    """A bare axis-mode histogram: a Regular value axis and a `"variation"` StrCategory, with
    distinct content per universe so the slices genuinely differ."""
    z = bh.Histogram(
        bh.axis.Regular(4, 0.0, 4.0), bh.axis.StrCategory(STORED_ORDER), storage=bh.storage.Weight()
    )
    z.axes[1].__dict__["name"] = "variation"  # the `hist` convention the codec round-trips
    z.fill([0.5, 1.5, 2.5], ["nominal"] * 3, weight=[1.0, 1.0, 1.0])
    z.fill([0.5, 1.5], ["jes_up"] * 2, weight=[2.0, 2.0])
    z.fill([3.5], ["jes_down"], weight=[5.0])
    return z


def test_the_variation_axis_name_survives_the_spec_round_trip() -> None:
    """The surviving oracle for "which axis is the variation one": `h.axes.name` MUST NOT be used
    (bh maps it over every axis and raises when one lacks a name — an `AttributeError` against a
    CORRECT histogram), so the invariant is that the name round-trips `spec_of` → `zero_of`."""
    z = zero_of(spec_of(_axis_histogram()))
    assert [a.__dict__.get("name") for a in z.axes] == [None, "variation"]


def test_graphed_labels_orders_nominal_first_while_the_stored_order_stays_lexicographic() -> None:
    z = _axis_histogram()
    assert graphed.labels(z) == LABELS_ORDER  # re-ordered, and NOT ["nominal"] (the unvaried answer)
    assert list(z.axes[var_index(z)]) == STORED_ORDER  # the stored bin order is unchanged


def test_graphed_nominal_is_the_nominal_slice_not_the_whole_histogram() -> None:
    z = _axis_histogram()
    got = graphed.nominal(z)
    assert len(got.axes) == 1  # the variation axis is sliced away, not returned whole
    assert views_equal(got, slice_label(z, "nominal"))


def test_graphed_universe_slices_each_label() -> None:
    z = _axis_histogram()
    with pytest.raises(TypeError):  # the named-dict form does not slice a bare bh.Histogram
        z[{"variation": "nominal"}]
    for label in LABELS_ORDER:
        got = graphed.universe(z, label)
        assert len(got.axes) == 1, label  # the variation axis is sliced away
        assert views_equal(got, slice_label(z, label)), label
