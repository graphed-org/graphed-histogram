"""m52/C5 — §5.1/§6.2: sibling mode and axis mode resolve a joint label by the same rule.

The private fallback rule had SEVEN call sites: one session read, three for the sibling-mode fill
(axis value, weight factor, `sample=`) and the same three again for the axis-mode fill. Routing only
the sibling branch through the point-aware accessor leaves the axis-mode sites on the old rule, and
the joint bin is then wrong in one mode and right in the other — which per-universe equality between
the two modes is exactly what catches. It is also what carries diff coverage over both branches.

Axis mode collapses weight-borne labels into one loop node against a FIXED axis column, so the joint
label — borne by the ambient b-tag weight — has to leave that loop (or the loop has to group by
resolved member) for its shifted kinematics to reach the fill at all.

Parity alone would also be satisfied by two identically-wrong modes, so each mode is separately
witnessed to have moved the joint universe off the b-tag-only one before the two are compared.
"""

from __future__ import annotations

from typing import Any

from m52_joint_fill_fixtures import (
    BTAG_ONLY,
    JOINT,
    LABELS,
    axis_inputs,
    execute,
    in_memory,
    joint_context,
    other_inputs,
    partitioned,
    slice_label,
    three_role_fill,
    views_equal,
)

import graphed_histogram as gh


def _sibling() -> dict[str, Any]:
    _session, events = partitioned()
    return gh.unpack(execute({"h": three_role_fill(joint_context(events), axis_mode=False)}))["h"]


def _axis() -> Any:
    _session, events = partitioned()
    return gh.unpack(execute({"h": three_role_fill(joint_context(events), axis_mode=True)}))["h"]


def test_the_fill_carries_all_three_operand_roles() -> None:
    """The premise of the anchors below: the recorded sibling fill node holds one axis value and
    three further operands — the ambient b-tag weight, the extra weight factor and `sample=` — so
    every role the resolution rule is reached from is genuinely engaged by this program."""
    session, events = in_memory()
    by_label = gh.fill_nodes_by_label(three_role_fill(joint_context(events), axis_mode=False))
    assert list(by_label) == list(LABELS)
    assert len(axis_inputs(session, by_label[JOINT])) == 1
    assert len(other_inputs(session, by_label[JOINT])) == 3


def test_each_mode_moves_the_joint_universe_off_the_btag_only_one() -> None:
    """Per mode, the mechanism witness: the joint universe is neither the b-tag-only universe (the
    silent-nominal signature) nor the pure JES one. Without this, the parity anchor below would pass
    on an implementation that left BOTH modes on the old rule."""
    sibling = _sibling()
    assert not views_equal(sibling[JOINT], sibling[BTAG_ONLY]), "sibling mode"
    assert not views_equal(sibling[JOINT], sibling["jes_up"]), "sibling mode"

    axis = _axis()
    assert not views_equal(slice_label(axis, JOINT), slice_label(axis, BTAG_ONLY)), "axis mode"
    assert not views_equal(slice_label(axis, JOINT), slice_label(axis, "jes_up")), "axis mode"


def test_both_fill_modes_resolve_the_joint_label_alike() -> None:
    """§6.2's equality, extended to the joint label: the axis-mode histogram sliced at each label
    equals the sibling fill for that label bin-for-bin, flow included, on a WeightedMean storage."""
    axis = _axis()
    sibling = _sibling()
    assert set(sibling) == set(LABELS)
    for label in LABELS:
        assert views_equal(slice_label(axis, label), sibling[label]), label
