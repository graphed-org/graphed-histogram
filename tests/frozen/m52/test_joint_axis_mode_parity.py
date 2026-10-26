"""m52/C5 — §5.1/§6.2: sibling mode and axis mode resolve a joint label by the same rule.

The private fallback rule had SEVEN call sites: one session read, three for the sibling-mode fill
(axis value, weight factor, `sample=`) and the same three again for the axis-mode fill. Routing
only the sibling branch through the point-aware accessor leaves the axis-mode sites on the old
rule, and the joint bin is then wrong in one mode and right in the other — which per-universe
equality between the two modes is exactly what catches. It is also what carries diff coverage over
both branches.

Axis mode collapses weight-borne labels into one loop node against a FIXED axis column, so the
joint label — borne by the ambient b-tag weight — has to leave that loop (or the loop has to group
by resolved member) for its shifted kinematics to reach the fill at all.

Parity alone would also be satisfied by two identically-wrong modes, so each mode is separately
witnessed to resolve the joint's axis value to the shifted member before the two are compared. That
witness has to be STRUCTURAL: the joint's ambient b-tag weight is already a different node from
`btag_hf_up`'s, so the executed inequality between those two labels holds even when the axis value
wrongly resolves to nominal.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import graphed
from m52_joint_fill_fixtures import (
    BTAG_ONLY,
    JOINT,
    LABELS,
    axis_inputs,
    execute,
    in_memory,
    joint_context,
    observable,
    other_inputs,
    partitioned,
    slice_label,
    three_role_fill,
    views_equal,
)

import graphed_histogram as gh


class Recorded(NamedTuple):
    """One mode's recorded axis-value member per label, beside the two named candidates."""

    axis_member: dict[str, list[int]]
    shifted: list[int]
    nominal: list[int]


def _recorded(*, axis_mode: bool) -> Recorded:
    session, events = in_memory()
    context = joint_context(events)
    obs = observable(context)
    by_label = gh.fill_nodes_by_label(three_role_fill(context, axis_mode=axis_mode))
    return Recorded(
        axis_member={label: axis_inputs(session, node) for label, node in by_label.items()},
        shifted=[graphed.universe(obs, "jes_up").node_id],
        nominal=[graphed.nominal(obs).node_id],
    )


def _sibling() -> dict[str, Any]:
    _session, events = partitioned()
    return gh.unpack(execute({"h": three_role_fill(joint_context(events), axis_mode=False)}))["h"]


def _axis() -> Any:
    _session, events = partitioned()
    return gh.unpack(execute({"h": three_role_fill(joint_context(events), axis_mode=True)}))["h"]


def test_the_fill_carries_all_three_operand_roles() -> None:
    """The premise of the anchors below: the recorded sibling fill node holds one axis value and
    three further operands — the ambient b-tag weight, the extra weight factor and `sample=` — so
    every role the resolution rule is reached from is genuinely engaged by this program.
    """
    session, events = in_memory()
    by_label = gh.fill_nodes_by_label(three_role_fill(joint_context(events), axis_mode=False))
    assert list(by_label) == list(LABELS)
    assert len(axis_inputs(session, by_label[JOINT])) == 1
    assert len(other_inputs(session, by_label[JOINT])) == 3


def test_each_mode_moves_the_joint_universe_off_the_btag_only_one() -> None:
    """Per mode, the mechanism witness. Without it the parity anchor below would pass on an
    implementation that left BOTH modes on the old rule.

    The structural half is the one that isolates §5.1: the joint's AXIS VALUE is the observable's
    shifted member in each mode, the b-tag-only label's is nominal. In axis mode that is the joint
    leaving the nominal weight loop — its recorded node is the shifted group's. The executed half
    only adds that each mode ran three distinct universes end to end; the fanout gives the joint
    its own ambient weight member, so those inequalities survive a wrong axis value and cannot
    stand in for the structural half.
    """
    for axis_mode in (False, True):
        recorded = _recorded(axis_mode=axis_mode)
        assert recorded.shifted != recorded.nominal, "the candidate axis members must be distinct"
        assert recorded.axis_member[JOINT] == recorded.shifted, axis_mode
        assert recorded.axis_member[BTAG_ONLY] == recorded.nominal, axis_mode

    sibling = _sibling()
    assert not views_equal(sibling[JOINT], sibling[BTAG_ONLY]), "sibling mode"
    assert not views_equal(sibling[JOINT], sibling["jes_up"]), "sibling mode"

    axis = _axis()
    assert not views_equal(slice_label(axis, JOINT), slice_label(axis, BTAG_ONLY)), "axis mode"
    assert not views_equal(slice_label(axis, JOINT), slice_label(axis, "jes_up")), "axis mode"


def test_both_fill_modes_resolve_the_joint_label_alike() -> None:
    """§6.2's equality, extended to the joint label: the axis-mode histogram sliced at each label
    equals the sibling fill for that label bin-for-bin, flow included, on a WeightedMean storage.
    """
    axis = _axis()
    sibling = _sibling()
    assert set(sibling) == set(LABELS)
    for label in LABELS:
        assert views_equal(slice_label(axis, label), sibling[label]), label
