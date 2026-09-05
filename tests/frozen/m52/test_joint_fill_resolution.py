"""m52/C5 — §5.1: the axis-value member a fill uses for a label is the one that label's POINT names.

The fill lowering re-implemented the fallback rule privately: it string-tested the label against the
operand's own label list before touching an accessor, so a joint label the operand does not carry
reduced to the operand's nominal member no matter what its point said. The joint universe then got a
JES-up-evaluated b-tag SF applied to JES-NOMINAL kinematics — a silent wrong number, not an error.

Two anchors, one discriminating pair. The joint label's axis-value member must be the SHIFTED one;
the b-tag-only label's must stay NOMINAL. A resolution that simply prefers the newest or the
most-derived member passes the first and fails the second.

Both legs are asserted: the recorded axis-input node id, as a RELATION between the two named
candidate members (never a pinned integer — node ids are fixture-dependent), and the executed bin
contents against an eager awkward oracle whose one-at-a-time rows are validated in the same run.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import graphed
from m52_joint_fill_fixtures import (
    BTAG_ONLY,
    JOINT,
    LABELS,
    UNIVERSE,
    ambient_only_fill,
    axis_inputs,
    eager_universe,
    in_memory,
    joint_context,
    observable,
    views_equal,
)

import graphed_histogram as gh


class Program(NamedTuple):
    session: Any
    by_label: dict[str, Any]
    filled: dict[str, Any]
    shifted_id: int
    nominal_id: int


def _program(*, register_point: bool) -> Program:
    session, events = in_memory()
    context = joint_context(events, register_point=register_point)
    obs = observable(context)
    h = ambient_only_fill(context)
    by_label = gh.fill_nodes_by_label(h)
    return Program(
        session=session,
        by_label=dict(by_label),
        filled={label: session.materialize(node) for label, node in by_label.items()},
        shifted_id=graphed.universe(obs, "jes_up").node_id,
        nominal_id=graphed.nominal(obs).node_id,
    )


def test_the_eager_universe_oracle_reproduces_the_one_at_a_time_universes() -> None:
    """The instrument for both anchors below, run WITHOUT `points=` so it stays live at every stage
    of the milestone: every one-at-a-time universe the program already lowers correctly equals its
    `(jes scale, SF coefficient)` oracle row bin-for-bin. An oracle reproducing none of them would
    make the joint row unfalsifiable arithmetic."""
    program = _program(register_point=False)
    assert list(program.by_label) == list(LABELS)
    for label in ("nominal", "jes_up", "jes_down", BTAG_ONLY, "btag_hf_down"):
        assert views_equal(program.filled[label], eager_universe(*UNIVERSE[label])), label
    assert not views_equal(eager_universe(*UNIVERSE["nominal"]), eager_universe(*UNIVERSE[JOINT]))


def test_the_joint_labels_axis_value_member_is_the_shifted_one() -> None:
    """§5.1's headline: for a label whose point is `{jes: up, btag: hf_up}` the fill's axis value is
    the observable's `jes_up` member, and the executed universe carries BOTH coordinates."""
    program = _program(register_point=True)
    assert program.shifted_id != program.nominal_id, "the candidate axis members must be distinct"

    assert axis_inputs(program.session, program.by_label[JOINT]) == [program.shifted_id]
    assert axis_inputs(program.session, program.by_label[JOINT]) != [program.nominal_id]

    assert views_equal(program.filled[JOINT], eager_universe(*UNIVERSE[JOINT]))
    assert not views_equal(program.filled[JOINT], program.filled[BTAG_ONLY]), "silent-nominal"
    assert not views_equal(program.filled[JOINT], program.filled["jes_up"])


def test_a_btag_only_labels_axis_value_member_stays_nominal() -> None:
    """The refusing half: a one-coordinate `{btag: hf_up}` point restricts to the origin over the
    observable's `jes` axis, so it takes the observable's NOMINAL member. A resolution that prefers
    the newest or most-derived member reddens here while passing the anchor above."""
    program = _program(register_point=True)
    assert axis_inputs(program.session, program.by_label[BTAG_ONLY]) == [program.nominal_id]
    assert axis_inputs(program.session, program.by_label[BTAG_ONLY]) != [program.shifted_id]
    assert axis_inputs(program.session, program.by_label[BTAG_ONLY]) == axis_inputs(
        program.session, program.by_label["nominal"]
    )
    assert views_equal(program.filled[BTAG_ONLY], eager_universe(*UNIVERSE[BTAG_ONLY]))
