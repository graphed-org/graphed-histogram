"""m52/C4 — §4.10: `graphed.points()` refuses a bare axis-mode histogram.

Points are a record-time fact. They are not carried on disk, and a label cannot be parsed back into
a point, so an executed histogram carrying a `"variation"` StrCategory axis has label strings and
nothing else. Answering `{label: {}}` there would assert that every executed universe is the origin
— a wrong answer dressed as an introspection result, which is why the refusal is the property.

The refusal needs its own control: an implementation that raises on everything would pass it. The
record-time half of the same program is asserted to answer, in the same file, with the two
coordinates the fanout mints the joint from.
"""

from __future__ import annotations

from typing import Any

import graphed
import pytest
from m52_joint_fill_fixtures import (
    GRID_LABELS,
    JOINT,
    JOINT_POINT,
    execute,
    joint_context,
    partitioned,
    three_role_fill,
    var_index,
)

import graphed_histogram as gh


def test_points_answers_on_the_record_time_shape() -> None:
    """The control: on the `Varied` the fill reads its ambient weight from, `points()` answers a
    label→point map keyed by exactly that container's labels, with `"nominal"` at the origin and
    the joint's TWO coordinates rendered under its machine label. The placement does not gate this
    — the fanout mints the joint and its point either way; what a placement changes is which joints
    survive, asserted where the labels are listed.
    """
    _session, events = partitioned()
    context = joint_context(events)
    result = graphed.points(graphed.weight(context))
    assert set(result) == set(graphed.labels(graphed.weight(context)))
    assert result["nominal"] == {}
    assert result[JOINT] == JOINT_POINT


def test_points_refuses_a_bare_axis_mode_histogram() -> None:
    """§4.10's refusal, on the executed axis-mode histogram — a `boost_histogram.Histogram` whose
    variation axis holds the label strings and no point at all.

    Built WITHOUT a placement: the refusal is about what an executed histogram can carry, so it
    must not depend on a joint SELECTION to reach the verb under test.
    """
    _session, events = partitioned()
    context = joint_context(events, select_joint=False)
    executed: Any = gh.unpack(execute({"h": three_role_fill(context, axis_mode=True)}))["h"]
    # it really is the axis-mode shape: one axis entry per label the unpruned fanout minted
    assert executed.axes[var_index(executed)].size == len(GRID_LABELS)

    with pytest.raises((graphed.GraphedError, TypeError), match=r"(?i)point"):
        graphed.points(executed)
