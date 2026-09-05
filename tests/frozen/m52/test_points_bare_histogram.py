"""m52/C4 — §4.10: `graphed.points()` refuses a bare axis-mode histogram.

Points are a record-time fact. They are not carried on disk, and a label cannot be parsed back into
a point, so an executed histogram carrying a `"variation"` StrCategory axis has label strings and
nothing else. Answering `{label: {}}` there would assert that every executed universe is the origin
— a wrong answer dressed as an introspection result, which is why the refusal is the property.

The refusal needs its own control: an implementation that raises on everything would pass it. The
record-time half of the same program is asserted to answer, in the same file, with the joint point
this tree's fixture registers.
"""

from __future__ import annotations

import graphed
import pytest
from m52_joint_fill_fixtures import (
    JOINT,
    JOINT_POINT,
    LABELS,
    execute,
    joint_context,
    partitioned,
    three_role_fill,
    var_index,
)

import graphed_histogram as gh


def test_points_answers_on_the_record_time_shape() -> None:
    """The control: on the `Varied` the fill reads its ambient weight from, `points()` answers a
    label→point map keyed by exactly that container's labels, with `"nominal"` at the origin and the
    registered joint point rendered under its own label."""
    _session, events = partitioned()
    context = joint_context(events)
    result = graphed.points(graphed.weight(context))
    assert set(result) == set(graphed.labels(graphed.weight(context)))
    assert result["nominal"] == {}
    assert result[JOINT] == JOINT_POINT


def test_points_refuses_a_bare_axis_mode_histogram() -> None:
    """§4.10's refusal, on the executed axis-mode histogram — a `boost_histogram.Histogram` whose
    variation axis holds the label strings and no point at all.

    Built WITHOUT `points=`: the refusal is about what an executed histogram can carry, so it must
    not depend on a joint registration to reach the verb under test."""
    _session, events = partitioned()
    context = joint_context(events, register_point=False)
    executed = gh.unpack(execute({"h": three_role_fill(context, axis_mode=True)}))["h"]
    assert executed.axes[var_index(executed)].size == len(LABELS)  # it really is the axis-mode shape

    with pytest.raises((graphed.GraphedError, TypeError), match=r"(?i)point"):
        graphed.points(executed)
