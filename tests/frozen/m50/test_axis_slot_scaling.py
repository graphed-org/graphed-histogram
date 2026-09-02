"""m50 — §6.2 scaling, STRUCTURALLY (R0.10a — no wall-clock in a frozen test).

Per partition, axis mode ships ONE combine payload entry against `N+1` in sibling mode — the length
of the per-partition combine payload under §6.1c's key shape: one `(output, label)` entry per label
in sibling mode, ONE `(output, None)` entry in axis mode. Scoped to WEIGHT labels only and one
output, so the `1`-vs-`N+1` count is unambiguous (a mixed program's axis side still records `1 + |S|`
fill nodes). Plus bin-for-bin equality, so the single slot is not achieved by dropping universes.

The "allocates one histogram object" and the N≈100 wall-clock comparison are implementer-report
measurements (R0.11), NOT frozen here. Exercises the axis-mode slot, absent at the baseline: FAILS
today at the opt-in keyword.
"""

from __future__ import annotations

from m50_axis_fixtures import (
    execute,
    fill_weight_program,
    partitioned,
    slice_label,
    views_equal,
)

import graphed_histogram as gh

WEIGHT_LABELS = ("nominal", "wgt_up", "wgt_down")  # N+1 = 3, N = 2 non-nominal weight variations


def test_axis_mode_ships_one_slot_where_sibling_ships_n_plus_one() -> None:
    _s, ea, _srca = partitioned()
    axis_value = execute({"h": fill_weight_program(True, source=ea)})
    _s2, es, _srcs = partitioned()
    sibling_value = execute({"h": fill_weight_program(False, source=es)})

    assert set(axis_value) == {("h", None)}  # exactly one combine entry, keyed by the MODE
    assert len(axis_value) == 1
    assert len(sibling_value) == len(WEIGHT_LABELS)  # N+1 = 3 (output, label) entries
    assert len(gh.unpack(axis_value)) == 1  # one output


def test_the_single_axis_slot_carries_every_universe_bin_for_bin() -> None:
    """The single slot is the WHOLE variation family, not a collapse that dropped universes: sliced
    per label it equals the sibling decomposition."""
    _s, ea, _srca = partitioned()
    axis = gh.unpack(execute({"h": fill_weight_program(True, source=ea)}))["h"]
    _s2, es, _srcs = partitioned()
    sibling = gh.unpack(execute({"h": fill_weight_program(False, source=es)}))["h"]
    for label in WEIGHT_LABELS:
        assert views_equal(slice_label(axis, label), sibling[label]), label
