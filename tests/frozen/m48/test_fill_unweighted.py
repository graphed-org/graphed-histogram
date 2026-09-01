"""m48/H11 — §6.1d: `unweighted=True` suppresses BOTH weight sources, and carries no labels.

Today's signature has no such parameter. §6.1d binds it as a dual suppression — the AMBIENT weight
AND any explicit `weight=[...]` factor — because a counts histogram carries no weight at all, and
supplying both `unweighted=True` and a non-`None` `weight=` in one call is a RECORD-TIME error
naming both (§2.5: validation over convention).

The label consequence is the sharp half. A fill's label set is computed over the factors it
ACTUALLY APPLIES, so a contexted `unweighted=True` fill whose only variation source is the ambient
registry is UNVARIED and returns a BARE `hist` (§6.1a) — not a `{label: hist}` of per-universe
identical counts, which is what an implementation that suppresses the weight VALUES but keeps the
label set would produce.

Stated consequence, not a defect: "suppress the ambient weight but apply my own factor" is not
expressible from a contexted program in v1 — the only opt-out kills the explicit factor too. That
is parked in §11.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
import pytest
from graphed import GraphedError
from graphed.awkward import gnano
from graphed.core.execution import SequentialRunner
from vary_hist_fixtures import (
    EVENTS,
    counts,
    eager_counts,
    in_memory_events,
    partitioned_events,
    weighted,
)

import graphed_histogram as gh


def _weighted_context(source: Any) -> Any:
    """A context whose ONLY variation source is the ambient registry."""
    ambient = source.MET.pt * 0.01
    return graphed.vary(
        gnano.events(source), "pu", ambient, is_weight=True, up=ambient * 1.1, down=ambient * 0.9
    )


def test_an_unweighted_fill_equals_an_unweighted_eager_reference() -> None:
    session, root = in_memory_events()
    ctx = _weighted_context(root)
    h = counts()
    h.fill(ctx.MET.pt, unweighted=True)

    want = eager_counts()
    want.fill(ak.to_numpy(EVENTS.MET.pt))
    got = session.materialize(h.fill_nodes()[0])
    assert np.array_equal(got.view(flow=True), want.view(flow=True))


def test_the_suppressed_ambient_weight_contributes_NO_labels() -> None:
    """The discriminator against suppressing the weight VALUES while keeping the label set: that
    implementation returns three per-universe-identical counts histograms, this one returns one."""
    _session, source, _data = partitioned_events()
    ctx = _weighted_context(source)
    h = counts()
    h.fill(ctx.MET.pt, unweighted=True)
    assert set(gh.fill_nodes_by_label(h)) == {"nominal"}
    assert h.staged_fills() == 1

    result = gh.unpack(SequentialRunner().run(gh.plan({"met": h}, steps_per_file=3)).value)
    assert isinstance(result["met"], bh.Histogram)
    assert not isinstance(result["met"], Mapping)
    assert result["met"].sum(flow=True) == len(EVENTS)


def test_the_same_program_without_the_flag_does_carry_the_ambient_labels() -> None:
    """The instrument: the ambient registry really does reach this fill, so the bare shape above
    is the flag's doing and not an empty registry."""
    _session, source, _data = partitioned_events()
    ctx = _weighted_context(source)
    h = weighted()
    h.fill(ctx.MET.pt)
    assert set(gh.fill_nodes_by_label(h)) == {"nominal", "pu_up", "pu_down"}


def test_unweighted_together_with_an_explicit_weight_is_a_record_time_error() -> None:
    """§2.5: the two arguments contradict each other, so the call is refused rather than one of
    them silently winning. Record-time — nothing is staged and nothing runs."""
    _session, root = in_memory_events()
    ctx = _weighted_context(root)
    h = counts()
    with pytest.raises(GraphedError) as excinfo:
        h.fill(ctx.MET.pt, weight=[ctx.MET.pt * 0.01], unweighted=True)
    message = str(excinfo.value)
    assert "unweighted" in message and "weight" in message
    assert h.staged_fills() == 0
