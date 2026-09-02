"""m49/F6 — the m05 ordering witness, SCOPED to the monotone JES fixture (§5.1).

`jes_up > nominal > jes_down` is a property of a monotone jet-pt SCALE applied to a monotone
observable, NOT a property of shifts. §5.1 makes that explicit and this suite obeys it: the
ordering is asserted here and nowhere else. The JER-SF anchor asserts the opposite shape on a
shift whose scale is stochastic — bidirectional migration, no ordering at all.

It rides beside a cutflow witness because ordering alone is satisfied by a program that only
re-weights: the defining behaviour of a shift is per-universe re-derivation of the SELECTION.
"""

from __future__ import annotations

from typing import Any

import boost_histogram as bh
import graphed
import numpy as np
from graphed.core.execution import SequentialRunner
from m49_hist_fixtures import TOY_EVENTS, btag_sf, partitioned, shift_jets, ttbar_slice

import graphed_histogram as gh

REGION = "4j2b"
STEPS = 3
JES_LABELS = ("jes_down", "nominal", "jes_up")


def _program() -> dict[str, Any]:
    """One monotone-JES ttbar slice: the weighted HT observable, and an unweighted count of the
    events each universe selects."""
    _session, events, _data = partitioned(TOY_EVENTS)
    observable, sel_jets = ttbar_slice(shift_jets(events.Jet), REGION)
    central = graphed.nominal(sel_jets)
    weight = graphed.vary(
        btag_sf(sel_jets, 1.0), "btag", up=btag_sf(central, 1.03), down=btag_sf(central, 0.97)
    )
    ht = gh.boost.Histogram(bh.axis.Regular(40, 0, 800), storage=bh.storage.Double())
    ht.fill(observable, weight=[weight])
    counts = gh.boost.Histogram(bh.axis.Regular(40, 0, 800), storage=bh.storage.Int64())
    counts.fill(observable)
    return gh.unpack(SequentialRunner().run(gh.plan({"ht": ht, "n": counts}, steps_per_file=STEPS)).value)


def _mean_bin(hist: bh.Histogram) -> float:
    values = np.asarray(hist.view(flow=True), dtype=float)
    return float(np.average(np.arange(values.size, dtype=float), weights=np.maximum(values, 0.0)))


def test_the_monotone_jes_orders_the_ht_observable() -> None:
    ht = _program()["ht"]
    down, nominal, up = (_mean_bin(ht[label]) for label in JES_LABELS)
    assert down < nominal < up


def test_the_jes_universes_re_derive_their_own_selection() -> None:
    """§5.1's defining behaviour, on the unweighted count. An implementation that shifted only the
    observable and reused nominal's selection leaves these three equal."""
    counts = _program()["n"]
    selected = {label: int(counts[label].sum()) for label in JES_LABELS}
    assert len(set(selected.values())) == 3, selected
    assert all(n > 0 for n in selected.values())


def test_the_unweighted_count_carries_only_the_shift_labels() -> None:
    """The instrument for the cutflow witness: the counts must come from the observable's own
    labels. A count histogram that had picked up the weight labels would be comparing universes
    whose selections agree by construction."""
    assert sorted(_program()["n"]) == sorted(JES_LABELS)
