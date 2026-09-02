"""The member §7.2's shortfall refusal ADMITS: two staged fills that intern to ONE record node.

`marked` counts DISTINCT record ids, so an interned pair leaves marked == compiled and nothing
refuses — correctly, since record-time interning is the supported dedup path. What must then hold
is that both consumers still count the fill twice: `evaluate_ir` returns one value per distinct
output, so a reduce that ITERATES those values silently answers at half strength.

Both consumers, plus a single fill of the same data as the factor-two reference — the shape that
names the wrong answer rather than merely differing from it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import numpy as np
from graphed.core.execution import SequentialRunner

# the frozen tree's own fixtures, without putting its dir on the packaged pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2] / "frozen" / "m49"))

from m49_hist_fixtures import eager_weighted, partitioned

import graphed_histogram as gh

STEPS = 2
#: `w` and `w2` carry the same values through DIFFERENT source fields, so two fills over them
#: neither intern nor merge
EVENTS = ak.Array(
    {
        "x": [1.0, 4.0, 7.0, 2.5, 6.0] * 8,
        "w": [0.5, 1.0, 2.0, 1.5, 0.2] * 8,
        "w2": [0.5, 1.0, 2.0, 1.5, 0.2] * 8,
    }
)


def _hist(fills: int) -> Any:
    """`fills` IDENTICAL fill calls: identical inputs and params intern to ONE record node."""
    _session, events, _data = partitioned(EVENTS)
    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight())
    for _ in range(fills):
        h.fill(events.x, weight=[events.w])
    return h


def _values(hist: bh.Histogram) -> np.ndarray:
    return np.asarray(hist.view(flow=True)["value"])


def _run(hist: Any) -> np.ndarray:
    return _values(SequentialRunner().run(hist.plan(steps_per_file=STEPS)).value)


def test_two_interned_fills_intern_to_one_node_and_still_count_twice() -> None:
    twice = _hist(2)
    assert len({node.node_id for node in twice.fill_nodes()}) == 1, (
        "the instrument: without interning this is not the member the refusal admits"
    )

    want = eager_weighted(bins=4, lo=0.0, hi=8.0)
    for _ in range(2):
        want.fill(ak.to_numpy(EVENTS.x), weight=ak.to_numpy(EVENTS.w))

    once = _run(_hist(1))
    assert once.sum() > 0
    assert np.allclose(_run(twice), _values(want), rtol=1e-12)  # the 0.5x under-sum dies here
    assert np.allclose(_values(want), 2.0 * once, rtol=1e-12)

    grouped = gh.unpack(SequentialRunner().run(gh.plan({"h": _hist(2)}, steps_per_file=STEPS)).value)
    assert np.allclose(_values(grouped["h"]), _values(want), rtol=1e-12)  # already replicated


def test_a_pair_that_does_not_intern_still_sums_both_fills() -> None:
    """The control against "multiply by the staged count": two fills over DISTINCT expressions of
    the same values compile to two outputs, and must still total exactly twice."""
    _session, events, _data = partitioned(EVENTS)
    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight())
    h.fill(events.x, weight=[events.w])
    h.fill(events.x, weight=[events.w2])  # same values, a different node: no interning, no merge

    assert len({node.node_id for node in h.fill_nodes()}) == 2
    assert np.allclose(_run(h), 2.0 * _run(_hist(1)), rtol=1e-12)
