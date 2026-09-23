"""G16: a pool rerun reproduces inexact growth-axis sums bit for bit, whatever the completion order."""

from __future__ import annotations

import pickle
from functools import reduce
from typing import Any

import boost_histogram as bh
import numpy as np
from m64_growth_fixtures import axis_state, eager, run, slices

import graphed_histogram as gh

# enough rows that bins collect from several partitions, so the float sums see the fold shape
_RNG = np.random.default_rng(16)
CATS = _RNG.integers(0, 12, 400)
KS = _RNG.integers(-6, 10, 400)
WK = _RNG.integers(1, 1000, 400)


def _empty() -> bh.Histogram:
    return bh.Histogram(
        bh.axis.IntCategory([], growth=True),
        bh.axis.Regular(4, 0.0, 1.0, growth=True),
        storage=bh.storage.Weight(),
    )


def _build(ev: Any) -> Any:
    h = gh.boost.Histogram(*_empty().axes, storage=bh.storage.Weight())
    return h.fill(ev[:, 0], (ev[:, 1] + 0.5) * 0.25, weight=ev[:, 2] * 0.001 + 0.0137)


def test_g16_pool_reruns_are_bitwise_identical() -> None:
    parts = [
        eager(_empty, CATS[s], (KS[s] + 0.5) * 0.25, weight=WK[s] * 0.001 + 0.0137) for s in slices(len(CATS))
    ]
    left = reduce(gh.add_histograms, parts)
    right = reduce(lambda acc, h: gh.add_histograms(h, acc), reversed(parts))
    assert [axis_state(a) for a in left.axes] == [axis_state(a) for a in right.axes]
    assert np.asarray(left.view(flow=True)).tobytes() != np.asarray(right.view(flow=True)).tobytes()

    raw = np.stack([CATS, KS, WK], axis=1)
    first = run(raw, _build, "procpool3", "h.plan")
    second = run(raw, _build, "procpool3", "h.plan", reversed=True)
    assert first.order() != second.order()
    assert pickle.dumps(first.result) == pickle.dumps(second.result)
