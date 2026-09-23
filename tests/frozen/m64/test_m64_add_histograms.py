"""G12: `gh.add_histograms` on eager histograms merges growth axes and refuses the rest."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any

import boost_histogram as bh
import numpy as np
import pytest
from m64_growth_fixtures import STORAGES, assert_bitwise

import graphed_histogram as gh

MEANS = ("Mean", "WeightedMean")


def _grow() -> Any:
    return bh.axis.Regular(4, 0.0, 1.0, growth=True)


def _filled(storage: str, axis: Any, x: Any, w: Any, s: Any, **meta: Any) -> bh.Histogram:
    h = bh.Histogram(axis, storage=STORAGES[storage](), **meta)
    kwargs: dict[str, Any] = {}
    if storage in ("Double", "Unlimited", "Weight", "WeightedMean"):
        kwargs["weight"] = w
    if storage in MEANS:
        kwargs["sample"] = s
    h.fill(x, **kwargs)
    return h


def test_g12a_fixed_axes_still_refuse_to_merge() -> None:
    with pytest.raises(ValueError):
        gh.add_histograms(bh.Histogram(bh.axis.Regular(4, 0, 1)), bh.Histogram(bh.axis.Regular(4, 0, 2)))


@pytest.mark.parametrize("storage", STORAGES)
def test_g12b_disjoint_growing_regular_extents(storage: str) -> None:
    rng = np.random.default_rng(64)
    # a grows only below, b only above: no bin receives both, and [0, 1) stays an empty gap
    xa, xb = -3 + 2 * rng.random(20) + 0.125, 3 + 2 * rng.random(20) + 0.125
    wa, wb, sa, sb = rng.random(20), rng.random(20), rng.normal(size=20), rng.normal(size=20)
    a = _filled(storage, _grow(), xa, wa, sa, metadata="ma")
    b = _filled(storage, _grow(), xb, wb, sb)
    before = pickle.dumps(a), pickle.dumps(b)

    got = gh.add_histograms(a, b)
    want = _filled(storage, _grow(), np.r_[xa, xb], np.r_[wa, wb], np.r_[sa, sb])
    assert (pickle.dumps(a), pickle.dumps(b)) == before
    assert type(got) is bh.Histogram
    assert got.metadata == "ma"
    assert np.asarray(got.axes[0].edges).tobytes() == np.asarray(want.axes[0].edges).tobytes()
    gv, wv = got.view(flow=True), want.view(flow=True)
    if storage not in MEANS:
        assert np.asarray(gv).tobytes() == np.asarray(wv).tobytes()
        return
    for name in wv.dtype.names:
        assert not np.isnan(gv[name]).any(), name
        assert np.allclose(gv[name], wv[name]), name
    edges = want.axes[0].edges

    def fixed() -> Any:
        return bh.axis.Regular(len(want.axes[0]), edges[0], edges[-1])

    fa, fb = _filled(storage, fixed(), xa, wa, sa), _filled(storage, fixed(), xb, wb, sb)
    assert np.asarray(gv).tobytes() == np.asarray((fa + fb).view(flow=True)).tobytes()


def _two_d(cats: list[int], ks: list[int]) -> bh.Histogram:
    h = bh.Histogram(bh.axis.IntCategory([], growth=True), _grow(), storage=bh.storage.Int64())
    h.fill(cats, (np.asarray(ks) + 0.5) * 0.25)
    return h


def test_g12c_the_merge_is_associative() -> None:
    a = _two_d([5, 3, 5], [-3, 0, 2])
    b = _two_d([3, 8, 1], [5, 1, -1])
    c = _two_d([1, 9, 5], [-6, 9, 3])
    assert len({np.asarray(h.axes[1].edges).tobytes() for h in (a, b, c)}) == 3
    add = gh.add_histograms
    assert_bitwise(add(add(a, b), c), add(a, add(b, c)))


def test_g12d_category_order_is_left_first() -> None:
    def cat(values: list[str]) -> bh.Histogram:
        h = bh.Histogram(bh.axis.StrCategory([], growth=True), storage=bh.storage.Int64())
        h.fill(values)
        return h

    a, b = cat(["c", "a"]), cat(["b", "a", "d"])
    assert list(gh.add_histograms(a, b).axes[0]) == ["c", "a", "b", "d"] == list((a + b).axes[0])
    assert list(gh.add_histograms(b, a).axes[0]) == ["b", "a", "d", "c"]


MISFITS: dict[str, Callable[[], Any]] = {
    "wider bins": lambda: bh.axis.Regular(4, 0.0, 2.0, growth=True),
    "shifted grid": lambda: bh.axis.Regular(4, 0.1, 1.1, growth=True),
    "other bin count": lambda: bh.axis.Regular(3, 0.0, 1.0, growth=True),
    "not growing": lambda: bh.axis.Regular(4, 0.0, 1.0),
    "other axis metadata": lambda: bh.axis.Regular(4, 0.0, 1.0, growth=True, metadata="other"),
}


@pytest.mark.parametrize("case", MISFITS)
def test_g12e_misaligned_growing_regulars_are_refused(case: str) -> None:
    a = bh.Histogram(_grow(), storage=bh.storage.Int64())
    a.fill([0.1, 1.3])
    assert a.axes[0].edges[-1] > 1.0
    b = bh.Histogram(MISFITS[case](), storage=bh.storage.Int64())
    b.fill([0.2])
    with pytest.raises(ValueError):
        gh.add_histograms(a, b)
