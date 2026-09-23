"""G11: weighted and sample storages on growth axes keep every stored field."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import boost_histogram as bh
import numpy as np
import pytest
from m64_growth_fixtures import TWO, assert_bitwise, eager, reference, run, slices

import graphed_histogram as gh

CATS = np.array([9, 7, 9, 7, 7, 5, 9, 5, 5, 3, 7, 3, 3, 1, 9, 1])
KS = np.array([-2, 0, 4, 2, -4, 2, 1, 0, 1, 6, -1, 3, 8, -3, 3, 0])
QUARTERS = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3])

AXES: dict[str, tuple[Callable[[], Any], np.ndarray[Any, Any], Callable[[Any], Any]]] = {
    "IntCategory": (lambda: bh.axis.IntCategory([], growth=True), CATS, lambda v: v),
    "Regular": (lambda: bh.axis.Regular(4, 0.0, 1.0, growth=True), KS, lambda v: (v + 0.5) * 0.25),
}

#: the fill keywords per storage, from the weight column `q` (whole quarters: every sum exact)
KWARGS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "Weight": lambda q: {"weight": q * 0.25},
    "Mean": lambda q: {"sample": q * 0.3},
    "WeightedMean": lambda q: {"weight": q * 0.25, "sample": q * 0.3},
}


def _raw(axis: str) -> np.ndarray[Any, Any]:
    raw: np.ndarray[Any, Any] = np.stack([AXES[axis][1], QUARTERS], axis=1).astype(np.int64)
    return raw


@pytest.mark.parametrize(("runner", "how"), TWO)
@pytest.mark.parametrize("storage", KWARGS)
@pytest.mark.parametrize("axis", AXES)
def test_g11_storages_on_growth_axes(axis: str, storage: str, runner: str, how: str) -> None:
    make, column, value = AXES[axis]
    kwargs = KWARGS[storage]

    def empty() -> bh.Histogram:
        return bh.Histogram(make(), storage=getattr(bh.storage, storage)())

    def build(ev: Any) -> Any:
        h = gh.boost.Histogram(make(), storage=getattr(bh.storage, storage)())
        return h.fill(value(ev[:, 0]), **kwargs(ev[:, 1]))

    got = run(_raw(axis), build, runner, how).result
    x, q = value(column), QUARTERS
    want = eager(empty, x, **kwargs(q))
    if storage == "Weight":
        assert_bitwise(got, want)
        return
    gv, wv = got.view(flow=True), want.view(flow=True)
    assert list(got.axes[0]) == list(want.axes[0])
    assert gv.dtype.names == wv.dtype.names
    for name in wv.dtype.names:
        assert not np.isnan(gv[name]).any(), name
        assert np.allclose(gv[name], wv[name]), name
    if runner == "seq":
        ref = reference(empty, [[((x[s],), kwargs(q[s]))] for s in slices(len(x))])
        assert_bitwise(got, ref)
