"""G7-G10: growing Regular axes merge to one eager fill (or to D5's reference where it differs)."""

from __future__ import annotations

from typing import Any

import boost_histogram as bh
import numpy as np
import pytest
from m64_growth_fixtures import ALL, PROCESS_RUNNERS, TWO, assert_bitwise, eager, reference, run, slices

import graphed_histogram as gh

INF, NAN = np.inf, np.nan


def _regular() -> bh.Histogram:
    return bh.Histogram(bh.axis.Regular(4, 0.0, 1.0, growth=True), storage=bh.storage.Int64())


def _deferred_regular(ev: Any) -> Any:
    return gh.boost.Histogram(bh.axis.Regular(4, 0.0, 1.0, growth=True), storage=bh.storage.Int64()).fill(ev)


def _edges(h: bh.Histogram) -> bytes:
    return bytes(np.asarray(h.axes[0].edges).tobytes())


# Quarter-bin centres, so no value is within float rounding of an edge.
REGULAR_LAYOUTS = {
    # partition 0 grows only below, the last only above; value ranges never overlap
    "disjoint": [
        [-0.875, -0.625, -0.875, -0.625],
        [0.125, 0.375, 0.125, 0.375],
        [0.625, 1.125, 1.375, 0.875],
        [2.125, 2.375, 1.875, 2.125],
    ],
    "overlapping": [
        [-0.375, 0.125, 1.125, 0.625],
        [-0.875, 0.625, 0.375, 0.125],
        [0.375, 1.625, -0.125, 0.875],
        [2.125, -0.625, 0.875, 0.125],
    ],
}


@pytest.mark.parametrize(("runner", "how"), ALL)
@pytest.mark.parametrize("layout", REGULAR_LAYOUTS)
def test_g7_growing_regular_on_a_dyadic_grid_equals_one_eager_fill(
    layout: str, runner: str, how: str
) -> None:
    data = np.array(REGULAR_LAYOUTS[layout], dtype=np.float64).ravel()
    parts = [_edges(eager(_regular, data[s])) for s in slices(len(data))]
    assert len(set(parts)) == len(parts)

    got = run(data, _deferred_regular, runner, how)
    want = eager(_regular, data)
    assert want.axes[0].edges[0] < 0 and want.axes[0].edges[-1] > 1
    assert got.result.axes[0].traits.growth
    assert_bitwise(got.result, want)
    if runner in PROCESS_RUNNERS:
        assert got.scrambled(), got.stamps


G8_KS = np.array([-3, 0, 1, 2, 5, 2, -1, 6, 7, 3, 9, 0])


def _tenths() -> bh.Histogram:
    return bh.Histogram(bh.axis.Regular(4, 0.0, 0.4, growth=True), storage=bh.storage.Int64())


def _off_grid(base: Any, other: Any) -> bool:
    """`other`'s edges are not a float-exact whole number of `base`'s bins away, or its width differs."""
    width = (base.edges[-1] - base.edges[0]) / base.size
    offset = (other.edges[0] - base.edges[0]) / width
    return bool(offset != round(offset) or (other.edges[-1] - other.edges[0]) / other.size != width)


@pytest.mark.parametrize(("runner", "how"), TWO)
def test_g8_growing_regular_on_a_non_dyadic_grid(runner: str, how: str) -> None:
    data = (G8_KS + 0.5) * 0.1
    declared = _tenths().axes[0]
    assert any(_off_grid(declared, eager(_tenths, data[s]).axes[0]) for s in slices(len(data)))

    got = run(data, lambda ev: gh.boost.Histogram(declared, storage=bh.storage.Int64()).fill(ev), runner, how)
    want = eager(_tenths, data)
    assert got.result.axes[0].size == want.axes[0].size
    assert np.asarray(got.result.view(flow=True)).tobytes() == np.asarray(want.view(flow=True)).tobytes()
    assert np.allclose(got.result.axes[0].edges, want.axes[0].edges, rtol=0, atol=1e-9 * 0.1)


FLOW_FINITE = [[-0.375, 0.125], [1.375, 0.625], [-1.125, 0.375], [2.625, 0.875]]


@pytest.mark.parametrize(("runner", "how"), TWO)
def test_g9_flows_stay_in_flow_bins(runner: str, how: str) -> None:
    data = np.array([[*finite, NAN, INF, -INF] for finite in FLOW_FINITE]).ravel()
    finite = data[np.isfinite(data)]
    # two calls: a non-finite value filled before a later growth moves in one eager call (D5 (c))
    want = eager(_regular, finite)
    want.fill(data[~np.isfinite(data)])
    got = run(data, _deferred_regular, runner, how)
    assert_bitwise(got.result, want)


# Values on quarter edges, and a -inf before a growing value inside one chunk.
EDGE_CHUNKS = [
    [0.25, 0.5, -INF, 1.25],
    [-0.5, 0.0, 1.0, 0.75],
    [NAN, 2.0, -0.25, 0.5],
    [1.5, -INF, -1.0, 0.25],
]


@pytest.mark.parametrize(("runner", "how"), TWO)
def test_g9_edge_and_non_finite_values_follow_the_reference(runner: str, how: str) -> None:
    data = np.array(EDGE_CHUNKS).ravel()
    want = reference(_regular, [[((data[s],), {})] for s in slices(len(data))])
    assert want.view(flow=True).tobytes() != eager(_regular, data).view(flow=True).tobytes()
    got = run(data, _deferred_regular, runner, how)
    assert_bitwise(got.result, want)


G10_CATS = np.array([9, 7, 9, 7, 7, 5, 9, 5, 5, 3, 7, 3, 3, 1, 9, 1])
G10_KS = np.array([-2, 0, 4, 2, -4, 2, 1, 0, 1, 6, -1, 3, 8, -3, 3, 0])


def _two_d() -> bh.Histogram:
    return bh.Histogram(
        bh.axis.IntCategory([], growth=True),
        bh.axis.Regular(4, 0.0, 1.0, growth=True),
        storage=bh.storage.Int64(),
    )


@pytest.mark.parametrize(("runner", "how"), ALL)
def test_g10_two_growth_axes(runner: str, how: str) -> None:
    raw = np.stack([G10_CATS, G10_KS], axis=1).astype(np.int64)

    def build(ev: Any) -> Any:
        h = gh.boost.Histogram(*_two_d().axes, storage=bh.storage.Int64())
        return h.fill(ev[:, 0], (ev[:, 1] + 0.5) * 0.25)

    got = run(raw, build, runner, how)
    want = eager(_two_d, G10_CATS, (G10_KS + 0.5) * 0.25)
    assert want.axes[1].edges[0] < 0 and want.axes[1].edges[-1] > 1
    assert_bitwise(got.result, want)
    if runner in PROCESS_RUNNERS:
        assert got.scrambled(), got.stamps
