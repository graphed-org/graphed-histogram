"""G5-G6: growth categories and growing Integer axes over a partitioned source equal one eager fill."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import boost_histogram as bh
import numpy as np
import pytest
from m64_growth_fixtures import ALL, PROCESS_RUNNERS, assert_bitwise, eager, run, slices

import graphed_histogram as gh

# Four partitions of four rows. First appearance over the dataset is reverse-sorted, so neither a
# sorted union nor a right-first (partition-reversed) union reproduces it.
LAYOUTS = {
    "disjoint": np.array([90, 80, 90, 80, 70, 60, 70, 70, 50, 40, 50, 40, 30, 20, 10, 30]),
    "overlapping": np.array([9, 7, 9, 7, 7, 5, 9, 5, 5, 3, 7, 3, 3, 1, 9, 1]),
}

AXES: dict[str, tuple[Callable[[], Any], Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]]] = {
    "IntCategory": (lambda: bh.axis.IntCategory([], growth=True), lambda a: a.astype(np.int64)),
    "StrCategory": (lambda: bh.axis.StrCategory([], growth=True), lambda a: a.astype(str)),
}


def _partition_categories(empty: Callable[[], bh.Histogram], data: np.ndarray[Any, Any]) -> list[list[Any]]:
    return [list(eager(empty, data[s]).axes[0]) for s in slices(len(data))]


@pytest.mark.parametrize(("runner", "how"), ALL)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("axis", AXES)
def test_g5_growth_categories_equal_one_eager_fill(axis: str, layout: str, runner: str, how: str) -> None:
    make, cast = AXES[axis]
    data = cast(LAYOUTS[layout])

    def empty() -> bh.Histogram:
        return bh.Histogram(make(), storage=bh.storage.Int64())

    parts = _partition_categories(empty, data)
    assert all(parts[i] != parts[j] for i in range(len(parts)) for j in range(i))
    first_appearance = list(dict.fromkeys(data.tolist()))
    assert first_appearance == sorted(first_appearance, reverse=True)
    assert first_appearance != list(dict.fromkeys(itertools.chain(*reversed(parts))))

    def build(ev: Any) -> Any:
        return gh.boost.Histogram(make(), storage=bh.storage.Int64()).fill(ev)

    got = run(data, build, runner, how)
    want = eager(empty, data)
    assert list(got.result.axes[0]) == first_appearance == list(want.axes[0])
    assert got.result.axes[0].traits.growth
    assert_bitwise(got.result, want)
    assert got.loader_calls == 0
    if runner in PROCESS_RUNNERS:
        assert got.scrambled(), got.stamps


# Partitions grow the declared [0, 4) below and above, and overlap one another.
INTEGER_DATA = np.array([-2, 0, 1, 3, 5, 2, 6, 1, -3, 4, 0, 2, 7, -1, 3, 8], dtype=np.int64)


@pytest.mark.parametrize(("runner", "how"), ALL)
def test_g6_growing_integer_equals_one_eager_fill(runner: str, how: str) -> None:
    def empty() -> bh.Histogram:
        return bh.Histogram(bh.axis.Integer(0, 4, growth=True), storage=bh.storage.Int64())

    edges = [eager(empty, INTEGER_DATA[s]).axes[0].edges.tobytes() for s in slices(len(INTEGER_DATA))]
    assert len(set(edges)) == len(edges)

    def build(ev: Any) -> Any:
        return gh.boost.Histogram(bh.axis.Integer(0, 4, growth=True), storage=bh.storage.Int64()).fill(ev)

    got = run(INTEGER_DATA, build, runner, how)
    want = eager(empty, INTEGER_DATA)
    assert want.axes[0].edges[0] < 0 and want.axes[0].edges[-1] > 4
    assert_bitwise(got.result, want)
    if runner in PROCESS_RUNNERS:
        assert got.scrambled(), got.stamps
