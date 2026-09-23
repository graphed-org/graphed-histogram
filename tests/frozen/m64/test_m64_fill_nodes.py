"""G13-G14: several fill nodes give partition-major category order; one fill node keeps eager order."""

from __future__ import annotations

import itertools
from typing import Any

import boost_histogram as bh
import graphed
import numpy as np
import pytest
from m64_growth_fixtures import TWO, assert_bitwise, eager, reference, run, slices

import graphed_histogram as gh

# Two string columns, two rows per partition.
FIRST = np.array(["e", "e", "d", "e", "c", "d", "e", "c"])
SECOND = np.array(["a", "a", "b", "a", "a", "b", "b", "a"])


def _strcat() -> bh.Histogram:
    return bh.Histogram(bh.axis.StrCategory([], growth=True), storage=bh.storage.Int64())


@pytest.mark.parametrize(("runner", "how"), TWO)
def test_g13_two_fill_calls_are_partition_major(runner: str, how: str) -> None:
    parts = slices(len(FIRST))
    partition_major = reference(_strcat, [[((FIRST[s],), {}), ((SECOND[s],), {})] for s in parts])
    by_partition = itertools.chain(*[[*FIRST[s].tolist(), *SECOND[s].tolist()] for s in parts])
    assert list(partition_major.axes[0]) == list(dict.fromkeys(by_partition))
    fill_major = eager(_strcat, FIRST)
    fill_major.fill(SECOND)
    assert list(partition_major.axes[0]) != list(fill_major.axes[0])

    def build(ev: Any) -> Any:
        h = gh.boost.Histogram(bh.axis.StrCategory([], growth=True), storage=bh.storage.Int64())
        return h.fill(ev[:, 0]).fill(ev[:, 1])

    got = run(np.stack([FIRST, SECOND], axis=1), build, runner, how)
    assert_bitwise(got.result, partition_major)


SHIFTED = np.array([5, 3, 7, 5, 3, 9, 1, 7], dtype=np.int64)


def _category_axis(h: bh.Histogram, kind: type) -> Any:
    (axis,) = (a for a in h.axes if type(a) is kind)
    return axis


@pytest.mark.parametrize(("runner", "how"), TWO)
def test_g13_an_axis_mode_value_variation_is_partition_major(runner: str, how: str) -> None:
    per_partition = [[*SHIFTED[s].tolist(), *(SHIFTED[s] + 100).tolist()] for s in slices(len(SHIFTED))]
    partition_major = list(dict.fromkeys(itertools.chain(*per_partition)))
    label_major = list(dict.fromkeys([*SHIFTED.tolist(), *(SHIFTED + 100).tolist()]))
    assert partition_major != label_major

    def build(ev: Any) -> Any:
        h = gh.boost.Histogram(bh.axis.IntCategory([], growth=True), storage=bh.storage.Int64())
        return h.fill(graphed.vary(ev, "sh", up=ev + 100), variation_axis=True)

    got = run(SHIFTED, build, runner, how, unpack=True).result
    assert list(_category_axis(got, bh.axis.IntCategory)) == partition_major


CATS = np.array([9, 7, 9, 7, 7, 5, 9, 5, 5, 3, 7, 3, 3, 1, 9, 1], dtype=np.int64)
QUARTERS = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3], dtype=np.int64)
FACTORS = {"nominal": 1.0, "wgt_up": 1.2, "wgt_down": 0.8}


def _weighted() -> bh.Histogram:
    return bh.Histogram(bh.axis.IntCategory([], growth=True), storage=bh.storage.Weight())


def _eager_label(label: str) -> bh.Histogram:
    return eager(_weighted, CATS, weight=QUARTERS * 0.25 * FACTORS[label])


def _weight_varied(axis_mode: bool) -> Any:
    def build(ev: Any) -> Any:
        w = ev[:, 1] * 0.25
        h = gh.boost.Histogram(bh.axis.IntCategory([], growth=True), storage=bh.storage.Weight())
        return h.fill(
            ev[:, 0], weight=graphed.vary(w, "wgt", up=w * 1.2, down=w * 0.8), variation_axis=axis_mode
        )

    return build


@pytest.mark.parametrize(("runner", "how"), TWO)
def test_g14_sibling_weight_variations_keep_eager_categories(runner: str, how: str) -> None:
    raw = np.stack([CATS, QUARTERS], axis=1)
    got = run(raw, _weight_varied(False), runner, how, unpack=True).result
    assert sorted(got) == sorted(FACTORS)
    for label, h in got.items():
        want = _eager_label(label)
        assert list(h.axes[0]) == list(want.axes[0]), label
        assert np.allclose(h.values(flow=True), want.values(flow=True)), label
        assert np.allclose(h.variances(flow=True), want.variances(flow=True)), label


@pytest.mark.parametrize(("runner", "how"), TWO)
def test_g14_axis_mode_weight_variations_keep_eager_categories(runner: str, how: str) -> None:
    raw = np.stack([CATS, QUARTERS], axis=1)
    got = run(raw, _weight_varied(True), runner, how, unpack=True).result
    categories = _category_axis(got, bh.axis.IntCategory)
    variation = _category_axis(got, bh.axis.StrCategory)
    assert list(categories) == list(_eager_label("nominal").axes[0])
    assert sorted(variation) == sorted(FACTORS)
    at = [type(a) for a in got.axes].index(bh.axis.StrCategory)
    for label in FACTORS:
        want = _eager_label(label)
        index = list(variation).index(label)
        assert np.allclose(np.take(got.values(), index, axis=at), want.values()), label
        assert np.allclose(np.take(got.variances(), index, axis=at), want.variances()), label
