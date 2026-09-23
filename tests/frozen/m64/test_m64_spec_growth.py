"""G1-G4: the spec codec carries growth, keeps fixed specs' bytes, and refuses what it cannot carry."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import boost_histogram as bh
import hist
import pytest
from m64_growth_fixtures import STORAGES

import graphed_histogram as gh

FIXED_V1 = (
    '{"axes":[{"bins":4,"metadata":{},"overflow":true,"start":0.0,"stop":8.0,'
    '"type":"Regular","underflow":true}],"storage":"Double","version":1}'
)

GROWTH_AXES: dict[str, tuple[Callable[[], Any], Callable[[], Any]]] = {
    "StrCategory": (lambda: bh.axis.StrCategory([], growth=True), lambda: bh.axis.StrCategory([])),
    "IntCategory": (lambda: bh.axis.IntCategory([], growth=True), lambda: bh.axis.IntCategory([])),
    "Integer": (lambda: bh.axis.Integer(0, 4, growth=True), lambda: bh.axis.Integer(0, 4)),
    "Regular": (lambda: bh.axis.Regular(4, 0.0, 1.0, growth=True), lambda: bh.axis.Regular(4, 0.0, 1.0)),
}


def test_g1_fixed_specs_keep_their_bytes() -> None:
    assert gh.spec_of(bh.Histogram(bh.axis.Regular(4, 0.0, 8.0))) == FIXED_V1


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("axis", GROWTH_AXES)
def test_g2_growth_specs_round_trip(axis: str, storage: str) -> None:
    grow, twin = GROWTH_AXES[axis]
    fixed = bh.axis.Regular(2, 0.0, 1.0)
    spec = gh.spec_of(bh.Histogram(grow(), fixed, storage=STORAGES[storage]()))
    parsed = json.loads(spec)
    assert parsed["version"] == 2
    assert parsed["axes"][0]["growth"] is True
    assert "growth" not in parsed["axes"][1]
    zero = gh.zero_of(spec)
    assert zero.axes[0] == grow()
    assert zero.axes[0].traits.growth
    assert zero.axes[1] == fixed
    assert gh.spec_of(zero) == spec
    twin_spec = gh.spec_of(bh.Histogram(twin(), fixed, storage=STORAGES[storage]()))
    assert gh.content_hash(spec) != gh.content_hash(twin_spec)
    assert gh.boost.Histogram(grow(), storage=STORAGES[storage]()).axes[0].traits.growth


def test_g3_unknown_versions_are_refused() -> None:
    spec = json.loads(FIXED_V1)
    spec["version"] = 3
    with pytest.raises(ValueError, match="unsupported histogram spec version 3"):
        gh.zero_of(json.dumps(spec))


REFUSED: dict[str, tuple[Callable[[], Any], str]] = {
    "growth Variable": (lambda: bh.axis.Variable([0, 0.5, 1], growth=True), "growth"),
    "circular Regular": (lambda: bh.axis.Regular(4, 0, 1, circular=True), "circular"),
    "circular Integer": (lambda: bh.axis.Integer(0, 4, circular=True), "circular"),
    "log Regular": (lambda: bh.axis.Regular(4, 1, 100, transform=bh.axis.transform.log), "transform"),
    "StrCategory no overflow": (lambda: bh.axis.StrCategory(["a"], overflow=False), "overflow"),
    "IntCategory no overflow": (lambda: bh.axis.IntCategory([1], overflow=False), "overflow"),
}

ADMITTED: dict[str, Callable[[], Any]] = {
    "dict metadata": lambda: bh.axis.Regular(4, 0, 1, metadata={"name": "pt"}),
    "hist named": lambda: hist.axis.Regular(4, 0, 1, name="x", label="x"),
    "no flow": lambda: bh.axis.Regular(4, 0, 1, underflow=False, overflow=False),
}


@pytest.mark.parametrize("case", REFUSED)
def test_g4_the_codec_refuses_what_it_cannot_carry(case: str) -> None:
    axis, lost = REFUSED[case]
    with pytest.raises(TypeError, match=lost):
        gh.spec_of(bh.Histogram(axis()))


@pytest.mark.parametrize("case", ADMITTED)
def test_g4_the_codec_admits_what_it_carries(case: str) -> None:
    axis = ADMITTED[case]()
    zero = gh.zero_of(gh.spec_of(bh.Histogram(axis)))
    assert zero.axes[0].traits == axis.traits
