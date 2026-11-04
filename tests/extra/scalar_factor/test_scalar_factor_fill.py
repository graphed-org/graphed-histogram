"""A `graphed.vary(1.0, ...)` factor on a fill's `weight=`."""

from __future__ import annotations

from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
import pytest
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak

from graphed_histogram import Histogram
from graphed_histogram.boost import fill_nodes_by_label

EVENTS = ak.Array({"x": [0.5, 1.5, 2.5, 3.5], "w": [2.0, 3.0, 5.0, 7.0]})
JETS = ak.Array({"pt": [[0.5, 1.5], [], [2.5], [3.5, 0.5]]})
UP, DOWN = 1.025, 0.975


def _events(events: ak.Array = EVENTS) -> Any:
    return from_awkward("Events", events, session=Session(AwkwardBackend()))


def _lumi() -> Any:
    return graphed.vary(1.0, "lumi", up=UP, down=DOWN)


def _filled(array: Any, weight: Any) -> dict[str, np.ndarray[Any, Any]]:
    hist = Histogram(bh.axis.Regular(4, 0, 4), storage=bh.storage.Weight())
    hist.fill(array, weight=weight)
    return {
        label: array.session.materialize(node).view()
        for label, node in fill_nodes_by_label(hist).items()
    }


def _same(got: dict[str, Any], want: dict[str, Any]) -> None:
    assert list(got) == list(want) == ["nominal", "lumi_up", "lumi_down"]
    for label, view in want.items():
        assert np.array_equal(got[label], view)


def _ones_form(ev: Any) -> Any:
    one = gak.ones_like(ev.w)
    return ev.w * graphed.vary(one, "lumi", up=one * UP, down=one * DOWN)


@pytest.mark.parametrize(
    "weight",
    [
        lambda ev, lumi: [ev.w, lumi],
        lambda ev, lumi: [lumi, ev.w],
    ],
)
def test_a_factor_beside_an_array_fills_as_the_ones_form_does(weight: Any) -> None:
    ev, by_hand = _events(), _events()
    _same(_filled(ev.x, weight(ev, _lumi())), _filled(by_hand.x, [_ones_form(by_hand)]))


def test_a_scalar_constant_keeps_its_place_in_the_product() -> None:
    ev, by_hand = _events(), _events()
    _same(_filled(ev.x, [2.0, ev.w, _lumi()]), _filled(by_hand.x, [2.0, _ones_form(by_hand)]))


def test_a_factor_alone_is_ones_times_it() -> None:
    ev, by_hand = _events(), _events()
    one = gak.ones_like(by_hand.x)
    varied = graphed.vary(one, "lumi", up=one * UP, down=one * DOWN)
    _same(_filled(ev.x, _lumi()), _filled(by_hand.x, varied))


def test_two_factors_add_their_universes() -> None:
    ev = _events()
    pileup = graphed.vary(1.0, "pileup", up=1.05)
    views = _filled(ev.x, [ev.w, _lumi(), pileup])
    assert list(views) == ["nominal", "lumi_up", "lumi_down", "pileup_up"]
    assert views["pileup_up"]["value"].tolist() == [2.0 * 1.05, 3.0 * 1.05, 5.0 * 1.05, 7.0 * 1.05]


def test_a_jagged_value_takes_a_factor_alone() -> None:
    ev = _events(JETS)
    views = _filled(ev.pt, _lumi())
    assert views["lumi_up"]["value"].tolist() == [2 * UP, 1 * UP, 1 * UP, 1 * UP]


def test_a_factor_lands_on_the_varied_array_beside_it() -> None:
    ev = _events()
    varied = graphed.vary(ev.w, "sf", up=ev.w * 1.1)
    assert list(_filled(ev.x, [varied, _lumi()])) == ["nominal", "sf_up", "lumi_up", "lumi_down"]


@pytest.mark.parametrize(
    ("fill", "message"),
    [
        (lambda ev: ([ev[["w"]], _lumi()], ev.x), "weight got a record"),
        (lambda ev: (_lumi(), ev.pt), "axis 1 needs one value per entry of axis 0"),
    ],
)
def test_a_refused_fill_with_a_factor_names_its_type_and_adds_no_node(
    fill: Any, message: str
) -> None:
    ev = _events(ak.zip({"x": EVENTS.x, "w": EVENTS.w, "pt": JETS.pt}, depth_limit=1))
    (weight, second), first = fill(ev), ev.x
    hist = Histogram(bh.axis.Regular(4, 0, 4), bh.axis.Regular(4, 0, 4))
    before = ev.session.node_count()
    with pytest.raises(graphed.GraphedTypeError, match=message):
        hist.fill(first, second, weight=weight)
    assert ev.session.node_count() == before


STRINGS = ak.Array({"s": ["a", "b", "a", "b"], "x": [0.5, 1.5, 2.5, 3.5]})


def test_a_factor_alone_takes_its_ones_from_a_numeric_value_after_a_string() -> None:
    ev = _events(STRINGS)
    hist = Histogram(bh.axis.StrCategory(["a", "b"]), bh.axis.Regular(4, 0, 4))
    hist.fill(ev.s, ev.x, weight=_lumi())
    views = {k: ev.session.materialize(n).view() for k, n in fill_nodes_by_label(hist).items()}
    assert views["lumi_up"].sum() == pytest.approx(4 * UP)


def test_a_factor_alone_on_strings_only_is_refused_and_adds_no_node() -> None:
    s = _events(STRINGS).s
    before = s.session.node_count()
    with pytest.raises(graphed.GraphedTypeError, match="needs a numeric value or an array weight"):
        Histogram(bh.axis.StrCategory(["a", "b"])).fill(s, weight=_lumi())
    assert s.session.node_count() == before
