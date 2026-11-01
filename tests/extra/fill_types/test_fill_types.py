"""A fill the recorded types rule out is refused at the fill line, before any data is read."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import hist.graphed as hg
import numpy as np
import pytest
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.errors import GraphedTypeError
from graphed.numpy import NumpyBackend, from_array

import graphed_histogram as gh

EVENTS = ak.Array(
    {
        "met": [1.0, 2.0, 3.0],
        "njet": [2, 0, 1],
        "w": [1.0, 2.0, 3.0],
        "maybe": [1.0, None, 3.0],
        "cat": ["a", "b", "a"],
        "jet_pt": [[1.0, 2.0], [], [3.0]],
        "jet_eta": [[0.5, 1.5], [], [2.5]],
        "jet_maybe": [[1.0, None], [], [3.0]],
        "muon": [{"pt": 1.0}, {"pt": 2.0}, {"pt": 3.0}],
    }
)


def _events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, from_awkward("events", EVENTS, session=session)


def _one_axis() -> Any:
    return hg.Hist.new.Reg(10, 0, 5, name="met").Double()


def _two_axes() -> Any:
    return hg.Hist.new.Reg(10, 0, 5, name="met").Reg(5, 0, 5, name="njet").Double()


def _sum(session: Session, histogram: Any) -> float:
    return float(session.materialize(histogram.fill_nodes()[0]).sum(flow=True))  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("fill", "message"),
    [
        (
            lambda ev: _two_axes().fill(met=ev.met, njet=ev.jet_pt),
            r"axis 'njet' needs one value per entry of axis 'met' \('## \* float64'\); "
            r"got '## \* var \* float64'; flatten it or pick one per entry",
        ),
        (
            lambda ev: _two_axes().fill(met=ev.jet_pt, njet=ev.met),
            r"axis 'met' needs one value per entry of axis 'njet'",
        ),
        (
            lambda ev: _two_axes().fill(met=ev, njet=ev.njet),
            r"axis 'met' got a record \('## \* \{",
        ),
        (lambda ev: _one_axis().fill(met=ev.muon), r"axis 'met' got a record"),
        (lambda ev: _one_axis().fill(met=ev.met, weight=ev.muon), r"weight got a record"),
        (lambda ev: _one_axis().fill(met=ev.met, weight=[ev.w, ev.muon]), r"weight got a record"),
    ],
)
def test_a_ruled_out_fill_is_refused_before_anything_is_recorded(
    fill: Callable[[Any], Any], message: str
) -> None:
    _, ev = _events()
    with pytest.raises(GraphedTypeError, match=message) as caught:
        fill(ev)
    assert caught.value.provenance.filename == __file__


def test_a_record_sample_is_refused() -> None:
    _, ev = _events()
    with pytest.raises(GraphedTypeError, match="sample got a record"):
        hg.Hist.new.Reg(10, 0, 5, name="met").Mean().fill(met=ev.met, sample=ev.muon)


def test_a_refusal_points_at_the_fill_line_and_adds_no_node() -> None:
    session, ev = _events()
    met, jet_pt, histogram = ev.met, ev.jet_pt, _two_axes()
    before = session.node_count()
    line = sys._getframe().f_lineno + 2
    with pytest.raises(GraphedTypeError) as caught:
        histogram.fill(met=met, njet=jet_pt)
    assert caught.value.provenance.lineno == line
    assert session.node_count() == before
    assert histogram.fill_nodes() == []


def test_an_unnamed_axis_is_named_by_position() -> None:
    _, ev = _events()
    h = gh.boost.Histogram(bh.axis.Regular(5, 0, 5), bh.axis.Regular(5, 0, 5))
    with pytest.raises(GraphedTypeError, match="axis 1 needs one value per entry of axis 0"):
        h.fill(ev.met, ev.jet_pt)


def test_a_varied_operand_is_checked_on_its_nominal_member() -> None:
    _, ev = _events()
    shifted = graphed.vary(ev.jet_pt, "jes", up=ev.jet_pt * 2.0, down=ev.jet_pt * 0.5)
    with pytest.raises(GraphedTypeError, match="axis 'njet' needs one value per entry"):
        _two_axes().fill(met=ev.met, njet=shifted)


def test_two_numpy_ranks_are_refused_too() -> None:
    session = Session(NumpyBackend())
    flat = from_array("flat", np.zeros(3), session=session)
    wide = from_array("wide", np.zeros((3, 2)), session=session)
    with pytest.raises(GraphedTypeError, match="axis 'njet' needs one value per entry"):
        _two_axes().fill(met=flat, njet=wide)


@pytest.mark.parametrize(
    ("fill", "total"),
    [
        (lambda ev: _one_axis().fill(met=ev.jet_pt), 3.0),
        (lambda ev: _one_axis().fill(met=gak.flatten(ev.jet_pt)), 3.0),
        (lambda ev: _one_axis().fill(met=ev.maybe), 2.0),
        (lambda ev: _one_axis().fill(met=ev.jet_maybe), 2.0),
        (lambda ev: _two_axes().fill(met=ev.jet_pt, njet=ev.jet_eta), 3.0),
        (lambda ev: _two_axes().fill(met=ev.jet_pt, njet=1.0), 3.0),
        (lambda ev: _two_axes().fill(met=2.0, njet=ev.jet_pt), 3.0),
        (lambda ev: _one_axis().fill(met=ev.met, weight=[ev.w, 2.0]), 12.0),
        (lambda ev: _one_axis().fill(met=ev.jet_pt, weight=ev.w), 5.0),
        (lambda ev: _one_axis().fill(met=ev.met, weight=ev.jet_pt), 6.0),
        (lambda ev: _one_axis().fill(met=ev.met, weight=[gak.sum(ev.met, axis=None)]), 18.0),
        (lambda ev: _one_axis().fill(met=ev.muon.pt), 3.0),
        (
            lambda ev: hg.Hist.new.StrCat(["a", "b"], name="c").Reg(5, 0, 5, name="met").Double()
            .fill(c=ev.cat, met=ev.met),
            3.0,
        ),
    ],
)  # fmt: skip
def test_a_valid_fill_records_and_evaluates(fill: Callable[[Any], Any], total: float) -> None:
    session, ev = _events()
    assert _sum(session, fill(ev)) == total


def test_a_sampled_fill_of_a_jagged_value_stays_valid() -> None:
    session, ev = _events()
    filled = hg.Hist.new.Reg(10, 0, 5, name="met").Mean().fill(met=ev.jet_pt, sample=ev.met)
    assert session.materialize(filled.fill_nodes()[0]).sum(flow=True).count == 3.0  # type: ignore[attr-defined]
