"""A deferred histogram holding counts no fill recorded is refused: a run builds from zero."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import pytest
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet
from graphed.core.execution import SequentialRunner

import graphed_histogram as gh

_AXIS = bh.axis.Regular(4, 0.0, 4.0)
_ENTRIES: dict[str, Callable[[gh.Histogram], Any]] = {
    "Histogram.plan": lambda h: SequentialRunner().run(h.plan()).value,
    "gh.plan": lambda h: gh.unpack(SequentialRunner().run(gh.plan({"h": h})).value)["h"],
    "graphed.materialize": graphed.materialize,
}


def _counts() -> bh.Histogram:
    eager = bh.Histogram(_AXIS)
    eager.fill([0.5, 1.5, 2.5])
    return eager


@pytest.fixture
def x(tmp_path: Path) -> Any:
    ak.to_parquet(ak.Array({"x": [1.5, 3.5]}), tmp_path / "x.parquet")
    session = Session(AwkwardBackend())
    return from_parquet("events", str(tmp_path / "x.parquet"), session=session).x


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("staged", [True, False], ids=["fills", "no-fills"])
def test_added_counts_are_refused(entry: str, staged: bool, x: Any) -> None:
    h = gh.Histogram(_AXIS) + _counts()
    if staged:
        h.fill(x)
    with pytest.raises(ValueError, match=r"no \.fill\(\) recorded \(sum 3\.0\).*add_histograms"):
        _ENTRIES[entry](h)


def _added(h: gh.Histogram) -> None:
    h += _counts()


def _written(h: gh.Histogram) -> None:
    h.view(flow=True)[0] = 2.0  # the underflow bin: the check reads flow bins too


@pytest.mark.parametrize("put", [_added, _written], ids=["+=", "view write"])
def test_in_place_add_and_view_writes_are_refused(put: Callable[[Any], None], x: Any) -> None:
    h = gh.Histogram(_AXIS)
    put(h)
    with pytest.raises(ValueError, match=r"no \.fill"):
        graphed.materialize(h.fill(x))


@pytest.mark.parametrize("entry", _ENTRIES)
def test_a_histogram_of_fills_alone_still_runs(entry: str, x: Any) -> None:
    filled = _ENTRIES[entry](gh.Histogram(_AXIS).fill(x))
    assert filled.values().tolist() == [0.0, 1.0, 0.0, 1.0]


def test_the_documented_fix_keeps_the_counts(x: Any) -> None:
    result = graphed.materialize(gh.Histogram(_AXIS).fill(x))
    assert gh.add_histograms(result, _counts()).values().tolist() == [1.0, 2.0, 1.0, 1.0]


def test_gh_plan_names_the_refused_output(x: Any) -> None:
    with pytest.raises(ValueError, match=r"^histogram 'DY/zmass' holds counts"):
        gh.plan(
            {"DY": {"pt": gh.Histogram(_AXIS).fill(x), "zmass": gh.Histogram(_AXIS) + _counts()}}
        )
