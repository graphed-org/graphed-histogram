"""`gh.plan` and `Histogram.plan` take the source's `steps_per_file` unless told otherwise."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import pytest
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet
from graphed.core.execution import SequentialRunner

sys.path.append(str(Path(__file__).resolve().parents[2] / "frozen" / "m49"))

from m49_hist_fixtures import partitioned

import graphed_histogram as gh

MET = [10.0, 40.0, 70.0, 120.0, 30.0, 90.0]
COUNTS = [3, 2, 1, 0]
ROUTES = ("h.plan", "gh.plan")


def _plan(route: str, h: gh.Histogram, **kwargs: Any) -> Any:
    return h.plan(**kwargs) if route == "h.plan" else gh.plan({"met": h}, **kwargs)


def _counts(route: str, plan: Any) -> Any:
    value = SequentialRunner().run(plan).value
    filled: Any = value if route == "h.plan" else gh.unpack(value)["met"]
    return filled.values().tolist()


def _fresh() -> gh.Histogram:
    return gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())


@pytest.fixture
def parquet(tmp_path: Path) -> str:
    path = tmp_path / "events.parquet"
    ak.to_parquet(ak.Array({"met": MET}), path)
    return str(path)


def _parquet_hist(path: str, **read: Any) -> gh.Histogram:
    events = from_parquet("events", path, session=Session(AwkwardBackend()), **read)
    return _fresh().fill(events.met)


@pytest.mark.parametrize(("read", "steps"), [({"steps_per_file": 3}, 3), ({}, 1)])
@pytest.mark.parametrize("route", ROUTES)
def test_the_read_sets_the_task_count(
    route: str, read: dict[str, int], steps: int, parquet: str
) -> None:
    plan = _plan(route, _parquet_hist(parquet, **read))
    assert [t.partition.blind_step for t in plan.tasks] == list(range(steps))
    assert _counts(route, plan) == COUNTS


@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("route", ROUTES)
def test_an_explicit_value_overrides_the_read(route: str, steps: int, parquet: str) -> None:
    plan = _plan(route, _parquet_hist(parquet, steps_per_file=3), steps_per_file=steps)
    assert [t.partition.blind_step for t in plan.tasks] == list(range(steps))
    assert _counts(route, plan) == COUNTS


@pytest.mark.parametrize("route", ROUTES)
def test_a_source_without_its_own_steps_is_one_task(route: str) -> None:
    _, events, _ = partitioned(ak.Array({"met": MET}))
    plan = _plan(route, _fresh().fill(events.met))
    assert len(plan.tasks) == 1
    assert _counts(route, plan) == COUNTS
