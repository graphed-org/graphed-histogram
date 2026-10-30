"""`gh.plan` and `Histogram.plan` take the source's `steps_per_file` unless told otherwise."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import numpy as np
import pytest
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet
from graphed.core import Partition
from graphed.core.execution import SequentialRunner
from graphed.numpy import NumpyBackend
from graphed.numpy.forms import NumpyForm

import graphed_histogram as gh

MET = [10.0, 40.0, 70.0, 120.0, 30.0, 90.0]
ROUTES = ("h.plan", "gh.plan")


def _plan(route: str, h: gh.Histogram, **kwargs: Any) -> Any:
    return h.plan(**kwargs) if route == "h.plan" else gh.plan({"met": h}, **kwargs)


def _counts(route: str, plan: Any) -> list[int]:
    value = SequentialRunner().run(plan).value
    filled = value if route == "h.plan" else gh.unpack(value)["met"]
    counts: list[int] = np.asarray(filled.values()).tolist()
    return counts


def _fresh() -> gh.Histogram:
    return gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())


@pytest.fixture
def parquet(tmp_path: Path) -> Iterator[str]:
    path = tmp_path / "events.parquet"
    ak.to_parquet(ak.Array({"met": MET}), path)
    yield str(path)


def _parquet_hist(path: str, **read: Any) -> gh.Histogram:
    events = from_parquet("events", path, session=Session(AwkwardBackend()), **read)
    return _fresh().fill(events.met)


@pytest.mark.parametrize("route", ROUTES)
def test_the_reads_steps_set_the_task_count(route: str, parquet: str) -> None:
    plan = _plan(route, _parquet_hist(parquet, steps_per_file=3))
    assert [t.partition.blind_step for t in plan.tasks] == [0, 1, 2]
    assert _counts(route, plan) == [3, 2, 1, 0]


@pytest.mark.parametrize("route", ROUTES)
def test_an_explicit_value_overrides_the_read(route: str, parquet: str) -> None:
    plan = _plan(route, _parquet_hist(parquet, steps_per_file=3), steps_per_file=2)
    assert [t.partition.blind_step for t in plan.tasks] == [0, 1]
    assert _counts(route, plan) == [3, 2, 1, 0]


@pytest.mark.parametrize("route", ROUTES)
def test_a_read_that_sets_nothing_is_one_task(route: str, parquet: str) -> None:
    assert len(_plan(route, _parquet_hist(parquet)).tasks) == 1


class _StepsLessSource:
    """A partitioned source declaring no `steps_per_file` of its own."""

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind("mem://x", "", s, steps_per_file) for s in range(steps_per_file)
        )

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> Any:
        part = partition.resolve(len(MET))
        return np.asarray(MET)[part.entry_start : part.entry_stop]


@pytest.mark.parametrize("route", ROUTES)
def test_a_source_without_its_own_steps_is_one_task(route: str) -> None:
    form = NumpyForm(np.dtype("float64"), shape=(None,))
    x = Session(NumpyBackend()).source("x", form=form, data=_StepsLessSource())
    assert len(_plan(route, _fresh().fill(x)).tasks) == 1
