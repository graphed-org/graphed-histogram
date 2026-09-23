"""Shared fixtures for the m64 growth-axis anchors.

Module-level so spawned workers can import the source. The `m64_` prefix keeps this helper from
binding another milestone's module: `tests/frozen` is collected in one pytest process.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import boost_histogram as bh
import numpy as np
from graphed import Session
from graphed.core import Partition
from graphed.core.execution import SequentialRunner, WorkerResources
from graphed.numpy import NumpyBackend
from graphed.numpy.forms import NumpyForm
from graphed_exec_local import ProcessExecutor, ProcessPoolExecutor

import graphed_histogram as gh

N_PARTITIONS = 4
#: per-partition sleep unit on the process runners; stamp gaps dwarf any clock resolution
DT = 0.1
LOADER_MARK = "loader"

RUNNERS: dict[str, Callable[[], Any]] = {
    "seq": SequentialRunner,
    "procexec2": lambda: ProcessExecutor(max_workers=2),
    "procpool3": lambda: ProcessPoolExecutor(max_workers=3),
}
PROCESS_RUNNERS = ("procexec2", "procpool3")
#: "all runners": every runner through both plan builders
ALL = [(r, how) for r in RUNNERS for how in ("h.plan", "gh.plan")]
#: "two runners": the sequential reference and a pool, through `gh.plan`
TWO = [("seq", "gh.plan"), ("procpool3", "gh.plan")]

STORAGES: dict[str, Callable[[], Any]] = {
    "Int64": bh.storage.Int64,
    "Double": bh.storage.Double,
    "AtomicInt64": bh.storage.AtomicInt64,
    "Unlimited": bh.storage.Unlimited,
    "Weight": bh.storage.Weight,
    "Mean": bh.storage.Mean,
    "WeightedMean": bh.storage.WeightedMean,
}

Fill = tuple[tuple[Any, ...], dict[str, Any]]


@dataclass
class ArraySource:
    """A `PartitionedSource` over one numpy array that records completion stamps.

    Partition `i` of `n` sleeps `(n - i) * dt` (or `(i + 1) * dt` reversed) and then writes
    `<stamp_dir>/<i>`; the stamp follows the sleep so it records completion, not key order.
    """

    data: np.ndarray[Any, Any]
    stamp_dir: str
    dt: float = 0.0
    reversed: bool = False

    def __call__(self) -> np.ndarray[Any, Any]:
        # a marker file, because a counter on this object stays in each worker's own copy
        with open(os.path.join(self.stamp_dir, f"{LOADER_MARK}-{os.getpid()}-{time.time_ns()}"), "w"):
            pass
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://m64", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(
        self, partition: Partition, columns: Any, resources: WorkerResources
    ) -> np.ndarray[Any, Any]:
        i, n = partition.blind_step, partition.blind_n_steps
        assert i is not None and n is not None
        time.sleep(((i + 1) if self.reversed else (n - i)) * self.dt)
        with open(os.path.join(self.stamp_dir, str(i)), "w") as f:
            f.write(str(time.time_ns()))
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


@dataclass
class Run:
    result: Any
    stamps: dict[int, int]
    loader_calls: int

    def scrambled(self) -> bool:
        """Some partition completed before a lower-keyed one."""
        return any(self.stamps[j] < self.stamps[i] for i in self.stamps for j in self.stamps if j > i)

    def order(self) -> list[int]:
        return sorted(self.stamps, key=self.stamps.__getitem__)


def run(
    data: np.ndarray[Any, Any],
    build: Callable[[Any], Any],
    runner: str,
    how: str,
    *,
    n: int = N_PARTITIONS,
    reversed: bool = False,
    unpack: bool = False,
) -> Run:
    """Record `build(source_array)` in a fresh `Session` over `data` and run it partitioned.

    `build` returns the deferred histogram. `unpack=True` returns `gh.unpack(value)["h"]`.
    """
    stamp_dir = tempfile.mkdtemp(prefix="m64-")
    session = Session(NumpyBackend())
    src = ArraySource(data, stamp_dir, 0.0 if runner == "seq" else DT, reversed)
    ev = session.source("ev", form=NumpyForm(data.dtype, shape=(None, *data.shape[1:])), data=src)
    h = build(ev)
    plan = h.plan(steps_per_file=n) if how == "h.plan" else gh.plan({"h": h}, steps_per_file=n)
    value = RUNNERS[runner]().run(plan).value
    if how == "gh.plan":
        value = gh.unpack(value)["h"] if unpack else value["h"]
    names = os.listdir(stamp_dir)
    stamps = {}
    for name in names:
        if not name.startswith(LOADER_MARK):
            with open(os.path.join(stamp_dir, name)) as f:
                stamps[int(name)] = int(f.read())
    return Run(value, stamps, sum(name.startswith(LOADER_MARK) for name in names))


def slices(n_rows: int, n: int = N_PARTITIONS) -> list[slice]:
    """The row ranges `Partition.resolve` gives partition `i` of `n`."""
    out = []
    for i in range(n):
        part = Partition.blind("toy://m64", "", i, n).resolve(n_rows)
        out.append(slice(part.entry_start, part.entry_stop))
    return out


def reference(empty: Callable[[], bh.Histogram], chunks: Sequence[Sequence[Fill]]) -> bh.Histogram:
    """D5's fold: per partition in key order, each fill node's chunk filled into a fresh empty
    histogram and added in recording order; the partitions then added left to right."""
    acc = empty()
    for nodes in chunks:
        part = empty()
        for args, kwargs in nodes:
            fresh = empty()
            fresh.fill(*args, **kwargs)
            part = gh.add_histograms(part, fresh)
        acc = gh.add_histograms(acc, part)
    return acc


def eager(empty: Callable[[], bh.Histogram], *args: Any, **kwargs: Any) -> bh.Histogram:
    h = empty()
    h.fill(*args, **kwargs)
    return h


def axis_state(axis: Any) -> Any:
    """What makes two axes the same bins: the category list, or the edges."""
    if isinstance(axis, bh.axis.IntCategory | bh.axis.StrCategory):
        return list(axis)
    return np.asarray(axis.edges).tobytes()


def assert_bitwise(got: bh.Histogram, want: bh.Histogram) -> None:
    """Same axes (categories and their order, or edges bit for bit) and the same view bytes."""
    assert len(got.axes) == len(want.axes)
    for g, w in zip(got.axes, want.axes, strict=True):
        assert axis_state(g) == axis_state(w), (g, w)
        assert g.traits.growth == w.traits.growth
    gv, wv = np.asarray(got.view(flow=True)), np.asarray(want.view(flow=True))
    assert gv.dtype == wv.dtype and gv.shape == wv.shape
    assert gv.tobytes() == wv.tobytes()
