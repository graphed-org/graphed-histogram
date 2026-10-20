"""Counts densifies of chunked partials, in this process and in every worker that unpickles `Events`."""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

import awkward as ak
from graphed.core import Partition
from graphed.core.execution import WorkerResources

from graphed_histogram._chunked import Chunked

#: inherited by spawned workers, so their densifies land beside the driver's
DIR = os.environ.setdefault("GH_CHUNKED_CENSUS", tempfile.mkdtemp(prefix="chunked-census-"))
_dense = Chunked._dense


def _counted(self: Chunked) -> Any:
    open(os.path.join(DIR, f"{os.getpid()}-{uuid.uuid4().hex}"), "w").close()
    return _dense(self)


Chunked._dense = _counted  # type: ignore[method-assign]


def take() -> list[int]:
    """The pid of each densify since the last call."""
    names = os.listdir(DIR)
    for name in names:
        os.remove(os.path.join(DIR, name))
    return [int(name.split("-")[0]) for name in names]


@dataclass
class Events:
    data: ak.Array

    def __call__(self) -> ak.Array:
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("census://ev", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: WorkerResources) -> ak.Array:
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]
