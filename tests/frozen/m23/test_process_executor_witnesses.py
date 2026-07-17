"""M23 process-executor witnesses (freeze-M23-2, user-directed additions).

Weighted fills are HEP's most common case and behavior loss plagued distributed ragged analyses
— both MUST be controlled early, across a REAL process boundary:

- a Weight-storage weighted fill through the spawned process pool equals the sequential and
  eager results exactly (values AND variances);
- a ragged awkward fill through the process pool equals its eager twin;
- a session recorded with a behavior-carrying backend FAILS LOUDLY under the default plan()
  (workers construct a bare backend — behaviors are never silently dropped), and
  `plan(backend="module:attr")` is the supported path: the worker IMPORTS the factory (no
  behavior dict is ever pickled) and reproduces the sequential result bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import boost_histogram as bh
import numpy as np
import pytest
from graphed import Session
from graphed.core import Partition
from graphed.core.execution import SequentialRunner

import graphed_histogram as gh

pytest.importorskip("graphed_exec_local")
from graphed_exec_local import ProcessExecutor

RNG = np.random.default_rng(11)
DATA = RNG.normal(5.0, 2.0, 600)
WEIGHTS = RNG.uniform(0.1, 2.0, 600)


@dataclass
class ChunkedSource:
    data: object
    whole_calls: list = field(default_factory=list)

    def __call__(self) -> object:
        self.whole_calls.append(1)
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://chunks", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition, columns, resources):  # type: ignore[no-untyped-def]
        part = partition.resolve(len(self.data))  # type: ignore[arg-type]
        return self.data[part.entry_start : part.entry_stop]  # type: ignore[index]


def test_weighted_fill_survives_the_process_boundary_exactly() -> None:
    pytest.importorskip("graphed.numpy")
    from graphed.numpy import NumpyBackend  # noqa: PLC0415  (importorskip-gated)
    from graphed.numpy.forms import NumpyForm  # noqa: PLC0415

    s = Session(NumpyBackend())
    src = ChunkedSource(np.stack([DATA, WEIGHTS], axis=1))
    ev = s.source("ev", form=NumpyForm(DATA.dtype, shape=(None, 2)), data=src)
    x, w = ev[:, 0], ev[:, 1]
    h = gh.boost.Histogram(bh.axis.Regular(24, 0.0, 10.0), storage=bh.storage.Weight())
    h.fill(x, weight=w)

    sequential = SequentialRunner().run(h.plan(steps_per_file=3)).value
    parallel = ProcessExecutor(max_workers=2).run(h.plan(steps_per_file=3)).value
    eager = bh.Histogram(bh.axis.Regular(24, 0.0, 10.0), storage=bh.storage.Weight())
    eager.fill(DATA, weight=WEIGHTS)

    # exact, not approximate: identical per-partition fills + the fixed combine tree
    assert np.array_equal(np.asarray(parallel.values(flow=True)), np.asarray(sequential.values(flow=True)))
    assert np.array_equal(
        np.asarray(parallel.variances(flow=True)), np.asarray(sequential.variances(flow=True))
    )
    assert np.allclose(np.asarray(parallel.values(flow=True)), eager.values(flow=True))
    assert np.allclose(np.asarray(parallel.variances(flow=True)), eager.variances(flow=True))


def test_ragged_awkward_fill_survives_the_process_boundary() -> None:
    ak = pytest.importorskip("awkward")
    pytest.importorskip("graphed.awkward")
    from graphed.awkward import AwkwardBackend, AwkwardForm  # noqa: PLC0415  (importorskip-gated)

    events = ak.Array({"Jet_pt": [[50.0, 30.0], [], [70.0, 20.0, 10.0], [5.0]] * 40})
    s = Session(AwkwardBackend())
    src = ChunkedSource(events)
    tt = ak.Array(events.layout.to_typetracer(forget_length=True))
    g = s.source("events", form=AwkwardForm(tt), data=src)

    h = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 100.0), storage=bh.storage.Int64())
    h.fill(g.Jet_pt * 1.0)

    parallel = ProcessExecutor(max_workers=2).run(h.plan(steps_per_file=4)).value
    eager = bh.Histogram(bh.axis.Regular(20, 0.0, 100.0), storage=bh.storage.Int64())
    eager.fill(ak.flatten(events.Jet_pt * 1.0, axis=None))
    assert np.array_equal(np.asarray(parallel.values(flow=True)), eager.values(flow=True))
    assert src.whole_calls == []


def _behavior_session():  # type: ignore[no-untyped-def]
    ak = pytest.importorskip("awkward")
    pytest.importorskip("graphed.awkward")
    from behavior_toy import BEHAVIOR, make_backend  # noqa: PLC0415
    from graphed.awkward import AwkwardForm, gak  # noqa: PLC0415

    events = ak.Array({"x": [3.0, 0.0, 6.0, 8.0] * 50, "y": [4.0, 1.0, 8.0, 15.0] * 50})
    s = Session(make_backend())
    src = ChunkedSource(events)
    tt = ak.Array(events.layout.to_typetracer(forget_length=True))
    g = s.source("events", form=AwkwardForm(tt), data=src)
    rec = gak.with_name(gak.zip({"x": g.x, "y": g.y}), "planar")
    mag = rec.mag  # a behavior PROPERTY: evaluation NEEDS the behavior registered
    h = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 20.0), storage=bh.storage.Int64())
    h.fill(mag)
    reference = np.hypot(np.asarray(events.x), np.asarray(events.y))
    del BEHAVIOR
    return h, reference


def test_default_plan_fails_loudly_when_behaviors_would_be_lost() -> None:
    h, _ref = _behavior_session()
    # the DEFAULT worker backend is constructed bare: the behavior property cannot evaluate.
    # Losing behaviors must be LOUD — never a silently wrong histogram (the dask-awkward lesson).
    with pytest.raises(Exception, match=r"mag|field|behavior"):
        ProcessExecutor(max_workers=2).run(h.plan(steps_per_file=2))


def test_behavior_backends_forward_by_import_ref() -> None:
    h, reference = _behavior_session()
    # the supported path: workers IMPORT the factory — no behavior dict is ever pickled
    plan = h.plan(steps_per_file=2, backend="behavior_toy:make_backend")
    parallel = ProcessExecutor(max_workers=2).run(plan).value
    sequential = SequentialRunner().run(h.plan(steps_per_file=2, backend="behavior_toy:make_backend")).value
    eager = bh.Histogram(bh.axis.Regular(20, 0.0, 20.0), storage=bh.storage.Int64())
    eager.fill(reference)
    assert np.array_equal(np.asarray(parallel.values(flow=True)), eager.values(flow=True))
    assert np.array_equal(np.asarray(parallel.values(flow=True)), np.asarray(sequential.values(flow=True)))
