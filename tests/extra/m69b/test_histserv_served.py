"""m69b served process beyond the frozen rows: what a worker's unpickled copy does, an empty run,
an unbound call, the plaintext ``grpc://`` wire, and one creation under concurrent first calls."""

from __future__ import annotations

import functools
import pickle
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Any

import boost_histogram as bh
import pytest
from graphed.core.execution import LocalResources, Plan, SequentialRunner
from graphed.services import UnboundService, bind_services, resolve_services

sys.path.append(str(Path(__file__).resolve().parents[2] / "frozen" / "m69b"))

from histserv_harness import bind, events, run_bounded, same, servers, write_events

import graphed_histogram as gh
from graphed_histogram import histserv

pytestmark = pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")), reason="grpcio ships no free-threaded wheel"
)


def _plans(path: str, name: str, **kwargs: Any) -> tuple[Plan[Any], Plan[Any]]:
    """A backed plan over two contexts (two servers) and its unbacked twin."""

    def hists(back: bool) -> dict[str, gh.boost.Histogram]:
        _session, ev = events(path)
        out = {
            "h": gh.boost.Histogram(bh.axis.Regular(10, -2.0, 6.0), storage=bh.storage.Weight()),
            "g": gh.boost.Histogram(bh.axis.Regular(7, -2.0, 6.0), storage=bh.storage.Int64()),
        }
        out["h"].fill(ev.x, weight=[ev.w])
        out["g"].fill(ev.x)
        if back:
            for key, hist in out.items():
                histserv.backed(hist, histserv.Context(memory_mb=1024, workers=1, name=f"{name}-{key}"))
        return out

    return gh.plan(hists(True), steps_per_file=3, **kwargs), gh.plan(hists(False), steps_per_file=3, **kwargs)


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> str:
    return write_events(str(tmp_path_factory.mktemp("m69b-extra-served") / "e.parquet"))


def test_an_unpickled_copy_fills_the_histograms_the_driver_created(path: str) -> None:
    """A worker's copy carries the created ids: it creates none and fills the same histograms.
    Creating in ``__setstate__`` (or per copy) doubles the creations and splits the sums."""
    plan, twin = _plans(path, "m69b-extra-unpickled")
    with servers(len(plan.services)) as started:
        bound, by_name = bind(plan, started)
        worker = pickle.loads(pickle.dumps(bound.process))
        created = {name: server.count("Init") for name, server in by_name.items()}
        assert created == dict.fromkeys(by_name, 1)
        res = LocalResources()
        tasks = sorted(plan.tasks, key=lambda t: t.key)
        total = bound.empty()
        for task in tasks:
            total = bound.combine(total, run_bounded(functools.partial(worker, task.partition, res)))
        assert {name: server.count("Init") for name, server in by_name.items()} == created
        resolved = run_bounded(lambda: resolve_services(bound, total))
    want = SequentialRunner().run(twin).value
    assert set(resolved) == set(want) and all(same(resolved[k], want[k]) for k in want)


def test_a_served_plan_over_no_partition_resolves_to_empty_histograms(path: str) -> None:
    """The empty value of a backed slot is the empty receipt, which resolves to the zero histogram
    without a server; an empty value of zero histograms would not add to the first receipt."""
    plan, twin = _plans(path, "m69b-extra-empty", partitions=[])
    # nothing listens on port 1: any RPC fails the test
    bound = bind_services(plan, {s.name: "tcp://127.0.0.1:1" for s in plan.services})
    value = SequentialRunner().run(bound).value
    assert all(isinstance(v, histserv.Receipt) and v.hist_id is None for v in value.values())
    resolved = resolve_services(bound, value)
    want = SequentialRunner().run(twin).value
    assert set(resolved) == set(want) and all(same(resolved[k], want[k]) for k in want)


def test_an_unbound_served_process_refuses_its_task_naming_every_server(path: str) -> None:
    plan, _twin = _plans(path, "m69b-extra-unbound")
    with pytest.raises(UnboundService) as raised:
        plan.process(plan.tasks[0].partition, LocalResources())
    assert sorted(raised.value.names) == sorted(s.name for s in plan.services)


def test_a_grpc_endpoint_is_the_plaintext_wire(path: str) -> None:
    plan, twin = _plans(path, "m69b-extra-grpc")
    with servers(len(plan.services)) as started:
        bound = bind_services(
            plan,
            {
                spec.name: f"grpc://{server.address}"
                for spec, server in zip(plan.services, started, strict=False)
            },
        )
        value = run_bounded(lambda: SequentialRunner().run(bound).value)
        assert all(v.endpoint.startswith("grpc://") for v in value.values())
        resolved = run_bounded(lambda: resolve_services(bound, value))
    want = SequentialRunner().run(twin).value
    assert all(same(resolved[k], want[k]) for k in want)


def test_concurrent_first_calls_create_each_histogram_once(path: str) -> None:
    """Eight threads released together all reach the first use; without the handles' lock several
    of them create, and the server holds more histograms than slots."""
    plan, _twin = _plans(path, "m69b-extra-race")
    with servers(len(plan.services)) as started:
        bound, by_name = bind(plan, started)
        barrier = threading.Barrier(8)
        errors: list[BaseException] = []

        def call(i: int) -> None:
            try:
                barrier.wait(timeout=60)
                bound.process(plan.tasks[i % len(plan.tasks)].partition, LocalResources())
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call, args=(i,), daemon=True) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        assert not errors and not any(t.is_alive() for t in threads)
        assert {name: server.count("Init") for name, server in by_name.items()} == dict.fromkeys(by_name, 1)
