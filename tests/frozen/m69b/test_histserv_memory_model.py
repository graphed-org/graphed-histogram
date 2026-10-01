"""m69b histserv backend: a server's measured peak memory against its plan-time prediction (Linux VmHWM).

Each scenario starts its own server, warms it as the model's probe did (one histogram created, filled, snapshotted
and deleted), resets the high-water mark, runs, then reads ``VmHWM`` minus the warm RSS.
"""

from __future__ import annotations

import concurrent.futures
import importlib
import sys
import threading
from collections.abc import Callable
from typing import Any

import boost_histogram as bh
import graphed
import pytest
from graphed.core.execution import LocalResources, Plan, SequentialRunner
from graphed.services import bind_services, resolve_services
from histserv_harness import (
    BASE,
    PER_CONN,
    PER_HIST,
    RUN_TIMEOUT_S,
    MiB,
    Server,
    events,
    histserv_api,
    peak,
    reset_peak,
    run_bounded,
    servers,
    write_events,
)

import graphed_histogram as gh

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/<pid>/status")
pytest.importorskip("histserv")


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> str:
    return write_events(str(tmp_path_factory.mktemp("m69b-memory") / "e.parquet"))


def _bound(plan: Plan[Any], server: Server) -> Plan[Any]:
    return bind_services(plan, {spec.name: server.endpoint for spec in plan.services})


def _warm(path: str, server: Server, name: str) -> int:
    _session, ev = events(path, 1)
    hist = gh.boost.Histogram(bh.axis.Regular(6, 0.0, 1.0), storage=bh.storage.Double())
    hist.fill(ev.x)
    hs = histserv_api()
    hs.backed(hist, hs.Context(memory_mb=1024, workers=1, name=f"{name}-warm"))
    bound = _bound(gh.plan({"warm": hist}, steps_per_file=1), server)
    run_bounded(lambda: resolve_services(bound, SequentialRunner().run(bound).value))
    return reset_peak(server.proc.pid)


def _measure(
    path: str,
    name: str,
    hists: dict[str, gh.boost.Histogram],
    workers: int,
    steps: int,
    memory_mb: int,
    run: Callable[[Plan[Any], Server], object],
) -> tuple[int, int, int]:
    """``(warm RSS, VmHWM - warm RSS, predicted bytes)`` of the one server ``hists`` land on."""
    hs = histserv_api()
    ctx = hs.Context(memory_mb=memory_mb, workers=workers, name=name)
    for hist in hists.values():
        hs.backed(hist, ctx)
    plan = gh.plan(hists, steps_per_file=steps)
    ((_name, _size, predicted_bytes, count),) = ctx.servers()
    assert count == len(hists) and len(plan.tasks) == steps
    with servers(1) as (server,):
        warm = _warm(path, server, name)
        run(_bound(plan, server), server)
        return warm, peak(server.proc.pid) - warm, predicted_bytes


def _sequential(bound: Plan[Any], _server: Server) -> None:
    run_bounded(lambda: resolve_services(bound, SequentialRunner().run(bound).value))


def _double(path: str, bins: int) -> gh.boost.Histogram:
    _session, ev = events(path)
    hist = gh.boost.Histogram(bh.axis.Regular(bins, 0.0, 1.0), storage=bh.storage.Double())
    hist.fill(ev.x)
    return hist


def test_a_64_mib_double_slot_filled_and_resolved_stays_under_its_prediction(path: str) -> None:
    dense = 64 * MiB
    hists = {"h": _double(path, dense // 8 - 2)}
    warm, grew, predicted_bytes = _measure(path, "m69b-mem-64", hists, 1, 2, 1024, _sequential)
    assert warm < BASE
    assert 2 * dense + dense < grew <= predicted_bytes - BASE


def test_four_fillers_released_together_stay_under_the_four_worker_prediction(path: str) -> None:
    dense = 32 * MiB
    hists = {"h": _double(path, dense // 8 - 2)}

    def together(bound: Plan[Any], server: Server) -> None:
        before = server.count("FillMany")
        barrier = threading.Barrier(4)

        def fill(partition: Any) -> Any:
            barrier.wait(timeout=RUN_TIMEOUT_S)
            return bound.process(partition, LocalResources())

        def run() -> None:
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                values = list(pool.map(fill, [task.partition for task in bound.tasks]))
            resolve_services(bound, values[0])

        run_bounded(run)
        assert server.count("FillMany") - before == 4

    warm, grew, predicted_bytes = _measure(path, "m69b-mem-barrier", hists, 4, 4, 1024, together)
    assert warm < BASE
    assert 2 * dense + dense < grew <= predicted_bytes - BASE


def test_an_eight_label_weight_slot_stays_under_its_prediction(path: str) -> None:
    dense = 4 * MiB
    _session, ev = events(path)
    hist = gh.boost.Histogram(bh.axis.Regular(dense // 16 - 2, 0.0, 1.0), storage=bh.storage.Weight())
    w = graphed.vary(ev.w, "sf", points={str(i): ev.w * (1 + (i + 1) / 64) for i in range(7)})
    hist.fill(ev.x, weight=[w], variation_axis=True)
    warm, grew, predicted_bytes = _measure(path, "m69b-mem-labels", {"h": hist}, 1, 2, 1024, _sequential)
    assert warm < BASE
    assert 9 * dense + 8 * dense < grew <= predicted_bytes - BASE


def test_two_thousand_one_bin_slots_stay_under_a_prediction_made_of_per_histogram_terms(path: str) -> None:
    _session, ev = events(path, 1)
    hists = {}
    for i in range(2000):
        hist = gh.boost.Histogram(bh.axis.Regular(1, 0.0, 1.0), storage=bh.storage.Weight())
        hist.fill(ev.x, weight=[ev.w])
        hists[f"h{i:04d}"] = hist

    def filled_once(bound: Plan[Any], server: Server) -> None:
        run_bounded(lambda: SequentialRunner().run(bound))
        assert server.stats()["histogram_count"] == 2000

    warm, grew, predicted_bytes = _measure(path, "m69b-mem-overhead", hists, 1, 1, 160, filled_once)
    assert 0.9 * (predicted_bytes - BASE) <= 2000 * PER_HIST
    assert warm < BASE
    assert grew <= predicted_bytes - BASE


def test_one_connection_per_worker_stays_under_the_prediction(path: str) -> None:
    grpc = importlib.import_module("grpc")
    pb2 = importlib.import_module("histserv.protos.hist_pb2")
    pb2_grpc = importlib.import_module("histserv.protos.hist_pb2_grpc")
    n = 256
    _session, ev = events(path, 1)
    hist = gh.boost.Histogram(bh.axis.Regular(1, 0.0, 1.0), storage=bh.storage.Weight())
    hist.fill(ev.x, weight=[ev.w])

    def dialled(bound: Plan[Any], server: Server) -> None:
        _sequential(bound, server)
        channels = []
        try:
            for _ in range(n):
                channel = grpc.insecure_channel(
                    server.address, options=[("grpc.use_local_subchannel_pool", 1)]
                )
                pb2_grpc.HistogrammerServiceStub(channel).Stats(pb2.StatsRequest(), timeout=60)
                channels.append(channel)
        finally:
            for channel in channels:
                channel.close()

    warm, grew, predicted_bytes = _measure(path, "m69b-mem-connections", {"h": hist}, n, 1, 1024, dialled)
    assert warm < BASE
    assert PER_CONN * n / 2 < grew <= predicted_bytes - BASE
