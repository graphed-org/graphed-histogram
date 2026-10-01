"""Shared harness for the m69b frozen suite: histserv 0.2.1 servers as subprocesses, events on parquet,
and the measured size model.

The implementation under test is reached only through ``histserv_api()`` and ``boost_api()``, inside
test bodies, so the suite collects before ``graphed_histogram.histserv`` exists and each test then
fails at the accessor naming the missing module or symbol. Every server a test starts is killed when
its ``servers()`` block exits.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import awkward as ak
import boost_histogram as bh
import numpy as np
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet
from graphed.core.execution import Plan
from graphed.services import bind_services

RUN_TIMEOUT_S = 240.0
SERVER_START_S = 60.0

MiB = 1 << 20
# The size model's constants, each the maximum over the MODEL lines of the plan's
# probes/m69b/probe_histserv_memory.txt, probe_histserv_memory.amd64.txt and probe_histserv_memory.gha.txt,
# and (PER_CONN) probe_histserv_connections{,.amd64}.txt.
BASE = 164 * MiB
PER_HIST = 4000
PER_TASK = 160
FILL_A = 5.5
FILL_B = 3.5
PER_CONN = 19 * 1024
CEILING = 1 << 29
ITEM = {"Double": 8, "Int64": 8, "Weight": 16}


def histserv_api() -> Any:
    return importlib.import_module("graphed_histogram.histserv")


def boost_api() -> Any:
    return importlib.import_module("graphed_histogram.boost")


def run_bounded(fn: Callable[[], Any], timeout_s: float = RUN_TIMEOUT_S) -> Any:
    """Run ``fn`` on a daemon thread and fail if it does not finish in ``timeout_s``: a hang is a
    failure, never a wedged CI job. Returns the value or re-raises the call's exception."""
    out: dict[str, Any] = {}

    def _drive() -> None:
        try:
            out["result"] = fn()
        except BaseException as exc:
            out["error"] = exc

    thread = threading.Thread(target=_drive, daemon=True)
    thread.start()
    thread.join(timeout_s)
    assert not thread.is_alive(), f"HARD TIMEOUT: call did not finish within {timeout_s}s"
    if "error" in out:
        raise out["error"]
    return out["result"]


# ---- the size model ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """One backed slot's sizes: ``stored = chunks * dense``; ``tasks`` is its plan's task count."""

    chunks: int
    dense: int
    tasks: int

    @property
    def stored(self) -> int:
        return self.chunks * self.dense


def slot_of(axes: Sequence[Any], storage: str, tasks: int, labels: int = 1) -> Slot:
    """The model's sizes for a slot over ``axes`` (the non-variation axes), ``labels`` its chunk count."""
    extent = 1
    for axis in axes:
        extent *= axis.extent
    return Slot(labels, ITEM[storage] * extent, tasks)


def predicted(slots: Sequence[Slot], workers: int) -> float:
    """A server's predicted peak bytes holding ``slots`` for ``workers`` worker processes."""
    m = max(s.stored for s in slots)
    held = sum(PER_HIST + PER_TASK * s.tasks + (s.chunks + 1) * s.dense for s in slots)
    return BASE + held + (FILL_A + FILL_B * workers) * m + PER_CONN * workers


# ---- servers -----------------------------------------------------------------------------------

_PORTS: set[int] = set()


def free_port() -> int:
    # gRPC binds with SO_REUSEPORT, so two servers on one port would both start: never hand a port out twice
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = int(s.getsockname()[1])
        if port not in _PORTS:
            _PORTS.add(port)
            return port


@dataclass
class Server:
    proc: subprocess.Popen[bytes]
    port: int

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.port}"

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.address}"

    def stats(self) -> dict[str, Any]:
        with importlib.import_module("histserv").Client(self.address) as client:
            return dict(client.stats())

    def count(self, rpc: str) -> int:
        return int(self.stats()["rpc_calls_total"].get(rpc, 0))

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=30)


def start_server() -> Server:
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "histserv", "--port", str(port), "--log-level", "ERROR"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    server = Server(proc, port)
    deadline = time.monotonic() + SERVER_START_S
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            socket.create_connection(("127.0.0.1", port), 0.5).close()
            return server
        except OSError:
            time.sleep(0.05)
    server.kill()
    raise AssertionError(f"histserv did not listen on {port} within {SERVER_START_S}s (exit {proc.poll()})")


@contextlib.contextmanager
def servers(n: int) -> Iterator[list[Server]]:
    started: list[Server] = []
    try:
        for _ in range(n):
            started.append(start_server())
        yield started
    finally:
        for server in started:
            server.kill()


def server_names(plan: Plan[Any]) -> list[str]:
    return [spec.name for spec in plan.services if spec.kind == "histserv"]


def bind(plan: Plan[Any], started: Sequence[Server]) -> tuple[Plan[Any], dict[str, Server]]:
    """``plan`` bound by hand, one started server per histserv name in ``plan.services`` order."""
    names = server_names(plan)
    assert len(started) >= len(names), (names, len(started))
    by_name = dict(zip(names, started, strict=False))
    return bind_services(plan, {name: server.endpoint for name, server in by_name.items()}), by_name


def assigned(plan: Plan[Any]) -> dict[Any, str]:
    """The served plan's slot -> server name map (``_Served.slots``)."""
    process: Any = plan.process
    return {slot: str(getattr(server, "name", server)) for slot, server in dict(process.slots).items()}


# ---- events and comparisons --------------------------------------------------------------------

CATEGORIES = ("ee", "mm", "em", "tt")


def write_events(path: str, n: int = 2000, seed: int = 7) -> str:
    """A parquet file of ``n`` events; ``w`` holds dyadic weights, ``wi`` integer ones, so any fold
    order sums them exactly; ``c``, ``k`` and ``b`` fill category and boolean axes past their bins."""
    rng = np.random.default_rng(seed)
    ak.to_parquet(
        ak.Array(
            {
                "x": rng.normal(2.0, 1.5, n),
                "w": rng.choice([0.25, 0.5, 1.0, 1.5, 2.0], n),
                "wi": rng.integers(1, 4, n).astype(np.float64),
                "c": rng.choice(CATEGORIES, n),
                "k": rng.integers(-1, 4, n),
                "b": rng.integers(0, 2, n).astype(bool),
            }
        ),
        path,
    )
    return path


def events(path: str, steps: int = 3) -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, from_parquet(session, "events", path, steps_per_file=steps)


def same(a: Any, b: Any) -> bool:
    """Bit-for-bit equality of two boost histograms: axes, storage, and every field of the flow view."""
    if not (isinstance(a, bh.Histogram) and isinstance(b, bh.Histogram)):
        return False
    if a.axes != b.axes or type(a.storage_type()) is not type(b.storage_type()):
        return False
    va, vb = np.asarray(a.view(flow=True)), np.asarray(b.view(flow=True))
    if va.dtype.names:
        return all(np.array_equal(va[f], vb[f]) for f in va.dtype.names)
    return va.dtype == vb.dtype and np.array_equal(va, vb)


def child_env(**extra: str) -> dict[str, str]:
    """The environment for a child interpreter that imports what this one does."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p), **extra}


# ---- Linux server memory -----------------------------------------------------------------------


def _status(pid: int, field: str) -> int:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)


def reset_peak(pid: int) -> int:
    """The settled RSS, then the high-water mark reset to it (``5`` to ``/proc/<pid>/clear_refs``)."""
    time.sleep(0.6)
    rss = _status(pid, "VmRSS")
    with open(f"/proc/{pid}/clear_refs", "w") as f:
        f.write("5")
    return rss


def peak(pid: int) -> int:
    time.sleep(0.6)
    return _status(pid, "VmHWM")
