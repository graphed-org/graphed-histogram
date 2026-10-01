"""Filling on histserv servers: a histogram backend beside ``gh.boost``, sized before the run.

Four parts, four owners. **Recording** stays ``gh.boost``'s: a backed histogram records exactly the
graph its unbacked twin records. The **context** (:class:`Context`) sizes each backed slot with a
measured model, packs the slots first-fit decreasing onto servers of the sizes it offers, and puts
one :class:`~graphed.services.ServiceSpec` per server in ``plan.services`` at ``pieces.serve``. The
**executor** places, binds and closes those servers and never learns what a histogram is. The
**served process** creates the server-side histograms, ships each partition's partial of a backed
slot, and hands a receipt through the reduction tree instead of the histogram.

Importing this module, building a context, backing a histogram and planning it import neither
``histserv`` nor ``grpc``.
"""

from __future__ import annotations

import json
import math
import os
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import boost_histogram as bh
import numpy as np
from graphed import GraphedError
from graphed.core import Partition
from graphed.core.execution import Plan, WorkerResources
from graphed.services import Bindable, Launch, Resolvable, ServiceSpec, UnboundService, split_endpoint

from . import boost
from ._spec import _make_axis, zero_of
from .boost import SlotKey

H = TypeVar("H", bound=boost.Histogram)
V = TypeVar("V")

MiB = 1 << 20
# The size model's constants, each the maximum over the MODEL lines of the plan's
# probes/m69b/probe_histserv_memory.txt, probe_histserv_memory.amd64.txt and probe_histserv_memory.gha.txt,
# and (_PER_CONN) probe_histserv_connections{,.amd64}.txt.
_BASE = 164 * MiB
_PER_HIST = 4000
_PER_TASK = 160
_FILL_A = 5.5
_FILL_B = 3.5
_PER_CONN = 19 * 1024
#: histserv 0.2.1's client refuses a message past 2**29 bytes; a fill message is the slot's stored
#: bytes plus at most this much
CEILING = 1 << 29
_ENVELOPE = 64 * 1024
_ITEM = {"Double": 8, "Int64": 8, "Weight": 16}
#: the prune age argv passes, ten years: the default drops a histogram idle for a day
_ARGV = (
    "{python}",
    "-m",
    "histserv",
    "--port",
    "{port}",
    "--prune-after-seconds",
    "315360000",
    "--log-level",
    "WARNING",
)

#: one lock over the name registry and every serve's placement
_LOCK = threading.Lock()


@dataclass(frozen=True)
class _Size:
    """A backed slot in the size model: ``stored = chunks * dense`` bytes on the server."""

    chunks: int
    dense: int
    tasks: int

    @property
    def stored(self) -> int:
        return self.chunks * self.dense


def _predict(sizes: Sequence[_Size], workers: int) -> float:
    """A server's predicted peak bytes while it holds ``sizes`` and ``workers`` worker processes
    dial it."""
    m = max(s.stored for s in sizes)
    held = sum(_PER_HIST + _PER_TASK * s.tasks + (s.chunks + 1) * s.dense for s in sizes)
    return _BASE + held + (_FILL_A + _FILL_B * workers) * m + _PER_CONN * workers


def _labels(key: SlotKey, spec: str) -> tuple[str, ...] | None:
    """An axis-mode slot's variation labels, the chunks it is filled in; ``None`` for any other."""
    if not (isinstance(key, tuple) and key[1] is None):
        return None
    return tuple(json.loads(spec)["axes"][-1]["categories"])


def _size(key: SlotKey, spec: str, tasks: int) -> _Size:
    payload = json.loads(spec)
    labels = _labels(key, spec)
    dense_axes = payload["axes"] if labels is None else payload["axes"][:-1]
    extent = math.prod(_make_axis(axis).extent for axis in dense_axes)
    return _Size(1 if labels is None else len(labels), _ITEM[payload["storage"]] * extent, tasks)


@dataclass
class _Server:
    size: int  # MiB
    #: opened for one slot no offered size holds; nothing else lands on it
    alone: bool
    sizes: list[_Size]


@dataclass
class _Pack:
    """One context name's arguments and servers, held for the life of the process."""

    memory_mb: tuple[int, ...]
    workers: int
    ports: tuple[int, int]
    timeout_s: float
    servers: list[_Server]

    def args(self) -> tuple[Any, ...]:
        return (self.memory_mb, self.workers, self.ports, self.timeout_s)

    def place(self, size: _Size) -> int:
        """First fit over the open servers in index order, else a new server at the smallest
        offered size that holds the slot alone, else one sized to it."""
        for i, server in enumerate(self.servers):
            if not server.alone and _predict([*server.sizes, size], self.workers) <= server.size * MiB:
                server.sizes.append(size)
                return i
        alone = _predict([size], self.workers)
        fits = [s for s in self.memory_mb if alone <= s * MiB]
        server = _Server(fits[0], False, [size]) if fits else _Server(math.ceil(alone / MiB), True, [size])
        self.servers.append(server)
        return len(self.servers) - 1


_PACKS: dict[str, _Pack] = {}


class Context:
    """The servers a set of histograms fills on, sized before the run.

    ``memory_mb`` is the server sizes offered, in MiB (one or several); ``workers`` bounds the fills
    in flight to one server and the worker processes that dial it. A ``name`` holds one packing
    state for the life of the process: a second context under that name with equal arguments
    shares it (and warns), one with other arguments is refused. Server ``i`` is the service
    ``f"{name}-{i}"`` listening on a port in ``ports``; ``timeout_s`` is its start-up bound."""

    def __init__(
        self,
        *,
        memory_mb: int | Sequence[int],
        workers: int,
        name: str = "histserv",
        ports: tuple[int, int] = (10000, 10100),
        timeout_s: float = 600.0,
    ) -> None:
        offered = tuple(sorted({int(s) for s in ([memory_mb] if isinstance(memory_mb, int) else memory_mb)}))
        if not offered:
            raise ValueError(f"histserv Context {name!r}: memory_mb offers no server size")
        if offered[0] * MiB < _BASE:
            raise ValueError(
                f"histserv Context {name!r}: a {offered[0]} MiB server is below the "
                f"{_BASE // MiB} MiB a histserv server holds before any histogram"
            )
        if workers < 1:
            raise ValueError(f"histserv Context {name!r}: workers is at least 1, not {workers}")
        pack = _Pack(offered, int(workers), (int(ports[0]), int(ports[1])), float(timeout_s), [])
        with _LOCK:
            held = _PACKS.setdefault(name, pack)
        if held is not pack:
            if held.args() != pack.args():
                raise ValueError(
                    f"histserv Context name {name!r} is held in this process by memory_mb="
                    f"{list(held.memory_mb)}, workers={held.workers}, ports={held.ports}, "
                    f"timeout_s={held.timeout_s}; this one asks for memory_mb={list(offered)}, "
                    f"workers={pack.workers}, ports={pack.ports}, timeout_s={pack.timeout_s}. Pass a "
                    "fresh name= or use a new process"
                )
            warnings.warn(
                f"histserv Context {name!r} already exists in this process with these arguments; "
                "this one shares its servers and packing",
                stacklevel=2,
            )
        self.name = name
        self.memory_mb = offered
        self.workers = pack.workers
        self.ports = pack.ports
        self.timeout_s = pack.timeout_s
        self._pack = held

    def servers(self) -> tuple[tuple[str, int, int, int], ...]:
        """``(name, memory_mb, predicted_bytes, n_histograms)`` per server opened so far."""
        with _LOCK:
            return tuple(
                (f"{self.name}-{i}", s.size, math.ceil(_predict(s.sizes, self.workers)), len(s.sizes))
                for i, s in enumerate(self._pack.servers)
            )

    def _spec(self, i: int) -> ServiceSpec:
        return ServiceSpec(
            f"{self.name}-{i}",
            kind="histserv",
            check="tcp",
            ports=self.ports,
            launch=Launch(_ARGV, resources={"memory_mb": self._pack.servers[i].size}),
            timeout_s=self.timeout_s,
        )


_STORAGES = ("Double", "Int64", "Weight")


def backed(h: H, context: Context) -> H:
    """``h``, filled on ``context``'s servers from its next plan on; its graph is unchanged.

    A growth axis is refused (a server is sized before the run), as is a storage other than
    ``Double``, ``Int64`` or ``Weight``."""
    for axis in h.axes:
        if axis.traits.growth:
            raise TypeError(
                f"histserv sizes its servers before the run, so it cannot hold a growth axis: "
                f"{type(axis).__name__}(growth=True)"
            )
    storage = type(h.storage_type()).__name__
    if storage not in _STORAGES:
        raise TypeError(f"histserv holds {', '.join(_STORAGES)} storage, not {storage}")
    h._histserv = context
    return h


class Histogram(boost.Histogram):
    """A ``gh.boost.Histogram`` filled on ``context``'s histserv servers (:func:`backed`)."""

    def __init__(self, *axes: Any, storage: Any = None, metadata: Any = None, context: Context) -> None:
        super().__init__(*axes, storage=storage, metadata=metadata)
        backed(self, context)


class HistservError(GraphedError):
    """A histserv RPC failed: the server's ``endpoint``, the gRPC status ``code`` name and its
    ``details``. Unlike grpc's own error, it pickles, so it crosses a process boundary intact."""

    def __init__(self, endpoint: str, code: str, details: str) -> None:
        super().__init__(endpoint, code, details)
        self.endpoint = endpoint
        self.code = code
        self.details = details

    def __str__(self) -> str:
        return f"histserv at {self.endpoint} answered {self.code}: {self.details}"


#: per process, so a forked worker never reuses its parent's channel
_CLIENTS: dict[tuple[int, str], Any] = {}
_CLIENTS_LOCK = threading.Lock()


def _call(endpoint: str, rpc: Callable[[Any], V]) -> V:
    """``rpc(client)`` against ``endpoint``'s cached client, a gRPC error raised as a
    :class:`HistservError`."""
    import grpc  # noqa: PLC0415  (only a bound run talks to a server)

    with _CLIENTS_LOCK:
        key = (os.getpid(), endpoint)
        client = _CLIENTS.get(key)
        if client is None:
            from histserv import Client  # noqa: PLC0415

            client = _CLIENTS[key] = Client(split_endpoint(endpoint)[1])
    try:
        return rpc(client)
    except grpc.RpcError as err:
        raise HistservError(endpoint, err.code().name, str(err.details())) from err


@dataclass(frozen=True)
class Receipt:
    """A backed slot's value between its reduce and ``resolve_services``: the server histogram
    holding it. Receipts add by identity; one without ``hist_id`` is the empty value."""

    spec: str
    endpoint: str | None = None
    hist_id: str | None = None

    def __add__(self, other: Receipt) -> Receipt:
        if other.hist_id is None or other == self:
            return self
        if self.hist_id is None:
            return other
        raise GraphedError(
            f"receipts of two histserv histograms do not add: {self.hist_id} at {self.endpoint} and "
            f"{other.hist_id} at {other.endpoint}"
        )

    def snapshot(self, *, delete: bool = False) -> Any:
        """The server histogram's contents as the slot's boost histogram (empty for the empty
        receipt); ``delete`` also drops it from the server."""
        out = zero_of(self.spec)
        if self.endpoint is None or self.hist_id is None:
            return out
        from histserv.protos import hist_pb2  # noqa: PLC0415
        from histserv.serialize import deserialize_chunked_hist_payload  # noqa: PLC0415

        request = hist_pb2.SnapshotRequest(hist_id=self.hist_id, delete_from_server=delete)
        reply = _call(self.endpoint, lambda c: c.stub.Snapshot(request, timeout=_RPC_TIMEOUT_S))
        view = out.view(flow=True)
        for key, chunk in deserialize_chunked_hist_payload(reply.payload).items():
            # a keyed chunk is one label of the variation axis, the spec's last
            view[(..., out.axes[-1].index(key[0])) if key else ...] = chunk
        return out


_RPC_TIMEOUT_S = 600


def _template(key: SlotKey, spec: str) -> Any:
    """The server-side histogram of a slot: a no-flow ``Regular`` per non-variation axis over its
    flow extent, so category and boolean flow bins survive, and the variation axis as the chunk
    axis."""
    import hist  # noqa: PLC0415
    from histserv.chunked_hist import ChunkedHist  # noqa: PLC0415

    payload = json.loads(spec)
    labels = _labels(key, spec)
    dense = payload["axes"] if labels is None else payload["axes"][:-1]
    axes = [
        hist.axis.Regular(_make_axis(a).extent, 0.0, 1.0, underflow=False, overflow=False, name=f"a{i}")
        for i, a in enumerate(dense)
    ]
    if labels is not None:
        axes.append(hist.axis.StrCategory(list(labels), name="variation"))
    return ChunkedHist(*axes, storage=getattr(bh.storage, payload["storage"])())


#: the running task's ``(unique id, {slot: (endpoint, hist_id)})``, set by `_Served.__call__`
_SCOPE: ContextVar[tuple[str, dict[SlotKey, tuple[str, str]]] | None] = ContextVar(
    "graphed_histogram_histserv_scope", default=None
)


def _ship(key: SlotKey, spec: str, partial: Any) -> Receipt:
    """One partition's partial of a backed slot, sent as one ``FillMany`` keyed by the partition
    (a chunk per variation label); a replay of that partition is ``ALREADY_EXISTS``, already done."""
    scope = _SCOPE.get()
    if scope is None or key not in scope[1]:
        raise GraphedError(
            f"histserv-backed slot {key!r} fills on a server, so it runs only in a plan served by "
            "pieces.serve(plan) and bound to its servers"
        )
    from histserv.protos import hist_pb2  # noqa: PLC0415
    from histserv.serialize import serialize_chunk_payload, serialize_unique_id  # noqa: PLC0415

    unique_id, homes = scope
    endpoint, hist_id = homes[key]
    view = np.asarray(partial.view(flow=True))
    labels = _labels(key, spec)
    if labels is None:
        chunks = [serialize_chunk_payload((), view, shape=view.shape, dtype=view.dtype)]
    else:
        chunks = [
            serialize_chunk_payload((label,), view[..., i], shape=view.shape[:-1], dtype=view.dtype)
            for i, label in enumerate(labels)
        ]
    request = hist_pb2.FillManyRequest(hist_id=hist_id, chunks=chunks)
    request.unique_id = serialize_unique_id(unique_id)
    try:
        _call(endpoint, lambda c: c.stub.FillMany(request, timeout=_RPC_TIMEOUT_S))
    except HistservError as err:
        if err.code != "ALREADY_EXISTS":
            raise
    return Receipt(spec, endpoint, hist_id)


@dataclass(frozen=True)
class _Home:
    """Where a backed slot fills: its server's service name, and the slot's histogram spec."""

    name: str
    spec: str


def _create(endpoint: str, key: SlotKey, spec: str) -> str:
    template = _template(key, spec)
    return str(_call(endpoint, lambda c: c.init(template, timeout=_RPC_TIMEOUT_S)).hist_id)


class _Handles:
    """A bound plan's server-side histogram ids, created once, under a lock, on first use."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: dict[SlotKey, str] | None = None

    def ids(self, endpoints: Mapping[str, str], slots: Sequence[tuple[SlotKey, _Home]]) -> dict[SlotKey, str]:
        with self._lock:
            if self._ids is None:
                self._ids = {key: _create(endpoints[home.name], key, home.spec) for key, home in slots}
            return self._ids

    def __getstate__(self) -> dict[SlotKey, str] | None:
        return self._ids

    def __setstate__(self, ids: dict[SlotKey, str] | None) -> None:
        self._lock = threading.Lock()
        self._ids = ids


@dataclass(frozen=True)
class _Served:
    """A served plan's process: the plan's own, around which each task's backed slots know their
    server histograms. Those are created on the first call or the first pickle, which every runner
    reaches in the driver, so a worker receives their ids and creates none."""

    inner: Callable[[Partition, WorkerResources], Any]
    slots: tuple[tuple[SlotKey, _Home], ...]
    endpoints: dict[str, str] | None = None
    handles: _Handles | None = None

    def _names(self) -> list[str]:
        return sorted({home.name for _key, home in self.slots})

    def bind_services(self, endpoints: Mapping[str, str]) -> _Served:
        """A copy bound to ``endpoints`` over those it holds, with fresh handles; no RPC."""
        merged = {**(self.endpoints or {}), **endpoints}
        missing = [name for name in self._names() if name not in merged]
        if missing:
            raise UnboundService(*missing)
        for name in self._names():
            if split_endpoint(merged[name])[0] not in ("tcp", "grpc"):
                raise ValueError(
                    f"histserv server {name!r} speaks plaintext gRPC only, so its endpoint is "
                    f"tcp:// or grpc://, not {merged[name]!r}"
                )
        inner = self.inner.bind_services(endpoints) if isinstance(self.inner, Bindable) else self.inner
        bound = {name: merged[name] for name in self._names()}
        return replace(self, inner=inner, endpoints=bound, handles=_Handles())

    def __call__(self, partition: Partition, resources: WorkerResources) -> Any:
        if self.endpoints is None or self.handles is None:
            raise UnboundService(*self._names())
        ids = self.handles.ids(self.endpoints, self.slots)
        homes = {key: (self.endpoints[home.name], ids[key]) for key, home in self.slots}
        token = _SCOPE.set((str(partition), homes))
        try:
            return self.inner(partition, resources)
        finally:
            _SCOPE.reset(token)

    def __reduce__(self) -> tuple[Any, ...]:
        if self.endpoints is not None and self.handles is not None:
            self.handles.ids(self.endpoints, self.slots)
        return (_Served, (self.inner, self.slots, self.endpoints, self.handles))

    def resolve_services(self, value: Any) -> Any:
        return self.inner.resolve_services(value) if isinstance(self.inner, Resolvable) else value

    def part_paths(self, partition: Partition) -> Sequence[str]:
        hook = getattr(self.inner, "part_paths", None)
        return () if hook is None else hook(partition)


def _serve(plan: Plan[V], session: Any, backing: Sequence[tuple[SlotKey, str, Context]]) -> Plan[V]:
    """Size, pack and declare ``backing``'s slots for ``plan``: the work of ``pieces.serve``."""
    if plan.next_tasks is not None:
        raise GraphedError(
            "histserv sizes its servers by the plan's tasks before the run, so a plan with "
            "next_tasks cannot be served"
        )
    seen: set[str] = set()
    for task in plan.tasks:
        if str(task.partition) in seen:
            raise GraphedError(
                f"histserv keys each fill by its partition, and partition {task.partition} is in two "
                "of this plan's tasks: give each task its own partition"
            )
        seen.add(str(task.partition))
    sized = [(key, spec, ctx, _size(key, spec, len(plan.tasks))) for key, spec, ctx in backing]
    for key, _spec, _ctx, size in sized:
        if size.stored + _ENVELOPE > CEILING:
            raise ValueError(
                f"histserv slot {key!r} needs a {size.stored + _ENVELOPE}-byte fill message "
                f"({size.stored} bytes stored + 64 KiB), past histserv's {CEILING}-byte message "
                "ceiling: split it into smaller histograms"
            )
    homes: dict[SlotKey, tuple[Context, int]] = {}
    with _LOCK:
        for key, _spec, ctx, size in sorted(sized, key=lambda s: (-s[3].stored, str(s[0]))):
            homes[key] = (ctx, ctx._pack.place(size))
        landed = {f"{ctx.name}-{i}": ctx._spec(i) for ctx, i in homes.values()}
    for spec in landed.values():
        session.declare_service(spec)
    services = {s.name: s for s in plan.services} | landed
    slots = tuple((key, _Home(f"{homes[key][0].name}-{homes[key][1]}", spec)) for key, spec, _ctx in backing)
    return replace(
        plan, process=_Served(plan.process, slots), services=tuple(services[n] for n in sorted(services))
    )


__all__ = ["CEILING", "Context", "Histogram", "HistservError", "Receipt", "backed"]
