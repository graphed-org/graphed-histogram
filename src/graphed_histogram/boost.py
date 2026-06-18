"""The deferred ``boost_histogram.Histogram`` — fills RECORD; executors aggregate.

Each ``.fill(...)`` records one External node (the M3 correctionlib/ONNX family) whose evaluator
returns a FILLED boost histogram for its chunk; the node's identity is the content hash of the
canonical axes/storage spec plus its inputs, so identical fills intern. Evaluation is graphed's
own machinery — there is no ``compute()`` here: ``plan()`` exports the R15.4 task graph (one
fill task per partition over a ``graphed.write.PartitionedSource``; the whole-dataset loader is
never invoked) whose tree-combine is native ``+``, and ANY R7 executor's ``run(plan).value`` IS
the aggregated histogram; the reference ``session.materialize(fill_node)`` evaluates a fill
eagerly. Int64 counts are exact under any combine tree; float storages are deterministic per
fixed-tree executor configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import boost_histogram as bh
import numpy as np
from graphed import Array, compile_ir, evaluate_ir, read_columns
from graphed.write import PartitionedSource
from graphed_core import Partition, PayloadDescriptor
from graphed_core.execution import Plan, Task, WorkerResources

from ._spec import content_hash, spec_of, zero_of


@dataclass(frozen=True)
class HistogramForm:
    """The recorded form of a fill node: a histogram, identified by its spec hash."""

    spec_hash: str

    def describe(self) -> str:
        return f"histogram[{self.spec_hash}]"


def _flat(values: object) -> object:
    """Fill values flattened to 1-D: ragged arrays flatten completely (the corpus `stable`
    semantics); rectilinear arrays ravel; scalars pass through for boost broadcasting."""
    if hasattr(values, "layout"):  # an awkward array, ragged or not (lazy import boundary)
        import awkward as ak  # noqa: PLC0415

        return ak.to_numpy(ak.flatten(values, axis=None))
    arr = np.asarray(values)
    return arr.reshape(-1) if arr.ndim > 0 else arr


@dataclass(frozen=True)
class FillEvaluator:
    """The External evaluator: fill ONE chunk into a fresh zero histogram (picklable)."""

    spec: str
    n_axes: int
    has_weight: bool
    has_sample: bool
    n_weights: int = 1  # M29: multiple multiplicative weight inputs (default keeps old pickles valid)

    def __call__(self, *values: object) -> bh.Histogram:
        h = zero_of(self.spec)
        axes = [_flat(v) for v in values[: self.n_axes]]
        rest = list(values[self.n_axes :])
        weight: Any = None
        if self.has_weight:
            weight = _flat(rest.pop(0))
            for _ in range(self.n_weights - 1):
                weight = weight * _flat(rest.pop(0))  # elementwise product of the weight factors
        sample = _flat(rest.pop(0)) if self.has_sample else None
        h.fill(*axes, weight=weight, sample=sample)
        return h


@dataclass(frozen=True)
class _ZeroHist:
    spec: str

    def __call__(self) -> bh.Histogram:
        return zero_of(self.spec)


def add_histograms(a: bh.Histogram, b: bh.Histogram) -> bh.Histogram:
    """The combine: histograms form a monoid under native addition (every standard storage)."""
    return a + b


def _resolve_backend(ref: Callable[[], Any] | str) -> Any:
    """A worker's evaluation backend: a zero-arg factory/class, or an importable "module:attr"
    reference resolved HERE in the worker — behavior-carrying backends travel by import ref,
    never by pickling (behavior dicts contain lambdas; losing them must be loud, not silent)."""
    if isinstance(ref, str):
        import importlib  # noqa: PLC0415

        mod_name, _, attr = ref.partition(":")
        target = getattr(importlib.import_module(mod_name), attr)
        return target() if callable(target) else target
    return ref()


@dataclass(frozen=True)
class _FillPartition:
    """One partition's work: read, evaluate every staged fill through the compiled IR, sum."""

    ir: bytes
    source_name: str
    backend_factory: Callable[[], Any] | str
    reader: PartitionedSource
    evaluators: tuple[tuple[str, FillEvaluator], ...]
    spec: str
    columns: tuple[str, ...] | None = None  # projected read set; None reads the source's full selection

    def __call__(self, partition: Partition, resources: WorkerResources) -> bh.Histogram:
        chunk = self.reader.read_partition(partition, self.columns, resources)
        fills = evaluate_ir(
            self.ir,
            _resolve_backend(self.backend_factory),
            {self.source_name: chunk},
            externals=dict(self.evaluators),
        )
        total = zero_of(self.spec)
        for h in fills:
            total = total + h
        return total


class Histogram(bh.Histogram):
    """A ``boost_histogram.Histogram`` whose fills are DEFERRED graphed computations.

    ``fill`` records and returns ``self`` (fills accumulate). Evaluation is graphed's, not a
    method of this class: ``plan()`` exports the compute-disabled task graph (R15.4) for any R7
    executor — the executor's result IS the aggregated histogram — and the reference
    ``session.materialize(fill_node)`` evaluates one fill eagerly (an in-memory source's whole
    dataset in one chunk). The eager boost API (axes, storage, views of the EMPTY state) remains
    available.
    """

    def __init__(self, *axes: Any, storage: Any = None, metadata: Any = None) -> None:
        if storage is None:
            storage = bh.storage.Double()
        super().__init__(*axes, storage=storage, metadata=metadata)
        self._spec: str = spec_of(self)
        self._fill_nodes: list[Array] = []
        self._evaluators: dict[str, FillEvaluator] = {}

    # ---- recording -------------------------------------------------------------------------
    def fill(
        self,
        *args: Array,
        weight: Array | Sequence[Array] | None = None,
        sample: Array | None = None,
        threads: int | None = None,
    ) -> Histogram:
        if len(args) != len(self.axes):
            raise TypeError(f"this histogram has {len(self.axes)} axes; fill got {len(args)} arrays")
        if not all(isinstance(a, Array) for a in args):
            raise TypeError("deferred fills take graphed Arrays; use boost_histogram for eager data")
        del threads  # parallelism belongs to the executor, not the fill
        # M29: weight= accepts a SEQUENCE of multiplicative factors (genWeight x SFs ...); each is
        # a real graph input and evaluation multiplies them elementwise
        if weight is None:
            weights: list[Array] = []
        elif isinstance(weight, (list, tuple)):
            weights = list(weight)
        else:
            weights = [cast("Array", weight)]
        if not all(isinstance(w, Array) for w in weights):
            raise TypeError("weights must be graphed Arrays")
        inputs: list[Array] = list(args)
        inputs.extend(weights)
        if sample is not None:
            inputs.append(sample)
        session = inputs[0].session
        evaluator = FillEvaluator(
            spec=self._spec,
            n_axes=len(args),
            has_weight=bool(weights),
            has_sample=sample is not None,
            n_weights=max(len(weights), 1),
        )
        chash = content_hash(self._spec)
        descriptor = PayloadDescriptor(
            kind="histogram",
            content_hash=chash,
            framework="boost_histogram",
            version=bh.__version__,
            io_schema="uhi",
            preprocessing_ref=None,
        )
        node = session.record_external(
            "histogram.fill",
            evaluator,
            inputs,
            {
                "spec": self._spec,
                "n_axes": len(args),
                "weighted": bool(weights),
                "sampled": sample is not None,
                # only multi-weight fills carry the param: single-weight node identity unchanged
                **({"n_weights": len(weights)} if len(weights) > 1 else {}),
            },
            descriptor=descriptor,
            form=HistogramForm(chash),
        )
        self._fill_nodes.append(node)
        self._evaluators[chash] = evaluator
        return self

    def staged_fills(self) -> int:
        return len(self._fill_nodes)

    def fill_nodes(self) -> list[Array]:
        return list(self._fill_nodes)

    def evaluators(self) -> dict[str, FillEvaluator]:
        """content hash -> evaluator, for resolving this histogram's External nodes."""
        return dict(self._evaluators)

    # ---- aggregation -----------------------------------------------------------------------
    def _session_and_source(self) -> tuple[Any, int, object]:
        if not self._fill_nodes:
            raise ValueError("nothing staged: call .fill(...) before computing")
        session = self._fill_nodes[0].session
        if any(n.session is not session for n in self._fill_nodes):
            raise TypeError("all fills of one histogram must record into one session")
        sources = session.sources()
        partitioned = {nid: d for nid, d in sources.items() if isinstance(d, PartitionedSource)}
        if len(partitioned) > 1:
            raise TypeError(
                f"deferred histogram aggregation supports exactly one partitioned source; "
                f"this session has {len(partitioned)}"
            )
        if partitioned:
            ((nid, data),) = partitioned.items()
            return session, nid, data
        return session, -1, None

    def plan(
        self,
        *,
        steps_per_file: int = 1,
        backend: Callable[[], Any] | str | None = None,
        partitions: Sequence[Partition] | None = None,
    ) -> Plan[bh.Histogram]:
        """The compute-disabled task graph (R15.4): one fill task per partition, combined by
        histogram addition. Run it later with any R7 executor.

        ``backend`` is each worker's evaluation backend: a zero-arg factory/class or an
        importable ``"module:attr"`` reference resolved IN the worker (required for
        behavior-carrying backends — behavior dicts do not pickle). ``partitions`` lets the
        caller shape partitioning itself (e.g. absolute entry-count chunks for a benchmark
        sweep) instead of the source's ``steps_per_file`` split."""
        session, nid, data = self._session_and_source()
        if not isinstance(data, PartitionedSource):
            raise TypeError(
                "plan() needs a partitioned source; evaluate in-memory sources with the "
                "reference session.materialize on each fill node"
            )
        compiled = compile_ir(session, *self._fill_nodes)
        # Column projection (M5): read only the source columns the staged fills syntactically touch
        # (e.g. one MET_pt branch, not all 86), the way dask-awkward's necessary_columns does. None
        # (whole-record/opaque consumption, or an in-memory source) widens to the full selection.
        columns = read_columns(self._fill_nodes, nid)  # nid >= 0 here (plan needs a partitioned source)
        process = _FillPartition(
            ir=bytes(compiled.ir),
            source_name=session.source_name(nid),
            backend_factory=backend if backend is not None else type(session.backend),
            reader=data,
            evaluators=tuple(self._evaluators.items()),
            spec=self._spec,
            columns=columns,
        )
        if partitions is None:
            partitions = data.partitions(steps_per_file)
        tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
        return Plan(process=process, combine=add_histograms, empty=_ZeroHist(self._spec), tasks=tasks)


def factory(
    *arrays: Array,
    histref: bh.Histogram,
    weight: Array | None = None,
    sample: Array | None = None,
) -> Histogram:
    """A deferred histogram from a reference histogram's axes/storage plus one staged fill
    (the dask-histogram ``factory`` shape)."""
    out = Histogram(*histref.axes, storage=histref.storage_type())
    return out.fill(*arrays, weight=weight, sample=sample)


def _regular_axes(
    bins: int | Sequence[int], range_: Sequence[Any] | None, ndim: int
) -> list[bh.axis.Regular]:
    if isinstance(bins, list | tuple):
        bins_per = [int(b) for b in bins]
    else:
        assert isinstance(bins, int)
        bins_per = [bins] * ndim
    if range_ is None or len(bins_per) != ndim:
        raise TypeError("deferred numpy-like histograms need explicit bins and range per dimension")
    ranges = list(range_) if ndim > 1 else [range_]
    return [
        bh.axis.Regular(int(b), float(lo), float(hi)) for b, (lo, hi) in zip(bins_per, ranges, strict=True)
    ]


def histogram(
    x: Array, *, bins: int = 10, range: Sequence[float] | None = None, weights: Array | None = None
) -> Histogram:
    """numpy-like 1-D entry point: a deferred Regular-axis histogram (Int64-exact when unweighted)."""
    (axis,) = _regular_axes(bins, range, 1)
    storage = bh.storage.Weight() if weights is not None else bh.storage.Int64()
    return Histogram(axis, storage=storage).fill(x, weight=weights)


def histogram2d(
    x: Array,
    y: Array,
    *,
    bins: int | Sequence[int] = 10,
    range: Sequence[Sequence[float]] | None = None,
    weights: Array | None = None,
) -> Histogram:
    ax, ay = _regular_axes(bins, range, 2)
    storage = bh.storage.Weight() if weights is not None else bh.storage.Int64()
    return Histogram(ax, ay, storage=storage).fill(x, y, weight=weights)


def histogramdd(
    sample: Sequence[Array],
    *,
    bins: int | Sequence[int] = 10,
    range: Sequence[Sequence[float]] | None = None,
    weights: Array | None = None,
) -> Histogram:
    axes = _regular_axes(bins, range, len(sample))
    storage = bh.storage.Weight() if weights is not None else bh.storage.Int64()
    return Histogram(*axes, storage=storage).fill(*sample, weight=weights)
