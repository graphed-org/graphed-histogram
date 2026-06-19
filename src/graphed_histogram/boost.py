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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import boost_histogram as bh
import numpy as np
from graphed import Array, aggregate_plan
from graphed_core import Partition, PayloadDescriptor
from graphed_core.execution import Plan

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


@dataclass(frozen=True)
class _SumFills:
    """Reduce one partition's evaluated fills to a single histogram (the single-histogram case): the
    partition result is the sum of that histogram's own fills."""

    spec: str

    def __call__(self, fills: list[object]) -> bh.Histogram:
        total = zero_of(self.spec)
        for f in fills:
            total = total + f
        return total


@dataclass(frozen=True)
class _GroupReduce:
    """Reduce one partition's evaluated fills to ``{label: histogram}`` — each histogram is the sum of
    its OWN fills, sliced out of the single shared one-pass evaluation by ``layout``."""

    layout: tuple[tuple[str, int, str], ...]  # (label, n_fills, spec), in compiled-fill order

    def __call__(self, fills: list[object]) -> dict[str, bh.Histogram]:
        out: dict[str, bh.Histogram] = {}
        i = 0
        for label, k, spec in self.layout:
            total = zero_of(spec)
            for j in range(i, i + k):
                total = total + fills[j]
            out[label] = total
            i += k
        return out


def _add_groups(a: dict[str, bh.Histogram], b: dict[str, bh.Histogram]) -> dict[str, bh.Histogram]:
    """Combine: histogram groups add key-wise (each histogram is a monoid under native +)."""
    return {label: a[label] + b[label] for label in a}


@dataclass(frozen=True)
class _GroupZero:
    layout: tuple[tuple[str, int, str], ...]

    def __call__(self) -> dict[str, bh.Histogram]:
        return {label: zero_of(spec) for label, _k, spec in self.layout}


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
    def plan(
        self,
        *,
        steps_per_file: int = 1,
        backend: Callable[[], Any] | str | None = None,
        partitions: Sequence[Partition] | None = None,
    ) -> Plan[bh.Histogram]:
        """The compute-disabled task graph (R15.4): one fill task per partition, combined by
        histogram addition. Run it later with any R7 executor.

        Thin specialization of :func:`graphed.aggregate_plan` — this histogram's fills are the
        outputs, summed per partition and added across them; ``backend`` is each worker's evaluation
        backend (factory/class or ``"module:attr"`` import ref for behavior-carrying backends, which
        do not pickle); ``partitions`` lets the caller shape partitioning itself. For several
        histograms that share a sub-graph, plan them together with :func:`plan` so the shared work
        runs ONCE."""
        if not self._fill_nodes:
            raise ValueError("nothing staged: call .fill(...) before computing")
        return aggregate_plan(
            *self._fill_nodes,
            reduce=_SumFills(self._spec),
            combine=add_histograms,
            empty=_ZeroHist(self._spec),
            externals=self._evaluators,
            backend=backend,
            steps_per_file=steps_per_file,
            partitions=partitions,
        )


def plan(
    histograms: Mapping[str, Histogram] | Sequence[Histogram],
    *,
    steps_per_file: int = 1,
    backend: Callable[[], Any] | str | None = None,
    partitions: Sequence[Partition] | None = None,
) -> Plan[dict[str, bh.Histogram]]:
    """One plan that aggregates SEVERAL deferred histograms sharing a source in a SINGLE pass.

    All their fills compile into ONE IR, so a sub-graph feeding multiple histograms (e.g. a trijet
    selection feeding both a pT and a b-tag histogram) is read and evaluated ONCE — not once per
    histogram as separate ``Histogram.plan()`` calls would. The dask-histogram
    ``compute(dict_of_hists)`` analogue; ``run(plan).value`` is the matching ``{label: histogram}``
    mapping (string keys for a Mapping input, ``"0"``,``"1"``,... for a plain sequence). Column
    projection covers the union of all histograms' fills."""
    items = (
        [(str(k), v) for k, v in histograms.items()]
        if isinstance(histograms, Mapping)
        else [(str(i), h) for i, h in enumerate(histograms)]
    )
    if not items:
        raise ValueError("plan() needs at least one histogram")
    hists = [h for _, h in items]
    if any(not h._fill_nodes for h in hists):
        raise ValueError("every histogram must have at least one staged fill before planning")
    fill_nodes = [n for h in hists for n in h._fill_nodes]
    layout = tuple((label, len(h._fill_nodes), h._spec) for label, h in items)
    evaluators: dict[str, FillEvaluator] = {}
    for h in hists:
        evaluators.update(h._evaluators)
    return aggregate_plan(  # the shared engine: one IR, read+evaluate once, reduce per histogram
        *fill_nodes,
        reduce=_GroupReduce(layout),
        combine=_add_groups,
        empty=_GroupZero(layout),
        externals=evaluators,
        backend=backend,
        steps_per_file=steps_per_file,
        partitions=partitions,
    )


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
