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
from typing import Any

import boost_histogram as bh
import graphed
import numpy as np
from graphed import Array, GraphedError, Varied, aggregate_plan
from graphed.core import Partition, PayloadDescriptor
from graphed.core.execution import Plan

from ._spec import content_hash, spec_of, zero_of

#: §6.1c's plan-value key: a bare output name (no variation reaches it), ``(output, label)`` for a
#: varied sibling output, ``(output, None)`` for §6.2's axis mode (m50).
SlotKey = str | tuple[str, str | None]

Operand = Any  # an `Array` or a `Varied` of them (§2.2's container carries `Array`'s surface)


@dataclass(frozen=True)
class HistogramForm:
    """The recorded form of a fill node: a histogram, identified by its spec hash."""

    spec_hash: str

    def describe(self) -> str:
        return f"histogram[{self.spec_hash}]"


def _flat(values: object) -> object:
    """Fill values flattened to 1-D: ragged arrays flatten completely (the corpus ``stable``
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


#: §6.1d's three execution-time row-space contracts. The two factor messages point at the fix (the
#: value was flattened, so nothing is left to broadcast against); the loose-value one must not —
#: nothing was passed flattened, the value simply carries no handle to re-index it by.
_UNFLATTEN = (
    "pass the value unflattened (h.fill(sel.Jet.pt), never gak.flatten(...)), so the factor can be "
    "broadcast to its structure"
)
_AMBIENT_ROWS = "the ambient event weight is not at this fill's row space: " + _UNFLATTEN
_FACTOR_ROWS = "weight[{index}] is not at this fill's row space: " + _UNFLATTEN
_LOOSE_ROWS = (
    "value[{index}] carries no event context, so its rows were never re-indexed to the selection "
    "the rest of this fill's inputs live in; read that value through the same context"
)


def _rows(values: object) -> int | None:
    """A chunk's outer length, ``None`` for a scalar (which broadcasts against anything)."""
    try:
        return len(values)  # type: ignore[arg-type]
    except TypeError:
        return None


@dataclass(frozen=True)
class _WeightGuard:
    """§6.1d's row-space guard, recorded UPSTREAM of the broadcast seam.

    The seam is a backend op, so an offending factor would otherwise die inside awkward's
    broadcast with a shape message naming neither the fill nor the factor. This node runs first
    (the seam consumes its output) and carries the one thing only the fill knows: WHICH operand is
    at the wrong row space. Record-time detection is impossible — a per-event value and a
    flattened per-object value have identical 1-D forms.
    """

    message: str

    def __call__(self, factor: object, value: object) -> object:
        wide, tall = _rows(factor), _rows(value)
        if wide is not None and tall is not None and wide != tall:
            raise GraphedError(self.message)
        return factor


@dataclass(frozen=True)
class _ZeroHist:
    spec: str

    def __call__(self) -> bh.Histogram:
        return zero_of(self.spec)


def _factor_list(weight: Operand | Sequence[Operand] | None) -> list[Operand]:
    if weight is None:
        return []
    return list(weight) if isinstance(weight, (list, tuple)) else [weight]


def _member(value: Operand, label: str) -> Any:
    """§2.4's per-label narrowing: the container's own member for ``label``, else its central one."""
    if isinstance(value, Varied):
        members = graphed.labels(value)
        return graphed.universe(value, label) if label in members else graphed.nominal(value)
    return value


def _fold_labels(operands: Sequence[Operand]) -> tuple[str, ...]:
    """§6.1d's bound union ORDER, folded LEFT over the operands as given — axis values in argument
    order, then the ambient weight, then explicit factors in list order, then ``sample=`` last. An
    unbound order would give two conforming implementations different label orders for one program
    (a §3.2 determinism difference and a different §6.1c layout)."""
    out: dict[str, None] = {"nominal": None}
    for operand in operands:
        if isinstance(operand, Varied):
            for label in graphed.labels(operand):
                out.setdefault(label, None)
    return tuple(out)


def _blame(args: Sequence[Operand], ctx: object, has_ambient: bool, n_factors: int) -> tuple[str, ...]:
    """One §6.1d length message per APPLIED factor, in fold order.

    A loose value adopts the unified context for label alignment only — its row space is never
    adjusted, no mask being known — so when one is present it, not a factor, is what the reader
    must fix, and the message says so instead of pointing at the flatten."""
    loose = (
        None
        if ctx is None
        else next((i for i, value in enumerate(args) if graphed.context_of(value) is None), None)
    )
    if loose is not None:
        return (_LOOSE_ROWS.format(index=loose),) * (int(has_ambient) + n_factors)
    ambient = (_AMBIENT_ROWS,) if has_ambient else ()
    return ambient + tuple(_FACTOR_ROWS.format(index=i) for i in range(n_factors))


def fill_nodes_by_label(hist: Histogram) -> dict[str, Array]:
    """§9.1's per-label fill-node accessor: ``{label: node}`` in §2.4 order, nominal first.

    ``Histogram.fill_nodes()`` is a flat list with no label attribution, and one `fill` call's
    siblings are the only place the correspondence exists — so a histogram carrying several fill
    calls has no single answer and says so rather than hiding one of them."""
    if len(hist._label_maps) != 1:
        raise GraphedError(
            f"fill_nodes_by_label reads ONE fill call's sibling nodes; this histogram has "
            f"{len(hist._label_maps)} of them — read the flat list from Histogram.fill_nodes()"
        )
    return dict(hist._label_maps[0])


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
        self._evaluators: dict[str, Callable[..., object]] = {}
        #: one ``{label: node}`` per `fill` call, in §6.1d's fold order — the label attribution
        #: `_fill_nodes` (a flat list) cannot carry
        self._label_maps: list[dict[str, Array]] = []

    # ---- recording -------------------------------------------------------------------------
    def fill(
        self,
        *args: Operand,
        weight: Operand | Sequence[Operand] | None = None,
        sample: Operand | None = None,
        threads: int | None = None,
        unweighted: bool = False,
    ) -> Histogram:
        """Record this fill — one SIBLING node per §2.4 label (§6.1d), and return ``self``.

        The fill is the first place independent axis, weight and ``sample=`` handles meet, so it
        runs the §2.3e unification itself, re-indexes every ancestor-context input into the
        winning context's row space, and auto-applies that context's ambient weight (§2.6's
        register-then-forget, completed here). ``unweighted=True`` opts out of BOTH weight sources
        and, applying no factor, carries none of their labels.
        """
        if len(args) != len(self.axes):
            raise TypeError(f"this histogram has {len(self.axes)} axes; fill got {len(args)} arrays")
        if not all(isinstance(a, Array | Varied) for a in args):
            raise TypeError("deferred fills take graphed Arrays; use boost_histogram for eager data")
        del threads  # parallelism belongs to the executor, not the fill
        if unweighted and weight is not None:
            raise GraphedError(
                "unweighted=True suppresses the ambient event weight AND every weight= factor, so "
                "passing weight= in the same call contradicts it — drop one"
            )
        # M29: weight= accepts a SEQUENCE of multiplicative factors (genWeight x SFs ...); each is
        # a real graph input and evaluation multiplies them elementwise
        weights = _factor_list(weight)
        if not all(isinstance(w, Array | Varied) for w in weights):
            raise TypeError("weights must be graphed Arrays")
        if sample is not None and not isinstance(sample, Array | Varied):
            raise TypeError("sample= must be a graphed Array")

        given = [*args, *weights, *([] if sample is None else [sample])]
        ctx = graphed.unify_contexts(*(graphed.context_of(value) for value in given))
        ambient = None if unweighted or ctx is None else graphed.weight(ctx)
        axes = [graphed.reindex_to(value, ctx) for value in args]
        factors = [graphed.reindex_to(value, ctx) for value in weights]  # the ambient one already is
        sampled = None if sample is None else graphed.reindex_to(sample, ctx)

        applied: list[Operand] = ([] if ambient is None else [ambient]) + factors
        labels = _fold_labels([*axes, *applied, *([] if sampled is None else [sampled])])
        # §6.3(2): a fill carrying NEITHER a context handle nor a `Varied` input records exactly as
        # it did before m48 — no seam, one node, the pre-m48 golden graph
        seam = ctx is not None or any(isinstance(value, Varied) for value in given)
        blame = _blame(args, ctx, ambient is not None, len(factors))
        session = _member(axes[0], "nominal").session

        evaluator = FillEvaluator(
            spec=self._spec,
            n_axes=len(args),
            has_weight=bool(applied),
            has_sample=sample is not None,
            n_weights=max(len(applied), 1),
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
        params: dict[str, Any] = {
            "spec": self._spec,
            "n_axes": len(args),
            "weighted": bool(applied),
            "sampled": sample is not None,
            # only multi-weight fills carry the param: single-weight node identity unchanged
            **({"n_weights": len(applied)} if len(applied) > 1 else {}),
        }
        per_label: dict[str, Array] = {}
        for label in labels:
            inputs: list[Array] = [_member(value, label) for value in axes]
            value = inputs[0]
            for message, factor in zip(blame, applied, strict=True):
                narrowed = _member(factor, label)
                if seam:
                    guarded = self._guard(session, narrowed, value, message)
                    narrowed = graphed.broadcast_like(value, guarded)
                inputs.append(narrowed)
            if sampled is not None:
                inputs.append(_member(sampled, label))
            per_label[label] = session.record_external(
                "histogram.fill",
                evaluator,
                inputs,
                params,
                descriptor=descriptor,
                form=HistogramForm(chash),
            )
        self._fill_nodes.extend(per_label.values())
        self._label_maps.append(per_label)
        self._evaluators[chash] = evaluator
        return self

    def _guard(self, session: Any, factor: Array, value: Array, message: str) -> Array:
        """Record §6.1d's row-space guard for one factor (see :class:`_WeightGuard`)."""
        guard = _WeightGuard(message)
        chash = content_hash("weight-guard:" + message)
        self._evaluators[chash] = guard
        return session.record_external(  # type: ignore[no-any-return]
            "histogram.weight_guard",
            guard,
            [factor, value],
            {},
            descriptor=PayloadDescriptor(
                kind="histogram.weight_guard",
                content_hash=chash,
                framework="graphed_histogram",
                version="",
                io_schema="array",
                preprocessing_ref=None,
            ),
            form=session.form(factor),
        )

    def staged_fills(self) -> int:
        return len(self._fill_nodes)

    def fill_nodes(self) -> list[Array]:
        return list(self._fill_nodes)

    def evaluators(self) -> dict[str, Callable[..., object]]:
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
    mapping (string keys for a Mapping input, ``"0"``, ``"1"``, ... for a plain sequence). Column
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
    evaluators: dict[str, Callable[..., object]] = {}
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
