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
from importlib.metadata import version
from typing import Any

import boost_histogram as bh
import graphed
import numpy as np
from graphed import Array, GraphedError, Varied, aggregate_plan, compile_ir
from graphed.core import GraphStore, Partition, PayloadDescriptor
from graphed.core.execution import Plan

from ._spec import content_hash, spec_of, zero_of

#: §6.1c's plan-value key: a bare output name (no variation reaches it), ``(output, label)`` for a
#: varied sibling output, ``(output, None)`` for §6.2's axis mode (m50).
SlotKey = str | tuple[str, str | None]

Operand = Any  # an `Array` or a `Varied` of them (§2.2's container carries `Array`'s surface)

#: this package's own version, recorded on the payloads it descriptors (the fill nodes carry
#: boost-histogram's, being boost payloads; the row-space guard is ours)
_VERSION = version("graphed-histogram")


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


def _blame(
    args: Sequence[Operand], ctx: object, has_ambient: bool, n_factors: int
) -> tuple[tuple[str, str], ...]:
    """One ``(coordinate, message)`` per APPLIED factor, in fold order.

    The COORDINATE — ``ambient``, ``weight[i]``, ``value[0]`` — is the guard's identity: it enters
    the node's params and payload hash, so the diagnostic's wording can be improved without moving
    any recorded graph, while two different offenders stay two different nodes. The message is the
    evaluator's alone.

    A loose value adopts the unified context for label alignment only — its row space is never
    adjusted, no mask being known — so it is what the reader must fix. Only ``args[0]`` can take
    that blame: the guard compares each factor against the fill's FIRST value, so a loose value at
    any other axis position is not what the comparison is about."""
    if ctx is not None and graphed.context_of(args[0]) is None:
        return (("value[0]", _LOOSE_ROWS.format(index=0)),) * (int(has_ambient) + n_factors)
    ambient = (("ambient", _AMBIENT_ROWS),) if has_ambient else ()
    return ambient + tuple((f"weight[{i}]", _FACTOR_ROWS.format(index=i)) for i in range(n_factors))


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


#: §6.1c's slot layout: per slot, its key, the OUTPUT INDICES it sums, and its spec. Indices, not
#: counts — two labels whose members are structurally identical intern to ONE node (§1.2), and
#: `evaluate_ir` returns one value per DISTINCT output, so a shared index simply replicates.
Layout = tuple[tuple[SlotKey, tuple[int, ...], str], ...]


@dataclass(frozen=True)
class _GroupReduce:
    """Reduce one partition's evaluated fills to ``{slot: histogram}`` — each histogram is the sum of
    its OWN fills, sliced out of the single shared one-pass evaluation by ``layout``."""

    layout: Layout

    def __call__(self, fills: list[object]) -> dict[SlotKey, bh.Histogram]:
        out: dict[SlotKey, bh.Histogram] = {}
        for key, indices, spec in self.layout:
            total = zero_of(spec)
            for i in indices:
                total = total + fills[i]
            out[key] = total
        return out


def _add_groups(
    a: dict[SlotKey, bh.Histogram], b: dict[SlotKey, bh.Histogram]
) -> dict[SlotKey, bh.Histogram]:
    """Combine: histogram groups add key-wise (each histogram is a monoid under native +)."""
    return {key: a[key] + b[key] for key in a}


@dataclass(frozen=True)
class _GroupZero:
    layout: Layout

    def __call__(self) -> dict[SlotKey, bh.Histogram]:
        return {key: zero_of(spec) for key, _indices, spec in self.layout}


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
        #: the spec each `fill` call RECORDED. §6.1c keys the layout on this, not on `_spec`, which
        #: is fixed in `__init__` and under m50's fill-time axis declaration would lack the
        #: variation axis the fill results carry
        self._fill_specs: list[str] = []

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
            for (coordinate, message), factor in zip(blame, applied, strict=True):
                narrowed = _member(factor, label)
                if seam:
                    guarded = self._guard(session, narrowed, value, coordinate, message)
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
        self._fill_specs.append(evaluator.spec)
        self._evaluators[chash] = evaluator
        return self

    def _guard(self, session: Any, factor: Array, value: Array, coordinate: str, message: str) -> Array:
        """Record §6.1d's row-space guard for one factor (see :class:`_WeightGuard`).

        Identity is the blame COORDINATE, carried in the params and hashed into the payload, so
        the node is derivable from what the graph records — a preservation plugin can rebuild the
        evaluator from `params["blame"]`, and rewording the diagnostic moves no bytes."""
        guard = _WeightGuard(message)
        chash = content_hash("weight-guard:" + coordinate)
        self._evaluators[chash] = guard
        return session.record_external(  # type: ignore[no-any-return]
            "histogram.weight_guard",
            guard,
            [factor, value],
            {"blame": coordinate},
            descriptor=PayloadDescriptor(
                kind="histogram.weight_guard",
                content_hash=chash,
                framework="graphed_histogram",
                version=_VERSION,
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
        # §6.1c: `_SumFills` adds every staged fill into ONE histogram, which for a varied
        # histogram would silently merge the universes into a plausible, physically wrong answer.
        # The trigger is the merge hazard, not a fill COUNT — a single varied fill refuses too.
        if any(len(labels) > 1 for labels in self._label_maps):
            raise GraphedError(
                "this histogram's fills carry variation labels, and .plan() sums every staged fill "
                "into one histogram — which would merge the universes; plan it through "
                "graphed_histogram.plan({name: hist}), whose per-slot results keep them apart"
            )
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


def _output_labels(hist: Histogram) -> tuple[str, ...]:
    """The §2.4-ordered union of the labels reaching one output, over all its fill calls."""
    out: dict[str, None] = {"nominal": None}
    for per_label in hist._label_maps:
        for label in per_label:
            out.setdefault(label, None)
    return tuple(out)


def _slots(name: str, hist: Histogram, rank: Mapping[int, int]) -> Layout:
    """§6.1a/§6.1c's keying for ONE output: a bare `name` when no variation reaches it — which is
    what keeps unvaried programs' plan values exactly as they were — and one `(name, label)` slot
    per label otherwise, each gathering that label's node from every fill call (§2.4's fallback:
    a fill that does not carry the label contributes its central one)."""
    labels = _output_labels(hist)
    spec = hist._fill_specs[0]  # §6.1c: the FILL node's spec (m50's §6.2(i) forces one per output)
    if len(labels) == 1:
        return ((name, tuple(rank[node.node_id] for node in hist._fill_nodes), spec),)
    return tuple(
        (
            (name, label),
            tuple(rank[per_label.get(label, per_label["nominal"]).node_id] for per_label in hist._label_maps),
            spec,
        )
        for label in labels
    )


def plan(
    histograms: Mapping[str, Histogram] | Sequence[Histogram],
    *,
    steps_per_file: int = 1,
    backend: Callable[[], Any] | str | None = None,
    partitions: Sequence[Partition] | None = None,
) -> Plan[dict[SlotKey, bh.Histogram]]:
    """One plan that aggregates SEVERAL deferred histograms sharing a source in a SINGLE pass.

    All their fills compile into ONE IR, so a sub-graph feeding multiple histograms (e.g. a trijet
    selection feeding both a pT and a b-tag histogram) is read and evaluated ONCE — not once per
    histogram as separate ``Histogram.plan()`` calls would. The dask-histogram
    ``compute(dict_of_hists)`` analogue; ``run(plan).value`` is the flat slot-keyed mapping §6.1c
    binds — a bare output name for an output no variation reaches, ``(output, label)`` for a varied
    one — which :func:`graphed_histogram.unpack` turns into the user-facing per-output shape.
    Column projection covers the union of all histograms' fills."""
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
    # §6.1c/§7.2: a slot's operand is the rank of its node id in the DEDUPLICATED id list, which
    # matches `evaluate_ir`'s one-value-per-distinct-output list element for element (`Array` is
    # unhashable, so the dedup runs over ids). A raw index into the staged list overruns it.
    rank = {nid: i for i, nid in enumerate(dict.fromkeys(n.node_id for n in fill_nodes))}
    layout = tuple(slot for name, hist in items for slot in _slots(name, hist, rank))
    evaluators: dict[str, Callable[..., object]] = {}
    for h in hists:
        evaluators.update(h._evaluators)
    varied = any(len(_output_labels(h)) > 1 for _name, h in items)
    return aggregate_plan(  # the shared engine: one IR, read+evaluate once, reduce per slot
        *fill_nodes,
        reduce=_GroupReduce(layout),
        combine=_add_groups,
        empty=_GroupZero(layout),
        externals=evaluators,
        backend=backend,
        steps_per_file=steps_per_file,
        partitions=partitions,
        on_compiled=_merge_guard(items, len(rank)) if varied else None,
    )


def _merge_guard(items: Sequence[tuple[str, Histogram]], marked: int) -> Callable[[Any], None]:
    """§7.2's optimizer-merge refusal, at the group-plan builder — the only site holding both the
    marked record ids and the compiled artifact.

    The M4 reducer merges DISTINCT record ids too (``x * 1.0`` is an identity token), so two fills
    differing only in ``weight=[w]`` versus ``weight=[w * 1.0]`` compile to ONE output while the
    slot layout still expects two. The sound key — the record-to-reduced map — does not exist until
    m49, so a shortfall is refused rather than mis-sliced (a mis-slice surfaces as an opaque
    worker-side ``IndexError``). Scoped to varied programs: an unvaried one whose fills the same
    rules merge must keep running exactly as it did."""

    def shrinks(hist: Histogram) -> bool:
        """Does THIS output's own compile lose fills? The refusal must name the histogram whose
        fills merged, which in a mixed plan need not be a varied one."""
        ids = dict.fromkeys(node.node_id for node in hist._fill_nodes)
        compiled = compile_ir(hist._fill_nodes[0].session, *hist._fill_nodes)
        return len(GraphStore.deserialize(compiled.ir).outputs()) < len(ids)

    def check(compiled: Any) -> None:
        outputs = len(GraphStore.deserialize(compiled.ir).outputs())
        if outputs >= marked:
            return
        # the shortfall is real; re-compiling per output to attribute it costs nothing on a path
        # that is about to raise. No single output shrinking means the merge crossed two of them.
        culprits = [(name, hist) for name, hist in items if shrinks(hist)] or list(items)
        detail = "; ".join(f"{name} carries {list(_output_labels(hist))}" for name, hist in culprits)
        workaround = (
            " Spell a label whose value equals another's with the SAME expression "
            '(variations={"1": w}, not w * 1.0), which routes it through the supported '
            "record-time dedup instead."
            if any(len(_output_labels(hist)) > 1 for _name, hist in culprits)
            else ""
        )
        raise GraphedError(
            f"the optimizer merged fills that record as distinct nodes ({marked} marked, {outputs} "
            f"compiled), so this plan's slots can no longer be told apart: {detail}.{workaround}"
        )

    return check


def unpack(value: Mapping[SlotKey, bh.Histogram]) -> dict[str, bh.Histogram | dict[str, bh.Histogram]]:
    """§6.1a's result unpacker: the executed plan's flat slot-keyed value as the per-output shape.

    The shape is decided by the KEY FORM, which is total and per output — a bare output name is
    that output's bare histogram, ``(output, label)`` keys gather into ``{label: hist}``, and
    ``(output, None)`` is §6.2's axis-mode histogram (m50), which carries its variations on an axis
    rather than in the mapping. A varied sibling output always carries at least two labels, so no
    output's shape is ambiguous, in a mixed plan exactly as in a single-mode one.
    ``graphed.labels``/``universe``/``nominal`` read both shapes uniformly."""
    out: dict[str, Any] = {}
    for key, hist in value.items():
        if not isinstance(key, tuple):
            out[key] = hist
            continue
        name, label = key
        if label is None:
            out[name] = hist
        else:
            out.setdefault(name, {})[label] = hist
    return out


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
