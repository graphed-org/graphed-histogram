"""The deferred ``boost_histogram.Histogram`` — fills RECORD; runners aggregate.

Each ``.fill(...)`` records one step the runner performs later, recorded the same way a
correction lookup or an ONNX model evaluation is; its evaluator returns a FILLED boost histogram
for one chunk. The step's identity is the content hash of the canonical axes/storage spec plus
its inputs, so identical fills collapse to one. Evaluation is graphed's own machinery — there is
no ``compute()`` here: ``plan()`` exports a plan (one fill task per partition over a
``graphed.write.PartitionedSource``; the whole-dataset loader is never invoked) whose combine
is :func:`add_histograms`, and ANY runner's ``run(plan).value`` IS the aggregated histogram;
``session.materialize(fill_node)`` evaluates a fill on the spot. On fixed axes, Int64 counts are
exact under any combine tree, and float sums' bits are fixed by the runner family and the
partition count. Growth axes combine to the reference stated in the docs' "Growth axes" section.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import boost_histogram as bh
import graphed
from graphed import Array, CompiledGraph, GraphedError, Varied, aggregate_plan
from graphed.core import Partition, PayloadDescriptor
from graphed.core.execution import Plan
from graphed.execute import Key, external_key

from ._chunked import Kind, dense
from ._combine import (
    Layout,
    _add_groups,
    _dense_slots,
    _GroupReduce,
    _GroupZero,
    _positions,
    _SumFills,
    _ZeroHist,
)
from ._combine import (
    add_histograms as add_histograms,
)
from ._fill import (
    _SCALARS,
    _VERSION,
    Operand,
    _blame,
    _constants,
    _declare_axis_spec,
    _factor_list,
    _fill_chash,
    _fill_params,
    _fold_labels,
    _mixed_row_spaces,
    _WeightGuard,
)
from ._fill import (
    FillEvaluator as FillEvaluator,
)
from ._fill import (
    HistogramForm as HistogramForm,
)
from ._histserv import Receipt, SlotKey, _serve
from ._spec import content_hash, spec_of

# named by the module users import them from, so pickled plans and docs keep that name
for _obj in (
    FillEvaluator,
    HistogramForm,
    _GroupReduce,
    _GroupZero,
    _SumFills,
    _WeightGuard,
    _ZeroHist,
    _add_groups,
    _dense_slots,
    add_histograms,
):
    _obj.__module__ = __name__
    if isinstance(_obj, type):
        # a bound method (`reduce.resolve_services`) pickles by its function's module
        for _attr in vars(_obj).values():
            if callable(_attr):
                _attr.__module__ = __name__
del _obj, _attr

V = TypeVar("V")
H = TypeVar("H", bound="Histogram")


@dataclass(frozen=True)
class Recording:
    """What a histogram's fills recorded and how its partials are held, as planning reads them."""

    #: one ``{label: node}`` per ``fill`` call, in fold order
    label_maps: tuple[Mapping[str, Array], ...]
    #: the spec each ``fill`` call recorded
    fill_specs: tuple[str, ...]
    #: whether the fills carry their variations on an axis
    axis_mode: bool
    #: the ``histserv.Context`` whose servers fill it, or None
    histserv: Any
    #: whether its partials and merges are dense
    dense: bool
    #: each fill's evaluator returning its chunked partial, keyed as a plan's ``externals=``
    partials: Mapping[str, Callable[..., object]]


def fill_nodes_by_label(hist: Histogram) -> dict[str, Array]:
    """Return the per-label fill nodes, ``{label: node}`` in fold order, nominal first.

    ``Histogram.fill_nodes()`` is a flat list with no label attribution, and one `fill` call's
    siblings are the only place the correspondence exists — so a histogram carrying several fill
    calls has no single answer and says so rather than hiding one of them.
    """
    label_maps = hist.recording().label_maps
    if len(label_maps) != 1:
        raise GraphedError(
            f"fill_nodes_by_label reads ONE fill call's sibling nodes; this histogram has "
            f"{len(label_maps)} of them — read the flat list from Histogram.fill_nodes()"
        )
    return dict(label_maps[0])


@dataclass(frozen=True)
class _Lowering:
    """What lowering one fill reads, in either mode, once its operands are unified."""

    session: Any
    shape: FillEvaluator
    axes: Sequence[Operand]
    applied: Sequence[Operand]
    sampled: Operand | None
    blame: Sequence[tuple[str, str]]
    broadcast: bool
    labels: tuple[str, ...]


class Histogram(bh.Histogram):
    """A ``boost_histogram.Histogram`` whose fills are DEFERRED graphed computations.

    ``fill`` records and returns ``self`` (fills accumulate). Evaluation is graphed's, not a
    method of this class: ``plan()`` exports a plan for any runner — the runner's result IS the
    aggregated histogram — and ``session.materialize(fill_node)`` evaluates one fill on the spot
    (an in-memory source's whole dataset in one chunk). The eager boost API (axes, storage, views
    of the EMPTY state) remains available.
    """

    def __init__(self, *axes: Any, storage: Any = None, metadata: Any = None) -> None:
        """Build an empty deferred histogram; ``storage`` defaults to ``Double``."""
        if storage is None:
            storage = bh.storage.Double()
        super().__init__(*axes, storage=storage, metadata=metadata)
        self._spec: str = spec_of(self)
        self._fill_nodes: list[Array] = []
        self._evaluators: dict[str, Callable[..., object]] = {}
        #: the fill evaluators a plan runs, by `graphed.execute.external_key`; they return chunked
        #: partials, where `_evaluators`, which `session.materialize` runs, return dense ones
        self._partials: dict[str, Callable[..., object]] = {}
        #: one ``{label: node}`` per `fill` call, in fold order — the label attribution
        #: `_fill_nodes` (a flat list) cannot carry
        self._label_maps: list[dict[str, Array]] = []
        #: the spec each `fill` call RECORDED. The slot layout keys on this, not on `_spec`, which
        #: is fixed in `__init__` and would lack the variation axis a fill-time axis declaration
        #: puts on the results
        self._fill_specs: list[str] = []
        #: the MODE is a property of the HISTOGRAM, fixed by the first fill and remembered —
        #: a later fill in the OTHER mode is a hard error. `False` until any fill records.
        self._axis_mode: bool = False
        #: the axis-mode inferred label set of the FIRST axis fill; a later axis fill whose
        #: set differs is refused (the declared axis, hence the spec, would disagree)
        self._axis_labels: tuple[str, ...] | None = None
        #: the `histserv.Context` whose servers fill this histogram (`histserv.backed`); outside
        #: graph identity
        self._histserv: Any = None
        #: partials and merges dense (`dense_backed`); outside graph identity
        self._dense: bool = False

    # ---- recording -------------------------------------------------------------------------
    def fill(
        self,
        *args: Operand,
        weight: Operand | Sequence[Operand] | None = None,
        sample: Operand | None = None,
        threads: int | None = None,
        unweighted: bool = False,
        variation_axis: bool = False,
    ) -> Histogram:
        """Record this fill and return ``self``.

        The fill is the first place independent axis, weight and ``sample=`` handles meet, so it
        unifies their event contexts itself, re-indexes every ancestor-context input into the
        winning context's row space, and auto-applies that context's ambient weight — the event
        weight you registered once and then forgot about. ``unweighted=True`` opts out of BOTH
        weight sources and, applying no factor, carries none of their labels.

        Default lowering is one SIBLING node per label. ``variation_axis=True`` opts into AXIS
        mode: labels group by the value/``sample=`` member they RESOLVE to, and each group folds
        into ONE evaluator-loop node over a frontend-declared ``"variation"`` StrCategory axis. The
        group resolving to nominal's member is the weight-only collapse; a label whose point names
        a shifted axis coordinate its own name does not leaves that group and joins the node of the
        member it names (still `1 + |S|` nodes). The MODE is a property of the histogram — fixed by
        the first fill and remembered; a later fill in the OTHER mode is a hard error.
        """
        del threads  # parallelism belongs to the executor, not the fill
        values, all_weights, weights, recorded_sample = self._checked_operands(
            args, weight, sample, unweighted=unweighted, variation_axis=variation_axis
        )

        given = [*values, *weights, *([] if recorded_sample is None else [recorded_sample])]
        ctx = graphed.unify_contexts(*(graphed.context_of(value) for value in given))
        ambient = None if unweighted or ctx is None else graphed.weight(ctx)
        axes = [graphed.reindex_to(value, ctx) for value in values]
        # the ambient one already is
        factors = [graphed.reindex_to(value, ctx) for value in weights]
        sampled = None if recorded_sample is None else graphed.reindex_to(recorded_sample, ctx)

        applied: list[Operand] = ([] if ambient is None else [ambient]) + factors
        labels = _fold_labels([*axes, *applied, *([] if sampled is None else [sampled])])
        blame = _blame(values, ctx, has_ambient=ambient is not None, n_factors=len(factors))
        n_weights = len(applied) + len(all_weights) - len(weights)
        shape = FillEvaluator(
            spec=self._spec,
            n_axes=len(args),
            has_weight=n_weights > 0,
            has_sample=sample is not None,
            n_weights=max(n_weights, 1),
            constants=_constants(
                args, ([] if ambient is None else [ambient]) + all_weights, sample
            ),
        )
        session = graphed.member_of(axes[0], "nominal").session
        # a fill whose factors already share the value's row space and carries neither a context
        # handle nor a `Varied` input records exactly as it did before variations existed: no
        # broadcast step, one node, the same graph as before
        broadcast = (
            ctx is not None
            or any(isinstance(value, Varied) for value in given)
            or _mixed_row_spaces(session, axes[0], applied)
        )

        low = _Lowering(session, shape, axes, applied, sampled, blame, broadcast, labels)
        if variation_axis:
            self._record_axis_fill(low)
        else:
            self._record_sibling_fill(low)
        return self

    def _checked_operands(
        self,
        args: Sequence[Operand],
        weight: Operand | Sequence[Operand] | None,
        sample: Operand | None,
        *,
        unweighted: bool,
        variation_axis: bool,
    ) -> tuple[list[Operand], list[Operand], list[Operand], Operand | None]:
        """Refuse a malformed fill; return its values, weight factors, graph factors and sample."""
        if len(args) != len(self.axes):
            raise TypeError(
                f"this histogram has {len(self.axes)} axes; fill got {len(args)} arrays"
            )
        if not all(isinstance(a, (Array, Varied, *_SCALARS)) for a in args):
            raise TypeError(
                "deferred fills take graphed Arrays or scalar constants; use boost_histogram for "
                "eager data"
            )
        values = [a for a in args if not isinstance(a, _SCALARS)]
        if not values:
            raise TypeError(
                "a deferred fill needs a graphed Array on some axis for its constants to "
                "broadcast against"
            )
        if unweighted and weight is not None:
            raise GraphedError(
                "unweighted=True suppresses the ambient event weight AND every weight= factor, so "
                "passing weight= in the same call contradicts it — drop one"
            )
        # weight= accepts a SEQUENCE of multiplicative factors (genWeight x SFs ...); each is
        # a real graph input and evaluation multiplies them elementwise
        all_weights = _factor_list(weight)
        if not all(isinstance(w, (Array, Varied, *_SCALARS)) for w in all_weights):
            raise TypeError("weights must be graphed Arrays or scalar constants")
        if sample is not None and not isinstance(sample, (Array, Varied, *_SCALARS)):
            raise TypeError("sample= must be a graphed Array or a scalar constant")
        weights = [w for w in all_weights if not isinstance(w, _SCALARS)]
        recorded_sample = None if isinstance(sample, _SCALARS) else sample

        if self._fill_specs and variation_axis != self._axis_mode:
            prior, now = ("axis", "sibling") if self._axis_mode else ("sibling", "axis")
            raise GraphedError(
                f"this histogram's mode is fixed to {prior}-fill by its first fill; this fill is "
                f"{now}-mode — the variation-axis MODE is a property of the histogram, not one "
                "fill() call (mixing them would key one output two different ways). Use one mode."
            )
        if variation_axis and any(ax.__dict__.get("name") == "variation" for ax in self.axes):
            raise GraphedError(
                "this histogram already carries an axis named 'variation'; the variation-axis "
                "fill mode declares that axis itself, so passing variation_axis=True would "
                "collide — drop the pre-existing 'variation' axis or use sibling fills"
            )

        return values, all_weights, weights, recorded_sample

    def _record_sibling_fill(self, low: _Lowering) -> None:
        """Lower one fill in sibling mode: one node per label."""
        evaluator = low.shape
        chash = _fill_chash(
            self._spec,
            has_weight=low.shape.has_weight,
            n_weights=low.shape.n_weights,
            constants=low.shape.constants,
        )
        descriptor = PayloadDescriptor(
            kind="histogram",
            content_hash=chash,
            framework="boost_histogram",
            version=bh.__version__,
            io_schema="uhi",
            preprocessing_ref=None,
        )
        params = _fill_params(self._spec, low.shape, len(low.axes))
        per_label: dict[str, Array] = {}
        for label in low.labels:
            inputs: list[Array] = [graphed.member_of(value, label) for value in low.axes]
            value = inputs[0]
            for (coordinate, message), factor in zip(low.blame, low.applied, strict=True):
                narrowed = graphed.member_of(factor, label)
                if low.broadcast:
                    guarded = self._guard(low.session, narrowed, value, coordinate, message)
                    narrowed = graphed.broadcast_like(value, guarded)
                inputs.append(narrowed)
            if low.sampled is not None:
                inputs.append(graphed.member_of(low.sampled, label))
            per_label[label] = low.session.record_external(
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
        self._register(chash, params, evaluator)

    def _register(self, chash: str, params: dict[str, Any], evaluator: FillEvaluator) -> None:
        self._evaluators[chash] = evaluator
        key = external_key({"descriptor": {"content_hash": chash}, "params": params})
        self._partials[key] = replace(evaluator, partial=True)

    def _guard(
        self, session: Any, factor: Array, value: Array, coordinate: str, message: str
    ) -> Array:
        """Record the row-space guard for one factor (see :class:`_WeightGuard`).

        Identity is the blame COORDINATE, carried in the params and hashed into the payload, so
        the node is derivable from what the graph records — a preservation plugin can rebuild the
        evaluator from `params["blame"]`, and rewording the diagnostic moves no bytes.
        """
        guard = _WeightGuard(message)
        chash = content_hash("weight-guard:" + coordinate)
        self._evaluators[chash] = guard
        node: Array = session.record_external(
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
        return node

    def _record_axis_fill(self, low: _Lowering) -> None:
        """Lower one fill in axis mode.

        A node's value and ``sample=`` columns are FIXED and only its weight block loops, so the
        fold labels group by the value/`sample=` MEMBERS they resolve to: labels sharing them share
        one evaluator-loop node, and the group whose members are nominal's is the `W` collapse.
        Grouping on the RESOLVED member rather than on label membership is what lets a joint label
        — one whose point names a shifted axis coordinate its own name does not — leave the nominal
        loop and carry its own kinematics; on default points every label resolves to its own
        member or to nominal, which is the `1 + |S|` split unchanged. Declare the sorted
        ``"variation"`` axis from the label set; each node's `content_hash((spec, variation))` is
        DISTINCT, so it resolves to its own evaluator.
        """
        if (
            self._fill_specs
            and self._axis_labels is not None
            and set(low.labels) != set(self._axis_labels)
        ):
            raise GraphedError(
                f"this axis-mode histogram's first fill declared variations "
                f"{sorted(self._axis_labels)}; this fill declares {sorted(low.labels)} — the "
                "variation axis (hence the spec) would disagree across fills. Match them."
            )
        self._axis_mode = True
        self._axis_labels = low.labels
        fixed = [*low.axes, *([] if low.sampled is None else [low.sampled])]
        groups: dict[tuple[int, ...], list[str]] = {}
        for label in low.labels:
            key = tuple(graphed.member_of(operand, label).node_id for operand in fixed)
            groups.setdefault(key, []).append(label)

        axis_spec = _declare_axis_spec(self, sorted(low.labels))
        by_label: dict[str, Array] = {}
        for group in groups.values():  # first-appearance order: nominal's group leads
            node = self._record_axis_node(
                low, axis_spec, carrier=group[0], node_labels=tuple(group)
            )
            self._fill_nodes.append(node)
            by_label.update(dict.fromkeys(group, node))
        # fold order, nominal first — the label listing and `_output_labels` read these keys
        self._label_maps.append({label: by_label[label] for label in low.labels})
        self._fill_specs.append(axis_spec)

    def _record_axis_node(
        self, low: _Lowering, axis_spec: str, *, carrier: str, node_labels: tuple[str, ...]
    ) -> Array:
        """Record one axis-mode fill node.

        Its value + `sample=` come from `carrier` (fixed), with one guarded and broadcast weight
        block per label in `node_labels` (guard and broadcast recorded per COLUMN, upstream of the
        loop). The evaluator writes each label's scalar category on the variation axis; its
        content hash folds `node_labels` in so the node owns its evaluator.
        """
        value_members = [graphed.member_of(operand, carrier) for operand in low.axes]
        value = value_members[0]
        inputs: list[Array] = list(value_members)
        if low.sampled is not None:
            inputs.append(graphed.member_of(low.sampled, carrier))
        for label in node_labels:
            for (coordinate, message), factor in zip(low.blame, low.applied, strict=True):
                narrowed = graphed.member_of(factor, label)
                if low.broadcast:
                    guarded = self._guard(low.session, narrowed, value, coordinate, message)
                    narrowed = graphed.broadcast_like(value, guarded)
                inputs.append(narrowed)
        evaluator = replace(low.shape, spec=axis_spec, variation=tuple(node_labels))
        chash = _fill_chash(
            axis_spec,
            has_weight=low.shape.has_weight,
            n_weights=low.shape.n_weights,
            variation=node_labels,
            constants=low.shape.constants,
        )
        descriptor = PayloadDescriptor(
            kind="histogram",
            content_hash=chash,
            framework="boost_histogram",
            version=bh.__version__,
            io_schema="uhi",
            preprocessing_ref=None,
        )
        params = _fill_params(axis_spec, low.shape, len(low.axes), node_labels)
        node: Array = low.session.record_external(
            "histogram.fill",
            evaluator,
            inputs,
            params,
            descriptor=descriptor,
            form=HistogramForm(chash),
        )
        self._register(chash, params, evaluator)
        return node

    def staged_fills(self) -> int:
        """Return how many fill nodes this histogram has recorded."""
        return len(self._fill_nodes)

    def fill_nodes(self) -> list[Array]:
        """Return the recorded fill nodes, in recording order, without label attribution."""
        return list(self._fill_nodes)

    def evaluators(self) -> dict[str, Callable[..., object]]:
        """Content hash -> evaluator, for resolving this histogram's External nodes."""
        return dict(self._evaluators)

    def recording(self) -> Recording:
        """Return what this histogram's fills recorded and how its partials are held."""
        return Recording(
            tuple(self._label_maps),
            tuple(self._fill_specs),
            self._axis_mode,
            self._histserv,
            self._dense,
            dict(self._partials),
        )

    def hold_partials(self, *, histserv: Any = None, dense: bool = False) -> None:
        """From the next plan on, fill on ``histserv``'s servers, or keep partials ``dense``.

        Its graph is unchanged. ``histserv.backed`` checks the histogram fits a server first.
        """
        if histserv is not None:
            self._histserv = histserv
        if dense:
            self._dense = True

    # ---- aggregation -----------------------------------------------------------------------
    def plan(
        self,
        *,
        steps_per_file: int = 1,
        backend: Callable[[], Any] | str | None = None,
        partitions: Sequence[Partition] | None = None,
    ) -> Plan[bh.Histogram]:
        """Build a plan: one fill task per partition, combined by :func:`add_histograms`.

        Run it later with any runner. Thin specialization of :func:`graphed.aggregate_plan` — this
        histogram's fills are the outputs, summed per partition and added across them; ``backend``
        is each worker's evaluation backend (factory/class or ``"module:attr"`` import ref for
        behavior-carrying backends, which do not pickle); ``partitions`` lets the caller shape
        partitioning itself. For several histograms that share a sub-graph, plan them together with
        :func:`plan` so the shared work runs ONCE.
        """
        if not self._fill_nodes:
            raise ValueError("nothing staged: call .fill(...) before computing")
        if self._histserv is not None:
            raise GraphedError(
                ".plan() runs no servers, and this histogram is histserv-backed; plan it through "
                "graphed_histogram.plan({name: hist}), which declares the servers it fills on"
            )
        # `.plan()` starts from `self._spec` (fixed in __init__, no variation axis) and
        # `_SumFills` adds every staged fill into ONE histogram — merging a varied histogram's
        # universes, and unable to render an axis-mode result at all. The refusal is GENERAL over
        # the MODE (an unvaried AXIS-mode histogram still declares the 1-bin axis), so it cannot
        # fall through into an opaque bh error. Not a fill COUNT — one varied fill refuses.
        if self._axis_mode or any(len(labels) > 1 for labels in self._label_maps):
            raise GraphedError(
                "this histogram's fills carry variations (a variation axis or sibling labels), "
                "and .plan() sums every staged fill into one histogram — which would merge the "
                "universes; plan it through graphed_histogram.plan({name: hist}), whose per-slot "
                "results keep them apart"
            )
        kind = _kind(self)
        reduce = _SumFills(
            self._spec, tuple(n.node_id for n in self._fill_nodes), dense=self._dense, kind=kind
        )

        def on_compiled(compiled: CompiledGraph) -> Any:
            reduce.positions = _positions(compiled, self._fill_nodes)
            return _variation_labels((("this histogram", self),), compiled)

        return aggregate_plan(
            *self._fill_nodes,
            reduce=reduce,
            combine=add_histograms,
            empty=_ZeroHist(self._spec, self._dense, kind),
            externals=dict(self._partials),
            backend=backend,
            steps_per_file=steps_per_file,
            partitions=partitions,
            on_compiled=on_compiled,
            finalize=dense,
        )


def _kind(h: Histogram) -> Kind:
    """Return what ``h``'s results are.

    That is the in-memory class its deferred class extends (``hist.Hist`` for
    ``hist.graphed.Hist``), carrying the public attributes ``h`` set (None is every one's
    default).
    """
    cls = next(
        c for c in type(h).__mro__ if issubclass(c, bh.Histogram) and not issubclass(c, Histogram)
    )
    return cls, tuple(
        (k, v) for k, v in vars(h).items() if not k.startswith("_") and v is not None
    )


def _output_labels(hist: Histogram) -> tuple[str, ...]:
    """Return the fold-ordered union of the labels reaching one output, over all its fills."""
    out: dict[str, None] = {"nominal": None}
    for per_label in hist.recording().label_maps:
        for label in per_label:
            out.setdefault(label, None)
    return tuple(out)


def _slots(name: str, hist: Histogram) -> Layout:
    """Return the plan-value keying for ONE output.

    That is a bare `name` when no variation reaches it — which is what keeps unvaried programs'
    plan values exactly as they were — and one `(name, label)` slot per label otherwise, each
    gathering that label's node from every fill call (the fallback: a fill that does not carry the
    label contributes its central one). Axis mode is the third key form: ONE `(name, None)` slot
    gathering every fill-node index, whatever the label count — the MODE decides the key (the
    bare-name rule is sibling-scoped), the value is the bare variation-axis histogram, the combine
    is the same `add_histograms`.
    """
    recording = hist.recording()
    spec = recording.fill_specs[0]  # the FILL node's spec (axis mode forces one per output)
    if recording.axis_mode:
        return (((name, None), tuple(node.node_id for node in hist.fill_nodes()), spec),)
    labels = _output_labels(hist)
    if len(labels) == 1:
        return ((name, tuple(node.node_id for node in hist.fill_nodes()), spec),)
    return tuple(
        (
            (name, label),
            tuple(
                per_label.get(label, per_label["nominal"]).node_id
                for per_label in recording.label_maps
            ),
            spec,
        )
        for label in labels
    )


@dataclass
class HistogramPieces:
    """The histogram half of one plan, for a plan composing these fills with outputs of its own.

    Mark the plan's own outputs, then ``fill_nodes``; hand ``reduce`` the whole value list and
    forward ``reduce.resolve_services`` and ``finalize``; pass ``combine``, ``empty`` and
    ``externals`` as they are and ``on_compiled=pieces.on_compiled``; then run ``serve(plan)``.
    ``reduce`` reads each fill at its compiled position, which ``on_compiled`` records, so a pieces
    belongs to ONE plan.
    """

    fill_nodes: tuple[Array, ...]
    reduce: _GroupReduce
    combine: Callable[[dict[SlotKey, Any], dict[SlotKey, Any]], dict[SlotKey, Any]]
    empty: _GroupZero
    externals: dict[str, Callable[..., object]]
    items: tuple[tuple[str, Histogram], ...]
    #: per histserv-backed slot: its key, spec and context
    backing: tuple[tuple[SlotKey, str, Any], ...]
    served: bool = False
    #: makes each slot the dense histogram of its class, on the head once the run is combined
    finalize: Callable[[dict[SlotKey, Any]], dict[SlotKey, Any]] = _dense_slots

    def on_compiled(self, compiled: CompiledGraph) -> Any:
        """Record where each fill landed: the plan's ``aggregate_plan(on_compiled=)`` hook.

        Returns the variation-label payload the shipped closure carries.
        """
        if self.reduce.positions:
            raise GraphedError(
                "these pieces already fed a plan; graphed_histogram.boost.pieces(...) belongs to "
                "one plan, so build fresh pieces for another"
            )
        self.reduce.positions = _positions(compiled, self.fill_nodes)
        return _variation_labels(self.items, compiled)

    def serve(self, plan: Plan[V]) -> Plan[V]:
        """Return ``plan`` with its histserv-backed slots served.

        The slots are sized, packed onto servers and declared in ``plan.services``; ``plan`` itself
        comes back when no slot is backed.
        """
        if not self.backing:
            return plan
        if self.served:
            raise GraphedError(
                "these pieces already served a plan; graphed_histogram.boost.pieces(...) belongs "
                "to one plan, so build fresh pieces for another"
            )
        served = _serve(plan, self.fill_nodes[0].session, self.backing)
        self.served = True
        return served


def pieces(histograms: Mapping[str, Histogram] | Sequence[Histogram]) -> HistogramPieces:
    """Return the parts :func:`plan` builds its plan from, for a plan with outputs of its own."""
    items = tuple(
        [(str(k), v) for k, v in histograms.items()]
        if isinstance(histograms, Mapping)
        else [(str(i), h) for i, h in enumerate(histograms)]
    )
    if not items:
        raise ValueError("plan() needs at least one histogram")
    if any(not h.staged_fills() for _, h in items):
        raise ValueError("every histogram must have at least one staged fill before planning")
    layout = tuple(slot for name, hist in items for slot in _slots(name, hist))
    evaluators: dict[str, Callable[..., object]] = {}
    for _, h in items:
        evaluators.update(h.recording().partials)
    backing = tuple(
        (key, spec, h.recording().histserv)
        for name, h in items
        if h.recording().histserv is not None
        for key, _ids, spec in _slots(name, h)
    )
    backed = tuple(key for key, _spec, _ctx in backing)
    # a histserv server is dense, so its slot sums dense before it ships
    dense = tuple(
        key
        for name, h in items
        if h.recording().dense or h.recording().histserv is not None
        for key, _ids, _spec in _slots(name, h)
    )
    kinds = tuple((key, _kind(h)) for name, h in items for key, _ids, _spec in _slots(name, h))
    return HistogramPieces(
        fill_nodes=tuple(n for _, h in items for n in h.fill_nodes()),
        reduce=_GroupReduce(layout, backed, dense=dense, kinds=kinds),
        combine=_add_groups,
        empty=_GroupZero(layout, backed, dense, kinds),
        externals=evaluators,
        items=items,
        backing=backing,
    )


def plan(
    histograms: Mapping[str, Histogram] | Sequence[Histogram],
    *,
    steps_per_file: int = 1,
    backend: Callable[[], Any] | str | None = None,
    partitions: Sequence[Partition] | None = None,
) -> Plan[dict[SlotKey, Any]]:
    """One plan that aggregates SEVERAL deferred histograms sharing a source in a SINGLE pass.

    All their fills compile into ONE IR, so a sub-graph feeding multiple histograms (e.g. a trijet
    selection feeding both a pT and a b-tag histogram) is read and evaluated ONCE — not once per
    histogram as separate ``Histogram.plan()`` calls would. The dask-histogram
    ``compute(dict_of_hists)`` analogue; ``run(plan).value`` is a flat slot-keyed mapping — a bare
    output name for an output no variation reaches, ``(output, label)`` for a varied one — which
    :func:`graphed_histogram.unpack` turns into the user-facing per-output shape.
    Column projection covers the union of all histograms' fills. Histserv-backed histograms are
    served: their servers are in ``plan.services``.
    """
    p = pieces(histograms)
    return p.serve(
        aggregate_plan(  # the shared engine: one IR, read+evaluate once, reduce per slot
            *p.fill_nodes,
            reduce=p.reduce,
            combine=p.combine,
            empty=p.empty,
            externals=p.externals,
            backend=backend,
            steps_per_file=steps_per_file,
            partitions=partitions,
            on_compiled=p.on_compiled,
            finalize=p.finalize,
        )
    )


def _cone(node: Array) -> set[int]:
    """Every record node id reachable from `node`, via `session.walk`.

    Spelled here rather than imported from `graphed.by_label`: that name is not on `graphed`'s
    exported surface, so nothing outside this file depends on how it is spelled.
    """
    seen: set[int] = set()

    def note(nid: int, *_rest: object) -> None:
        seen.add(nid)

    node.session.walk(node, source=note, op=note, external=note)
    return seen


def _variation_labels(
    items: Sequence[tuple[str, Histogram]], compiled: CompiledGraph
) -> tuple[tuple[Key, tuple[tuple[str, ...], Any]], ...] | None:
    """Which labels' universes reach each node of the compiled reduced store.

    Per label, the WHOLE record cone of that label's marked fill nodes (the central-member fallback
    included) — not a reachability difference against nominal, because the shared prefix is exactly
    where a fused failure raises — folded onto the artifact's own record→reduced keys and UNIONED,
    which is what makes the map set-valued. ``"nominal"`` never enters that union: a key no varied
    universe reaches keeps its entry with an EMPTY tuple, the single encoding of nominal/unvaried,
    and still carries the user's line.

    The artifact's ``frames`` is already one entry per key of the map's image, in order, so it is
    both the key enumeration and the frame source; nothing here computes either.
    """
    node_map = compiled.correspondence.node_map
    reached: dict[Key, set[str]] = {}
    for _name, hist in items:
        for label in _output_labels(hist):
            if label == "nominal":
                continue
            for per_label in hist.recording().label_maps:
                node = per_label.get(label, per_label["nominal"])
                for nid in _cone(node):
                    # total on this cone: every node feeding a MARKED fill survives DCE
                    reached.setdefault(node_map[nid], set()).add(label)
    if not reached:  # the predicate is over the LABELS, hence over the compiled program
        return None
    return tuple(
        (key, (tuple(sorted(reached.get(key, ()))), frame))
        for key, frame in compiled.correspondence.frames
    )


def unpack(value: Mapping[SlotKey, Any]) -> dict[str, bh.Histogram | dict[str, bh.Histogram]]:
    """Unpack the executed plan's flat slot-keyed value into the per-output shape.

    The shape is decided by the KEY FORM, which is total and per output — a bare output name is
    that output's bare histogram, ``(output, label)`` keys gather into ``{label: hist}``, and
    ``(output, None)`` is the axis-mode histogram, which carries its variations on an axis
    rather than in the mapping. A varied sibling output always carries at least two labels, so no
    output's shape is ambiguous, in a mixed plan exactly as in a single-mode one.
    ``graphed.labels``/``universe``/``nominal`` read both shapes uniformly. A histserv receipt
    unpacks to its server histogram's snapshot, which stays on the server.
    """
    out: dict[str, Any] = {}
    for key, held in value.items():
        hist = held.snapshot() if isinstance(held, Receipt) else dense(held)
        if not isinstance(key, tuple):
            out[key] = hist
            continue
        name, label = key
        if label is None:
            out[name] = hist
        else:
            out.setdefault(name, {})[label] = hist
    return out


def dense_backed(h: H) -> H:
    """Return ``h``, its partials and merges dense boost histograms from its next plan on.

    Its graph is unchanged. The default backing keeps a histogram with a category axis chunked:
    one dense slice per filled category key (``hist.chunked.ChunkedHist``). Either way the result
    is the same dense histogram, bin for bin.
    """
    h.hold_partials(dense=True)
    return h


def label_listing(histograms: Mapping[str, Histogram]) -> dict[str, list[str]]:
    """List each output's variation labels: the plan-level ``{output: [labels]}`` listing.

    Each output maps to its variation labels in FOLD order (nominal first, then vary-tag insertion
    order); an unvaried output maps to ``["nominal"]``.

    MODE-INDEPENDENT: an axis-mode output lists the SAME labels as its sibling twin, because both
    modes populate ``_label_maps`` with the same fold-ordered set. The labels come from the fill's
    DECLARED label set the builder holds (``_output_labels``), NOT from the ``(output, None)`` slot
    key — which carries no label, so a key-reading listing would answer ``[None]``/``[]`` for an
    axis-mode output.
    """
    return {name: list(_output_labels(hist)) for name, hist in histograms.items()}


def factory(
    *arrays: Array,
    histref: bh.Histogram,
    weight: Array | None = None,
    sample: Array | None = None,
) -> Histogram:
    """Build a deferred histogram from a reference histogram's axes/storage plus one staged fill.

    This is the dask-histogram ``factory`` shape.
    """
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
        raise TypeError(
            "deferred numpy-like histograms need explicit bins and range per dimension"
        )
    ranges = list(range_) if ndim > 1 else [range_]
    return [
        bh.axis.Regular(int(b), float(lo), float(hi))
        for b, (lo, hi) in zip(bins_per, ranges, strict=True)
    ]


def histogram(
    x: Array, *, bins: int = 10, range: Sequence[float] | None = None, weights: Array | None = None
) -> Histogram:
    """Fill a deferred Regular-axis histogram: the numpy-like 1-D entry point.

    Its storage is Int64, exact, when unweighted.
    """
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
    """Fill a deferred 2-D Regular-axis histogram: the numpy-like ``histogram2d``."""
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
    """Fill a deferred N-D Regular-axis histogram: the numpy-like ``histogramdd``."""
    axes = _regular_axes(bins, range, len(sample))
    storage = bh.storage.Weight() if weights is not None else bh.storage.Int64()
    return Histogram(*axes, storage=storage).fill(*sample, weight=weights)
