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

A fill leaves this object's storage empty, so reading one with staged fills warns
(:class:`UnrunHistogramWarning`).
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any, TypeVar, overload

import boost_histogram as bh
import graphed
from graphed import Array, GraphedError, PlanPieces, Varied, aggregate_plan, collate
from graphed.aggregate import fold_of
from graphed.core import DurablePlanV2, Partition, PayloadDescriptor
from graphed.core.execution import Plan
from graphed.execute import CompiledGraph, Key, external_key
from graphed.provenance import capture
from graphed.systematics import (
    ScalarFactor,
    broadcast_like,
    context_of,
    member_of,
    reindex_to,
    unify_contexts,
)
from graphed.write import PartWrite

from ._chunked import Kind, dense
from ._combine import (
    PATHS,
    Layout,
    _AddGroups,
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
    _lifted,
    _mixed_row_spaces,
    _node_factors,
    _WeightGuard,
    check_fill_types,
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
    _AddGroups,
    _GroupReduce,
    _GroupZero,
    _SumFills,
    _WeightGuard,
    _ZeroHist,
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


class UnrunHistogramWarning(UserWarning):
    """A histogram with staged fills was read, or used, as though it held data.

    Its storage is empty until it runs, and what is done with it is not recorded into the run. To
    make this an error: ``warnings.filterwarnings("error", category=UnrunHistogramWarning)``.
    """


def _warns(base: Callable[..., Any], *, reflected: bool = False) -> Any:
    """Return ``base`` warning first for each operand with staged fills."""

    # not all of `functools.WRAPPER_ASSIGNMENTS`: the docs build mocks boost_histogram, and a
    # mock's `__type_params__` is not the tuple a function must hold
    @wraps(base, assigned=("__name__", "__qualname__", "__doc__"))
    def seam(self: Histogram, *args: Any, **kwargs: Any) -> Any:
        for operand in (self, *args):
            if isinstance(operand, Histogram):
                _warn_if_unrun(operand)
        if reflected and isinstance(args[0], bh.Histogram):
            return NotImplemented
        return base(self, *args, **kwargs)

    return seam


class Histogram(bh.Histogram):
    """A ``boost_histogram.Histogram`` whose fills are DEFERRED graphed computations.

    ``fill`` records and returns ``self`` (fills accumulate). Evaluation is graphed's, not a
    method of this class: ``plan()`` exports a plan for any runner — the runner's result IS the
    aggregated histogram — and ``session.materialize(fill_node)`` evaluates one fill on the spot
    (an in-memory source's whole dataset in one chunk). This object never holds the result: its
    axes and storage type stay available, and with no fill recorded it reads as the empty histogram
    it is, but reading it with fills staged warns (:class:`UnrunHistogramWarning`).
    """

    view = _warns(bh.Histogram.view)
    sum = _warns(bh.Histogram.sum)
    empty = _warns(bh.Histogram.empty)
    project = _warns(bh.Histogram.project)
    reset = _warns(bh.Histogram.reset)
    __getitem__ = _warns(bh.Histogram.__getitem__)
    __eq__ = _warns(bh.Histogram.__eq__)
    # boost-histogram 1.7 reaches none of the other seams from these three
    __ne__ = _warns(bh.Histogram.__ne__)
    __mul__ = _warns(bh.Histogram.__mul__)
    __truediv__ = _warns(bh.Histogram.__truediv__)
    # ahead of `__iadd__`'s and `__isub__`'s seams, so the copy they run on inherits the warning
    __add__ = _warns(bh.Histogram.__add__)
    __sub__ = _warns(bh.Histogram.__sub__)
    __iadd__ = _warns(bh.Histogram.__iadd__)
    __isub__ = _warns(bh.Histogram.__isub__)
    __imul__ = _warns(bh.Histogram.__imul__)
    __itruediv__ = _warns(bh.Histogram.__itruediv__)
    # a right operand of a subclass goes first, so `eager + h` lands here before eager's `__add__`
    __radd__ = _warns(bh.Histogram.__radd__, reflected=True)
    __rmul__ = _warns(bh.Histogram.__rmul__, reflected=True)
    if hasattr(bh.Histogram, "__rsub__"):  # boost-histogram 1.8 added these two together
        __rsub__ = _warns(bh.Histogram.__rsub__, reflected=True)
        __rtruediv__ = _warns(bh.Histogram.__rtruediv__, reflected=True)

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

    def _import_bh_(self) -> None:
        # what a fill records, which a copy must not share with the original
        for name in ("_fill_nodes", "_evaluators", "_partials", "_label_maps", "_fill_specs"):
            self.__dict__[name] = copy.copy(self.__dict__[name])

    def __deepcopy__(self, memo: Any) -> Any:
        """Copy storage, axes and fill lists; the recorded nodes stay on the session's graph."""
        memo.update((id(item), item) for item in (self._histserv, *self._fill_nodes))
        return super().__deepcopy__(memo)

    @classmethod
    def _export_bh_(cls, target: bh.Histogram) -> None:
        if not isinstance(target, Histogram):
            _warn_if_unrun(target)
        super()._export_bh_(target)

    def __repr__(self) -> str:
        """Show the histogram, marked with its staged fills when it has any."""
        n = self.staged_fills()
        # a clone holds no fills, and its repr reads no staged storage
        plain = repr(type(self)._clone(self._hist)) if n else super().__repr__()
        return f"{plain} # {n} staged fill{'s' if n > 1 else ''}, not run" if n else plain

    def __str__(self) -> str:
        """Draw the histogram; with staged fills there is nothing to draw, so show its repr."""
        return repr(self) if self.staged_fills() else super().__str__()

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

        >>> h = gh.boost.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Weight())
        >>> h.fill(events.x, weight=[events.w, 2.0]) is h  # weights multiply; nothing is read yet
        True
        >>> h.staged_fills()
        1
        """
        del threads  # parallelism belongs to the executor, not the fill
        values, all_weights, weights, recorded_sample = self._checked_operands(
            args, weight, sample, unweighted=unweighted, variation_axis=variation_axis
        )

        given = [*values, *weights, *([] if recorded_sample is None else [recorded_sample])]
        ctx = unify_contexts(*(context_of(value) for value in given))
        ambient = None if unweighted or ctx is None else graphed.systematics.weight(ctx)
        check_fill_types(self.axes, args, ambient, weights, recorded_sample)
        # lifting records nodes, so it waits until every refusal has had its say
        all_weights = _lifted(all_weights, values)
        weights = _node_factors(all_weights)
        axes = [reindex_to(value, ctx) for value in values]
        # the ambient one already is
        factors = [reindex_to(value, ctx) for value in weights]
        sampled = None if recorded_sample is None else reindex_to(recorded_sample, ctx)

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
        session = member_of(axes[0], "nominal").session
        # a fill whose factors already share the value's row space and carries neither a context
        # handle nor a `Varied` input records exactly as it did before variations existed: no
        # broadcast step, one node, the same graph as before
        broadcast = (
            ctx is not None
            or any(isinstance(value, Varied) for value in (*given, *weights))
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
        if not all(isinstance(w, (Array, Varied, ScalarFactor, *_SCALARS)) for w in all_weights):
            raise TypeError("weights must be graphed Arrays or scalar constants")
        if sample is not None and not isinstance(sample, (Array, Varied, *_SCALARS)):
            raise TypeError("sample= must be a graphed Array or a scalar constant")
        weights = _node_factors(all_weights)
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
            inputs: list[Array] = [member_of(value, label) for value in low.axes]
            value = inputs[0]
            for (coordinate, message), factor in zip(low.blame, low.applied, strict=True):
                narrowed = member_of(factor, label)
                if low.broadcast:
                    guarded = self._guard(low.session, narrowed, value, coordinate, message)
                    narrowed = broadcast_like(value, guarded)
                inputs.append(narrowed)
            if low.sampled is not None:
                inputs.append(member_of(low.sampled, label))
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
            key = tuple(member_of(operand, label).node_id for operand in fixed)
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
        value_members = [member_of(operand, carrier) for operand in low.axes]
        value = value_members[0]
        inputs: list[Array] = list(value_members)
        if low.sampled is not None:
            inputs.append(member_of(low.sampled, carrier))
        for label in node_labels:
            for (coordinate, message), factor in zip(low.blame, low.applied, strict=True):
                narrowed = member_of(factor, label)
                if low.broadcast:
                    guarded = self._guard(low.session, narrowed, value, coordinate, message)
                    narrowed = broadcast_like(value, guarded)
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
        return len(self.__dict__.get("_fill_nodes", ()))

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
        steps_per_file: int | None = None,
        events_per_task: int | None = None,
        backend: Callable[[], Any] | str | None = None,
        partitions: Sequence[Partition] | None = None,
    ) -> Plan[bh.Histogram] | DurablePlanV2:
        """Build a plan: one fill task per partition, combined by :func:`add_histograms`.

        Run it later with any runner. Thin specialization of :func:`graphed.aggregate_plan` — this
        histogram's fills are the outputs, summed per partition and added across them; ``backend``
        is each worker's evaluation backend (factory/class or ``"module:attr"`` import ref for
        behavior-carrying backends, which do not pickle); ``steps_per_file`` overrides the source's
        own setting (``None`` keeps it); ``events_per_task`` instead cuts each file into tasks of
        that many entries; ``partitions`` lets the caller shape partitioning itself.
        Over a ``graphed.join`` the plan is the join's ``DurablePlanV2``, with no ``tasks`` or
        ``process``: the fills run per join destination, and ``partitions`` and
        ``events_per_task`` are refused. For several histograms that share a sub-graph, plan them
        together with :func:`plan` so the shared work runs ONCE.

        The run's value is the filled histogram, of the class this one extends:

        >>> h = gh.boost.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Int64())
        >>> plan = h.fill(events.x).plan()
        >>> SequentialRunner().run(plan).value.values().tolist()
        [1, 2, 1]
        """
        _refuse_outside_counts(self)
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
            events_per_task=events_per_task,
            partitions=partitions,
            on_compiled=on_compiled,
            finalize=dense,
        )

    def __graphed_pieces__(self) -> PlanPieces:
        """Return this histogram's share of the plan :func:`graphed.materialize` runs it in.

        Its result is a new histogram, as a run builds it, and this one stays un-run: what
        :func:`unpack` gives for it, the in-memory class filled, or for a varied one
        ``{label: histogram}``. With no fills it is that class, empty.
        """
        _refuse_outside_counts(self)
        if self._histserv is not None:
            raise GraphedError(
                "graphed.materialize runs no servers, so it cannot fill this histserv-backed "
                "histogram: run the analysis to fill it, through graphed_histogram.plan({name: "
                "hist}) on an executor that starts them"
            )
        if not self._fill_nodes:
            return PlanPieces(
                outputs=(),
                reduce=dense,
                combine=add_histograms,
                empty=_ZeroHist(self._spec, True, _kind(self)),
                finalize=dense,
                on_compiled=lambda _: None,
            )
        p = pieces({"filled": self})
        return PlanPieces(
            outputs=p.fill_nodes,
            reduce=p.reduce,
            combine=p.combine,
            empty=p.empty,
            finalize=lambda value: unpack(value)["filled"],
            on_compiled=p.on_compiled,
            externals=p.externals,
        )


def _refuse_outside_counts(h: Histogram, name: str | None = None) -> None:
    """Raise if ``h`` has counts no fill recorded (``+``, ``+=``, view writes): runs lose them."""
    if not bh.Histogram.empty(h, flow=True):
        which = "this histogram" if name is None else f"histogram {name!r}"
        raise ValueError(
            f"{which} holds counts that no .fill() recorded (sum "
            f"{bh.Histogram.sum(h, flow=True)}), and a run builds its result from zero, so they "
            "would be lost: plan a histogram that has only fills, then add the counts to the "
            "result with gh.add_histograms(result, counts), or fill them"
        )


def _warn_if_unrun(h: bh.Histogram) -> None:
    """Warn at the user's line if ``h`` has staged fills; ``hist.Hist(h)`` carries them too."""
    if Histogram.staged_fills(h):
        where = capture()
        warnings.warn_explicit(
            "this histogram has not run, so it reads as empty, and what you do with it here "
            "(arithmetic, h[...], project, ==, hist.Hist(h)) is not recorded into the run. Use "
            "`filled = graphed.materialize(h)`, or run a plan, and read `filled`: `h` itself "
            "stays un-run.",
            UnrunHistogramWarning,
            where.filename,
            where.lineno,
            # per file, as its key has no filename; per histogram, or the next one's read of this
            # line would be hidden
            registry=h.__dict__.setdefault("_warned", {}).setdefault(where.filename, {}),
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


#: one whole-dataset reduction among a plan's outputs: its slot key and its Array
_Scalar = tuple[SlotKey, Array]


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
    scalars: tuple[_Scalar, ...] = ()
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
        return _variation_labels(self.items, compiled, self.scalars)

    @overload
    def serve(self, plan: Plan[V]) -> Plan[V]: ...
    @overload
    def serve(self, plan: DurablePlanV2) -> DurablePlanV2: ...
    def serve(self, plan: Plan[V] | DurablePlanV2) -> Plan[V] | DurablePlanV2:
        """Return ``plan`` with its histserv-backed slots served.

        The slots are sized, packed onto servers and declared in ``plan.services``; ``plan`` itself
        comes back when no slot is backed. A backed slot refuses a plan with ``next_tasks`` or over
        a join, which has no tasks to size by.
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
    return _pieces(items)


def _pieces(
    items: tuple[tuple[str, Histogram], ...],
    scalars: tuple[_Scalar, ...] = (),
    writes: tuple[str, ...] = (),
) -> HistogramPieces:
    """Return the pieces of ``items``, with ``scalars`` and ``writes`` slots beside theirs."""
    for name, h in items:
        _refuse_outside_counts(h, name)
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
    folds = (*((key, fold_of(array)) for key, array in scalars), *((w, PATHS) for w in writes))
    return HistogramPieces(
        fill_nodes=(*(n for _, h in items for n in h.fill_nodes()), *(a for _, a in scalars)),
        reduce=_GroupReduce(
            layout,
            backed,
            dense=dense,
            kinds=kinds,
            scalars=tuple((key, array.node_id) for key, array in scalars),
            writes=writes,
        ),
        combine=_AddGroups(folds),
        empty=_GroupZero(layout, backed, dense, kinds, folds),
        externals=evaluators,
        items=items,
        backing=backing,
        scalars=scalars,
    )


#: where one output sits in what the caller passed to :func:`plan`: the mapping keys down to it
_Path = tuple[Any, ...]


#: what :func:`plan` takes as one output: a histogram, or a whole-dataset reduction (an Array or a
#: ``Varied`` of them)
_Output = Histogram | Array | Varied

_TAKES = "gh.plan takes histograms and whole-dataset reductions"


def _universes(output: Array | Varied) -> dict[str | None, Array]:
    """Return a reduction's Array per label, under ``None`` when it is not varied."""
    if isinstance(output, Array):
        return {None: output}
    return {
        label: graphed.systematics.universe(output, label)
        for label in graphed.systematics.labels(output)
    }


def _leaves(histograms: Any, path: _Path = ()) -> list[tuple[_Path, _Output]]:
    """Return each output in a nesting of mappings, with its path, in mapping order."""
    if isinstance(histograms, Histogram):
        return [(path, histograms)]
    if isinstance(histograms, (Array, Varied)):
        for array in _universes(histograms).values():
            try:
                fold_of(array)
            except TypeError as err:
                raise TypeError(f"{_TAKES}; output {_named(path)!r} {err}") from None
        return [(path, histograms)]
    if isinstance(histograms, Mapping):
        for key in histograms:
            if isinstance(key, tuple):  # :func:`unpack` reads a tuple as (output, label)
                raise TypeError(
                    f"gh.plan takes no tuple keys; nest the mapping instead of {key!r}"
                )
        return [leaf for key, value in histograms.items() for leaf in _leaves(value, (*path, key))]
    if isinstance(histograms, Iterable) and not path:
        return _leaves({str(i): h for i, h in enumerate(histograms)})
    held = _named(path) or "the argument"
    raise TypeError(f"{_TAKES}; {held} is {type(histograms).__name__}")


def _named(path: _Path) -> str:
    return "/".join(map(str, path))


def _session_of(output: _Output | PartWrite) -> Any:
    if isinstance(output, PartWrite):
        return output.array.session
    if isinstance(output, Histogram):
        return output.fill_nodes()[0].session if output.staged_fills() else None
    return next(iter(_universes(output).values())).session


@dataclass(frozen=True)
class _Nested:
    """A plan's finalize: its value in the shape the caller passed to :func:`plan`."""

    paths: tuple[_Path, ...]
    collated: bool = False

    def __call__(self, value: Mapping[Any, Any]) -> Any:
        slots = (
            {k: v for part in value.values() for k, v in part.items()} if self.collated else value
        )
        results = unpack(slots)
        if self.paths == ((),):
            return next(iter(results.values()))
        out: dict[Any, Any] = {}
        for path in self.paths:
            if _named(path) in results:
                node = out
                for key in path[:-1]:
                    node = node.setdefault(key, {})
                node[path[-1]] = results[_named(path)]
        return out


def _one_plan(
    outputs: Mapping[str, _Output | PartWrite],
    *,
    finalize: Callable[[Any], Any] | None = None,
    **kwargs: Any,
) -> Plan[Any] | DurablePlanV2:
    writes = {name: w for name, w in outputs.items() if isinstance(w, PartWrite)}
    scalars = tuple(
        (name if label is None else (name, label), array)
        for name, output in outputs.items()
        if isinstance(output, (Array, Varied))
        for label, array in _universes(output).items()
    )
    hists = tuple((name, h) for name, h in outputs.items() if isinstance(h, Histogram))
    p = _pieces(hists, scalars, tuple(writes))
    built = aggregate_plan(  # the shared engine: one IR, read+evaluate once, reduce per slot
        *p.fill_nodes,
        writes=tuple(writes.values()),
        reduce=p.reduce,
        combine=p.combine,
        empty=p.empty,
        externals=p.externals,
        on_compiled=p.on_compiled,
        finalize=finalize or p.finalize,
        **kwargs,
    )
    return p.serve(built)


def plan(
    histograms: Any,
    *,
    steps_per_file: int | None = None,
    events_per_task: int | None = None,
    backend: Callable[[], Any] | str | None = None,
    partitions: Sequence[Partition] | None = None,
    writes: Mapping[str, PartWrite] | None = None,
) -> Plan[Any] | DurablePlanV2:
    """One plan that aggregates SEVERAL deferred histograms sharing a source in a SINGLE pass.

    All their fills compile into ONE IR, so a sub-graph feeding multiple histograms (e.g. a trijet
    selection feeding both a pT and a b-tag histogram) is read and evaluated ONCE — not once per
    histogram as separate ``Histogram.plan()`` calls would. The dask-histogram
    ``compute(dict_of_hists)`` analogue. ``histograms`` is a histogram, a mapping or sequence of
    them, or mappings nested to any depth.

    An output may also be a whole-dataset reduction (``sum``, ``count``, ``count_nonzero``,
    ``any``, ``all``, ``min`` or ``max`` with no axis, per :func:`graphed.aggregate.fold_of`),
    read in the same pass; any other array raises a ``TypeError`` naming the rule. ``writes``
    maps names to :class:`graphed.write.PartWrite` (``parquet_write(...)``), each writing one part
    per task from the same read, its result the list of part paths.

    ``run(plan).value`` comes back in the nesting that went in, each output a histogram or a
    value, or a ``{label: ...}`` of them for a varied one, and each write under its name. Over
    several Sessions, a coffea fileset's for one, each Session is one plan, collated
    (:func:`graphed.collate`) into one with the sum of their tasks, so ``partitions`` needs one
    Session, and a dataset with no tasks is absent. ``steps_per_file`` overrides the source's own
    setting (``None`` keeps it); ``events_per_task`` instead cuts each file into tasks of that many
    entries. Column projection covers the union of all outputs' columns.

    Histserv-backed histograms are served, in a flat mapping over one Session, and their servers
    are in ``plan.services``. That value stays slot-keyed, a bare output name or
    ``(output, label)`` for a varied one, and :func:`graphed_histogram.unpack` turns it into the
    same shape, reading any receipts.

    Over a ``graphed.join`` the plan is the join's ``DurablePlanV2``, for ``SequentialRunner`` or
    ``SubmitRunner``. Every output then reads past the join, in one Session, and neither
    ``partitions``, ``events_per_task`` nor histserv applies.

    >>> w = graphed.vary(events.w, "jes", up=events.w * 1.5)
    >>> met = hg.Hist.new.Reg(3, 0, 3, name="x").Weight().fill(x=events.x, weight=w)
    >>> sumw, n = gak.sum(w, axis=None), gak.count(events.x, axis=None)
    >>> plan = gh.plan({"met": met, "sumw": sumw, "n": n})
    >>> out = SequentialRunner().run(plan).value
    >>> {label: h.sum().value for label, h in out["met"].items()}
    {'nominal': 5.0, 'jes_up': 7.5}
    >>> {label: float(v) for label, v in out["sumw"].items()}, int(out["n"])
    ({'nominal': 5.0, 'jes_up': 7.5}, 4)
    """
    outputs = _leaves(histograms)
    if not outputs:
        raise ValueError("gh.plan needs at least one histogram or reduction")
    for name, w in (writes or {}).items():
        if not isinstance(w, PartWrite):
            raise TypeError(
                f"gh.plan(writes=) takes PartWrite values; {name!r} is {type(w).__name__}"
            )
    if writes and any(path == () for path, _ in outputs):
        raise TypeError("gh.plan(writes=) names its results, so pass the outputs as a mapping")
    for path, _ in outputs:
        if writes and path[0] in writes:
            raise ValueError(f"gh.plan: {path[0]!r} names both an output and a write")
    leaves = [*outputs, *(((name,), w) for name, w in (writes or {}).items())]
    flat = {_named(path): h for path, h in leaves}
    if len(flat) != len(leaves):
        raise ValueError("gh.plan: two outputs have the same '/'-joined name")
    by_session: dict[Any, dict[str, _Output | PartWrite]] = {}
    for name, h in flat.items():
        by_session.setdefault(_session_of(h), {})[name] = h
    kwargs: dict[str, Any] = {
        "steps_per_file": steps_per_file,
        "events_per_task": events_per_task,
        "backend": backend,
        "partitions": partitions,
    }
    paths = tuple(path for path, _ in leaves)
    served = [
        name
        for name, h in flat.items()
        if isinstance(h, Histogram) and h.recording().histserv is not None
    ]
    if served:
        if len(by_session) > 1 or any(len(path) != 1 for path in paths):
            raise TypeError(
                f"histserv-backed output {served[0]!r} needs a flat mapping over one Session, "
                "whose receipts graphed_histogram.unpack reads"
            )
        return _one_plan(flat, **kwargs)
    if len(by_session) < 2:
        return _one_plan(flat, finalize=_Nested(paths), **kwargs)
    if partitions is not None:
        raise TypeError("gh.plan(partitions=) names one source's tasks, and these are several")
    groups = {next(iter(group)): group for group in by_session.values()}
    collated = collate({name: _one_plan(group, **kwargs) for name, group in groups.items()})
    return replace(collated, finalize=_Nested(paths, collated=True))


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
    items: Sequence[tuple[str, Histogram]],
    compiled: CompiledGraph,
    scalars: Sequence[_Scalar] = (),
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
    universes = [
        (label, per_label.get(label, per_label["nominal"]))
        for _name, hist in items
        for label in _output_labels(hist)
        for per_label in hist.recording().label_maps
    ]
    # a varied reduction's universe is keyed (output, label)
    universes += [
        (label, array)
        for key, array in scalars
        if isinstance(key, tuple) and (label := key[1]) is not None
    ]
    reached: dict[Key, set[str]] = {}
    for label, node in universes:
        if label != "nominal":
            for nid in _cone(node):
                # total on this cone: every node feeding a MARKED output survives DCE
                reached.setdefault(node_map[nid], set()).add(label)
    if not reached:  # the predicate is over the LABELS, hence over the compiled program
        return None
    return tuple(
        (key, (tuple(sorted(reached.get(key, ()))), frame))
        for key, frame in compiled.correspondence.frames
    )


def unpack(value: Mapping[Any, Any]) -> dict[str, bh.Histogram | dict[str, bh.Histogram]]:
    """Unpack a slot-keyed value, a served plan's or ``pieces``', into the per-output shape.

    The shape is decided by the KEY FORM, which is total and per output — a bare output name is
    that output's bare histogram, ``(output, label)`` keys gather into ``{label: hist}``, and
    ``(output, None)`` is the axis-mode histogram, which carries its variations on an axis
    rather than in the mapping. A varied sibling output always carries at least two labels, so no
    output's shape is ambiguous, in a mixed plan exactly as in a single-mode one.
    ``graphed.systematics.labels``/``universe``/``nominal`` read both shapes uniformly. A
    histserv receipt unpacks to its server histogram's snapshot, which stays on the server.

    A collated ``{dataset: value}`` unpacks per dataset, and an unpacked value unpacks to itself.

    >>> a = bh.Histogram(bh.axis.Regular(2, 0, 2))
    >>> out = gh.unpack({("met", "nominal"): a, ("met", "jes_up"): a, "n": a})
    >>> {name: list(v) if isinstance(v, dict) else type(v).__name__ for name, v in out.items()}
    {'met': ['nominal', 'jes_up'], 'n': 'Histogram'}
    """
    out: dict[str, Any] = {}
    for key, held in value.items():
        if isinstance(held, Mapping):
            out[key] = unpack(held)
            continue
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


def label_listing(histograms: Mapping[Any, Any]) -> dict[Any, Any]:
    """List each output's variation labels, before anything runs.

    Each output maps to its labels, nominal first, then each ``graphed.vary`` label as it was
    declared; an unvaried output maps to ``["nominal"]``. Nested mappings list in the same nesting.
    A histogram filled with ``variation_axis=True`` lists the same labels its variation axis
    holds, though the axis orders its categories its own way.

    >>> w = graphed.vary(events.w, "jes", up=events.w * 1.1, down=events.w * 0.9)
    >>> met = hg.Hist.new.Reg(3, 0, 3, name="x").Weight().fill(x=events.x, weight=w)
    >>> n = hg.Hist.new.Reg(3, 0, 3, name="x").Int64().fill(x=events.x)
    >>> gh.label_listing({"met": met, "n": n})
    {'met': ['nominal', 'jes_up', 'jes_down'], 'n': ['nominal']}
    """
    # declared labels, not slot keys: an axis-mode output's ``(output, None)`` key carries none
    return {
        name: label_listing(h) if isinstance(h, Mapping) else list(_output_labels(h))
        for name, h in histograms.items()
    }


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
