"""Fill evaluation and fill recording: the evaluator, the row-space guard, the node identity."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import boost_histogram as bh
import graphed
import numpy as np
from graphed import GraphedError, Varied

from ._chunked import Partial, dense, empty
from ._spec import content_hash, spec_of

Operand = Any  # an `Array`, or a `Varied` of them (which carries `Array`'s surface)


#: this package's own version, recorded on the payloads it descriptors (the fill nodes carry
#: boost-histogram's, being boost payloads; the row-space guard is ours)
try:
    _VERSION = version("graphed-histogram")
except PackageNotFoundError:  # a src-only PYTHONPATH run: degrade the field, never the import
    _VERSION = ""


@dataclass(frozen=True)
class HistogramForm:
    """The recorded form of a fill node: a histogram, identified by its spec hash."""

    spec_hash: str

    def describe(self) -> str:
        """Return the form's display name, ``histogram[<spec hash>]``."""
        return f"histogram[{self.spec_hash}]"


def _flat(values: object) -> object:
    """Flatten fill values to 1-D.

    Ragged arrays flatten completely (the corpus ``stable`` semantics); rectilinear arrays ravel;
    scalars pass through for boost broadcasting.
    """
    if hasattr(values, "layout"):  # an awkward array, ragged or not (lazy import boundary)
        import awkward as ak

        return ak.to_numpy(ak.flatten(values, axis=None))
    arr = np.asarray(values)
    return arr.reshape(-1) if arr.ndim > 0 else arr


@dataclass(frozen=True)
class FillEvaluator:
    """The External evaluator: fill ONE chunk into a fresh zero histogram (picklable).

    Sibling mode (``variation is None``, the default — old pickles restore into it) fills once.
    In variation-axis mode ``variation`` is the ordered tuple of labels this node writes: the value
    and ``sample=`` columns are FIXED for the node and only the weight block varies across the
    loop, so the evaluator fills each label's own weight block against that label's scalar
    category value on the pre-declared ``"variation"`` axis. That is how every label resolving to
    the SAME value and ``sample=`` members collapses into ONE node — the weight-only labels onto
    nominal's, a joint label onto the shifted member its point names.

    Attributes:
        constants: scalar fill values recorded in place of inputs, as :func:`_constants` encodes
            them (None keeps old pickles valid); boost broadcasts each one, as an eager fill does
        partial: return the chunked partial, which only a plan reads, rather than the dense
            histogram
    """

    spec: str
    n_axes: int
    has_weight: bool
    has_sample: bool
    n_weights: int = (
        1  # multiple multiplicative weight inputs (the default keeps old pickles valid)
    )
    variation: tuple[str, ...] | None = None  # axis-mode label loop (None keeps old pickles valid)
    constants: str | None = None
    partial: bool = False

    def __call__(self, *values: object) -> Partial:
        """Fill one chunk's ``values`` into a fresh empty histogram and return it."""
        h = self._fill(*values)
        return h if self.partial else dense(h)

    def _fill(self, *values: object) -> Partial:
        h = empty(self.spec)
        fixed = json.loads(self.constants) if self.constants else {}
        axis_fixed = dict(fixed.get("axes", []))
        weight_fixed = dict(fixed.get("weights", []))
        rest = list(values)

        def weight_of() -> Any:
            if not self.has_weight:
                return None
            factors = _place(rest, self.n_weights, weight_fixed)
            weight = factors[0]
            for factor in factors[1:]:
                weight = weight * factor  # elementwise product of the weight factors
            return weight

        def sample_of() -> Any:
            if not self.has_sample:
                return None
            return fixed["sample"][0] if "sample" in fixed else _flat(rest.pop(0))

        axes = _place(rest, self.n_axes, axis_fixed)
        if self.variation is None:
            weight = weight_of()
            return h.fill(*axes, weight=weight, sample=sample_of())
        # axis mode: [*value_axes, sample?, *per-label weight blocks]; value/sample fixed
        sample = sample_of()
        for label in self.variation:
            # scalar category broadcasts (probe-verified)
            h = h.fill(*axes, label, weight=weight_of(), sample=sample)
        return h


#: the scalar fill values a deferred fill records as constants rather than as graph inputs
_SCALARS = (str, int, float, bool, np.integer, np.floating, np.bool_)


def _place(rest: list[object], n: int, fixed: Mapping[int, object]) -> list[Any]:
    """``n`` fill columns: each constant in its own slot, the others flattened off ``rest``."""
    return [fixed[i] if i in fixed else _flat(rest.pop(0)) for i in range(n)]


def _plain(value: object) -> object:
    """Return a numpy scalar as its Python value (``int``, ``float`` or ``bool``) for JSON."""
    return value.item() if isinstance(value, np.generic) else value


def _constants(args: Sequence[object], factors: Sequence[object], sample: object) -> str | None:
    """Encode the fill's scalar values by slot as JSON; None when there are none.

    The encoding gives node params, content hash and evaluator one identity (``1``, ``1.0``,
    ``True`` and ``"1"`` stay distinct).
    """
    fixed: dict[str, Any] = {}
    for key, column in (("axes", args), ("weights", factors)):
        slots = [[i, _plain(v)] for i, v in enumerate(column) if isinstance(v, _SCALARS)]
        if slots:
            fixed[key] = slots
    if isinstance(sample, _SCALARS):
        fixed["sample"] = [_plain(sample)]
    return json.dumps(fixed, sort_keys=True) if fixed else None


#: the three execution-time row-space contracts. The two factor messages point at the fix (the
#: value was flattened, so nothing is left to broadcast against); the loose-value one must not —
#: nothing was passed flattened, the value simply carries no handle to re-index it by.
_UNFLATTEN = (
    "pass the value unflattened (h.fill(sel.Jet.pt), never gak.flatten(...)), so the factor can "
    "be broadcast to its structure"
)
_AMBIENT_ROWS = "the ambient event weight is not at this fill's row space: " + _UNFLATTEN
_FACTOR_ROWS = "weight[{index}] is not at this fill's row space: " + _UNFLATTEN
_LOOSE_ROWS = (
    "value[{index}] carries no event context, so its rows were never re-indexed to the selection "
    "the rest of this fill's inputs live in; read that value through the same context"
)


def _depth(form: object) -> int | None:
    """Count the nested lengths a recorded form carries; ``None`` for NO row space at all.

    A per-EVENT column is 1, a per-object one 2. ``None`` means either the backend's form does not
    say (a rectilinear idiom needs no broadcast) or the operand is a scalar, which broadcasts
    against anything. That last case is `_rows`'s runtime contract read at record time;
    disagreeing with it would fire the seam on a fill whose guard then provably cannot trip,
    moving a working program's graph.

    Read off the form rather than off the data: the fill records, so this is the only handle on the
    row space at record time, and it is the one the awkward idiom already uses (``.tt.ndim``).
    """
    tt = getattr(form, "tt", None)  # the awkward form's typetracer
    ndim = getattr(tt if tt is not None else form, "ndim", None)
    return ndim if isinstance(ndim, int) and ndim > 0 else None


def _option_anywhere(layout_form: object) -> bool:
    """Tell whether an awkward layout form carries a missing value at any level.

    Record and union forms nest through ``contents``; a list or option form through ``content``.
    The list is read first because a record form also answers to ``content`` — with a bound METHOD,
    which recursing into would walk right past an optional field.
    """
    if getattr(layout_form, "is_option", False):
        return True
    nested = getattr(layout_form, "contents", None)
    if nested is None:
        content = getattr(layout_form, "content", None)
        nested = () if content is None or callable(content) else (content,)
    return any(_option_anywhere(inner) for inner in nested)


def _drops_rows(value: object) -> bool:
    """Tell whether this value — a recorded form, or a chunk of one — carries MISSING entries.

    `_flat` drops them, so such a value's leaf row space is not the one its structure counts out:
    a factor already flat at it cannot be re-nested into the value (the counts that would carry it
    no longer describe it), and at record time a factor at that row space cannot be told from one
    at the event row space either. So a fill over such a value keeps the graph and the elementwise
    semantics it had before the seam existed, and the guard blames rather than repairing.
    """
    tt = getattr(value, "tt", None)  # a recorded awkward form wraps its typetracer; a chunk is one
    layout = getattr(tt if tt is not None else value, "layout", None)
    return layout is not None and _option_anywhere(layout.form)


def _mixed_row_spaces(session: Any, value: Operand, applied: Sequence[Operand]) -> bool:
    """Tell whether any applied weight factor sits SHALLOWER than the fill's value.

    That is where the seam repairs. The evaluator flattens each input independently, so an
    un-broadcast per-event weight against a per-object value reaches boost with the wrong length
    and dies there naming neither. Deciding on the FORMS, not on the presence of a context or a
    `Varied`, is what makes the plain fill — the one every non-systematics analysis writes — take
    the same seam.

    Only in that one direction, though. `broadcast_like` lines a shallower factor up with a deeper
    value and nothing else, so firing on a DEEPER factor (or on an idiom whose backend supplies no
    seam at all, where it is the bound no-op) records a step that cannot repair anything and can
    only move — or break — a program that worked.
    """
    if getattr(session.backend, "broadcast_like", None) is None:
        return False
    value_form = session.form(graphed.member_of(value, "nominal"))
    depth = _depth(value_form)
    if depth is None or _drops_rows(value_form):
        return False
    for factor in applied:
        other = _depth(session.form(graphed.member_of(factor, "nominal")))
        if other is not None and other < depth:
            return True
    return False


def _rows(values: object) -> int | None:
    """Return a chunk's outer length, ``None`` for a scalar (which broadcasts against anything)."""
    try:
        return len(values)  # type: ignore[arg-type]
    except TypeError:
        return None


def _renest(value: object, factor: object) -> object:
    """Give `factor`, already flat at `value`'s LEAF row space, the value's nesting.

    The broadcast downstream of the guard is then the identity rather than a shape error.

    A rectilinear idiom needs nothing: its seam is the bound no-op and the evaluator ravels both.
    """
    if not hasattr(value, "layout"):  # an awkward array (the `_flat` lazy-import boundary)
        return factor
    import awkward as ak

    for axis in range(int(getattr(value, "ndim", 1)) - 1, 0, -1):
        factor = ak.unflatten(factor, ak.flatten(ak.num(value, axis=axis), axis=None))
    return factor


@dataclass(frozen=True)
class _WeightGuard:
    """The row-space guard, recorded UPSTREAM of the broadcast that lines a factor up with a value.

    That broadcast is a backend op, so an offending factor would otherwise die inside awkward's
    broadcast with a shape message naming neither the fill nor the factor. This node runs first
    (the broadcast consumes its output) and carries the one thing only the fill knows: WHICH
    operand is at the wrong row space. Record-time detection is impossible — a per-event value and
    a flattened per-object value have identical 1-D forms — so the two row spaces a fill accepts
    are told apart HERE, where both lengths are known: the value's OUTER one, which the broadcast
    then lines up, and its LEAF one, an already-flat per-object factor the evaluator would
    multiply elementwise unaided. Anything else is the blame.
    """

    message: str

    def __call__(self, factor: object, value: object) -> object:
        wide, tall = _rows(factor), _rows(value)
        if wide is None or tall is None or wide == tall:
            return factor
        if not _drops_rows(value) and wide == _rows(_flat(value)):
            return _renest(value, factor)
        raise GraphedError(self.message)


def _factor_list(weight: Operand | Sequence[Operand] | None) -> list[Operand]:
    if weight is None:
        return []
    return list(weight) if isinstance(weight, (list, tuple)) else [weight]


def _fold_labels(operands: Sequence[Operand]) -> tuple[str, ...]:
    """Return the label union in a FIXED order, folded LEFT over the operands as given.

    The order is axis values in argument order, then the ambient weight, then explicit factors in
    list order, then ``sample=`` last. Leaving the order free would let two conforming
    implementations disagree on it for one program, so the same script would give different label
    orders and a different result layout.
    """
    out: dict[str, None] = {"nominal": None}
    for operand in operands:
        if isinstance(operand, Varied):
            for label in graphed.labels(operand):
                out.setdefault(label, None)
    return tuple(out)


def _blame(
    args: Sequence[Operand], ctx: object, *, has_ambient: bool, n_factors: int
) -> tuple[tuple[str, str], ...]:
    """Return one ``(coordinate, message)`` per APPLIED factor, in fold order.

    The COORDINATE — ``ambient``, ``weight[i]``, ``value[0]`` — is the guard's identity: it enters
    the node's params and payload hash, so the diagnostic's wording can be improved without moving
    any recorded graph, while two different offenders stay two different nodes. The message is the
    evaluator's alone.

    A loose value adopts the unified context for label alignment only — its row space is never
    adjusted, no mask being known — so it is what the reader must fix. Only ``args[0]`` can take
    that blame: the guard compares each factor against the fill's FIRST value, so a loose value at
    any other axis position is not what the comparison is about.
    """
    if ctx is not None and graphed.context_of(args[0]) is None:
        return (("value[0]", _LOOSE_ROWS.format(index=0)),) * (int(has_ambient) + n_factors)
    ambient = (("ambient", _AMBIENT_ROWS),) if has_ambient else ()
    return ambient + tuple(
        (f"weight[{i}]", _FACTOR_ROWS.format(index=i)) for i in range(n_factors)
    )


def _fill_chash(
    spec: str,
    *,
    has_weight: bool,
    n_weights: int,
    variation: Sequence[str] | None = None,
    constants: str | None = None,
) -> str:
    """Return the fill node's descriptor content hash, also its External evaluator's registry key.

    `evaluate_ir` resolves `externals[content_hash]`, and the group plan merges every histogram's
    evaluators into ONE dict keyed that way. Two same-spec fills whose evaluators DIFFER must get
    distinct hashes, or both distinct nodes resolve to whichever evaluator registered last (a
    weighted fill silently evaluated unweighted — the four-output plan's `plain`/`sib` collision).

    Within one spec the evaluator can differ in three ways: unweighted vs weighted, the
    weight-factor count, and the axis-mode label loop. `has_sample` and `n_axes` cannot —
    `sample=` is only valid on Mean/WeightedMean storage and the axis count is in the spec, both
    already spec-borne. The CANONICAL single-weight sibling fill keeps `content_hash(spec)`
    verbatim, so graphs recorded before variations existed keep their identity; any deviation
    folds a discriminator suffix in.
    """
    disc: list[str] = []
    if not has_weight:
        disc.append("unweighted")
    if n_weights != 1:
        disc.append(f"n_weights={n_weights}")
    if variation is not None:
        disc.append("variation=" + json.dumps(list(variation)))
    if constants is not None:
        disc.append("constants=" + constants)
    if not disc:
        return content_hash(spec)
    return content_hash(spec + "\x00" + "\x00".join(disc))


def _fill_params(
    spec: str, shape: FillEvaluator, n_recorded: int, variation: Sequence[str] | None = None
) -> dict[str, Any]:
    """Return a fill node's params.

    ``n_axes`` counts the RECORDED axis inputs, which lead the node's inputs.
    """
    return {
        "spec": spec,
        "n_axes": n_recorded,
        "weighted": shape.has_weight,
        "sampled": shape.has_sample,
        # store params are scalars
        **({} if variation is None else {"variation": json.dumps(list(variation))}),
        # only multi-weight fills carry the param: single-weight node identity unchanged
        **({"n_weights": shape.n_weights} if shape.n_weights > 1 else {}),
        **({} if shape.constants is None else {"constants": shape.constants}),
    }


def _declare_axis_spec(hist: bh.Histogram, sorted_labels: Sequence[str]) -> str:
    """Declare a sorted `"variation"` axis at FILL time and return the fill node's spec.

    The non-growth StrCategory axis holds the inferred label set and is appended after the value
    axes. The spec is identity-bearing, hence byte-stable per partition, so every partition's
    result has the same variation axis to add on.
    """
    # non-growth by default; stored order = as given
    var = bh.axis.StrCategory(list(sorted_labels))
    var.__dict__["name"] = "variation"  # the `hist` name convention the spec codec round-trips
    declared = bh.Histogram(*hist.axes, var, storage=hist.storage_type())
    return spec_of(declared)
