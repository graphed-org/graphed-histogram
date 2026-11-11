"""Adding histograms, and the reduce/combine/empty callables a plan runs."""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import boost_histogram as bh
import numpy as np
from graphed import Array, CompiledGraph, GraphedError
from graphed.aggregate import Fold
from graphed.core import GraphStore

from ._chunked import BH, Chunked, Kind, Partial, as_kind, dense, empty
from ._histserv import Receipt, SlotKey, _ship
from ._spec import _make_axis, zero_of


@dataclass(frozen=True)
class _ZeroHist:
    spec: str
    dense: bool = False
    kind: Kind = BH

    def __call__(self) -> Partial:
        return (
            as_kind(zero_of(self.spec), self.kind) if self.dense else empty(self.spec, self.kind)
        )


def _grows_regular(axis: Any) -> bool:
    return isinstance(axis, bh.axis.Regular) and bool(axis.traits.growth)


def _width(axis: Any) -> float:
    return float(axis.edges[-1] - axis.edges[0]) / int(axis.size)


def _union(x: Any, y: Any) -> Any:
    """Return the growing ``Regular`` axis spanning ``x`` and ``y``.

    ``None`` when they are not two differing growing ``Regular`` axes on one grid with equal
    metadata.
    """
    if not (_grows_regular(x) and _grows_regular(y) and vars(x) == vars(y) and x != y):
        return None
    first = (float(y.edges[0]) - float(x.edges[0])) / _width(x)
    last = (float(y.edges[-1]) - float(x.edges[0])) / _width(x)
    k = round(first)
    # 1e-6 of a bin admits grids that drifted by float rounding while they grew
    if abs(first - k) > 1e-6 or abs(last - round(last)) > 1e-6 or round(last) - k != y.size:
        return None
    lo = y.edges[0] if k < 0 else x.edges[0]
    hi = y.edges[-1] if k + y.size > x.size else x.edges[-1]
    union = bh.axis.Regular(max(x.size, k + y.size) - min(0, k), lo, hi, growth=True)
    union.__dict__.update(vars(x))
    return union


def _onto(hist: bh.Histogram, unions: Sequence[Any]) -> bh.Histogram:
    """``hist`` re-binned onto ``unions`` (``None`` keeps that axis), flow bins to flow bins."""
    axes = [ax if u is None else u for ax, u in zip(hist.axes, unions, strict=True)]
    out = type(hist)(*axes, storage=hist.storage_type())
    out.__dict__.update(vars(hist))
    index = []
    for ax, u in zip(hist.axes, unions, strict=True):
        if u is None:
            index.append(np.arange(ax.extent))
        else:  # a growing Regular always has both flow bins
            first = round((float(ax.edges[0]) - float(u.edges[0])) / _width(u))
            index.append(np.r_[0, np.arange(ax.size) + first + 1, u.extent - 1])
    out.view(flow=True)[np.ix_(*index)] = hist.view(flow=True)
    return out


def add_histograms(a: bh.Histogram, b: bh.Histogram) -> bh.Histogram:
    """Add two histograms: the plan's combine.

    It is native ``+``, after widening each pair of growing ``Regular`` axes on one grid (equal
    bin width, edges a whole number of bins apart, equal metadata) to their bin-aligned union.
    Every pair native ``+`` refuses otherwise stays refused.

    Returns a new ``boost_histogram.Histogram`` carrying ``a``'s metadata; neither operand
    changes. Growth categories come out as ``a``'s, then ``b``'s new ones, so ``a`` must hold
    the lower-keyed partitions (the docs' "Growth axes" section). Two chunked partials of one spec
    merge chunked; any other pair is made dense first.

    >>> a = bh.Histogram(bh.axis.Regular(3, 0, 3)).fill([0.5, 1.5])
    >>> gh.add_histograms(a, bh.Histogram(bh.axis.Regular(3, 0, 3)).fill([1.5])).values().tolist()
    [1.0, 2.0, 0.0]
    >>> grew_up = bh.Histogram(bh.axis.Regular(1, 0, 1, growth=True)).fill([2.5])
    >>> grew_down = bh.Histogram(bh.axis.Regular(1, 0, 1, growth=True)).fill([-0.5])
    >>> gh.add_histograms(grew_up, grew_down).axes[0].edges.tolist()
    [-1.0, 0.0, 1.0, 2.0, 3.0]
    """
    merged = a.merged(b) if isinstance(a, Chunked) else None
    if merged is not None:
        return merged
    a, b = dense(a), dense(b)
    unions = [_union(x, y) for x, y in zip(a.axes, b.axes, strict=False)]
    # a rank mismatch skips the widening so native + refuses it with boost's own error
    if a.ndim == b.ndim and any(u is not None for u in unions):
        a, b = _onto(a, unions), _onto(b, unions)
    return a + b


def _positions(compiled: CompiledGraph, nodes: Sequence[Array]) -> dict[int, int]:
    """Return each marked fill's index among the values ``compiled`` evaluates to.

    The optimizer may merge two marked fills into one output (``weight=[w]`` and
    ``weight=[w * 1.0]``); reading each fill at its own position reads that output once per marked
    fill, which is what filling twice means.
    """
    order = {cid: i for i, cid in enumerate(GraphStore.deserialize(bytes(compiled.ir)).outputs())}
    return {n.node_id: order[compiled.correspondence.node_map[n.node_id][0]] for n in nodes}


_NO_POSITIONS = (
    "this histogram reduce has no fill positions: its plan's aggregate_plan(on_compiled=) never "
    "fired; pass on_compiled=pieces.on_compiled"
)


@dataclass
class _SumFills:
    """Reduce one partition's evaluated fills to a single histogram.

    That is the sum of this histogram's staged fills, each read at its compiled position
    (``positions``, set by the plan's hook).
    """

    spec: str
    node_ids: tuple[int, ...]
    positions: dict[int, int] = field(default_factory=dict)
    dense: bool = False
    kind: Kind = BH

    def output_summaries(self) -> list[str]:
        """Describe this histogram for a plan's repr."""
        return [_summary("", self.spec, self.kind, 1)]

    def __call__(self, fills: list[object]) -> Partial:
        """Sum the fills into this slot's partial, as ``kind``.

        A dense slot sums into the dense zero, adding each chunked fill dense; a chunked slot adds
        each fill into the total it owns, in place.
        """
        total: Any = _ZeroHist(self.spec, self.dense, self.kind)()
        for nid in self.node_ids:
            fill = fills[self.positions[nid]]
            if not (isinstance(total, Chunked) and total.iadd(fill)):
                total = add_histograms(total, fill)
        return total


def _summary(name: str, spec: str, kind: Kind, labels: int) -> str:
    """Say what one output is: ``mass (Hist, 120 bins, 3 labels)``."""
    axes = [_make_axis(axis) for axis in json.loads(spec)["axes"]]
    bins = math.prod(max(axis.size, 1) for axis in axes)
    grows = "+" if any(axis.traits.growth for axis in axes) else ""
    shape = f"{kind[0].__name__}, {bins}{grows} bin{'' if bins == 1 and not grows else 's'}"
    if labels > 1:
        shape += f", {labels} labels"
    return f"{name} ({shape})" if name else shape


#: the slot layout: per slot, its key, the RECORD ids of the fills it sums (one per staged fill, so
#: an id interned twice repeats), and its spec
Layout = tuple[tuple[SlotKey, tuple[int, ...], str], ...]


@dataclass
class _GroupReduce:
    """Reduce one partition's evaluated fills to ``{slot: histogram}``.

    Each slot sums its own fills, each read at its compiled position (``positions``, set by
    ``pieces.on_compiled``). A ``backed`` slot's sum is shipped to its histserv server and its
    value is the receipt; a ``dense`` slot sums dense; each slot's value is its ``kinds`` entry's
    class. Each ``scalars`` slot is its reduction's chunk value, read the same way, and each
    ``writes`` slot lists this chunk's part path, which the plan's values end with.
    """

    layout: Layout
    backed: tuple[SlotKey, ...] = ()
    positions: dict[int, int] = field(default_factory=dict)
    dense: tuple[SlotKey, ...] = ()
    kinds: tuple[tuple[SlotKey, Kind], ...] = ()
    #: per whole-dataset reduction slot: its key and its record node id
    scalars: tuple[tuple[SlotKey, int], ...] = ()
    writes: tuple[SlotKey, ...] = ()

    def output_summaries(self) -> list[str]:
        """Describe each output for a plan's repr, counting the labels that are slots of it."""
        kinds = dict(self.kinds)
        outputs: dict[str, list[tuple[SlotKey, str]]] = {}
        for key, _ids, spec in self.layout:
            outputs.setdefault(key[0] if isinstance(key, tuple) else key, []).append((key, spec))
        return [
            _summary(name, slots[0][1], kinds.get(slots[0][0], BH), len(slots))
            for name, slots in outputs.items()
        ] + [
            f"{name} (reduction)"
            for name in dict.fromkeys(
                key[0] if isinstance(key, tuple) else key for key, _ in self.scalars
            )
        ]

    def __call__(self, fills: list[object]) -> dict[SlotKey, Any]:
        if not self.positions:
            raise GraphedError(_NO_POSITIONS)
        kinds = dict(self.kinds)
        out = {
            key: _SumFills(spec, ids, self.positions, key in self.dense, kinds.get(key, BH))(fills)
            for key, ids, spec in self.layout
        }
        if self.backed:
            backed = set(self.backed)
            out = {
                key: replace(_ship(key, spec, out[key]), kind=kinds.get(key))
                if key in backed
                else out[key]
                for key, _ids, spec in self.layout
            }
        out.update((key, fills[self.positions[nid]]) for key, nid in self.scalars)
        paths = fills[len(fills) - len(self.writes) :]
        out.update((key, [path]) for key, path in zip(self.writes, paths, strict=True))
        return out

    def resolve_services(self, value: dict[SlotKey, Any]) -> dict[SlotKey, Any]:
        """Replace each receipt in ``value`` by its server histogram, deleted from the server."""
        return {
            key: h.snapshot(delete=True) if isinstance(h, Receipt) else h
            for key, h in value.items()
        }


#: a write slot's fold: its part paths, in combine order
PATHS = Fold(operator.add, [])

#: per slot that is not a histogram, a reduction's or a write's: its key and its fold
Folds = tuple[tuple[SlotKey, Fold], ...]


@dataclass(frozen=True)
class _AddGroups:
    """Add two slot groups key-wise: the group plan's combine.

    A ``folds`` slot folds by its :class:`~graphed.aggregate.Fold`; histograms add through
    :func:`add_histograms`, and receipts by their own ``+``.
    """

    folds: Folds = ()

    def __call__(self, a: dict[SlotKey, Any], b: dict[SlotKey, Any]) -> dict[SlotKey, Any]:
        folds = dict(self.folds)
        return {
            key: folds[key](a[key], b[key])
            if key in folds
            else a[key] + b[key]
            if isinstance(a[key], Receipt)
            else add_histograms(a[key], b[key])
            for key in a
        }


@dataclass(frozen=True)
class _GroupZero:
    layout: Layout
    backed: tuple[SlotKey, ...] = ()
    dense: tuple[SlotKey, ...] = ()
    kinds: tuple[tuple[SlotKey, Kind], ...] = ()
    folds: Folds = ()

    def __call__(self) -> dict[SlotKey, Any]:
        kinds = dict(self.kinds)

        def zero(key: SlotKey, spec: str) -> Any:
            kind = kinds.get(key, BH)
            if key in self.backed:
                return Receipt(spec, kind=kind)
            return as_kind(zero_of(spec), kind) if key in self.dense else empty(spec, kind)

        return {
            **{key: zero(key, spec) for key, _ids, spec in self.layout},
            **{key: fold.empty for key, fold in self.folds},
        }


def _dense_slots(value: dict[SlotKey, Any]) -> dict[SlotKey, Any]:
    """Return ``value`` with each chunked slot dense: the group plan's finalize."""
    return {key: dense(h) for key, h in value.items()}
