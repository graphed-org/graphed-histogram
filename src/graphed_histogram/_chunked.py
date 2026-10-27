"""Chunked partials: a fill's histogram held as a ``hist.chunked.ChunkedHist``.

A ChunkedHist stores one dense slice of the non-category axes per filled category key, so a
partial and every merge of partials cost what was filled rather than the whole category product.
:class:`Chunked` is that partial, an internal type: fills and
:func:`~graphed_histogram.add_histograms` keep it chunked, and a plan's ``finalize`` turns the
result into the class the user instantiated, bin for bin what a dense fill gives. ChunkedHist takes
one scalar key per fill, so :meth:`Chunked.fill` groups a fill's events by key, stably, which keeps
every bin's sum in event order.
"""

from __future__ import annotations

import json
from typing import Any, TypeAlias

import boost_histogram as bh
import hist
import numpy as np
from hist.chunked import ChunkedHist

from ._spec import _GROWTH_SPEC_VERSION, _STORAGES, SPEC_VERSION, _make_axis, zero_of

# axis type -> the dtype kinds bh fills it with
_CATEGORY = {"StrCategory": "U", "IntCategory": "iu"}


def chunkable(spec: str) -> bool:
    """Tell whether a ChunkedHist holds this spec's histogram with something to gain.

    That takes a category axis and a non-category one, no growing non-category axis, and not a
    Mean storage.
    """
    payload = json.loads(spec)
    cat = [a["type"] in _CATEGORY for a in payload["axes"]]
    return (
        payload["version"] in (SPEC_VERSION, _GROWTH_SPEC_VERSION)  # else zero_of refuses it
        and payload["storage"] not in ("Mean", "WeightedMean")
        and any(cat)
        and not all(cat)
        and not any(a.get("growth") for a, c in zip(payload["axes"], cat, strict=True) if not c)
    )


def _internal_axis(i: int, a: dict[str, Any]) -> Any:
    """Build axis ``i`` as the ChunkedHist holds it.

    A declared category axis is keyed by bin INDEX, its overflow one more, so an undeclared label
    lands where a dense fill puts it.
    """
    name = f"a{i}"
    if a["type"] not in _CATEGORY:
        return _make_axis({**a, "metadata": {"name": name}})
    if not a.get("growth"):
        return hist.axis.IntCategory([], growth=True, name=name)
    return getattr(hist.axis, a["type"])(a["categories"], growth=True, name=name)


#: what a slot's result is: the in-memory class the user instantiated, and that histogram's public
#: attributes (``name``, ``label``, ``metadata``)
Kind = tuple[type, tuple[tuple[str, Any], ...]]
BH: Kind = (bh.Histogram, ())


def as_kind(h: bh.Histogram, kind: Kind) -> bh.Histogram:
    """Return ``h`` as ``kind``'s class with its attributes.

    hist's own constructor shares ``h``'s storage.
    """
    cls, attrs = kind
    out = h if type(h) is cls else cls(h)
    for name, value in attrs:
        setattr(out, name, value)
    return out


def _variances_unknown(attrs: tuple[tuple[str, Any], ...]) -> tuple[tuple[str, Any], ...]:
    """Return ``attrs`` marking the variances unknown, so ``variances()`` is None.

    bh does so on a sum, or on a weighted fill of a storage without variances.
    """
    return (*(a for a in attrs if a[0] != "_variance_known"), ("_variance_known", False))


#: what a fill, a sum or a merge yields: a chunked partial, or a dense histogram
Partial: TypeAlias = "bh.Histogram | Chunked"


class Chunked:
    """A partial held as a ChunkedHist: its spec, chunks, and the ``kind`` it reads as."""

    __slots__ = ("chunked", "kind", "spec")

    def __init__(self, spec: str, chunked: ChunkedHist | None = None, kind: Kind = BH) -> None:
        """Hold ``chunked``, or empty chunks for ``spec``, as a partial of ``kind``."""
        if chunked is None:
            payload = json.loads(spec)
            axes = [_internal_axis(i, a) for i, a in enumerate(payload["axes"])]
            chunked = ChunkedHist(*axes, storage=_STORAGES[payload["storage"]]())
        self.spec = spec
        self.chunked = chunked
        self.kind = kind

    def _unknown(self) -> None:
        cls, attrs = self.kind
        self.kind = (cls, _variances_unknown(attrs))

    def _grown(self) -> tuple[dict[str, Any], list[dict[Any, int] | None]]:
        """Return the spec with its growth categories as filled, and the category axes' keys.

        The keys are, per category axis, its label -> index map (None: keyed by index).
        """
        payload = json.loads(self.spec)
        known = iter(c.known_keys for c in self.chunked.chunk_axes)
        where: list[dict[Any, int] | None] = []
        for a in payload["axes"]:
            if a["type"] in _CATEGORY:
                keys = next(known)
                where.append({k: j for j, k in enumerate(keys)} if a.get("growth") else None)
                if a.get("growth"):
                    a["categories"] = list(keys)
        return payload, where

    def dense(self) -> bh.Histogram:
        """Return the dense histogram of ``kind`` this partial stands for; it stays chunked."""
        payload, where = self._grown()
        out = zero_of(json.dumps(payload))
        view = out.view(flow=True)
        chunked = self.chunked
        for key, chunk in chunked.items():
            index: list[Any] = [slice(None)] * out.ndim
            for spec, part, lookup in zip(chunked.chunk_axes, key, where, strict=True):
                index[spec.index] = part if lookup is None else lookup[part]
            view[tuple(index)] = chunk
        return as_kind(out, self.kind)

    def fill(self, *args: Any, weight: Any = None, sample: Any = None) -> Partial:
        """Fill as a dense fill would, one ChunkedHist fill per category key in first-seen order.

        Values the grouping cannot reproduce bh on, or a ``sample=``, fill :meth:`dense` instead,
        which is returned, and raise as bh does.
        """
        payload = json.loads(self.spec)
        groups = None if sample is not None else _groups(payload, args, weight)
        if groups is None:
            out = self.dense()
            out.fill(*args, weight=weight, sample=sample)
            return out
        if weight is not None and payload["storage"] != "Weight":
            self._unknown()
        for key, dense, w in groups:
            self.chunked.fill(**dense, **key, weight=w)
        return self

    def iadd(self, other: Any) -> bool:
        """Do ``self += other`` in place when both hold one spec; the caller owns ``self``."""
        if not (isinstance(other, Chunked) and self.spec == other.spec):
            return False
        chunked = self.chunked
        chunked += other.chunked
        self._unknown()
        return True

    def merged(self, other: Any) -> Chunked | None:
        """Return ``self + other`` as a new partial when both hold one spec; None otherwise."""
        if not (isinstance(other, Chunked) and self.spec == other.spec):
            return None
        out = Chunked(self.spec, self.chunked + other.chunked, self.kind)
        out._unknown()
        return out


def dense(h: Any) -> Any:
    """Return ``h`` with a chunked partial made dense; anything else is returned as it is."""
    return h.dense() if isinstance(h, Chunked) else h


def _codes(a: dict[str, Any], value: Any) -> tuple[Any, list[Any]] | None:
    """Return per-event codes into the key parts they stand for.

    None when bh would refuse the value.
    """
    arr = np.asarray(value)
    labels = a["type"] == "StrCategory"
    if arr.ndim > 1 or arr.dtype.kind not in _CATEGORY[a["type"]] + ("O" if labels else ""):
        return None
    if arr.dtype.kind == "O" and not all(isinstance(u, str) for u in arr.ravel()):
        return None
    if arr.ndim == 0:
        uniq, codes = [arr.item()], np.intp(0)
    else:
        found, codes = np.unique(arr, return_inverse=True)
        uniq = found.tolist()
    if a.get("growth"):
        return codes, uniq
    # coded by bin, so every undeclared label shares the overflow group and its event order
    position = {c: j for j, c in enumerate(a["categories"])}
    bins = np.array([position.get(u, len(position)) for u in uniq], dtype=np.intp)
    return bins[codes], list(range(len(position) + 1))


def _groups(payload: dict[str, Any], args: Any, weight: Any) -> list[Any] | None:
    """Return ``(key kwargs, dense kwargs, weight)`` per category key, in first-seen order.

    None when the values are not ones this grouping reproduces bh on.
    """
    if len(args) != len(payload["axes"]):
        return None
    cat = [i for i, a in enumerate(payload["axes"]) if a["type"] in _CATEGORY]
    coded = []
    for i in cat:
        c = _codes(payload["axes"][i], args[i])
        if c is None:
            return None
        coded.append(c)
    columns = {f"a{i}": np.asarray(v) for i, v in enumerate(args) if i not in cat}
    lengths = {
        len(v)
        for v in (*columns.values(), np.asarray(weight), *(np.asarray(args[i]) for i in cat))
        if v.ndim
    }
    if len(lengths) > 1:
        return None
    n = lengths.pop() if lengths else 1
    if n == 0:
        return []
    names = [f"a{i}" for i in cat]
    if all(np.ndim(codes) == 0 for codes, _ in coded):  # one key: the fill is the dense one
        key = {name: parts[int(codes)] for name, (codes, parts) in zip(names, coded, strict=True)}
        return [(key, columns, weight)]
    dims = [len(parts) for _, parts in coded]
    combined = np.ravel_multi_index([np.broadcast_to(codes, n) for codes, _ in coded], dims)
    found, first, inverse = np.unique(combined, return_index=True, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    columns = {k: np.broadcast_to(v, n) for k, v in columns.items()}
    w = None if weight is None else np.broadcast_to(weight, n)
    out = []
    for g in np.argsort(first):
        parts = np.unravel_index(found[g], dims)
        key = {name: p[j] for name, (_, p), j in zip(names, coded, parts, strict=True)}
        sel = members[g]
        out.append((key, {k: v[sel] for k, v in columns.items()}, None if w is None else w[sel]))
    return out


def empty(spec: str, kind: Kind = BH) -> Partial:
    """Return the empty partial of ``spec`` as ``kind``, chunked where :func:`chunkable`."""
    return Chunked(spec, None, kind) if chunkable(spec) else as_kind(zero_of(spec), kind)
