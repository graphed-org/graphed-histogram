"""Chunked partials: a fill's histogram held as a ``hist.chunked.ChunkedHist``.

A ChunkedHist stores one dense slice of the non-category axes per filled category key, so a partial
and every merge of partials cost what was filled rather than the whole category product.
:class:`Chunked` is that partial wearing the ``boost_histogram.Histogram`` type: fills and
:func:`~graphed_histogram.add_histograms` keep it chunked, and anything else that reads it first
turns it, in place, into the plain dense histogram the spec describes, bin for bin what a dense fill
gives. ChunkedHist takes one scalar key per fill, so :meth:`Chunked.fill` groups a fill's events by
key, stably, which keeps every bin's sum in event order.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import boost_histogram as bh
import hist
import numpy as np
from hist.chunked import ChunkedHist

from ._spec import _GROWTH_SPEC_VERSION, _STORAGES, SPEC_VERSION, _make_axis, zero_of

_CATEGORY = {"StrCategory": "U", "IntCategory": "iu"}  # axis type -> the dtype kinds bh fills it with
_LOCK = threading.Lock()


def chunkable(spec: str) -> bool:
    """Whether a ChunkedHist holds this spec's histogram with something to gain: a category axis and
    a non-category one, no growing non-category axis, and not a Mean storage."""
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
    """Axis ``i`` as the ChunkedHist holds it. A declared category axis is keyed by bin INDEX, its
    overflow one more, so an undeclared label lands where a dense fill puts it."""
    name = f"a{i}"
    if a["type"] not in _CATEGORY:
        return _make_axis({**a, "metadata": {"name": name}})
    if not a.get("growth"):
        return hist.axis.IntCategory([], growth=True, name=name)
    return getattr(hist.axis, a["type"])(a["categories"], growth=True, name=name)


class Chunked(bh.Histogram):
    """A partial held as a ChunkedHist, typed as the dense histogram it stands for (see module doc)."""

    __slots__ = ()

    def __init__(self, spec: str, chunked: ChunkedHist | None = None) -> None:
        payload = json.loads(spec)
        if chunked is None:
            axes = [_internal_axis(i, a) for i, a in enumerate(payload["axes"])]
            chunked = ChunkedHist(*axes, storage=_STORAGES[payload["storage"]]())
        self.__dict__.update(_gh_spec=spec, _gh_chunked=chunked)

    def __reduce__(self) -> tuple[Any, ...]:
        return (Chunked, (self.__dict__["_gh_spec"], self.__dict__["_gh_chunked"]))

    @property
    def chunked(self) -> ChunkedHist:
        """The ChunkedHist this partial is held in."""
        return self.__dict__["_gh_chunked"]

    def _dense(self) -> bh.Histogram:
        payload, chunked = json.loads(self.__dict__["_gh_spec"]), self.chunked
        known = iter(c.known_keys for c in chunked.chunk_axes)
        where = []  # per category axis, label -> index (None: keyed by index)
        for a in payload["axes"]:
            if a["type"] in _CATEGORY:
                keys = next(known)
                where.append({k: j for j, k in enumerate(keys)} if a.get("growth") else None)
                if a.get("growth"):
                    a["categories"] = list(keys)
        out = zero_of(json.dumps(payload))
        view = out.view(flow=True)
        for key, chunk in chunked.items():
            index: list[Any] = [slice(None)] * out.ndim
            for spec, part, lookup in zip(chunked.chunk_axes, key, where, strict=True):
                index[spec.index] = part if lookup is None else lookup[part]
            view[tuple(index)] = chunk
        return out

    def _materialize(self) -> None:
        """Become the dense histogram: its slots and ``__dict__``, and then its class."""
        with _LOCK:
            if type(self) is not Chunked:
                return
            dense = self._dense()
            bh.Histogram._hist.__set__(self, dense._hist)
            bh.Histogram.axes.__set__(self, dense.axes)
            state = self.__dict__
            state.update(vars(dense))  # before the swap: a reader that sees bh.Histogram finds its state
            object.__setattr__(self, "__class__", bh.Histogram)  # past our __setattr__, which would re-lock
            del state["_gh_spec"], state["_gh_chunked"]

    def __setattr__(self, name: str, value: Any) -> None:
        # only spec and ChunkedHist live on a Chunked, so a write lands on the dense histogram
        Chunked._materialize(self)  # by class: another thread may have swapped ours
        object.__setattr__(self, name, value)

    @property
    def _hist(self) -> Any:
        Chunked._materialize(self)
        return bh.Histogram._hist.__get__(self)

    @property
    def axes(self) -> Any:
        Chunked._materialize(self)
        return bh.Histogram.axes.__get__(self)

    def __getattr__(self, name: str) -> Any:
        # protocol probes (``__dask_graph__``, ...) must not densify a partial in transit
        if name.startswith(("__", "_gh_")):
            raise AttributeError(name)
        Chunked._materialize(self)
        return getattr(self, name)

    def fill(
        self,
        *args: Any,
        weight: Any = None,
        sample: Any = None,
        threads: int | None = None,
    ) -> bh.Histogram:
        """Fill as a dense fill would, one ChunkedHist fill per category key in first-seen order.
        Values bh refuses, or a ``sample=``, take bh's own dense fill, which raises as it does."""
        groups = None if sample is not None else _groups(json.loads(self.__dict__["_gh_spec"]), args, weight)
        if groups is None:
            return bh.Histogram.fill(self, *args, weight=weight, sample=sample, threads=threads)
        for key, dense, w in groups:
            self.chunked.fill(**dense, **key, weight=w)
        return self


def _codes(a: dict[str, Any], value: Any) -> tuple[Any, list[Any]] | None:
    """Per-event codes into the key parts they stand for, or None when bh would refuse the value."""
    arr = np.asarray(value)
    if arr.ndim > 1 or arr.dtype.kind not in _CATEGORY[a["type"]]:
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
    """``(key kwargs, dense kwargs, weight)`` per category key, in first-seen order; None when the
    values are not ones this grouping reproduces bh on."""
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
        len(v) for v in (*columns.values(), np.asarray(weight), *(np.asarray(args[i]) for i in cat)) if v.ndim
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


def empty(spec: str) -> bh.Histogram:
    """The empty partial of ``spec``: chunked where :func:`chunkable`, else the dense zero."""
    return Chunked(spec) if chunkable(spec) else zero_of(spec)
