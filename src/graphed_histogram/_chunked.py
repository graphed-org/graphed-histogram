"""Chunked partials: a fill's histogram held as a ``hist.chunked.ChunkedHist``.

A ChunkedHist stores one dense slice of the non-category axes per filled category key, so a partial
and every merge of partials cost what was filled rather than the whole category product.
:class:`Chunked` is that partial wearing the type the user instantiated (``bh.Histogram``, or a
per-class surrogate subclass for ``hist.Hist`` and kin): fills and
:func:`~graphed_histogram.add_histograms` keep it chunked, and the first read turns it, in place,
into that class's dense histogram, bin for bin what a dense fill gives. ChunkedHist takes one scalar key per fill, so :meth:`Chunked.fill` groups a fill's events by
key, stably, which keeps every bin's sum in event order.
"""

from __future__ import annotations

import html
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


#: what a slot's result is: the in-memory class the user instantiated, and that histogram's public
#: attributes (``name``, ``label``, ``metadata``)
Kind = tuple[type, tuple[tuple[str, Any], ...]]
BH: Kind = (bh.Histogram, ())


def as_kind(h: bh.Histogram, kind: Kind) -> bh.Histogram:
    """``h`` as ``kind``'s class with its attributes; hist's own constructor shares ``h``'s storage."""
    cls, attrs = kind
    out = h if type(h) is cls else cls(h)
    out.__dict__.update(attrs)
    return out


_SURROGATES: dict[type, type] = {}


def _surrogate(cls: type) -> type[Chunked]:
    """The Chunked subclass standing for ``cls``. ``cls`` is its first base, so the ``__class__`` swap
    finds the layout ``cls`` has (hist adds a weakref slot); Chunked's own members are copied over
    ``cls``'s, which would otherwise come first."""
    if cls is bh.Histogram:
        return Chunked
    own = {k: v for k, v in vars(Chunked).items() if k not in ("__dict__", "__module__", "__doc__")}
    with _LOCK:
        return _SURROGATES.setdefault(cls, type(cls.__name__, (cls, Chunked), own))


def _target(cls: type) -> type:
    return bh.Histogram if cls is Chunked else cls.__bases__[0]


def _restore(cls: type, spec: str, chunked: ChunkedHist, attrs: tuple[tuple[str, Any], ...]) -> Chunked:
    return _surrogate(cls)(spec, chunked, attrs)


class Chunked(bh.Histogram):
    """A partial held as a ChunkedHist, typed as the histogram it stands for (see module doc). It
    holds its spec, chunks and result attributes; reading it, writing an attribute or pickling its
    state densifies it, while ``axes``, a category-axis selection and its repr read the chunks."""

    __slots__ = ()

    def __init__(
        self, spec: str, chunked: ChunkedHist | None = None, attrs: tuple[tuple[str, Any], ...] = ()
    ) -> None:
        payload = json.loads(spec)
        if chunked is None:
            axes = [_internal_axis(i, a) for i, a in enumerate(payload["axes"])]
            chunked = ChunkedHist(*axes, storage=_STORAGES[payload["storage"]]())
        state = self.__dict__  # written directly: __setattr__ densifies
        state.update(_gh_spec=spec, _gh_chunked=chunked, _gh_attrs=tuple(attrs), _gh_lock=threading.Lock())

    def __reduce__(self) -> tuple[Any, ...]:
        state = self.__dict__
        return (_restore, (_target(type(self)), state["_gh_spec"], state["_gh_chunked"], state["_gh_attrs"]))

    def __getstate__(self) -> Any:
        Chunked._materialize(self)
        return self.__getstate__()

    @property
    def chunked(self) -> ChunkedHist:
        """The ChunkedHist this partial is held in."""
        return self.__dict__["_gh_chunked"]

    def _like(self, spec: str, chunked: ChunkedHist | None = None, summed: bool = False) -> Chunked:
        out = type(self)(spec, chunked, self.__dict__["_gh_attrs"])
        if summed:
            out._unknown()
        return out

    def _unknown(self) -> None:
        """As bh does on a sum, or a weighted fill of a storage without variances: ``variances()`` is None."""
        attrs = [a for a in self.__dict__["_gh_attrs"] if a[0] != "_variance_known"]
        self.__dict__["_gh_attrs"] = (*attrs, ("_variance_known", False))

    def _grown(self) -> tuple[dict[str, Any], list[dict[Any, int] | None]]:
        """The spec with its growth categories as filled, and per category axis its label -> index
        map (None: keyed by index)."""
        payload = json.loads(self.__dict__["_gh_spec"])
        known = iter(c.known_keys for c in self.chunked.chunk_axes)
        where: list[dict[Any, int] | None] = []
        for a in payload["axes"]:
            if a["type"] in _CATEGORY:
                keys = next(known)
                where.append({k: j for j, k in enumerate(keys)} if a.get("growth") else None)
                if a.get("growth"):
                    a["categories"] = list(keys)
        return payload, where

    def _dense(self) -> bh.Histogram:
        payload, where = self._grown()
        out = zero_of(json.dumps(payload))
        view = out.view(flow=True)
        chunked = self.chunked
        for key, chunk in chunked.items():
            index: list[Any] = [slice(None)] * out.ndim
            for spec, part, lookup in zip(chunked.chunk_axes, key, where, strict=True):
                index[spec.index] = part if lookup is None else lookup[part]
            view[tuple(index)] = chunk
        return as_kind(out, (_target(type(self)), self.__dict__["_gh_attrs"]))

    def _materialize(self) -> None:
        """Become the dense histogram: its slots and ``__dict__``, and then its class."""
        lock = self.__dict__.get("_gh_lock")
        if lock is None:  # densified, and its chunked state gone
            return
        with lock:
            if not isinstance(self, Chunked):
                return
            dense = self._dense()
            for read, axis in zip(self.__dict__.get("_gh_axes", ()), dense.axes, strict=False):
                axis.__dict__.update(read.__dict__)  # an edit made through `axes` before the read
            bh.Histogram._hist.__set__(self, dense._hist)
            bh.Histogram.axes.__set__(self, dense.axes)
            state = self.__dict__
            state.update(vars(dense))  # before the swap: a reader that sees the dense class finds its state
            object.__setattr__(self, "__class__", type(dense))  # past our __setattr__, which would re-lock
            for key in [k for k in state if k.startswith("_gh_")]:
                del state[key]

    def __setattr__(self, name: str, value: Any) -> None:
        # only spec, chunks and attributes live on a Chunked, so a write lands on the dense histogram
        Chunked._materialize(self)  # by class: another thread may have swapped ours
        object.__setattr__(self, name, value)

    @property
    def _hist(self) -> Any:
        Chunked._materialize(self)
        return bh.Histogram._hist.__get__(self)

    @property
    def axes(self) -> Any:
        """The axes the dense histogram will have, read from the spec and chunks."""
        read = self.__dict__.get("_gh_axes")
        if read is None:
            ns = hist if issubclass(_target(type(self)), hist.BaseHist) else bh
            tuple_ = hist.axis.NamedAxesTuple if ns is hist else bh.axis.AxesTuple
            read = tuple_(_make_axis(a, ns.axis) for a in self._grown()[0]["axes"])
            self.__dict__["_gh_axes"] = read
        return read

    def __getattr__(self, name: str) -> Any:
        # protocol probes (``__dask_graph__``, ...) must not densify a partial in transit
        if name.startswith(("__", "_gh_")):
            raise AttributeError(name)
        Chunked._materialize(self)
        return getattr(self, name)

    def __getitem__(self, index: Any) -> Any:
        cut = self._cut(index)
        if cut is not None:
            return cut
        Chunked._materialize(self)
        return self[index]

    def _cut(self, index: Any) -> Chunked | None:
        """``self[{axis: i}]`` for an integer position ``i`` on a category axis (how
        ``graphed.universe`` reads a variation axis) as a partial without that axis."""
        if not (isinstance(index, dict) and len(index) == 1):
            return None
        ((j, i),) = index.items()
        payload, _ = self._grown()
        if not (isinstance(j, int) and isinstance(i, int | np.integer) and 0 <= j < len(payload["axes"])):
            return None
        cut = payload["axes"][j]
        n = len(cut.get("categories", ()))
        if cut["type"] not in _CATEGORY or not -n <= i < n:
            return None
        part = cut["categories"][i] if cut.get("growth") else int(i) % n
        c = [spec.index for spec in self.chunked.chunk_axes].index(j)
        rest = [a for k, a in enumerate(payload["axes"]) if k != j]
        version = _GROWTH_SPEC_VERSION if any(a.get("growth") for a in rest) else SPEC_VERSION
        out = self._like(json.dumps({"version": version, "storage": payload["storage"], "axes": rest}))
        for key, chunk in self.chunked.items():
            if key[c] == part:
                out.chunked.add_dense_view(key[:c] + key[c + 1 :], chunk)
        return out

    def __repr__(self) -> str:
        axes = ", ".join(repr(a) for a in self.axes)
        storage = self.chunked.storage_type()
        return (
            f"{_target(type(self)).__name__}({axes}, storage={storage!r}) # chunked: {len(self.chunked)} "
            "filled slices; reading it (e.g. .values()) makes it dense"
        )

    __str__ = __repr__

    def _repr_html_(self) -> str:
        return f"<pre>{html.escape(repr(self))}</pre>"

    def fill(
        self,
        *args: Any,
        weight: Any = None,
        sample: Any = None,
        threads: int | None = None,
    ) -> bh.Histogram:
        """Fill as a dense fill would, one ChunkedHist fill per category key in first-seen order.
        Values bh refuses, or a ``sample=``, take bh's own dense fill, which raises as it does."""
        payload = json.loads(self.__dict__["_gh_spec"])
        groups = None if sample is not None else _groups(payload, args, weight)
        if groups is None:
            return bh.Histogram.fill(self, *args, weight=weight, sample=sample, threads=threads)
        self.__dict__.pop("_gh_axes", None)
        if weight is not None and payload["storage"] != "Weight":
            self._unknown()
        for key, dense, w in groups:
            self.chunked.fill(**dense, **key, weight=w)
        return self

    def iadd(self, other: Any) -> bool:
        """``self += other`` in place when both hold one spec; the caller owns ``self``."""
        if not isinstance(other, Chunked) or other.__dict__["_gh_spec"] != self.__dict__["_gh_spec"]:
            return False
        chunked = self.chunked
        chunked += other.chunked
        self.__dict__.pop("_gh_axes", None)
        self._unknown()
        return True


def _codes(a: dict[str, Any], value: Any) -> tuple[Any, list[Any]] | None:
    """Per-event codes into the key parts they stand for, or None when bh would refuse the value."""
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


def empty(spec: str, kind: Kind = BH) -> bh.Histogram:
    """The empty partial of ``spec`` as ``kind``: chunked where :func:`chunkable`, else dense."""
    return _surrogate(kind[0])(spec, None, kind[1]) if chunkable(spec) else as_kind(zero_of(spec), kind)
