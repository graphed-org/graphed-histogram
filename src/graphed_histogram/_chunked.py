"""Chunked partials: a fill's histogram held as a ``hist.chunked.ChunkedHist``.

A ChunkedHist stores one dense slice of the non-category axes per filled category key, so a
partial and every merge of partials cost what was filled rather than the whole category product.
:class:`Chunked` is that partial wearing the type the user instantiated (``bh.Histogram``, or a
per-class surrogate subclass for ``hist.Hist`` and kin): fills and
:func:`~graphed_histogram.add_histograms` keep it chunked, and the first read turns it, in place,
into that class's dense histogram, bin for bin what a dense fill gives. ChunkedHist takes one
scalar key per fill, so :meth:`Chunked.fill` groups a fill's events by key, stably, which keeps
every bin's sum in event order.
"""

from __future__ import annotations

import html
import json
import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar

import boost_histogram as bh
import hist
import numpy as np
from hist.chunked import ChunkedHist

from ._spec import _GROWTH_SPEC_VERSION, _STORAGES, SPEC_VERSION, _make_axis, zero_of

# axis type -> the dtype kinds bh fills it with
_CATEGORY = {"StrCategory": "U", "IntCategory": "iu"}
_LOCK = threading.Lock()


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


_SURROGATES: dict[type, type] = {}


def _surrogate(cls: type) -> type[Chunked]:
    """Return the Chunked subclass standing for ``cls``.

    ``cls`` must be its first base, or CPython refuses the class change a read makes ("object
    layout differs": hist adds a weakref slot); so Chunked's own members are copied over ``cls``'s,
    which would otherwise come first.
    """
    if cls is bh.Histogram:
        return Chunked
    own = {
        k: v for k, v in vars(Chunked).items() if k not in ("__dict__", "__module__", "__doc__")
    }
    with _LOCK:
        return _SURROGATES.setdefault(
            cls, type(cls.__name__, (cls, Chunked), {**own, "_dense_class": cls})
        )


def _restore(
    cls: type, spec: str, chunked: ChunkedHist, attrs: tuple[tuple[str, Any], ...]
) -> Chunked:
    return _surrogate(cls)(spec, chunked, attrs)


@dataclass
class _Held:
    """What a chunked partial holds until its first read."""

    spec: str
    chunked: ChunkedHist
    attrs: tuple[tuple[str, Any], ...]
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: the axes read before the first read, whose edits the dense axes take over
    axes: Any = None


class Chunked(bh.Histogram):
    """A partial held as a ChunkedHist, typed as the histogram it stands for (see module doc).

    It holds its spec, chunks and result attributes; reading it, writing an attribute or pickling
    its state densifies it, while ``axes``, a category-axis selection and its repr read the chunks.
    """

    __slots__ = ()
    #: the class a read turns this partial into
    _dense_class: ClassVar[type] = bh.Histogram
    _gh_held: _Held

    def __init__(
        self,
        spec: str,
        chunked: ChunkedHist | None = None,
        attrs: tuple[tuple[str, Any], ...] = (),
    ) -> None:
        """Hold ``chunked``, or empty chunks for ``spec``, and the result's ``attrs``."""
        payload = json.loads(spec)
        if chunked is None:
            axes = [_internal_axis(i, a) for i, a in enumerate(payload["axes"])]
            chunked = ChunkedHist(*axes, storage=_STORAGES[payload["storage"]]())
        # past Chunked.__setattr__, which densifies
        object.__setattr__(self, "_gh_held", _Held(spec, chunked, tuple(attrs)))

    def __reduce__(self) -> tuple[Any, ...]:
        held = self._gh_held
        return (_restore, (self._dense_class, held.spec, held.chunked, held.attrs))

    def __getstate__(self) -> Any:
        Chunked._densify(self)
        return self.__getstate__()

    @property
    def chunked(self) -> ChunkedHist:
        """The ChunkedHist this partial is held in."""
        return self._gh_held.chunked

    def _like(
        self, spec: str, chunked: ChunkedHist | None = None, summed: bool = False
    ) -> Chunked:
        attrs = self._gh_held.attrs
        return type(self)(spec, chunked, _variances_unknown(attrs) if summed else attrs)

    def _unknown(self) -> None:
        held = self._gh_held
        held.attrs = _variances_unknown(held.attrs)

    def _grown(self) -> tuple[dict[str, Any], list[dict[Any, int] | None]]:
        """Return the spec with its growth categories as filled, and the category axes' keys.

        The keys are, per category axis, its label -> index map (None: keyed by index).
        """
        payload = json.loads(self._gh_held.spec)
        known = iter(c.known_keys for c in self.chunked.chunk_axes)
        where: list[dict[Any, int] | None] = []
        for a in payload["axes"]:
            if a["type"] in _CATEGORY:
                keys = next(known)
                where.append({k: j for j, k in enumerate(keys)} if a.get("growth") else None)
                if a.get("growth"):
                    a["categories"] = list(keys)
        return payload, where

    def _dense_copy(self) -> bh.Histogram:
        """Return the dense histogram this partial reads as, leaving the partial chunked."""
        payload, where = self._grown()
        out = zero_of(json.dumps(payload))
        view = out.view(flow=True)
        chunked = self.chunked
        for key, chunk in chunked.items():
            index: list[Any] = [slice(None)] * out.ndim
            for spec, part, lookup in zip(chunked.chunk_axes, key, where, strict=True):
                index[spec.index] = part if lookup is None else lookup[part]
            view[tuple(index)] = chunk
        return as_kind(out, (self._dense_class, self._gh_held.attrs))

    def _densify(self) -> None:
        """Turn this partial, in place, into the dense histogram it reads as.

        It changes its class, so every reference the user holds reads the dense histogram.
        """
        state = self.__dict__
        held = state.get("_gh_held")
        if held is None:  # densified already
            return
        with held.lock:
            if not isinstance(self, Chunked):
                return
            dense = self._dense_copy()
            for read, axis in zip(held.axes or (), dense.axes, strict=False):
                vars(axis).update(vars(read))  # an edit made through `axes` before the read
            # Chunked's properties shadow bh's slots, so bh's descriptors write them, and before
            # the class changes, so a reader that sees the dense class finds its state
            bh.Histogram._hist.__set__(self, dense._hist)
            bh.Histogram.axes.__set__(self, dense.axes)
            state.update(vars(dense))
            object.__setattr__(self, "__class__", type(dense))  # past Chunked.__setattr__
            del state["_gh_held"]

    def __setattr__(self, name: str, value: Any) -> None:
        # only spec, chunks and attributes live on a Chunked, so a write lands on the dense
        # histogram
        Chunked._densify(self)  # by class: another thread may have swapped ours
        object.__setattr__(self, name, value)

    @property
    def _hist(self) -> Any:
        Chunked._densify(self)
        return self._hist

    @property
    def axes(self) -> Any:
        """The axes the dense histogram will have, read from the spec and chunks."""
        held = self._gh_held
        if held.axes is None:
            ns = hist if issubclass(self._dense_class, hist.BaseHist) else bh
            tuple_ = hist.axis.NamedAxesTuple if ns is hist else bh.axis.AxesTuple
            held.axes = tuple_(_make_axis(a, ns.axis) for a in self._grown()[0]["axes"])
        return held.axes

    def __getattr__(self, name: str) -> Any:
        # protocol probes (``__dask_graph__``, ...) must not densify a partial in transit
        if name.startswith(("__", "_gh_")):
            raise AttributeError(name)
        Chunked._densify(self)
        return getattr(self, name)

    def __getitem__(self, index: Any) -> Any:
        cut = self._cut(index)
        if cut is not None:
            return cut
        Chunked._densify(self)
        return self[index]

    def _cut(self, index: Any) -> Chunked | None:
        """Return ``self[{axis: i}]`` as a partial without that axis.

        ``i`` is an integer position on a category axis, which is how ``graphed.universe`` reads a
        variation axis.
        """
        if not (isinstance(index, dict) and len(index) == 1):
            return None
        ((j, i),) = index.items()
        payload, _ = self._grown()
        if not (
            isinstance(j, int)
            and isinstance(i, int | np.integer)
            and 0 <= j < len(payload["axes"])
        ):
            return None
        cut = payload["axes"][j]
        n = len(cut.get("categories", ()))
        if cut["type"] not in _CATEGORY or not -n <= i < n:
            return None
        part = cut["categories"][i] if cut.get("growth") else int(i) % n
        c = [spec.index for spec in self.chunked.chunk_axes].index(j)
        rest = [a for k, a in enumerate(payload["axes"]) if k != j]
        version = _GROWTH_SPEC_VERSION if any(a.get("growth") for a in rest) else SPEC_VERSION
        out = self._like(
            json.dumps({"version": version, "storage": payload["storage"], "axes": rest})
        )
        for key, chunk in self.chunked.items():
            if key[c] == part:
                out.chunked.add_dense_view(key[:c] + key[c + 1 :], chunk)
        return out

    def __repr__(self) -> str:
        axes = ", ".join(repr(a) for a in self.axes)
        storage = self.chunked.storage_type()
        return (
            f"{self._dense_class.__name__}({axes}, storage={storage!r}) # chunked: "
            f"{len(self.chunked)} filled slices; reading it (e.g. .values()) makes it dense"
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

        Values bh refuses, or a ``sample=``, take bh's own dense fill, which raises as it does.
        """
        payload = json.loads(self._gh_held.spec)
        groups = None if sample is not None else _groups(payload, args, weight)
        if groups is None:
            return bh.Histogram.fill(self, *args, weight=weight, sample=sample, threads=threads)
        self._gh_held.axes = None
        if weight is not None and payload["storage"] != "Weight":
            self._unknown()
        for key, dense, w in groups:
            self.chunked.fill(**dense, **key, weight=w)
        return self

    def iadd(self, other: Any) -> bool:
        """Do ``self += other`` in place when both hold one spec; the caller owns ``self``."""
        if not (isinstance(other, Chunked) and self._holds_spec_of(other)):
            return False
        chunked = self.chunked
        chunked += other.chunked
        self._gh_held.axes = None
        self._unknown()
        return True

    def _merged(self, other: Any) -> Chunked | None:
        """Return ``self + other`` as a new partial when both hold one spec; None otherwise."""
        if not (isinstance(other, Chunked) and self._holds_spec_of(other)):
            return None
        return self._like(self._gh_held.spec, self.chunked + other.chunked, summed=True)

    def _holds_spec_of(self, other: Chunked) -> bool:
        return self._gh_held.spec == other._gh_held.spec


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


def empty(spec: str, kind: Kind = BH) -> bh.Histogram:
    """Return the empty partial of ``spec`` as ``kind``, chunked where :func:`chunkable`."""
    return (
        _surrogate(kind[0])(spec, None, kind[1])
        if chunkable(spec)
        else as_kind(zero_of(spec), kind)
    )
