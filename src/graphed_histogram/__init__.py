"""graphed-histogram: deferred boost-histogram/hist filling on graphed task graphs (M23).

The dask-histogram analogue with graphed's own evaluation idiom: ``.fill(...)`` records an
External node (content-addressed canonical axes/storage spec; backends know nothing about
histograms); ``plan()`` exports the R15.4 task graph an R7 executor aggregates (partition-wise
through the compiled IR, native ``+`` tree-combine); the reference ``session.materialize``
evaluates a fill eagerly.
"""

from __future__ import annotations

from collections.abc import Callable

from . import boost
from ._spec import content_hash, zero_of
from ._spec import spec_of as _spec_of_hist
from .boost import (
    FillEvaluator,
    Histogram,
    add_histograms,
    factory,
    fill_nodes_by_label,
    histogram,
    histogram2d,
    histogramdd,
    label_listing,
    plan,
    unpack,
)


def spec_of(hist: object) -> str:
    """The canonical spec string of a (deferred or eager) boost histogram."""
    import boost_histogram as bh  # noqa: PLC0415

    assert isinstance(hist, bh.Histogram)
    return _spec_of_hist(hist)


def evaluators(*histograms: Histogram) -> dict[str, Callable[..., object]]:
    """Merged content-hash -> evaluator registry for ``evaluate_ir(externals=...)``."""
    out: dict[str, Callable[..., object]] = {}
    for h in histograms:
        out.update(h.evaluators())
    return out


__all__ = [
    "FillEvaluator",
    "Histogram",
    "add_histograms",
    "boost",
    "content_hash",
    "evaluators",
    "factory",
    "fill_nodes_by_label",
    "histogram",
    "histogram2d",
    "histogramdd",
    "label_listing",
    "plan",
    "spec_of",
    "unpack",
    "zero_of",
]
