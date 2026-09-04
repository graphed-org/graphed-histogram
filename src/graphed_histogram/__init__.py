"""Deferred boost-histogram/hist filling for graphed: a fill records, a runner computes.

The dask-histogram analogue, with graphed's own evaluation idiom. ``.fill(...)`` records a step
the runner performs later, identified by the content hash of a canonical axes/storage
description — so identical fills collapse to one, and nothing in graphed or in any array backend
needs an opinion about histograms. ``plan()`` exports the plan any runner aggregates, one fill
task per chunk combined by histogram addition. ``session.materialize(node)`` evaluates a single
fill on the spot, for a source that has no chunks to hand out.
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
    """Merged ``{content hash: evaluator}`` registry, for wiring these fills into a graph you
    evaluate yourself."""
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
