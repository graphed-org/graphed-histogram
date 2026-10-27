"""Filling on histserv servers: a histogram backend beside ``gh.boost``, sized before the run.

Four parts, four owners. **Recording** stays ``gh.boost``'s: a backed histogram records exactly the
graph its unbacked twin records. The **context** (:class:`Context`) sizes each backed slot with a
measured model, packs the slots first-fit decreasing onto servers of the sizes it offers, and puts
one :class:`~graphed.services.ServiceSpec` per server in ``plan.services`` at ``pieces.serve``. The
**executor** places, binds and closes those servers and never learns what a histogram is. The
**served process** creates the server-side histograms, ships each partition's partial of a backed
slot, and hands a receipt through the reduction tree instead of the histogram.

Importing this module, building a context, backing a histogram and planning it import neither
``histserv`` nor ``grpc``.
"""

from __future__ import annotations

from typing import Any, TypeVar

from . import boost
from ._histserv import CEILING, Context, HistservError, Receipt

H = TypeVar("H", bound=boost.Histogram)


_STORAGES = ("Double", "Int64", "Weight")


def backed(h: H, context: Context) -> H:
    """``h``, filled on ``context``'s servers from its next plan on; its graph is unchanged.

    A growth axis is refused (a server is sized before the run), as is a storage other than
    ``Double``, ``Int64`` or ``Weight``.
    """
    for axis in h.axes:
        if axis.traits.growth:
            raise TypeError(
                f"histserv sizes its servers before the run, so it cannot hold a growth axis: "
                f"{type(axis).__name__}(growth=True)"
            )
    storage = type(h.storage_type()).__name__
    if storage not in _STORAGES:
        raise TypeError(f"histserv holds {', '.join(_STORAGES)} storage, not {storage}")
    h.hold_partials(histserv=context)
    return h


class Histogram(boost.Histogram):
    """A ``gh.boost.Histogram`` filled on ``context``'s histserv servers (:func:`backed`)."""

    def __init__(
        self, *axes: Any, storage: Any = None, metadata: Any = None, context: Context
    ) -> None:
        """Build an empty deferred histogram backed by ``context``."""
        super().__init__(*axes, storage=storage, metadata=metadata)
        backed(self, context)


__all__ = ["CEILING", "Context", "Histogram", "HistservError", "Receipt", "backed"]
