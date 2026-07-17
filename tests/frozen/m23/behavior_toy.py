"""An importable toy awkward behavior for the process-executor behavior witnesses.

Module-level and dependency-light on purpose: process workers resolve it by IMPORT REF
("behavior_toy:make_backend"), never by pickling the behavior dict (behavior dicts in the wild
— e.g. vector's — contain lambdas and do not pickle; this is the exact failure mode that
plagued distributed awkward analyses)."""

from __future__ import annotations

from typing import Any

import awkward as ak


class PlanarRecord(ak.Record):  # type: ignore[misc]
    @property
    def mag(self) -> Any:
        return (self["x"] ** 2 + self["y"] ** 2) ** 0.5


class PlanarArray(ak.Array):  # type: ignore[misc]
    @property
    def mag(self) -> Any:
        return (self["x"] ** 2 + self["y"] ** 2) ** 0.5


BEHAVIOR: dict[Any, Any] = {"planar": PlanarRecord, ("*", "planar"): PlanarArray}


def make_backend() -> Any:
    """Zero-arg factory a worker imports and calls (the supported behavior-forwarding path)."""
    from graphed.awkward import AwkwardBackend  # noqa: PLC0415

    return AwkwardBackend(behavior=BEHAVIOR)
