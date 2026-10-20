"""Nothing densifies a chunked partial before the user reads it, and the read hands back the type instantiated."""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Callable
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import hist
import hist.graphed as hg
import numpy as np
import pytest
from chunked_census import Events, take
from graphed.awkward import AwkwardBackend, AwkwardForm
from graphed.core.execution import SequentialRunner
from graphed_exec_local import ProcessPoolExecutor

import graphed_histogram as gh
from graphed_histogram._chunked import Chunked

RNG = np.random.default_rng(67)
N = 600
EVENTS = ak.zip({"x": RNG.uniform(0.0, 1.0, N), "w": RNG.normal(1.0, 0.3, N), "c": RNG.integers(0, 3, N)})
X, W, C = (np.asarray(EVENTS[k]) for k in ("x", "w", "c"))
RUNNERS: dict[str, Callable[[], Any]] = {
    "sequential": SequentialRunner,
    "process": lambda: ProcessPoolExecutor(max_workers=2, comms=None),
    "peer-ipc": lambda: ProcessPoolExecutor(max_workers=3, comms="ipc"),
}


def _events() -> Any:
    session = graphed.Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    return session.source("ev", form=form, data=Events(EVENTS))


def _census_hists(control: bool) -> dict[str, Any]:
    ev = _events()
    sf = graphed.vary(ev.w, "sf", up=ev.w * 1.5)

    def weighted() -> gh.boost.Histogram:
        return gh.boost.Histogram(
            bh.axis.Regular(4, 0, 1), bh.axis.IntCategory([0, 1]), storage=bh.storage.Weight()
        )

    out = {
        "sibling": weighted().fill(ev.x, ev.c, weight=sf),
        "axis": weighted().fill(ev.x, ev.c, weight=sf, variation_axis=True),
        "hist": hg.Hist(
            hist.axis.Regular(4, 0, 1, name="x"),
            hist.axis.IntCategory([0, 1], name="c"),
            name="h",
            metadata="m",
        ).fill(x=ev.x, c=ev.c, weight=ev.w),
    }
    if control:
        out["control"] = gh.dense_backed(weighted().fill(ev.x, ev.c, weight=ev.w))
    return out


def _bytes(h: bh.Histogram) -> bytes:
    return bytes(np.ascontiguousarray(h.view(flow=True)).tobytes())


def _public(h: bh.Histogram) -> dict[str, Any]:
    return {k: v for k, v in vars(h).items() if not k.startswith("_")}


def _leaves(out: dict[str, Any]) -> list[Any]:
    return [h for v in out.values() for h in (v.values() if isinstance(v, dict) else [v])]


@pytest.mark.parametrize("runner", RUNNERS)
def test_nothing_densifies_a_partial_before_the_users_first_read(runner: str) -> None:
    take()
    RUNNERS[runner]().run(gh.plan(_census_hists(True), steps_per_file=4))
    control = take()
    assert control and (runner == "sequential" or set(control) - {os.getpid()})  # the counter reaches workers
    blob = pickle.dumps(RUNNERS[runner]().run(gh.plan(_census_hists(False), steps_per_file=4)).value)
    out = gh.unpack(pickle.loads(blob))
    assert all(isinstance(h, Chunked) for h in _leaves(out))
    for h in _leaves(out):
        graphed.labels(h)
        logging.getLogger(__name__).warning("%r %s %s", h, h, h._repr_html_())
    for v in out.values():
        graphed.nominal(v)
    axis: Any = out["axis"]
    universe = graphed.universe(axis, "sf_up")
    assert isinstance(universe, Chunked) and "chunked" in repr(universe) and ".values()" in repr(universe)
    assert take() == []
    out["sibling"]["nominal"].values()
    assert take() == [os.getpid()]
    assert _bytes(universe) == _bytes(axis[{2: 1}])


def _bh(cls: type, storage: str) -> Any:
    return cls(
        bh.axis.Regular(4, 0, 1),
        bh.axis.IntCategory([0, 1]),
        storage=getattr(bh.storage, storage)(),
        metadata="m",
    )


def _hist(cls: type, storage: str, extra: dict[str, Any]) -> Any:
    x, c = hist.axis.Regular(4, 0, 1, name="x", label="X"), hist.axis.IntCategory([0, 1], name="c")
    return cls(x, c, storage=getattr(hist.storage, storage)(), **extra)


#: lazy class -> (the class its result is, its histogram attributes)
HISTS: dict[type, tuple[type, dict[str, Any]]] = {
    hg.Hist: (hist.Hist, {"name": "hn", "label": "hl", "metadata": "m"}),
    hg.NamedHist: (hist.NamedHist, {}),
}


def _typed(ev: Any) -> dict[str, tuple[Any, Any]]:
    """slot -> (the lazy histogram, its eager twin)."""
    out = {}
    for storage in ("Weight", "Mean"):
        field = "weight" if storage == "Weight" else "sample"
        for backing in ("chunked", "dense"):
            made = {
                "bh": (
                    _bh(gh.boost.Histogram, storage).fill(ev.x, ev.c, **{field: ev.w}),
                    _bh(bh.Histogram, storage).fill(X, C, **{field: W}),
                )
            }
            for lazy, (eager, extra) in HISTS.items():
                made[lazy.__name__] = (
                    _hist(lazy, storage, extra).fill(x=ev.x, c=ev.c, **{field: ev.w}),
                    _hist(eager, storage, extra).fill(x=X, c=C, **{field: W}),
                )
            for name, (lazy_h, eager_h) in made.items():
                out[f"{name}-{storage}-{backing}"] = (
                    gh.dense_backed(lazy_h) if backing == "dense" else lazy_h,
                    eager_h,
                )
    return out


@pytest.mark.parametrize("runner", ["sequential", "process"])
def test_the_read_hands_back_the_instantiated_type(runner: str) -> None:
    typed = _typed(_events())
    value = gh.unpack(
        RUNNERS[runner]().run(gh.plan({k: v[0] for k, v in typed.items()}, steps_per_file=3)).value
    )
    lazy = [k for k, v in value.items() if type(v) is not type(typed[k][1])]
    assert sorted(lazy) == sorted(k for k in typed if k.endswith("Weight-chunked"))  # chunked until read
    for key, (_, eager) in typed.items():
        got = value[key]
        assert isinstance(got, type(eager)), key  # before the read, too
        np.testing.assert_allclose(got.values(), eager.values(), rtol=1e-13)  # three partials, one fill
        assert type(got) is type(eager), key
        twin = value[key.replace("chunked", "dense")]
        assert got.axes == eager.axes and _public(got) == _public(eager) and vars(got) == vars(twin), key
        assert _bytes(got) == _bytes(twin)
