"""Partials stay chunked through every task and merge; the plan's finalize, on the head, hands
back the dense histogram of the class instantiated.
"""

from __future__ import annotations

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
from graphed.core.execution import LocalResources, SequentialRunner
from graphed_exec_local import ProcessPoolExecutor

import graphed_histogram as gh
from graphed_histogram._chunked import Chunked

RNG = np.random.default_rng(67)
N = 600
EVENTS = ak.zip(
    {"x": RNG.uniform(0.0, 1.0, N), "w": RNG.normal(1.0, 0.3, N), "c": RNG.integers(0, 3, N)}
)
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
def test_only_the_heads_finalize_densifies_once_per_slot(runner: str) -> None:
    take()
    RUNNERS[runner]().run(gh.plan(_census_hists(True), steps_per_file=4))
    control = take()
    # the counter reaches workers: the dense slot adds each chunked fill dense where it sums
    assert control and (runner == "sequential" or set(control) - {os.getpid()})
    value = RUNNERS[runner]().run(gh.plan(_census_hists(False), steps_per_file=4)).value
    # sibling: 2 labels, axis: 1, hist: 1
    assert take() == [os.getpid()] * 4
    out = gh.unpack(pickle.loads(pickle.dumps(value)))
    assert take() == []
    assert all(type(h) in (bh.Histogram, hist.Hist) for h in _leaves(out))
    assert [graphed.labels(v) for v in out.values()] == [("nominal", "sf_up")] * 2 + [("nominal",)]
    axis: Any = out["axis"]
    assert _bytes(graphed.universe(axis, "sf_up")) == _bytes(axis[{2: 1}])
    assert _bytes(graphed.nominal(out["sibling"])) == _bytes(out["sibling"]["nominal"])


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
    """Slot -> (the lazy histogram, its eager twin)."""
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
def test_the_run_value_is_the_instantiated_class(runner: str) -> None:
    typed = _typed(_events())
    plan = gh.plan({k: v[0] for k, v in typed.items()}, steps_per_file=3)
    partial = plan.process(plan.tasks[0].partition, LocalResources())
    assert sorted(k for k, v in partial.items() if type(v) is Chunked) == sorted(
        k for k in typed if k.endswith("Weight-chunked")
    )
    raw = RUNNERS[runner]().run(plan).value
    assert all(type(raw[k]) is type(eager) for k, (_, eager) in typed.items())
    value = gh.unpack(raw)
    for key, (_, eager) in typed.items():
        got: Any = value[key]
        assert type(got) is type(eager), key
        # three partials, one fill
        np.testing.assert_allclose(got.values(), eager.values(), rtol=1e-13)
        twin = value[key.replace("chunked", "dense")]
        assert (
            got.axes == eager.axes and _public(got) == _public(eager) and vars(got) == vars(twin)
        ), key
        assert _bytes(got) == _bytes(twin)
        if eager.variances() is not None:
            np.testing.assert_allclose(got.variances(), eager.variances(), rtol=1e-13)


def test_a_partial_has_no_histogram_surface() -> None:
    spec = gh.spec_of(bh.Histogram(bh.axis.Regular(4, 0, 1), bh.axis.IntCategory([0, 1])))
    h = gh.FillEvaluator(spec, 2, False, False, partial=True)(X, C)
    take()
    assert type(h) is Chunked and not isinstance(h, bh.Histogram)
    for name in ("values", "axes", "view", "__array__", "_hist", "__dask_graph__"):
        assert not hasattr(h, name), name
    probe: Any = h
    with pytest.raises(TypeError):
        probe[{1: 0}]
    with pytest.raises(AttributeError):
        probe.metadata = "m"
    assert take() == [] and type(h) is Chunked


def test_materialize_on_a_fill_node_is_dense() -> None:
    ev = _events()
    h = gh.boost.Histogram(
        bh.axis.Regular(4, 0, 1), bh.axis.IntCategory([0, 1]), storage=bh.storage.Weight()
    ).fill(ev.x, ev.c, weight=ev.w)
    (node,) = h.fill_nodes()
    got = ev.session.materialize(node)
    eager = bh.Histogram(*h.axes, storage=bh.storage.Weight()).fill(X, C, weight=W)
    assert type(got) is bh.Histogram and _bytes(got) == _bytes(eager)
