"""The chunked default and the dense opt-in give the same histograms, byte for byte, on every
route.
"""

from __future__ import annotations

import os
import pickle
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
import pytest
from chunked_census import take
from graphed.awkward import AwkwardBackend, AwkwardForm
from graphed.core import Partition
from graphed.core.execution import LocalResources, SequentialRunner, WorkerResources
from graphed_exec_local import ProcessPoolExecutor

import graphed_histogram as gh
from graphed_histogram._chunked import Chunked, dense

RNG = np.random.default_rng(66)
N = 3000
EVENTS = ak.zip(
    {"x": RNG.uniform(0.0, 1.0, N), "w": RNG.normal(1.0, 0.3, N), "c": RNG.integers(0, 6, N)}
)
LABELS = RNG.choice(["a", "b", "zz", "q"], N)


@dataclass
class Events:
    data: ak.Array

    def __call__(self) -> ak.Array:
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind("chunked://ev", "", s, steps_per_file) for s in range(steps_per_file)
        )

    def read_partition(
        self, partition: Partition, columns: Any, resources: WorkerResources
    ) -> ak.Array:
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


def _hists(dense: bool) -> dict[str, gh.boost.Histogram]:
    session = graphed.Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    ev = session.source("ev", form=form, data=Events(EVENTS))
    sf = graphed.vary(ev.w, "sf", up=ev.w * 1.5, down=ev.w * 0.5)
    reg = bh.axis.Regular(8, 0.0, 1.0)
    out = {
        # declared categories: c in 3..5 and "zz" are undeclared and land in overflow
        "weight": gh.boost.Histogram(
            reg,
            bh.axis.IntCategory([0, 1, 2]),
            bh.axis.StrCategory(["d1", "d2"]),
            storage=bh.storage.Weight(),
        ),
        "int64": gh.boost.Histogram(
            bh.axis.Integer(0, 6), bh.axis.IntCategory([], growth=True), storage=bh.storage.Int64()
        ),
        "double": gh.boost.Histogram(
            reg, bh.axis.StrCategory(["sr", "cr"]), storage=bh.storage.Double()
        ),
        "growth": gh.boost.Histogram(
            reg, bh.axis.StrCategory([], growth=True), storage=bh.storage.Weight()
        ),
        "axis": gh.boost.Histogram(reg, bh.axis.IntCategory([0, 1]), storage=bh.storage.Weight()),
        "mean": gh.boost.Histogram(reg, bh.axis.IntCategory([0, 1]), storage=bh.storage.Mean()),
    }
    out["weight"].fill(ev.x, ev.c, "d1", weight=ev.w).fill(ev.x * 0.5, ev.c, "zz", weight=sf)
    out["int64"].fill(ev.c, ev.c, unweighted=True).fill(3, ev.c, unweighted=True)
    out["double"].fill(ev.x, "sr", weight=2.0).fill(ev.x * 0.25, "cr", weight=ev.w)  # GW30 scalars
    out["growth"].fill(ev.x, "d2", weight=sf).fill(ev.x, "d1", weight=ev.w)
    out["axis"].fill(ev.x, ev.c, weight=sf, variation_axis=True)
    out["mean"].fill(ev.x, ev.c, sample=ev.w)
    if dense:
        for h in out.values():
            gh.dense_backed(h)
    return out


RUNNERS: dict[str, Callable[[], Any]] = {
    "sequential": SequentialRunner,
    "process": lambda: ProcessPoolExecutor(max_workers=2, comms=None),
    "peer-ipc": lambda: ProcessPoolExecutor(max_workers=3, comms="ipc"),
}


def _bytes(h: bh.Histogram) -> tuple[bytes, list[Any], bool]:
    return (
        np.ascontiguousarray(h.view(flow=True)).tobytes(),
        [list(ax) for ax in h.axes],
        h._variance_known,
    )


def _flat(value: Mapping[Any, Any]) -> dict[Any, bh.Histogram]:
    out = gh.unpack(value)
    return {
        (k, label): h
        for k, v in out.items()
        for label, h in (v.items() if isinstance(v, dict) else [(None, v)])
    }


@pytest.mark.parametrize("runner", RUNNERS)
def test_chunked_and_dense_backings_agree_byte_for_byte(runner: str) -> None:
    take()
    chunked = _flat(RUNNERS[runner]().run(gh.plan(_hists(False), steps_per_file=4)).value)
    # the witness: chunked merges ran end to end, and only the head's finalize made each dense
    assert take() == [os.getpid()] * sum(k[0] != "mean" for k in chunked)
    dense = _flat(RUNNERS[runner]().run(gh.plan(_hists(True), steps_per_file=4)).value)
    assert chunked.keys() == dense.keys() and len(chunked) == 10
    assert all(type(h) is bh.Histogram for h in (*chunked.values(), *dense.values()))
    for key in chunked:
        assert _bytes(chunked[key]) == _bytes(dense[key]), key
    # both overflows filled
    assert chunked[("weight", "nominal")].view(flow=True)["value"][:, 3, 2].sum() > 0


def test_a_partial_ships_chunked_and_densifies_into_first_seen_order() -> None:
    plan = gh.plan(_hists(False), steps_per_file=4)
    partial = plan.process(plan.tasks[0].partition, LocalResources())
    shipped = pickle.loads(pickle.dumps(partial[("growth", "nominal")]))
    assert type(shipped) is Chunked and len(shipped.chunked) == 2
    assert type(plan.empty()[("growth", "nominal")]) is Chunked
    # first-seen order, as a growing dense axis has
    assert list(shipped.dense().axes[1]) == ["d2", "d1"]


def test_a_sparse_partial_pickles_its_filled_slices_only() -> None:
    session = graphed.Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    ev = session.source("ev", form=form, data=Events(EVENTS))
    h = gh.boost.Histogram(
        bh.axis.Regular(100, 0.0, 1.0),
        bh.axis.StrCategory([f"d{i}" for i in range(50)]),
        storage=bh.storage.Weight(),
    ).fill(ev.x, "d7", weight=ev.w)
    plan = gh.plan({"h": h}, steps_per_file=4)
    partial = plan.process(plan.tasks[0].partition, LocalResources())["h"]
    size = len(pickle.dumps(partial))
    assert type(partial) is Chunked and size * 10 < len(pickle.dumps(dense(partial)))


def test_a_single_histogram_plan_and_a_zero_partition_plan_run_dense() -> None:
    h = gh.boost.Histogram(
        bh.axis.Regular(4, 0.0, 1.0), bh.axis.IntCategory([0, 1]), storage=bh.storage.Weight()
    )
    session = graphed.Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    ev = session.source("ev", form=form, data=Events(EVENTS))
    h.fill(ev.x, ev.c, weight=ev.w)
    value = SequentialRunner().run(h.plan(steps_per_file=3)).value
    twin = SequentialRunner().run(gh.dense_backed(h).plan(steps_per_file=3)).value
    assert type(value) is type(twin) is bh.Histogram and _bytes(value) == _bytes(twin)
    zero = SequentialRunner().run(gh.plan({"h": h}, partitions=[])).value["h"]
    assert type(zero) is bh.Histogram and zero == gh.zero_of(gh.spec_of(h))


X, W, C = np.asarray(EVENTS.x), np.asarray(EVENTS.w), np.asarray(EVENTS.c)
S, IC, ST = bh.axis.StrCategory, bh.axis.IntCategory, bh.storage
R = bh.axis.Regular(10, 0.0, 1.0)
#: case -> (axes, storage, fill columns, weighted); a Mean storage also takes ``sample=X``
CHUNKED: dict[str, tuple[list[Any], Any, list[Any], bool]] = {
    "declared str, undeclared array labels": ([R, S(["a", "b"])], ST.Weight(), [X, LABELS], True),
    "growth str with initial categories": (
        [R, S(["q"], growth=True)],
        ST.Double(),
        [X, LABELS],
        True,
    ),
    "declared int + growth str, scalar x": (
        [R, IC([1, 2, 3]), S([], growth=True)],
        ST.Weight(),
        [0.5, C, LABELS],
        True,
    ),
    "undeclared scalar label": ([R, S(["a"])], ST.Int64(), [X, "nope"], False),
    "Unlimited storage": ([R, IC([0, 1, 2])], ST.Unlimited(), [X, C], False),
    "AtomicInt64 storage": ([R, IC([0, 1])], ST.AtomicInt64(), [X, C], False),
    "no event": ([R, S([], growth=True)], ST.Double(), [X[:0], "a"], False),
    "object-dtype labels": ([R, S(["a", "b"])], ST.Double(), [X, LABELS.astype(object)], False),
}
#: the dense fallbacks: what ChunkedHist cannot hold, holds for no gain, or values bh refuses
DENSE: dict[str, tuple[list[Any], Any, list[Any], bool]] = {
    "Mean storage": ([R, S(["a", "b"])], ST.Mean(), [X, LABELS], False),
    "WeightedMean storage": ([R, S(["a", "b"])], ST.WeightedMean(), [X, LABELS], True),
    "growing Regular": (
        [bh.axis.Regular(10, 0.0, 1.0, growth=True), S(["a"])],
        ST.Double(),
        [X, LABELS],
        False,
    ),
    "growing Integer": (
        [bh.axis.Integer(0, 3, growth=True), S(["a"])],
        ST.Double(),
        [C, LABELS],
        False,
    ),
    "no category axis": ([R], ST.Double(), [X], True),
    "only category axes": ([S(["a"]), IC([], growth=True)], ST.Double(), [LABELS, C], False),
}


def _evaluate(case: str, dense: bool) -> bh.Histogram:
    axes, storage, cols, weighted = {**CHUNKED, **DENSE}[case]
    sample = isinstance(storage, (ST.Mean, ST.WeightedMean))
    spec = gh.spec_of(bh.Histogram(*axes, storage=storage))
    if dense:  # the oracle: one eager bh fill
        return gh.zero_of(spec).fill(
            *cols, weight=W if weighted else None, sample=X if sample else None
        )
    ev = gh.FillEvaluator(spec, len(axes), weighted, sample, partial=True)
    return ev(*cols, *([W] if weighted else []), *([X] if sample else []))


@pytest.mark.parametrize("case", [*CHUNKED, *DENSE])
def test_a_fill_and_a_merge_are_the_dense_ones_byte_for_byte(case: str) -> None:
    chunked = gh.add_histograms(_evaluate(case, False), _evaluate(case, False))
    eager = gh.add_histograms(_evaluate(case, True), _evaluate(case, True))
    assert (type(chunked) is Chunked) == (case in CHUNKED)
    assert _bytes(dense(chunked)) == _bytes(eager)


@pytest.mark.parametrize(
    "cols",
    [[X, LABELS[:10]], [X, C], [X, np.array([1, "a"] * (N // 2), dtype=object)]],
    ids=["lengths differ", "int labels", "object labels, not all str"],
)
def test_values_bh_refuses_raise_as_a_dense_fill_does(cols: list[Any]) -> None:
    spec = gh.spec_of(bh.Histogram(R, S(["a", "b"])))
    with pytest.raises(Exception) as dense:
        gh.zero_of(spec).fill(*cols)
    with pytest.raises(type(dense.value), match=re.escape(str(dense.value))):
        gh.FillEvaluator(spec, 2, False, False, partial=True)(*cols)


def test_transformed_regular_never_reaches_a_fill() -> None:
    with pytest.raises(TypeError, match="transform"):
        gh.spec_of(
            bh.Histogram(bh.axis.Regular(4, 1.0, 10.0, transform=bh.axis.transform.log), S(["a"]))
        )


def test_a_wrong_arity_fill_raises_as_a_dense_fill_does() -> None:
    spec = gh.spec_of(bh.Histogram(R, S(["a", "b"])))
    for cols in ([X], [X, LABELS, X]):
        with pytest.raises(ValueError, match="Wrong number of args"):
            Chunked(spec).fill(*cols)


def test_only_partials_of_one_spec_merge_chunked() -> None:
    spec = gh.spec_of(bh.Histogram(R, S(["a", "b"])))
    a, b = Chunked(spec), Chunked(spec)
    a.fill(X, LABELS)
    b.fill(X, LABELS)
    assert type(a.merged(b)) is Chunked
    assert a.merged(Chunked(gh.spec_of(bh.Histogram(R, S(["a"]))))) is None
    assert a.merged(bh.Histogram(R, S(["a", "b"]))) is None
