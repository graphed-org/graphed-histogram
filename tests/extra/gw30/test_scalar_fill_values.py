"""A scalar fill value (``h.fill(dataset="x", pt=arr)``) broadcasts per entry, as an eager fill does."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import hist
import hist.graphed
import numpy as np
import pytest
from graphed.awkward import AwkwardBackend, AwkwardForm, gnano
from graphed.core import Partition
from graphed.core.execution import SequentialRunner, WorkerResources

import graphed_histogram as gh

RNG = np.random.default_rng(30)
COUNTS = RNG.integers(0, 4, 200)
EVENTS = ak.zip(
    {
        "pt": ak.unflatten(RNG.uniform(0.0, 100.0, int(COUNTS.sum())), COUNTS),
        "w": RNG.uniform(0.5, 1.5, 200),
    },
    depth_limit=1,
)
PT = ak.to_numpy(ak.flatten(EVENTS.pt))
N_PARTITIONS = 3


@dataclass
class Events:
    data: ak.Array

    def __call__(self) -> ak.Array:
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("gw30://events", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: WorkerResources) -> ak.Array:
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


def _events() -> Any:
    session = graphed.Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    return session.source("events", form=form, data=Events(EVENTS))


def _run(histograms: dict[str, Any], partitions: int = N_PARTITIONS) -> dict[str, Any]:
    """Float sums are byte-equal to one eager fill only from one partition; counts from any."""
    plan = gh.plan(histograms, steps_per_file=partitions)
    return dict(gh.unpack(SequentialRunner().run(plan).value))


def _assert_same(got: bh.Histogram, want: bh.Histogram) -> None:
    assert list(got.axes) == list(want.axes)
    assert got.view(flow=True).tobytes() == want.view(flow=True).tobytes()


PT_AXIS = hist.axis.Regular(20, 0.0, 100.0, name="pt")
CONSTANT_AXES = [
    (hist.axis.StrCategory([], growth=True, name="c"), "DYJets"),
    (hist.axis.IntCategory([], growth=True, name="c"), 2018),
    (hist.axis.Regular(4, 0.0, 4.0, name="c"), 2.5),
]


@pytest.mark.parametrize(("axis", "constant"), CONSTANT_AXES, ids=["str", "int", "regular"])
def test_a_scalar_axis_value_matches_eager_hist(axis: Any, constant: Any) -> None:
    events = _events()
    deferred = hist.graphed.Hist(axis, PT_AXIS).fill(c=constant, pt=events.pt)
    got = _run({"h": deferred})["h"]
    _assert_same(got, hist.Hist(axis, PT_AXIS).fill(c=constant, pt=PT))


def test_scalar_weight_and_sample_match_eager_hist() -> None:
    events = _events()
    weighted = hist.graphed.Hist(PT_AXIS, storage=hist.storage.Weight()).fill(pt=events.pt, weight=0.5)
    mean = hist.graphed.Hist(PT_AXIS, storage=hist.storage.Mean()).fill(pt=events.pt, sample=3.0)
    product = gh.boost.Histogram(PT_AXIS, storage=bh.storage.Weight()).fill(
        events.pt, weight=[events.pt, 0.5]
    )
    got = _run({"weighted": weighted, "mean": mean, "product": product}, partitions=1)
    _assert_same(got["weighted"], hist.Hist(PT_AXIS, storage=hist.storage.Weight()).fill(pt=PT, weight=0.5))
    _assert_same(got["mean"], hist.Hist(PT_AXIS, storage=hist.storage.Mean()).fill(pt=PT, sample=3.0))
    _assert_same(got["product"], bh.Histogram(PT_AXIS, storage=bh.storage.Weight()).fill(PT, weight=PT * 0.5))


def test_two_constants_in_one_plan_are_two_nodes() -> None:
    events = _events()
    dataset = hist.axis.StrCategory([], growth=True, name="dataset")
    a = hist.graphed.Hist(dataset, PT_AXIS).fill(dataset="a", pt=events.pt)
    b = hist.graphed.Hist(dataset, PT_AXIS).fill(dataset="b", pt=events.pt)
    both = hist.graphed.Hist(dataset, PT_AXIS).fill(dataset="a", pt=events.pt).fill(dataset="b", pt=events.pt)
    assert a.fill_nodes()[0].node_id != b.fill_nodes()[0].node_id
    assert len({node.node_id for node in both.fill_nodes()}) == 2
    assert len(both.evaluators()) == 2
    records = {n["id"]: n for n in events.session._store.nodes()}
    assert [json.loads(records[node.node_id]["params"]["constants"]) for node in both.fill_nodes()] == [
        {"axes": [[0, "a"]]},
        {"axes": [[0, "b"]]},
    ]
    got = _run({"a": a, "b": b, "both": both})
    _assert_same(got["a"], hist.Hist(dataset, PT_AXIS).fill(dataset="a", pt=PT))
    _assert_same(got["b"], hist.Hist(dataset, PT_AXIS).fill(dataset="b", pt=PT))
    _assert_same(got["both"], hist.Hist(dataset, PT_AXIS).fill(dataset="a", pt=PT).fill(dataset="b", pt=PT))


def test_an_axis_mode_fill_carries_the_constant() -> None:
    events = _events()
    dataset = hist.axis.StrCategory(["x"], name="dataset")
    weight = graphed.vary(events.w, "w", up=events.w * 1.1)
    axis_mode = hist.graphed.Hist(dataset, PT_AXIS, storage=hist.storage.Weight())
    axis_mode.fill(dataset="x", pt=events.pt, weight=weight, variation_axis=True)
    got = _run({"h": axis_mode}, partitions=1)["h"]
    for label, w in (("nominal", EVENTS.w), ("w_up", EVENTS.w * 1.1)):
        flat_w = ak.to_numpy(ak.flatten(ak.broadcast_arrays(w, EVENTS.pt)[0]))
        want = hist.Hist(dataset, PT_AXIS, storage=hist.storage.Weight()).fill(
            dataset="x", pt=PT, weight=flat_w
        )
        assert got[..., hist.loc(label)].view(flow=True).tobytes() == want.view(flow=True).tobytes(), label


def test_a_fill_of_constants_alone_is_refused() -> None:
    _events()
    with pytest.raises(TypeError, match="needs a graphed Array on some axis"):
        gh.boost.Histogram(bh.axis.StrCategory([], growth=True)).fill("x")


def test_a_scalar_factor_keeps_its_slot_after_the_ambient_weight() -> None:
    events = gnano.events(_events())
    ctx = graphed.vary(events, "pu", events.w, is_weight=True, up=events.w * 1.1)
    h = gh.boost.Histogram(PT_AXIS, storage=bh.storage.Weight()).fill(ctx.pt, weight=[ctx.pt, 0.3])
    got = _run({"h": h}, partitions=1)["h"]
    for label, ambient in (("nominal", EVENTS.w), ("pu_up", EVENTS.w * 1.1)):
        flat = ak.to_numpy(ak.flatten(ak.broadcast_arrays(ambient, EVENTS.pt)[0]))
        want = bh.Histogram(PT_AXIS, storage=bh.storage.Weight()).fill(PT, weight=flat * PT * 0.3)
        _assert_same(got[label], want)
