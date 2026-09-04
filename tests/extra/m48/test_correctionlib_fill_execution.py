"""A correctionlib `External` in a fill's input cone must EXECUTE through the plan path.

The frozen `test_correctionlib_multiparam` proves the RECORDING side (one payload, N systematic
universes as distinct nodes); it never runs a plan. The corpus weight matrix DOES run a plan, but
its b-tag SF is a pure-Python stand-in, not an `External`. So no frozen anchor filled a
`hist.graphed` histogram whose cone reads a real correctionlib node — and that path was broken:
`graphed_histogram.plan` wired only the fills' own evaluators, never the upstream corrections, so a
plan raised ``External payload '…' needs an evaluator``. This suite pins the fixed path.

Discriminators (each fails on a specific regression, not merely "is correct"):

* the plan RUNS at all — pre-fix it raised "needs an evaluator" before producing a histogram;
* the three universes are DISTINCT — a single content hash across ``central/up/down`` must resolve
  to THREE evaluators (one per ``systematic`` param), not collapse to one via the shared hash;
* SequentialRunner and a process POOL agree — the correctionlib evaluator pickled and re-loaded in
  a worker (the "hundreds of histograms with systematic variations" case is distributed);
* each universe matches an eager `materialize` reference — distinctness alone could be three WRONG
  numbers; the eager leg pins the values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
import pytest
from graphed import Array, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, gak
from graphed.core import Partition
from graphed.core.execution import SequentialRunner, WorkerResources
from graphed.preserve import CORRECTIONLIB_PLUGIN, record_external
from graphed_corpus import make_events

import graphed_histogram as gh

pytest.importorskip("correctionlib")

EVENTS = make_events(n_events=6_000, seed=48)
N_PARTITIONS = 4
SYSTEMATICS = ("nominal", "up", "down")
AXIS = bh.axis.Regular(40, 0, 800)

#: a per-event b-tag SF, binned in njet, with a `systematic` category axis. ONE payload; the three
#: universes differ ONLY in the `systematic` param passed to `evaluate`.
_SF_EDGES = [0.0, 4.0, 5.0, 6.0, 100.0]
_SF_CONTENT = {
    "nominal": [1.00, 1.00, 1.00, 1.00],
    "up": [1.05, 1.10, 1.15, 1.20],
    "down": [0.95, 0.90, 0.85, 0.80],
}


def _correctionlib_json() -> bytes:
    content = {
        syst: {"nodetype": "binning", "input": "x", "edges": _SF_EDGES, "content": vals, "flow": "clamp"}
        for syst, vals in _SF_CONTENT.items()
    }
    cset = {
        "schema_version": 2,
        "corrections": [
            {
                "name": "btag_sf",
                "version": 1,
                "inputs": [{"name": "systematic", "type": "string"}, {"name": "x", "type": "real"}],
                "output": {"name": "sf", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "input": "systematic",
                    "content": [{"key": k, "value": v} for k, v in content.items()],
                },
            }
        ],
    }
    return json.dumps(cset, sort_keys=True).encode("utf-8")


@dataclass
class CorpusEvents:
    """A `graphed.write.PartitionedSource` over the corpus events (the m48 weight-matrix shape)."""

    data: ak.Array
    part_reads: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self) -> ak.Array:
        return self.data  # the whole-dataset loader, used only by the eager `materialize` reference

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("corpus://events", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: WorkerResources) -> ak.Array:
        part = partition.resolve(len(self.data))
        self.part_reads.append((part.entry_start, part.entry_stop))
        return self.data[part.entry_start : part.entry_stop]


def _record() -> tuple[Session, CorpusEvents, gh.boost.Histogram, Array, graphed.Varied]:
    """One session: HT observable, an njet-binned correctionlib SF as a varied event weight."""
    session = Session(AwkwardBackend())
    source = CorpusEvents(EVENTS)
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    events = session.source("events", form=form, data=source)

    good = events.Jet[events.Jet.pt > 25]
    observable = gak.sum(good.pt, axis=1)  # per-event HT
    njet = gak.num(good, axis=1)  # per-event bin coordinate for the SF

    payload = _correctionlib_json()
    sf = {
        syst: record_external(
            session, CORRECTIONLIB_PLUGIN, payload, [njet], params={"name": "btag_sf", "systematic": syst}
        )
        for syst in SYSTEMATICS
    }
    btag = graphed.vary(sf["nominal"], "btag", up=sf["up"], down=sf["down"])

    h = gh.boost.Histogram(AXIS, storage=bh.storage.Double())
    h.fill(observable, weight=[btag])
    return session, source, h, observable, btag


def _run(executor: Any) -> dict[str, dict[str, gh.boost.Histogram]]:
    _session, _source, h, _obs, _btag = _record()
    plan = gh.plan({"ht": h}, steps_per_file=N_PARTITIONS)
    return gh.unpack(executor.run(plan).value)


def _eager_reference() -> dict[str, np.ndarray]:
    """The build-time histogram per universe, via `materialize` (originals present) — no plan."""
    session, _source, _h, observable, btag = _record()
    v = np.asarray(ak.to_numpy(ak.Array(session.materialize(observable))), dtype="float64")
    out: dict[str, np.ndarray] = {}
    for label in graphed.labels(btag):
        w = np.asarray(
            ak.to_numpy(ak.Array(session.materialize(graphed.universe(btag, label)))), dtype="float64"
        )
        counts, _ = np.histogram(v, bins=AXIS.size, range=(AXIS.edges[0], AXIS.edges[-1]), weights=w)
        out[label] = counts
    return out


def _values(h: gh.boost.Histogram) -> np.ndarray:
    return np.asarray(h.view(), dtype="float64")


def test_the_plan_runs_and_yields_all_three_universes() -> None:
    """Pre-fix this raised ``External payload '…' needs an evaluator`` — the fill's cone read a
    correctionlib node whose evaluator `graphed_histogram.plan` never wired."""
    seq = _run(SequentialRunner())
    assert sorted(seq["ht"]) == ["btag_down", "btag_up", "nominal"]


def test_the_three_universes_are_distinct() -> None:
    """One payload content hash, three `systematic` params → THREE evaluators. A collision onto the
    shared hash would make the universes identical (and `up`/`down` are chosen to bracket nominal)."""
    seq = _run(SequentialRunner())["ht"]
    nominal, up, down = _values(seq["nominal"]), _values(seq["btag_up"]), _values(seq["btag_down"])
    assert not np.array_equal(nominal, up)
    assert not np.array_equal(nominal, down)
    assert not np.array_equal(up, down)
    # up scales weights >= 1, down <= 1: the total yields bracket nominal (the SF was actually read)
    assert up.sum() > nominal.sum() > down.sum()


def test_the_process_pool_agrees_with_the_sequential_run() -> None:
    """The correctionlib evaluator must pickle and re-load in a worker: the distributed path is the
    whole point. A non-picklable evaluator (the pre-fix local closure) raised on plan submission."""
    ProcessPoolExecutor = pytest.importorskip("graphed_exec_local").ProcessPoolExecutor
    seq = _run(SequentialRunner())["ht"]
    par = _run(ProcessPoolExecutor(max_workers=2))["ht"]
    assert sorted(par) == sorted(seq)
    for label in seq:
        assert np.allclose(_values(par[label]), _values(seq[label]), rtol=1e-12, atol=0)


def test_each_universe_matches_its_eager_reference() -> None:
    """Distinctness alone could be three WRONG numbers; pin the values to the in-process `materialize`
    of the same recorded program (partition summation vs one pass → float tolerance, not bit-equal)."""
    seq = _run(SequentialRunner())["ht"]
    reference = _eager_reference()
    for label in seq:
        assert np.allclose(_values(seq[label]), reference[label], rtol=1e-9, atol=1e-9), label
