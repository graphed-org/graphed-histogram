"""m48/H1 — the corpus WEIGHT-variation references, reproduced through the `vary` frontend.

Nine of the fifteen reference histograms (§10/m48): ttbar 4j1b/4j2b x {nominal, btag_up,
btag_down} and ttgamma x {nominal, pho_up, pho_down}. One Session, one plan, one pass — §5.2b's
single-read witness rides THIS run, not a toy graph, so a per-variation re-run loop cannot pass it.

Traps this file is written around (§10/m48, all binding):

* the ttgamma flat SF is `gak.full_like(<a per-event Array>, sf)` — no constant Array exists
  without a shape donor (§4.1, §11);
* the corpus rounds the observable to 6 decimals BEFORE the fill; the recorded program
  re-expresses that as `np.rint(x * 1e6) / 1e6` through `Array.__array_ufunc__`, since neither
  gak nor graphed exposes `rint` or `round(x, decimals)` and `np.round(x, 6)` records a `field`
  access and raises;
* the comparison rides `bin_values`/`fingerprint` (the m05 `test_fixtures_reproduce.py` form).
  Raw-view bit-identity against the references MUST NOT be asserted: driver-side rounding is what
  absorbs per-partition summation-order differences;
* the b-tag SF's operand is the pt-CUT jets, not every jet in the selected events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
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
from graphed_corpus import fingerprint, make_events
from graphed_corpus.histograms import bin_values

import graphed_histogram as gh

REFERENCES = Path(__file__).resolve().parents[2] / "_corpus" / "references"

#: the references' OWN dataset — `make_events()` at its defaults (20_000 events, seed 1234)
EVENTS = make_events()

N_PARTITIONS = 4

#: (output, label) -> reference file stem
MATRIX = (
    {("ttbar_4j1b", label): f"ttbar_4j1b_{label}" for label in ("nominal", "btag_up", "btag_down")}
    | {("ttbar_4j2b", label): f"ttbar_4j2b_{label}" for label in ("nominal", "btag_up", "btag_down")}
    | {("ttgamma", label): f"ttgamma_{label}" for label in ("nominal", "pho_up", "pho_down")}
)


@dataclass
class CorpusEvents:
    """A `graphed.write.PartitionedSource` over the corpus events, counting partition reads."""

    data: ak.Array
    part_reads: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("corpus://events", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: WorkerResources) -> ak.Array:
        part = partition.resolve(len(self.data))
        self.part_reads.append((part.entry_start, part.entry_stop))
        return self.data[part.entry_start : part.entry_stop]


def _source(session: Session) -> tuple[Any, CorpusEvents]:
    data = CorpusEvents(EVENTS)
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    return session.source("events", form=form, data=data), data


def _stable(values: Array) -> Array:
    """The corpus's pre-fill 6-decimal rounding, as a recorded ufunc (`np.round`'s own lowering)."""
    return np.rint(values * 1e6) / 1e6


def _ttbar(events: Any, region: str) -> tuple[Array, Array]:
    """AGC-style ttbar slice: >=4 jets pt>25, ==1 (4j1b) or >=2 (4j2b) b-tags; observable HT."""
    good = events.Jet[events.Jet.pt > 25]
    base = gak.num(good, axis=1) >= 4
    n_b = gak.sum(good.btag > 0.7, axis=1)
    selected = base & (n_b == 1) if region == "4j1b" else base & (n_b >= 2)
    sel_jets = good[selected]  # the pt-CUT jets of the selected events: the SF's operand
    return _stable(gak.sum(sel_jets.pt, axis=1)), sel_jets


def _btag_sf(sel_jets: Array, scale: float) -> Array:
    """The corpus's per-jet b-tag SF producted into a per-event weight."""
    return gak.prod((0.95 + 0.10 * sel_jets.btag) * scale, axis=1)


def _ttgamma(events: Any) -> Array:
    """TTGamma-style slice: >=1 photon pt>20, >=1 muon pt>30, >=2 jets pt>25; leading photon pT."""
    photons = events.Photon[events.Photon.pt > 20]
    muons = events.Muon[events.Muon.pt > 30]
    good_jets = events.Jet[events.Jet.pt > 25]
    selected = (
        (gak.num(photons, axis=1) >= 1) & (gak.num(muons, axis=1) >= 1) & (gak.num(good_jets, axis=1) >= 2)
    )
    return _stable(gak.drop_none(gak.firsts(photons[selected].pt)))


@lru_cache(maxsize=1)
def _matrix() -> tuple[dict[str, Any], CorpusEvents, int]:
    """ONE session, ONE plan, ONE pass over the source — the run both anchors below read."""
    session = Session(AwkwardBackend())
    events, source = _source(session)

    hists: dict[str, gh.boost.Histogram] = {}
    for region in ("4j1b", "4j2b"):
        observable, sel_jets = _ttbar(events, region)
        btag = graphed.vary(
            _btag_sf(sel_jets, 1.0),
            "btag",
            up=_btag_sf(sel_jets, 1.03),
            down=_btag_sf(sel_jets, 0.97),
        )
        h = gh.boost.Histogram(bh.axis.Regular(40, 0, 800), storage=bh.storage.Double())
        h.fill(observable, weight=[btag])
        hists[f"ttbar_{region}"] = h

    photon_pt = _ttgamma(events)
    pho = graphed.vary(
        gak.full_like(photon_pt, 0.98),
        "pho",
        up=gak.full_like(photon_pt, 1.01),
        down=gak.full_like(photon_pt, 0.95),
    )
    h = gh.boost.Histogram(bh.axis.Regular(30, 0, 300), storage=bh.storage.Double())
    h.fill(photon_pt, weight=[pho])
    hists["ttgamma"] = h

    plan = gh.plan(hists, steps_per_file=N_PARTITIONS)
    unpacked = gh.unpack(SequentialRunner().run(plan).value)
    return unpacked, source, len(plan.tasks)


@pytest.mark.parametrize(("slot", "reference"), sorted(MATRIX.items()))
def test_the_weight_matrix_reproduces_its_corpus_reference(slot: tuple[str, str], reference: str) -> None:
    output, label = slot
    unpacked, _source, _tasks = _matrix()
    got = unpacked[output][label]
    stored = json.loads((REFERENCES / f"{reference}.json").read_text(encoding="utf-8"))
    assert bin_values(got) == stored["values"], f"{reference}: bin contents drifted from the corpus"
    assert fingerprint(got) == stored["fingerprint"], f"{reference}: fingerprint drifted"


def test_every_output_carries_exactly_its_own_labels() -> None:
    """Absent labels are absent (§6.1a): the pho labels never leak onto a ttbar output, and the
    btag labels never onto ttgamma, so the reference lookups above cannot silently agree by
    reading a mapping that duplicates nominal under every label in the plan."""
    unpacked, _source, _tasks = _matrix()
    assert sorted(unpacked["ttbar_4j1b"]) == ["btag_down", "btag_up", "nominal"]
    assert sorted(unpacked["ttbar_4j2b"]) == ["btag_down", "btag_up", "nominal"]
    assert sorted(unpacked["ttgamma"]) == ["nominal", "pho_down", "pho_up"]


def test_the_reference_run_reads_each_partition_exactly_once() -> None:
    """§5.2b, bound to the reference-matrix run ITSELF. Nine histograms come out of one pass: a
    per-variation re-run loop reads `n_partitions x n_labels` and a per-histogram plan reads
    `n_partitions x n_outputs`; only the shared-IR lowering reads `n_partitions`."""
    _unpacked, source, tasks = _matrix()
    assert tasks == N_PARTITIONS
    assert len(source.part_reads) == N_PARTITIONS
    assert sorted(source.part_reads) == sorted(set(source.part_reads))
