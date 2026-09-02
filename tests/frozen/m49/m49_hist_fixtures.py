"""Shared fixtures for the m49 shift-path anchors.

Self-contained on purpose: `tests/frozen/m48` is not on `pythonpath`, and this tree's helper
basename is prefixed so the two dirs — collected in ONE pytest process — cannot silently bind
each other's module under prepend import mode.

Three fixture families live here:

* the **mixed shift+weight corpus program** the full 15-reference matrix runs through: JES varies
  the jets record BEFORE the pt cut, so each universe re-derives its own selection (§5.1), while
  the b-tag / photon scale factors vary only the weight;
* the **shared-node program** §8.2(i)'s label population keys on — one derived node consumed by
  two NON-nominal universes and by neither the nominal one, which is the only shape whose
  correspondence key is reached from two labels' cones;
* the **JER-SF stochastic program** of §5.5: one content-seeded draw shared by every universe,
  SF-varied per label.

The corpus is the VENDORED copy under `tests/_corpus` (already on `pythonpath`) — never an
`importorskip`, which would silently skip the milestone's headline gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
from graphed import Array, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.core import Partition
from graphed.core.execution import WorkerResources
from graphed_corpus import make_events

import graphed_histogram as gh

REFERENCES = Path(__file__).resolve().parents[2] / "_corpus" / "references"

#: the references' OWN dataset — `make_events()` at its defaults (20_000 events, seed 1234)
CORPUS_EVENTS = make_events()

#: a small dataset for the structural fixtures, where 20_000 events buy nothing
TOY_EVENTS = make_events(n_events=500, seed=4949)

N_PARTITIONS = 4

#: the corpus JES scales (`graphed_corpus.analyses.systematics._apply_jes`)
JES = {"nominal": 1.0, "jes_up": 1.05, "jes_down": 0.95}

#: JER scale factors: BOTH varied labels ABOVE 1 at DIFFERENT magnitudes (§5.5b) — at SF <= 1 the
#: `max(SF**2 - 1, 0)` floor smears by exactly 1, which equals nominal and kills both the
#: migration and the partition-invariance witnesses
JER_SF = {"nominal": 1.0, "jer_up": 1.20, "jer_down": 1.08}

#: inside the jet-pt bulk (the sample median sits at ~24.5), so a signed draw moves jets across it
#: in BOTH directions
JER_THRESHOLD = 25.0

#: the full m49 matrix: (output, label) -> reference file stem
MATRIX = (
    {
        ("ttbar_4j1b", label): f"ttbar_4j1b_{label}"
        for label in ("nominal", "jes_up", "jes_down", "btag_up", "btag_down")
    }
    | {
        ("ttbar_4j2b", label): f"ttbar_4j2b_{label}"
        for label in ("nominal", "jes_up", "jes_down", "btag_up", "btag_down")
    }
    | {
        ("ttgamma", label): f"ttgamma_{label}"
        for label in ("nominal", "jes_up", "jes_down", "pho_up", "pho_down")
    }
)


@dataclass
class CountingSource:
    """A `graphed.write.PartitionedSource` over in-memory events, counting partition reads."""

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


def partitioned(events: ak.Array) -> tuple[Session, Any, CountingSource]:
    session = Session(AwkwardBackend())
    data = CountingSource(events)
    form = AwkwardForm(ak.Array(events.layout.to_typetracer(forget_length=True)))
    return session, session.source("events", form=form, data=data), data


def in_memory(events: ak.Array) -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "events", events)


# ---- the mixed shift + weight corpus program --------------------------------------------------
def stable(values: Array) -> Array:
    """The corpus's pre-fill 6-decimal rounding, as a recorded ufunc (`np.round`'s own lowering):
    neither gak nor graphed exposes `rint` or `round(x, decimals)`."""
    return np.rint(values * 1e6) / 1e6


def shift_jets(jets: Any) -> Any:
    """§5.1's shift: the JES varies the jets record BEFORE any selection reads it."""
    return graphed.vary(
        jets,
        "jes",
        up=gak.with_field(jets, jets.pt * 1.05, "pt"),
        down=gak.with_field(jets, jets.pt * 0.95, "pt"),
    )


def btag_sf(sel_jets: Any, scale: float) -> Array:
    """The corpus b-tag SF: the scale multiplies each JET's factor, inside the product."""
    return gak.prod((0.95 + 0.10 * sel_jets.btag) * scale, axis=1)


def ttbar_slice(jets: Any, region: str) -> tuple[Array, Any]:
    """AGC ttbar slice: >=4 jets pt>25, ==1 (4j1b) or >=2 (4j2b) b-tags; observable HT."""
    good = jets[jets.pt > 25]
    base = gak.num(good, axis=1) >= 4
    n_b = gak.sum(good.btag > 0.7, axis=1)
    selected = base & (n_b == 1) if region == "4j1b" else base & (n_b >= 2)
    sel_jets = good[selected]
    return stable(gak.sum(sel_jets.pt, axis=1)), sel_jets


def ttgamma_slice(events: Any, jets: Any) -> Array:
    """TTGamma slice: >=1 photon pt>20, >=1 muon pt>30, >=2 jets pt>25; leading photon pT."""
    photons = events.Photon[events.Photon.pt > 20]
    muons = events.Muon[events.Muon.pt > 30]
    good_jets = jets[jets.pt > 25]
    selected = (
        (gak.num(photons, axis=1) >= 1) & (gak.num(muons, axis=1) >= 1) & (gak.num(good_jets, axis=1) >= 2)
    )
    return stable(gak.drop_none(gak.firsts(photons[selected].pt)))


def matrix_program() -> tuple[Session, dict[str, gh.boost.Histogram], CountingSource]:
    """The 15-reference program: one Session, three outputs, five labels each.

    Per §2.4 the shift labels fill with the central weight AS EVALUATED IN THEIR OWN universe
    (the b-tag SF reads that universe's selected jets) while the weight labels fill with NOMINAL
    kinematics — the corpus semantics exactly.
    """
    session, events, data = partitioned(CORPUS_EVENTS)
    jets = shift_jets(events.Jet)

    hists: dict[str, gh.boost.Histogram] = {}
    for region in ("4j1b", "4j2b"):
        observable, sel_jets = ttbar_slice(jets, region)
        central = graphed.nominal(sel_jets)
        weight = graphed.vary(
            btag_sf(sel_jets, 1.0),
            "btag",
            up=btag_sf(central, 1.03),
            down=btag_sf(central, 0.97),
        )
        h = gh.boost.Histogram(bh.axis.Regular(40, 0, 800), storage=bh.storage.Double())
        h.fill(observable, weight=[weight])
        hists[f"ttbar_{region}"] = h

    photon_pt = ttgamma_slice(events, jets)
    nominal_pt = graphed.nominal(photon_pt)
    photon_weight = graphed.vary(
        gak.full_like(photon_pt, 0.98),
        "pho",
        up=gak.full_like(nominal_pt, 1.01),
        down=gak.full_like(nominal_pt, 0.95),
    )
    h = gh.boost.Histogram(bh.axis.Regular(30, 0, 300), storage=bh.storage.Double())
    h.fill(photon_pt, weight=[photon_weight])
    hists["ttgamma"] = h
    return session, hists, data


# ---- the §8.2(i) shared-node program ----------------------------------------------------------
def shared_node_program() -> tuple[Session, gh.boost.Histogram, Any, CountingSource]:
    """§3.4's shape: ONE derived node consumed by two NON-nominal universes and by neither the
    nominal one. Interning keys on input ids, so nothing DOWNSTREAM of the fork can be shared by
    two labels with distinct members — the sharing has to sit upstream.

    Returns the session, the varied histogram, an unvaried histogram recorded in the SAME session
    (the admitted member of the hook's `None` rule), and the read-counting source.
    """
    session, events, data = partitioned(TOY_EVENTS)
    base = events.MET.pt
    shared = base * 2.0  # reached from BOTH non-nominal cones, from NEITHER the nominal one
    varied = graphed.vary(base, "s", up=shared + base, down=shared * base)

    hist = gh.boost.Histogram(bh.axis.Regular(10, 0.0, 400.0), storage=bh.storage.Int64())
    hist.fill(varied)

    plain = gh.boost.Histogram(bh.axis.Regular(10, 0.0, 400.0), storage=bh.storage.Int64())
    plain.fill(base)
    return session, hist, plain, data


# ---- the §5.5 JER-SF stochastic program --------------------------------------------------------
def content_seeded_normal(values: Array) -> Array:
    """A standard normal drawn as a PURE FUNCTION OF EACH ROW'S OWN CONTENT (§5.5a).

    Two hashed uniforms of the row value, Box-Muller'd into a normal. Everything is a numpy ufunc
    dispatched through `Array.__array_ufunc__`, so the draw is one recorded sub-graph and NOT an
    External — `graphed_histogram.plan` ships only the histograms' own evaluators, so an External
    draw would be unresolvable in the worker. No global RNG, no wall clock, no partition index
    reaches it, which is what makes the same row draw the same value under any partitioning.
    """

    def hashed_uniform(x: Array, a: float, b: float) -> Array:
        raw = np.sin(x * a + b) * 43758.5453123
        return raw - np.floor(raw)

    u1 = hashed_uniform(values, 12.9898, 4.1414) * 0.9998 + 1e-4  # off the log's pole
    u2 = hashed_uniform(values, 78.233, 1.7182)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def jer_program() -> tuple[Session, Any, dict[str, dict[str, Array]], Array, CountingSource]:
    """One shared draw, SF-varied per label (§5.5b).

    The smear is coffea's stochastic shape ``1 + sqrt(max(SF**2 - 1, 0)) * g``. Returns the
    session, the varied smeared-pt container, ``{label: {"smeared": ..., "mask": ...}}`` per
    universe, the shared draw node, and the read-counting source.
    """
    session, events, data = partitioned(TOY_EVENTS)
    pt = events.Jet.pt
    draw = content_seeded_normal(pt)
    # each member re-records the draw from scratch: interning is what collapses them to ONE node
    members = {
        tag: pt
        * (1.0 + float(np.sqrt(max(JER_SF[f"jer_{tag}"] ** 2 - 1.0, 0.0))) * content_seeded_normal(pt))
        for tag in ("up", "down")
    }
    smeared = graphed.vary(pt, "jer", variations=members)  # nominal is UNSMEARED (§5.5b)

    per_label: dict[str, dict[str, Array]] = {}
    for label in graphed.labels(smeared):
        value = graphed.universe(smeared, label)
        per_label[label] = {"smeared": gak.flatten(value), "mask": gak.flatten(value > JER_THRESHOLD)}
    return session, smeared, per_label, draw, data


# ---- eager oracles ------------------------------------------------------------------------------
def eager_jes_mask(events: ak.Array, label: str, n_jets: int) -> ak.Array:
    """The same JES selection computed by hand from the same array."""
    jets = ak.with_field(events.Jet, events.Jet.pt * JES[label], "pt")
    return ak.num(jets[jets.pt > 25.0], axis=1) >= n_jets


def eager_counts(bins: int = 10, lo: float = 0.0, hi: float = 400.0) -> bh.Histogram:
    return bh.Histogram(bh.axis.Regular(bins, lo, hi), storage=bh.storage.Int64())


def eager_weighted(bins: int = 20, lo: float = 0.0, hi: float = 400.0) -> bh.Histogram:
    return bh.Histogram(bh.axis.Regular(bins, lo, hi), storage=bh.storage.Weight())
