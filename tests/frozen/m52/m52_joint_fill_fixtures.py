"""Shared fixtures for the m52 joint-point fill anchors (§5.1 / C5).

Self-contained: the sibling m48/m49/m50 helper dirs are not on `pythonpath`, and the whole
`tests/frozen` tree is collected in ONE pytest process, so the `m52_` prefix is what keeps this
module from binding — or being bound as — another milestone's helper.

The program is the single-carrier shape §4.11-4 admits: a `jes` SHIFT registered on the event
context, then a `btag` WEIGHT family whose members are computed over the SHIFTED jets. That
dependency is what makes the family fan out: `graphed.vary` mints the full `jes` x `btag` grid, so
the two-coordinate universe `{btag: hf_up, jes: up}` exists under the machine label
`btag_hf_up__jes_up` with no tag of its own. A `points=` PLACEMENT entry selects that one joint and
prunes its siblings, leaving the one-at-a-time union untouched.

`btag_hf_up` and the joint come from the same `hf_up` member expression re-resolved at each `jes`
coordinate, so their ambient WEIGHT members are already DIFFERENT nodes. An executed inequality
between the two labels therefore does not separate a correct axis-value resolution from one that
kept nominal kinematics — C5 has to be read off the axis-value member structurally.

The oracle is eager awkward: each universe is `(jes scale, SF coefficient)`, and a JOINT row
carries BOTH coordinates. `test_joint_fill_resolution` validates the oracle against the unpruned
grid in the same run before leaning on the pruned program's joint row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
from graphed import Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak, gnano
from graphed.core import Partition
from graphed.core.execution import SequentialRunner, WorkerResources
from graphed_corpus import make_events

import graphed_histogram as gh

#: small enough to keep the suite fast, large enough that every jet multiplicity below is populated
EVENTS = make_events(n_events=400, seed=52)

#: m50's frozen per-`fill()` axis-mode opt-in spelling; this tree inherits it
AXIS_MODE_KW = "variation_axis"

JES_UP = 1.05
JES_DOWN = 0.95
SF_CENTRAL = 0.02
SF_HF_UP = 0.05
SF_HF_DOWN = -0.01

JOINT = "btag_hf_up__jes_up"
JOINT_POINT = {"btag": "hf_up", "jes": "up"}
BTAG_ONLY = "btag_hf_up"

#: the one-at-a-time union: nominal, the shift's own labels and the b-tag family's, all of which
#: the fanout leaves untouched whether or not a placement prunes the joints
BASE_LABELS = ("nominal", "jes_up", "jes_down", BTAG_ONLY, "btag_hf_down")

#: what the placement leaves standing — the union plus the ONE selected joint
LABELS = (*BASE_LABELS, JOINT)

#: the unpruned fanout: the union plus every `btag` x `jes` joint, in `graphed.labels` order
GRID_LABELS = (
    *BASE_LABELS,
    JOINT,
    "btag_hf_up__jes_down",
    "btag_hf_down__jes_up",
    "btag_hf_down__jes_down",
)

#: `(jes scale, per-jet SF coefficient)` per universe. A JOINT row carries BOTH coordinates: the
#: shifted kinematics AND the heavy-flavour SF evaluated at the shifted pT.
UNIVERSE = {
    "nominal": (1.0, SF_CENTRAL),
    "jes_up": (JES_UP, SF_CENTRAL),
    "jes_down": (JES_DOWN, SF_CENTRAL),
    BTAG_ONLY: (1.0, SF_HF_UP),
    "btag_hf_down": (1.0, SF_HF_DOWN),
    JOINT: (JES_UP, SF_HF_UP),
    "btag_hf_up__jes_down": (JES_DOWN, SF_HF_UP),
    "btag_hf_down__jes_up": (JES_UP, SF_HF_DOWN),
    "btag_hf_down__jes_down": (JES_DOWN, SF_HF_DOWN),
}

AXIS = bh.axis.Regular(40, 0, 800)


# ---- sessions ---------------------------------------------------------------------------------
def in_memory() -> tuple[Session, Any]:
    """Record-time fixture: no plan, so no partitioned source."""
    session = Session(AwkwardBackend())
    return session, gnano.events(from_awkward(session, "events", EVENTS))


@dataclass
class _Source:
    """A minimal `graphed.write.PartitionedSource` over in-memory events."""

    data: ak.Array

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind("toy://events", "", s, steps_per_file) for s in range(steps_per_file)
        )

    def read_partition(
        self, partition: Partition, columns: Any, resources: WorkerResources
    ) -> ak.Array:
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


def partitioned() -> tuple[Session, Any]:
    """What `graphed_histogram.plan` binds a plan to — axis mode has no in-memory route."""
    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    root = session.source("events", form=form, data=_Source(EVENTS))
    return session, gnano.events(root)


# ---- the program ------------------------------------------------------------------------------
def _sf(jets: Any, coeff: float) -> Any:
    """The per-event b-tag SF: a per-JET factor multiplied INSIDE the product, so the SF inherits
    the jet pT of whichever universe evaluates it.
    """
    return gak.prod(1.0 + coeff * (jets.pt / 100.0), axis=1)


def joint_context(source: Any, *, select_joint: bool = True) -> Any:
    """`jes` shift ⊗ a `btag` weight family computed over the shifted jets, so the family fans out.

    `select_joint=True` adds the `JOINT_POINT` placement, which prunes the grid to `LABELS`;
    `select_joint=False` declares through the plain mapping channel and leaves the whole
    `GRID_LABELS` fanout — the shape the eager oracle is validated against.
    """
    jets = source.Jet
    shifted = graphed.vary(
        source,
        "jes",
        Jet={
            "up": gak.with_field(jets, jets.pt * JES_UP, "pt"),
            "down": gak.with_field(jets, jets.pt * JES_DOWN, "pt"),
        },
    )
    sjets = shifted.Jet
    declares = {"hf_up": _sf(sjets, SF_HF_UP), "hf_down": _sf(sjets, SF_HF_DOWN)}
    points: Any = [*declares.items(), dict(JOINT_POINT)] if select_joint else declares
    return graphed.vary(
        shifted,
        "btag",
        _sf(sjets, SF_CENTRAL),
        is_weight=True,
        points=points,
    )


def observable(context: Any) -> Any:
    return gak.sum(context.Jet.pt, axis=1)


def ambient_only_fill(context: Any) -> gh.boost.Histogram:
    """One axis value and the context's AMBIENT b-tag weight — nothing else.

    `weight=` is deliberately absent: the ambient weight is applied automatically, and passing
    `graphed.weight(context)` back in would apply it a second time and square the SF.
    """
    h = gh.boost.Histogram(AXIS, storage=bh.storage.Double())
    h.fill(observable(context))
    return h


def three_role_fill(context: Any, *, axis_mode: bool) -> gh.boost.Histogram:
    """Axis value, an extra weight FACTOR and `sample=`, each `Varied` over `jes`, on top of the
    ambient b-tag weight: every operand role the fill lowering resolves a label through.

    `sample=` needs a Mean/WeightedMean storage; `bh` rejects it on `Double()`.
    """
    obs = observable(context)
    h = gh.boost.Histogram(AXIS, storage=bh.storage.WeightedMean())
    h.fill(
        obs,
        weight=[1.0 + 0.001 * obs],
        sample=obs * 0.5,
        **({AXIS_MODE_KW: True} if axis_mode else {}),
    )
    return h


# ---- eager oracle -----------------------------------------------------------------------------
def eager_universe(jes_scale: float, sf_coeff: float) -> bh.Histogram:
    """The universe at one `(jes, btag)` coordinate pair, in pure awkward — independent of the
    resolution rule under test.
    """
    pt = EVENTS.Jet.pt * jes_scale
    h = bh.Histogram(AXIS, storage=bh.storage.Double())
    h.fill(
        ak.to_numpy(ak.sum(pt, axis=1)),
        weight=ak.to_numpy(ak.prod(1.0 + sf_coeff * (pt / 100.0), axis=1)),
    )
    return h


# ---- structural + comparison helpers ----------------------------------------------------------
def recorded(session: Session, node: Any) -> dict[str, Any]:
    return next(n for n in session._store.nodes() if n["id"] == node.node_id)


def axis_inputs(session: Session, node: Any) -> list[int]:
    """The fill node's recorded axis-value input prefix (m48's `inputs[:n_axes]` idiom)."""
    rec = recorded(session, node)
    return list(rec["inputs"][: int(rec["params"]["n_axes"])])


def other_inputs(session: Session, node: Any) -> list[int]:
    rec = recorded(session, node)
    return list(rec["inputs"][int(rec["params"]["n_axes"]) :])


def execute(histograms: dict[str, gh.boost.Histogram], steps: int = 3) -> dict[Any, Any]:
    return dict(SequentialRunner().run(gh.plan(histograms, steps_per_file=steps)).value)


def var_index(hist: bh.Histogram) -> int:
    """The variation axis's position, read per-axis from `__dict__` — `h.axes.name` raises unless
    EVERY axis carries a name.
    """
    return next(i for i, a in enumerate(hist.axes) if a.__dict__.get("name") == "variation")


def slice_label(hist: bh.Histogram, label: str) -> bh.Histogram:
    """A pure-`bh` positional slice of the variation axis; the named-dict form is a `TypeError` on
    a bare `bh.Histogram`.
    """
    return hist[{var_index(hist): bh.loc(label)}]


def views_equal(a: bh.Histogram, b: bh.Histogram) -> bool:
    """Bin-for-bin equality including flow, NaN-tolerant per storage field."""
    va, vb = a.view(flow=True), b.view(flow=True)
    if va.dtype.names is None:
        return bool(np.allclose(np.asarray(va), np.asarray(vb), rtol=1e-12, equal_nan=True))
    return all(
        np.allclose(np.asarray(va[name]), np.asarray(vb[name]), rtol=1e-12, equal_nan=True)
        for name in va.dtype.names
    )
