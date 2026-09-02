"""m48/H9 — §6.1d: unification is not enough; ancestor VALUES are re-indexed too.

Inputs whose contexts sit on one ancestry chain unify to the most-derived context, and every
ancestor-context value is then re-expressed in that context's row space across the intervening
links, label-aligned per §2.4. Two of the three link kinds are exercised over the VALUE here:

* **(1) mask-derivation** — each label's ancestor value re-indexed by THAT label's own mask,
  nominal's by nominal's. An implementation that unifies the handles but never re-indexes fails on
  the row COUNT alone, so the assertion bites before the contents are even compared;
* **(3) universe/nominal projection** — the ancestor `Varied` is projected to that label's member,
  which carries NO labels, so the fill is UNVARIED and its result is a BARE `hist` (§6.1a). That
  result-shape assertion is the discriminator against an implementation that keeps the labels and
  projects only the contents.

Both axis values are PER-EVENT and this fixture must not be "improved" to a per-object second
axis: the evaluator flattens each axis independently and boost-histogram 1.8.0 requires equal
lengths across axes (`ValueError: spans must have compatible lengths`), so it would red for a
reason unrelated to what this file asserts. §6.1d's broadcast seam is scoped to weight factors —
nothing broadcasts one axis value against another.

The `sample=` extension rides link kind (1): an ancestor-context `sample=` is re-indexed like any
other ancestor VALUE, which is why this fixture carries a `WeightedMean` storage — bh 1.8.0
rejects `sample=` on `Double()` and on `Weight()` while the evaluator passes it straight through,
so a default-storage fixture records cleanly and dies at evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
from graphed.awkward import gak, gnano
from graphed.core.execution import SequentialRunner
from vary_hist_fixtures import (
    EVENTS,
    eager_sampled_2d,
    eager_weighted_2d,
    in_memory_events,
    partitioned_events,
    sampled_2d,
    weighted_2d,
)

import graphed_histogram as gh

JES = {"nominal": 1.0, "jes_up": 1.05, "jes_down": 0.95}


def _shifted_jets(source: Any, scale: float) -> Any:
    jets = source.Jet
    return gak.with_field(jets, jets.pt * scale, "pt")


def _derived(events_ctx: Any) -> Any:
    """A VARIED mask derivation: each label selects its own row set, so re-indexing everything by
    nominal's mask is a different answer for every non-nominal label."""
    shifted = graphed.vary(
        events_ctx,
        "jes",
        Jet={"up": _shifted_jets(events_ctx, 1.05), "down": _shifted_jets(events_ctx, 0.95)},
    )
    return shifted[gak.num(shifted.Jet[shifted.Jet.pt > 25.0], axis=1) >= 2]


def _eager_mask(label: str) -> ak.Array:
    """The same mask computed by hand from the same array — the reference's own row set."""
    jets = ak.with_field(EVENTS.Jet, EVENTS.Jet.pt * JES[label], "pt")
    return ak.num(jets[jets.pt > 25.0], axis=1) >= 2


def test_an_ancestor_value_is_re_indexed_per_label_by_that_labels_own_mask() -> None:
    session, root = in_memory_events()
    events = gnano.events(root)
    sel = _derived(events)

    h = sampled_2d()
    h.fill(events.MET.pt, sel.MET.pt, sample=events.MET.phi)  # axis 0 and the sample are ANCESTORS
    per_label = gh.fill_nodes_by_label(h)
    assert set(per_label) == set(JES)

    for label in per_label:
        mask = _eager_mask(label)
        rows = ak.to_numpy(EVENTS.MET.pt[mask])
        want = eager_sampled_2d()
        want.fill(rows, rows, sample=ak.to_numpy(EVENTS.MET.phi[mask]))
        got = session.materialize(per_label[label])
        for field in ("sum_of_weights", "value"):
            assert np.allclose(
                got.view(flow=True)[field], want.view(flow=True)[field], rtol=1e-12, equal_nan=True
            )


def test_the_labels_row_sets_really_differ_so_the_re_indexing_is_observable() -> None:
    """The instrument for the assertion above: if every label's mask selected the same rows,
    re-indexing by nominal's mask would be indistinguishable from re-indexing per label."""
    rows = {label: int(ak.sum(_eager_mask(label))) for label in JES}
    assert len(set(rows.values())) == len(rows)
    assert all(0 < n < len(EVENTS) for n in rows.values())


def test_a_projection_link_yields_an_unvaried_fill_whose_result_is_a_BARE_hist() -> None:
    """§6.1d computes the fill's label set AFTER the lineage step, so a value reached across a
    projection link contributes no labels and §6.1a's unvaried shape applies."""
    _session, source, _data = partitioned_events()
    sel = _derived(gnano.events(source))
    child = graphed.nominal(sel)

    h = weighted_2d()
    h.fill(child.MET.pt, sel.MET.pt)
    assert set(gh.fill_nodes_by_label(h)) == {"nominal"}

    result = gh.unpack(SequentialRunner().run(gh.plan({"met": h}, steps_per_file=3)).value)
    assert isinstance(result["met"], bh.Histogram)
    assert not isinstance(result["met"], Mapping), "a projected fill is UNVARIED: a bare hist"
    assert tuple(graphed.labels(result["met"])) == ("nominal",)

    rows = ak.to_numpy(EVENTS.MET.pt[_eager_mask("nominal")])
    want = eager_weighted_2d()
    want.fill(rows, rows)
    assert np.array_equal(result["met"].view(flow=True)["value"], want.view(flow=True)["value"])
