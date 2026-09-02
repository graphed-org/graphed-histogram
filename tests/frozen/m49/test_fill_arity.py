"""m49/F5 — §2.4/§6.1b: a mixed shift+weight fill records `1 + |S| + |W|` siblings, never a product.

Scoped to SIBLING mode, which is what m48 and m49 lower to. §6.2's axis mode records `1 + |S|`
over the same definitions and lands at m50, so the count is asserted against the program's own
S and W rather than a literal that a later mode would have to contradict.

S and W are read off LOWERING BEHAVIOUR, not off the shift/weight vocabulary: `S` is the labels
borne by an axis value (or a `Varied` `sample=`), `W` the labels borne only by weight factors.
Here the JES labels reach the observable and the b-tag labels reach only the weight.
"""

from __future__ import annotations

from typing import Any

import boost_histogram as bh
import graphed
from m49_hist_fixtures import TOY_EVENTS, btag_sf, partitioned, shift_jets, ttbar_slice

import graphed_histogram as gh


def _mixed() -> tuple[Any, Any, gh.boost.Histogram]:
    """One fill combining shift-varied kinematics with a stacked weight `Varied` (§2.4)."""
    _session, events, _data = partitioned(TOY_EVENTS)
    observable, sel_jets = ttbar_slice(shift_jets(events.Jet), "4j1b")
    central = graphed.nominal(sel_jets)
    weight = graphed.vary(
        btag_sf(sel_jets, 1.0), "btag", up=btag_sf(central, 1.03), down=btag_sf(central, 0.97)
    )
    hist = gh.boost.Histogram(bh.axis.Regular(40, 0, 800), storage=bh.storage.Double())
    hist.fill(observable, weight=[weight])
    return observable, weight, hist


def test_a_mixed_fill_records_one_sibling_per_label_and_not_their_product() -> None:
    observable, weight, hist = _mixed()
    shift_labels = set(graphed.labels(observable)) - {"nominal"}
    weight_labels = set(graphed.labels(weight)) - {"nominal"} - shift_labels
    assert shift_labels and weight_labels, "the fixture must carry BOTH label classes"

    assert hist.staged_fills() == 1 + len(shift_labels) + len(weight_labels)
    assert hist.staged_fills() != (1 + len(shift_labels)) * (1 + len(weight_labels))


def test_each_sibling_is_a_distinct_node_carrying_its_own_label() -> None:
    """The count alone is satisfied by an implementation that records one node N times, which the
    §6.1c slot layout then sums into the wrong universes."""
    _observable, weight, hist = _mixed()
    per_label = gh.fill_nodes_by_label(hist)
    assert set(per_label) == set(graphed.labels(weight))
    assert len({node.node_id for node in per_label.values()}) == len(per_label)


def test_the_labels_reach_the_fill_from_two_different_operands() -> None:
    """§2.4's alignment: shift labels fill with the central weight AS EVALUATED IN THEIR OWN
    universe, weight labels with NOMINAL kinematics. A fixture whose classes both came from the
    weight would count `1 + |S| + |W|` without ever exercising the no-cross-product rule."""
    observable, weight, _hist = _mixed()
    assert set(graphed.labels(observable)) == {"nominal", "jes_up", "jes_down"}
    assert set(graphed.labels(weight)) == {"nominal", "jes_up", "jes_down", "btag_up", "btag_down"}
