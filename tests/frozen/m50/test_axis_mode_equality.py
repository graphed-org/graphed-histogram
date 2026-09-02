"""m50 — §6.2 axis-vs-sibling equality: weight labels collapse into one histogram.

An opt-in per-`fill()` mode lands weight-label variations in ONE histogram carrying a pre-declared
`"variation"` StrCategory axis via an evaluator-side loop, while shift and `sample=` labels stay
sibling fills targeting that same axis. The result equals the frozen SIBLING-mode decomposition
label-by-label — the sibling lowering (m48/m49, eager-referenced) is the oracle. The value is
per-object and the weight factor per-event, so §6.1d's broadcast seam is genuinely engaged.

These anchors exercise source that does NOT ship at the m50 baseline (the axis-mode opt-in), so
each FAILS today at the opt-in keyword and passes once H1/H2 land.
"""

from __future__ import annotations

import numpy as np
from m50_axis_fixtures import (
    execute,
    fill_mixed_program,
    fill_weight_program,
    in_memory,
    node_chash,
    partitioned,
    slice_label,
    views_equal,
)

import graphed_histogram as gh

#: `S` = labels that lower as siblings (borne by an axis value or a `Varied` sample=); `W` = labels
#: borne only by weight factors, which collapse into the loop. The mixed program's split.
MIXED_LABELS = ("nominal", "jes_up", "jes_down", "wgt_up", "wgt_down", "smp_up", "smp_down")
SIBLINGS = ("jes_up", "jes_down", "smp_up", "smp_down")  # S
WEIGHTS = ("wgt_up", "wgt_down")  # W


def _axis_result(program_fn) -> object:
    _s, events, _src = partitioned()
    return gh.unpack(execute({"h": program_fn(True, source=events)}))["h"]


def _sibling_result(program_fn) -> dict:
    _s, events, _src = partitioned()
    return gh.unpack(execute({"h": program_fn(False, source=events)}))["h"]


def test_weight_labels_collapse_into_one_axis_histogram_equal_to_the_siblings() -> None:
    """Per-object value, per-event varied weight: the axis-mode histogram sliced at each weight
    label equals the sibling fill for that label bin-for-bin, flow included."""
    axis = _axis_result(fill_weight_program)
    sibling = _sibling_result(fill_weight_program)
    for label in ("nominal", "wgt_up", "wgt_down"):
        assert views_equal(slice_label(axis, label), sibling[label]), label


def test_the_weight_equality_is_not_vacuous() -> None:
    """The instrument: every universe genuinely differs from nominal and nominal is non-empty, so
    the equality above cannot pass on an implementation that never applied a variation."""
    axis = _axis_result(fill_weight_program)
    nominal = np.asarray(slice_label(axis, "nominal").view(flow=True)["value"])
    assert float(nominal.sum()) > 0.0
    for label in ("wgt_up", "wgt_down"):
        assert not np.array_equal(np.asarray(slice_label(axis, label).view(flow=True)["value"]), nominal)


def test_mixed_shift_weight_sample_axis_equals_the_sibling_decomposition() -> None:
    """WeightedMean storage, a shift (`S`), a weight (`W`) and a sample-only variation (`S`) with
    per-label sample values that DIFFER: the single axis-mode histogram equals the seven-way sibling
    decomposition label-by-label."""
    axis = _axis_result(fill_mixed_program)
    sibling = _sibling_result(fill_mixed_program)
    assert set(sibling) == set(MIXED_LABELS)
    for label in MIXED_LABELS:
        assert views_equal(slice_label(axis, label), sibling[label]), label


def test_a_sample_only_label_lowers_as_a_sibling_not_into_the_loop() -> None:
    """§6.1b's `S`/`W` split, observable only in axis mode: `smp_*` is borne solely by a `Varied`
    sample= and must lower as a SIBLING (the loop re-fills against a FIXED sample column). Axis
    arity is `1 + |S|` = 5; misclassing either `smp` label into `W` would drop it to 3."""
    _s, events = in_memory()
    axis = fill_mixed_program(True, source=events)
    _s2, events2 = in_memory()
    sibling = fill_mixed_program(False, source=events2)
    assert axis.staged_fills() == 1 + len(SIBLINGS)  # 5: the weight loop plus |S| sibling nodes
    assert sibling.staged_fills() == 1 + len(SIBLINGS) + len(WEIGHTS)  # 7: no collapse in sibling mode


def test_axis_mode_fill_nodes_carry_distinct_evaluators() -> None:
    """§6.2's per-fill carrier: the `1 + |S|` axis-mode fill nodes hash on
    `content_hash((spec, variation_payload))`, so each resolves to its OWN evaluator. The registry
    keys on payload hash alone — collapse them to one hash and every sibling resolves to whichever
    evaluator was registered last, which bin-for-bin equality can mask when two siblings agree.
    Sibling mode is the contrast: one shared `content_hash(spec)` for the whole fill call."""
    session, events = in_memory()
    axis = fill_mixed_program(True, source=events)
    axis_chashes = [node_chash(session, node) for node in axis.fill_nodes()]
    assert len(set(axis_chashes)) == len(axis_chashes) == 1 + len(SIBLINGS)
    assert all(chash in axis.evaluators() for chash in axis_chashes)

    sib_session, sib_events = in_memory()
    sibling = fill_mixed_program(False, source=sib_events)
    sib_chashes = {node_chash(sib_session, node) for node in sibling.fill_nodes()}
    assert len(sib_chashes) == 1, "sibling mode shares one content_hash(spec) across the fill call"
