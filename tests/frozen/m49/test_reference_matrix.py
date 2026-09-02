"""m49/F1 — the FULL 15-reference corpus matrix through the `vary` frontend (§10/m49 anchor (i)).

m48 reproduced the nine weight-variation references from a weight-only program. This is the
mixed one: JES varies the jets record before the pt cut, so the five ttbar/ttgamma labels split
into two shift labels that re-derive their own selection and two weight labels that do not — one
Session, one plan, one pass over the source.

The comparison rides `bin_values`/`fingerprint` (the m05 `test_fixtures_reproduce.py` form).
Raw-view bit-identity against the references is NOT asserted: the corpus rounds driver-side, and
that rounding is what absorbs per-partition summation-order differences. "Bit-for-bit" is claimed
run-to-run only, against a second independent build of the same program.

§5.2b's read witness binds to THIS run, and so does §8.2(i)'s label channel: the shipped closure
of this plan is where the milestone's transport either carries the program's labels or does not.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from graphed.core.execution import SequentialRunner
from graphed_corpus import fingerprint
from graphed_corpus.histograms import bin_values
from m49_hist_fixtures import MATRIX, N_PARTITIONS, REFERENCES, matrix_program

import graphed_histogram as gh

#: the labels the program registers, nominal excluded (§8.2(i)'s nominal-exclusion rule)
NON_NOMINAL = {"jes_up", "jes_down", "btag_up", "btag_down", "pho_up", "pho_down"}

_RUN: dict[str, object] = {}


def _run() -> dict[str, object]:
    """ONE session, ONE plan, ONE pass — the run every anchor below reads."""
    if not _RUN:
        _session, hists, source = matrix_program()
        plan = gh.plan(hists, steps_per_file=N_PARTITIONS)
        _RUN.update(
            unpacked=gh.unpack(SequentialRunner().run(plan).value),
            source=source,
            tasks=len(plan.tasks),
            labels=plan.process.variation_labels,
        )
    return _RUN


def test_every_reference_the_matrix_claims_is_actually_vendored() -> None:
    """The live instrument for the parametrized comparison: a missing vendored reference would
    otherwise shrink the matrix silently rather than fail it."""
    missing = sorted(stem for stem in MATRIX.values() if not (REFERENCES / f"{stem}.json").is_file())
    assert missing == []
    assert len(set(MATRIX.values())) == 15


@pytest.mark.parametrize(("slot", "reference"), sorted(MATRIX.items()))
def test_the_matrix_reproduces_its_corpus_reference(slot: tuple[str, str], reference: str) -> None:
    output, label = slot
    got = _run()["unpacked"][output][label]  # type: ignore[index]
    stored = json.loads((REFERENCES / f"{reference}.json").read_text(encoding="utf-8"))
    assert bin_values(got) == stored["values"], f"{reference}: bin contents drifted from the corpus"
    assert fingerprint(got) == stored["fingerprint"], f"{reference}: fingerprint drifted"


def test_every_output_carries_exactly_its_own_five_labels() -> None:
    """Absent labels are absent (§6.1a), so the reference lookups above cannot agree by reading a
    mapping that replicates nominal under every label the plan knows."""
    unpacked = _run()["unpacked"]
    assert sorted(unpacked["ttbar_4j1b"]) == ["btag_down", "btag_up", "jes_down", "jes_up", "nominal"]
    assert sorted(unpacked["ttbar_4j2b"]) == ["btag_down", "btag_up", "jes_down", "jes_up", "nominal"]
    assert sorted(unpacked["ttgamma"]) == ["jes_down", "jes_up", "nominal", "pho_down", "pho_up"]


def test_the_matrix_run_reads_each_partition_exactly_once() -> None:
    """§5.2b, bound to the matrix run ITSELF. Fifteen histograms come out of one pass: a
    per-variation re-run loop reads `n_partitions x n_labels`, a per-histogram plan reads
    `n_partitions x n_outputs`; only the shared-IR lowering reads `n_partitions`."""
    run = _run()
    source = run["source"]
    assert run["tasks"] == N_PARTITIONS
    assert len(source.part_reads) == N_PARTITIONS  # type: ignore[union-attr]
    assert sorted(source.part_reads) == sorted(set(source.part_reads))  # type: ignore[union-attr]


def test_a_second_independent_build_is_bit_for_bit_identical() -> None:
    """Run-to-run byte identity, the only form in which this suite claims it (§10/m49(i)): a
    second Session builds and runs the same program, and every view compares `array_equal`."""
    _session, hists, _source = matrix_program()
    again = gh.unpack(SequentialRunner().run(gh.plan(hists, steps_per_file=N_PARTITIONS)).value)
    first = _run()["unpacked"]
    for output, label in sorted(MATRIX):
        left = first[output][label].view(flow=True)  # type: ignore[index]
        right = again[output][label].view(flow=True)
        assert np.array_equal(left, right), f"{output}/{label} is not run-to-run identical"
        assert left.sum() > 0, f"{output}/{label} is empty, so the comparison witnesses nothing"


def test_the_matrix_plan_ships_a_populated_variation_label_channel() -> None:
    """§8.2(i) on a real analysis plan: the shipped closure's channel carries this program's
    non-nominal labels and no other. `None` is what m48 ships and is the state m49 replaces."""
    payload = _run()["labels"]
    assert payload is not None, "the shipped closure carries no variation labels"
    seen: set[str] = set()
    for _key, (labels, _frame) in payload:  # type: ignore[union-attr]
        seen.update(labels)
    assert seen == NON_NOMINAL
