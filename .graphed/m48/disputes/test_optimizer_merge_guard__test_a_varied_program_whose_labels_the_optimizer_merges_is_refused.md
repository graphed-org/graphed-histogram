# Test Dispute — `tests/frozen/m48/test_optimizer_merge_guard.py::test_a_varied_program_whose_labels_the_optimizer_merges_is_refused`

Status: **CLOSED** — owner ruling 2026-09-30: refreeze authorized (refrozen in `test(frozen): merged fills are read once per marked fill`).

## The test

`tests/frozen/m48/test_optimizer_merge_guard.py::test_a_varied_program_whose_labels_the_optimizer_merges_is_refused` asserted that `gh.plan({"met": _merging(events)})` raises `GraphedError` naming `met`, `nominal`, `sig_up` and `points=`.

## The clause it contradicts

plan-services.md §5.1 `boost.py`: "The reduce reads every fill at its position, so a fill the optimizer merged with another along the weights axis is read once per marked fill, whatever else the plan marks and in whatever order (`probes/m69b/probe_merged_leaves.txt` B) … `gh.plan` (backed or not) builds from a fresh `pieces` per call … and `Histogram.plan()` reads its fills the same way, so neither refuses a merge: `_refuse_shortfall` goes." With the refusal gone the test asserts behaviour the plan removes.

## Correction

Renamed `test_a_varied_program_whose_labels_the_optimizer_merges_fills_every_label`: the plan runs on `SequentialRunner`, `gh.unpack` gives `nominal` and `sig_up`, nominal is non-empty, and `sig_up`'s flow view (value and variance) equals `nominal`'s bit for bit. The positive control and the instrument's assertions stay; the docstring sentences describing the refusal are reworded to the read.
