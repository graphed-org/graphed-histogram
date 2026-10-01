# Test Dispute — `tests/frozen/m49/test_merge_shortfall.py::test_the_single_histogram_plan_refuses_a_merged_program`

Status: **CLOSED** — owner ruling 2026-09-30: refreeze authorized (refrozen in `test(frozen): merged fills are read once per marked fill`).

## The test

`tests/frozen/m49/test_merge_shortfall.py::test_the_single_histogram_plan_refuses_a_merged_program` asserted that `_merging().plan()` raises `GraphedError` containing `merged`.

## The clause it contradicts

plan-services.md §5.1 `boost.py`: "The reduce reads every fill at its position, so a fill the optimizer merged with another along the weights axis is read once per marked fill, whatever else the plan marks and in whatever order (`probes/m69b/probe_merged_leaves.txt` B) … `gh.plan` (backed or not) builds from a fresh `pieces` per call … and `Histogram.plan()` reads its fills the same way, so neither refuses a merge: `_refuse_shortfall` goes." With the refusal gone the test asserts behaviour the plan removes.

## Correction

Renamed `test_the_single_histogram_plan_sums_both_fills`: `Histogram.plan()`'s run of `_merging()` equals two direct fills to `rtol=1e-12`. The positive control and the instrument's assertions stay; the docstring sentences describing the refusal are reworded to the read.
