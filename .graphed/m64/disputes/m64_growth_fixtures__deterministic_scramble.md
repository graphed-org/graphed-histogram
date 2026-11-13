# Test Dispute: `tests/frozen/m64/m64_growth_fixtures.py::ArraySource.read_partition` (every process-runner `assert got.scrambled()`)

Status: CLOSED. Owner ruling 2026-11-13: refreeze authorized, refrozen in `test(frozen): m64 partition 0 completes after partition 1`, tag freeze-m64-fixup3.

## The test

Partition `i` of `n` sleeps `(n - i) * DT`, so on a process runner a later partition should finish first and `scrambled()` should hold. The pre-warm (freeze-m64-fixup, #28) took worker spawn out of the timed run, but staggered task starts can still cancel the sleeps. Merge-group run 43239110985 (graphed-histogram#69) failed `test_g7_...[disjoint-procpool3-h.plan]` on windows-latest py3.13 with stamps in key order: partitions 0–2 finished within 0.25 ms of each other, and partition 3 about `DT` later (graphed-histogram#27).

## The clause it contradicts

The stagger is the fixture's stated mechanism for completion order. When tasks start is not part of what the m64 anchors test.

## Correction

When `dt` is set and the run is not reversed, partition 0 waits for partition 1's stamp, with a `10 * n * dt` deadline, before its own sleep. Both run in the first wave on every process runner, so partition 1 completes before partition 0. The stamp trails partition 1's by `n * dt`, well above Windows py3.11/3.12's 15.6 ms `time.time_ns()` tick. `scrambled()`, `DT` and every assertion are unchanged. m69b's digest of `tests/frozen/m64` follows (freeze-m69b-fixup7).

## Ruling

Owner, 2026-11-13: "histogram#27 is allowed to edit a frozen test".
