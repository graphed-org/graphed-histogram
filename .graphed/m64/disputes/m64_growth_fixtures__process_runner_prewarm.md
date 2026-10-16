# Test Dispute — `tests/frozen/m64/m64_growth_fixtures.py::run` (every process-runner `assert got.scrambled()`)

Status: CLOSED — owner ruling 2026-10-16: refreeze authorized, refrozen in `test(frozen): m64 process runners spawn their workers before the timed run`, tag freeze-m64-fixup.

## The test

Partition `i` of `n` sleeps `(n - i) * DT` (`DT = 0.1 s`), so on a process runner a later partition finishes first and `scrambled()` holds. The runners spawn their workers lazily, and on GitHub-hosted runners each spawn takes about `DT`, so worker `k` starts its partition about `k * DT` late and cancels the stagger: partitions 0, 1, 2 finished 2 ms apart in key order. This failed main's push run 38816427450 and three runs of graphed-histogram#26, and ejected #26 from the merge queue (graphed-histogram#27).

## The clause it contradicts

The stagger is the fixture's stated mechanism for completion order ("the stamp follows the sleep so it records completion, not key order"); worker start-up time is not part of what the m64 anchors test.

## Correction

`run` builds the process runner with `persistent=True` and runs the same plan shape once untimed, over an unstamped source with no sleep, before the timed run, so every worker is up before the stagger starts. `scrambled()`, `DT` and every assertion are unchanged. `RUNNERS` passes keyword arguments through to the executors. m69b's digest of `tests/frozen/m64` follows.

## Ruling

Owner, 2026-10-16: one frozen harness edit pre-warming the process runners; the `scrambled()` assertion and `DT` stay as they are.
