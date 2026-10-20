# Test Dispute — `tests/frozen/m69b/histserv_harness.py` `BASE` (every `test_histserv_memory_model.py` test: `assert warm < BASE`)

Status: CLOSED — owner ruling 2026-10-20: refreeze authorized ("fix the issues for 0.0.6", graphed-histogram#31), refrozen in `test(frozen): m69b size model B leaves room for a runner-image change`, tag freeze-m69b-fixup2.

## The test

`BASE = 164 * MiB` is the largest MODEL `B` (server at rest + announcer) in `probe_histserv_memory{,.amd64,.gha}.txt`, with no headroom. On GitHub's ubuntu-24.04 image `20261018.339.1` (x86_64) the warm server measures 170.2–170.5 MiB under py3.14 and 163.5–163.8 MiB under py3.11–3.13, so `warm < BASE` fails on py3.14 (graphed-histogram#31) before any growth assertion runs. On `20261012.334.1` the same legs measure 129.9–137.1 MiB.

The growth assertions already measure against the same run: `grew = VmHWM − warm`, bounded by `prediction − BASE`, which does not contain `BASE`. Only `warm < BASE` depends on the runner image.

## The clause it contradicts

plan-services.md §5.1: the model must hold for a server as it runs, with constants from driven probes. Re-running the plan's probe on both images (`probe_histserv_memory.gha-20261018.txt`) gives MODEL `B` 161 MiB on `20261012.334.1` and 194 MiB on `20261018.339.1` (x86_64, py3.14). The images differ in kernel (6.17 → 7.0) and transparent hugepages (`always` → `madvise`); with identical wheels and CPython builds, `import histserv.service` leaves 60.6 MiB file-backed resident on the old image and 103.4 MiB on the new. `I` reaches 170 B (model 160). `a` reaches 6.5 (model 5.5), but every measured `(a, b)` pair has `a + b ≤ 9.0` and `b ≤ 3.5`, so `a + b × workers` stays under the model's `5.5 + 3.5 × workers` for every `workers ≥ 1`.

## Correction

- `histserv_harness.py` and src: `BASE = 227 * MiB`, the largest MODEL `B` (194 MiB) plus the largest change in `B` between two runner images (194 − 161 = 33 MiB). `PER_TASK = 170`, the new maximum. Every other constant and every assertion is unchanged.
- `design.rst` and the changelog state the new `B` and `I`.

A slot-handling regression still fails both before and after: with histserv's `ChunkedHist._save_chunk_view` patched to hold 4 extra copies of each stored chunk, the 64 MiB, 8-label and 2000-slot tests fail `grew <= prediction − BASE` on all 13 probe legs over both images, at base and after. One extra copy stays under the prediction at base and after (the smallest margin is 9.3 MiB on the 8-label test); the tests are meant to catch "a server past its prediction", and one extra copy does not push the server past it.

## Ruling

Owner, 2026-10-20 (via team-lead): graphed-histogram 0.0.6 ships with #31 closed. The refreeze is authorized as part of fixing it at the cause.
