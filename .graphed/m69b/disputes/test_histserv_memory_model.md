# Test Dispute — `tests/frozen/m69b/test_histserv_memory_model.py` (every test: `assert warm < BASE`)

Status: **OPEN** — awaiting an owner ruling.

## The test

The harness's size-model constants (`histserv_harness.py` `BASE = 129 MiB`, `FILL_B = 3.0`) are the maximum over the plan's `probes/m69b/probe_histserv_memory{,.amd64}.txt`. Those probes ran histserv in a minimal environment (histserv + psutil). On graphed-histogram#24's CI (run 36809889213), the warm server measures 132.6–140.5 MiB on all four ubuntu-latest legs and 129.8 MiB on ubuntu-24.04-arm py3.14, so `warm < BASE` fails before any growth assertion runs.

## The clause it contradicts

plan-services.md §5.1: the model must hold for "a server … that won't crash", with constants from driven probes of the server as it runs. histserv imports `hist`, and `hist.basehist` imports `hist.interop`, which imports pandas (and pandas imports pyarrow) whenever pandas is installed. The chain was measured with `-X importtime`. The server's baseline therefore depends on its environment. The test environment and analysis environments (coffea) have pandas; the probe's environment did not. Re-running the same probe on GitHub-hosted runners in the CI test environment (`probes/m69b/probe_histserv_memory.gha.txt`) gives MODEL `B` 142–164 MiB and `b` up to 3.5. `O`, `I`, `a` and `K` stay within the existing maxima.

## Correction

- `histserv_harness.py`: `BASE = 164 * MiB` and `FILL_B = 3.5`, the maximum over `probe_histserv_memory{,.amd64,.gha}.txt`. Its comment names the three files.
- The sizes in `test_histserv_packing.py` and `test_histserv_memory_model.py` that are tuned to `BASE = 129 MiB` (the 128/140/160 MiB servers and their slot fractions) are rewritten relative to `BASE`, so that re-measuring `B` again changes one constant.
- src `_BASE`/`_FILL_B`, `design.rst`'s statement of B, and the plan's probe clause (probes run in the server's environment, pandas included) follow.
