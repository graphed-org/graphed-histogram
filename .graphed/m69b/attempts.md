# m69b attempts — graphed-histogram (histserv backend)

Frozen suite: `tests/frozen/m69b/**` at `freeze-m69b` (56992b7), plus the refrozen m48/m49 tests
(`freeze-m48-fixup2`, `freeze-m49-fixup` = 67131cb). Baseline at 56992b7: 50 failed (the 4 refrozen
m48/m49 tests and 46 m69b items; the 5 memory-model items skip off Linux).

## Iteration 1 — 2026-09-30 — context, backing, size model, packing, server specs

- `graphed_histogram.histserv`: `Context` (sorted distinct `memory_mb`, a process-wide name
  registry under one module lock with the packing state; equal arguments share and warn, others
  refuse naming both and the remedy), `backed`/`Histogram` (growth axes and storages other than
  Double/Int64/Weight refused naming histserv), the size model with the probes' max-over-MODEL
  constants (B 129 MiB, O 4000, I 160, a 5.5, b 3.0, K 19 KiB), first-fit decreasing placement
  ((a) open servers, (b) smallest offered size, (c) an own server at the alone prediction rounded
  up, closed to other slots), the 2^29 ceiling refusal, the `next_tasks`/repeated-partition
  refusals (all before placement), one `ServiceSpec` per landed server declared on the session,
  `plan.services` sorted by name. `_Served` carries `slots` only here.
- `boost.py`: `pieces()` / `HistogramPieces` (`on_compiled` refuses a second firing before
  recording, records each fill's compiled position, returns `_variation_labels`; `serve` sizes
  backed slots, returns the plan unchanged otherwise, refuses a second serve). Both reducers read
  each staged fill at its compiled position, so the m48/m49 refreeze (merged fills read once per
  marked fill) lands here with `pieces`, which needs it; `_refuse_shortfall` goes. `gh.plan` builds
  through `pieces` and returns `pieces.serve(plan)`; `Histogram.plan()` refuses a backed histogram.
- `tests/extra/m48/test_review_witnesses.py`'s A4 refusal witness becomes the read it now gets.
- Floor: `graphed.services` exists only at the unreleased graphed a0638719 (PyPI 0.0.6 lacks it,
  and the sha reports 0.0.6 too), so no truthful version floor covers it; the base floor stays
  `graphed>=0.0.5` (boost.py needs nothing newer) and §5.1's "the floor becomes the release
  holding services" applies when that release exists.
- Gates: frozen m48/m49 + m69b packing/surface green; the m69b server files fail by construction
  (no served process yet); rest of `tests/frozen`+`tests/extra` green; ruff, format, mypy clean;
  sphinx -W ok; precommit --fast ok. Extra-test mutants (8) all killed.
