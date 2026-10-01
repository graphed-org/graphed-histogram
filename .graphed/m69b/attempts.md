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

## Iteration 2 — 2026-09-30 — served process, fills, receipts, resolve, unpack

- `_Served(inner, slots, endpoints, handles)`: `bind_services` makes no RPC, merges the endpoints
  it already holds (a bound plan re-bound by `require_bound` with `{}` must not report its own
  servers unbound), refuses a missing server with `UnboundService` and a non-plaintext wire
  (`grpcs`/`http`/`https`) before dialling; `_Handles` creates every slot's no-flow stand-in
  `ChunkedHist` once, under a lock, on the first call or the first bound pickle (`__reduce__`); a
  worker's unpickled copy carries the ids. `__call__` sets the task scope (a `ContextVar`) around
  the inner process; `resolve_services`/`part_paths` forward.
- The pieces' reduce ships a backed slot's partial as one `FillMany` keyed `str(partition)`
  (`ALREADY_EXISTS` is done) and returns `Receipt(spec, endpoint, hist_id)`; outside a scope it
  raises naming `pieces.serve`. The empty value of a backed slot is the empty receipt; receipts add
  by identity. `resolve_services` snapshots with delete, `unpack` without. RPC errors become the
  picklable `HistservError(endpoint, code, details)` (fields through `args`); clients are cached per
  (pid, endpoint); every RPC carries a 600 s deadline.
- First run: a bound plan failed `require_bound` (the held-endpoints merge above); the snapshot
  placed chunks by the slot key's form, which a receipt does not carry — it now places a keyed
  chunk by the spec's last (variation) axis.
- Gates: frozen + extra all green on macOS (memory model skipped off Linux); the whole m69b frozen
  suite green on Linux arm64 (python:3.12-slim, graphed a0638719 built in the container), memory
  model included. Coverage: histserv.py 100 %, boost.py 98.9 %, _spec.py 95.9 % (min); diff-cover
  100 % vs origin/main. ruff, format, mypy clean. Extra-test mutants (5) killed, the lock mutant
  4/4 runs (8 creations against 1).
