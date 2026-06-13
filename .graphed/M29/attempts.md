# M29 attempts — graphed-histogram (multiple multiplicative weights)

## Iteration 0 — 2026-06-12 (freeze-M29-0)

- The M27 replay contract (preserve eval_histogram n_weights) had no producer. Now fill()
  accepts weight= as a SEQUENCE of graphed Arrays: each factor is a real graph input; params
  gain n_weights ONLY when >1 (single-weight node identity byte-for-byte unchanged — pinned);
  FillEvaluator gains n_weights (default 1: old pickles/evaluators valid) and multiplies the
  factors elementwise before filling — the package's OWN plan()/executor path agrees with the
  preserve replay by sharing the evaluator.
- frozen m29 (5): two-weight materialize == eager (values AND variances, Weight storage);
  the plan()/SequentialRunner path (per-partition fills: deterministic byte-identical across
  runs, allclose(rtol=1e-12) vs single-pass eager — float summation ORDER differs across
  partitions, an honest pin); three weights + the params contract (n_weights=3, 4 graph
  inputs); single-weight unchanged (no n_weights param); jagged weight factors flatten
  consistently. Non-vacuous: weight=[w1,w2] crashed recording pre-impl.
- Gates: 24 passed · coverage 95.71% · ruff/format/mypy/sphinx clean. Cross-repo: preserve
  frozen m30 pins the bundle replay; the full 11-repo + 3-fork sweep ran green pre-commit.

## Iteration 1 — 2026-06-12

- CI exposed a test-dependency gap invisible to the local cross-repo sweep: the frozen m29
  plan-path test builds its partitioned source with ak.to_parquet -> pyarrow, present locally
  but not in this repo's CI install. pyarrow added to the dev extra (the frozen test is
  untouched). Lesson: a green local sweep validates code, not CI environments — new frozen
  tests must declare their dependencies in the repo they land in.

## Iteration 2 — 2026-06-12 (freeze-M29-1)

- CI round 2: ak.to_parquet routes through pyarrow's PANDAS SHIM -> ModuleNotFoundError in CI
  (pandas exists locally only; this ecosystem is deliberately pandas-free). FREEZE AMENDMENT
  (sanctioned, dispute-correction path): the plan-path test's fixture now writes parquet via
  pure pyarrow (pq.write_table) — the assertions are byte-identical, only the fixture I/O
  changed. Re-frozen as freeze-M29-1.
- Iteration 1's pyarrow dev-dep edit ALSO shipped invalid TOML (a regex grabbed the wrong
  bracket; the tomllib check ran but its failure was swallowed by statement chaining). Both
  failure modes are now ENCODED in graphed-orchestrator's new pre-commit gate
  (python -m graphed_orchestrator.precommit, commit 4f0abf5), which gated THIS commit:
  toml-valid ok, integrity-scan REFREEZE:tests/frozen/m29/... (loud, sanctioned), full suite ok.

## Iteration 3 — 2026-06-12

- CI round 3: the pandas dependency is in awkward ITSELF (arrow form derivation calls
  pyarrow's to_pandas_dtype on every from_parquet schema read) — the fixture's write side was
  never the whole story. graphed-awkward's dev extra carries pyarrow+pandas for exactly this
  reason (the established precedent); this repo's dev extra now matches. No test changes.
