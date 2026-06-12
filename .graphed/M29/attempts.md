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
