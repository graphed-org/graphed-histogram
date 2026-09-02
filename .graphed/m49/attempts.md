# m49 — implementer iterations (`graphed-histogram`, C4)

## Iteration 1 — the `variation_labels` producer + the widened §7.2 refusal

Changed: `src/graphed_histogram/boost.py` only (`+88/−33`).

- `_variation_labels(items, compiled)` — per non-nominal label of each output, the WHOLE record
  cone (`graphed.by_label.cone`) of that label's marked fill nodes (§2.4's `per_label.get(label,
  per_label["nominal"])` fallback), folded through `compiled.correspondence.node_map` and unioned
  per key. `"nominal"` is excluded, so a key no varied universe reaches keeps an empty tuple.
  The entry enumeration and the frames both come from `correspondence.frames`, which is already
  one entry per key of the map's image in §8.2(i)'s bound order — nothing here re-derives either.
  Returns `None` when no key carries a label.
- `_merge_guard` → `_refuse_shortfall(items, marked, compiled)`, now on EVERY program and on both
  consumers (`plan()` and `Histogram.plan()`); the message drops the label list where the only
  label is `nominal`.
- `_on_compiled(items, marked)` wraps the two: refuse first, then produce.

Result: `pytest tests/frozen` 154 passed / 0 failed (13 red at `ddcf48e`, all m49). ruff, ruff
format, `mypy --strict`, coverage 96% (fail_under 90), sphinx `-W` and the integrity scan clean;
`precommit .` full: ok. `workflows-valid` reports `-- pyyaml not installed` (tool resolution, not
a finding) — the same skip C1–C3 recorded.

R0.11 (import-site spy on `graphed.aggregate.compile_ir` + `graphed_histogram.boost.compile_ir`):
one `gh.plan({…})` = **1** compile on both a varied toy program and the 15-reference matrix; the
definition-site spy on `graphed.execute.compile_ir` sees **0**, confirming §7.2's claim that it
measures nothing. The refusal path is 2 (one re-compile per output, on a path about to raise).
