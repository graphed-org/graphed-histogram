# m50 axis-mode — implementer attempts log

## Contents
- [H1 — axis-mode fill lowering](#h1)
- [H2 — axis slot + label listing](#h2)

<a id="h1"></a>
## H1 — axis-mode fill lowering (boost.py)

Target: §6.2 per-`fill()` axis mode. `FillEvaluator` gains a `variation` field (None = sibling,
unchanged; a label tuple = axis-mode loop). `fill(variation_axis=True)` declares a sorted non-growth
`"variation"` StrCategory axis at fill time (`_declare_axis_spec`), splits fold labels into `W`
(weight-only → one evaluator-loop node) and `S` (axis-value / `Varied` sample= → own sibling nodes),
records `1 + |S|` nodes each with a distinct `content_hash((spec, variation))` (`_axis_chash`). Four
refusals: mode mismatch, cross-fill label disagreement, user `"variation"` axis, and `.plan()` axis
arm. §6.1d seam recorded per weight COLUMN upstream (reuses `_guard` + `broadcast_like`).

Trap hit: store params must be scalar — encoded `variation` as `json.dumps(list(...))`.

Results after H1:
- tests/frozen/m50: 12 passed / 12 failed. The 12 green = all 6 H1-scoped anchors (cross-fill,
  mode-mismatch, user-axis-refused, user-axis-untouched, sample-only-sibling counts,
  distinct-evaluators) + `plan_refuses` + `axis_mode_no_variation` (bare-key coincidence, stays green
  under H2) + 4 result-shape (G1). The 12 red all route through `execute`/`unpack`/`label_listing`
  → await H2's axis slot.
- Whole tests/frozen flat run: 166 passed, 12 failed — ALL failures are m50 (no non-m50 regression).
- ruff clean; ruff format clean; mypy (project config, `disallow_subclassing_any=false`) clean.
- Determinism: 5 distinct axis-mode chashes, byte-identical across two runs; variation StrCategory
  stored order lexicographic (sorted) and identical across runs.
