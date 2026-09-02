# m50 axis-mode — implementer attempts log

## Contents
- [H1 — axis-mode fill lowering](#h1)
- [H2 — axis slot + label listing](#h2)
- [D1 — docs (how variations work)](#d1)

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

<a id="h2"></a>
## H2 — axis slot + label listing (boost.py, __init__.py)

`_slots` gains an axis-mode arm: ONE `(name, None)` slot gathering every fill-node index, spec from
the fill node — checked BEFORE the sibling `len(labels)==1` bare-key path so an unvaried axis-mode
output is still keyed `(name, None)`. `label_listing(histograms)` returns `{name: list(_output_labels)}`
— §2.4 fold order, MODE-independent (both modes populate `_label_maps` identically), unvaried →
`["nominal"]`. Exported from `__init__`.

**Collision found + fixed (deviation from brief's "sibling hashing UNCHANGED"; flagged to reviewer).**
The four-output frozen test combines a WEIGHTED output (`sib`, Jet.pt) and an UNWEIGHTED output
(`plain`, MET.pt) of the SAME spec (Regular(20,0,200)+Weight). `evaluate_ir` resolves an External's
evaluator by `externals[content_hash]` alone, and `content_hash(self._spec)` — the pre-m48 sibling
key — is identical for both, so the merged registry resolves both distinct nodes to whichever
evaluator registered last: `sib` nominal silently evaluated UNWEIGHTED (232.16 → 1037.0). PRE-EXISTING
latent m48/m49 collision (reproduced on sibling-only histograms, no axis code involved); the m50
four-output plan is the first program to combine a weighted + unweighted same-spec output.

Fix (`_fill_chash`): the fill-node descriptor content hash folds the evaluator-distinguishing fields
that can differ WITHIN one spec — unweighted, weight-factor count, §6.2 variation — while the
CANONICAL single-weight sibling fill still reduces to `content_hash(spec)` verbatim, so the committed
m48 golden (`test_variation_goldens`, pinning the pre-m48 serialized IR of exactly that fill) is
byte-unchanged. `has_sample`/`n_axes` are excluded — both spec-borne (sample only on Mean/WeightedMean
storage; axis count in the spec), so neither collides within a spec. First attempt (folding
everything) broke the golden; the baseline-delta form restores it.

Results:
- tests/frozen/m50: 24/24 green. Whole tests/frozen flat run: 178/178 green (no regression; m48
  goldens intact).
- ruff clean; ruff format clean; mypy (project config) clean.
- Coverage: boost.py 97% line+branch from the frozen suite. Two new spots uncovered by the fixtures:
  the axis-mode multi-weight product and the seam-free axis branch (both degenerate for the frozen
  programs); the multi-weight-axis path verified bin-for-bin correct out-of-band (mirrors the tested
  sibling path).
- Determinism: slot keys, axis histogram bytes, label_listing, and the 5 mixed-program chashes all
  byte-identical across two runs.

<a id="d1"></a>
## D1 — docs (how variations work: axis-mode fills)
Changed:
- docs/design.rst: new "Systematic variations: the variation axis" section —
  `Histogram.fill(..., variation_axis=True)`, the `StrCategory("variation")` axis, the S/W split
  (weight labels collapse into the evaluator loop; shift/`Varied` sample= stay sibling fills at
  `1 + |S|` arity), the four refusals, recognition via `axis.__dict__["name"]=="variation"`,
  `graphed.labels/universe/nominal` over the result, and `graphed_histogram.label_listing`
  (mode-independent `{output:[labels]}` without executing).

Executed examples (docs-sweep rule; run against the shared venv, editable graphed_histogram @ H1/H2):
- axis-mode fill over a per-event varied weight → result axes `['Regular','StrCategory']`, axis
  name `'variation'`, stored bins `['nominal','sf_down','sf_up']` (sorted), `graphed.labels` =
  `('nominal','sf_down','sf_up')`, nominal slice `[3.1 3.2 0. 0.]`, sf_up slice
  `[3.72 3.84 0. 0.]` (exact reprs pasted).
- sibling mode → dict per label, `sib["nominal"]` bit-for-bit equal to the axis nominal slice.
- `label_listing({axis, sib, plain})` → `{'axis':['nominal','sf_up','sf_down'],
  'sib':[...same...], 'plain':['nominal']}` — axis entry == sibling entry (mode-independent).
- Update-updated the doc's numpy reprs to match `repr()` exactly after a first-pass spacing miss.

Update-added the Contents anchor `[D1 — docs](#d1)`.

Gates:
- sphinx -W (docs/ → docs/_build/html): EXIT 0, zero warnings (`grep -c WARNING` = 0).
- precommit --fast: toml/workflows/integrity/prek(ruff check + ruff format) → all ok.
- Docs-only change: pytest suite unaffected by an .rst edit (not re-run).
