# `tests/frozen/m49` — traceability

The `graphed-histogram` half of m49 (shift path + impact + executor end-to-end). Frozen from the
freeze tag: read-only, never edited, skipped, xfailed or weakened. A test that looks wrong is a
Test Dispute at `.graphed/m49/disputes/<test_id>.md`, never a repair in place.

Plan anchors are cited by SECTION. Every fixture is same-directory (`m49_hist_fixtures.py`); the
`m49_` prefix is the tree's helper-basename rule — `tests/frozen` is collected in ONE pytest
process and a bare `vary_hist_fixtures` would silently bind m48's module instead.

The corpus is the copy VENDORED under `tests/_corpus` (already on `pythonpath`). Nothing here is
`importorskip`-guarded: a skipped headline gate is a silently discharged milestone.

| File | Plan anchor | What it holds |
|---|---|---|
| `test_reference_matrix.py` | §10/m49 anchor (i), §5.2b, §8.2(i) | the full 15-reference corpus matrix through one mixed shift+weight program; the single-read witness bound to THAT run; run-to-run `array_equal` against a second independent build; the shipped closure's label channel over a real analysis plan |
| `test_variation_labels_payload.py` | §8.2(i), §7.2 (β) | `variation_labels` POPULATION at its only bound producer: the bound entry layout, the sort key, no `set`/`frozenset`, keys that name nodes of the reduced store shipped beside them, the multi-label key, the nominal-exclusion clause and its empty-tuple member, and the `None` rule's admitted member |
| `test_merge_shortfall.py` | §7.2 | the merge-shortfall refusal widened to its CLASS — the group builder on an UNVARIED program, and `Histogram.plan()`, where nothing raises today; both asserted at the BUILDER, with merge-free positive controls on each consumer |
| `test_jer_stochastic.py` | §5.5a, §5.5b | the JER-SF stochastic shift: one content-seeded draw interned once, pairwise-distinct counts with no ordering, bidirectional migration, run-to-run identity, and partition invariance at two `steps_per_file` values from a plan run |
| `test_fill_arity.py` | §2.4, §6.1b | `1 + \|S\| + \|W\|` sibling fill nodes on a mixed shift+weight fill, counted against the program's own S and W rather than a literal (§6.2's axis mode is `1 + \|S\|`, m50) |
| `test_shift_ordering.py` | §5.1 | `jes_up > nominal > jes_down`, scoped to the monotone JES fixture and asserted nowhere else, beside the cutflow-divergence witness that makes it a shift and not a re-weighting |
| `test_weight_factor_reindex.py` | §6.1d, §9.1 | an ancestor-context WEIGHT FACTOR re-indexed per label by that label's own mask, on both evaluation routes |
| `test_blame_parity.py` | §6.1d | the plan/executor path blames the same operand `Session.materialize` does — message equality plus the named coordinate, with a matching-factor control |
| `test_golden_side_strip.py` | §6.3 | the golden comparison strips each side of the version IT carries: the two-release pair that separates per-side stripping from one live-derived pattern |

## Fixture families (`m49_hist_fixtures.py`)

* `matrix_program()` — the corpus mixed program. JES varies the jets record BEFORE the pt cut, so
  each universe re-derives its own selection; the b-tag and photon SFs vary only the weight. §2.4
  alignment falls out: shift labels fill with the central weight as evaluated in their own
  universe, weight labels with nominal kinematics.
* `shared_node_program()` — §3.4's shape for §8.2(i): one derived node consumed by two NON-nominal
  universes and by neither the nominal one. Interning keys on input ids, so the sharing must sit
  upstream of the fork; §6.1b's count then guarantees no fill-node key is multi-label. It also
  returns an unvaried histogram recorded in the SAME session — the admitted member of the hook's
  `None` rule.
* `jer_program()` — §5.5's stochastic fixture. The draw is a pure elementwise function of each
  row's own content, recorded as ufuncs rather than an External: `graphed_histogram.plan` forwards
  only the histograms' own evaluators, so an External draw is unresolvable in a worker.

## Traps this tree is written around

* the corpus b-tag SF's scale multiplies each JET's factor INSIDE the product, not the product;
* the corpus rounds the observable to 6 decimals BEFORE the fill, re-expressed here as
  `np.rint(x * 1e6) / 1e6` through `Array.__array_ufunc__` — neither `gak` nor `graphed` exposes
  `rint` or `round(x, decimals)`;
* raw-view bit-identity against the references is NOT asserted; the driver-side rounding is what
  absorbs per-partition summation-order differences, so "bit-for-bit" is claimed run-to-run only;
* `Session.materialize` is partition-blind and cannot be the oracle for a partition-invariance
  witness (§5.5a);
* `unweighted=True` and `weight=` contradict each other, so the cutflow count histogram carries
  the shift labels alone.
