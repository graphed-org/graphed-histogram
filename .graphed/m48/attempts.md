# m48 attempts — graphed-histogram (`vary` weight path, the fill-shaped half)

Frozen suite: `tests/frozen/m48/**` (66 items). Baseline at `freeze-m48` (40f9b48): **16 passed /
50 failed** — the H8 divergence sentinels and the H12 goldens are the green-on-HEAD ones.

## Iteration 0 — 2026-09-01 — H-I1 fill lowering

- `Histogram.fill` rewritten as §6.1d's lowering: unify the input context handles (axis values,
  explicit factors AND `sample=`, which nothing upstream checks) before anything is staged;
  auto-apply the winning context's ambient weight; re-index every ancestor-context input into that
  context's row space through `graphed.reindex_to`; fold labels LEFT in the bound operand order
  (axes → ambient → explicit factors → `sample=`) and emit one sibling fill node per label.
  `unweighted=True` suppresses both weight sources and, applying no factor, contributes none of
  their labels; `weight=` beside it is a record-time refusal naming both.
- §6.1d's broadcast seam records through `graphed.broadcast_like` for the ambient factor and each
  explicit one independently, under the §6.3(2) disjunction trigger (a context handle OR any
  `Varied` input) — so the golden's unvaried, uncontexted fill still records ONE node.
- The three length messages needed a raiser that knows which operand is at fault, which the
  backend seam cannot know: `_WeightGuard` is a tiny External recorded UPSTREAM of the seam (the
  seam consumes its output, so it runs first) carrying the record-time blame — the ambient factor,
  the offending `weight=[i]` by position, or a loose VALUE by position when one is present, since
  a loose value's row space is never adjusted and "pass the value unflattened" would be wrong
  advice there.
- `fill_nodes_by_label(h)` is the §9.1 accessor over one fill call's siblings; a histogram with
  several fill calls has no single label→node answer and refuses rather than hiding one.
- **Dispute filed and UPHELD**: H9's `test_an_ancestor_value_..._by_that_labels_own_mask` read its
  ancestor operands off the raw source Array, which is context-FREE (`gnano.events` leaves its
  argument alone), so no re-indexing applies to them and H7's loose-VALUE refusal freezes exactly
  that shape. Re-frozen at `freeze-m48-fixup` (90c5d6d) reading them through the context; no source
  was written for or around the pre-fixup wording.
- Gates: frozen m48 **42 passed / 24 failed** (all 24 are H-I2's `plan()`/`unpack` surface, absent
  by construction at this commit); the rest of `tests/frozen` green with no regression; `ruff
  check`/`ruff format --check`/`mypy --strict` clean.
