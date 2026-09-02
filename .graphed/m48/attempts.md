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

## Iteration 1 — 2026-09-01 — H-I2 group/plan surface

- `_GroupReduce`'s layout is now slot-keyed and carries per-slot output INDICES, not counts: the
  rank of each fill node's id in the DEDUPLICATED id list, which matches `evaluate_ir`'s
  one-value-per-distinct-output list element for element. Two labels whose members intern to one
  node therefore replicate off one evaluated fill instead of overrunning the value list.
- `plan()`'s value is §6.1c's flat slot keying — a bare output name for an output no variation
  reaches (what keeps the frozen m23 indexing idiom working), `(output, label)` for a varied one —
  and its declared type widened to `Plan[dict[str | tuple[str, str | None], bh.Histogram]]` whole,
  `_add_groups`/`_GroupZero` included. `unpack` reads the per-output shape off the KEY FORM alone.
- `Histogram.plan()` refuses a varied histogram — `_SumFills` would merge the universes into a
  plausible wrong answer — and points at the group API. The trigger is the merge hazard, so a
  single varied fill refuses too.
- §7.2's optimizer-merge shortfall is refused at the group-plan builder through
  `aggregate_plan(on_compiled=...)`, comparing distinct compiled outputs against distinct marked
  record ids, over varied programs only. Measured: **1** `compile_ir` call per `gh.plan({…})` call,
  varied or unvaried (`graphed.aggregate.compile_ir` counted through a wrapper) — §7.2's
  anti-quadratic rule holds because the guard rides the seam rather than compiling again.
- `tests/extra/m48/test_lowering_edges.py` covers what no frozen anchor reaches: the guard's scalar
  pass-through (with the mismatched-array leg as its discriminator), the accessor's several-fills
  refusal, `fill`'s operand type checks, and `unpack`'s `(output, None)` key form.
- Gates: **93/93 frozen** (66 m48 + 27 m23/m29) and 97/97 including `tests/extra`; coverage on the
  CI gate (`pytest tests/frozen --cov=graphed_histogram --cov-branch`) **96.16%**, `boost.py` 97%;
  `ruff check`, `ruff format --check`, `mypy --strict`, toml and the integrity scan all clean;
  determinism measured on a VARIED program — byte-identical `compile_ir` output and identical label
  order across `PYTHONHASHSEED` 1 and 424242, with `hash('graphed')` differing across the two, so
  the instrument was live. `sphinx -W` cannot run here: no sphinx in the interpreter (environmental).

## Iteration 2 — 2026-09-01 — review repairs (5 MED)

- **A1/A6/A7 (one edit at `_guard`)**: the guard's identity is now the blame COORDINATE
  (`ambient` / `weight[i]` / `value[0]`), carried in `params["blame"]` and hashed into the payload;
  the sentence stays evaluator-side and the descriptor records this package's real version. So the
  node is derivable from what the graph records (a preservation plugin can rebuild the evaluator),
  two offenders stay two nodes on the plan path, and rewording a diagnostic moves no bytes.
- **A2**: the loose branch fires only when `args[0]` is the loose one — the guard compares each
  factor against the fill's FIRST value, so a loose value at another axis position was never what
  the comparison was about, and blaming it named an innocent operand.
- **A4**: the merge refusal now attributes the shortfall by re-compiling per output (free: that
  path is about to raise) and names the histogram whose own fills merged, which in a mixed plan
  need not be a varied one; the `variations=` workaround is offered only when a named output is
  varied. No single output shrinking means the merge crossed two, and all are named.
- **A8**: the layout's third element reads the FILL's recorded spec (`_fill_specs`), not the
  histogram object's. **No discriminating witness is possible at m48** — the two are equal until
  m50's §6.2 fill-time axis declaration makes them diverge — so none was written; mutating the read
  back to `hist._spec` leaves all 106 tests green, measured.
- **A5/A9 + the A6/A7 witnesses** live in `tests/extra/m48/test_review_witnesses.py`. The compile
  spy patches every module binding of `graphed.execute.compile_ir`, since it is imported by name in
  several modules and patching one re-export would miss calls made through another.
- Mutation table (each mutation applied to `boost.py`, target test run, source restored):
  hash on the prose → A7 witness FAILS; guard identity collapsed → A6 plan-path parity FAILS;
  loose-on-any-arg → A2 witness FAILS; blame-only-varied-outputs → A4 witness FAILS;
  `factors = list(weights)` → A5 witness FAILS; a second `compile_ir` in `plan()` → A9 witness
  FAILS; `_fill_specs[0]` → `_spec` → nothing fails (A8, as stated above).
- Gates: **93/93 frozen** (66 m48), 106/106 with `tests/extra`; coverage 96.25% (`boost.py` 97%);
  `graphed_orchestrator.precommit .` **ok** end to end — sphinx -W now green, the lead having
  installed it. Determinism re-measured after the identity change: byte-identical IR and label
  order across `PYTHONHASHSEED` 1/424242 with `hash('graphed')` differing.
