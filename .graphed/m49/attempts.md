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

## Fix cycle 1 — review wf_fa3d406b (A-2, A-9, A-10; A-1 disputed)

**A-2 (MED)** — the member §7.2's refusal admits: two fills that intern to ONE record node left
`marked == outputs`, nothing refused, and `_SumFills` ITERATED the evaluated values, answering at
half strength (`[0, 4, 12, 8, 17.6]` against `[0, 8, 24, 16, 35.2]`, the adjudicator's 0.500). The
repair is the group builder's own shape rather than a wider refusal: `Histogram.plan` builds the
same `rank` map `plan()` slices with and hands `_SumFills` the per-staged-fill OUTPUT INDICES, so a
repeated index replicates — which is what filling twice means, and is the supported record-time
dedup path §7.2 names. `marked` still comes from that map, so the optimizer-merge refusal (distinct
ids merged) is unchanged and still fires before any mis-index. Witness
`tests/extra/m49/test_duplicate_fill_multiplicity.py`: RED at 0.5x before, green after, with the
one-node instrument asserted and the non-interning pair (`w` vs `w2`) as the control against
"multiply by the staged count".

**A-9 (NIT)** — the `if key is not None` guard in `_variation_labels` is gone; every node in a
marked fill's cone survives DCE, so `node_map[nid]` is total there and a miss should say so.

**A-10 (NIT)** — `graphed.by_label.cone` is no longer imported across the repo boundary (it is not
on `graphed`'s exported surface, so nothing in graphed's frozen suite holds its spelling for us).
Chose the plan's own spelling over exporting it: a four-line local `_cone` over `session.walk`,
which §8.2(i) names as the producer's walk and which `cone` was a wrapper for.

**A-1 (HIGH, graphed-side)** — the repair reds this repo's `tests/frozen/m49/test_blame_parity.py`
(4 cases). Dispute filed at `.graphed/m49/disputes/test_blame_parity.md`; the graphed change was
not shipped and nothing here was weakened.

Gates: `pytest tests/frozen` 154 passed / 0 failed, coverage 96.16% (fail_under 90); whole `tests/`
tree green; `graphed-exec-local tests/frozen/m49` 31 passed (read-only cross-check); ruff check +
format clean; `mypy` (strict, `files = ["src"]`) clean; `precommit . --fast` ok with
`workflows-valid` now LIVE.

## post-1b correction (re-review RR-1)
The cycle-1 note "the graphed change was not shipped" is superseded: fix cycle 1b (graphed
4364beb) shipped the A-1 repair under the adjudicated plan-r42 §8.2(ii) carve-out; the dispute
in .graphed/m49/disputes/test_blame_parity.md is CLOSED and the four blame-parity anchors are
green with the external arm dispatched.
