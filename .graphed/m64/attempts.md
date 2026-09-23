# m64 graphed-histogram implementer — iteration log

Freeze: `freeze-m64` (+ `freeze-m23-m64` for the ruled m23 witness). `git diff freeze-m64 -- tests/frozen/`
stays empty for the life of the milestone.

## Iteration 0 — baseline (ffe6342, no m64 code)
- `pytest tests/frozen tests/extra`: 221 pass, 124 fail, 1 skip. Red = 123 m64 clauses + the ruled m23
  guardrail (growth `Variable` DID NOT RAISE), as TEST_SANITY recorded.

## S1 — floors
- boost-histogram>=1.4.1 (runtime), hist>=2.12 and uhi>=1.0 (dev) per plan S1. No test changes state.

## S2 — the codec refuses what it cannot carry (D2)
- `_axis_spec` = `_encode_axis` + a decode comparison of transform and traits; the two category-only
  growth raises go. Suite: 228 pass / 117 fail; newly red none; G4 and the m23 witness turn green.

## S3 — growth on category and Integer axes (D1)
- `"growth": true` on growing IntCategory/StrCategory/Integer entries, version 2 when any axis grows,
  `zero_of` reads 1 and 2. Suite: 293 pass / 52 fail, newly red none; G5, G6, G13, G14 and G11's
  category legs green. Left red: every growing-Regular clause (still refused by D2 at this tip).

## S4 — growing Regular merges by a bin-aligned union (D4)
- `add_histograms` widens each admitted growing-Regular pair (`_union`: both growing Regular, equal
  `vars`, unequal, offsets whole bins within 1e-6) and re-bins both operands (`_onto`, flow to flow,
  `vars(h)` carried), then native `+`; `_SumFills`, `_GroupReduce`, `_add_groups` route through it;
  Regular entries carry growth. Suite: 345 pass / 0 fail / 1 skip (pre-existing: extra m48
  correctionlib absent from the lane venv). `git diff freeze-m64 -- tests/frozen/` empty.
