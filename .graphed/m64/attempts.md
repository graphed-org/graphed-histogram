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
