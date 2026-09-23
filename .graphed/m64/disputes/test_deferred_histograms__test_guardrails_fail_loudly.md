# Test Dispute — `tests/frozen/m23/test_deferred_histograms.py::test_guardrails_fail_loudly`

Filed by: m64 (histogram-growth) plan stage, before test authoring. Nothing has been edited,
skipped or weakened; the anchor is green at HEAD.

**Status:** RESOLVED — owner ruling 2026-09-23

> (1) D7 APPROVED: move the frozen m23 growth-refusal witness to a `Variable(..., growth=True)`
> axis (Variable growth stays refused with the same message) and take growth off the repo
> CLAUDE.md Phase-2 list; the dispute file gets a `**Status:** RESOLVED — owner ruling
> 2026-09-23` block quoting this ruling, so the test-author and implementer can act on it.
> (2) D5 APPROVED: the growing-Regular promises are P4–P6 as narrowed; the brief's 'bit-for-bit'
> binds Int64 counts and, for Regular growth, the bin count and logical offset for any
> partitioning, with edges bit-for-bit on dyadic grids only. (3) D6 APPROVED AS WIDENED by the r5
> cut: several fill NODES (several calls, or one axis-mode fill whose value varies) give
> partition-major category order, deterministic for a given partitioning; a single fill node
> keeps exact first-appearance-in-dataset order on every runner and partitioning (P1). Node-major
> reconstruction from per-node first-appearance lists carried by the partial is recorded under
> 'Follow-ups (reported, not built)' as the way to restore call-major order later; it is NOT
> built in m64. (4) APPLY the r5 cut exactly as the review states it, once, in plan.md: the
> reference is per-partition, per-node eager fills combined in partition order by
> add_histograms; the closed list of differences from one whole-dataset eager fill is (a)
> partition-major order with several fill nodes, (b) a value on a bin edge (P5), (c) a
> non-finite value before a growth event of its axis (P6), (d) float bits (P3); P2 points at (b)
> and (c); the promise set gains design.rst's 'The two modes agree…' paragraph, which B2
> rewrites as per-category, per-label agreement; plan-B C13 gains the axis-mode member; plan-C
> C2's closing check (b) requires the non-finite/flow clause wherever an Int64 promise is
> stated; the dispute file's D5 and D6 bullets are corrected to (a)–(c). (5) Unit A proceeds; B
> and C proceed under this ruling — the lane no longer waits.

In the loop-2 plan (`plan.md`), "D7" here is R1 (plan steps T2 and S5), "D5"/"D6" are R2 and the
deviations (a)–(d) below; the `Int64` promise carries (b) and (c).

## The test

One of its seven guardrail assertions:

```python
with pytest.raises(TypeError, match="growth"):
    gh.spec_of(bh.Histogram(bh.axis.IntCategory([], growth=True)))
```

## The clause it contradicts

m64's brief (the growth-axes item the owner pulled from Phase 2 into scope on 2026-09-23):
`StrCategory([], growth=True)`, `IntCategory([], growth=True)` and `Regular(..., growth=True)` must
fill over a partitioned source and combine to the eager result. A deferred `Histogram` computes
`spec_of(self)` in `__init__`, so this assertion and the brief cannot both hold: any implementation
of the brief makes this line fail, and any code that keeps it green refuses the axis the brief
requires.

## Measurement

The m64 probe copy (plan D1 + D2 + D4: growth encoded in the spec, the codec refusal, the union
merge in `add_histograms`; `lanes/histogram/probes/probe-p5.patch`) against the whole current suite:

```
PYTHONPATH=<p5> python -m pytest tests/frozen tests/extra \
    -o "pythonpath=<p5> tests/frozen/m23 tests/_corpus" --junit-xml=p5.xml
-> 211 tests, 1 failure, 1 skip; the failure is
   test_deferred_histograms::test_guardrails_fail_loudly — Failed: DID NOT RAISE TypeError
```

No other frozen test fails.

## Proposed correction

Keep the guardrail's intent (an axis the spec codec cannot carry fails loudly at spec time, naming
the option it would lose) and move its witness to a growth axis m64 still refuses:

```python
with pytest.raises(TypeError, match="growth"):
    gh.spec_of(bh.Histogram(bh.axis.Variable([0.0, 0.5, 1.0], growth=True)))
```

At HEAD a growth `Variable` axis is silently accepted with its growth dropped. m64's spec step
refuses it, so this replacement is red at HEAD and green after m64: it discriminates.
The m64 test-author applies the correction under the ruling (`--allow-refreeze tests/frozen/m23`),
as its own commit.

## Also for the same ruling

The m64 plan is `graphed-workdir/lanes/histogram/plan.md`. Its section "Owner ruling" names two
parts, R1 and R2.

- **R1.** This dispute's correction (plan step T2). In the same ruling, the repo `CLAUDE.md`
  "Phase 2 (do NOT build)" line drops "growth axes" (plan step S5).
- **R2.** The meaning of the brief's "the same histogram an eager `hist` fill produces". The
  plan's section "Reference and deviations (D5)" states the reference: per-partition eager fills,
  one per fill node, combined left to right in partition order by `gh.add_histograms`. It also
  lists the closed set of ways the reference differs from one whole-dataset eager fill:
  - (a) several fill nodes feeding one histogram give partition-major category order;
  - (b) a value on, or within float rounding of, a growing `Regular` bin edge follows its chunk's
    growth history, both its bin and, when it is an extreme, the extent;
  - (c) a non-finite value followed in the same eager fill by a growing value is moved by boost
    into a regular bin;
  - (d) float bits of inexact sums, and of growing-`Regular` edges off dyadic grids, follow the
    runner's fold shape.

  Everything else is bit for bit (`Int64` counts outside (b) and (c)). Call-major order for
  several fill nodes is the plan's F2, reported and not built.
