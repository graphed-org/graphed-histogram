# Test dispute — m48 / H9

**Test:** `tests/frozen/m48/test_fill_reindexing.py::test_an_ancestor_value_is_re_indexed_per_label_by_that_labels_own_mask`
**Filed:** 2026-09-01, isolated implementer, `m48-vary` @ `freeze-m48` (40f9b48)
**Claim:** the fixture is unsatisfiable simultaneously with plan §6.1d and with a second frozen
m48 anchor. Exactly one of the two anchors can be green under any lowering.

## What the test requires

```python
session, root = in_memory_events()
sel = _derived(gnano.events(root))          # the context is built and DISCARDED
h.fill(root.MET.pt, sel.MET.pt, sample=root.MET.phi)   # "axis 0 and the sample are ANCESTORS"
```

and then asserts, per label, that axis 0 and the sample equal `EVENTS.MET.pt[_eager_mask(label)]`
— i.e. that `root.MET.pt` was re-indexed by each label's own derivation mask.

## Why it cannot hold

`root.MET.pt` carries **no context handle**: `gnano.events(root)` leaves its argument
context-free by construction (`python/graphed/awkward/gnano.py`: "`root` itself stays
context-free — only reads performed THROUGH the returned context carry its handle"), and the
handle is propagated only through `Session._wrap` from inputs that already carry one. Measured
against the frozen fixture (interpreter `/Users/lgray/vibe-coding/graphed/.venv/bin/python`,
`graphed @ 79a61a7`):

```
ctx(root.MET.pt)   = None
ctx(sel.MET.pt)    = EventContext(#2 ...)
graphed.reindex_to(root.MET.pt, ctx(sel.MET.pt)) is root.MET.pt   -> True   (identity)
rows: root.MET.pt = 400   vs   nominal sel.MET.pt = 234
```

§6.1d's seam is the bound one: `reindex_to` is *identity when `value` ... carries none*
(`python/graphed/accessors.py`), and no public verb stamps a context onto a loose value
(`accessors.with_context` is not exported). So a conforming fill leaves axis 0 at 400 rows
against axis 1's 234 and boost-histogram refuses on unequal axis lengths.

## The plan clause it contradicts

§6.1d (plan r33, lines 1385-1390):

> Context-free (loose) inputs alongside contexted ones adopt the unified context **for LABEL
> ALIGNMENT only; their row space is NOT adjusted** (a loose value carries no handle, so no
> intervening mask is known and no re-indexing is possible).

## The second frozen anchor it contradicts

`tests/frozen/m48/test_fill_flatten_refusals.py::test_a_loose_VALUE_at_the_wrong_row_space_gets_its_own_message`
freezes the *same shape* — a loose axis value beside a contexted operand at a different row
count — as an execution-time **refusal** naming `value[0]`, and its own comment states the rule:
"a loose input adopts the unified context for LABEL ALIGNMENT only — its row space is NOT
adjusted". Measured on that fixture: loose = 400 rows, factor = 172 rows; the only rule that
makes H9 green ("re-index a loose value from the context's root row space") makes both 172 and
silences H7's refusal. The two anchors are mutually exclusive.

## Proposed correction

The anchor's *intent* — §6.1d link kind (1), an **ancestor-context** value re-indexed per label —
is reachable with a one-line fixture change: keep the context and read the ancestor values
through it, instead of reading them off the raw source Array.

```python
    session, root = in_memory_events()
    events = gnano.events(root)
    sel = _derived(events)

    h = sampled_2d()
    h.fill(events.MET.pt, sel.MET.pt, sample=events.MET.phi)  # axis 0 and the sample are ANCESTORS
```

`events` is a genuine ancestor context of `sel`, so `graphed.reindex_to` composes the links below
it (a `vary` identity link, then the varied mask link) and yields each label's member re-indexed
by that label's own mask — exactly the row set `_eager_mask(label)` computes, leaving every
assertion in the test body unchanged.

## Status

No source was written to satisfy or to route around this test. The rest of the m48 lowering is
implemented per the plan (loose values keep their row space), so this anchor stays red pending
adjudication; the second H9 anchor (`test_a_projection_link_...`) is unaffected and green.
