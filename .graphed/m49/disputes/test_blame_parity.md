# Test Dispute — `tests/frozen/m49/test_blame_parity.py` vs review finding A-1

Filed by: m49 fix-cycle-1 implementer. Status: **OPEN — A-1 NOT implemented**, code left at
freeze-m49 behaviour. §A.7: not routed around, not weakened, no `xfail`.

## The test

`tests/frozen/m49/test_blame_parity.py` (frozen at `freeze-m49`, commit ddcf48e), four cases:

* `test_the_plan_path_reports_the_same_message_materialize_reports[0|1]` —
  `assert str(excinfo.value) == from_materialize["nominal"]` inside `pytest.raises(GraphedError)`,
  around `SequentialRunner().run(gh.plan({...}))`;
* `test_the_plan_path_names_the_offending_factor_and_not_its_neighbour[0|1]` — same
  `pytest.raises(GraphedError)` wrapper.

The failing node on the plan path is the §6.1d row-space guard, which is an **External** node
(`Histogram._guard` -> `session.record_external`), and the program is varied, so
`_variation_labels` carries an entry for its correspondence key.

## The clause it contradicts

Review finding A-1 (HIGH), from plan §8.2(ii)/(iii): `evaluate_ir`'s `external` arm sits in "the
top-level node loop", one of the "two dispatch points" the attribution hook binds to, and §4.1
makes an External payload the canonical carrier of a weight variation. Under that reading an
External failure with an entry MUST become a `StageError` carrying the label and the user's frame.

The frozen anchor requires the opposite for the same node class: the plan path must re-raise the
evaluator's own `GraphedError` **verbatim** (string equality with `session.materialize`'s message),
and `StageError` is a bare `Exception`, not a `GraphedError`, so `pytest.raises(GraphedError)` does
not even catch it. The two cannot both hold; §8.2(iii)'s parenthetical is also readable as naming
the two `backend.eval_stage` call sites, which is how the suite was frozen.

## Measurement

Same command, same fixture, only `python/graphed/execute.py` differs (graphed `m49-vary`,
shared venv):

```
# external arm routed through _dispatch (A-1's repair)
pytest tests/frozen/m49/test_blame_parity.py -q   -> EXIT=1, 4 failed
  E graphed.debug.errors.StageError: StageError in op 'external:histogram.weight_guard'
    at .../test_blame_parity.py:53 (partition ..., opt_level=1):
    GraphedError: weight[0] is not at this fill's row space: ...
# external arm unwrapped (freeze-m49 behaviour, what is committed)
pytest tests/frozen/m49/test_blame_parity.py -q   -> EXIT=0, 5 passed
```

`graphed`'s own frozen tree does not discriminate: `COV=1 ./scripts/run-tests.sh` is green with the
repair in place, as is `graphed-executors tests/frozen/m49`. This anchor is the only frozen
evidence either way, and it favours the un-attributed arm.

## Proposed correction

Either

1. amend this anchor at m50 to compare the *cause* rather than the rendered string — e.g. accept a
   `StageError` whose `cause_message` equals `from_materialize["nominal"]` (and keep the
   `weight[i]` / not-`weight[1-i]` discrimination on that field), alongside the m50 External
   attribution anchor the ledger already carries; **or**
2. rule that §8.2(iii)'s "two dispatch points" are the two `backend.eval_stage` sites, close A-1 as
   not-a-defect, and say so in §8.2 so the next reader does not re-open it.

The witness A-1 asks for, ready to freeze under (1) — it passed against the repair:

```python
# an External evaluator that raises, compiled to a boundary External node
with pytest.raises(ValueError, match=rf"jes_up at \({output}, None\)"):
    evaluate_ir(compiled, backend, sources, externals={CHASH: _boom}, on_failure=with_an_entry)
assert seen == [((output, None), "external:probe")]   # the arm reached the hook
# and the negative half: a hook returning None re-raises the original untouched
with pytest.raises(Boom):
    evaluate_ir(compiled, backend, sources, externals={CHASH: _boom}, on_failure=lambda *_: None)
```

(An External node carries no `name` in the IR — its identity is the descriptor — so an attributed
External failure has to name the payload `kind`; that spelling is part of what (1) would freeze.)
