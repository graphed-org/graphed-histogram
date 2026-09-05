# m52 graphed-histogram implementer — iteration log

Freeze: `freeze-m52`. Scope C5. `git diff freeze-m52 -- tests/frozen/` MUST stay empty for the life
of the milestone.

## Iteration 0 — baseline (origin/main, no m52 implementation)
- tests/frozen/m52: 8 failed / 1 passed. Whole tree collects in ONE process: 8 failed / 179 passed
  (every pre-existing frozen test green). Failure set byte-identical across PYTHONHASHSEED 1/7/424242.
Baseline reasons match the (adjudicated) decomposition §5 table: 5.1/5.2/5.3 fail at fixture
construction (`GraphedError: ...got dict`, `points=` absent), 5.4 `AttributeError: no 'points'`, and
only 5.5 fails as an ASSERTION (the G8 rule-grep hits boost.py). The "5.1 gets the JES-nominal member"
reason is the C1-C4-landed/C5-absent intermediate, not origin/main.

## C5 scope note (measured, both test-authors) — read before implementing
Deleting `_member` + routing its seven call sites through `graphed.member_of` fixes SIBLING mode but
NOT axis mode: the axis-mode loop node groups labels under ONE (nominal) axis input, so the joint
label stays on nominal kinematics and row 5.3 (`test_joint_axis_mode_parity`) reds on
`btag_jesup_hf_up` alone. C5 must ALSO make the axis-mode loop group by *resolved* member (or let the
joint label leave the loop). See design §5.1 / decomposition §4-C5.

## Iteration 0b — implementer baseline (freeze + graphed C1–C4 at 9940a5f)


`pytest tests/frozen` → 2 failed / 185 passed.

* `m52/test_joint_fill_resolution.py::test_the_joint_labels_axis_value_member_is_the_shifted_one`
  — the joint label's axis input is the NOMINAL observable member.
* `m52/test_single_fallback_rule.py::test_no_second_implementation_of_the_fallback_rule_exists`
  — one hit, `boost.py:174` (`_member`).

Rows 5.2/5.3/5.4/5.5-controls already green: C1–C4 make the WEIGHT-side member point-aware, so the
joint universe already differed from the b-tag-only one on both fill modes; only the value/`sample=`
side was stuck at nominal, identically in both modes, which is why parity held while both were wrong.

## Iteration 1 — C5

`src/graphed_histogram/boost.py` only:

1. deleted `_member` and routed its seven call sites through the public `graphed.member_of`
   (behaviour-identical pre-C3, point-aware after it);
2. `_record_axis_fill` now groups the fold labels by the **resolved value/`sample=` members** they
   share instead of by label membership on those operands. The nominal group is the old `W` collapse;
   a label whose point names a shifted axis coordinate its own name does not now leaves that loop and
   joins (or forms) the group carrying its own kinematics.

`pytest tests/frozen` → 187 passed. `pytest tests` → 249 passed.

### Gates
* frozen diff vs `freeze-m52`: empty.
* `ruff check .` / `ruff format --check src tests` / `mypy` (configured, `--strict` over `src`): clean.
* coverage (repo gate, `fail_under = 90`): `boost.py` 99%, total 98.36%. Frozen-suite-only run:
  `boost.py` 98%, total 97.04% — no new line depends on `tests/extra`.
* determinism: `compile_ir` bytes over the joint program are identical across `PYTHONHASHSEED`
  1 / 7 / 12345 in both fill modes.

### Witnesses
* axis-mode grouping, joint program (3 nodes, unchanged arity):
  `nominal|btag_hf_up|btag_hf_down -> axis input 39` (nominal HT),
  `jes_up|btag_jesup_hf_up -> 40` (the `jes_up` member), `jes_down -> 41`.
* mutation probe — restoring the pre-C5 label-membership grouping (member_of swap kept) reds exactly
  `m52/test_joint_axis_mode_parity.py::test_both_fill_modes_resolve_the_joint_label_alike` and
  nothing else, so the grouping change is necessary and m50's `1 + |S|` arity is preserved by both
  rules.

## Iteration (REVIEW repair) — R1-B2
REVIEW R1-B2: boost.Histogram.fill()'s public docstring still stated the pre-C5 axis-mode rule
("weight-only labels collapse ... shift/sample= stay sibling"), which C5 falsified (grouping is now
by RESOLVED member; a weight-only joint label leaves the nominal loop). The two sibling docstrings
were already correct; this user-facing one (rendered into the autosummary API ref) was missed.
Rewrote it to match. Comment-only: tests/frozen -q 187 passed, frozen diff vs freeze-m52 empty.
