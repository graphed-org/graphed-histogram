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
