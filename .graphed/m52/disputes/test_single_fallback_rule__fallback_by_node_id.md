# Dispute: `test_no_second_implementation_of_the_fallback_rule_exists` pins the fallback by object identity

**Test.** `tests/frozen/m52/test_single_fallback_rule.py::test_no_second_implementation_of_the_fallback_rule_exists`, its behavioral leg:
`graphed.member_of(varied, "no_such_label") is graphed.nominal(varied)`, plus the `is not` for `jer_up` and the observable's `is`.

**What changed.** graphed#231 (graphed#298, merged as b1b1380f) makes every universe reader return a proxy of its member node, not the member object itself: `member_of`, `nominal`, `universe`. Assigning into the returned value then leaves the container as it was. Two reads of the same member are equal by node but are no longer the same Python object. So `is` now fails for the fallback, and `is not` passes whatever the fallback does.

**Spec clause.** The m52 C5 intent (README row "vary-m52-C5 5.5") is that the missing label falls back to the nominal member through the one fallback rule. The member is the node. Object identity was only how the pin was written before #231.

**Proposed correction (applied in this branch).** Compare `.node_id`: the missing label's member is the nominal node, and `jer_up`'s is not. The same re-aim was made for graphed's frozen m49/m52 reader pins in #298 (`freeze-m49-fixup5`, `freeze-m52-fixup3`).

**Resolution.** Closed. team-lead, 2026-11-20: "It is covered by 'refreezes are authorized' (scope: the frozen edits these issues' decisions require)". This is the same rewrite #231's ruling authorized in graphed, caused by #298.
