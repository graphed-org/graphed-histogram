# Test Dispute — `tests/frozen/m23/test_process_executor_witnesses.py::test_default_plan_fails_loudly_when_behaviors_would_be_lost`

Status: CLOSED — owner ruling 2026-10-29: approved; #123 ships the behavior by value. Re-aimed in `test(frozen): m23 default plan carries behaviors; a bare worker backend fails loudly`, tags freeze-m23-fixup2 and freeze-m69b-fixup4.

## The test

It asserts that `ProcessExecutor(max_workers=2).run(h.plan(steps_per_file=2))` raises when the Session's backend carries an awkward behavior (`behavior_toy.BEHAVIOR`), on the premise that the default worker backend is constructed bare.

## The change it contradicts

graphed#123 (`fix(plans): workers build the backend with the Session backend's behavior`, in graphed 0.0.10) makes the default worker backend `graphed.aggregate.worker_backend`, which ships an `AwkwardBackend`'s behavior by value. The default plan now evaluates the behavior property on workers and fills the right histogram, so the test fails with `DID NOT RAISE` against graphed 31ca47a (graphed-histogram#41, run 40942937541).

## Correction

- The guard stays: a worker backend that really lacks the behavior (`backend="graphed.awkward:AwkwardBackend"`, constructed bare) must still raise on `mag|field|behavior`.
- The default plan must succeed and equal the eager `boost_histogram` fill of `hypot(x, y)`.
- The test is renamed `test_default_plan_carries_behaviors_and_a_bare_backend_fails_loudly`, since the default plan no longer fails; the module docstring says the same.
- m69b digest repinned as the recorded consequence of the approved m23 refreeze: `tests/frozen/m69b/test_histserv_surface.py` `FROZEN_BEFORE_M69B["m23"]` becomes the re-aimed directory's digest; no other entry changes.
