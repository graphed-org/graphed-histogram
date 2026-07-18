# Contributing to graphed-histogram

Part of the `graphed` project, governed by the gated three-role pipeline. The root
[`graphed-project/CLAUDE.md`](https://github.com/graphed-org/graphed-project-mvp) and the project plan
are authoritative; the plan always wins.

## Guardrails (M8)

- **Local filesystem store only** (no distributed store in MVP); **single machine**.
- M8 is checkpoint/resume — analysis **preservation** is M9, not here.
- The canonical durable form is the **serializable IR** (`graphed_core.DurablePlan`), never
  cloudpickle except for genuinely opaque callables (flagged `opaque=True`).
- Resume must be correct under interruption: **no double-count, no lost partition**; a resumed run
  matches an uninterrupted one bit-for-bit. `task_id` must stay content-addressed (cache-poisoning-safe).

## Integrity rules — NON-NEGOTIABLE (plan A.7 / B.6)

Never edit/skip/weaken `tests/frozen/**`; never lower a threshold or relax CI; never stub the thing
under test. Dispute a frozen test via `.graphed/<Mx>/disputes/<test_id>.md`.

## Local gates

```bash
pip install "graphed[awkward,numpy] @ git+https://github.com/graphed-org/graphed@main"   # needs Rust
pip install "graphed-executors @ git+https://github.com/graphed-org/graphed-executors@main"
pip install -e ".[dev,docs]"
ruff check . && ruff format --check . && mypy
pytest tests/frozen --cov=graphed_histogram --cov-branch
sphinx-build -W -b html docs docs/_build/html
```
