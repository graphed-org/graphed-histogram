# CLAUDE.md — graphed-histogram

Defers to the root **`graphed-project/CLAUDE.md`**; the **project plan
(`graphed-project-plan-gated.md`) always wins.** This file distills milestone **M23**
(P0.1 of the ADL-benchmarks port, user-confirmed 2026-06-10).

## What this repo is

**Deferred boost-histogram/hist filling on graphed task graphs** — the `dask-histogram`
analogue. A `.fill(...)` RECORDS instead of executing: each fill is an **External node** in the
graphed IR (the M3 correctionlib/ONNX family), carrying a `PayloadDescriptor` whose
`content_hash` is the SHA-256 of the **canonical axes/storage spec** (declarative params — never
cloudpickle; UHI in, UHI out, invent no formats). Backends know NOTHING about histograms: fills
record through `record_external(descriptor=, form=)` (graphed M23) and evaluate through
`evaluate_ir`'s `externals=` registry.

Aggregation rides the M7/M8 seam with graphed's OWN evaluation idiom (no `compute()` helper —
user-directed, 2026-06-11): `plan()` builds the `Plan(process=fill-partition-through-the-
compiled-IR, combine=histogram add, empty=zero-hist)`; an R7 executor's `run(plan).value` IS the
aggregated histogram; the reference `session.materialize(fill_node)` evaluates a fill eagerly.
Sources implementing `graphed.write.PartitionedSource` are filled partition by partition (their
whole-dataset loader is NEVER invoked); Int64 counts are exact under any combine tree, float
storages are deterministic per fixed-tree executor configuration.

## Surface (dask-histogram parity)

- `graphed_histogram.boost.Histogram` — deferred `boost_histogram.Histogram`: `.fill()` records
  and returns self (multiple fills accumulate); `.plan()` exports the task graph.
- `factory(*arrays, histref=, weight=, sample=)`.
- numpy-like `histogram` / `histogram2d` / `histogramdd`.
- All standard boost storages (combine is native `+`); axes Regular/Variable/Integer/
  IntCategory/StrCategory/Boolean. **Phase 2 (do NOT build):** growth axes, dask-style
  persist/delayed beyond Plan export.

## Hard rules

Frozen tests under `tests/frozen/m23/` — never weakened. One source family per histogram
(PartitionedSource or in-memory; mixtures rejected). Ragged fill values flatten at fill time.
The canonical spec encoding is VERSIONED and byte-stable (the content hash is identity).

Gates: ruff + ruff format · mypy --strict · sphinx -W · coverage (every source file >=90%
line+branch, never lowered; every PR >=98% diff coverage vs `main`).
Status: see `.graphed/state.json`.
