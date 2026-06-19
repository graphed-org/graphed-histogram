# graphed-histogram

Deferred [boost-histogram](https://github.com/scikit-hep/boost-histogram) /
[hist](https://github.com/scikit-hep/hist) filling on [graphed](https://github.com/graphed-org)
task graphs — the [dask-histogram](https://github.com/dask-contrib/dask-histogram) analogue, built
on graphed's own evaluation idiom (milestone **M23**; P0.1 of the ADL-benchmarks port).

A `.fill(...)` **records** instead of executing. Each fill becomes an **External node** in the
graphed IR (the same M3 family as correctionlib and ONNX nodes): a call into foreign machinery,
carried in the IR with reproducibility metadata, evaluated later by a registered evaluator.
Backends know nothing about histograms — fills record through the frontend's
`record_external(descriptor=, form=)` seam and resolve through `evaluate_ir`'s `externals=`
registry.

## The deferred histogram in one example

```python
import boost_histogram as bh
import graphed_histogram as gh
from graphed_core.execution import SequentialRunner

h = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 10.0), storage=bh.storage.Int64())
h.fill(x)                 # x is a graphed Array: RECORDS a fill node, returns h
h.fill(x * 0.5 + 1.0)     # fills accumulate — more nodes, same histogram

plan   = h.plan(steps_per_file=4)            # the deferred task graph
result = SequentialRunner().run(plan).value  # a CONCRETE boost histogram
# any R7 executor accepts the same plan:
#   ProcessExecutor(max_workers=4, persistent=True).run(plan).value
```

The eager boost API stays available on `h` (axes, storage, views of the empty state); what
changes is that filling stages graph nodes and evaluation belongs to executors.

## Why it is built this way

- **Fills are External nodes.** The package supplies a `PayloadDescriptor`
  (`kind="histogram"`, `content_hash=sha256(spec)`, `io_schema="uhi"`) and an opaque histogram
  form; the backend is never consulted. Nothing in graphed-core, graphed, or any backend mentions
  histograms.
- **The canonical spec is the identity.** A histogram's identity is the SHA-256 of its
  **canonical, versioned axes/storage spec** — key-sorted JSON covering every supported axis and
  storage (declarative params, never cloudpickle; UHI in, UHI out, no invented formats). Identical
  fills intern to one graph node; the spec string is the fill's preservation payload; a plan
  re-run on another machine resolves its evaluator by the same hash. `spec_of(h)` reads it;
  `zero_of(spec)` rebuilds the empty histogram anywhere.
- **Aggregation is plans and executors, not `compute()`.** There is deliberately no `compute()`
  method — evaluation is graphed's machinery. `h.plan(...)` builds a
  `Plan(process=fill-partition-through-the-compiled-IR, combine=native +, empty=zero)`, and any R7
  executor's `run(plan).value` **is** the aggregated histogram. Histograms form a monoid under
  native `+` for every standard storage, so the executor's fixed combine tree applies unchanged:
  Int64 counts are exact under any tree, float storages are deterministic per fixed-tree executor
  configuration. The reference path for in-memory sources is `session.materialize(fill_node)`.

## Public surface

| | |
|---|---|
| `graphed_histogram.boost.Histogram` | deferred `boost_histogram.Histogram`; `.fill()` records and returns self (fills accumulate), `.plan()` exports the task graph |
| `factory(*arrays, histref=, weight=, sample=)` | a deferred histogram from a reference histogram's axes/storage plus one staged fill (the dask-histogram `factory` shape) |
| `histogram` / `histogram2d` / `histogramdd` | numpy-like deferred entry points (explicit bins + range) |
| `plan(histograms, ...)` | one plan aggregating **several** deferred histograms that share a source in a **single pass** (the `compute(dict_of_hists)` analogue) |
| `spec_of` / `zero_of` / `content_hash` | the canonical-spec helpers |
| `evaluators(*histograms)` | merged content-hash -> evaluator registry for `evaluate_ir(externals=...)` |
| `add_histograms` | native-`+` combine helper for multi-fill sums |

All standard boost storages (combine is native `+`); axes
Regular/Variable/Integer/IntCategory/StrCategory/Boolean. Sources implementing
`graphed.write.PartitionedSource` are filled partition by partition — their whole-dataset loader is
never invoked. One source family per histogram (PartitionedSource or in-memory; mixtures rejected).
Ragged fill values flatten completely at fill time.

### Multiplicative weights (M29)

HEP event weights arrive as several factors (generator weight x pileup x trigger SFs ...).
`fill(weight=...)` accepts `weight=` as a **sequence** of graphed Arrays, a first-class fill
signature: each weight is recorded as a real graph input and evaluation multiplies them
elementwise into the single fill weight.

```python
h.fill(g.pt, weight=[g.genweight, g.pileup_sf, g.trigger_sf])
```

A single weight records exactly as before — no `n_weights` param — so pre-M29 node identities,
specs, and preservation bundles are untouched.

### One pass over several histograms

`plan(histograms, ...)` compiles all the fills of several histograms into **one** multi-output IR,
so a sub-graph feeding multiple histograms (e.g. a trijet selection feeding both a pT and a b-tag
histogram) is read and evaluated **once** — not once per histogram. `run(plan).value` is the
matching `{label: histogram}` mapping (string keys for a `Mapping`, `"0"`, `"1"`, ... for a plain
sequence). Column projection covers the union of all the histograms' fills.

### Worker backends

`plan(backend=...)` accepts a zero-arg factory/class or an importable `"module:attr"` string
resolved **in the worker** — the required form for behavior-carrying backends, because behavior
dicts contain lambdas and do not pickle. A worker built without required behaviors fails loudly; it
never silently fills the wrong thing.

## The hist integration

`hist.graphed` (in the `hist` fork) supplies `Hist`/`NamedHist` as thin MRO sandwiches over this
package's `Histogram`: the familiar QuickConstruct
(`Hist.new.Reg(100, 0, 200, name="met").Double()`) and named-axis fills record deferred; executor
results wrap back into in-memory `hist.Hist` objects with names and labels intact (they ride the
canonical spec). The eight ADL benchmark queries run on exactly this surface.

## Phase 2 (deliberately not built)

Growth axes (combining grown category axes across partitions needs a category-union merge,
rejected at spec time for now); dask-style collection protocols (`persist`, `to_delayed`) — the
durable artifact is the compiled IR / `Plan`; behavior-reference forwarding by default.

## Status and gates

Frozen tests under `tests/frozen/m23/` and `tests/frozen/m29/` — never weakened. Gates: ruff +
ruff format · `mypy --strict` · pytest (>= 90% branch coverage) · `sphinx -W`. See `CLAUDE.md`
for the milestone digest, `docs/design.rst` for the engineering walkthrough, and
`.graphed/state.json` for current status.
