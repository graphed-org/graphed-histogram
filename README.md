# graphed-histogram

Deferred [boost-histogram](https://github.com/scikit-hep/boost-histogram) /
[hist](https://github.com/scikit-hep/hist) filling for
[graphed](https://github.com/graphed-org) — the
[dask-histogram](https://github.com/dask-contrib/dask-histogram) shape, without a
`.compute()`: **`.fill()` records, a runner computes.**

- Fill with the boost-histogram API you already use; nothing runs until you hand a plan to a
  runner, so a thousand-file fill costs nothing to describe.
- Several histograms sharing a selection run in **one pass** over the data — a shared
  sub-expression is read and evaluated once, not once per histogram.
- Histograms add, so partial results merge in any order: the total is the same on 1 worker
  or 100, and integer counts are exact under any combine order.

## Install

```bash
pip install "graphed[awkward]" graphed-histogram   # awkward events + deferred fills
pip install graphed-executors                      # process-pool runners
pip install "graphed-executors[dask]"              # or [parsl] for a cluster
```

Building `graphed` from source (no wheel for your platform) needs a Rust toolchain.

## Your first deferred histogram

A complete program: awkward events in, a filled `boost_histogram.Histogram` out. The only
new ingredient over eager boost-histogram is a *source* — the object that hands your dataset
out in chunks, so workers each fill their piece. Here it is a parquet file; ROOT files work
the same way.

```python
import awkward as ak
import boost_histogram as bh
import graphed_histogram as gh
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet
from graphed.core.execution import SequentialRunner

events = ak.Array({
    "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0],
                                   [80.0], [15.0, 45.0], [70.0, 10.0]])}),
    "MET": ak.zip({"pt": [10.0, 40.0, 70.0, 120.0, 30.0, 90.0]}),
    "genweight": [1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
    "pu_sf": [0.9, 1.1, 1.0, 0.95, 1.05, 1.0],
})
ak.to_parquet(events, "events.parquet")    # stand in for your dataset

s = Session(AwkwardBackend())
evt = from_parquet(s, "events", "events.parquet", steps_per_file=2)

h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 100.0), storage=bh.storage.Int64())
h.fill(evt.Jet.pt)                         # records the fill; nothing is read yet

plan = h.plan(steps_per_file=2)            # 2 chunks -> 2 fill tasks + a combine
result = SequentialRunner().run(plan).value
print(result.values())
# [3 4 3 1]
```

`h` keeps the eager boost API (axes, storage, views of the empty state); what changed is
that `.fill()` stages work and evaluation belongs to a runner. Ragged values flatten at fill
time, exactly as an eager `fill(ak.flatten(...))` would.

## The one thing that's different

There is **no `.compute()`**. You export a plan and run it — and every runner accepts the
same plan:

```python
# keep the imports and `events` from the program above, and replace everything after
# them with this — needs graphed-executors. A process pool spawns its workers and each
# re-imports your file, so anything with an effect goes under a __main__ guard: writing
# the file, staging the fill, running. Otherwise a re-importing worker rewrites the very
# file the run is reading.
from graphed_executors.local import ProcessPoolExecutor

if __name__ == "__main__":
    ak.to_parquet(events, "events.parquet")

    s = Session(AwkwardBackend())
    evt = from_parquet(s, "events", "events.parquet", steps_per_file=2)

    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 100.0), storage=bh.storage.Int64())
    h.fill(evt.Jet.pt)

    result = ProcessPoolExecutor(max_workers=2).run(h.plan(steps_per_file=2)).value
    print(result.values())
    # [3 4 3 1]
```

Same numbers, on two processes.

The dask and parsl runners in `graphed-executors` take the identical plan onto a cluster;
they come with the `[dask]` and `[parsl]` extras.

## Weights come in factors

HEP event weights arrive as several factors. `weight=` takes a **list** of per-event arrays
and multiplies them elementwise — no pre-multiplying in your own code:

```python
# continuing from the first example (same evt)
hmet = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Weight())
hmet.fill(evt.MET.pt, weight=[evt.genweight, evt.pu_sf])
```

A single array (`weight=w`) works as before.

## Several histograms, one pass

`gh.plan({...})` is the `compute(dict_of_hists)` analogue: all the fills compile into one
graph, so a selection feeding several histograms is read and evaluated once, and only the
columns any fill touches are read off disk.

```python
plan = gh.plan({"met": hmet, "jet_pt": h}, steps_per_file=2)
out = gh.unpack(SequentialRunner().run(plan).value)   # {name: histogram}
print(out["jet_pt"].values())
# [3 4 3 1]
```

If your fills carry systematic variations (via `graphed.vary`), each name maps to
`{label: histogram}` instead — or pass `variation_axis=True` to `fill()` to get one
histogram with a `variation` axis rather than one histogram per label.
`gh.label_listing(...)` shows which labels reach which histogram before you run anything.
The [variations walkthrough](docs/design.rst) covers the choice.

## Which entry point do I want?

| You write | You get |
|---|---|
| `gh.boost.Histogram(*axes, storage=...)` | a deferred `boost_histogram.Histogram`: `.fill()` records and returns self, fills accumulate, `.plan()` exports the plan |
| `Hist.new.Reg(100, 0, 200, name="met").Double()` | the same, through the `hist` integration — QuickConstruct and named-axis fills |
| `gh.factory(*arrays, histref=...)` | dask-histogram's `factory`: a reference histogram's axes plus one staged fill |
| `gh.histogram` / `histogram2d` / `histogramdd` | numpy-like one-liners (explicit `bins=` and `range=`) |
| `gh.plan({name: hist, ...})` | one plan for several histograms in a single pass |

Whichever you build with, a run hands back `boost_histogram.Histogram` objects with your
axis names and labels intact. Wrap one in `hist.Hist(result)` to get `.plot()` and
name-based indexing back.

The `hist` builder lives in a fork of `hist` that carries the `hist.graphed` module; upstream
`hist` does not ship it yet:

```bash
pip install "hist @ git+https://github.com/graphed-org/hist-graphed-mvp@graphed-mvp"
```

All standard boost storages and the Regular / Variable / Integer / IntCategory /
StrCategory / Boolean axes are supported.

Beyond those, the toolbox splits by task:

- **run**: `h.plan(...)`, `gh.plan(...)`, `gh.unpack(...)`, `gh.add_histograms(a, b)`
- **inspect variations**: `gh.label_listing(...)`, `gh.fill_nodes_by_label(h)`
- **identity and reproducibility** (advanced): `gh.spec_of(h)` — the canonical axes/storage
  description that doubles as the histogram's fingerprint; `gh.zero_of(spec)` rebuilds the
  empty histogram anywhere; `gh.content_hash`, `gh.evaluators` wire fills into a graph you
  evaluate yourself. That fingerprint is why a plan re-run on another machine fills the
  same histogram.

## What you can count on

- Fills read partition by partition; a source's whole-dataset loader is never invoked.
- Integer-count storages are exact for any worker count; float storages are reproducible
  for a fixed runner configuration (floating-point addition is order-sensitive, and the
  combine order is fixed up front).
- Worker backends are passed as a factory/class or an importable `"module:attr"` string and
  built **in the worker**; a worker missing a required behavior fails loudly rather than
  filling the wrong thing.

## Not supported yet

- **Growth axes.** Declare the categories you expect with an explicit `StrCategory` /
  `IntCategory` instead.
- **dask-style `persist` / `to_delayed`.** A plan is a live object your script builds, not
  a file format — rebuild it from the script and hand it to whichever runner you have.
- **Two datasets in one plan.** A plan reads one chunked dataset: every fill in it records
  into the same session, and that session has exactly one partitioned source. Run a plan
  per dataset and add the results — histograms add.

## Next

- [How graphed-histogram works](docs/design.rst) — why filling is free until you run, how
  many histograms share one pass, and the variations walkthrough.
- [API reference](docs/api.rst).
- Siblings: [graphed](https://github.com/graphed-org) (the frontend your arrays come from)
  and [graphed-executors](https://github.com/graphed-org) (process-pool, dask and parsl
  runners).
