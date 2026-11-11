# graphed-histogram

Deferred [boost-histogram](https://github.com/scikit-hep/boost-histogram) /
[hist](https://github.com/scikit-hep/hist) filling for
[graphed](https://github.com/graphed-org/graphed) — the
[dask-histogram](https://github.com/dask-contrib/dask-histogram) shape, without a
`.compute()`: **`.fill()` records, a runner computes.**

- Fill with the boost-histogram API you already use; nothing runs until you hand a plan to a
  runner, so a thousand-file fill costs nothing to describe.
- Several histograms sharing a selection run in **one pass** over the data — a shared
  sub-expression is read and evaluated once, not once per histogram.
- On fixed axes histograms add, so partial results merge in any combine tree: integer counts
  are exact however the run is split, and nothing depends on the worker count. Axes with
  `growth=True` merge too, to the result [Growth axes](docs/design.rst#growth-axes) defines.

## Install

```bash
pip install "graphed[awkward]" graphed-histogram "uproot>=5.7.7"  # ROOT events + deferred fills
pip install graphed-executors                      # process-pool runners
pip install "graphed-executors[dask]"              # or [parsl] for a cluster
```

Building `graphed` from source (no wheel for your platform) needs a Rust toolchain.

## A quick look

A ROOT file in, a `hist.Hist` out, in this process. `uproot.graphed` (uproot 5.7.7 or later)
records the file's events the way `uproot.dask` does, `.fill()` records the fill, and
`graphed.materialize` runs it, reading only the branches the fill uses:

```python
import awkward as ak
import graphed
import numpy as np
import uproot
from hist.graphed import Hist

with uproot.recreate("events.root") as f:  # stand in for your dataset
    f.mktree("Events", {"Jet_pt": "var * float64", "MET_pt": "float64",
                        "genWeight": "float64", "pu_sf": "float64"})
    f["Events"].extend({
        "Jet_pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0],
                            [80.0], [15.0, 45.0], [70.0, 10.0]]),
        "MET_pt": np.array([10.0, 40.0, 70.0, 120.0, 30.0, 90.0]),
        "genWeight": np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0]),
        "pu_sf": np.array([0.9, 1.1, 1.0, 0.95, 1.05, 1.0]),
    })

events = uproot.graphed("events.root:Events")  # reads the file's form, no events
h = Hist.new.Reg(4, 0, 100, name="pt", label="jet $p_T$ [GeV]").Int64().fill(pt=events.Jet_pt)

filled = graphed.materialize(h)  # a new, filled hist.Hist; h itself stays un-run
print(type(filled).__name__, filled.values())
# Hist [3 4 3 1]
artists = filled.plot()
```

Fitting and plotting are `hist`, scipy and matplotlib code on `filled`, as they would be on any
`hist.Hist`.

## Your first deferred histogram

The same file, through a plan a runner executes: this is how a fill runs over a whole dataset, a
chunk per task. A runner hands back the in-memory class your histogram extends; here a
`boost_histogram.Histogram`.

```python
import boost_histogram as bh
import graphed_histogram as gh
from graphed.core.execution import SequentialRunner

h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 100.0), storage=bh.storage.Int64())
h.fill(events.Jet_pt)                      # records the fill; nothing is read yet

plan = h.plan(steps_per_file=2)            # 2 chunks -> 2 fill tasks + a combine
result = SequentialRunner().run(plan).value
print(result.values())
# [3 4 3 1]
```

`h` keeps its axes and storage type, and warns if you read its bins before it runs; what changed is
that `.fill()` stages work and evaluation belongs to a runner. Ragged values flatten at fill
time, exactly as an eager `fill(ak.flatten(...))` would.

## Results in this process: `graphed.materialize`

A runner is for the full dataset. To look at a value while you write the analysis,
`graphed.materialize` evaluates anything recorded, here in this process, reading only the columns
it uses:

```python
import graphed
from graphed.awkward import gak

print(graphed.materialize(gak.sum(events.genWeight, axis=None)))
# 4.0
print(graphed.materialize(events.MET_pt[events.MET_pt > 50.0]).tolist())
# [70.0, 120.0, 90.0]
```

It takes a nesting of them too, `graphed.materialize({"n": ..., "met": ...})`, and evaluates them
in one read.

## The one thing that's different

There is **no `.compute()`**. You export a plan and run it — and every runner accepts the
same plan:

```python
# needs graphed-executors. Each worker re-imports this file, so everything with an effect
# goes under the __main__ guard, or a worker would rewrite the file the run is reading.
import awkward as ak
import boost_histogram as bh
import graphed_histogram as gh
import uproot
from graphed_executors.local import ProcessPoolExecutor

if __name__ == "__main__":
    with uproot.recreate("jets.root") as f:
        f.mktree("Events", {"Jet_pt": "var * float64"})
        f["Events"].extend({"Jet_pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0],
                                                [80.0], [15.0, 45.0], [70.0, 10.0]])})

    events = uproot.graphed("jets.root:Events")
    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 100.0), storage=bh.storage.Int64())
    h.fill(events.Jet_pt)

    result = ProcessPoolExecutor(workers=2).run(h.plan(steps_per_file=2)).value
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
# continuing from the first example (same events)
hmet = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Weight())
hmet.fill(events.MET_pt, weight=[events.genWeight, events.pu_sf])
```

A single array (`weight=w`) works as before.

## Several histograms, one pass

`gh.plan({...})` is the `compute(dict_of_hists)` analogue: all the fills compile into one
graph, so a selection feeding several histograms is read and evaluated once, and only the
columns any fill touches are read off disk.

```python
plan = gh.plan({"met": hmet, "jet_pt": h}, steps_per_file=2)
out = SequentialRunner().run(plan).value   # {name: histogram}
print(out["jet_pt"].values())
# [3 4 3 1]
```

If your fills carry systematic variations (via `graphed.vary`), each name maps to
`{label: histogram}` instead — or pass `variation_axis=True` to `fill()` to get one
histogram with a `variation` axis rather than one histogram per label.
`gh.label_listing(...)` shows which labels reach which histogram before you run anything.
The [variations walkthrough](docs/design.rst) covers the choice.

## Several datasets, one plan

Each dataset is its own source, and `gh.plan` takes histograms of several in one call: one plan
with the sum of their tasks, its result in the nesting you passed, such as a coffea fileset's.

```python
import shutil

shutil.copy("events.root", "ttbar.root")  # a second dataset

def met(events):
    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())
    return h.fill(events.MET_pt)

fileset = {"dy": uproot.graphed("events.root:Events"),
           "ttbar": uproot.graphed("ttbar.root:Events")}
plan = gh.plan({name: {"met": met(ev)} for name, ev in fileset.items()}, steps_per_file=2)
out = SequentialRunner().run(plan).value   # {dataset: {name: histogram}}
print(len(plan.tasks), out["ttbar"]["met"].values())
# 4 [3 2 1 0]
```

That is `graphed.collate` underneath, which joins any plans, one per dataset, into one, its value a
result per dataset:

```python
plan = graphed.collate({name: met(ev).plan(steps_per_file=2) for name, ev in fileset.items()})
print(SequentialRunner().run(plan).value["dy"].values())
# [3 2 1 0]
```

One histogram still fills from one dataset: filling one histogram from both raises `TypeError`, so
add the per-dataset results with `gh.add_histograms`.

## Which entry point do I want?

| You write | You get |
|---|---|
| `gh.boost.Histogram(*axes, storage=...)` | a deferred `boost_histogram.Histogram`: `.fill()` records and returns self, fills accumulate, `.plan()` exports the plan |
| `Hist.new.Reg(100, 0, 200, name="met").Double()` | the same, through the `hist` integration — QuickConstruct and named-axis fills |
| `gh.factory(*arrays, histref=...)` | dask-histogram's `factory`: a reference histogram's axes plus one staged fill |
| `gh.histogram` / `histogram2d` / `histogramdd` | numpy-like one-liners (explicit `bins=` and `range=`) |
| `gh.plan({name: hist, ...})` | one plan for several histograms in a single pass |

Whichever you build with, a run hands back the class you instantiated, with your axis names
and labels intact: `boost_histogram.Histogram` for `gh.boost.Histogram`, and `hist.Hist` or
`hist.NamedHist` for the `hist.graphed` builders.

The `hist` builder lives in the `hist.graphed` module, which `hist` ships from 2.12 on:

```bash
pip install "hist>=2.12"
```

All standard boost storages and the Regular / Variable / Integer / IntCategory /
StrCategory / Boolean axes are supported, and `growth=True` on Regular, Integer, IntCategory and
StrCategory ([Growth axes](docs/design.rst#growth-axes)).

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
- On fixed axes, integer counts are exact under any combine tree. Inexact float sums (and
  every `Mean` / `WeightedMean` field) keep their last bits for a given runner family and
  partitioning, whatever the worker count: floating-point addition is order-sensitive, and
  each runner fixes its combine tree up front. For growth axes, including the values whose
  bins depend on the partitioning, see [Growth axes](docs/design.rst#growth-axes).
- Worker backends are passed as a factory/class or an importable `"module:attr"` string and
  built **in the worker**; a worker missing a required behavior fails loudly rather than
  filling the wrong thing.

## Not supported yet

- **Growing `Variable` axes.** They raise `TypeError`; declare the edges up front. Other axes
  grow ([Growth axes](docs/design.rst#growth-axes)).
- **dask-style `persist` / `to_delayed`.** A plan is a live object your script builds, not
  a file format — rebuild it from the script and hand it to whichever runner you have.

## Next

- [How graphed-histogram works](docs/design.rst) — why filling is free until you run, how
  many histograms share one pass, and the variations walkthrough.
- [API reference](docs/api.rst).
- Siblings: [graphed](https://github.com/graphed-org/graphed) (the frontend your arrays come
  from) and [graphed-executors](https://github.com/graphed-org/graphed-executors) (process-pool,
  dask and parsl runners).
