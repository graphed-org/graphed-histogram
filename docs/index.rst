graphed-histogram
=================

Deferred `boost-histogram <https://boost-histogram.readthedocs.io>`_ / `hist
<https://hist.readthedocs.io>`_ filling for ``graphed``: the ``dask-histogram`` shape,
without a ``.compute()``. **A fill records; a runner computes.**

You keep the boost-histogram API you already use. What changes is that ``.fill(...)``
stages work into a plan instead of reading data, so describing a thousand-file fill costs
nothing, several histograms sharing a selection run in **one pass**, and the same plan runs
on your laptop or on a cluster without editing it.

Install
-------

.. code-block:: bash

   pip install "graphed[awkward]" graphed-histogram   # awkward events + deferred fills
   pip install graphed-executors                      # process-pool runners
   pip install "graphed-executors[dask]"              # or [parsl] for a cluster

Building ``graphed`` from source — when there is no wheel for your platform — needs a Rust
toolchain.

Your first deferred histogram
-----------------------------

Awkward events in, two filled ``boost_histogram.Histogram`` objects out. The one new
ingredient over eager boost-histogram is a **source**: the object that hands your dataset
out in chunks, so each task fills its own piece. Here it is a parquet file; ROOT files work
the same way.

.. code-block:: python

   import awkward as ak
   import boost_histogram as bh
   import graphed_histogram as gh
   from graphed import Session
   from graphed.awkward import AwkwardBackend, from_parquet
   from graphed.core.execution import SequentialRunner

   events = ak.Array({
       "met": [10.0, 40.0, 70.0, 120.0, 30.0, 90.0, 150.0, 60.0],
       "jet_pt": [[40.0, 25.0], [55.0], [30.0, 60.0, 20.0], [80.0],
                  [15.0, 45.0], [70.0, 10.0], [95.0], [35.0, 35.0]],
   })
   ak.to_parquet(events, "events.parquet")    # stand in for your dataset

   s = Session(AwkwardBackend())
   evt = from_parquet(s, "events", "events.parquet", steps_per_file=4)

   met = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())
   met.fill(evt.met)                          # records the fill; nothing is read yet

   jets = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())
   jets.fill(evt.jet_pt)                      # ragged values flatten at fill time

   plan = gh.plan({"met": met, "jet_pt": jets}, steps_per_file=4)
   out = gh.unpack(SequentialRunner().run(plan).value)

   print(out["met"].values())
   print(out["jet_pt"].values())

.. code-block:: text

   [3 3 1 1]
   [9 5 0 0]

Three lines carry the whole idea. ``met.fill(evt.met)`` records and returns the histogram —
fills accumulate, and ``met`` keeps its eager axes, storage and (empty) views throughout.
``gh.plan({...})`` compiles every staged fill of every named histogram into one graph, so
the four chunks are each read once and both histograms are filled in the same pass.
``gh.unpack`` turns the runner's result into the ``{name: histogram}`` mapping you asked
for.

There is no ``.compute()``
--------------------------

That is the only API difference worth memorising. You export a plan and hand it to a
runner, and every runner takes the same plan — swapping ``SequentialRunner`` for a process
pool, a dask cluster or a parsl pool changes no analysis code:

.. code-block:: python

   # keep the imports and ``events`` from the program above and replace everything after
   # them with this; needs graphed-executors. A process pool spawns its workers and each
   # re-imports your file, so anything with an effect goes under a __main__ guard —
   # writing the file, staging the fills, running — or a re-importing worker rewrites the
   # very file the run is reading.
   from graphed_executors.local import ProcessPoolExecutor

   if __name__ == "__main__":
       ak.to_parquet(events, "events.parquet")

       s = Session(AwkwardBackend())
       evt = from_parquet(s, "events", "events.parquet", steps_per_file=4)

       met = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())
       met.fill(evt.met)
       jets = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())
       jets.fill(evt.jet_pt)

       plan = gh.plan({"met": met, "jet_pt": jets}, steps_per_file=4)
       out = gh.unpack(ProcessPoolExecutor(max_workers=2).run(plan).value)
       print(out["met"].values())
       print(out["jet_pt"].values())

.. code-block:: text

   [3 3 1 1]
   [9 5 0 0]

Same numbers, on two processes. Because histograms add, the partial results merge in any
order: your total is the same on one worker and on a hundred, and integer-count storages
are exact whatever the worker count.

Where to go next
----------------

* :doc:`design` — how it works and why it holds: why filling is free until you run, why
  many histograms cost one pass, why a plan re-run elsewhere fills the same histogram, and
  the walkthrough for systematic variations (one histogram per variation, or one histogram
  with a ``variation`` axis).
* :doc:`api` — the public surface grouped by task, and what the runner hands back.
* :doc:`improvements` — the limitations you may hit, each with its workaround.

.. toctree::
   :maxdepth: 2

   design
   api
   improvements
