Limitations and gotchas
=======================

What doesn't work yet, and what to do instead.

Growth axes
-----------

``Regular``, ``Integer``, ``IntCategory`` and ``StrCategory`` axes grow (:ref:`growth-axes`).
Two things are missing:

* A growing ``Variable`` axis raises ``TypeError`` when you construct the histogram: its new
  edges come from the data, so two partitions' axes share no grid to merge on. Declare the
  edges up front.
* Several fills into one growth-category histogram (two ``fill`` calls, or an axis-mode fill
  whose values vary) list their categories partition by partition, not fill by fill as one
  eager fill per call would. Declare the categories up front if you need fill order.

No ``.compute()``, ``persist``, or ``to_delayed``
-------------------------------------------------

Deferred histograms are not dask collections, so the dask collection protocols don't apply.
You build a plan — ``h.plan()``, or ``gh.plan({"pt": h1, "mass": h2})`` to run several
histograms in one pass — and hand it to a runner, which returns concrete boost histograms.
See :doc:`design` for the full path from fill to result.

Float storages and reproducibility
----------------------------------

On fixed axes, integer count storages (``Int64``) give bit-identical totals however the run is
split; on growth axes, see the edge and non-finite exceptions in :ref:`growth-axes`. Inexact
float sums in any storage, and every ``Mean`` and ``WeightedMean`` field, depend on order:
each runner fixes its combine tree from the runner family and the partition count, so
re-running reproduces your totals exactly, and so does changing the worker count — but
changing the runner family or the partitioning can change the last bits.

For bit-for-bit comparisons between runs, keep the runner configuration fixed, or compare an
``Int64`` count alongside the weighted result.

One partitioned source per plan
-------------------------------

A plan reads one dataset: every fill in it must record into the same session, and that session
must have exactly one partitioned source. Fills drawing on two different datasets cannot be
planned together.

Run one plan per dataset and add the results
(``gh.add_histograms(results_2017["pt"], results_2018["pt"])``).

Backend behaviors are not carried to workers
--------------------------------------------

Each worker builds its own array backend, and by default a bare one of the same class. Coffea-style
behaviors attached to your record types live in a dictionary of Python functions that does not
pickle, so nothing recovers them from the recording session automatically.

Pass ``backend="my_analysis:make_backend"`` to ``plan()`` — an importable zero-argument callable
resolved in the worker. A worker without the behaviors an analysis asks for raises rather than
filling the wrong thing.
