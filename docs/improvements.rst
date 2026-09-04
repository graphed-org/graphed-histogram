Limitations and gotchas
=======================

What doesn't work yet, and what to do instead.

Growth axes
-----------

Axes created with ``growth=True`` are rejected with a ``TypeError`` when you construct the
histogram. Each partition fills its own partial histogram, and two partials that each grew a
different category set cannot yet be merged.

Instead, declare the categories you expect up front — ``bh.axis.StrCategory(["ee", "mm",
"em"])`` or an ``bh.axis.IntCategory`` of the codes you use, without ``growth=True``. Values
outside your list land in overflow, exactly as they do for any non-growth category axis.

No ``.compute()``, ``persist``, or ``to_delayed``
-------------------------------------------------

Deferred histograms are not dask collections, so the dask collection protocols don't apply.
You build a plan — ``h.plan()``, or ``gh.plan({"pt": h1, "mass": h2})`` to run several
histograms in one pass — and hand it to a runner, which returns concrete boost histograms.
See :doc:`design` for the full path from fill to result.

Float storages and reproducibility
----------------------------------

Integer count storages (``Int64``) give bit-identical totals however the run is split.
``Weight`` and ``Mean`` storages accumulate floats, and floating-point addition depends on
order: for a fixed runner configuration the combine order is fixed, so re-running reproduces
your totals exactly — but changing the worker count or the partitioning can change the last
bits.

For bit-for-bit comparisons between runs, keep the runner configuration fixed, or compare an
``Int64`` count alongside the weighted result.

One partitioned source per plan
-------------------------------

A plan reads one dataset: every fill in it must record into the same session, and that session
must have exactly one partitioned source. Fills drawing on two different datasets cannot be
planned together.

Histograms add, so run one plan per dataset and sum the results
(``results_2017["pt"] + results_2018["pt"]``).

Backend behaviors are not carried to workers
--------------------------------------------

Each worker builds its own array backend, and by default a bare one of the same class. Coffea-style
behaviors attached to your record types live in a dictionary of Python functions that does not
pickle, so nothing recovers them from the recording session automatically.

Pass ``backend="my_analysis:make_backend"`` to ``plan()`` — an importable zero-argument callable
resolved in the worker. A worker without the behaviors an analysis asks for raises rather than
filling the wrong thing.
