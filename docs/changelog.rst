What changed
============

Unreleased
----------

Fixed
~~~~~

* ``gh.plan`` and ``h.plan()`` keep the partitioning chosen at read time: ``steps_per_file`` now
  defaults to ``None``, the source's own setting, as in ``graphed.aggregate_plan``, so
  ``from_parquet(..., steps_per_file=3)`` gives three tasks without repeating it on the plan. An
  explicit value still overrides, and a source that sets nothing is still one step per file.

0.0.7
-----

Requires ``graphed`` 0.0.10 or later.

Changed
~~~~~~~

* A result is dense when the run returns it: the plan's ``finalize`` runs once, on the head, after
  the last combine, and makes each chunked result the class you instantiated, so
  ``run(plan).value`` and ``gh.unpack`` hold a plain ``boost_histogram.Histogram``, ``hist.Hist``
  or ``hist.NamedHist`` (``type(result) is hist.Hist``). There is no chunked-until-read phase: no
  stand-in subclass, no densifying on the first read, no "chunked" repr. Partials and merges stay
  chunked in every task (:ref:`chunked-partials`).
* ``session.materialize`` on a fill node gives the dense histogram, as does ``gh.FillEvaluator``;
  ``FillEvaluator(..., partial=True)`` gives the chunked partial a plan's tasks fill.
* ``gh.boost.pieces`` carries a ``finalize`` for the plan it feeds, and its ``externals`` are keyed
  by ``graphed.execute.external_key``.

0.0.6
-----

Requires ``graphed`` 0.0.9 or later.

Changed
~~~~~~~

* A histogram with a category axis keeps its partials and merges chunked, one dense slice per
  filled category key (``hist.chunked.ChunkedHist``), so a task that fills a few of many category
  combinations ships and merges only those. The result is the same dense
  ``boost_histogram.Histogram``, bin for bin; partials stay dense where chunking cannot hold them
  (:ref:`chunked-partials`). ``hist`` 2.11 or later is now a dependency, and with it
  ``boost-histogram`` 1.7 or later.
* ``gh.dense_backed(h)`` keeps ``h``'s partials and merges dense.
* A plan's result is the class you instantiated. ``hist.graphed.Hist`` and ``NamedHist`` results
  are now ``hist.Hist`` and ``hist.NamedHist`` with the histogram's ``name``, ``label`` and
  ``metadata``, and a ``gh.boost.Histogram(metadata=...)`` result keeps its ``metadata``; both
  were a plain ``boost_histogram.Histogram`` without them.
* A fill takes a scalar (``str``, ``int``, ``float``, ``bool`` or a numpy scalar) for any axis, ``weight=`` factor
  or ``sample=`` and broadcasts it per entry, as an eager fill does:
  ``h.fill(dataset="DYJets", pt=events.Muon.pt)``. Fills with different constants are different
  nodes. Every fill needs a graphed Array on at least one axis.
* A histserv server's predicted size counts ``B`` = 227 MiB before any histogram (was 164 MiB) and
  ``I`` = 170 B per histogram per task (was 160 B), so offered ``memory_mb`` sizes below 227 are
  refused. GitHub's ubuntu-24.04 runner image 20261018 grew an idle server's resident memory by
  33 MiB; ``B`` now leaves room for another change of that size.

0.0.5
-----

Requires ``graphed`` 0.0.7 or later.

Filling on histserv servers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``graphed_histogram.histserv`` fills histograms on `histserv <https://pypi.org/project/histserv/>`_
  servers instead of carrying them through the reduction tree: back a histogram with a
  ``histserv.Context`` and ``gh.plan`` sizes it, packs it onto servers of the sizes you offer and
  declares those servers in ``plan.services`` for the executor to start (:ref:`histserv`). Install
  with ``pip install "graphed-histogram[histserv]"``; it uses
  graphed 0.0.7's service surface (``graphed.services``).
* ``gh.boost.pieces(histograms)`` hands out the parts ``gh.plan`` builds from, so a plan can put
  your histograms beside outputs of its own.

Changed
~~~~~~~

* Two fills the optimizer merges into one (``weight=[w]`` and ``weight=[w * 1.0]``) are now read
  once per fill, so the histogram is what filling twice gives, in ``gh.plan`` and ``h.plan()``
  alike. Both used to refuse the plan.
* ``h.plan()`` refuses a histserv-backed histogram; plan it with ``gh.plan``.

0.0.4
-----

Requires ``graphed`` 0.0.5 or later.

Growth axes
~~~~~~~~~~~

* ``StrCategory([], growth=True)``, ``IntCategory([], growth=True)``, ``Integer(...,
  growth=True)`` and ``Regular(..., growth=True)`` now fill over a partitioned source: each
  partition grows its own axes, and ``gh.add_histograms`` merges the partial results — a
  category union in partition order, and a bin-aligned union for growing ``Regular`` axes. The
  result, and the closed list of ways it differs from one eager fill of the whole dataset, is
  in :ref:`growth-axes`. A growing ``Variable`` axis still raises ``TypeError``.
* ``gh.spec_of`` refuses, with a ``TypeError`` naming it, any axis option its description cannot
  carry — a transform, ``circular``, category ``overflow=False``, a growing ``Variable`` — where
  it used to drop the option silently. A spec with a growth axis is version 2; every other spec
  keeps its version-1 bytes and content hash.
* Requires boost-histogram 1.4.1 or later.

0.0.3
-----

Packaging
~~~~~~~~~

* The MIT ``LICENSE`` file ships in the repository and in the sdist, so a conda-forge recipe can
  take the license from the package it builds (#16). Nothing else changes: this release is
  0.0.2's code, tested against graphed 0.0.3.

0.0.2
-----

Systematic variations reach your histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fill with a varied array — a jet-energy scale shifted up and down, a scale factor with an
  uncertainty — and you get one histogram per universe, keyed by label, out of the same single
  pass over your files. ``fill`` lines the axis values, the weight factors and ``sample=`` up
  itself, whatever selection each of them came through (#2).
* ``variation_axis=True`` puts those universes on a ``StrCategory("variation")`` axis of *one*
  histogram instead. A variation that only changes the weight then costs one fill over shared
  values rather than one fill per label, and the result is an ordinary ``bh.Histogram`` that
  merges, slices and plots like any other (#4).
* ``gh.label_listing({name: hist})`` tells you which variations reach which output before you
  read a byte — the check to run before you submit a job (#3).
* A variation whose point names another variation's coordinate now lands in that universe
  rather than in one of its own, so joint points give you the histogram you meant (#8).
* ``gh.fill_nodes_by_label(h)`` maps each label to the fill it was recorded on, when you want
  to see where a label ended up (#2).

Fixed
~~~~~

* A fill whose weight is per-event while its values are per-object used to multiply the two
  elementwise and quietly fill the wrong thing. The weight is now broadcast into the values'
  row space, and a pairing that cannot be lined up names the operand instead of guessing (#11).
* A fill whose inputs come from a correctionlib correction runs, rather than failing with a
  missing evaluator, and its systematic universes stay distinct when the run goes through a
  process pool (#5, with the fix in ``graphed``).

Upgrading from 0.0.1
~~~~~~~~~~~~~~~~~~~~

* What a runner hands back for ``gh.plan({...})`` is now keyed by ``(name, label)`` wherever
  variations are involved. Call ``gh.unpack(value)`` to get ``{name: histogram}`` — or
  ``{name: {label: histogram}}`` for a varied one — the way you read results before. An output
  no variation reaches keeps its bare-name key, so an analysis without systematics reads
  exactly as it did (#2).
* Fills that carry variations cannot go through the single-histogram ``h.plan()``, which would
  sum the universes into one histogram. Plan them through ``gh.plan({...})``, whose per-name
  results keep them apart (#2).

Documentation
~~~~~~~~~~~~~

* The README and every docs page were rewritten for someone with an analysis to run: a
  complete program in the first screen, the variations walkthrough, and the limitations each
  paired with its workaround (#6, #7).
* Read the Docs builds the documentation again (#1).

Under the hood: the merge queue is gated by CI, the ``graphed`` build the tests run against is
pinned, and the type checker now covers the test tree as well as the package (#9, #10, #13, #14).

0.0.1
-----

First release: deferred ``boost_histogram`` and ``hist`` filling on ``graphed`` task graphs.
``.fill()`` records and a runner computes, several histograms share one pass over the data, and
a histogram's axes-and-storage description is what a worker resolves a fill by — so the same
plan fills the same histogram wherever it runs.
