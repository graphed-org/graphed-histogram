API reference
=============

Most analyses use four calls: build a histogram, ``fill`` it, ``plan`` it, ``unpack`` the
result. The rest of the surface is there when you need variations, several histograms in
one pass, or the identity a worker resolves a fill by. Everything below except
``hist.graphed.Hist`` is exported from ``graphed_histogram``
(``import graphed_histogram as gh``).

Build a histogram
-----------------

``gh.boost.Histogram(*axes, storage=..., metadata=...)``
   A ``boost_histogram.Histogram`` whose fills are deferred. Same axes, same storages,
   same eager API for axes/storage/views of the empty state. All standard boost storages
   and the Regular / Variable / Integer / IntCategory / StrCategory / Boolean axes work.

``gh.factory(*arrays, histref=h, weight=None, sample=None)``
   ``dask-histogram``'s ``factory``: take a reference histogram's axes and storage, and
   stage one fill in a single call.

``gh.histogram(x, *, bins, range)`` · ``gh.histogram2d`` · ``gh.histogramdd``
   numpy-shaped one-liners. ``bins=`` and ``range=`` are required per dimension — neither has
   a usable default, and omitting ``range=`` raises. Unweighted gives you an exact ``Int64``
   storage; passing ``weights=`` gives you ``Weight()``.

``hist.graphed.Hist``
   The same deferred histogram behind ``hist``'s builder — ``Hist.new.Reg(100, 0, 200,
   name="met").Double()`` — with named-axis fills, and names and labels that survive the run.
   It ships in a fork of ``hist`` carrying the ``hist.graphed`` module, which upstream ``hist``
   does not have yet: ``pip install "hist @
   git+https://github.com/graphed-org/hist-graphed-mvp@graphed-mvp"``.

Fill it
-------

``h.fill(*arrays, weight=None, sample=None, threads=None, unweighted=False, variation_axis=False)``
   Records the fill and returns ``h``, so fills accumulate. Values must be ``graphed``
   arrays; ragged values flatten, exactly as an eager ``fill(ak.flatten(...))`` would.
   ``threads=`` is accepted so boost-histogram call sites port unchanged, and ignored:
   how many threads or workers touch this fill is the runner's business, set where you
   build the runner.

   ``weight=`` takes **one array or a list of them**, which is what HEP weights actually
   look like — ``fill(pt, weight=[genweight, pu_sf, lepton_sf])`` multiplies the factors
   elementwise for you, and each stays a real input to the graph rather than something you
   pre-multiplied by hand. If your session carries an ambient event weight, the fill applies
   it too; ``unweighted=True`` opts out of both that and every ``weight=`` factor.

   ``variation_axis=True`` puts your systematic variations on a ``StrCategory("variation")``
   axis of *one* histogram instead of giving each variation its own histogram. The choice is
   remembered per histogram — a later fill in the other mode is an error — and
   :doc:`design` walks through which variations can share a pass and which cannot.

Run it
------

``gh.plan(histograms, *, steps_per_file=1, backend=None, partitions=None)``
   One plan for several histograms — ``gh.plan({"pt": h1, "mass": h2})`` — and the
   ``compute(dict_of_hists)`` analogue. All their fills compile into one graph, so a
   selection feeding both histograms is read and evaluated once, and only the columns some
   fill touches are read off disk. A plain sequence works too — ``gh.plan([h1, h2])`` —
   and names the results ``"0"``, ``"1"``, … by position.

   Everything after ``histograms`` is keyword-only. ``steps_per_file=`` sets how many chunks
   each file becomes. ``partitions=`` takes the chunking itself instead, a
   ``Sequence[graphed.core.Partition]`` — ``partitions=[Partition.blind("events.parquet", "",
   i, 4) for i in range(4)]`` is what ``steps_per_file=4`` builds for you (``blind`` means the
   chunk's row range is resolved when a worker opens the file, not now). ``backend=`` is the
   evaluation backend each worker builds, as a class/factory or an importable
   ``"module:attr"`` string.

``h.plan(*, steps_per_file=1, backend=None, partitions=None)``
   The same for a single histogram; its run value is that one filled histogram rather than
   a mapping, so there is nothing to unpack. It refuses a histogram whose fills carry variations, because summing every staged fill
   into one histogram would silently merge the variations together — use ``gh.plan`` there,
   whose per-name results keep them apart.

``gh.unpack(value)``
   Turns what a runner returns into ``{name: histogram}``, or ``{name: {label: histogram}}``
   for a name whose fills carry variations. A histogram filled with ``variation_axis=True``
   comes back as a single histogram under its name, carrying the variations on its axis.

``gh.add_histograms(a, b)``
   Adds two filled histograms. Runners use it to combine partial results; you need it only
   if you are merging results yourself.

What you get back, in one program
---------------------------------

Two per-event weight factors, one of them varied up and down, and the two shapes a result
can take. Needs ``graphed[awkward]``.

.. code-block:: python

   import awkward as ak
   import boost_histogram as bh
   import graphed_histogram as gh
   from graphed import Session, vary
   from graphed.awkward import AwkwardBackend, from_parquet
   from graphed.core.execution import SequentialRunner

   events = ak.Array({"met": [10.0, 40.0, 70.0, 120.0, 30.0, 90.0, 150.0, 60.0]})
   ak.to_parquet(events, "met8.parquet")

   s = Session(AwkwardBackend())
   evt = from_parquet(s, "events", "met8.parquet", steps_per_file=2)

   lumi = evt.met * 0.0 + 2.0                        # one weight factor
   sf = evt.met * 0.0 + 1.0                          # another
   sf = vary(sf, "sf", up=sf * 1.5, down=sf * 0.5)   # up/down variations of it

   h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Weight())
   h.fill(evt.met, weight=[lumi, sf])                # factors multiply elementwise

   print(gh.label_listing({"met": h}))               # before running anything

   value = SequentialRunner().run(gh.plan({"met": h}, steps_per_file=2)).value
   print(sorted(value))                              # what the runner returns
   out = gh.unpack(value)                            # what you work with
   print(sorted(out["met"]))
   print(out["met"]["nominal"].values())
   print(out["met"]["sf_up"].values())

.. code-block:: text

   {'met': ['nominal', 'sf_up', 'sf_down']}
   [('met', 'nominal'), ('met', 'sf_down'), ('met', 'sf_up')]
   ['nominal', 'sf_down', 'sf_up']
   [6. 6. 2. 2.]
   [9. 9. 3. 3.]

Whatever you built with, the histograms in that mapping are ``boost_histogram.Histogram``
objects with your axis names and labels intact. ``hist.Hist(result)`` wraps one back up when
you want ``.plot()`` or name-based indexing.

The runner's own value is keyed by ``(name, label)`` pairs; ``unpack`` is what turns that
into the nested mapping. A name no variation reaches keeps a bare-name key and unpacks to a
bare histogram, so an analysis without systematics sees exactly what it always saw.

Inspect variations before running
---------------------------------

``gh.label_listing({name: hist})``
   ``{name: [labels]}`` without executing anything — nominal first, then your variations in
   the order you declared them. The answer does not depend on which variation mode a
   histogram is in, so it is the same question asked of an axis-mode and a per-variation
   histogram.

``gh.fill_nodes_by_label(h)``
   The ``{label: node}`` map for one histogram, when you want to see which recorded fill a
   label ended up on.

Identity and reproducibility
----------------------------

You need these when a fill has to be recognised somewhere other than where you wrote it — on
a worker, or in the histogram side of a preservation bundle. Day-to-day filling never touches
them.

``gh.spec_of(h)``
   The canonical axes-and-storage description of a histogram — versioned, key-sorted JSON
   that also carries axis metadata like ``hist``'s ``name`` and ``label``, so named axes
   survive a round trip. It doubles as the histogram's fingerprint: two fills with the same
   description merge.

``gh.content_hash(spec)``
   The hash of that description. It is how a fill recorded on your machine finds its
   evaluator again on a worker.

``gh.zero_of(spec)``
   Rebuilds the empty histogram from a description, anywhere, with no access to the original
   object. The round trip closes — ``gh.spec_of(gh.zero_of(spec)) == spec`` — which is why a
   description that travels on its own still names exactly one histogram at the other end.

``gh.evaluators(*histograms)``
   The merged ``{content hash: evaluator}`` registry, for wiring these fills into a graph
   you evaluate yourself.

``gh.FillEvaluator``
   The object that fills one chunk on a worker. Named here because it appears in
   ``evaluators()``; you construct it only if you are extending the package.

Generated reference
-------------------

.. autosummary::
   :toctree: generated
   :recursive:

   graphed_histogram
