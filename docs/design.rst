How graphed-histogram works
===========================

You call ``h.fill(jets.pt)`` and nothing happens. No entries land, the view stays zero, and the
call hands you the histogram back. That single change — the fill **records**, and a runner
computes it later — is the whole of this package, and everything below follows from it: why
writing the same fill twice costs once, why twenty histograms cost one pass over your files, why
your numbers come out the same on one core and on a hundred, and how a systematic variation
becomes an axis instead of a directory of files.

.. contents::
   :local:
   :depth: 2


One complete run
----------------

Needs ``graphed[awkward]`` and ``graphed-histogram``. A **source** is the read side — anything
that can hand a dataset out in slices; here a parquet file, in real work your ROOT reader.

.. code-block:: python

    import awkward as ak
    import boost_histogram as bh
    import graphed_histogram as gh
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_parquet
    from graphed.core.execution import SequentialRunner

    EVENTS = ak.Array({
        "MET": ak.zip({"pt": [10.0, 40.0, 70.0, 120.0, 30.0, 90.0]}),
        "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0],
                                       [80.0], [15.0, 45.0], [70.0, 10.0]])}),
    })
    ak.to_parquet(EVENTS, "events.parquet")

    session = Session(AwkwardBackend())
    events = from_parquet(session, "events", "events.parquet", steps_per_file=3)

    jet_pt = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())
    met_pt = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())
    jet_pt.fill(events.Jet.pt)
    met_pt.fill(events.MET.pt)

    plan = gh.plan({"jet_pt": jet_pt, "met_pt": met_pt}, steps_per_file=3)
    out = gh.unpack(SequentialRunner().run(plan).value)
    print(out["jet_pt"].view())
    print(out["met_pt"].view())
    print(out["jet_pt"].sum())

Printed output::

    [7 4 0 0]
    [3 2 1 0]
    11.0

Three things to notice. ``gh.boost.Histogram`` is a ``boost_histogram.Histogram`` — the eager API
(axes, storage, views of the empty state) is all still there, and ragged fill values flatten
completely, exactly as boost does them. There is no ``.compute()``: you build a plan and hand it
to a runner, and ``run(plan).value`` **is** the aggregated result. And you name your histograms
when you plan them, so the results come back keyed by those names.


Why filling is free until you run
---------------------------------

A fill is not computed when you write it. It is recorded as a step the runner performs later —
the same treatment a correction lookup or an ONNX model evaluation gets (``graphed`` calls these
*External* steps: calls out to machinery it does not look inside, carried in the recorded graph
with enough metadata to be re-run anywhere).

The consequence that matters is what the rest of the system does **not** have to know. Nothing in
``graphed`` itself, and nothing in any array backend, has an opinion about histograms. A fill is a
step with array inputs and an opaque result, so it rides through the optimizer, the plan
serializer and the workers on the same rails as every other step. The histogram machinery is
entirely on this side of the line: the description of the axes and storage, and a small picklable
object that fills one chunk into a fresh empty histogram.

That is also why filling is where independent handles finally meet. The axis values, the weights
and ``sample=`` may have been derived through different selections; ``fill`` re-indexes them all
into one common row space and applies the event weight the context is carrying, so you do not
line them up by hand.


Why writing the same fill twice costs once
------------------------------------------

Write the same expression twice and you get one recorded step, not two — identical expressions
collapse on the way in (*interning*). A fill is no exception: the same axes, the same storage and
the same inputs is the same step.

.. code-block:: python

    import awkward as ak
    import boost_histogram as bh
    import graphed_histogram as gh
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_awkward

    EVENTS = ak.Array({"Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0]])})})

    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", EVENTS)

    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())
    h.fill(events.Jet.pt[events.Jet.pt > 20.0])
    h.fill(events.Jet.pt[events.Jet.pt > 20.0])   # written out twice, letter for letter

    print(h.staged_fills(), "fills recorded")
    print(len({node.node_id for node in h.fill_nodes()}), "distinct node to evaluate")
    print(gh.spec_of(h))
    print(gh.content_hash(gh.spec_of(h)))
    print(session.materialize(h.fill_nodes()[0]).view())

Printed output::

    2 fills recorded
    1 distinct node to evaluate
    {"axes":[{"bins":4,"metadata":{},"overflow":true,"start":0.0,"stop":200.0,"type":"Regular","underflow":true}],"storage":"Int64","version":1}
    sha256:82860a80339d51cc3a2290596f9ebee5b9f957fe63934b41ee0e56e3b117ecc4
    [3 2 0 0]

``from_awkward`` wraps an array that is already in memory, which is all you need to look at what
was recorded. It has no chunks to hand out, though, so it cannot be planned: evaluate one of its
fills with ``session.materialize(node)`` — the last line above — and use a chunked source like
``from_parquet`` when you want ``.plan()``.

Two fills, one evaluation — and the result still counts both, because the histogram remembers it
asked for that step twice. You get the deduplication without changing what your program means.

The last two lines are the mechanism. A histogram's identity is a canonical description of its
axes and storage: sorted-key JSON, one entry per axis with its bins, edges and flow flags, plus
the axis attributes ``hist`` uses for ``name`` and ``label``. Its SHA-256 is what makes two fills
the same fill. Notice what is *not* in there: no pickled function, no memory address, no file
path — which is what lets the same description be rebuilt into the same empty histogram on
another machine.


Why several histograms cost one pass over the data
--------------------------------------------------

``gh.plan({...})`` takes all the histograms you want at once. Their fills compile into one graph,
so a selection feeding both a jet-pT histogram and a b-tag histogram is read and evaluated once —
not once per histogram, which is what planning each separately would cost you.
Only the columns that graph touches are read off disk, over the union of every histogram's fills.

Counting the reads takes a source you wrote yourself, which is also the shortest look at what a
source owes a runner: ``partitions(steps_per_file)`` names the slices the dataset will be cut into
(``Partition.blind`` when the row count is not known until a worker opens the file), and
``read_partition`` is handed one of them plus the columns the graph asked for, and returns that
slice. ``from_parquet`` is exactly this, written for you. The ``form`` says what the columns look
like without reading any of them — an awkward typetracer array, which is what
``to_typetracer(forget_length=True)`` builds — so the graph can be typed before a byte moves.

.. code-block:: python

    from dataclasses import dataclass, field

    import awkward as ak
    import boost_histogram as bh
    import graphed_histogram as gh
    from graphed import Session
    from graphed.awkward import AwkwardBackend, AwkwardForm
    from graphed.core import Partition
    from graphed.core.execution import SequentialRunner

    EVENTS = ak.Array({
        "MET": ak.zip({"pt": [10.0, 40.0, 70.0, 120.0, 30.0, 90.0]}),
        "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0],
                                       [80.0], [15.0, 45.0], [70.0, 10.0]])}),
    })

    @dataclass
    class CountingChunks:
        data: ak.Array
        reads: list = field(default_factory=list)

        def partitions(self, steps_per_file=1):
            return tuple(Partition.blind("mem://events", "", s, steps_per_file)
                         for s in range(steps_per_file))

        def read_partition(self, partition, columns, resources):
            part = partition.resolve(len(self.data))
            self.reads.append((part.entry_start, part.entry_stop))
            return self.data[part.entry_start:part.entry_stop]

    source = CountingChunks(EVENTS)
    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    events = session.source("events", form=form, data=source)

    selected = events.Jet.pt[events.Jet.pt > 20.0]        # the shared sub-graph
    lead = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Int64())
    wide = gh.boost.Histogram(bh.axis.Regular(2, 0.0, 200.0), storage=bh.storage.Int64())
    lead.fill(selected)
    wide.fill(selected)

    value = SequentialRunner().run(gh.plan({"lead": lead, "wide": wide}, steps_per_file=3)).value
    print(sorted(value))
    print(gh.unpack(value)["lead"].view(), gh.unpack(value)["wide"].view())
    print(len(source.reads), "partition reads for 2 histograms")

Printed output::

    ['lead', 'wide']
    [4 4 0 0] [8 0]
    3 partition reads for 2 histograms

Three partitions, three reads — for both histograms together. The run's value is a flat mapping,
which ``gh.unpack`` turns into the shape you want: ``{name: histogram}`` here, and
``{name: {label: histogram}}`` once variations are involved. Unpack it and index by the names you
planned with; the flat keys are the plan's business, not yours.

If you only ever have one histogram, ``h.plan(...)`` skips the naming and gives you the histogram
directly. It refuses fills that carry variations, because summing them into one histogram would
merge universes that need to stay apart — plan those through ``gh.plan({...})``.


Why the same analysis reproduces on another machine
---------------------------------------------------

Nothing about a fill is carried as pickled code. The recorded step carries the canonical axes and
storage description, a content hash of it, and the framework and version that produced it. On the
worker, the hash is what resolves the step to the object that performs it; the description is what
rebuilds the empty histogram to fill into. Both are plain declarative data.

Two things fall out of that. A plan re-run somewhere else resolves to the same fill by the same
hash, so "the same analysis" is a claim you can check rather than assume. And the description is
also the histogram's preservation payload — it is enough, on its own, to reconstruct the shape of
every result the analysis produced, without the code that produced it.

Named axes survive the trip. ``hist`` stores ``name`` and ``label`` as axis attributes; the
description captures and restores them, so a histogram that comes back from a cluster is still
indexable by axis name.


Why the total does not depend on how many workers you used
----------------------------------------------------------

Histograms add. Every standard boost storage supports ``+``, so a partial result from one
partition merges with a partial result from another in any order, and the runner's combine tree
needs no special case for histograms — it needs an empty value and an addition, and the axes
description supplies the first while boost supplies the second.

Integer counts are exact under any combine tree: whatever order the partials merge in, you get the
same integers. Float storages (``Weight``, ``Mean``, ``WeightedMean``) are a different matter,
because floating-point addition is not associative — ``(a + b) + c`` and ``a + (b + c)`` can differ
in the last bits. The runner fixes the merge order up front rather than letting it depend on which
worker finished first, so re-running the same plan on the same runner configuration reproduces the
same floats. Change the worker count and the merge tree changes shape with it, so the last bits
may move.


Running fills on several processes
----------------------------------

Needs ``graphed-executors`` alongside ``graphed`` and ``graphed-histogram``. A process pool spawns
its workers and each one re-imports your file, so everything with an effect goes under the usual
``if __name__ == "__main__":`` guard — including writing the data file, which a re-importing
worker would otherwise rewrite underneath the run.

.. code-block:: python

    import awkward as ak
    import boost_histogram as bh
    import graphed_histogram as gh
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_parquet
    from graphed.core.execution import SequentialRunner
    from graphed_executors.local import ProcessPoolExecutor

    EVENTS = ak.Array({"Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0],
                                                      [80.0], [15.0, 45.0], [70.0, 10.0]])})})

    def staged():
        session = Session(AwkwardBackend())
        events = from_parquet(session, "events", "jets.parquet", steps_per_file=3)
        h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Weight())
        h.fill(events.Jet.pt, weight=[events.Jet.pt * 0.01])
        return h

    if __name__ == "__main__":
        ak.to_parquet(EVENTS, "jets.parquet")
        plan = gh.plan({"jet_pt": staged()}, steps_per_file=3,
                       backend="graphed.awkward:AwkwardBackend")
        parallel = gh.unpack(ProcessPoolExecutor(max_workers=3).run(plan).value)["jet_pt"]
        serial = gh.unpack(SequentialRunner().run(plan).value)["jet_pt"]
        print(parallel.view()["value"])
        print(parallel.view()["variance"])
        print((parallel.view() == serial.view()).all())

Printed output::

    [1.85 2.65 0.   0.  ]
    [0.5875 1.7925 0.     0.    ]
    True

Values *and* variances match the single-process run exactly. The same plan object runs under
either runner; nothing about the plan is specific to how you execute it.

The one thing to get right is ``backend=``. Each worker builds its own array backend, and by
default it builds a bare one of the same class. If your backend carries behaviors — coffea-style
methods attached to your record types — those live in a dictionary of Python functions that does
not pickle, so pass an importable reference instead: ``backend="my_analysis:make_backend"``, a
zero-argument callable resolved in the worker. A worker that ends up without the behaviors it
needs raises when the analysis asks for one; it never quietly fills the wrong thing.

The same ``weight=`` shown above takes a *list* of factors — ``weight=[gen_weight, lepton_sf,
btag_sf]`` — multiplied elementwise at fill time. Each factor is a real input to the recorded
step, so a factor shared between two histograms is computed once for both. Pass
``unweighted=True`` to opt a fill out of both the explicit factors and the ambient event weight.


Five histograms, or one with a variation axis?
----------------------------------------------

You have a jet-energy-scale shift up and down, and a scale-factor up and down. That is five
universes counting the nominal one. Do you want five histograms, or one histogram with a
``variation`` axis you can slice?

Both are supported and the choice is yours. By default each universe is its own histogram, keyed
by label. Passing ``variation_axis=True`` to a fill instead puts them on a
``StrCategory("variation")`` axis of one histogram. The mode belongs to the histogram, not the
call: the first fill fixes it, and a later fill in the other mode is refused, as is one whose
label set disagrees with the first fill's — the declared axis would no longer describe the result.
Give a histogram an axis already named ``variation`` and axis mode is refused too, since it would
collide with the one the fill declares. An axis gets that name the way ``hist`` gives axes names:
boost axis constructors take no ``name=`` keyword, so the name is an attribute you set on the axis
— ``ax.__dict__["name"] = "variation"`` — and that is where both ``hist`` and the check look.

.. code-block:: python

    import awkward as ak
    import boost_histogram as bh
    import graphed
    import graphed_histogram as gh
    from graphed import Session, vary
    from graphed.awkward import AwkwardBackend, from_parquet
    from graphed.core.execution import SequentialRunner

    EVENTS = ak.Array({
        "MET": ak.zip({"pt": [10.0, 40.0, 70.0, 120.0, 30.0, 90.0]}),
        "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0],
                                       [80.0], [15.0, 45.0], [70.0, 10.0]])}),
    })
    ak.to_parquet(EVENTS, "events.parquet")

    def staged(*, variation_axis):
        session = Session(AwkwardBackend())
        events = from_parquet(session, "events", "events.parquet", steps_per_file=2)
        w = events.MET.pt * 0.01
        sf = vary(w, "sf", up=w * 1.2, down=w * 0.8)   # a per-event scale factor, up and down
        h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Weight())
        h.fill(events.Jet.pt, weight=[sf], variation_axis=variation_axis)
        return h

    def run(h):
        return gh.unpack(SequentialRunner().run(gh.plan({"h": h}, steps_per_file=2)).value)["h"]

    one = run(staged(variation_axis=True))
    print([type(ax).__name__ for ax in one.axes])
    print(graphed.labels(one))
    print(graphed.nominal(one).view()["value"])
    print(graphed.universe(one, "sf_up").view()["value"])

    many = run(staged(variation_axis=False))
    print(sorted(many))
    print(many["nominal"].view()["value"])

Printed output::

    ['Regular', 'StrCategory']
    ('nominal', 'sf_down', 'sf_up')
    [3.1 3.2 0.  0. ]
    [3.72 3.84 0.   0.  ]
    ['nominal', 'sf_down', 'sf_up']
    [3.1 3.2 0.  0. ]

The two modes agree bin for bin: the nominal slice of the axis-mode histogram and the ``nominal``
entry of the sibling-mode mapping are the same numbers. What differs is the container, and
``graphed.labels`` / ``graphed.nominal`` / ``graphed.universe`` read both shapes the same way — in
axis mode they slice the axis away, in sibling mode they index the mapping.

The axis-mode result is an ordinary ``bh.Histogram`` with one extra axis, so nothing downstream
needs a special case: it merges under ``+`` like any other, and plotting or ``hist`` indexing work
on it unchanged. The ``variation`` axis is a non-growth ``StrCategory`` whose categories are your
labels in sorted order, so two runs of the same program give you the same axis in the same order.
``graphed.labels`` reports them nominal-first, which is the order you usually want to read.


What folds onto the axis and what stays a sibling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rule of thumb: **a variation that only changes the weight folds onto the axis; one that
changes the values does not.**

That is not a policy, it is arithmetic. Filling a variation axis means filling the same value
column several times, once per label, with a different weight each time — so the values, the
selection and the per-object broadcast are computed once and reused across every label. That is
the payoff, and it only exists while the values are shared. A jet-energy-scale shift re-derives
``Jet.pt`` itself, and a varied ``sample=`` changes what is being averaged; neither can share a
value column, so each stays its own fill writing into its own category of the same axis.

.. code-block:: python

    import awkward as ak
    import boost_histogram as bh
    import graphed_histogram as gh
    from graphed import Session, vary
    from graphed.awkward import AwkwardBackend, from_awkward

    EVENTS = ak.Array({
        "MET": ak.zip({"pt": [10.0, 40.0, 70.0]}),
        "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0]])}),
    })

    def staged(*, vary_the_weight, vary_the_value):
        session = Session(AwkwardBackend())
        events = from_awkward(session, "events", EVENTS)
        w = events.MET.pt * 0.01
        pt = events.Jet.pt
        if vary_the_weight:
            w = vary(w, "sf", up=w * 1.2, down=w * 0.8)
        if vary_the_value:
            pt = vary(pt, "jes", up=pt * 1.05, down=pt * 0.95)
        h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Weight())
        h.fill(pt, weight=[w], variation_axis=True)
        return h

    for name, kwargs in [
        ("weight only    ", dict(vary_the_weight=True, vary_the_value=False)),
        ("value only     ", dict(vary_the_weight=False, vary_the_value=True)),
        ("weight + value ", dict(vary_the_weight=True, vary_the_value=True)),
    ]:
        h = staged(**kwargs)
        labels = gh.label_listing({"h": h})["h"]
        print(name, len(labels), "labels ->", h.staged_fills(), "fill node(s)")

Printed output::

    weight only     3 labels -> 1 fill node(s)
    value only      3 labels -> 3 fill node(s)
    weight + value  5 labels -> 3 fill node(s)

Read the third row: five universes, three fills. The nominal and both scale-factor labels share
one fill that loops over their weights; the two jet-energy-scale labels each get their own. In
general a mixed program costs one fill plus one per value-changing label — which is also the
number you would have paid in sibling mode for the value-changing ones alone. Every one of them
still lands in the same histogram, on the same axis.


Seeing your labels without running anything
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gh.label_listing({name: histogram})`` reports which variations reach each output, without
touching a file — the check you want before submitting a job, and the analogue of ROOT
RDataFrame's ``GetVariations``. It answers the same way in both modes, because it reads the labels
your fills declared rather than the shape of the result:

.. code-block:: python

    import awkward as ak
    import boost_histogram as bh
    import graphed_histogram as gh
    from graphed import Session, vary
    from graphed.awkward import AwkwardBackend, from_awkward

    EVENTS = ak.Array({
        "MET": ak.zip({"pt": [10.0, 40.0, 70.0]}),
        "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0]])}),
    })

    def staged(*, variation_axis, varied):
        session = Session(AwkwardBackend())
        events = from_awkward(session, "events", EVENTS)
        w = events.MET.pt * 0.01
        if varied:
            w = vary(w, "sf", up=w * 1.2, down=w * 0.8)
        h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Weight())
        h.fill(events.Jet.pt, weight=[w], variation_axis=variation_axis)
        return h

    print(gh.label_listing({
        "one":   staged(variation_axis=True,  varied=True),
        "many":  staged(variation_axis=False, varied=True),
        "plain": staged(variation_axis=False, varied=False),
    }))

Printed output::

    {'one': ['nominal', 'sf_up', 'sf_down'], 'many': ['nominal', 'sf_up', 'sf_down'], 'plain': ['nominal']}

Same labels in both modes, so the listing is a straight answer to "which variations does this
output carry" regardless of how you asked for them. The order is nominal first, then the order you
declared your variations in. An output no variation reaches lists ``['nominal']``.

For the fills themselves rather than their names, ``gh.fill_nodes_by_label(h)`` gives you
``{label: node}`` for a histogram with a single ``fill`` call — a histogram with several fill
calls has no one answer and says so instead of picking one.


The hist builder you already use
--------------------------------

``hist.graphed.Hist`` and ``hist.graphed.NamedHist`` give you the QuickConstruct builder and
named-axis fills unchanged. They live in a fork of ``hist`` that adds the ``hist.graphed`` module;
upstream ``hist`` does not ship it yet, so install it from there:

.. code-block:: bash

    pip install "hist @ git+https://github.com/graphed-org/hist-graphed-mvp@graphed-mvp"

.. code-block:: python

    import awkward as ak
    import hist
    import hist.graphed
    from graphed import Session
    from graphed.awkward import AwkwardBackend, from_parquet
    from graphed.core.execution import SequentialRunner

    EVENTS = ak.Array({"MET": ak.zip({"pt": [10.0, 40.0, 70.0, 120.0, 30.0, 90.0]})})
    ak.to_parquet(EVENTS, "met.parquet")

    session = Session(AwkwardBackend())
    events = from_parquet(session, "events", "met.parquet", steps_per_file=2)

    h = hist.graphed.Hist.new.Reg(4, 0, 200, name="met", label=r"$p_T^{miss}$").Double()
    h.fill(met=events.MET.pt)

    print(h.staged_fills(), "fill recorded")

    out = SequentialRunner().run(h.plan(steps_per_file=2)).value
    print(type(out).__name__, [axis.name for axis in out.axes], out.axes[0].label)

    plotted = hist.Hist(out)
    print(plotted[{"met": sum}], type(plotted).__name__)

Printed output::

    1 fill recorded
    Histogram ['met'] $p_T^{miss}$
    6.0 Hist

The fill records like any other, and you plan and run it exactly as above. Note the type on the
second line: a runner hands back a ``boost_histogram.Histogram``, not a ``hist.Hist`` — the names
and labels survive (they ride along in the axes description), but ``.plot()`` and name-based
indexing are ``hist``'s own additions. One call gets them back: ``hist.Hist(result)``, as the last
two lines show.


Not supported yet
-----------------

**Growth axes.** A category axis that grows as it sees new values cannot be combined across
partitions without a category-union merge, so constructing a deferred histogram with one raises
``TypeError`` right there. Declare the categories you expect up front:
``bh.axis.StrCategory(["ee", "emu", "mumu"])``.

**dask-style collection methods.** There is no ``persist`` or ``to_delayed``. A plan is a live
object your script builds, not a file format — rebuild it from the script and hand it to whichever
runner you have.

**Bit-identical float storages across different worker counts.** ``Weight``, ``Mean`` and
``WeightedMean`` reproduce exactly for a fixed runner configuration, not across configurations
that merge in a different tree shape. ``Int64`` counts are exact everywhere. If you need
bit-identical floats across machines, fix the worker count.

**Two datasets in one plan.** Every fill in a plan must record into the same session, and a
session plans against exactly one partitioned source. Run the datasets separately and add the
results — histograms add.

**Behaviors carried automatically to workers.** Pass ``backend="module:attr"`` when your backend
carries behaviors; there is no default that recovers them from the recording session.

See :doc:`improvements` for the tracked list.
