How graphed-histogram works
===========================

Histograms are how a HEP analysis ends: nearly every real query terminates in a fill.
``graphed-histogram`` makes that terminal step a first-class citizen of the deferred graph — a
``.fill(...)`` **records** instead of executing, the fill is an ordinary IR node with a
content-addressed identity, and aggregation is a task graph any executor runs. It is the
``dask-histogram`` analogue, built on graphed's own idioms.

Three design decisions define the package; each gets a section below: fills as *External
nodes* (so backends know nothing about histograms), the *canonical spec* as content identity,
and aggregation through *plans and executors* rather than a ``compute()`` method.

.. contents::
   :local:
   :depth: 2


The deferred histogram in one example
-------------------------------------

::

    import boost_histogram as bh
    import graphed_histogram as gh
    from graphed_core.execution import SequentialRunner

    h = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 10.0), storage=bh.storage.Int64())
    h.fill(x)                  # x is a graphed Array: RECORDS a fill node, returns h
    h.fill(x * 0.5 + 1.0)      # fills accumulate — more nodes, same histogram

    plan   = h.plan(steps_per_file=4)            # the R15.4 task graph
    result = SequentialRunner().run(plan).value  # a CONCRETE boost histogram
    # any R7 executor accepts the same plan:
    #   ProcessExecutor(max_workers=4, persistent=True).run(plan).value

The eager boost API stays available on ``h`` (axes, storage, views of the empty state); what
changed is that filling stages graph nodes and evaluation belongs to executors.


Fills are External nodes — backends know nothing
------------------------------------------------

A fill records through the frontend's ``record_external(descriptor=, form=)`` seam: the
package supplies the ``PayloadDescriptor`` (``kind="histogram"``,
``content_hash=sha256(spec)``, ``io_schema="uhi"``) and an opaque histogram form itself — the
backend is **never consulted**. This is the same architectural family as correctionlib and
ONNX nodes (M3): a call into foreign machinery, carried in the IR with reproducibility
metadata, evaluated by a registered evaluator.

The evaluator (``FillEvaluator``) is a frozen, picklable dataclass: given a chunk's input
arrays it builds a zero histogram from the spec and fills it — ragged inputs flatten
completely, weights and samples ride as additional *graph inputs* (never parameters).
``evaluate_ir`` resolves it by content hash through its ``externals=`` registry, failing
loudly if unregistered. Nothing in graphed-core, graphed, or any backend mentions histograms.

The canonical spec: identity you can ship
-----------------------------------------

A histogram's identity is the SHA-256 of its **canonical axes/storage spec** — versioned,
key-sorted JSON covering every supported axis (Regular/Variable/Integer/IntCategory/
StrCategory/Boolean, with flow flags) and storage (all standard boost storages), plus axis
user attributes: boost axes carry metadata like hist's ``name``/``label`` in their
``__dict__``, and the spec captures and restores it, so named axes survive a round trip.
``zero_of(spec)`` rebuilds the empty histogram anywhere; ``spec_of(h)`` is a fixed point
through rebuild (pinned).

Why this matters beyond tidiness: identical fills **intern** (same spec + same inputs = one
graph node); the spec string itself is the fill's *preservation payload* (graphed-preserve's
histogram plugin synthesizes it from the node's own parameters at bundle-build time); and a
plan re-run on another machine resolves its evaluator by the same hash. Identity, payload, and
registry key are one object.

Aggregation: plans and executors, not ``compute()``
---------------------------------------------------

There is deliberately no ``compute()`` method — evaluation is graphed's machinery, not a
collection protocol:

* ``h.plan(steps_per_file=..., partitions=..., backend=...)`` builds a
  ``Plan(process=fill-partition-through-the-compiled-IR, combine=native +, empty=zero)``. All
  of the histogram's fills compile into **one** multi-output graph, evaluated in a single pass
  per partition. Sources implementing the ``PartitionedSource`` protocol are read partition by
  partition — the whole-dataset loader is never invoked (counter-witnessed in the frozen
  suite). ``partitions=`` lets a caller shape chunking explicitly (benchmark sweeps use
  absolute entry counts).
* Any R7 executor's ``run(plan).value`` **is** the aggregated histogram. Histograms form a
  monoid under native ``+`` for every standard storage, so the executor's fixed combine tree
  applies unchanged: integer counts are exact under any tree; float storages are
  deterministic per fixed-tree configuration.
* The reference path for in-memory sources is ``session.materialize(fill_node)`` — the
  evaluated fill *is* a filled histogram — with the ``zero_of``/``add_histograms`` helpers for
  multi-fill sums.

Worker backends travel safely
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``plan(backend=...)`` accepts a zero-arg factory/class or an importable ``"module:attr"``
string resolved *in the worker* — the required form for behavior-carrying backends, because
behavior dicts contain lambdas and do not pickle. The failure direction is pinned the safe way
round: a worker built without required behaviors **fails loudly** on the behavior property; it
never silently fills the wrong thing. Weighted fills through a spawned pool are pinned exact
(values *and* variances) against the sequential run.

The hist integration
--------------------

``hist.graphed`` (in the ``hist`` fork) supplies ``Hist``/``NamedHist`` as thin MRO sandwiches
over this package's ``Histogram``: the familiar QuickConstruct
(``Hist.new.Reg(100, 0, 200, name="met").Double()``) and named-axis fills record deferred;
executor results wrap back into in-memory ``hist.Hist`` objects with names and labels intact
(they ride the canonical spec). The eight ADL benchmark queries run on exactly this surface.


Systematic variations: the variation axis
-----------------------------------------

A systematic variation is the same analysis re-run with one knob moved. ``graphed`` records it as
a family of labelled universes (``graphed.vary`` → a ``Varied``; see graphed's own design doc);
by default each universe is its own output — its own histogram. A fill can instead opt **one**
histogram into carrying those universes on a ``StrCategory("variation")`` axis, so N universes
land in ONE histogram rather than N siblings::

    from dataclasses import dataclass
    from typing import Any
    import awkward as ak, boost_histogram as bh
    import graphed, graphed_histogram as gh
    from graphed import Session, vary
    from graphed.awkward import AwkwardBackend, AwkwardForm, gnano
    from graphed.core import Partition
    from graphed.core.execution import SequentialRunner

    @dataclass
    class InMemorySource:                       # a minimal graphed.write.PartitionedSource
        data: ak.Array
        def __call__(self): raise AssertionError("the whole-dataset loader must not run")
        def partitions(self, steps_per_file=1):
            return tuple(Partition.blind("mem://events", "", s, steps_per_file)
                         for s in range(steps_per_file))
        def read_partition(self, partition, columns, resources):
            p = partition.resolve(len(self.data))
            return self.data[p.entry_start:p.entry_stop]

    EVENTS = ak.Array({
        "MET": ak.zip({"pt": [10.0, 40.0, 70.0, 120.0, 30.0, 90.0]}),
        "Jet": ak.zip({"pt": ak.Array([[40.0, 25.0], [55.0], [30.0, 60.0, 20.0],
                                       [80.0], [15.0, 45.0], [70.0, 10.0]])}),
    })

    def program(*, axis_mode):
        s = Session(AwkwardBackend())
        form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
        ctx = gnano.events(s.source("events", form=form, data=InMemorySource(EVENTS)))
        w  = ctx.MET.pt * 0.01
        wv = vary(w, "sf", up=w * 1.2, down=w * 0.8)         # a per-event varied weight factor
        h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 200.0), storage=bh.storage.Weight())
        h.fill(ctx.Jet.pt, weight=[wv], variation_axis=axis_mode)
        return h

    def run(h):                                              # plan -> execute -> unpack one output
        return gh.unpack(dict(SequentialRunner().run(gh.plan({"h": h}, steps_per_file=2)).value))["h"]

    axis = run(program(axis_mode=True))
    [type(a).__name__ for a in axis.axes]      # -> ['Regular', 'StrCategory']
    axis.axes[1].__dict__.get("name")          # -> 'variation'
    list(axis.axes[1])                          # -> ['nominal', 'sf_down', 'sf_up']  (stored: sorted)
    graphed.labels(axis)                        # -> ('nominal', 'sf_down', 'sf_up')  (nominal-first)
    graphed.nominal(axis).view()["value"]      # -> [3.1 3.2 0.  0. ]
    graphed.universe(axis, "sf_up").view()["value"]   # -> [3.72 3.84 0.   0.  ]

    sib = run(program(axis_mode=False))         # sibling mode: one histogram PER label
    sorted(sib)                                  # -> ['nominal', 'sf_down', 'sf_up']
    sib["nominal"].view()["value"]              # -> [3.1 3.2 0.  0. ]  (bit-for-bit vs the axis slice)

The opt-in is per ``fill()`` (``variation_axis=True``) and remembered by the histogram — a second
fill in the other mode on the same histogram is refused, as is a fill that disagrees with an
earlier fill's label set, or one onto a histogram the user already gave a ``"variation"`` axis.
The variation axis is a non-growth ``StrCategory`` whose bins are the inferred label set in
**sorted** order (recognised by ``axis.__dict__["name"] == "variation"`` — the ``hist`` metadata
convention the spec codec round-trips, since ``StrCategory(..., name=)`` is itself a ``TypeError``).

**What collapses and what stays a sibling.** Only *weight-label* universes fold into the axis: the
evaluator runs one inner loop over the weight variations, re-filling the same value column with
each label's weight — this is the mode's payoff (the per-object broadcast and value evaluation
happen once, not once per universe). Universes that change the *value* — a shift like a JES that
re-derives ``Jet.pt``, or a ``Varied`` ``sample=`` — cannot share the loop's value column and stay
**sibling** fills that target the same axis. So a mixed program's axis has arity ``1 + |S|`` (one
loop entry plus one per shift/sample label). Each fill node still carries its own
``content_hash((spec, variation_payload))``, so every universe resolves to its own evaluator.

The result is type-identical to an unvaried ``bh.Histogram`` carrying an extra axis, so
``graphed.labels``/``universe``/``nominal`` (graphed's introspection verbs) read it directly:
``labels`` reorders ``"nominal"``-first-then-axis-order while the stored bin order stays sorted,
and ``universe``/``nominal`` slice the axis away. Aggregation needs no special case — every slot's
value is a plain ``bh.Histogram``, a monoid under ``+``.

The plan-level label listing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``graphed_histogram.label_listing({name: Histogram})`` reports each output's variation labels
**without executing** — the RDF ``GetVariations`` analogue. It answers uniformly across both
modes, which is exactly why it cannot read the labels off the slot key: an axis-mode output
contributes one ``(output, None)`` slot that carries no label, so the listing reads that output's
labels from the fill's declared label set (the builder holds it while building the layout). The
labels come back in fold order — ``"nominal"`` first, then vary-tag insertion order::

    gh.label_listing({"axis": program(axis_mode=True),      # axis-mode varied
                      "sib":  program(axis_mode=False),      # sibling-mode, same variations
                      "plain": ...})                          # a fill no variation reaches
    # -> {'axis': ['nominal', 'sf_up', 'sf_down'],
    #     'sib':  ['nominal', 'sf_up', 'sf_down'],   # MODE-independent: same variations, same list
    #     'plain': ['nominal']}


Phase 2 (deliberately not built)
--------------------------------

* **Growth axes** — combining grown category axes across partitions needs a category-union
  merge; rejected explicitly at spec time for now.
* **Dask-style collection protocols** (``persist``, ``to_delayed``) — the durable artifact is
  the compiled IR / ``DurablePlan``; no parallel collection API is planned.
* **Behavior-reference forwarding by default** — the ``"module:attr"`` mechanism exists;
  defaulting it from the recording session (rather than the bare backend class) awaits the
  same behavior-reference carriage as preservation.

See :doc:`improvements` for the live tracked list.
