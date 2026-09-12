What changed
============

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
