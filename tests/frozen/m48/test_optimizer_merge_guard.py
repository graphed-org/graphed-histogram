"""m48/H5 — §7.2: the two ways two labels can collapse onto one node, and the two answers.

RECORD-TIME collapse (§1.2) is SUPPORTED: two labels whose members are structurally identical
intern to one node id, and the result carries BOTH keys off one evaluated fill — the frontend owns
`(output, label) -> node id` and derives the position as the rank in the DEDUPLICATED id list, so
many labels may resolve to one position and the unpacker replicates that value.

OPTIMIZER collapse is REFUSED. The M4 reducer also merges DISTINCT record ids (`x * 1.0` is an
identity token), so two fills differing only in `weight=[w]` versus `weight=[w * 1.0]` record two
nodes and compile to ONE output. The sound key — the record-to-reduced map — does not exist until
m49, so m48 refuses rather than mis-slicing; a mis-slice surfaces as an opaque worker-side
`IndexError`. The guard's SITE is the group-plan builder, not `compile_ir` and not
`aggregate_plan`, and it fires over a VARIED program only.

§1.1's stringified-float families make `variations={s: w * float(s)}` — which contains a literal
`w * 1.0` member — a natural spelling, which is why this is guarded rather than documented.
"""

from __future__ import annotations

import graphed
import numpy as np
import pytest
from graphed import GraphedError, compile_ir
from graphed.core import GraphStore
from graphed.core.execution import SequentialRunner
from vary_hist_fixtures import in_memory_events, partitioned_events, weighted

import graphed_histogram as gh


def _merging(events: object) -> gh.boost.Histogram:
    """`sig_up`'s member is `nominal * 1.0`: a DISTINCT record id the optimizer merges away."""
    factor = events.MET.pt * 0.01
    h = weighted()
    h.fill(events.MET.pt, weight=[graphed.vary(factor, "sig", up=factor * 1.0)])
    return h


def _deduping(events: object) -> gh.boost.Histogram:
    """`sig_up`'s member IS nominal's: §1.2's record-time dedup, which is supported."""
    factor = events.MET.pt * 0.01
    h = weighted()
    h.fill(events.MET.pt, weight=[graphed.vary(factor, "sig", up=factor, down=factor * 0.8)])
    return h


def test_the_optimizer_merge_is_real_before_the_guard_is_asserted() -> None:
    """The instrument: without this the refusal below could be firing on a program whose labels
    the optimizer never merged, and the guard would be untested."""
    session, events = in_memory_events()
    h = _merging(events)
    marked = [node.node_id for node in h.fill_nodes()]
    assert len(set(marked)) == 2, "the two fills must record as DISTINCT nodes"
    compiled = compile_ir(session, *h.fill_nodes())
    assert len(GraphStore.deserialize(compiled.ir).outputs()) == 1


def test_a_varied_program_whose_labels_the_optimizer_merges_is_refused() -> None:
    _session, events, _source = partitioned_events()
    with pytest.raises(GraphedError) as excinfo:
        gh.plan({"met": _merging(events)})
    message = str(excinfo.value)
    assert "met" in message
    assert "nominal" in message and "sig_up" in message
    assert "variations=" in message, "the refusal must carry the same-expression workaround"


def test_a_varied_program_the_optimizer_does_not_merge_plans_normally() -> None:
    """The guard is scoped to the shortfall; an ordinary varied program must not trip it."""
    _session, events, _source = partitioned_events()
    factor = events.MET.pt * 0.01
    h = weighted()
    h.fill(events.MET.pt, weight=[graphed.vary(factor, "sig", up=factor * 1.2, down=factor * 0.8)])
    result = gh.unpack(SequentialRunner().run(gh.plan({"met": h}, steps_per_file=3)).value)
    assert list(result["met"]) == ["nominal", "sig_up", "sig_down"]


def test_a_record_time_dedup_keeps_both_keys_off_one_evaluated_fill() -> None:
    """§1.2's dedup case, the result-mapping half. Three labels, TWO distinct fill nodes: the
    shared id replicates into both slots. A raw index into the undeduplicated fill list overruns
    the value list `evaluate_ir` returns, which is an opaque worker-side `IndexError`."""
    session, events = in_memory_events()
    h = _deduping(events)
    per_label = gh.fill_nodes_by_label(h)
    assert per_label["nominal"].node_id == per_label["sig_up"].node_id
    assert per_label["sig_down"].node_id != per_label["nominal"].node_id
    assert len({node.node_id for node in h.fill_nodes()}) == 2
    del session

    _s2, partitioned, _source = partitioned_events()
    shared = _deduping(partitioned)
    value = SequentialRunner().run(gh.plan({"met": shared}, steps_per_file=3)).value
    assert set(value) == {("met", "nominal"), ("met", "sig_up"), ("met", "sig_down")}
    nominal = np.asarray(value[("met", "nominal")].view(flow=True)["value"])
    assert np.array_equal(np.asarray(value[("met", "sig_up")].view(flow=True)["value"]), nominal)
    assert not np.array_equal(np.asarray(value[("met", "sig_down")].view(flow=True)["value"]), nominal)
