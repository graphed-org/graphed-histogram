"""Witnesses for the m48 review repairs — the mechanisms no frozen anchor reaches yet.

Frozen is law; these live here until m49 freezes real anchors for them. Each carries the mutation
that reds it, stated in the docstring so the check's own discriminating power is checkable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
import pytest
from graphed import GraphedError, compile_ir
from graphed.awkward import gak, gnano
from graphed.core.execution import SequentialRunner

# the frozen tree's own fixtures, without putting its dir on the packaged pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2] / "frozen" / "m48"))

from vary_hist_fixtures import (
    EVENTS,
    in_memory_events,
    partitioned_events,
    weighted,
    weighted_2d,
)

import graphed_histogram as gh
from graphed_histogram import boost


def _pu(source: Any, scale: float = 1.0) -> Any:
    return source.MET.pt * 0.01 * scale


def _weighted_context(source: Any) -> Any:
    events = gnano.events(source)
    return graphed.vary(events, "pu", _pu(events), is_weight=True, up=_pu(events, 1.1))


# --- A7/A6: the guard's identity is the blame COORDINATE, not the diagnostic's prose ------------
def _guarded_blob(monkeypatch: Any, factor_prose: str) -> bytes:
    monkeypatch.setattr(boost, "_FACTOR_ROWS", factor_prose)
    session, root = in_memory_events()
    ctx = _weighted_context(root)
    h = weighted(bins=20, lo=0.0, hi=400.0)
    h.fill(ctx.Jet.pt, weight=[ctx.MET.pt * 0.5 + 1.0])
    return bytes(compile_ir(session, *h.fill_nodes()).ir)


def test_rewording_a_diagnostic_moves_no_recorded_bytes(monkeypatch: Any) -> None:
    """A7: the message is evaluator-side only. Keying the payload hash on the prose instead of the
    coordinate reds this — and with it every varied fill's checkpoint identity."""
    before = _guarded_blob(monkeypatch, "weight[{index}] is not at this fill's row space")
    after = _guarded_blob(monkeypatch, "weight[{index}] sits at a different row count entirely")
    assert before == after


def test_two_different_offenders_stay_two_different_nodes() -> None:
    """A6: collapsing the guards' identities would let one evaluator answer for both coordinates,
    which mis-blames on the plan path. Ambient and explicit guards must differ."""
    session, root = in_memory_events()
    ctx = _weighted_context(root)
    h = weighted(bins=20, lo=0.0, hi=400.0)
    h.fill(ctx.Jet.pt, weight=[ctx.MET.pt * 0.5 + 1.0])
    guards = {
        node["params"]["blame"]
        for node in session._store.nodes()
        if node.get("params", {}).get("blame") is not None
    }
    assert guards == {"ambient", "weight[0]"}


# --- A6: the plan path blames the same operand the materialize path does -----------------------
def _offender_at(index: int) -> gh.boost.Histogram:
    """Two explicit factors, one of them per-EVENT against a flattened per-object value."""
    _session, source, _data = partitioned_events()
    events = gnano.events(source)
    value = gak.flatten(events.Jet.pt)
    fine = gak.flatten(events.Jet.pt * 0.0 + 1.0)
    offender = events.MET.pt * 0.01
    factors = [offender, fine] if index == 0 else [fine, offender]
    h = weighted(bins=20, lo=0.0, hi=400.0)
    h.fill(value, weight=factors)
    return h


@pytest.mark.parametrize("index", [0, 1])
def test_the_plan_path_names_the_same_offending_factor_as_materialize(index: int) -> None:
    with pytest.raises(GraphedError) as excinfo:
        SequentialRunner().run(gh.plan({"jets": _offender_at(index)}, steps_per_file=2))
    message = str(excinfo.value)
    assert f"weight[{index}]" in message
    assert f"weight[{1 - index}]" not in message


# --- A2: only the FIRST value can take the loose blame ----------------------------------------
def _two_axis(session_root: tuple[Any, Any], factor: Any) -> Any:
    """Axis 0 contexted, axis 1 LOOSE but at the correct row count — the shape that separates
    "any loose arg" from "args[0] is loose"."""
    _session, root = session_root
    events = gnano.events(root)
    sel = events[events.MET.pt > 20.0]
    loose_at_the_right_rows = root.MET.phi[root.MET.pt > 20.0]
    h = weighted_2d()
    h.fill(sel.MET.pt, loose_at_the_right_rows, weight=[factor(sel)])
    return h


def test_a_loose_value_away_from_index_zero_does_not_take_the_blame() -> None:
    """The guard compares each factor against the fill's FIRST value, so a loose value at another
    axis position is not what the comparison is about. Blaming on ANY loose arg reds this."""
    session, root = in_memory_events()
    h = _two_axis((session, root), lambda sel: gak.flatten(sel.Jet.pt))
    with pytest.raises(GraphedError) as excinfo:
        session.materialize(h.fill_nodes()[0])
    message = str(excinfo.value)
    assert "weight[0]" in message
    assert "value[0]" not in message


def test_the_same_program_with_a_matching_factor_runs() -> None:
    """The positive control: same shape, factor at the fill's row space."""
    session, root = in_memory_events()
    h = _two_axis((session, root), lambda sel: sel.MET.pt * 0.01)
    assert session.materialize(h.fill_nodes()[0]).sum(flow=True).value > 0


# --- A4: the merge refusal names the output whose fills actually merged ------------------------
def test_a_merge_inside_an_unvaried_sibling_names_that_sibling() -> None:
    """In a mixed plan the shortfall is global but the culprit need not be varied; naming only the
    varied outputs sends the reader to the wrong histogram with an inapplicable workaround."""
    _session, source, _data = partitioned_events()
    events = gnano.events(source)
    factor = events.MET.pt * 0.01

    varied = weighted()
    varied.fill(events.MET.pt, weight=[graphed.vary(factor, "sig", up=factor * 1.2)])
    merging = weighted()  # two fills the M4 identity rules collapse into one output
    merging.fill(events.MET.pt, weight=[factor])
    merging.fill(events.MET.pt, weight=[factor * 1.0])

    with pytest.raises(GraphedError) as excinfo:
        gh.plan({"varied": varied, "merging": merging}, steps_per_file=2)
    message = str(excinfo.value)
    assert "merging" in message
    assert "varied carries" not in message
    assert "variations=" not in message, "the workaround does not apply to an unvaried output"


# --- A5: ancestor-context weight factors are re-indexed ----------------------------------------
def test_an_ancestor_context_weight_factor_is_re_indexed_to_the_fill() -> None:
    """Mutating `factors = [reindex_to(w, ctx) ...]` to `list(weights)` reds this: the parent-space
    factor then reaches the guard at the parent's row count."""
    session, root = in_memory_events()
    events = gnano.events(root)
    parent_factor = events.MET.pt * 0.0 + 2.0  # read at the PARENT, applied at the child
    sel = events[events.MET.pt > 20.0]

    h = weighted()
    h.fill(sel.MET.pt, weight=[parent_factor])
    got = session.materialize(h.fill_nodes()[0])

    mask = ak.to_numpy(EVENTS.MET.pt) > 20.0
    want = bh.Histogram(*weighted().axes, storage=bh.storage.Weight())
    want.fill(ak.to_numpy(EVENTS.MET.pt)[mask], weight=np.full(int(mask.sum()), 2.0))
    assert np.array_equal(got.view(flow=True)["value"], want.view(flow=True)["value"])
    assert got.sum(flow=True).value > 0


# --- A9: one compile per plan, spied on every binding of the definition ------------------------
def test_one_compile_ir_per_group_plan_call(monkeypatch: Any) -> None:
    """§7.2's anti-quadratic budget. `compile_ir` is imported BY NAME in several modules, so each
    binding is patched from the defining function — patching one module's re-export alone would
    miss a call made through another's."""
    import graphed.execute as execute  # noqa: PLC0415  (the spy starts from the definition)

    calls: list[int] = []
    real = execute.compile_ir

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    for module in list(sys.modules.values()):
        if getattr(module, "compile_ir", None) is real:
            monkeypatch.setattr(module, "compile_ir", spy)

    _session, source, _data = partitioned_events()
    ctx = _weighted_context(source)
    h = weighted()
    h.fill(ctx.MET.pt)
    gh.plan({"met": h}, steps_per_file=2)
    assert calls == [1]
