"""m69b histogram pieces: fills composed with another output in one plan, read at their compiled positions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
import pytest
from graphed import Array, CompiledGraph, aggregate_plan, compile_ir
from graphed.aggregate import collate
from graphed.core import GraphStore
from graphed.core.execution import Plan, SequentialRunner
from graphed.services import resolve_services
from histserv_harness import (
    assigned,
    bind,
    boost_api,
    events,
    histserv_api,
    run_bounded,
    same,
    servers,
    write_events,
)

import graphed_histogram as gh

pytest.importorskip("histserv")


@dataclass
class Composed:
    """A composing plan's reduce: the histograms through ``pieces.reduce``, plus an event count."""

    hists: Any
    at: dict[str, int] = field(default_factory=dict)

    def __call__(self, values: list[Any]) -> dict[str, Any]:
        return {"hists": self.hists(values), "n": int(ak.sum(values[self.at["counter"]]))}

    def resolve_services(self, value: dict[str, Any]) -> dict[str, Any]:
        return {"hists": self.hists.resolve_services(value["hists"]), "n": value["n"]}


def _position(compiled: CompiledGraph, node: Array) -> int:
    order = {cid: i for i, cid in enumerate(GraphStore.deserialize(bytes(compiled.ir)).outputs())}
    return order[compiled.correspondence.node_map[node.node_id][0]]


def compose(p: Any, counter: Array, *, counter_first: bool, hook: bool = True) -> Plan[dict[str, Any]]:
    reduce = Composed(p.reduce)

    def on_compiled(compiled: CompiledGraph) -> Any:
        reduce.at["counter"] = _position(compiled, counter)
        return p.on_compiled(compiled)

    def skip_hook(compiled: CompiledGraph) -> None:
        reduce.at["counter"] = _position(compiled, counter)

    def combine(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return {"hists": p.combine(a["hists"], b["hists"]), "n": a["n"] + b["n"]}

    def empty() -> dict[str, Any]:
        return {"hists": p.empty(), "n": 0}

    marked = [counter, *p.fill_nodes] if counter_first else [*p.fill_nodes, counter]
    fire: Callable[[CompiledGraph], Any] = on_compiled if hook else skip_hook
    return aggregate_plan(
        *marked, reduce=reduce, combine=combine, empty=empty, externals=p.externals, on_compiled=fire
    )


def _weighted(ev: Any) -> gh.boost.Histogram:
    hist = gh.boost.Histogram(bh.axis.Regular(12, -2.0, 6.0), storage=bh.storage.Weight())
    hist.fill(ev.x, weight=[ev.w])
    return hist


def _merged(ev: Any) -> gh.boost.Histogram:
    hist = gh.boost.Histogram(bh.axis.Regular(12, -2.0, 6.0), storage=bh.storage.Weight())
    hist.fill(ev.x, weight=[ev.w])
    hist.fill(ev.x, weight=[ev.w * 1.0])
    return hist


def _direct(path: str, times: int) -> bh.Histogram:
    data = ak.from_parquet(path)
    want = bh.Histogram(bh.axis.Regular(12, -2.0, 6.0), storage=bh.storage.Weight())
    for _ in range(times):
        want.fill(ak.to_numpy(data.x), weight=ak.to_numpy(data.w))
    return want


def _hists(ev: Any, ctx: Any) -> dict[str, gh.boost.Histogram]:
    w = graphed.vary(ev.w, "sf", up=ev.w * 1.5)
    varied = gh.boost.Histogram(bh.axis.Regular(6, -2.0, 6.0), storage=bh.storage.Double())
    varied.fill(ev.x, weight=[w])
    out = {"h": _weighted(ev), "v": varied}
    if ctx is not None:
        for hist in out.values():
            histserv_api().backed(hist, ctx)
    return out


def test_a_composed_served_plan_equals_gh_plan_and_counts_events(tmp_path: Path) -> None:
    path = write_events(str(tmp_path / "e.parquet"))
    _session, ev = events(path)
    ctx = histserv_api().Context(memory_mb=1024, workers=1, name="m69b-pieces-compose")
    p = boost_api().pieces(_hists(ev, ctx))
    plan = p.serve(compose(p, ev.x * 0 + 1, counter_first=True))
    _twin_session, twin_ev = events(path)
    want = gh.unpack(SequentialRunner().run(gh.plan(_hists(twin_ev, None), steps_per_file=3)).value)
    with servers(len(plan.services)) as started:
        bound, _by_name = bind(plan, started)
        value = run_bounded(lambda: resolve_services(bound, SequentialRunner().run(bound).value))
    got = gh.unpack(value["hists"])
    assert value["n"] == len(ak.from_parquet(path))
    assert same(got["h"], want["h"])
    assert isinstance(got["v"], dict) and isinstance(want["v"], dict)
    assert set(got["v"]) == {"nominal", "sf_up"} and all(same(got["v"][k], want["v"][k]) for k in want["v"])


def test_an_unserved_composed_plan_fails_its_first_task_naming_serve(tmp_path: Path) -> None:
    _session, ev = events(write_events(str(tmp_path / "e.parquet")))
    ctx = histserv_api().Context(memory_mb=1024, workers=1, name="m69b-pieces-unserved")
    p = boost_api().pieces(_hists(ev, ctx))
    plan = compose(p, ev.x * 0 + 1, counter_first=True)
    with pytest.raises(Exception, match="serve"):
        run_bounded(lambda: SequentialRunner().run(plan))


def test_a_reduce_whose_hook_never_ran_fails_naming_on_compiled(tmp_path: Path) -> None:
    _session, ev = events(write_events(str(tmp_path / "e.parquet")))
    p = boost_api().pieces(_hists(ev, None))
    plan = compose(p, ev.x * 0 + 1, counter_first=True, hook=False)
    with pytest.raises(Exception, match="on_compiled"):
        run_bounded(lambda: SequentialRunner().run(plan))


def test_one_pieces_feeds_one_plan_and_the_first_still_runs_right(tmp_path: Path) -> None:
    path = write_events(str(tmp_path / "e.parquet"))
    _session, ev = events(path)
    counter = ev.x * 0 + 1
    p = boost_api().pieces({"h": _weighted(ev)})
    first = compose(p, counter, counter_first=True)
    with pytest.raises(Exception, match="pieces"):
        compose(p, counter, counter_first=False)
    value = SequentialRunner().run(first).value
    assert value["n"] == len(ak.from_parquet(path))
    assert same(gh.unpack(value["hists"])["h"], _direct(path, 1))


def test_gh_plan_of_backed_merged_fills_equals_a_direct_fill_done_twice(tmp_path: Path) -> None:
    path = write_events(str(tmp_path / "e.parquet"))
    _session, ev = events(path)
    hist = histserv_api().backed(
        _merged(ev), histserv_api().Context(memory_mb=1024, workers=1, name="m69b-pieces-merged")
    )
    plan = gh.plan({"h": hist}, steps_per_file=3)
    with servers(len(plan.services)) as started:
        bound, _by_name = bind(plan, started)
        value = run_bounded(lambda: resolve_services(bound, SequentialRunner().run(bound).value))
    assert same(value["h"], _direct(path, 2))


@pytest.mark.parametrize("counter_first", [True, False], ids=["counter-first", "counter-last"])
def test_merged_fills_composed_with_a_counter_each_read_once_per_marked_fill(
    tmp_path: Path, counter_first: bool
) -> None:
    path = write_events(str(tmp_path / "e.parquet"))
    session, ev = events(path)
    hists = {"b": _merged(ev), "l": _merged(ev)}
    pair = hists["b"].fill_nodes()
    assert len({n.node_id for n in pair}) == 2
    assert len(GraphStore.deserialize(bytes(compile_ir(session, *pair).ir)).outputs()) == 1
    tag = "first" if counter_first else "last"
    ctx = histserv_api().Context(memory_mb=1024, workers=1, name=f"m69b-pieces-merged-counter-{tag}")
    histserv_api().backed(hists["b"], ctx)
    p = boost_api().pieces(hists)
    plan = p.serve(compose(p, ev.x * 0 + 1, counter_first=counter_first))
    with servers(len(plan.services)) as started:
        bound, _by_name = bind(plan, started)
        value = run_bounded(lambda: resolve_services(bound, SequentialRunner().run(bound).value))
    twice = _direct(path, 2)
    got = gh.unpack(value["hists"])
    assert same(got["b"], twice) and same(got["l"], twice)
    assert np.asarray(twice.view(flow=True))["value"].sum() > 0
    assert value["n"] == len(ak.from_parquet(path))


def test_collated_served_plans_route_each_plan_to_its_own_servers(tmp_path: Path) -> None:
    paths = {
        name: write_events(str(tmp_path / f"{name}.parquet"), seed=seed)
        for name, seed in (("A", 1), ("B", 2))
    }
    plans, twins = {}, {}
    for name, path in paths.items():
        _session, ev = events(path)
        ctx = histserv_api().Context(memory_mb=1024, workers=1, name=f"m69b-pieces-collate-{name}")
        plans[name] = gh.plan(_hists(ev, ctx), steps_per_file=3)
        _twin_session, twin_ev = events(path)
        twins[name] = SequentialRunner().run(gh.plan(_hists(twin_ev, None), steps_per_file=3)).value
    collated = collate(plans)
    with servers(len(collated.services)) as started:
        bound, by_name = bind(collated, started)
        value = run_bounded(lambda: SequentialRunner().run(bound).value)
        for name, plan in plans.items():
            mine = {by_name[server].endpoint for server in assigned(plan).values()}
            others = {s.endpoint for s in by_name.values()} - mine
            receipts = [v for v in value[name].values() if isinstance(v, histserv_api().Receipt)]
            assert receipts and {r.endpoint for r in receipts} <= mine and others
        resolved = run_bounded(lambda: resolve_services(bound, value))
    for name, twin in twins.items():
        assert set(resolved[name]) == set(twin)
        assert all(same(resolved[name][slot], hist) for slot, hist in twin.items())
