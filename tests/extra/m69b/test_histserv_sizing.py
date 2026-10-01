"""m69b histserv sizing beyond the frozen rows: flow extents and chunks per slot kind, refusals that
leave the packing untouched, and a served plan keeping the services it already had."""

from __future__ import annotations

import sys
from pathlib import Path

import boost_histogram as bh
import graphed
import pytest
from graphed import GraphedError, aggregate_plan
from graphed.services import ServiceSpec

sys.path.append(str(Path(__file__).resolve().parents[2] / "frozen" / "m69b"))

from histserv_harness import Slot, events, predicted, write_events

import graphed_histogram as gh
from graphed_histogram import histserv


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> str:
    return write_events(str(tmp_path_factory.mktemp("m69b-extra-sizing") / "e.parquet"))


def test_slots_are_sized_by_flow_extents_with_one_chunk_per_axis_label(path: str) -> None:
    """Axis mode fills a chunk per label over the non-variation axes' flow extents; each sibling
    label is a slot of its own. Sizing by bins (no flow) or by the variation axis reds this."""
    ctx = histserv.Context(memory_mb=1024, workers=2, name="m69b-extra-extents")
    _session, ev = events(path)
    w = graphed.vary(ev.w, "sf", up=ev.w * 2.0, down=ev.w * 0.5)
    axis = gh.boost.Histogram(
        bh.axis.StrCategory(["ee", "mm"]), bh.axis.Boolean(), storage=bh.storage.Weight()
    )
    axis.fill(ev.c, ev.b, weight=[w], variation_axis=True)
    sibling = gh.boost.Histogram(bh.axis.IntCategory([0, 1, 2]), storage=bh.storage.Double())
    sibling.fill(ev.k, weight=[w])
    plan = gh.plan({"a": histserv.backed(axis, ctx), "s": histserv.backed(sibling, ctx)}, steps_per_file=3)
    slots = [Slot(3, 16 * 3 * 2, 3)] + [Slot(1, 8 * 4, 3)] * 3
    ((_name, size, got, count),) = ctx.servers()
    assert (size, count) == (1024, 4) and len(plan.tasks) == 3
    assert got == pytest.approx(predicted(slots, workers=2), abs=1)


def test_a_refused_serve_opens_no_server(path: str) -> None:
    """Every refusal precedes placement: refusing after packing would leave servers no plan
    declares, and the next serve would pack around them."""
    ctx = histserv.Context(memory_mb=160, workers=1, name="m69b-extra-refused")
    _session, ev = events(path)
    hist = gh.boost.Histogram(bh.axis.Regular(8, 0.0, 1.0), storage=bh.storage.Double())
    hist.fill(ev.x)
    part = gh.plan({"u": hist}, steps_per_file=3).tasks[0].partition
    with pytest.raises(GraphedError):
        gh.plan({"h": histserv.backed(hist, ctx)}, partitions=[part, part])
    assert ctx.servers() == ()
    gh.plan({"h": hist}, partitions=[part])
    assert [count for *_rest, count in ctx.servers()] == [1]


def test_a_served_plan_keeps_its_own_services_and_declares_the_servers(path: str) -> None:
    """``plan.services`` is the plan's own union the servers, by name; the servers are declared on
    the session, so a later plan may name them."""
    ctx = histserv.Context(memory_mb=1024, workers=1, name="m69b-extra-union")
    session, ev = events(path)
    other = ServiceSpec("m69b-extra-a-service", kind="other")
    session.declare_service(other)
    hist = gh.boost.Histogram(bh.axis.Regular(8, 0.0, 1.0), storage=bh.storage.Double())
    hist.fill(ev.x)
    p = gh.boost.pieces({"h": histserv.backed(hist, ctx)})
    built = aggregate_plan(
        *p.fill_nodes,
        reduce=p.reduce,
        combine=p.combine,
        empty=p.empty,
        externals=p.externals,
        on_compiled=p.on_compiled,
        services=[other.name],
        steps_per_file=3,
    )
    plan = p.serve(built)
    assert [s.name for s in plan.services] == [other.name, "m69b-extra-union-0"]
    assert session.service_for("m69b-extra-union-0") == plan.services[1]


def test_a_context_without_workers_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\bworkers\b"):
        histserv.Context(memory_mb=1024, workers=0, name="m69b-extra-no-workers")


def test_a_context_differing_only_in_ports_is_refused_naming_the_remedy() -> None:
    histserv.Context(memory_mb=1024, workers=1, name="m69b-extra-ports")
    with pytest.raises(ValueError, match=r"fresh name=.*new process"):
        histserv.Context(memory_mb=1024, workers=1, name="m69b-extra-ports", ports=(20000, 20100))
