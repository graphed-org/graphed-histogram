"""m69b histserv backend: sizing and packing at ``pieces.serve``, with no server and no allocation.

Sizes are written in MiB of ``stored`` bytes on one Double axis, so ``histserv_harness.predicted`` gives
each server's model prediction; the expected placements follow from it (stored descending, then key).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import boost_histogram as bh
import graphed
import pytest
from graphed import GraphedError, aggregate_plan
from graphed.aggregate import collate
from graphed.core.execution import Plan
from histserv_harness import (
    BASE,
    CEILING,
    MiB,
    Slot,
    assigned,
    boost_api,
    child_env,
    events,
    histserv_api,
    predicted,
    slot_of,
    write_events,
)

import graphed_histogram as gh

REFUSED = (GraphedError, TypeError, ValueError)
ARGV = (
    "{python}",
    "-m",
    "histserv",
    "--port",
    "{port}",
    "--prune-after-seconds",
    "315360000",
    "--log-level",
    "WARNING",
)
#: server sizes offset from the model's ``B``, so a re-measured ``B`` changes only ``BASE``
BASE_MB = BASE // MiB
SMALL_MB = BASE_MB + 11
LARGE_MB = BASE_MB + 32


def _bins(mib: float) -> int:
    return int(mib * MiB) // 8 - 2


def _slot(mib: float, tasks: int) -> Slot:
    return slot_of((bh.axis.Regular(_bins(mib), 0.0, 1.0),), "Double", tasks)


def _plan(path: str, steps: int, sizes: dict[str, tuple[float, Any]]) -> Plan[Any]:
    """``gh.plan`` over one Double histogram per name, ``sizes[name] = (stored MiB, context or None)``."""
    _session, ev = events(path, steps)
    hists = {}
    for name, (mib, ctx) in sizes.items():
        hist = gh.boost.Histogram(bh.axis.Regular(_bins(mib), 0.0, 1.0), storage=bh.storage.Double())
        hist.fill(ev.x)
        hists[name] = hist if ctx is None else histserv_api().backed(hist, ctx)
    return gh.plan(hists, steps_per_file=steps)


def _servers(ctx: Any) -> dict[str, Any]:
    return {name: (size, predicted_bytes, n) for name, size, predicted_bytes, n in ctx.servers()}


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> str:
    return write_events(str(tmp_path_factory.mktemp("m69b-packing") / "e.parquet"), n=4000)


def test_slots_land_first_fit_in_stored_then_key_order_with_the_model_prediction(path: str) -> None:
    hs = histserv_api()
    n = "m69b-pack-ffd"
    ctx = hs.Context(memory_mb=LARGE_MB, workers=1, name=n, ports=(20000, 20100), timeout_s=300.0)
    mib = {"z": 1.0, "k": 0.75, "q": 2.5, "b": 0.75, "m": 1.375}
    plan = _plan(path, 3, {key: (size, ctx) for key, size in mib.items()})
    want = {"q": f"{n}-0", "m": f"{n}-0", "b": f"{n}-0", "z": f"{n}-1", "k": f"{n}-1"}
    assert assigned(plan) == want
    held = {name: [_slot(mib[k], 3) for k, s in want.items() if s == name] for name in (f"{n}-0", f"{n}-1")}
    servers = _servers(ctx)
    assert set(servers) == set(held)
    for name, slots in held.items():
        size, got, count = servers[name]
        assert (size, count) == (LARGE_MB, len(slots))
        assert got == pytest.approx(predicted(slots, workers=1), abs=1)
        assert got <= LARGE_MB * MiB
    assert [s.name for s in plan.services] == [f"{n}-0", f"{n}-1"]
    for spec in plan.services:
        assert (spec.kind, spec.check, spec.ports, spec.timeout_s) == (
            "histserv",
            "tcp",
            (20000, 20100),
            300.0,
        )
        assert spec.launch is not None and spec.launch.argv == ARGV
        assert dict(spec.launch.resources) == {"memory_mb": LARGE_MB}


def _count(path: str, name: str, *, workers: int = 1, steps: int = 1, size: int = BASE_MB + 8) -> int:
    ctx = histserv_api().Context(memory_mb=size, workers=workers, name=name)
    plan = _plan(path, steps, {f"s{i:02d}": (0.25, ctx) for i in range(32)})
    servers = _servers(ctx)
    assert all(got <= s * MiB for s, got, _n in servers.values())
    assert len(plan.services) == len(servers)
    return len(servers)


def test_with_one_size_the_server_count_rises_with_workers_and_tasks_and_falls_with_the_size(
    path: str,
) -> None:
    by_workers = [_count(path, f"m69b-pack-workers-{w}", workers=w) for w in (1, 3, 6)]
    by_tasks = [_count(path, f"m69b-pack-tasks-{t}", steps=t) for t in (1, 1000, 4000)]
    by_size = [_count(path, f"m69b-pack-size-{s}", size=s) for s in (BASE_MB + 8, BASE_MB + 15, BASE_MB + 71)]
    assert by_workers[0] < by_workers[1] < by_workers[2]
    assert by_tasks[0] < by_tasks[1] < by_tasks[2]
    assert by_size[0] > by_size[1] > by_size[2]


#: one context offering s = SMALL_MB < L = LARGE_MB: ``only_l`` fits L alone, ``big`` fits neither
TWO_SIZE = {"big": 3.046875, "only_l": 2.75, "small1": 0.5, "small2": 0.5, "small3": 0.5, "tiny": 0.1}


def _two_size_plan(path: str, name: str) -> tuple[Plan[Any], Any]:
    ctx = histserv_api().Context(memory_mb=[LARGE_MB, SMALL_MB], workers=1, name=name)
    return _plan(path, 3, {key: (mib, ctx) for key, mib in TWO_SIZE.items()}), ctx


def test_one_context_with_two_sizes_opens_the_smallest_that_fits_and_never_refuses_on_size(path: str) -> None:
    n = "m69b-pack-two-size"
    plan, ctx = _two_size_plan(path, n)
    assert ctx.memory_mb == (SMALL_MB, LARGE_MB)
    alone = predicted([_slot(TWO_SIZE["big"], 3)], workers=1)
    assert alone > LARGE_MB * MiB and predicted([_slot(TWO_SIZE["only_l"], 3)], workers=1) > SMALL_MB * MiB
    want = {
        "big": f"{n}-0",
        "only_l": f"{n}-1",
        "small1": f"{n}-1",
        "tiny": f"{n}-1",
        "small2": f"{n}-2",
        "small3": f"{n}-2",
    }
    assert assigned(plan) == want
    sizes = {f"{n}-0": math.ceil(alone / MiB), f"{n}-1": LARGE_MB, f"{n}-2": SMALL_MB}
    servers = _servers(ctx)
    for name, size in sizes.items():
        held = [_slot(TWO_SIZE[k], 3) for k, s in want.items() if s == name]
        assert servers[name][0] == size and servers[name][2] == len(held)
        assert servers[name][1] == pytest.approx(predicted(held, workers=1), abs=1)
        assert servers[name][1] <= size * MiB
    launched = {s.name: s.launch for s in plan.services}
    assert {name: dict(launch.resources)["memory_mb"] for name, launch in launched.items() if launch} == sizes


def test_two_contexts_in_one_plan_never_share_a_server(path: str) -> None:
    hs = histserv_api()
    x = hs.Context(memory_mb=SMALL_MB, workers=1, name="m69b-pack-ctx-x")
    y = hs.Context(memory_mb=LARGE_MB, workers=1, name="m69b-pack-ctx-y")
    plan = _plan(path, 3, {"x_only_l": (2.8, x), "y_small": (0.5, y)})
    alone = predicted([_slot(2.8, 3)], workers=1)
    assert SMALL_MB * MiB < alone and predicted([_slot(2.8, 3), _slot(0.5, 3)], workers=1) <= LARGE_MB * MiB
    assert assigned(plan) == {"x_only_l": "m69b-pack-ctx-x-0", "y_small": "m69b-pack-ctx-y-0"}
    assert _servers(x) == {"m69b-pack-ctx-x-0": (math.ceil(alone / MiB), pytest.approx(alone, abs=1), 1)}
    assert [(name, size, count) for name, size, _p, count in y.servers()] == [
        ("m69b-pack-ctx-y-0", LARGE_MB, 1)
    ]


def _labelled(path: str, bins: int, ctx: Any) -> Plan[Any]:
    _session, ev = events(path)
    hist = gh.boost.Histogram(bh.axis.Regular(bins, 0.0, 1.0), storage=bh.storage.Weight())
    w = graphed.vary(ev.w, "sf", points={str(i): ev.w * (1 + (i + 1) / 64) for i in range(31)})
    hist.fill(ev.x, weight=[w], variation_axis=True)
    return gh.plan({"huge": histserv_api().backed(hist, ctx)}, steps_per_file=3)


def test_a_slot_past_the_message_ceiling_is_refused_naming_the_sizes(path: str) -> None:
    ctx = histserv_api().Context(memory_mb=LARGE_MB, workers=1, name="m69b-pack-ceiling")
    # 32 labels of 16 * (bins + 2) bytes: 1048446 bins put stored + 64 KiB exactly at the ceiling
    at = 32 * 16 * (1_048_446 + 2)
    assert at + 64 * 1024 == CEILING
    _labelled(path, 1_048_446, ctx)
    over = at + 32 * 16
    with pytest.raises(REFUSED) as raised:
        _labelled(path, 1_048_447, ctx)
    message = str(raised.value)
    assert re.search(r"\bhuge\b", message) and re.search(rf"\b{CEILING}\b", message)
    assert re.search(rf"\b({over}|{over + 64 * 1024})\b", message)


def test_an_adaptive_plan_a_repeated_partition_and_a_second_serve_are_refused(path: str) -> None:
    ctx = histserv_api().Context(memory_mb=LARGE_MB, workers=1, name="m69b-pack-refusals")
    _session, ev = events(path)

    def hist(back: bool = True) -> gh.boost.Histogram:
        made = gh.boost.Histogram(bh.axis.Regular(8, 0.0, 1.0), storage=bh.storage.Double())
        made.fill(ev.x)
        return histserv_api().backed(made, ctx) if back else made

    def built(p: Any) -> Plan[Any]:
        return aggregate_plan(
            *p.fill_nodes,
            reduce=p.reduce,
            combine=p.combine,
            empty=p.empty,
            externals=p.externals,
            on_compiled=p.on_compiled,
            steps_per_file=3,
        )

    p = boost_api().pieces({"h": hist()})
    adaptive = dataclasses.replace(built(p), next_tasks=lambda _ctx: None)
    with pytest.raises(REFUSED, match=r"\bnext_tasks\b"):
        p.serve(adaptive)

    part = gh.plan({"u": hist(back=False)}, steps_per_file=3).tasks[0].partition
    unbacked = gh.plan({"u": hist(back=False)}, partitions=[part, part])
    assert len(unbacked.tasks) == 2 and unbacked.services == ()
    with pytest.raises(REFUSED, match=r"(?i)\bpartitions?\b"):
        gh.plan({"h": hist()}, partitions=[part, part])

    once = boost_api().pieces({"h": hist()})
    plan = built(once)
    once.serve(plan)
    with pytest.raises(REFUSED, match=r"(?i)\bserved?\b"):
        once.serve(plan)

    local = boost_api().pieces({"u": hist(back=False)})
    plain = built(local)
    assert local.serve(plain) is plain


def test_one_context_over_two_plans_fills_the_open_servers_first_and_collate_starts_each_once(
    tmp_path: Path,
) -> None:
    ctx = histserv_api().Context(memory_mb=LARGE_MB, workers=1, name="m69b-pack-two-plans")
    a = _plan(write_events(str(tmp_path / "a.parquet"), seed=1), 3, {"a1": (2.8, ctx), "a2": (2.0, ctx)})
    b = _plan(write_events(str(tmp_path / "b.parquet"), seed=2), 3, {"b1": (1.5, ctx), "b2": (2.6, ctx)})
    n = "m69b-pack-two-plans"
    assert assigned(a) == {"a1": f"{n}-0", "a2": f"{n}-1"}
    assert assigned(b) == {"b2": f"{n}-2", "b1": f"{n}-1"}
    assert [s.name for s in collate({"A": a, "B": b}).services] == [f"{n}-0", f"{n}-1", f"{n}-2"]
    shared = predicted([_slot(2.0, 3), _slot(1.5, 3)], workers=1)
    assert _servers(ctx)[f"{n}-1"] == (LARGE_MB, pytest.approx(shared, abs=1), 2) and shared <= LARGE_MB * MiB


def test_an_equal_second_context_shares_the_first_ones_servers_without_overfilling(tmp_path: Path) -> None:
    hs = histserv_api()
    n = "m69b-pack-equal"
    first = hs.Context(memory_mb=LARGE_MB, workers=1, name=n)
    p1 = _plan(write_events(str(tmp_path / "one.parquet"), seed=1), 3, {"p1a": (2.5, first)})
    with pytest.warns(Warning, match=rf"\b{re.escape(n)}\b"):
        second = hs.Context(memory_mb=LARGE_MB, workers=1, name=n)
    p2 = _plan(
        write_events(str(tmp_path / "two.parquet"), seed=2), 3, {"p2a": (2.7, second), "p2b": (0.25, second)}
    )
    declared = {s.name for s in p1.services}
    assert assigned(p2)["p2a"] not in declared
    held: dict[str, list[Slot]] = {}
    for plan, mib in ((p1, {"p1a": 2.5}), (p2, {"p2a": 2.7, "p2b": 0.25})):
        for key, server in assigned(plan).items():
            held.setdefault(server, []).append(_slot(mib[key], 3))
    assert assigned(p2)["p2b"] in declared
    for server, slots in held.items():
        assert predicted(slots, workers=1) <= LARGE_MB * MiB
        assert _servers(second)[server][1] == pytest.approx(predicted(slots, workers=1), abs=1)
    assert first.servers() == second.servers()


HELD = "m69b-pack-held-name"


def test_a_name_holds_its_arguments_for_the_process() -> None:
    hs = histserv_api()
    # sizes no line number of this file can echo
    hs.Context(memory_mb=1500, workers=1, name=HELD)
    with pytest.raises(REFUSED) as raised:
        hs.Context(memory_mb=1700, workers=1, name=HELD)
    assert re.search(r"\b1500\b", str(raised.value)) and re.search(r"\b1700\b", str(raised.value))
    hs.Context(memory_mb=[SMALL_MB, LARGE_MB], workers=1, name=f"{HELD}-sizes")
    with pytest.warns(Warning, match=rf"\b{re.escape(HELD)}-sizes\b"):
        hs.Context(memory_mb=[LARGE_MB, SMALL_MB], workers=1, name=f"{HELD}-sizes")
    with pytest.raises(REFUSED):
        hs.Context(memory_mb=[SMALL_MB], workers=1, name=f"{HELD}-sizes")
    with pytest.raises(REFUSED):
        hs.Context(memory_mb=[], workers=1, name=f"{HELD}-empty")
    with pytest.raises(REFUSED):
        hs.Context(memory_mb=BASE_MB - 1, workers=1, name=f"{HELD}-below")


def test_a_later_test_still_finds_the_name_held() -> None:
    """Runs after ``test_a_name_holds_its_arguments_for_the_process`` in file order, in one process."""
    with pytest.raises(REFUSED):
        histserv_api().Context(memory_mb=1700, workers=1, name=HELD)


def test_plan_services_are_sorted_by_name(path: str) -> None:
    hs = histserv_api()
    z = hs.Context(memory_mb=LARGE_MB, workers=1, name="m69b-pack-sort-z")
    a = hs.Context(memory_mb=LARGE_MB, workers=1, name="m69b-pack-sort-a")
    plan = _plan(path, 3, {"big": (2.5, z), "small": (0.25, a)})
    assert [s.name for s in plan.services] == ["m69b-pack-sort-a-0", "m69b-pack-sort-z-0"]


_CHILD = """
import hashlib, pickle, sys
from test_histserv_packing import _two_size_plan
plan, _ctx = _two_size_plan(sys.argv[1], "m69b-pack-seed")
print(hashlib.sha256(pickle.dumps((plan.services, plan.process.slots))).hexdigest())
"""


def test_the_two_size_plan_packs_identically_under_any_hash_seed(path: str) -> None:
    digests = []
    for seed in ("1", "2"):
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, path],
            env=child_env(PYTHONHASHSEED=seed),
            capture_output=True,
            text=True,
            timeout=240,
            check=True,
        )
        lines = done.stdout.splitlines()
        assert len(lines) == 1, done.stdout + done.stderr
        digests.append(lines[0])
    assert digests[0] == digests[1]
    assert hashlib.sha256(b"").hexdigest() not in digests
