"""m69b histserv backend: server-side histograms are created once, on first use in the driver process."""

from __future__ import annotations

import dataclasses
import json
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import boost_histogram as bh
import graphed
import pytest
from graphed.core.execution import LocalResources, Plan, SequentialRunner
from graphed.services import UnboundService, bind_services, require_bound, resolve_services
from graphed_exec_local import ProcessPoolExecutor
from histserv_harness import (
    assigned,
    bind,
    child_env,
    events,
    histserv_api,
    run_bounded,
    same,
    server_names,
    servers,
    write_events,
)

import graphed_histogram as gh

pytest.importorskip("histserv")


def _plan(path: str, tag: str | None, weight: str = "w") -> Plan[Any]:
    """Two contexts, so two servers: ``h`` and ``o``'s two labels on one, ``g`` on the other."""
    _session, ev = events(path)
    w = ev[weight]
    h = gh.boost.Histogram(bh.axis.Regular(20, -2.0, 6.0), storage=bh.storage.Weight())
    h.fill(ev.x, weight=[w])
    o = gh.boost.Histogram(bh.axis.Regular(6, -2.0, 6.0), storage=bh.storage.Double())
    o.fill(ev.x, weight=[graphed.vary(w, "sf", up=w * 2.0)])
    g = gh.boost.Histogram(bh.axis.Regular(9, -2.0, 6.0), storage=bh.storage.Int64())
    g.fill(ev.x)
    if tag is not None:
        hs = histserv_api()
        a = hs.Context(memory_mb=1024, workers=2, name=f"m69b-lazy-{tag}-a")
        b = hs.Context(memory_mb=1024, workers=2, name=f"m69b-lazy-{tag}-b")
        hs.backed(h, a)
        hs.backed(o, a)
        hs.backed(g, b)
    return gh.plan({"h": h, "o": o, "g": g}, steps_per_file=3)


def _per_server(plan: Plan[Any]) -> Counter[str]:
    return Counter(assigned(plan).values())


def test_require_bound_names_every_server_and_accepts_names_sharing_an_endpoint(tmp_path: Path) -> None:
    plan = _plan(write_events(str(tmp_path / "e.parquet")), "require")
    names = server_names(plan)
    assert len(names) >= 2
    with pytest.raises(UnboundService) as raised:
        require_bound(plan)
    assert sorted(raised.value.names) == sorted(names)
    require_bound(bind_services(plan, dict.fromkeys(names, "tcp://127.0.0.1:1")))


def test_histograms_are_created_on_the_first_call_once_and_on_a_bound_pickle_once(tmp_path: Path) -> None:
    plan = _plan(write_events(str(tmp_path / "e.parquet")), "first-use")
    want = _per_server(plan)
    first, second = sorted(plan.tasks, key=lambda t: t.key)[:2]
    res = LocalResources()
    with servers(len(plan.services)) as started:
        bound, by_name = bind(plan, started)

        def inits() -> dict[str, int]:
            return {name: server.count("Init") for name, server in by_name.items()}

        assert inits() == dict.fromkeys(by_name, 0)
        assert all(server.stats()["histogram_count"] == 0 for server in by_name.values())
        run_bounded(lambda: bound.process(first.partition, res))
        assert inits() == {name: want[name] for name in by_name}
        assert {n: s.stats()["histogram_count"] for n, s in by_name.items()} == inits()
        copy = dataclasses.replace(bound, tasks=(second,))
        run_bounded(lambda: copy.process(second.partition, res))
        pickle.dumps(plan.process)
        assert inits() == {name: want[name] for name in by_name}

        fresh, _ = bind(plan, started)
        run_bounded(lambda: pickle.dumps(fresh.process))
        run_bounded(lambda: pickle.dumps(fresh.process))
        assert inits() == {name: 2 * want[name] for name in by_name}


def test_a_process_pool_creates_once_and_equals_the_local_run(tmp_path: Path) -> None:
    path = write_events(str(tmp_path / "e.parquet"))
    plan = _plan(path, "pool", weight="wi")
    local = SequentialRunner().run(_plan(path, None, weight="wi")).value
    want = _per_server(plan)
    with servers(len(plan.services)) as started:
        bound, by_name = bind(plan, started)
        value = run_bounded(lambda: ProcessPoolExecutor(max_workers=2).run(bound).value)
        assert {n: s.count("Init") for n, s in by_name.items()} == {name: want[name] for name in by_name}
        got = resolve_services(bound, value)
    assert set(got) == set(local)
    assert all(same(got[slot], hist) for slot, hist in local.items())


_CHILD = """
import json, sys
import boost_histogram as bh
import graphed_histogram as gh
from graphed.core.execution import SequentialRunner
from graphed.services import UnboundService, bind_services, require_bound, resolve_services
from graphed_histogram import histserv
from histserv_harness import events
_session, ev = events(sys.argv[1])
hist = gh.boost.Histogram(bh.axis.Regular(20, -2.0, 6.0), storage=bh.storage.Weight())
hist.fill(ev.x, weight=[ev.w])
histserv.backed(hist, histserv.Context(memory_mb=1024, workers=1, name="m69b-lazy-child"))
plan = gh.plan({"h": hist}, steps_per_file=3)
try:
    require_bound(plan)
except UnboundService:
    pass
bound = bind_services(plan, {plan.services[0].name: sys.argv[2]})
before = [name for name in ("histserv", "grpc") if name in sys.modules]
SequentialRunner().run(bound)
print(json.dumps({"before": before, "after": "histserv" in sys.modules}))
"""


def test_nothing_imports_histserv_before_a_bound_run_does(tmp_path: Path) -> None:
    path = write_events(str(tmp_path / "e.parquet"))
    with servers(1) as started:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, path, started[0].endpoint],
            env=child_env(),
            capture_output=True,
            text=True,
            timeout=240,
            check=True,
        )
    lines = done.stdout.splitlines()
    assert len(lines) == 1, done.stdout + done.stderr
    assert json.loads(lines[0]) == {"before": [], "after": True}
