"""m69b histserv backend: the fill path from a partition's reduce to the server and back, run by hand."""

from __future__ import annotations

import pickle
import socket
from collections import Counter
from pathlib import Path
from typing import Any

import boost_histogram as bh
import graphed
import numpy as np
import pytest
from graphed import GraphedError
from graphed.core.execution import Plan, SequentialRunner
from graphed.services import bind_services, resolve_services
from histserv_harness import (
    MiB,
    assigned,
    bind,
    events,
    histserv_api,
    predicted,
    run_bounded,
    same,
    server_names,
    servers,
    slot_of,
    write_events,
)

import graphed_histogram as gh

pytest.importorskip("histserv")

REFUSED = (GraphedError, TypeError, ValueError)
SIZE_MB = 136
#: Double over 400000 bins: its alone prediction is past every size the context offers
BIG_BINS = 400_000


def _hists(path: str) -> dict[str, gh.boost.Histogram]:
    _session, ev = events(path)
    w = graphed.vary(ev.w, "sf", up=ev.w * 1.5, down=ev.w * 0.5)
    out = {
        "wax": gh.boost.Histogram(bh.axis.Regular(10_000, -4.0, 8.0), storage=bh.storage.Weight()),
        "dsib": gh.boost.Histogram(bh.axis.Regular(10, -4.0, 8.0), storage=bh.storage.Double()),
        "cnt": gh.boost.Histogram(bh.axis.Regular(10_000, -4.0, 8.0), storage=bh.storage.Int64()),
        "cat": gh.boost.Histogram(
            bh.axis.StrCategory(["ee", "mm", "em"]),
            bh.axis.Regular(8, -4.0, 8.0),
            storage=bh.storage.Weight(),
        ),
        "ib": gh.boost.Histogram(
            bh.axis.IntCategory([0, 1, 2]), bh.axis.Boolean(), storage=bh.storage.Double()
        ),
        "big": gh.boost.Histogram(bh.axis.Regular(BIG_BINS, -4.0, 8.0), storage=bh.storage.Double()),
        "local": gh.boost.Histogram(bh.axis.Regular(20, -4.0, 8.0), storage=bh.storage.Weight()),
    }
    out["wax"].fill(ev.x, weight=[w], variation_axis=True)
    out["dsib"].fill(ev.x, weight=[w])
    out["cnt"].fill(ev.x)
    out["cat"].fill(ev.c, ev.x, weight=[w], variation_axis=True)
    out["ib"].fill(ev.k, ev.b, weight=[w])
    out["big"].fill(ev.x, weight=[ev.w])
    out["local"].fill(ev.x, weight=[ev.w])
    return out


def _plans(path: str, name: str) -> tuple[Plan[Any], Plan[Any], Any]:
    hs = histserv_api()
    ctx = hs.Context(memory_mb=SIZE_MB, workers=1, name=name)
    backed = _hists(path)
    for key, hist in backed.items():
        if key != "local":
            hs.backed(hist, ctx)
    return gh.plan(backed, steps_per_file=3), gh.plan(_hists(path), steps_per_file=3), ctx


def _stored(slot: Any) -> int:
    """The model's ``stored`` bytes for each backed slot of ``_hists``."""
    name = slot if isinstance(slot, str) else slot[0]
    axes = {
        "wax": ((bh.axis.Regular(10_000, -4.0, 8.0),), "Weight", 3),
        "dsib": ((bh.axis.Regular(10, -4.0, 8.0),), "Double", 1),
        "cnt": ((bh.axis.Regular(10_000, -4.0, 8.0),), "Int64", 1),
        "cat": ((bh.axis.StrCategory(["ee", "mm", "em"]), bh.axis.Regular(8, -4.0, 8.0)), "Weight", 3),
        "ib": ((bh.axis.IntCategory([0, 1, 2]), bh.axis.Boolean()), "Double", 1),
        "big": ((bh.axis.Regular(BIG_BINS, -4.0, 8.0),), "Double", 1),
    }[name]
    return slot_of(axes[0], axes[1], tasks=3, labels=axes[2]).stored


def test_receipts_ride_the_tree_and_each_server_holds_exactly_its_slots(tmp_path: Path) -> None:
    hs = histserv_api()
    path = write_events(str(tmp_path / "e.parquet"))
    plan, twin_plan, _ctx = _plans(path, "m69b-fill-path")
    twin = SequentialRunner().run(twin_plan).value
    where = assigned(plan)
    per_server = Counter(where.values())
    assert len(per_server) >= 2
    assert per_server[where["big"]] == 1
    big = slot_of((bh.axis.Regular(BIG_BINS, -4.0, 8.0),), "Double", tasks=len(plan.tasks))
    assert predicted([big], workers=1) > SIZE_MB * MiB
    # the category axes' overflow bins (last along the flow view's first axis) hold fills
    assert np.asarray(twin[("cat", None)].view(flow=True))["value"][-1].sum() != 0
    assert np.asarray(twin[("ib", "nominal")].view(flow=True))[-1].sum() != 0
    with servers(len(plan.services)) as started:
        bound, by_name = bind(plan, started)
        value = run_bounded(lambda: SequentialRunner().run(bound).value)
        assert set(value) == set(twin)
        assert not isinstance(value["local"], hs.Receipt) and same(value["local"], twin["local"])
        for slot, home in where.items():
            receipt = value[slot]
            assert isinstance(receipt, hs.Receipt)
            assert receipt.endpoint == by_name[home].endpoint
            assert len(pickle.dumps(receipt)) < 1024
        assert all(len(pickle.dumps(twin[slot])) > 64 * 1024 for slot in (("wax", None), "cnt", "big"))
        for name, server in by_name.items():
            stats = server.stats()
            mine = [slot for slot, s in where.items() if s == name]
            assert stats["histogram_count"] == len(mine)
            assert stats["histogram_bytes"] == sum(_stored(slot) for slot in mine)
            assert server.count("Snapshot") == 0
            assert server.count("FillMany") == len(mine) * len(plan.tasks)
        for _ in range(2):
            unpacked = run_bounded(lambda: gh.unpack(value))
            expected = gh.unpack(twin)
            assert set(unpacked) == set(expected)
            for key, hist in expected.items():
                if isinstance(hist, dict):
                    got = unpacked[key]
                    assert isinstance(got, dict) and set(got) == set(hist)
                    assert all(same(got[label], h) for label, h in hist.items())
                else:
                    assert same(unpacked[key], hist)
            for name, server in by_name.items():
                assert server.stats()["histogram_count"] == sum(1 for s in where.values() if s == name)
        resolved = run_bounded(lambda: resolve_services(bound, value))
        assert set(resolved) == set(twin) and all(same(resolved[slot], twin[slot]) for slot in twin)
        assert all(server.stats()["histogram_count"] == 0 for server in by_name.values())


def test_server_names_bound_to_one_endpoint_run_on_one_server(tmp_path: Path) -> None:
    hs = histserv_api()
    path = write_events(str(tmp_path / "e.parquet"))
    plan, twin_plan, _ctx = _plans(path, "m69b-fill-shared-endpoint")
    names = server_names(plan)
    assert len(names) >= 2
    twin = SequentialRunner().run(twin_plan).value
    with servers(1) as (only,):
        bound = bind_services(plan, dict.fromkeys(names, only.endpoint))
        value = run_bounded(lambda: SequentialRunner().run(bound).value)
        receipts = {slot: value[slot] for slot in assigned(plan)}
        assert all(isinstance(r, hs.Receipt) and r.endpoint == only.endpoint for r in receipts.values())
        assert len({r.hist_id for r in receipts.values()}) == len(receipts)
        assert only.stats()["histogram_count"] == len(receipts)
        resolved = run_bounded(lambda: resolve_services(bound, value))
    assert all(same(resolved[slot], twin[slot]) for slot in twin)


@pytest.mark.parametrize("scheme", ["grpcs", "https", "http"])
def test_a_tls_or_http_endpoint_is_refused_and_never_dialled(tmp_path: Path, scheme: str) -> None:
    path = write_events(str(tmp_path / "e.parquet"))
    plan, _twin, _ctx = _plans(path, f"m69b-fill-scheme-{scheme}")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        listener.setblocking(False)
        port = listener.getsockname()[1]
        with pytest.raises(REFUSED):
            bind_services(plan, dict.fromkeys(server_names(plan), f"{scheme}://127.0.0.1:{port}"))
        with pytest.raises(BlockingIOError):
            listener.accept()
