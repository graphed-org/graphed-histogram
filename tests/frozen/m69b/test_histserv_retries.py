"""m69b histserv backend: a retried partition fills once, and a server failure surfaces as a picklable error."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import boost_histogram as bh
import graphed
import pytest
from graphed import GraphedError
from graphed.core.execution import LocalResources, Plan
from histserv_harness import bind, events, histserv_api, run_bounded, same, servers, write_events

import graphed_histogram as gh

histserv = pytest.importorskip("histserv")

REFUSED = (GraphedError, TypeError, ValueError)


def _plan(path: str, ctx: Any, back: bool) -> Plan[Any]:
    _session, ev = events(path)
    hist = gh.boost.Histogram(bh.axis.Regular(20, -2.0, 6.0), storage=bh.storage.Weight())
    hist.fill(ev.x, weight=[ev.w])
    other = gh.boost.Histogram(bh.axis.Regular(5, -2.0, 6.0), storage=bh.storage.Double())
    other.fill(ev.x, weight=[graphed.vary(ev.w, "sf", up=ev.w * 2.0)])
    if back:
        histserv_api().backed(hist, ctx)
        histserv_api().backed(other, ctx)
    return gh.plan({"h": hist, "o": other}, steps_per_file=3)


def test_a_partition_run_twice_fills_once_and_another_partition_adds(tmp_path: Path) -> None:
    hs = histserv_api()
    path = write_events(str(tmp_path / "e.parquet"))
    plan = _plan(path, hs.Context(memory_mb=1024, workers=1, name="m69b-retries-once"), back=True)
    twin = _plan(path, None, back=False)
    first, second = sorted(plan.tasks, key=lambda t: t.key)[:2]
    res = LocalResources()
    p0 = twin.process(first.partition, res)
    p01 = twin.combine(p0, twin.process(second.partition, res))
    with servers(len(plan.services)) as started:
        bound, _by_name = bind(plan, started)
        once = run_bounded(lambda: bound.process(first.partition, res))
        again = run_bounded(lambda: bound.process(first.partition, res))
        receipt = once["h"]
        assert isinstance(receipt, hs.Receipt) and again["h"].hist_id == receipt.hist_id
        assert started[0].address in receipt.endpoint
        remote = histserv.Client(started[0].address).connect(receipt.hist_id)
        assert remote.was_filled_with_unique_id(str(first.partition))
        assert not remote.was_filled_with_unique_id(str(second.partition))
        assert same(gh.unpack({"h": receipt})["h"], p0["h"])
        run_bounded(lambda: bound.process(second.partition, res))
        assert remote.was_filled_with_unique_id(str(second.partition))
        assert same(gh.unpack({"h": receipt})["h"], p01["h"])
        assert not same(p01["h"], p0["h"])


def test_a_killed_server_raises_a_histserv_error_that_survives_a_pickle(tmp_path: Path) -> None:
    hs = histserv_api()
    path = write_events(str(tmp_path / "e.parquet"))
    plan = _plan(path, hs.Context(memory_mb=1024, workers=1, name="m69b-retries-killed"), back=True)
    first, second = sorted(plan.tasks, key=lambda t: t.key)[:2]
    res = LocalResources()
    with servers(len(plan.services)) as started:
        bound, _by_name = bind(plan, started)
        run_bounded(lambda: bound.process(first.partition, res))
        started[0].kill()
        with pytest.raises(hs.HistservError) as raised:
            run_bounded(lambda: bound.process(second.partition, res))
    err = raised.value
    assert re.search(rf"\b{re.escape(started[0].address)}\b", str(err)) and re.search(
        r"\bUNAVAILABLE\b", str(err)
    )
    assert started[0].address in err.endpoint and err.code == "UNAVAILABLE"
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is type(err) and str(back) == str(err)
    assert (back.endpoint, back.code, back.details) == (err.endpoint, err.code, err.details)


def test_receipts_of_two_server_histograms_refuse_to_add(tmp_path: Path) -> None:
    hs = histserv_api()
    path = write_events(str(tmp_path / "e.parquet"))
    plan = _plan(path, hs.Context(memory_mb=1024, workers=1, name="m69b-retries-add"), back=True)
    first = sorted(plan.tasks, key=lambda t: t.key)[0]
    with servers(len(plan.services)) as started:
        bound, _by_name = bind(plan, started)
        value = run_bounded(lambda: bound.process(first.partition, LocalResources()))
    h, o = value["h"], value[("o", "nominal")]
    assert (h + h).hist_id == h.hist_id
    with pytest.raises(REFUSED) as raised:
        h + o
    assert all(re.search(rf"\b{hist_id}\b", str(raised.value)) for hist_id in (h.hist_id, o.hist_id))
