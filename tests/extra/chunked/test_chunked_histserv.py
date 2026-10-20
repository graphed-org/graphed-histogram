"""A histserv-backed slot sums dense for its server; a chunked slot beside it stays chunked."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path
from typing import Any

import boost_histogram as bh
import pytest
from graphed.core.execution import LocalResources, SequentialRunner
from graphed.services import resolve_services

sys.path.append(str(Path(__file__).resolve().parents[2] / "frozen" / "m69b"))

from histserv_harness import bind, events, run_bounded, same, servers, write_events

import graphed_histogram as gh
from graphed_histogram import histserv
from graphed_histogram._chunked import Chunked

pytestmark = pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")), reason="grpcio ships no free-threaded wheel"
)


def test_a_histserv_backed_slot_fills_dense_beside_a_chunked_one(tmp_path: Path) -> None:
    path = write_events(str(tmp_path / "e.parquet"))

    def plan(back: bool) -> Any:
        _session, ev = events(path)
        out = {
            name: gh.boost.Histogram(
                bh.axis.Regular(6, -2.0, 6.0), bh.axis.StrCategory(["ee", "mm"]), storage=bh.storage.Weight()
            ).fill(ev.x, ev.c, weight=ev.w)
            for name in ("served", "local")
        }
        if back:
            histserv.backed(out["served"], histserv.Context(memory_mb=256, workers=1, name="chunked-served"))
        else:
            for h in out.values():
                gh.dense_backed(h)
        return gh.plan(out, steps_per_file=3)

    backed, twin = plan(True), plan(False)
    with servers(len(backed.services)) as started:
        bound, _ = bind(backed, started)
        partial = run_bounded(lambda: bound.process(bound.tasks[0].partition, LocalResources()))
        assert isinstance(partial["served"], histserv.Receipt) and type(partial["local"]) is Chunked
        value = run_bounded(lambda: SequentialRunner().run(bound).value)
        assert type(value["local"]) is Chunked
        resolved = run_bounded(lambda: resolve_services(bound, value))
    want = SequentialRunner().run(twin).value
    assert all(same(resolved[k], want[k]) for k in ("served", "local"))
