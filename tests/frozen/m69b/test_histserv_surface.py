"""m69b histserv backend: the surface a user touches before any server runs."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import numpy as np
import pytest
from graphed import GraphedError, compile_ir
from graphed.core import GraphStore
from graphed.core.execution import SequentialRunner
from graphed.execute import external_key
from histserv_harness import events, histserv_api, same, write_events

import graphed_histogram as gh

REFUSED = (GraphedError, TypeError, ValueError)
FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
HAS_HISTSERV = importlib.util.find_spec("histserv") is not None


def _fills(ev: Any, hist: gh.boost.Histogram) -> gh.boost.Histogram:
    hist.fill(ev.x, ev.c, weight=[ev.w])
    return hist


def _axes() -> tuple[Any, ...]:
    return (bh.axis.Regular(12, -2.0, 6.0), bh.axis.StrCategory(["ee", "mm", "em"]))


def test_a_histserv_histogram_is_a_deferred_histogram_and_backed_returns_its_argument() -> None:
    hs = histserv_api()
    ctx = hs.Context(memory_mb=1024, workers=1, name="m69b-surface-subclass")
    made = hs.Histogram(*_axes(), storage=bh.storage.Weight(), context=ctx)
    assert isinstance(made, gh.boost.Histogram)
    plain = gh.boost.Histogram(*_axes(), storage=bh.storage.Weight())
    assert hs.backed(plain, ctx) is plain


def test_backing_leaves_the_graph_identity_of_the_unbacked_twin(tmp_path: Path) -> None:
    hs = histserv_api()
    ctx = hs.Context(memory_mb=1024, workers=1, name="m69b-surface-identity")
    path = write_events(str(tmp_path / "e.parquet"))
    irs, keys = [], []
    for back in ("constructed", "before-fill", "after-fill", None):
        session, ev = events(path)
        if back == "constructed":
            hist = hs.Histogram(*_axes(), storage=bh.storage.Weight(), context=ctx)
        else:
            hist = gh.boost.Histogram(*_axes(), storage=bh.storage.Weight())
        if back == "before-fill":
            hs.backed(hist, ctx)
        _fills(ev, hist)
        if back == "after-fill":
            hs.backed(hist, ctx)
        ir = bytes(compile_ir(session, *hist.fill_nodes()).ir)
        irs.append(ir)
        keys.append(
            sorted(
                external_key(n)
                for n in GraphStore.deserialize(ir).nodes()
                if n["kind"] == "external"
            )
        )
    assert all(ir == irs[-1] for ir in irs)
    assert all(key == keys[-1] for key in keys) and keys[-1]


@pytest.mark.parametrize(
    ("axis", "tag"),
    [
        (bh.axis.Regular(4, 0.0, 1.0, growth=True), "a"),
        (bh.axis.StrCategory([], growth=True), "b"),
        (bh.axis.IntCategory([], growth=True), "c"),
    ],
    ids=["Regular", "StrCategory", "IntCategory"],
)
def test_a_growth_axis_is_refused_naming_histserv_and_the_axis(axis: Any, tag: str) -> None:
    hs = histserv_api()
    # the name holds none of the asserted words, so only the refusal itself can supply them
    ctx = hs.Context(memory_mb=1024, workers=1, name=f"m69b-surface-growth-{tag}")
    with pytest.raises(REFUSED) as made:
        hs.Histogram(axis, context=ctx)
    with pytest.raises(REFUSED) as backed:
        hs.backed(gh.boost.Histogram(axis), ctx)
    for message in (str(made.value), str(backed.value)):
        assert re.search(r"(?i)\bhistserv\b", message) and re.search(
            rf"\b{type(axis).__name__}\b", message
        )


@pytest.mark.parametrize(
    ("storage", "tag"),
    [("Mean", "a"), ("WeightedMean", "b"), ("Unlimited", "c"), ("AtomicInt64", "d")],
    ids=["Mean", "WeightedMean", "Unlimited", "AtomicInt64"],
)
def test_a_storage_histserv_does_not_hold_is_refused_naming_it(storage: str, tag: str) -> None:
    hs = histserv_api()
    ctx = hs.Context(memory_mb=1024, workers=1, name=f"m69b-surface-storage-{tag}")
    make = getattr(bh.storage, storage)
    with pytest.raises(REFUSED) as made:
        hs.Histogram(bh.axis.Regular(4, 0.0, 1.0), storage=make(), context=ctx)
    with pytest.raises(REFUSED) as backed:
        hs.backed(gh.boost.Histogram(bh.axis.Regular(4, 0.0, 1.0), storage=make()), ctx)
    for message in (str(made.value), str(backed.value)):
        assert re.search(r"(?i)\bhistserv\b", message) and re.search(rf"\b{storage}\b", message)


@pytest.mark.parametrize("storage", ["Double", "Int64", "Weight"])
def test_category_and_boolean_axes_are_accepted(storage: str) -> None:
    hs = histserv_api()
    ctx = hs.Context(memory_mb=1024, workers=1, name=f"m69b-surface-accepted-{storage}")
    axes = (bh.axis.StrCategory(["a", "b"]), bh.axis.IntCategory([1, 2]), bh.axis.Boolean())
    made = hs.Histogram(*axes, storage=getattr(bh.storage, storage)(), context=ctx)
    assert isinstance(made, gh.boost.Histogram)


def test_the_single_histogram_plan_refuses_a_backed_histogram(tmp_path: Path) -> None:
    hs = histserv_api()
    ctx = hs.Context(memory_mb=1024, workers=1, name="m69b-surface-single-plan")
    _session, ev = events(write_events(str(tmp_path / "e.parquet")))
    hist = hs.Histogram(bh.axis.Regular(8, -2.0, 6.0), storage=bh.storage.Weight(), context=ctx)
    hist.fill(ev.x, weight=[ev.w])
    with pytest.raises(REFUSED):
        hist.plan(steps_per_file=3)


_CHILD = """
import importlib.util, json, sys
import boost_histogram as bh
import graphed_histogram as gh
from graphed_histogram import histserv
from histserv_harness import events
ctx = histserv.Context(memory_mb=1024, workers=1, name="m69b-surface-import")
_session, ev = events(sys.argv[1])
hist = histserv.Histogram(bh.axis.Regular(8, -2.0, 6.0), storage=bh.storage.Weight(), context=ctx)
hist.fill(ev.x, weight=[ev.w])
plan = gh.plan({"h": hist}, steps_per_file=3)
seen = [name for name in ("histserv", "grpc") if name in sys.modules]
control = None
if importlib.util.find_spec("histserv") is not None:
    import histserv
    control = "grpc" in sys.modules
print(json.dumps({"servers": len(plan.services), "seen": seen, "control": control}))
"""


def test_backing_and_planning_import_neither_histserv_nor_grpc(tmp_path: Path) -> None:
    from histserv_harness import child_env  # noqa: PLC0415

    path = write_events(str(tmp_path / "e.parquet"))
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, path],
        env=child_env(),
        capture_output=True,
        text=True,
        timeout=240,
        check=True,
    )
    lines = done.stdout.splitlines()
    assert len(lines) == 1, done.stdout + done.stderr
    report = json.loads(lines[0])
    assert report["servers"] >= 1
    assert report["seen"] == []
    if HAS_HISTSERV:
        assert report["control"] is True


# a frozen test ID: its length is fixed
def test_an_unbacked_merge_free_plan_equals_the_per_partition_direct_fill_folded_in_partition_order(  # noqa: E501
    tmp_path: Path,
) -> None:
    path = write_events(str(tmp_path / "e.parquet"))
    _session, ev = events(path)
    hist = gh.boost.Histogram(bh.axis.Regular(15, -2.0, 6.0), storage=bh.storage.Weight())
    hist.fill(ev.x, weight=[ev.x * 0.3 + 0.1])
    plan = gh.plan({"h": hist}, steps_per_file=3)
    assert plan.services == ()
    got = SequentialRunner().run(plan).value["h"]

    data = ak.from_parquet(path)
    x = ak.to_numpy(data.x)
    want = bh.Histogram(bh.axis.Regular(15, -2.0, 6.0), storage=bh.storage.Weight())
    for task in sorted(plan.tasks, key=lambda t: t.key):
        part = task.partition.resolve(len(x))
        one = bh.Histogram(bh.axis.Regular(15, -2.0, 6.0), storage=bh.storage.Weight())
        chunk = x[part.entry_start : part.entry_stop]
        one.fill(chunk, weight=chunk * 0.3 + 0.1)
        want = want + one
    assert np.asarray(want.view(flow=True))["value"].sum() > 0
    assert same(got, want)


#: per-directory digests of every frozen file before m69b, the m48/m49 refreeze and m64 fixup
#: included
FROZEN_BEFORE_M69B = {
    "m23": "7c0b3dae357934448aed47f30b9f218c8a32b3a5ad76796431841445b710e943",
    "m29": "031f00db67a9ae844569b7a7f8bc9fb03ab5e017a4fe71db912b5077237f0ab6",
    "m48": "09f30a9a392c77ca3c9ec094adf754ca0a805d593f6531fb7e9835eab4c60f7e",
    "m49": "6e8178246143759acd9406c8b806f46b586d792f82991ca81a1c6d58fa94a4d9",
    "m50": "ee6985354c682524472498da92097bbef739518d4c38ca560c61042fbcc5081d",
    "m52": "0a86a4237cb03c01ae13986f1d6e4f2c4a539fb164e00f3caa21e7b6d989a124",
    "m64": "1f61e8ceead25e3ba9e1878efb073eeb44e6e6774da9835ec834b3b97e8b56aa",
}


def _digest(directory: Path, root: Path) -> str:
    h = hashlib.sha256()
    files = (q for q in directory.rglob("*") if q.is_file() and "__pycache__" not in q.parts)
    # str parts order alike on every OS; WindowsPath ordering ignores case
    for f in sorted(files, key=lambda q: q.relative_to(root).parts):
        data = f.read_bytes()
        if f.suffix in (".py", ".md"):
            data = data.replace(b"\r\n", b"\n")
        h.update(f.relative_to(root).as_posix().encode() + b"\0" + data + b"\0")
    return h.hexdigest()


def test_the_frozen_suites_before_m69b_are_unmodified_but_for_the_refreeze() -> None:
    root = Path(__file__).resolve().parent.parent
    found = {name: _digest(root / name, root) for name in FROZEN_BEFORE_M69B}
    assert found == FROZEN_BEFORE_M69B


GIL_ENABLED = pytest.mark.skipif(FREE_THREADED, reason="grpcio ships no free-threaded wheel")


@GIL_ENABLED
def test_histserv_imports_wherever_the_gil_is_enabled() -> None:
    assert importlib.import_module("histserv").Client is not None
