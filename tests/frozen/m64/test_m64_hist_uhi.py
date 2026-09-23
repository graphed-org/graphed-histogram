"""G15: hist's named growth axes survive a parquet-read plan and the UHI JSON round trip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import awkward as ak
import hist
import hist.graphed
import numpy as np
import pytest
import uhi.io.json
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet
from m64_growth_fixtures import ALL, RUNNERS, assert_bitwise

import graphed_histogram as gh

CHANNELS = ["mu", "e", "mu", "e", "tau", "e", "mu", "tau", "jet", "e", "tau", "jet", "e", "mu", "jet", "gam"]
#: quarter-bin centres, growing below and above [0, 1)
X = [
    0.125,
    -0.375,
    0.625,
    1.125,
    0.375,
    -0.875,
    1.625,
    0.125,
    2.125,
    0.875,
    -0.125,
    0.375,
    0.625,
    2.625,
    -1.375,
    0.875,
]


def _eager() -> hist.Hist:
    h = hist.Hist.new.StrCat([], growth=True, name="ch").Reg(4, 0, 1, growth=True, name="x").Int64()
    h.fill(ch=np.asarray(CHANNELS), x=np.asarray(X))
    return h


@pytest.mark.parametrize(("runner", "how"), ALL)
def test_g15_hist_names_growth_and_uhi_round_trip(tmp_path: Path, runner: str, how: str) -> None:
    path = tmp_path / "ev.parquet"
    ak.to_parquet(ak.Array({"ch": CHANNELS, "x": X}), path)
    session = Session(AwkwardBackend())
    ev: Any = from_parquet(session, "ev", str(path), steps_per_file=4)
    h = hist.graphed.Hist.new.StrCat([], growth=True, name="ch").Reg(4, 0, 1, growth=True, name="x").Int64()
    h.fill(ch=ev.ch, x=ev.x)
    plan = h.plan(steps_per_file=4) if how == "h.plan" else gh.plan({"h": h}, steps_per_file=4)
    value = RUNNERS[runner]().run(plan).value
    got = hist.Hist(value if how == "h.plan" else value["h"])

    want = _eager()
    assert got.axes.name == ("ch", "x")
    assert all(axis.traits.growth for axis in got.axes)
    assert list(got.axes[0]) == list(dict.fromkeys(CHANNELS))
    assert_bitwise(got, want)
    loaded = hist.Hist(
        json.loads(json.dumps(got, default=uhi.io.json.default), object_hook=uhi.io.json.object_hook)
    )
    assert loaded == got
    assert all(axis.traits.growth for axis in loaded.axes)
