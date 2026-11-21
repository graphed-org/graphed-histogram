"""Histograms whose fills the optimizer merges each get their own fill (graphed#228)."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import hist
import hist.graphed as hg
import numpy as np
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet
from graphed.core.execution import SequentialRunner

import graphed_histogram as gh

X, Y = [1.5, 2.5, 2.5, 7.5], [4.5, 5.5, 6.5, 6.5]


def _filled(values: list[float]) -> hist.Hist:
    return hist.Hist.new.Reg(10, 0, 10, name="v").Double().fill(values)


def test_a_merge_ahead_of_another_fill_reads_each_fill(tmp_path: Path) -> None:
    ak.to_parquet(ak.Array({"x": X, "y": Y}), tmp_path / "e.parquet")
    ev = from_parquet("e", [str(tmp_path / "e.parquet")], session=Session(AwkwardBackend()))
    hists = [hg.Hist.new.Reg(10, 0, 10, name="v").Double() for _ in range(3)]
    for h, col in zip(hists, (ev.x, ev.x, ev.y), strict=True):
        h.fill(col)
    value = gh.unpack(SequentialRunner().run(gh.plan(hists)).value)
    for got, want in zip(value.values(), (X, X, Y), strict=True):
        np.testing.assert_array_equal(got.values(), _filled(want).values())
