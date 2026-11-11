"""``gh.plan`` takes whole-dataset reductions and writes beside histograms, each back by name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import graphed
import hist
import hist.graphed as hg
import pytest
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet, gak, parquet_write
from graphed.core import Partition
from graphed.core.execution import Plan, SequentialRunner
from graphed.write import sources_read
from graphed_exec_local import ProcessPoolExecutor, ThreadExecutor

import graphed_histogram as gh

RUNNERS: dict[str, Any] = {
    "sequential": SequentialRunner,
    "thread": lambda: ThreadExecutor(workers=2),
    "process": lambda: ProcessPoolExecutor(workers=2),
}


def _events(path: Path, offset: float = 0.0) -> Any:
    ak.to_parquet(ak.Array({"x": [offset + i for i in range(8)]}), path)
    return from_parquet(
        path.stem, [str(path)], session=Session(AwkwardBackend()), steps_per_file=2
    )


def _hist(x: Any, weight: Any = None) -> Any:
    return hg.Hist.new.Reg(20, 0, 20, name="x").Weight().fill(x, weight=weight)


def _part(partition: Partition) -> str:
    return f"part{partition.blind_step}.parquet"


@pytest.mark.parametrize("runner", RUNNERS)
def test_reductions_come_back_by_name_equal_to_materialize(tmp_path: Path, runner: str) -> None:
    ev = _events(tmp_path / "a.parquet")
    weight = graphed.vary(ev.x * 0.5, "s", up=ev.x * 0.75, down=ev.x * 0.25)
    scalars = {
        "sumw": gak.sum(weight, axis=None),
        "n_big": gak.sum(ev.x >= 5, axis=None),
        "top": gak.max(ev.x, axis=None),
    }
    out = RUNNERS[runner]().run(gh.plan({"h": _hist(ev.x, weight), **scalars})).value
    assert list(out) == ["h", "sumw", "n_big", "top"]
    assert out["sumw"] == graphed.materialize(scalars["sumw"])
    assert out["sumw"] == {"nominal": 14.0, "s_up": 21.0, "s_down": 7.0}
    assert (out["n_big"], out["top"]) == (3, 7.0)
    assert type(out["h"]["nominal"]) is hist.Hist


def test_a_skim_and_histograms_come_back_from_one_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ev = _events(tmp_path / "a.parquet")
    ((_, data),) = sources_read([ev.x]).items()
    reads: list[Partition] = []
    source: Any = type(data)
    original = source.read_partition

    def counted(self: Any, partition: Partition, *args: Any, **kwargs: Any) -> Any:
        reads.append(partition)
        return original(self, partition, *args, **kwargs)

    monkeypatch.setattr(source, "read_partition", counted)
    skim = parquet_write(ev[ev.x >= 5], str(tmp_path / "skim"), name=_part)
    plan = gh.plan({"h": _hist(ev.x), "sumw": gak.sum(ev.x, axis=None)}, writes={"skim": skim})
    assert isinstance(plan, Plan)
    out = SequentialRunner().run(plan).value
    assert len(reads) == len(plan.tasks) == 2
    assert out["sumw"] == 28.0 and out["h"].sum().value == 8
    assert sorted(Path(p).name for p in out["skim"]) == ["part0.parquet", "part1.parquet"]
    assert ak.concatenate([ak.from_parquet(p) for p in sorted(out["skim"])]).x.tolist() == [
        5.0,
        6.0,
        7.0,
    ]


def test_each_dataset_carries_its_own_reductions(tmp_path: Path) -> None:
    dy, tt = _events(tmp_path / "dy.parquet"), _events(tmp_path / "tt.parquet", 10.0)
    out = (
        SequentialRunner()
        .run(
            gh.plan(
                {
                    "DY": {"h": _hist(dy.x), "n": gak.count(dy.x, axis=None)},
                    "TT": {"h": _hist(tt.x), "sumx": gak.sum(tt.x, axis=None)},
                }
            )
        )
        .value
    )
    assert (out["DY"]["n"], out["TT"]["sumx"]) == (8, 108.0)
    assert out["TT"]["h"].sum().value == 8


def test_reductions_alone_make_a_plan(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet")
    plan = gh.plan({"n": gak.count(ev.x, axis=None)})
    assert "n (reduction)" in repr(plan)
    assert SequentialRunner().run(plan).value == {"n": 8}


@pytest.mark.parametrize(
    ("output", "rule"),
    [
        (lambda ev: ev.x * 2, "output 'bad' is per-event"),
        (lambda ev: gak.mean(ev.x, axis=None), "output 'bad' is ak.mean: a mean of chunk means"),
        (lambda ev: gak.sum(ev.x, axis=0), "output 'bad' is ak.sum with axis="),
    ],
)
def test_an_output_no_fold_covers_raises_naming_the_rule(
    tmp_path: Path, output: Any, rule: str
) -> None:
    ev = _events(tmp_path / "a.parquet")
    with pytest.raises(TypeError, match=r"gh\.plan takes histograms and whole-dataset reductions"):
        gh.plan({"h": _hist(ev.x), "bad": output(ev)})
    with pytest.raises(TypeError, match=rule):
        gh.plan({"h": _hist(ev.x), "bad": output(ev)})


def test_a_varied_per_event_output_raises(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet")
    with pytest.raises(TypeError, match="output 'w' is per-event"):
        gh.plan({"w": graphed.vary(ev.x, "s", up=ev.x * 2)})


def test_write_names_must_be_their_own(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet")
    skim = parquet_write(ev.x, str(tmp_path / "skim"), name=_part)
    with pytest.raises(ValueError, match="'h' names both an output and a write"):
        gh.plan({"h": _hist(ev.x)}, writes={"h": skim})
    with pytest.raises(TypeError, match="pass the outputs as a mapping"):
        gh.plan(_hist(ev.x), writes={"skim": skim})


def test_writes_need_an_output_and_a_part_write(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet")
    skim = parquet_write(ev.x, str(tmp_path / "skim"), name=_part)
    with pytest.raises(ValueError, match="at least one histogram or reduction"):
        gh.plan({}, writes={"skim": skim})
    with pytest.raises(TypeError, match="takes PartWrite values; 's' is str"):
        not_a_write: Any = "out/skim"
        gh.plan({"h": _hist(ev.x)}, writes={"s": not_a_write})


def test_a_varied_reductions_universes_carry_their_labels(tmp_path: Path) -> None:
    ev = _events(tmp_path / "a.parquet")
    shifted = ev.x * 0.75
    plan = gh.plan({"sumw": gak.sum(graphed.vary(ev.x, "s", up=shifted), axis=None)})
    assert isinstance(plan, Plan)
    process: Any = plan.process
    labelled = {label for _key, (labels, _frame) in process.variation_labels for label in labels}
    assert labelled == {"s_up"}
