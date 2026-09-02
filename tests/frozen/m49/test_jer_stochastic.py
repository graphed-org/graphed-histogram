"""m49/F4 — §5.5: the JER-SF stochastic shift, and the four witnesses a wrong RNG passes.

The smear is coffea's stochastic shape `1 + sqrt(max(SF**2 - 1, 0)) * g`, with `g` a per-row
content-seeded standard normal drawn ONCE and shared by every universe. Both varied labels carry
SF above 1 at different magnitudes: at SF <= 1 the `max(..., 0)` floor smears by exactly 1, which
equals nominal, passes the invariance leg vacuously and reds the migration witness.

PARTITION INVARIANCE is the witness that discriminates a per-partition seed — coffea's own
`rand_gauss` seeds PCG64 from the first and last elements of the array SLICE it is handed, which
passes every other witness here. The compared quantities are the per-label SMEARED VALUES and
selection MASKS, concatenated from a PLAN RUN in task order: a weighted float histogram is not
byte-invariant under re-partitioning (the combine tree regroups its additions), and
`Session.materialize` is partition-blind and cannot observe `steps_per_file` at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import awkward as ak
import boost_histogram as bh
import graphed
import numpy as np
from graphed import aggregate_plan
from graphed.awkward import gak
from graphed.core.execution import SequentialRunner
from m49_hist_fixtures import JER_SF, JER_THRESHOLD, jer_program

import graphed_histogram as gh


def _chunk(values: list[object]) -> list[np.ndarray]:
    """One partition's outputs as plain numpy, in the plan's output order."""
    return [ak.to_numpy(ak.flatten(v, axis=None)) for v in values]


def _concat(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    return [np.concatenate([x, y]) for x, y in zip(a, b, strict=True)]


@dataclass(frozen=True)
class _Empty:
    width: int

    def __call__(self) -> list[np.ndarray]:
        return [np.empty(0) for _ in range(self.width)]


def _run(steps: int) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """Per-label smeared values and masks, from a PLAN RUN at `steps` partitions."""
    _session, _smeared, per_label, _draw, source = jer_program()
    labels = sorted(per_label)
    outputs = [per_label[label][kind] for label in labels for kind in ("smeared", "mask")]
    plan = aggregate_plan(
        *outputs,
        reduce=_chunk,
        combine=_concat,
        empty=_Empty(len(outputs)),
        steps_per_file=steps,
    )
    flat = SequentialRunner().run(plan).value
    got = {
        label: {"smeared": flat[2 * i], "mask": flat[2 * i + 1].astype(bool)}
        for i, label in enumerate(labels)
    }
    return got, len(source.part_reads)


def test_the_universes_select_pairwise_distinct_event_counts() -> None:
    """No ordering is asserted (§5.1): a re-smearing shift migrates in both directions, so the
    counts have no expected sign."""
    got, _reads = _run(1)
    counts = {label: int(values["mask"].sum()) for label, values in got.items()}
    assert set(counts) == set(JER_SF)
    assert len(set(counts.values())) == len(counts), counts
    assert all(0 < n < len(got[label]["mask"]) for label, n in counts.items())


def test_no_universes_selection_is_a_subset_of_anothers() -> None:
    """Bidirectional migration — the non-monotone discriminator. A one-sided scale (and any
    SF <= 1 label, which the floor pins to nominal) makes one mask contain another."""
    got, _reads = _run(1)
    for left, right in permutations(got, 2):
        implied = np.all(~got[left]["mask"] | got[right]["mask"])
        assert not implied, f"{left}'s selection is contained in {right}'s"


def test_the_nominal_universe_is_unsmeared() -> None:
    """The instrument for the migration witness: nominal must be the raw quantity, or every
    universe is smeared and the comparisons above are between three smeared sets."""
    got, _reads = _run(1)
    assert not np.array_equal(got["nominal"]["smeared"], got["jer_up"]["smeared"])
    assert not np.array_equal(got["jer_up"]["smeared"], got["jer_down"]["smeared"])


def test_the_shared_draw_node_is_interned_exactly_once() -> None:
    """§5.5b: one draw, all universes. Each member re-records the draw expression from scratch, so
    a store carrying two structurally identical draw nodes means interning did not engage — and
    two independent draws would break the shared-prefix sharing the whole shift path rests on."""
    session, _smeared, _per_label, draw, _source = jer_program()
    nodes = session._store.nodes()
    (recorded,) = (node for node in nodes if node["id"] == draw.node_id)
    twins = [
        node for node in nodes if node["name"] == recorded["name"] and node["inputs"] == recorded["inputs"]
    ]
    assert len(twins) == 1


def test_the_smeared_values_and_masks_are_byte_identical_across_partitionings() -> None:
    """§5.5a's headline: the draw for a row is a pure function of THAT ROW's own content, so the
    same row draws the same value under any partitioning."""
    one, reads_one = _run(1)
    five, reads_five = _run(5)
    assert (reads_one, reads_five) == (1, 5), "the two runs did not actually re-partition"
    for label in sorted(one):
        for kind in ("smeared", "mask"):
            assert np.array_equal(one[label][kind], five[label][kind]), f"{label}/{kind} moved"
        assert one[label]["smeared"].size > 0


def test_the_same_program_is_byte_identical_run_to_run() -> None:
    first, _ = _run(3)
    second, _ = _run(3)
    for label in sorted(first):
        for kind in ("smeared", "mask"):
            assert np.array_equal(first[label][kind], second[label][kind])


def test_the_selection_counts_survive_the_histogram_sink() -> None:
    """The same quantity through this repo's own fill path, on an INTEGER storage — the only
    histogram §5.5a admits as a compared quantity."""
    _session, smeared, _per_label, _draw, _source = jer_program()
    selected = smeared[smeared > JER_THRESHOLD]
    hist = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 400.0), storage=bh.storage.Int64())
    hist.fill(gak.flatten(selected))
    assert sorted(graphed.labels(smeared)) == sorted(JER_SF)

    result = gh.unpack(SequentialRunner().run(gh.plan({"jer": hist}, steps_per_file=4)).value)
    expected, _reads = _run(4)
    for label in sorted(JER_SF):
        in_range = expected[label]["smeared"][expected[label]["mask"]]
        assert result["jer"][label].sum() == int(((in_range >= 0.0) & (in_range < 400.0)).sum())
