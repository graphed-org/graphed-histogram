"""M23 (graphed-histogram): deferred boost-histogram filling on graphed task graphs (P0.1).

`.fill(...)` RECORDS — an External node carrying the content-addressed canonical axes/storage
spec (the M3 correctionlib/ONNX family; backends know nothing about histograms). Evaluation is
graphed's own machinery [freeze-M23-1, user-directed: there is NO compute() helper]: `plan()`
exports the R15.4 task graph and an R7 executor's `run(plan).value` IS the aggregated histogram
(partition-wise through the compiled IR; the whole-dataset loader NEVER invoked —
counter-witnessed); the reference `session.materialize(fill_node)` evaluates a fill eagerly.
Counts are pinned BIT FOR BIT against eager boost-histogram; multiple fills accumulate; same
axes spec => one interned node.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import boost_histogram as bh
import numpy as np
import pytest
from graphed import Array, Session
from graphed.write import SequentialRunner
from graphed_core import Partition
from graphed_numpy import NumpyBackend
from graphed_numpy.forms import NumpyForm

import graphed_histogram as gh

RNG = np.random.default_rng(42)
DATA = RNG.normal(5.0, 2.0, 1000)
WEIGHTS = RNG.uniform(0.5, 1.5, 1000)


@dataclass
class ChunkedSource:
    """A PartitionedSource over an in-memory array, with the efficiency-witness counters."""

    data: np.ndarray
    whole_calls: list = field(default_factory=list)
    part_reads: list = field(default_factory=list)

    def __call__(self) -> np.ndarray:  # the whole-dataset loader: must NEVER run during a plan
        self.whole_calls.append(1)
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://chunks", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition, columns, resources) -> np.ndarray:  # type: ignore[no-untyped-def]
        part = partition.resolve(len(self.data))
        self.part_reads.append((part.entry_start, part.entry_stop))
        return self.data[part.entry_start : part.entry_stop]


def _source(session: Session, name: str, data: np.ndarray) -> tuple[Array, ChunkedSource]:
    src = ChunkedSource(data)
    arr = session.source(name, form=NumpyForm(data.dtype, shape=(None,)), data=src)
    return arr, src


def test_fill_records_instead_of_executing() -> None:
    s = Session(NumpyBackend())
    x, src = _source(s, "x", DATA)
    h = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 10.0))
    out = h.fill(x)
    assert out is h  # fill stages and returns self
    assert src.whole_calls == [] and src.part_reads == []  # nothing read, nothing computed
    assert h.staged_fills() == 1


def test_executor_aggregation_matches_eager_boost_bit_for_bit() -> None:
    s = Session(NumpyBackend())
    x, src = _source(s, "x", DATA)
    h = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 10.0), storage=bh.storage.Int64()).fill(x)
    got = SequentialRunner().run(h.plan(steps_per_file=4)).value  # an R7 executor aggregates
    eager = bh.Histogram(bh.axis.Regular(20, 0.0, 10.0), storage=bh.storage.Int64())
    eager.fill(DATA)
    assert np.array_equal(got.values(), eager.values())
    assert got.sum(flow=True) == eager.sum(flow=True)
    assert src.whole_calls == []  # partition-wise: the whole-dataset loader never ran
    assert len(src.part_reads) == 4  # one read per partition


def test_multiple_fills_accumulate() -> None:
    s = Session(NumpyBackend())
    x, _src = _source(s, "x", DATA)
    h = gh.boost.Histogram(bh.axis.Regular(10, 0.0, 10.0), storage=bh.storage.Int64())
    h.fill(x).fill(x * 0.5 + 1.0)
    got = SequentialRunner().run(h.plan(steps_per_file=3)).value
    eager = bh.Histogram(bh.axis.Regular(10, 0.0, 10.0), storage=bh.storage.Int64())
    eager.fill(DATA)
    eager.fill(DATA * 0.5 + 1.0)
    assert np.array_equal(got.values(), eager.values())


def test_weighted_fill_with_weight_storage() -> None:
    s = Session(NumpyBackend())
    x, _ = _source(s, "x", DATA)
    w, _ = _source(s, "w", WEIGHTS)
    with pytest.raises(TypeError, match="one partitioned source"):
        gh.boost.Histogram(bh.axis.Regular(10, 0.0, 10.0), storage=bh.storage.Weight()).fill(
            x, weight=w
        ).plan()
    # weight from the SAME source's record is fine
    s2 = Session(NumpyBackend())
    x2, _ = _source(s2, "x", DATA)
    h = gh.boost.Histogram(bh.axis.Regular(10, 0.0, 10.0), storage=bh.storage.Weight())
    h.fill(x2, weight=np.sqrt(abs(x2)))
    got = SequentialRunner().run(h.plan(steps_per_file=2)).value
    eager = bh.Histogram(bh.axis.Regular(10, 0.0, 10.0), storage=bh.storage.Weight())
    eager.fill(DATA, weight=np.sqrt(abs(DATA)))
    assert np.allclose(got.values(), eager.values())
    assert np.allclose(got.variances(), eager.variances())


def test_multidimensional_and_categorical_axes() -> None:
    s = Session(NumpyBackend())
    x, _ = _source(s, "x", DATA)
    h2 = gh.boost.Histogram(
        bh.axis.Regular(10, 0.0, 10.0),
        bh.axis.Variable([0.0, 2.0, 5.0, 10.0]),
        storage=bh.storage.Int64(),
    ).fill(x, x * 0.9)
    eager2 = bh.Histogram(
        bh.axis.Regular(10, 0.0, 10.0),
        bh.axis.Variable([0.0, 2.0, 5.0, 10.0]),
        storage=bh.storage.Int64(),
    )
    eager2.fill(DATA, DATA * 0.9)
    assert np.array_equal(SequentialRunner().run(h2.plan(steps_per_file=2)).value.values(), eager2.values())

    s3 = Session(NumpyBackend())
    cats = (DATA > 5.0).astype("int64") + 2 * (DATA > 7.0).astype("int64")
    c3, _ = _source(s3, "c", cats)
    h3 = gh.boost.Histogram(bh.axis.IntCategory([0, 1, 2]), storage=bh.storage.Int64()).fill(c3)
    eager3 = bh.Histogram(bh.axis.IntCategory([0, 1, 2]), storage=bh.storage.Int64())
    eager3.fill(cats)
    assert np.array_equal(SequentialRunner().run(h3.plan()).value.values(), eager3.values())


def test_content_addressed_identity_is_hash_consed() -> None:
    s = Session(NumpyBackend())
    x, _ = _source(s, "x", DATA)
    h1 = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 10.0)).fill(x)
    h2 = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 10.0)).fill(x)
    h3 = gh.boost.Histogram(bh.axis.Regular(21, 0.0, 10.0)).fill(x)
    assert h1.fill_nodes()[0].node_id == h2.fill_nodes()[0].node_id  # same spec + input -> interned
    assert h1.fill_nodes()[0].node_id != h3.fill_nodes()[0].node_id  # different bins -> different
    assert gh.spec_of(h1) == gh.spec_of(h2) and gh.spec_of(h1) != gh.spec_of(h3)


def test_plan_aggregates_identically_across_executors() -> None:
    pytest.importorskip("graphed_exec_local")
    from graphed_exec_local import ProcessExecutor  # noqa: PLC0415  (importorskip-gated)

    s = Session(NumpyBackend())
    x, _ = _source(s, "x", DATA)
    h = gh.boost.Histogram(bh.axis.Regular(16, 0.0, 10.0), storage=bh.storage.Int64()).fill(abs(x))
    sequential = SequentialRunner().run(h.plan(steps_per_file=3)).value
    parallel = ProcessExecutor(max_workers=2).run(h.plan(steps_per_file=3)).value
    assert np.array_equal(np.asarray(parallel.values()), np.asarray(sequential.values()))  # R15.4
    assert parallel.sum(flow=True) == sequential.sum(flow=True)


def test_in_memory_sources_evaluate_via_the_reference_materialize() -> None:
    s = Session(NumpyBackend())
    arr = s.source("m", form=NumpyForm(DATA.dtype, shape=(None,)), data=DATA)  # plain array data
    h = gh.boost.Histogram(bh.axis.Regular(12, 0.0, 10.0), storage=bh.storage.Int64()).fill(arr)
    h.fill(arr * 0.5)  # multi-fill: zero + sum of per-fill materializes (the monoid helpers)
    total = gh.zero_of(gh.spec_of(h))
    for node in h.fill_nodes():
        total = gh.add_histograms(total, s.materialize(node))
    eager = bh.Histogram(bh.axis.Regular(12, 0.0, 10.0), storage=bh.storage.Int64())
    eager.fill(DATA)
    eager.fill(DATA * 0.5)
    assert np.array_equal(total.values(), eager.values())


def test_numpy_like_entry_points() -> None:
    s = Session(NumpyBackend())
    x, _ = _source(s, "x", DATA)
    h = gh.histogram(x, bins=25, range=(0.0, 10.0))
    counts = np.asarray(SequentialRunner().run(h.plan(steps_per_file=2)).value.values())
    ref, _edges = np.histogram(DATA, bins=25, range=(0.0, 10.0))
    assert np.array_equal(counts.astype("int64"), ref)

    s2 = Session(NumpyBackend())
    x2, _ = _source(s2, "x", DATA)
    h2 = gh.histogram2d(x2, x2 * 0.5, bins=(10, 8), range=((0.0, 10.0), (0.0, 5.0)))
    ref2, _, _ = np.histogram2d(DATA, DATA * 0.5, bins=(10, 8), range=((0.0, 10.0), (0.0, 5.0)))
    assert np.array_equal(
        np.asarray(SequentialRunner().run(h2.plan()).value.values()).astype("int64"), ref2.astype("int64")
    )


def test_ir_is_deterministic_and_carries_the_descriptor() -> None:
    import graphed_core  # noqa: PLC0415
    from graphed import compile_ir  # noqa: PLC0415

    def build() -> tuple[bytes, dict]:
        s = Session(NumpyBackend())
        x, _ = _source(s, "x", DATA)
        h = gh.boost.Histogram(bh.axis.Regular(20, 0.0, 10.0)).fill(x)
        compiled = compile_ir(s, *h.fill_nodes())
        nodes = graphed_core.GraphStore.deserialize(compiled.ir).nodes()
        ext = next(n for n in nodes if n["kind"] == "external")
        return bytes(compiled.ir), ext["descriptor"]

    ir1, desc1 = build()
    ir2, desc2 = build()
    assert ir1 == ir2  # byte-identical across runs (the determinism gate)
    assert desc1 == desc2
    assert desc1["kind"] == "histogram" and desc1["io_schema"] == "uhi"
    assert desc1["content_hash"].startswith("sha256:")


def test_factory_and_histogramdd_match_eager() -> None:
    s = Session(NumpyBackend())
    x, _ = _source(s, "x", DATA)
    href = bh.Histogram(bh.axis.Regular(14, 0.0, 10.0), storage=bh.storage.Int64())
    h = gh.factory(x, histref=href)
    eager = bh.Histogram(bh.axis.Regular(14, 0.0, 10.0), storage=bh.storage.Int64())
    eager.fill(DATA)
    assert np.array_equal(SequentialRunner().run(h.plan(steps_per_file=2)).value.values(), eager.values())

    s2 = Session(NumpyBackend())
    x2, _ = _source(s2, "x", DATA)
    hdd = gh.histogramdd([x2, x2 * 0.5], bins=[6, 5], range=((0.0, 10.0), (0.0, 5.0)))
    refdd, _ = np.histogramdd([DATA, DATA * 0.5], bins=[6, 5], range=((0.0, 10.0), (0.0, 5.0)))
    assert np.array_equal(
        np.asarray(SequentialRunner().run(hdd.plan()).value.values()).astype("int64"), refdd.astype("int64")
    )


def test_spec_round_trips_every_supported_axis_and_storage() -> None:
    histos = [
        bh.Histogram(bh.axis.Integer(0, 7), storage=bh.storage.Double()),
        bh.Histogram(bh.axis.Boolean(), storage=bh.storage.Int64()),
        bh.Histogram(bh.axis.StrCategory(["a", "b"]), storage=bh.storage.Weight()),
        bh.Histogram(bh.axis.Regular(4, 0.0, 1.0, metadata={"name": "pt"}), storage=bh.storage.Mean()),
        bh.Histogram(bh.axis.Variable([0.0, 0.5, 1.0]), storage=bh.storage.WeightedMean()),
    ]
    for h in histos:
        spec = gh.spec_of(h)
        z = gh.zero_of(spec)
        assert gh.spec_of(z) == spec  # encode -> rebuild -> encode is a fixed point
        assert gh.content_hash(spec).startswith("sha256:")


def test_guardrails_fail_loudly() -> None:
    s = Session(NumpyBackend())
    x, _ = _source(s, "x", DATA)
    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 1.0))
    with pytest.raises(TypeError, match="2 axes" if False else "axes"):
        gh.boost.Histogram(bh.axis.Regular(4, 0.0, 1.0), bh.axis.Regular(4, 0.0, 1.0)).fill(x)
    with pytest.raises(TypeError, match="graphed Arrays"):
        h.fill(DATA)  # eager data is not a deferred fill
    with pytest.raises(ValueError, match="nothing staged"):
        gh.boost.Histogram(bh.axis.Regular(4, 0.0, 1.0)).plan()
    with pytest.raises(TypeError, match="growth"):
        gh.spec_of(bh.Histogram(bh.axis.IntCategory([], growth=True)))
    s2 = Session(NumpyBackend())
    arr = s2.source("m", form=NumpyForm(DATA.dtype, shape=(None,)), data=DATA)
    with pytest.raises(TypeError, match="partitioned source"):
        gh.boost.Histogram(bh.axis.Regular(4, 0.0, 10.0)).fill(arr).plan()
    with pytest.raises(TypeError, match="explicit bins and range"):
        gh.histogram(x, bins=10)  # no range


def test_evaluators_registry_merges_histograms() -> None:
    s = Session(NumpyBackend())
    x, _ = _source(s, "x", DATA)
    h1 = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 10.0)).fill(x)
    h2 = gh.boost.Histogram(bh.axis.Regular(8, 0.0, 10.0)).fill(x)
    reg = gh.evaluators(h1, h2)
    assert len(reg) == 2 and all(k.startswith("sha256:") for k in reg)


def test_plan_accepts_explicit_partitions() -> None:
    # the entry-target seam (ADL P2): a caller may shape partitioning itself — absolute
    # entry-count chunks for a benchmark sweep — instead of steps_per_file's per-file split
    s = Session(NumpyBackend())
    x, src = _source(s, "x", DATA)
    h = gh.boost.Histogram(bh.axis.Regular(10, 0.0, 10.0), storage=bh.storage.Int64()).fill(x)
    explicit = tuple(
        Partition("toy://chunks", "", lo, min(lo + 300, len(DATA))) for lo in range(0, len(DATA), 300)
    )
    plan = h.plan(partitions=explicit)
    assert len(plan.tasks) == len(explicit)
    got = SequentialRunner().run(plan).value
    eager = bh.Histogram(bh.axis.Regular(10, 0.0, 10.0), storage=bh.storage.Int64())
    eager.fill(DATA)
    assert np.array_equal(got.values(flow=True), eager.values(flow=True))
    assert src.whole_calls == []
