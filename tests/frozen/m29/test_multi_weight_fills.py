"""M29 — multiple multiplicative weights are a first-class fill signature.

HEP event weights arrive as several factors (generator x pileup x trigger SFs...). ``fill``
now accepts ``weight=`` as a SEQUENCE of graphed Arrays: every weight is recorded as a graph
input, the node's params carry ``n_weights``, and evaluation — the histogram package's own
``FillEvaluator`` (the ``plan()``/executor path) — multiplies them elementwise into the single
fill weight. Replay through preservation follows the same params (pinned in graphed-preserve's
cross-seam suite). A single weight records EXACTLY as before — no ``n_weights`` param, so
pre-M29 node identities, specs, and bundles are untouched.
"""

from __future__ import annotations

import awkward as ak
import boost_histogram as bh
import numpy as np
from graphed import Session
from graphed.write import SequentialRunner
from graphed_awkward import AwkwardBackend, from_awkward

import graphed_histogram as gh

EVENTS = ak.Array(
    {
        "x": [1.0, 4.0, 7.0, 2.5, 6.0] * 24,
        "w1": [0.5, 1.0, 2.0, 1.5, 0.2] * 24,
        "w2": [1.0, 0.5, 1.0, 2.0, 3.0] * 24,
    }
)


def _recorded(weight_spec):  # type: ignore[no-untyped-def]
    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", EVENTS)
    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight())
    weights = [getattr(g, n) for n in weight_spec]
    h.fill(g.x, weight=weights if len(weights) > 1 else weights[0])
    return s, h


def _eager(weight_names):  # type: ignore[no-untyped-def]
    h = bh.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight())
    w = np.ones(len(EVENTS))
    for n in weight_names:
        w = w * np.asarray(EVENTS[n])
    h.fill(np.asarray(EVENTS.x), weight=w)
    return h


def test_two_weights_multiply_via_materialize() -> None:
    s, h = _recorded(["w1", "w2"])
    got = s.materialize(h.fill_nodes()[0])
    want = _eager(["w1", "w2"])
    assert np.array_equal(got.view(flow=True)["value"], want.view(flow=True)["value"])
    assert np.array_equal(got.view(flow=True)["variance"], want.view(flow=True)["variance"])


def test_two_weights_multiply_through_the_plan_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # the executor path needs a PARTITIONED source: fill from a parquet dataset
    import graphed_awkward as gha  # noqa: PLC0415

    path = tmp_path / "events.parquet"
    ak.to_parquet(EVENTS, path)
    s = Session(AwkwardBackend())
    g = gha.io.from_parquet(s, "events", str(path))
    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight())
    h.fill(g.x, weight=[g.w1, g.w2])
    result = SequentialRunner().run(h.plan(steps_per_file=3)).value
    again = SequentialRunner().run(h.plan(steps_per_file=3)).value
    want = _eager(["w1", "w2"])
    # per-partition fills sum in tree order: deterministic (byte-identical across runs), and
    # equal to the single-pass eager fill up to float-summation order
    assert np.array_equal(result.view(flow=True)["value"], again.view(flow=True)["value"])
    assert np.allclose(result.view(flow=True)["value"], want.view(flow=True)["value"], rtol=1e-12)
    assert np.allclose(result.view(flow=True)["variance"], want.view(flow=True)["variance"], rtol=1e-12)


def test_three_weights_and_the_params_contract() -> None:
    s, h = _recorded(["w1", "w2", "w1"])
    node = next(n for n in s._store.nodes() if n["id"] == h.fill_nodes()[0].node_id)
    assert int(node["params"]["n_weights"]) == 3
    assert bool(node["params"]["weighted"]) is True
    assert len(node["inputs"]) == 4  # one axis + three weights, ALL graph inputs
    got = s.materialize(h.fill_nodes()[0])
    want = _eager(["w1", "w2", "w1"])
    assert np.array_equal(got.view(flow=True)["value"], want.view(flow=True)["value"])


def test_a_single_weight_records_exactly_as_before() -> None:
    s, h = _recorded(["w1"])
    node = next(n for n in s._store.nodes() if n["id"] == h.fill_nodes()[0].node_id)
    assert "n_weights" not in node["params"]  # pre-M29 node identity, byte-for-byte
    got = s.materialize(h.fill_nodes()[0])
    want = _eager(["w1"])
    assert np.array_equal(got.view(flow=True)["value"], want.view(flow=True)["value"])


def test_jagged_weights_flatten_consistently() -> None:
    events = ak.Array(
        {
            "pt": [[30.0, 50.0], [], [80.0]] * 20,
            "sf1": [[1.1, 0.9], [], [1.2]] * 20,
            "sf2": [[2.0, 1.0], [], [0.5]] * 20,
        }
    )
    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", events)
    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 100.0), storage=bh.storage.Weight())
    h.fill(g.pt, weight=[g.sf1, g.sf2])
    got = s.materialize(h.fill_nodes()[0])
    want = bh.Histogram(bh.axis.Regular(4, 0.0, 100.0), storage=bh.storage.Weight())
    want.fill(
        np.asarray(ak.flatten(events.pt, axis=None)),
        weight=np.asarray(ak.flatten(events.sf1 * events.sf2, axis=None)),
    )
    assert np.array_equal(got.view(flow=True)["value"], want.view(flow=True)["value"])
    assert np.array_equal(got.view(flow=True)["variance"], want.view(flow=True)["variance"])
