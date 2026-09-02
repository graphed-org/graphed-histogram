"""m48/H12 — §6.3: data / no-variation paths are unchanged, gated byte-for-byte.

Two in-tree golden patterns over the SAME unvaried single-weight fill: a committed GIR blob, and
the fill node's params KEY SET against a literally spelled expected set. m48 in sibling mode adds
no params key by design (§1.2 keeps labels out of params and hashes), so a key-absence placeholder
would have nothing to spell — the whole key set is the assertion.

The blob is the PRE-m48 one: captured against `graphed @ ff7c607` (the revision before m48's
frontend landed) and committed ALREADY STRIPPED of the fill node's `PayloadDescriptor.version`.
Captured after the implementation it would be a no-op tautology. Regenerate with:

    git -C <graphed> worktree add /tmp/graphed-pre-m48 ff7c607
    cp <graphed>/python/graphed/core/graphed_core*.so /tmp/graphed-pre-m48/python/graphed/core/
    PYTHONPATH=/tmp/graphed-pre-m48/python:src:tests/_corpus:tests/frozen/m48 python -c \
      "import boost_histogram as bh, test_variation_goldens as t; \
       t.GOLDEN.write_bytes(t.strip_version(t.live_blob(), bh.__version__))"

`Histogram.fill` hard-codes `version=bh.__version__` into the serialized descriptor with no
author-facing knob, and `boost-histogram>=1.4` is unpinned — an unstripped literal would red on
the next release, in a frozen file that cannot be repaired in place. The strip pattern is
content-derived and therefore PER SIDE: one live-derived pattern applied to both sides misses the
literal's capture-time version entirely, and the two agree at capture, so the miss stays silent
until the first bh bump. The closing monkeypatch leg is what discriminates that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import pytest
from graphed import Session, compile_ir
from graphed.awkward import AwkwardBackend, from_awkward

import graphed_histogram as gh

GOLDEN = Path(__file__).resolve().parent / "goldens" / "unvaried_single_weight_fill.gir"

#: fixed so the capture is reproducible: the blob carries the source's FORM, not its values
EVENTS = ak.Array({"x": [1.0, 4.0, 7.0, 2.5, 6.0] * 8, "w": [0.5, 1.0, 2.0, 1.5, 0.2] * 8})

PARAMS = {"spec", "n_axes", "weighted", "sampled"}


def _fill() -> tuple[Session, gh.boost.Histogram]:
    """The §6.3 case: NO context handle and NO `Varied` input, so §6.1d's broadcast seam is not
    triggered and this graph must record exactly as it did before m48."""
    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", EVENTS)
    h = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight())
    h.fill(events.x, weight=events.w)
    return session, h


def live_blob() -> bytes:
    session, h = _fill()
    return bytes(compile_ir(session, h.fill_nodes()[0]).ir)


def strip_version(blob: bytes, version: str) -> bytes:
    """Remove ONE side's own length-prefixed `bh.__version__` from a GIR blob."""
    return blob.replace(len(version).to_bytes(4, "little") + version.encode(), b"")


def test_the_unvaried_fill_graph_still_serializes_to_the_pre_m48_golden() -> None:
    assert GOLDEN.read_bytes() == strip_version(live_blob(), bh.__version__)


def test_the_golden_carries_no_version_bytes_of_its_own() -> None:
    """The instrument for the assertion above: a golden committed UNSTRIPPED would agree with a
    live blob stripped by a pattern that never matched, and both legs would pass vacuously."""
    blob = live_blob()
    pattern = len(bh.__version__).to_bytes(4, "little") + bh.__version__.encode()
    assert blob.count(pattern) == 1
    assert pattern not in GOLDEN.read_bytes()


def test_the_unvaried_single_weight_fill_records_exactly_these_params() -> None:
    session, h = _fill()
    node = next(n for n in session._store.nodes() if n["id"] == h.fill_nodes()[0].node_id)
    assert set(node["params"]) == PARAMS


def test_the_comparison_survives_a_boost_histogram_version_bump(monkeypatch: Any) -> None:
    """§6.3's closing leg. Each side is stripped of the version IT carries, so a bh release moves
    the live blob and the golden still matches; an implementation that strips both sides with one
    live-derived pattern reds here the moment the two versions differ."""
    monkeypatch.setattr(bh, "__version__", "0.0.0.dev-m48")
    assert GOLDEN.read_bytes() == strip_version(live_blob(), bh.__version__)


def test_the_stripping_is_what_makes_the_two_sides_agree() -> None:
    """Without it the live blob differs from the golden, so the assertions above are not passing
    on a pair that was equal anyway."""
    assert live_blob() != GOLDEN.read_bytes()


@pytest.mark.parametrize("version", ["1.8.0", "0.0.0.dev-m48"])
def test_stripping_is_a_no_op_on_a_blob_that_does_not_carry_that_version(version: str) -> None:
    golden = GOLDEN.read_bytes()
    assert strip_version(golden, version) == golden
