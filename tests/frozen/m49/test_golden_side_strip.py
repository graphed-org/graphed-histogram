"""m49/F9 — §6.3: the golden comparison strips each side of the version IT carries.

`Histogram.fill` hard-codes `version=bh.__version__` into the serialized `PayloadDescriptor` with
no author-facing knob, and `boost-histogram>=1.4` is unpinned, so a golden blob and a live blob
captured at different releases carry DIFFERENT version bytes. Stripping both sides with one
live-derived pattern therefore misses the golden's own version — and the two agree at capture
time, so the miss stays silent until the first bump, in a frozen file that cannot be repaired in
place.

m48's committed golden carries no version bytes at all, which makes one-pattern and per-side
stripping indistinguishable on it. This anchor builds the pair that separates them: the same
unvaried fill graph serialized under two different `bh.__version__` values.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import boost_histogram as bh
import pytest
from graphed import Session, compile_ir
from graphed.awkward import AwkwardBackend, from_awkward

import graphed_histogram as gh

#: fixed so the capture is reproducible: the blob carries the source's FORM, not its values
EVENTS = ak.Array({"x": [1.0, 4.0, 7.0, 2.5, 6.0] * 8, "w": [0.5, 1.0, 2.0, 1.5, 0.2] * 8})

OLD, NEW = "1.4.0", "99.0.0.dev-m49"


def _blob() -> bytes:
    """§6.3's case: NO context handle and NO `Varied` input, so the §6.1d broadcast seam is not
    recorded and this graph serializes exactly as it did before m48."""
    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", EVENTS)
    hist = gh.boost.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight())
    hist.fill(events.x, weight=events.w)
    return bytes(compile_ir(session, hist.fill_nodes()[0]).ir)


def _blob_at(monkeypatch: Any, version: str) -> bytes:
    monkeypatch.setattr(bh, "__version__", version)
    return _blob()


def _strip(blob: bytes, version: str) -> bytes:
    """Remove ONE side's own length-prefixed `bh.__version__` from a GIR blob."""
    return blob.replace(len(version).to_bytes(4, "little") + version.encode(), b"")


def test_the_version_really_reaches_the_serialized_bytes(monkeypatch: Any) -> None:
    """The live instrument: were the version absent, every comparison below would pass whichever
    pattern it stripped with."""
    old, new = _blob_at(monkeypatch, OLD), _blob_at(monkeypatch, NEW)
    assert old != new
    assert old.count(len(OLD).to_bytes(4, "little") + OLD.encode()) == 1
    assert new.count(len(NEW).to_bytes(4, "little") + NEW.encode()) == 1


def test_per_side_stripping_makes_two_release_versions_of_one_graph_agree(monkeypatch: Any) -> None:
    old, new = _blob_at(monkeypatch, OLD), _blob_at(monkeypatch, NEW)
    assert _strip(old, OLD) == _strip(new, NEW)


@pytest.mark.parametrize("pattern", [OLD, NEW])
def test_one_pattern_applied_to_both_sides_does_not(monkeypatch: Any, pattern: str) -> None:
    """The discriminator. A comparison that derives its pattern from the LIVE side only leaves the
    other side's version in place, so the two blobs differ by exactly those bytes."""
    old, new = _blob_at(monkeypatch, OLD), _blob_at(monkeypatch, NEW)
    assert _strip(old, pattern) != _strip(new, pattern)


def test_the_stripped_graph_carries_no_version_bytes_of_either_release(monkeypatch: Any) -> None:
    """What the strip is FOR: the comparable artifact is the graph, and the graph does not depend
    on which boost-histogram release recorded it."""
    for version in (OLD, NEW):
        stripped = _strip(_blob_at(monkeypatch, version), version)
        assert len(OLD).to_bytes(4, "little") + OLD.encode() not in stripped
        assert len(NEW).to_bytes(4, "little") + NEW.encode() not in stripped
        assert stripped
