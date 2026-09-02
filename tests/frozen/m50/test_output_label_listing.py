"""m50 — §9.1 plan-level `{output: [labels]}` listing.

Over a THREE-output program — a sibling-mode varied output, an AXIS-MODE varied output, and one no
variation reaches — the listing maps each output to its labels in §2.4 (fold) order, the unvaried
output to `["nominal"]`. The AXIS-MODE arm is what makes it discriminating: that output's slot key
is `(output, None)` and carries no label, so an implementation reading labels off the KEY answers
`[None]`/`[]` for it and is red, while a sibling-only program admits that implementation. The two
varied outputs carry the SAME variations, so the listing must not vary with the MODE.

The listing verb and the axis-mode opt-in are both absent at the m50 baseline: FAILS today.
"""

from __future__ import annotations

from m50_axis_fixtures import fill_weight_program, partitioned, weighted

import graphed_histogram as gh

FOLD_ORDER = ["nominal", "wgt_up", "wgt_down"]  # §2.4: nominal first, then vary-tag insertion order


def _three_output_hists() -> dict[str, gh.boost.Histogram]:
    _s, events, _src = partitioned()
    return {
        "sib": fill_weight_program(False, source=events),  # sibling-mode varied
        "ax": fill_weight_program(True, source=events),  # axis-mode varied — same variations
        "plain": weighted().fill(events.MET.pt),  # no variation reaches it
    }


def test_label_listing_maps_each_output_to_its_variations_in_fold_order() -> None:
    listing = gh.label_listing(_three_output_hists())
    assert listing["sib"] == FOLD_ORDER
    assert listing["ax"] == FOLD_ORDER  # MODE-independent — same variations, same list
    assert listing["plain"] == ["nominal"]


def test_the_axis_output_entry_equals_the_sibling_entry() -> None:
    """The `(output, None)` axis key carries no label; a key-reading implementation answers `[None]`
    or `[]` for `"ax"` and reds here, while the sibling output would admit it."""
    listing = gh.label_listing(_three_output_hists())
    assert listing["ax"] == listing["sib"]
    assert None not in listing["ax"]
    assert listing["ax"] != []
