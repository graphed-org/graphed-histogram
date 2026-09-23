"""A rank mismatch whose leading pair would widen still gets boost's own refusal."""

from __future__ import annotations

import boost_histogram as bh
import pytest

import graphed_histogram as gh


def test_rank_mismatch_refused_by_boost() -> None:
    a = bh.Histogram(bh.axis.Regular(4, 0, 1, growth=True), bh.axis.Regular(2, 0, 1))
    a.fill(1.3, 0.2)
    b = bh.Histogram(bh.axis.Regular(4, 0, 1, growth=True))
    b.fill(-0.3)
    with pytest.raises(ValueError, match="axes have different length"):
        gh.add_histograms(a, b)
