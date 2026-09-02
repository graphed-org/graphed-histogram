"""m48 edges the frozen anchors do not reach: the guard's scalar case, the accessor's
several-fills refusal, `fill`'s operand type checks, and `unpack`'s axis-mode key form."""

from __future__ import annotations

import boost_histogram as bh
import numpy as np
import pytest
from graphed import GraphedError, Session
from graphed.numpy import NumpyBackend, NumpyForm

import graphed_histogram as gh
from graphed_histogram.boost import _WeightGuard

DATA = np.linspace(0.0, 9.0, 40)


def _source() -> tuple[Session, object]:
    session = Session(NumpyBackend())
    return session, session.source("x", form=NumpyForm(DATA.dtype, shape=(None,)), data=DATA)


def _hist() -> gh.boost.Histogram:
    return gh.boost.Histogram(bh.axis.Regular(4, 0.0, 10.0), storage=bh.storage.Weight())


def test_a_scalar_factor_passes_the_row_space_guard() -> None:
    """A scalar weight broadcasts against any row count, so the guard must not refuse it — and the
    mismatched-array leg is what shows the guard is live rather than permissive."""
    guard = _WeightGuard("the factor is at the wrong row space")
    value = np.arange(5.0)
    assert guard(2.0, value) == 2.0
    with pytest.raises(GraphedError, match="wrong row space"):
        guard(np.arange(4.0), value)


def test_the_per_label_accessor_refuses_a_histogram_with_several_fill_calls() -> None:
    """Each fill call has its own label set, so `{label: node}` has no single answer across two of
    them; one fill still answers, which is what makes the refusal about the count."""
    session, x = _source()
    del session
    one = _hist().fill(x)
    assert list(gh.fill_nodes_by_label(one)) == ["nominal"]

    two = _hist().fill(x)
    two.fill(x * 0.5)
    with pytest.raises(GraphedError, match="fill_nodes"):
        gh.fill_nodes_by_label(two)


def test_fill_refuses_non_array_weight_and_sample_operands() -> None:
    session, x = _source()
    del session
    with pytest.raises(TypeError, match="weights must be graphed Arrays"):
        _hist().fill(x, weight=[1.5])
    with pytest.raises(TypeError, match="sample= must be a graphed Array"):
        _hist().fill(x, sample=1.5)


def test_unpack_reads_an_axis_mode_slot_as_a_bare_histogram() -> None:
    """§6.1a's third key form, `(output, None)`: the shape is decided by the KEY, so the unpacker
    answers with the histogram itself and never with a `{None: hist}` mapping."""
    axis_mode, plain = bh.Histogram(bh.axis.Regular(2, 0, 1)), bh.Histogram(bh.axis.Regular(2, 0, 1))
    result = gh.unpack({("met", None): axis_mode, "ht": plain})
    assert result == {"met": axis_mode, "ht": plain}
    assert result["met"] is axis_mode
