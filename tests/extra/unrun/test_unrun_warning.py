"""Reading a histogram with staged fills warns at the user's line, and only then."""

from __future__ import annotations

import copy
import operator
import sys
import warnings
from collections.abc import Callable
from typing import Any

import boost_histogram as bh
import hist
import hist.graphed as hg
import numpy as np
import pytest
from graphed import Session
from graphed.core.execution import SequentialRunner
from graphed.numpy import NumpyBackend
from test_deferred_histograms import _source

import graphed_histogram as gh

DATA = np.array([10.0, 20.0, 30.0, 150.0])


def _x(data: np.ndarray = DATA) -> Any:
    return _source(Session(NumpyBackend()), "x", data)[0]


def _empty() -> Any:
    return hg.Hist.new.Reg(50, 0, 200, name="x").Double()


def _staged() -> Any:
    return _empty().fill(_x())


def _eager() -> Any:
    return hist.Hist.new.Reg(50, 0, 200, name="x").Double()


def _run(h: Any) -> Any:
    return SequentialRunner().run(h.plan()).value


def _lineno() -> int:
    return sys._getframe(1).f_lineno


def _setitem(h: Any) -> None:
    h[3] = 1.0


#: every way to read or use a histogram's storage, reaching each seam directly or through hist
READS: dict[str, Callable[[Any], object]] = {
    "values": lambda h: h.values(),
    "variances": lambda h: h.variances(),
    "counts": lambda h: h.counts(),
    "sum": lambda h: h.sum(),
    "view": lambda h: h.view(),
    "empty": lambda h: h.empty(),
    "index": lambda h: h[3],
    "slice": lambda h: h[2:5],
    "sum-axis": lambda h: h[:: bh.sum],
    "setitem": _setitem,
    "project": lambda h: h.project("x"),
    "integrate": lambda h: h.integrate("x"),
    "T": lambda h: h.T,
    "to_numpy": lambda h: h.to_numpy(),
    "asarray": np.asarray,
    "density": lambda h: h.density(),
    "reset": lambda h: h.reset(),
    "eq": lambda h: h == _eager(),
    "ne": lambda h: h != _eager(),
    "req": lambda h: _eager() == h,
    "add": lambda h: h + h,
    "iadd": lambda h: h.__iadd__(_eager()),
    "sub": lambda h: h - _eager(),
    "isub": lambda h: h.__isub__(_eager()),
    "mul": lambda h: h * 2,
    "imul": lambda h: h.__imul__(_eager()),
    "div": lambda h: h / 2,
    "idiv": lambda h: h.__itruediv__(_eager()),
    "radd": lambda h: _eager() + h,
    "rsub": lambda h: _eager() - h,
    "rmul": lambda h: _eager() * h,
    "rdiv": lambda h: _eager() / h,
    "sum-builtin": lambda h: sum([h, h]),
    "scalar-radd": lambda h: 1.0 + h,
    "scalar-rsub": lambda h: 1.0 - h,
    "scalar-rmul": lambda h: 2.0 * h,
    "scalar-rdiv": lambda h: 1.0 / h,
    "hist.Hist": hist.Hist,
    "bh.Histogram": bh.Histogram,
}
#: staged in either operand, which only a same-class pair cannot reach through reflection
OTHER_OPERAND: dict[str, Callable[[Any], object]] = {
    "add": lambda h: _empty() + h,
    "sub": lambda h: _empty() - h,
    "mul": lambda h: _empty() * h,
    "div": lambda h: _empty() / h,
}
SEAMS = [*READS.values(), *OTHER_OPERAND.values()]
IDS = [*READS, *(f"other-{k}" for k in OTHER_OPERAND)]


def _unrun(record: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in record if w.category is gh.UnrunHistogramWarning]


def _recorded(call: Callable[[], object]) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        call()
    return _unrun(record)


@pytest.mark.parametrize("read", SEAMS, ids=IDS)
def test_a_read_of_a_staged_histogram_warns(read: Callable[[Any], object]) -> None:
    h = _staged()
    assert _recorded(lambda: read(h))


@pytest.mark.parametrize("read", SEAMS, ids=IDS)
def test_the_error_filter_makes_every_seam_raise(read: Callable[[Any], object]) -> None:
    h = _staged()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=gh.UnrunHistogramWarning)
        with pytest.raises(gh.UnrunHistogramWarning, match="has not run"):
            read(h)


@pytest.mark.parametrize("read", READS.values(), ids=READS)
@pytest.mark.parametrize(
    "make",
    [_empty, _eager, lambda: _run(_staged()), lambda: _run(_empty().fill(_x(DATA[:0])))],
    ids=["no-fills", "eager", "run-result", "zero-event-run-result"],
)
def test_nothing_else_warns(make: Callable[[], Any], read: Callable[[Any], object]) -> None:
    h = make()
    assert not _recorded(lambda: read(h))


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize("right", [_empty, _staged], ids=["no-fills", "staged"])
def test_a_histogram_on_the_left_keeps_its_own_operator(
    op: Callable[[Any, Any], Any], right: Callable[[], Any]
) -> None:
    left = _eager().fill([10.0, 20.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = op(left, right())
    want = op(left, _eager())
    assert type(got) is hist.Hist
    assert np.array_equal(got.view(flow=True), want.view(flow=True), equal_nan=True)


def test_a_zero_event_run_reads_as_empty_without_a_marker() -> None:
    result = _run(_empty().fill(_x(DATA[:0])))
    assert result.sum() == 0.0
    assert "staged" not in repr(result)


def test_the_warning_says_what_to_do_and_what_is_lost() -> None:
    h = _staged()
    (w,) = _recorded(h.values)
    text = str(w.message)
    assert "\n" not in text
    assert "not recorded into the run" in text
    assert "filled = graphed.materialize(h)" in text
    assert "`h` itself stays un-run" in text


def test_the_warning_names_the_users_line_through_hist() -> None:
    h = _staged()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        hist.Hist(h)
        line = _lineno() - 1
    (w,) = _unrun(record)
    assert (w.filename, w.lineno) == (__file__, line)


@pytest.mark.parametrize("plot", ["method", "mplhep"])
def test_the_warning_names_the_users_line_through_plotting(plot: str) -> None:
    mplhep = pytest.importorskip("mplhep")
    pytest.importorskip("matplotlib").use("Agg")
    h = _staged()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        if plot == "method":
            h.plot()
            line = _lineno() - 1
        else:
            mplhep.histplot(h)
            line = _lineno() - 1
    (w,) = _unrun(record)  # one, though plotting reads it several times
    assert (w.filename, w.lineno) == (__file__, line)


def test_repeat_reads_on_one_line_warn_once_per_histogram() -> None:
    h, other = _staged(), _staged()

    def reads() -> None:
        for each in (h, h, other, other):
            each.values()

    assert len(_recorded(reads)) == 2


def test_copies_of_one_template_warn_once_each() -> None:
    template, x = _empty(), _x()
    copies = [copy.copy(template).fill(x) for _ in range(3)]

    def reads() -> None:
        for each in copies:
            each.values()

    assert len(_recorded(reads)) == 3


@pytest.mark.parametrize(
    "use",
    [lambda h: h * 2, lambda h: h + h, lambda h: h - h, lambda h: 1.0 - h, lambda h: h[2:5]],
    ids=["mul", "add", "sub", "rsub", "slice"],
)
def test_one_use_warns_once_though_it_copies_the_histogram(use: Callable[[Any], object]) -> None:
    h = _staged()
    assert len(_recorded(lambda: use(h))) == 1


def test_the_suite_filter_sees_a_multi_axis_histogram() -> None:
    x = _x()
    h = hg.Hist.new.Reg(5, 0, 200, name="x").Reg(5, 0, 200, name="y").Double().fill(x, x)
    with pytest.raises(gh.UnrunHistogramWarning):
        h.values()


def test_a_read_on_another_line_warns_again() -> None:
    h = _staged()

    def reads() -> None:
        h.values()
        h.values()

    assert len(_recorded(reads)) == 2


def test_the_same_line_of_another_file_warns_again() -> None:
    h = _staged()

    def reads() -> None:
        for file in ("cell_one", "cell_two"):
            exec(compile("h.values()", file, "exec"), {"h": h})

    assert [w.filename for w in _recorded(reads)] == ["cell_one", "cell_two"]


def test_the_always_filter_shows_every_read() -> None:
    h = _staged()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            h.values()
    assert len(_unrun(record)) == 3


def test_the_error_filter_applies_after_the_default_showed_it_once() -> None:
    h = _staged()

    def read() -> None:
        h.values()

    assert _recorded(read)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=gh.UnrunHistogramWarning)
        with pytest.raises(gh.UnrunHistogramWarning):
            read()


def test_reads_still_return_what_the_empty_storage_holds() -> None:
    h = _staged()
    with pytest.warns(gh.UnrunHistogramWarning):
        assert np.array_equal(h.values(), np.zeros(50))
    with pytest.warns(gh.UnrunHistogramWarning):
        assert (h * 2).staged_fills() == 1


def test_running_the_histogram_ignores_what_was_done_with_it_and_warned_so() -> None:
    with pytest.warns(gh.UnrunHistogramWarning):
        doubled = _staged() * 2
    assert _run(doubled).sum() == len(DATA)


@pytest.mark.parametrize(
    ("n", "marker"), [(1, "# 1 staged fill, not run"), (2, "# 2 staged fills, not run")]
)
@pytest.mark.parametrize("text", [repr, str])
def test_the_text_repr_marks_staged_fills(
    n: int, marker: str, text: Callable[[object], str]
) -> None:
    h = _empty()
    for _ in range(n):
        h.fill(_x())
    assert not _recorded(lambda: text(h))
    assert text(h).endswith(marker)
    assert text(h).startswith("Hist(Regular(50, 0, 200, name='x')")


def test_the_marker_names_a_plain_deferred_histogram_too() -> None:
    h = gh.boost.Histogram(bh.axis.Regular(5, 0.0, 5.0)).fill(_x())
    assert repr(h).startswith("Histogram(Regular(5, 0, 5)")
    assert repr(h).endswith("# 1 staged fill, not run")


@pytest.mark.parametrize("make", [_empty, _eager, lambda: _run(_staged())])
def test_nothing_else_is_marked(make: Callable[[], Any]) -> None:
    assert "staged" not in repr(make())
    assert "staged" not in str(make())


def test_a_named_hist_warns_and_is_marked() -> None:
    h = hg.NamedHist.new.Reg(5, 0, 5, name="x").Double().fill(x=_x())
    assert repr(h).startswith("NamedHist(") and repr(h).endswith("# 1 staged fill, not run")
    with pytest.warns(gh.UnrunHistogramWarning):
        h.values()


def test_a_copys_fill_stages_on_the_copy_alone() -> None:
    x = _x()
    h = _empty().fill(x)
    c = copy.copy(h)
    c.fill(x * 0.5)
    assert (h.staged_fills(), c.staged_fills()) == (1, 2)
    assert (len(h.recording().fill_specs), len(c.recording().fill_specs)) == (1, 2)
    assert (_run(h).sum(), _run(c).sum()) == (len(DATA), 2 * len(DATA))
