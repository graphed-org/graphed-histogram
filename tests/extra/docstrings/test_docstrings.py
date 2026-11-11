"""The public docstrings' examples run, and no public docstring names a private name."""

from __future__ import annotations

import doctest
import inspect
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import graphed_histogram as gh
from graphed_histogram import histserv

_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
_PRIVATE = re.compile(r"(?<!\w)_[A-Za-z]\w*")


def _run(obj: Any, globs: dict[str, Any]) -> doctest.TestResults:
    runner = doctest.DocTestRunner(optionflags=_FLAGS)
    for test in doctest.DocTestFinder(recurse=False).find(obj, globs=globs):
        test.globs = globs  # the package docstring's setup carries into each example
        runner.run(test, clear_globs=False)
    return runner.summarize(verbose=False)


def _public() -> Iterator[tuple[str, Any]]:
    for module in (gh, gh.boost, histserv):
        yield module.__name__, module
        for name in getattr(module, "__all__", [n for n in vars(module) if n[0] != "_"]):
            obj = getattr(module, name)
            if getattr(obj, "__module__", "").startswith("graphed_histogram"):
                yield f"{module.__name__}.{name}", obj
                if inspect.isclass(obj):
                    for attr, member in vars(obj).items():
                        # a data attribute's __doc__ is its type's, not this package's
                        if attr[0] != "_" and member.__doc__ not in (None, type(member).__doc__):
                            yield f"{module.__name__}.{name}.{attr}", member


EXAMPLED = list(
    {
        id(obj): obj for _, obj in _public() if obj is not gh and ">>>" in (obj.__doc__ or "")
    }.values()
)


def test_each_named_entry_point_has_an_example() -> None:
    named = [gh.plan, gh.unpack, gh.label_listing, gh.add_histograms]
    named += [gh.Histogram.fill, gh.Histogram.plan]
    assert all(any(obj is e for e in EXAMPLED) for obj in named)


@pytest.mark.parametrize("obj", EXAMPLED, ids=lambda obj: obj.__qualname__)
def test_the_example_runs(obj: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    globs: dict[str, Any] = {}
    setup = _run(gh, globs)
    assert setup.failed == 0 and setup.attempted > 0
    result = _run(obj, globs)
    assert result.attempted > 0, f"{obj.__qualname__} has no example"
    assert result.failed == 0


def test_no_public_docstring_names_a_private_name() -> None:
    named = {
        (where, found)
        for where, obj in _public()
        for found in _PRIVATE.findall(inspect.getdoc(obj) or "")
    }
    assert named == set()


def test_the_scan_sees_a_private_name() -> None:
    assert _PRIVATE.findall("reads ``_label_maps``, _x and gh._y, not __init__") == [
        "_label_maps",
        "_x",
        "_y",
    ]
