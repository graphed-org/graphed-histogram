"""m52/C5 — §5.1/§8-h: this package holds no second implementation of the fallback rule.

`graphed_histogram` re-implemented "the member this label names, else nominal" privately, so a
point-aware resolution in `graphed` could never reach the fill lowering. The rule has exactly one
implementation after C5, and it lives behind `graphed.member_of`.

A grep that finds nothing is indistinguishable from a grep that never ran, so the null leg carries
both its controls in the SAME test. The MATCHER control puts one-line files the test writes itself
through the identical matcher, one line per alternation branch — no edit to either source tree can
silence it, and a branch that had gone dead would show up as a missing hit rather than as a quiet
empty result. The BEHAVIORAL control asserts the surviving implementation still answers the
fallback, which the rule cannot lose without losing the resolution property this milestone rests on.

The control is deliberately NOT a spelling in `graphed`'s own source: this milestone rewrites the
canonical `_member_for`, and a source-spelling control there would redden a frozen test in this
repository for a legal implementation, invisibly to its owner. The tree is resolved from
`graphed_histogram.__file__` — the editable install points at the real source — never a path literal.
"""

from __future__ import annotations

import re
from pathlib import Path

import graphed
from graphed.awkward import gak
from m52_joint_fill_fixtures import in_memory, joint_context, observable

import graphed_histogram

#: §8-h's rule-grep, verbatim, as a Python alternation
RULE = re.compile(r'else graphed\.nominal|else nominal\(|else .*_members\["nominal"\]')

#: one line per alternation branch — a matcher with a dead branch would read empty for a bad reason
CONTROL_LINES = (
    "value = members[label] if label in members else graphed.nominal(container)",
    "value = members[label] if label in members else nominal(container)",
    'value = self._members.get(label) if label else self._members["nominal"]',
)


def _hits(root: Path) -> list[tuple[str, int, str]]:
    """Every `*.py` line under `root` matching the rule, as `(relative path, line number, text)`."""
    found = []
    for path in sorted(root.rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if RULE.search(line):
                found.append((path.relative_to(root).as_posix(), number, line.strip()))
    return found


def _rendered(hits: list[tuple[str, int, str]]) -> str:
    return "\n".join(f"{path}:{number}: {text}" for path, number, text in hits)


def test_no_second_implementation_of_the_fallback_rule_exists(tmp_path: Path) -> None:
    """The null leg over the installed package's real source tree, run beside both of its controls
    so no leg can report green while the instrument was dead."""
    for index, line in enumerate(CONTROL_LINES):
        directory = tmp_path / f"branch{index}"
        directory.mkdir()
        (directory / "control.py").write_text(line + "\n")
        assert _hits(directory) == [("control.py", 1, line)]

    _session, events = in_memory()
    observable_ = observable(joint_context(events, register_point=False))
    counts = gak.num(events.Jet, axis=1) * 1.0
    varied = graphed.vary(counts, "jer", up=counts * 2.0)
    assert graphed.member_of(varied, "no_such_label") is graphed.nominal(varied)
    assert graphed.member_of(varied, "jer_up") is not graphed.nominal(varied)
    assert graphed.member_of(observable_, "no_such_label") is graphed.nominal(observable_)

    root = Path(graphed_histogram.__file__).parent
    assert (root / "boost.py").exists(), f"{root} is not the graphed_histogram source tree"
    hits = _hits(root)
    assert hits == [], _rendered(hits)
