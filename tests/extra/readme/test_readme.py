"""Every python block in README.md runs, in order, and prints what the README says it prints."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).parents[3] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def _printed(block: str) -> list[str]:
    """The ``# ...`` lines right after each ``print(...)`` line: what the README says it prints."""
    out, after_print = [], False
    for line in block.splitlines():
        stripped = line.strip()
        if after_print and stripped.startswith("# "):
            out.append(stripped[2:])
            continue
        after_print = stripped.startswith("print(")
    return out


def test_every_block_prints_what_the_readme_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace: dict[str, object] = {"__name__": "__main__"}
    assert len(BLOCKS) >= 5
    for i, block in enumerate(BLOCKS):
        # a block replacing the first program's body runs on a copy, as its own program
        scope = dict(namespace) if block.startswith("# keep the imports") else namespace
        exec(compile(block, f"README.md block {i}", "exec"), scope)
        assert capsys.readouterr().out.splitlines() == _printed(block), f"block {i}"
