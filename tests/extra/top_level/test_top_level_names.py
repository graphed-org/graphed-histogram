"""``graphed_histogram`` reports its version and lists only its public names."""

from __future__ import annotations

import importlib.metadata

import graphed_histogram as gh


def test_version_is_the_installed_distributions() -> None:
    assert gh.__version__ == importlib.metadata.version("graphed-histogram")


def test_dir_is_all() -> None:
    assert dir(gh) == sorted(gh.__all__)
