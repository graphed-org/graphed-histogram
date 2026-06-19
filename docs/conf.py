"""Sphinx configuration for graphed-histogram."""

from __future__ import annotations

project = "graphed-histogram"
author = "graphed-org"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
exclude_patterns = ["_build"]
templates_path = ["_templates"]
html_theme = "furo"
html_title = "graphed-histogram"
autodoc_typehints = "description"
autosummary_generate = True
autosummary_imported_members = False
# the dev-extra backends are not installed in the docs job; keep autodoc from importing them
autodoc_mock_imports = ["numpy", "boost_histogram"]
