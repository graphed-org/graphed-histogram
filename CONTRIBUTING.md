# Contributing to graphed-histogram

Thanks for pitching in. This package is pure Python, but its `graphed` dependency is a
compiled extension: **installing `graphed` from source requires a Rust toolchain**
(https://rustup.rs). That install is the only place the toolchain is used — everything else
here is Python.

## Set up a dev environment

From a clone of this repository:

```bash
python -m venv .venv && source .venv/bin/activate
pip install "graphed[awkward,numpy] @ git+https://github.com/graphed-org/graphed@main" graphed-executors
pip install -e ".[dev,docs]"
```

Install `graphed` from git **first**, exactly as CI does. A name-only `graphed[awkward,numpy]`
— which is what the `dev` extra asks for — resolves to the PyPI release, and these tests are
written against the git tip. Getting it from git first pins the version the later editable
install then leaves alone. (This is also the step that compiles, hence the Rust toolchain.)

The `dev` extra pulls in the rest of what the test suite uses: `hist`, `pyarrow`, `pandas`,
and the test/lint/type tools.

The `hist.graphed` builder lives in a fork of `hist`; the PyPI `hist` the `dev` extra installs
does not carry it. To work on that path:

```bash
pip install "hist @ git+https://github.com/graphed-org/hist-graphed-mvp@graphed-mvp"
```

## Run the checks

These are the same checks CI runs on a pull request:

```bash
ruff check . && ruff format --check .
mypy
pytest tests/frozen --cov=graphed_histogram --cov-branch
sphinx-build -W -b html docs docs/_build/html
```

Notes:

- `mypy` runs in strict mode over `src/` (configured in `pyproject.toml`).
- Coverage must stay at or above 90% branch coverage on `graphed_histogram`.
- A bare `pytest` also collects `tests/extra`, which needs the optional backends installed;
  CI runs the acceptance suite under `tests/frozen`, which is what the command above does.
- The docs build treats warnings as errors (`-W`); a broken cross-reference or a
  mismatched section underline fails the build.

## Propose a change

1. Open an issue or a draft PR describing the change first if it touches public API.
2. Add tests under `tests/` for new behaviour — a test should fail without your change.
3. Make the four checks above pass locally.
4. Open a pull request. Keep it to one logical change; docs updates ride along with the
   code they describe.
