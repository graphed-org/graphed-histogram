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
— which is what the `dev` extra asks for — resolves to the newest PyPI release, and these tests
track `graphed`'s git tip (`.github/workflows/ci.yml` pins the exact commit). Getting it from git
first pins the version the later editable install then leaves alone. (This is also the step that
compiles, hence the Rust toolchain.)

The `dev` extra pulls in the rest of what the test suite uses: `hist` (2.12 or later, which
carries the `hist.graphed` builder), `pyarrow`, `pandas`, and the test/lint/type tools.

## Run the checks

These are the same checks CI runs on a pull request:

```bash
ruff check . && ruff format --check .
mypy
pytest tests/frozen tests/extra --cov=graphed_histogram --cov-branch --cov-report=json --cov-report=xml
python scripts/coverage_gate.py coverage.json
sphinx-build -W -b html docs docs/_build/html
```

Or run the lint and type checks exactly as CI does, in one shot:

```bash
uvx prek@0.4.5 run --all-files
```

Notes:

- `mypy` runs in strict mode over `src/`, `tests/`, and `scripts/` (configured in `pyproject.toml`).
- Coverage policy: every source file must independently reach >=90% line+branch coverage
  (`scripts/coverage_gate.py`, never lowered), and every pull request must cover >=98% of its
  own added/changed lines (`diff-cover` against `main`, enforced once this policy is on `main`).
- CI runs both `tests/frozen` and `tests/extra`, which needs the optional backends installed.
- The docs build treats warnings as errors (`-W`); a broken cross-reference or a
  mismatched section underline fails the build.

## Propose a change

1. Open an issue or a draft PR describing the change first if it touches public API.
2. Add tests under `tests/` for new behaviour — a test should fail without your change.
3. Make the four checks above pass locally.
4. Open a pull request. Keep it to one logical change; docs updates ride along with the
   code they describe.
