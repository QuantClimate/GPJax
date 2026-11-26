# Repository Guidelines

## Communication

When responding locally or writing documents, always use ASD-STE100 Simplified Technical English unless otherwise instructed.

## What is GPJax?

GPJax is a Gaussian process library built on JAX. The API mirrors GP mathematics: you compose a `Prior` (kernel + mean function), multiply it by a `Likelihood` to get a `JointModel`, condition that model on data to get a `Posterior`, then optimise an objective. All modules are Equinox `eqx.Module` subclasses, so they are JAX pytrees and work with `jit`, `vmap`, and `grad`.

## Project Structure & Module Organization
Core GP primitives live in `gpjax/`, with modules such as `kernels/`, `mean_functions.py`, `likelihoods.py`, `variational_families.py`, `conditioning.py`, and `linalg/` mirroring the mathematical decomposition of a GP. Two subpackages hold larger model families: `gpjax/state_space/` (state-space GPs and SDE solvers) and `gpjax/models/` (OILMM). Tests that shadow these modules live in `tests/`—keep filenames aligned (for example, `tests/test_likelihoods.py`) so stack traces map to the right file. Example notebooks are `py:percent` scripts in `docs/examples/` and are rendered into the Sphinx site from `docs/` (output goes to `docs/_build/html`). Benchmarks live in `benchmarks/` (ASV). The root `static/` directory holds the JOSS paper and `CONTRIBUTING.md`; documentation assets live in `docs/static/`. Repo automation resides in the root config files.

## skgp

The `skgp/` folder holds skgp, a second package: scikit-learn-style Gaussian process models built on GPJax. skgp and GPJax make a uv workspace with one `uv.lock` (`docs/adr/0003-skgp-as-workspace-member.md`). Each package has its own version, PyPI listing, docs site and release tag (`v1.3.0` for GPJax, `skgp-v0.1.0` for skgp). When you work in `skgp/`, obey `skgp/AGENTS.md`. GPJax's commands below act on GPJax only; `uv sync` at the root does not install skgp.

## Build, Test, and Development Commands
Create the dev environment with `uv venv && uv sync`, then install the pre-commit hooks with `uv run prek install`. All commands must be prefixed with `uv run`:

```bash
uv run poe test           # pytest with xdist (8 workers), beartype shape checks, skips `slow`
uv run poe test-slow      # the same, including tests marked `slow`
uv run poe lint           # ruff format check + ruff check --fix
uv run poe typecheck      # ty static type check on gpjax/ (every diagnostic fails)
uv run poe format         # ruff import sorting + formatting (mutates files)
uv run poe docstrings     # xdoctest on gpjax/
uv run poe all-tests      # lint + typecheck + docstrings + bench-check + test (CI gate)
uv run poe coverage       # pytest with an XML coverage report for CI
uv run poe docs           # sphinx-build: executes + caches the example notebooks
uv run poe docs-ci        # smoke render, warnings-as-errors (the CI docs gate)
uv run poe docs-serve     # build, then serve docs/_build/html at localhost:8000
uv run poe docs-live      # live-reloading docs server for local authoring
uv run poe docs-linkcheck # sphinx linkcheck, no notebook execution
uv run poe docs-clean     # remove built docs (keeps the notebook exec cache)
uv run poe bench          # run the ASV benchmarks against HEAD
```

Run a single test file or test:
```bash
uv run pytest tests/test_kernels/test_stationary.py -v
uv run pytest tests/test_gps.py::test_conjugate_posterior -v
```

## Architecture

Read `ARCHITECTURE.md` before you change code in `gpjax/`. It describes the modelling pipeline, the parameter system (`val()` and wrapped models), kernels, linear algebra, objectives, the optimisers in `gpjax/fit.py`, and `Dataset`.

## Changing the public API

Other packages use GPJax's public API. For example, skgp calls `gpx.fit`, `gpx.gps.Prior`, `gpx.objectives.elbo` and `gpx.Dataset`. When you rename or remove a public function, class, argument or module:

1. Add the new name. Keep the old name. Make it emit a `DeprecationWarning` with `warnings.warn`, and give the new name in the message.
2. Remove the old name only in a major release (for example, 2.0.0), and only after skgp stops using it.

skgp's tests fail on a GPJax `DeprecationWarning`, so skgp changes to the new name before GPJax removes the old name. `docs/contributing.md` gives the same rule to human contributors.

## Coding Style & Naming Conventions
Code follows Ruff format and Ruff lint with an 88-character limit and google docstring convention. Use `snake_case` for functions, `UpperCamelCase` for classes, and keep module paths lowercase. Public APIs should ship with type hints—prefer jaxtyping annotations for shapes/dtypes—and docstrings describing inputs, outputs, and key JAX transforms (`jit`, `vmap`, etc.).

- `F722` suppressed (jaxtyping string annotations like `"N D"`)
- Unicode math identifiers allowed in docstrings and comments (`RUF002`/`RUF003` suppressed)
- Imports: `isort` via ruff with combined-as-imports and force-sort-within-sections
- Notebooks must stay in `py:percent` format so CI linters can process them consistently

## Testing Guidelines
CI runs the full test suites of GPJax and skgp on every pull request. Locally, run only the tests for the code you change, for example `uv run pytest tests/test_fit.py`. `CODING_STANDARDS.md` holds the rules that review applies to tests.

- `tests/conftest.py` enables `jax_enable_x64` and installs the jaxtyping import hook with beartype over `gpjax`, so every test checks array shapes and dtypes at runtime. beartype is a dev dependency only; installing GPJax does not install it
- `uv run poe typecheck` (ty) checks types statically. ty cannot check jaxtyping shapes; the beartype hook checks them. `[tool.ty.rules]` in `pyproject.toml` turns off the rules that have findings in the current code. When you fix all the findings of a rule, remove its line
- pytest config treats warnings as errors (`filterwarnings = ["error", "ignore::DeprecationWarning"]`), so solve noisy deprecations locally
- `poe test` runs 8 xdist workers and skips tests marked `slow`
- Hypothesis is configured with `deadline=None, max_examples=20`

## Examples

Stored in `docs/examples/` as `py:percent` format (jupytext), executed at docs
build time by MyST-NB (`nb_custom_formats` in `docs/conf.py`). Convert with:
```bash
jupytext --to notebook example.py       # .py -> .ipynb
jupytext --to py:percent example.ipynb  # .ipynb -> .py
```

## Commit & Pull Request Guidelines
Commits use short, imperative subjects and often tag the change scope (`ci(deps): bump action`, `fix: limit dtype promotion (#573)`). Reference related issues using `Fixes #123` so automation links threads. Pull requests should cover motivation, the solution, verification commands, and screenshots when docs change. The prek hooks run ruff and ty before each commit, and CI runs the tests. Rebuild the docs if user-facing copy moved, and add a CHANGELOG entry when behaviour changes.
