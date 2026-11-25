# Repository Guidelines

## Communication

When responding locally or writing documents, always use ASD-STE100 Simplified Technical English unless otherwise instructed.

## What is GPJax?

GPJax is a Gaussian process library built on JAX. The API mirrors GP mathematics: you compose a `Prior` (kernel + mean function), multiply it by a `Likelihood` to get a `JointModel`, condition that model on data to get a `Posterior`, then optimise an objective. All modules are Equinox `eqx.Module` subclasses, so they are JAX pytrees and work with `jit`, `vmap`, and `grad`.

## Project Structure & Module Organization
Core GP primitives live in `gpjax/`, with modules such as `kernels/`, `mean_functions.py`, `likelihoods.py`, `variational_families.py`, `conditioning.py`, and `linalg/` mirroring the mathematical decomposition of a GP. Two subpackages hold larger model families: `gpjax/state_space/` (state-space GPs and SDE solvers) and `gpjax/models/` (OILMM). Tests that shadow these modules live in `tests/`—keep filenames aligned (for example, `tests/test_likelihoods.py`) so stack traces map to the right file. Example notebooks are `py:percent` scripts in `docs/examples/` and are rendered into the Sphinx site from `docs/` (output goes to `docs/_build/html`). Benchmarks live in `benchmarks/` (ASV). The root `static/` directory holds the JOSS paper and `CONTRIBUTING.md`; documentation assets live in `docs/static/`. Repo automation resides in the root config files.

## Build, Test, and Development Commands
Create the dev environment with `uv venv && uv sync`. All commands must be prefixed with `uv run`:

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

### Core pipeline

```
Prior(kernel, mean_function)  *  Likelihood  -->  JointModel
         |                                            |
    prior(Xtest)                          model.condition(train_data) --> Posterior
         |                                            |
   GaussianDistribution                     posterior(Xtest) --> GaussianDistribution
```

`Prior.__mul__(likelihood)` dispatches through `construct_model()` and returns the correct joint model type:
- `Gaussian` likelihood -> `ConjugateModel` (closed-form conditioning, `ExactPosterior`)
- `Bernoulli`/`Poisson` -> `NonConjugateModel` (latent function values optimised, `LatentPosterior`)
- Heteroscedastic likelihoods carry a second (noise) prior, which the two-operand product cannot express. Build them directly: `HeteroscedasticModel(prior, likelihood, noise_prior=...)`.

`JointModel.predict(test_inputs, train_data)` is a shortcut for `model.condition(train_data)(test_inputs)`.

### Parameter system (`gpjax/parameters.py`)

Parameters are `paramax.AbstractUnwrappable` subclasses. Each class stores its value in an unconstrained internal field and implements `unwrap()` to apply the constraining bijection:

| Class | Bijection | Internal storage |
|---|---|---|
| `Real` | Identity | `value` (unchanged) |
| `PositiveReal` | Softplus | `_unconstrained` via `inv_softplus` |
| `NonNegativeReal` | Softplus | `_unconstrained` via `inv_softplus` |
| `SigmoidBounded` | Sigmoid scaled to `[low, high]` | `_unconstrained` via `logit` |
| `LowerTriangular` | Fill-triangular | `_flat` vector |

**Models are always held in wrapped form.** Wrapping happens once, at construction; nothing unwraps a model tree, including `fit`, which returns a wrapped model. The rule is:

> Call `val(param)` (from `gpjax.parameters`) wherever a parameter meets arithmetic. Pass models around untouched everywhere else.

`val` is idempotent and recursive: it returns the constrained value of a parameter (resolving nested wrappers such as `paramax.non_trainable`) and returns plain arrays or floats unchanged, so it never needs an `isinstance` guard. When writing a new kernel, mean function, likelihood, or objective, every parameter read needs a `val()`.

Forgetting `val()` is not silent — a parameter is a pytree node, not an array, so arithmetic raises `TypeError: unsupported operand type(s) ... 'PositiveReal'` immediately. The one quiet exception is `jax.tree_util.tree_map` over a model, which reaches the *unconstrained* leaf; that is what makes optimisation work, but it means tree-mapping arithmetic over a model is not the same as scaling its parameters.

Do not add `paramax.unwrap(model)` to a loss, objective, or prediction function — that is the pre-consolidation pattern, and it reintroduces an undocumented "must be unwrapped first" precondition on user-facing extension points. `paramax.unwrap` remains correct for generic tree introspection (see `gpjax/summary.py`). To freeze parameters, wrap them with `paramax.non_trainable(param)`.

### Kernel system (`gpjax/kernels/`)

`AbstractKernel` defines `__call__(x, y) -> scalar`, `gram(x) -> LinearOperator`, `cross_covariance(x, y) -> array`, and `diagonal(x) -> LinearOperator`. Kernels compose with `+` (SumKernel) and `*` (ProductKernel). Each kernel delegates matrix computation to a `compute_engine` (typically `DenseKernelComputation`).

Kernel categories: `stationary/` (RBF, Matern12/32/52, Periodic, PoweredExponential, RationalQuadratic, White), `nonstationary/` (Linear, Polynomial, ArcCosine), `non_euclidean/` (GraphKernel), `multioutput/` (ICMKernel, LCMKernel), `approximations/` (RFF), `additive/` (OrthogonalAdditiveKernel).

### Linear algebra (`gpjax/linalg/`)

Built on [Lineax](https://docs.kidger.site/lineax/). Kernel `gram()` returns `lx.AbstractLinearOperator` (typically `lx.MatrixLinearOperator`). Use `.as_matrix()` to materialise. Custom operators: `BlockDiag`, `Kronecker`. Key utilities: `cholesky_factor()` (singledispatch, returns lower-triangular operator), `stabilised_cholesky()`, `logdet()`, `logdet_from_factor()`, `add_jitter()`. Linear solves use `lx.linear_solve()`.

### Objectives (`gpjax/objectives.py`)

Functions `(model, Dataset) -> scalar`:
- `conjugate_mll` / `conjugate_loocv` -- for `ConjugateModel`
- `log_posterior_density` (alias `non_conjugate_mll`) -- for `NonConjugateModel`
- `elbo` / `collapsed_elbo` -- for variational families
- `dual_elbo` -- for `DualVariationalGaussian` (t-SVGP; same value as `elbo`, different hyperparameter gradient)
- `heteroscedastic_elbo` -- for heteroscedastic models

Optimise by negating: `nmll = lambda p, d: -conjugate_mll(p, d)`

### Fitting (`gpjax/fit.py`)

Four optimisers: `fit()` (Optax gradient descent with scan), `fit_scipy()` (SciPy L-BFGS-B), `fit_lbfgs()` (Optax L-BFGS with `while_loop`), `fit_natgrads()` (natural-gradient steps on a variational family, alternated with Optax steps on the hyperparameters). All handle the constrained/unconstrained bijection automatically, without converting the model: `eqx.partition`/`eqx.combine` with `eqx.is_array` manage trainable vs static parts, gradients are taken with respect to the internal unconstrained arrays, and the objective receives — and the optimiser returns — a wrapped model. The bijection is applied only at the `val()` call sites inside kernels, mean functions, and likelihoods.

### Variational inference (`gpjax/variational_families.py`)

`VariationalGaussian`, `WhitenedVariationalGaussian`, `DualVariationalGaussian`, `CollapsedVariationalGaussian`, `GraphVariationalGaussian`, `HeteroscedasticVariationalFamily`. All inherit from `AbstractVariationalFamily` and implement `predict()` + `prior_kl()`.

### NumPyro integration

Parameter constraints and bijections come from `numpyro.distributions` (`biject_to`, `constraints`), and `GaussianDistribution` is a NumPyro `Distribution`. For fully Bayesian inference over hyperparameters, write a NumPyro model that samples the hyperparameters with `numpyro.sample` and builds the GP inside the model function; see `docs/examples/numpyro_integration.py`.

### Dataset (`gpjax/dataset.py`)

`Dataset` is a `@dataclass(slots=True)` registered as a JAX pytree. Requires 2D arrays: `X` shape `(N, D)`, `y` shape `(N, Q)`. Warns if inputs are not float64.

## Coding Style & Naming Conventions
Code follows Ruff format and Ruff lint with an 88-character limit and google docstring convention. Use `snake_case` for functions, `UpperCamelCase` for classes, and keep module paths lowercase. Public APIs should ship with type hints—prefer jaxtyping annotations for shapes/dtypes—and docstrings describing inputs, outputs, and key JAX transforms (`jit`, `vmap`, etc.).

- `F722` suppressed (jaxtyping string annotations like `"N D"`)
- Unicode math identifiers allowed in docstrings and comments (`RUF002`/`RUF003` suppressed)
- Imports: `isort` via ruff with combined-as-imports and force-sort-within-sections
- Notebooks must stay in `py:percent` format so CI linters can process them consistently

## Testing Guidelines
Place tests next to their logical counterpart (e.g., kernel additions in `tests/test_kernels/`) and parametrize over dtype/precision combinations to guard against silent promotion. Run `uv run poe test` before every push. Execute `uv run poe all-tests` (lint + typecheck + docstrings + bench-check + pytest) for API or docs changes to confirm ancillary checks stay healthy. Every bug fix should add a regression test, and examples must be updated when behavior shifts.

- `tests/conftest.py` enables `jax_enable_x64` and installs the jaxtyping import hook with beartype over `gpjax`, so every test checks array shapes and dtypes at runtime. beartype is a dev dependency only; installing GPJax does not install it
- `uv run poe typecheck` (ty) checks types statically. ty cannot check jaxtyping shapes; the beartype hook checks them. `[tool.ty.rules]` in `pyproject.toml` turns off the rules that have findings in the current code. When you fix all the findings of a rule, remove its line
- pytest config treats warnings as errors (`filterwarnings = ["error", "ignore::DeprecationWarning"]`), so solve noisy deprecations locally
- Tests run with 8 xdist workers by default; `poe test` skips tests marked `slow`
- Hypothesis is configured with `deadline=None, max_examples=20`
- Use `pytest-mock` for mocking and `hypothesis` for property-based testing. Prefer functions over classes for test organization.

## Examples

Stored in `docs/examples/` as `py:percent` format (jupytext), executed at docs
build time by MyST-NB (`nb_custom_formats` in `docs/conf.py`). Convert with:
```bash
jupytext --to notebook example.py       # .py -> .ipynb
jupytext --to py:percent example.ipynb  # .ipynb -> .py
```

## Commit & Pull Request Guidelines
Commits use short, imperative subjects and often tag the change scope (`ci(deps): bump action`, `fix: limit dtype promotion (#573)`). Reference related issues using `Fixes #123` so automation links threads. Pull requests should cover motivation, the solution, verification commands, and screenshots when docs change. Before requesting review, run `uv run poe lint`, `uv run poe test`, rebuild docs if user-facing copy moved, and add release-note snippets when behavior changes.
