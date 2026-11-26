# Architecture

How GPJax's code is organised: the modelling pipeline, the parameter system and the main modules. Read it before you change code in `gpjax/`.

## Core pipeline

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

## Parameter system (`gpjax/parameters.py`)

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

## Kernel system (`gpjax/kernels/`)

`AbstractKernel` defines `__call__(x, y) -> scalar`, `gram(x) -> LinearOperator`, `cross_covariance(x, y) -> array`, and `diagonal(x) -> LinearOperator`. Kernels compose with `+` (SumKernel) and `*` (ProductKernel). Each kernel delegates matrix computation to a `compute_engine` (typically `DenseKernelComputation`).

Kernel categories: `stationary/` (RBF, Matern12/32/52, Periodic, PoweredExponential, RationalQuadratic, White), `nonstationary/` (Linear, Polynomial, ArcCosine), `non_euclidean/` (GraphKernel), `multioutput/` (ICMKernel, LCMKernel), `approximations/` (RFF), `additive/` (OrthogonalAdditiveKernel).

## Linear algebra (`gpjax/linalg/`)

Built on [Lineax](https://docs.kidger.site/lineax/). Kernel `gram()` returns `lx.AbstractLinearOperator` (typically `lx.MatrixLinearOperator`). Use `.as_matrix()` to materialise. Custom operators: `BlockDiag`, `Kronecker`. Key utilities: `cholesky_factor()` (singledispatch, returns lower-triangular operator), `stabilised_cholesky()`, `logdet()`, `logdet_from_factor()`, `add_jitter()`. Linear solves use `lx.linear_solve()`.

## Objectives (`gpjax/objectives.py`)

Functions `(model, Dataset) -> scalar`:
- `conjugate_mll` / `conjugate_loocv` -- for `ConjugateModel`
- `log_posterior_density` (alias `non_conjugate_mll`) -- for `NonConjugateModel`
- `elbo` / `collapsed_elbo` -- for variational families
- `dual_elbo` -- for `DualVariationalGaussian` (t-SVGP; same value as `elbo`, different hyperparameter gradient)
- `heteroscedastic_elbo` -- for heteroscedastic models

Optimise by negating: `nmll = lambda p, d: -conjugate_mll(p, d)`

## Fitting (`gpjax/fit.py`)

Four optimisers: `fit()` (Optax gradient descent with scan), `fit_scipy()` (SciPy L-BFGS-B), `fit_lbfgs()` (Optax L-BFGS with `while_loop`), `fit_natgrads()` (natural-gradient steps on a variational family, alternated with Optax steps on the hyperparameters). All handle the constrained/unconstrained bijection automatically, without converting the model: `eqx.partition`/`eqx.combine` with `eqx.is_array` manage trainable vs static parts, gradients are taken with respect to the internal unconstrained arrays, and the objective receives — and the optimiser returns — a wrapped model. The bijection is applied only at the `val()` call sites inside kernels, mean functions, and likelihoods.

## Variational inference (`gpjax/variational_families.py`)

`VariationalGaussian`, `WhitenedVariationalGaussian`, `DualVariationalGaussian`, `CollapsedVariationalGaussian`, `GraphVariationalGaussian`, `HeteroscedasticVariationalFamily`. All inherit from `AbstractVariationalFamily` and implement `predict()` + `prior_kl()`.

## NumPyro integration

Parameter constraints and bijections come from `numpyro.distributions` (`biject_to`, `constraints`), and `GaussianDistribution` is a NumPyro `Distribution`. For fully Bayesian inference over hyperparameters, write a NumPyro model that samples the hyperparameters with `numpyro.sample` and builds the GP inside the model function; see `docs/examples/numpyro_integration.py`.

## Dataset (`gpjax/dataset.py`)

`Dataset` is a `@dataclass(slots=True)` registered as a JAX pytree. Requires 2D arrays: `X` shape `(N, D)`, `y` shape `(N, Q)`. Warns if inputs are not float64.
