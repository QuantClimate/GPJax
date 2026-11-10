# Migrations

One section per release that changed a public API. Work through them in order
from whichever version you are on: each guide only describes the step it names,
so upgrading across two releases means reading two sections.

Releases not listed here made no breaking changes; see the
[changelog](gh:blob/main/CHANGELOG.md) for
the full history.

## 1.0.x → 1.1.0

GPJax `1.1` adds features and fixes. One change can affect existing code: the
`power` of `PoweredExponential` is now constrained to $(0, 2]$, the interval
where the kernel is positive definite.

- A `power` outside $(0, 2]$ now raises `ValueError`. Before, it was accepted
  and gave a Gram matrix that was not positive definite.
- A float `power` in $(0, 2)$ is now a trainable `SigmoidBounded` parameter, so
  `fit` learns it. Before, it was a plain float that `fit` did not change. To
  keep the power fixed, pass `paramax.non_trainable(jnp.asarray(power))`.
- `power=2.0` (the RBF case) stays fixed.
- `kernel.power` is no longer a float. Read its value with `val(kernel.power)`.

## 0.18.x → 1.0.0

GPJax `1.0` restructures the core API around **conditioning**: the joint
model and the conditioned posterior are now distinct objects, and the API
reads as the maths. The full decision record is
[ADR-0001](gh:blob/main/docs/adr/0001-conditioning-architecture.md); the
vocabulary lives in [CONTEXT.md](gh:blob/main/CONTEXT.md).

### The conditioning API

`prior * likelihood` now returns a `JointModel` — the joint p(f, y), the
object `gpx.fit` trains. Conditioning it on data yields the posterior
process, which caches its factorisation and is queried directly:

```python
import gpjax as gpx
import jax.numpy as jnp

xtrain = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
D = gpx.Dataset(X=xtrain, y=jnp.sin(xtrain))
xtest = jnp.linspace(0.0, 1.0, 50).reshape(-1, 1)
prior = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Zero(), kernel=gpx.kernels.RBF()
)

model = prior * gpx.likelihoods.Gaussian()  # JointModel (was: ConjugatePosterior)
posterior = model.condition(D)      # Posterior — equivalently: model | D
predictive = posterior(xtest)       # was: posterior.predict(xtest, D)
evidence = posterior.log_marginal_likelihood
```

`model.predict(xtest, D)` and `model(xtest, D)` remain as documented one-line
sugar for `model.condition(D)(xtest)` — existing prediction code keeps
working. When predicting repeatedly, condition once and reuse the returned
posterior.

### Renames

| 0.18.x | 1.0.0 |
| --- | --- |
| `ConjugatePosterior` | `ConjugateModel` |
| `NonConjugatePosterior` | `NonConjugateModel` |
| `HeteroscedasticPosterior` / `ChainedPosterior` | `HeteroscedasticModel` |
| `construct_posterior` | `construct_model` |
| `AbstractPrior` | folded into `Prior` |
| `AbstractPosterior` | split into `JointModel` and `Posterior` |
| `StateSpaceConjugatePosterior` | `StateSpaceConjugateModel` |
| `return_covariance_type=` kwarg | `covariance=` |

### Likelihoods are pure conditionals

`num_datapoints` is gone from every likelihood constructor:

```python
likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)   # was: Gaussian(num_datapoints=D.n, ...)
```

The minibatch ELBO scale is now derived from the dataset itself
(`get_batch` stamps the full size onto each minibatch as `Dataset.n_total`),
so it can no longer be wrong. `MultiOutputGaussian(num_outputs=P)` likewise
drops the argument.

`HeteroscedasticGaussian` no longer takes a `noise_prior`; the noise process
lives on the model, which is constructed directly because it holds two
priors:

```python
noise_prior = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Zero(), kernel=gpx.kernels.RBF()
)
het_model = gpx.gps.HeteroscedasticModel(
    prior=prior,
    likelihood=gpx.likelihoods.HeteroscedasticGaussian(),
    noise_prior=noise_prior,
)
```

### Non-conjugate latents initialise lazily

`NonConjugateModel` no longer sizes its latent vector at construction (that
was what `num_datapoints` was for). `gpx.fit` initialises it automatically on
first contact with the data; to work with the latent before fitting, call
`model = model.init_latent(D.n)` explicitly.

### One jitter knob

`Prior.jitter` is now the model's single stabilisation knob, applied exactly
once inside conditioning. The independent `Posterior.jitter` field is gone —
previously `predict` and `conjugate_mll` could factorise *different*
matrices when the two knobs diverged. The variational families' `jitter`
constructor argument is gone for the same reason (see below).

### Variational families condition like everything else

Variational families now hold the joint model in a field named `model`
(they approximate a posterior; they aren't one), and every Gaussian-output
family conditions through the same machinery as joint models:

```python
z = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)
q = gpx.variational_families.VariationalGaussian(
    model=model, inducing_inputs=z          # was: posterior=..., jitter=...
)

q_posterior = q.condition(D)   # a Posterior, exactly like model.condition(D)
predictive = q_posterior(xtest)
marginals = q_posterior(xtest, covariance="diagonal")   # new fast path
```

`condition` takes `train_data` everywhere, on families as on joint models,
and `q | D` is the same sugar. The uncollapsed families
(`VariationalGaussian`, `WhitenedVariationalGaussian`,
`DualVariationalGaussian`, `GraphVariationalGaussian`) already carry the
fitted `q(u)`, so they accept `train_data` for that uniformity and ignore it;
the collapsed family solves its optimal `q*(u)` from the data and consumes
it. `q.predict(xtest, D)` remains documented sugar for
`q.condition(D)(xtest)`.

`prior_kl` is *not* part of that uniform contract and keeps its per-family
signature: `q.prior_kl()` for the uncollapsed families, `q.prior_kl(D)` for
the collapsed one, whose KL genuinely depends on the data.

Three breaking changes follow:

- **`posterior=` → `model=`**: the constructor keyword and the field are
  renamed on every family (`q.posterior.likelihood` becomes
  `q.model.likelihood`).
- **`jitter=` removed**: conditioning stabilises `K_zz` with the model's
  `Prior.jitter`. If you passed a non-default family jitter, set it on the
  Prior instead: `gpx.gps.Prior(..., jitter=1e-8)`. The defaults agree
  (both were `1e-6`), so most code sees identical numbers.
- **`prior_kl` is part of the family contract**: custom subclasses of
  `AbstractVariationalFamily` must now implement `predict`, `prior_kl` and
  `condition`.

### OILMM joins the conditioning contract

`OILMMModel.condition_on_observations(D)` becomes `model.condition(D)` (or
`model | D`), and `OILMMPosterior.predict(x, return_full_cov=...)` becomes the
`covariance=` keyword every other process uses. Both old spellings still work
and emit a `DeprecationWarning`.

```python
import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
from gpjax.models import create_oilmm

X = jnp.linspace(0.0, 5.0, 20).reshape(-1, 1)
y = jnp.hstack([jnp.sin(X), jnp.cos(X), jnp.sin(2.0 * X)])
D = gpx.Dataset(X=X, y=y)
model = create_oilmm(
    num_outputs=3, num_latent_gps=2, kernel=gpx.kernels.RBF(), key=jr.key(0)
)

posterior = model | D                              # was: condition_on_observations(D)
predictive = posterior(X[:5], covariance="diagonal")  # was: return_full_cov=False
evidence = posterior.log_marginal_likelihood       # was: oilmm_mll(model, D)
```

Two behavioural notes. `OILMMPosterior` is now an `equinox.Module` and a
`gpjax.conditioning.Posterior`, so it is a pytree and can be passed through
`jit`, `grad` and `vmap`; its constructor takes `(model, train_data)` rather
than pre-built latent pieces. And because it caches the latent factorisations
at `condition` time rather than rebuilding them inside `predict`, repeated
prediction from one posterior is markedly cheaper — condition once, predict
many times.

`latent_datasets` is gone, and `latent_posteriors` now holds conditioned
`ExactPosterior`s rather than unconditioned `ConjugateModel`s. Each latent
process carries its own projected training set, so reach through it:

```python
latent = posterior.latent_posteriors[0]
projected_y = latent.train_data.y            # was: posterior.latent_datasets[0].y
latent_pred = latent(X[:5])                  # was: predict(x, train_data=...)
```

For this multi-output process `covariance="dense"` returns the joint
`(NP, NP)` covariance across test inputs *and* outputs, flattened
output-major, where single-output processes return `(N, N)`.

`HeteroscedasticVariationalFamily` is the exception: it approximates two
latent processes, so it has no single conditioned process and its
`condition` raises `NotImplementedError`. Use `predict_latents(xtest)`, or
condition its `signal_variational` / `noise_variational` components
individually.

## 0.14.x → 0.15.0

GPJax `0.15` adds the `gpjax.state_space` sub-package (state-space / Markovian
Gaussian processes) and makes **one** breaking change to likelihood prediction.
Everything else is additive.

### Breaking change: likelihood `.predict` return type

`gpjax.likelihoods.Gaussian.predict` and
`gpjax.likelihoods.HeteroscedasticGaussian.predict` now return a
`gpjax.distributions.GaussianDistribution` instead of a
`numpyro.distributions.MultivariateNormal`.

- If you read `mean`, `variance`, or `covariance_matrix`, **no change is
  needed** — these attributes exist on both types.
- If you relied on `MultivariateNormal`-specific attributes (`scale_tril`,
  `precision_matrix`, etc.), update your call site. The covariance is now backed
  by a [Lineax](https://docs.kidger.site/lineax/) operator: `Gaussian.predict`
  keeps a `lineax.DiagonalLinearOperator` scale on its diagonal fast path and
  wraps a `lineax.MatrixLinearOperator` on the dense path;
  `HeteroscedasticGaussian.predict` always wraps a `lineax.MatrixLinearOperator`.

```py
# Before (0.14.x)
dist = likelihood.predict(latent_dist)
tril = dist.scale_tril            # MultivariateNormal attribute

# After (0.15.0)
dist = likelihood.predict(latent_dist)
mean = dist.mean                  # unchanged
cov = dist.covariance_matrix      # unchanged
# need a Cholesky factor? materialise it explicitly:
import jax.numpy as jnp
tril = jnp.linalg.cholesky(dist.covariance_matrix)
```

### New: state-space Gaussian processes

`gpjax.state_space` is a new, opt-in sub-package — importing or upgrading does
not change any existing behaviour. See the
[State-Space GPs example](examples/state_space_gps.py) to get started.

## 0.13.x → 0.14.0

GPJax `0.14` replaces the Flax NNX backend with
[Equinox](https://docs.kidger.site/equinox/) +
[paramax](https://danielward27.github.io/paramax/), and introduces a linear-algebra layer via
[Lineax](https://docs.kidger.site/lineax/). It also removes the custom bijector
stack in favour of
[numpyro constraints](https://num.pyro.ai/en/stable/distributions.html#constraints).

The changes are mostly internal. They surface in three places:

1. How you **define custom modules and custom parameter classes**.
2. How you **read a parameter value** (`param.unwrap()` / `paramax.unwrap(model)`
   instead of `param.value`).
3. How you **freeze parameters** (`paramax.non_trainable(...)` instead of the
   `trainable=` filter argument on `fit`).

If you only use the high-level API (`gpx.Prior`, `gpx.Posterior`, `gpx.fit`,
etc.) most code keeps working once you update the two or three call sites
below.

### Installation

```bash
pip install "gpjax==0.14.0"
# or
uv add "gpjax==0.14.0"
```

New dependencies (pulled in automatically): `equinox>=0.11`, `paramax>=0.0.5`.
Flax is no longer a runtime dependency.

### Breaking changes

#### 1. Backend: flax.nnx.Module → equinox.Module

If you subclassed `nnx.Module` to build a custom model, kernel, mean function,
likelihood, or variational family, change the base class:

```py
# Before (0.13.x)
from flax import nnx

class MyKernel(nnx.Module):
    def __init__(self, lengthscale):
        self.lengthscale = gpx.parameters.PositiveReal(lengthscale)
```

```py
# After (0.14.0)
import equinox as eqx

class MyKernel(eqx.Module):
    lengthscale: gpx.parameters.PositiveReal

    def __init__(self, lengthscale):
        self.lengthscale = gpx.parameters.PositiveReal(lengthscale)
```

Equinox requires **class-level field annotations** for every attribute, and
static configuration fields should be marked with `eqx.field(static=True)`.

#### 2. Parameter classes are now paramax.AbstractUnwrappable

`PositiveReal`, `NonNegativeReal`, `Real`, `SigmoidBounded`, and
`LowerTriangular` all live in `gpjax.parameters` with the same names, but they
now inherit from `paramax.AbstractUnwrappable` and store their value in an
**unconstrained** internal field. The constraining bijection is applied at
read time through `unwrap()`.

```py
# Before (0.13.x) — nnx.Variable-based
length = gpx.parameters.PositiveReal(0.5)
length.value                # -> 0.5
length.value = 1.0          # in-place mutation (nnx)
```

```py
# After (0.14.0) — paramax.AbstractUnwrappable
length = gpx.parameters.PositiveReal(0.5)
length.unwrap()             # -> 0.5  (applies softplus to the stored unconstrained value)

# Unwrap an entire model tree in one call:
import paramax
model_resolved = paramax.unwrap(model)
```

`Parameter` (the old generic base class), `DEFAULT_BIJECTION`, the
`transform(...)` function, and `FillTriangularTransform` have been
**removed**. `numpyro.distributions.biject_to` now handles every
constraint → bijection mapping.

#### 3. LowerTriangular now requires a valid Cholesky factor

Previously `LowerTriangular` accepted any lower-triangular matrix (the
diagonal was unconstrained). It is now parameterised via
`numpyro.distributions.constraints.softplus_lower_cholesky`, so the diagonal
**must be strictly positive**.

- Passing a matrix with zero or negative diagonal entries produces `NaN`
  during construction (`inv_softplus` of a non-positive number).
- In-library usage is unaffected: the only consumer is
  `VariationalGaussian.variational_root_covariance`, which is initialised to
  the identity by default.
- If you previously supplied a custom `variational_root_covariance`, ensure
  its diagonal is strictly positive. Under the old parameterisation, zero or
  negative diagonals produced singular or sign-ambiguous variational
  covariances.

#### 4. `gpx.fit` / `fit_scipy` / `fit_lbfgs`: removed `params_bijection` and `trainable`

Bijection handling is now automatic via `paramax.unwrap` inside the loss
function, and freezing parameters is expressed by wrapping them in
`paramax.non_trainable`:

```py
# Before (0.13.x)
opt_model, history = gpx.fit(
    model=posterior,
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=D,
    optim=ox.adam(1e-2),
    params_bijection=gpx.parameters.DEFAULT_BIJECTION,
    trainable=gpx.parameters.Parameter,   # filter-based trainability
)
```

```py
# After (0.14.0)
import paramax

# Freeze specific parameters up-front by wrapping them:
posterior = eqx.tree_at(
    lambda m: m.prior.kernel.lengthscale,
    posterior,
    replace_fn=paramax.non_trainable,
)

opt_model, history = gpx.fit(
    model=posterior,
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=D,
    optim=ox.adam(1e-2),
)
```

Internally, `fit` now splits the model with `eqx.partition(model, eqx.is_array)`
so only concrete JAX arrays participate in the gradient update; everything
wrapped in `paramax.non_trainable` is held constant.

#### 5. register_parameters removed

The `gpx.parameters.register_parameters` decorator (added in 0.13.x to mark
NNX variables as GPJax parameters) is gone. With Equinox, GPJax identifies
parameter classes through `isinstance` checks on `AbstractUnwrappable`, so
registration is unnecessary.

#### 6. gpjax.linalg rewrite: cola → Lineax

Kernel `gram()` now returns a `lineax.AbstractLinearOperator`
(typically `lineax.MatrixLinearOperator`) instead of a `cola.LinearOperator`.
Materialise with `.as_matrix()`.

The following names have been **removed** from `gpjax.linalg`:

| Removed                     | Replacement                                        |
| --------------------------- | -------------------------------------------------- |
| `PSD`, `psd`                | Not needed — Lineax operators carry tags directly. |
| `Dense`, `Diagonal`, `Identity`, `Triangular` | `lineax.MatrixLinearOperator`, `lineax.DiagonalLinearOperator`, `lineax.IdentityLinearOperator`, `lineax.TriangularLinearOperator` |
| `LinearOperator`            | `lineax.AbstractLinearOperator`                    |
| `diag`, `solve`             | `operator.diagonal()`, `lineax.linear_solve(...)`  |
| `lower_cholesky`            | `gpjax.linalg.cholesky_factor` (singledispatch, returns a lower-triangular operator) |

`BlockDiag`, `Kronecker`, and `logdet` are unchanged. Use
`gpjax.linalg.add_jitter` to add a jitter term to a covariance operator.

#### 7. Custom bijectors replaced with numpyro constraints

If you had a custom `Parameter` subclass that declared a bijection, replace
the bijection with a numpyro constraint and use `biject_to`:

```py
# Before (0.13.x)
class MyParam(gpx.parameters.Parameter):
    # Custom bijection registered via DEFAULT_BIJECTION
    ...
```

```py
# After (0.14.0)
from numpyro.distributions import biject_to, constraints
from paramax import AbstractUnwrappable
import jax

class MyParam(AbstractUnwrappable):
    _constraint = constraints.positive
    _unconstrained: jax.Array

    def __init__(self, value):
        self._unconstrained = biject_to(self._constraint).inv(value)

    def unwrap(self):
        return biject_to(self._constraint)(self._unconstrained)
```

### Non-breaking cleanup

- `__description__` changed from `"Gaussian processes in JAX and Flax"` to
  `"Gaussian processes in JAX"`, since Flax is no longer a dependency.
- Many kernel `compute_engine` internals moved; the public `kernel(x, y)`,
  `kernel.gram(x)`, `kernel.cross_covariance(x, y)`, and `kernel.diagonal(x)`
  methods are unchanged.

### Upgrade checklist

- [ ] Replace `nnx.Module` base classes with `eqx.Module`, and add class-level
      type annotations for every field.
- [ ] Replace `param.value` reads with `param.unwrap()`, or call
      `paramax.unwrap(model)` once at the top of your loss / prediction
      function.
- [ ] Drop any `params_bijection=` / `trainable=` arguments passed to
      `gpx.fit`. To freeze parameters, wrap them with `paramax.non_trainable`
      using `eqx.tree_at`.
- [ ] Remove any `gpx.parameters.register_parameters` decorator calls.
- [ ] If you construct `LowerTriangular` from a custom matrix, verify the
      diagonal is strictly positive.
- [ ] If you used `gpjax.linalg` operators directly, switch to the Lineax
      equivalents listed above.

### Reporting issues

Please file migration issues at
[the issue tracker](gh:issues) with the `0.14-migration`
label.
