# Changelog

All notable changes to GPJax are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Changed

- Allow JAX and JAXlib 0.11 in downstream environments by removing the
  `<0.11` dependency bounds ([#801](https://github.com/QuantClimate/GPJax/issues/801)).

### Fixed

- Read the observation count from `y` for unsupervised `Dataset` objects, so
  `n` and `full_size` work when `X` is absent.

## [1.0.0] — 2026-09-28

### Added

- **`gpjax.xarray`: labelled gridded data in and out
  ([#782](https://github.com/QuantClimate/GPJax/issues/782)).**
  `from_xarray(ds, target=..., inputs=[...])` flattens an `xr.Dataset` into an
  ordinary `Dataset`. Inputs can be coordinates or other data variables; datetime
  inputs become days since the first timestamp, and NaN cells are dropped. It
  also returns a `GridSpec`, which builds inputs for a new grid
  (`spec.inputs_for(grid)`) and maps a predictive distribution back onto the
  labelled grid (`spec.to_xarray(dist)`). That gives the mean and variance by
  default, or joint posterior samples with a `sample` dimension when called with
  `num_samples=`. `Dataset` is unchanged, and `import gpjax` does not import
  xarray. Install with `pip install "gpjax[xarray]"`. A new example, *Gridded
  Data with xarray*, walks through the workflow.

- **`gpjax.parameters.val`.** Promoted from the private `_val` helper to documented
  public API, and now the single accessor for a parameter's constrained value:
  `val(kernel.lengthscale)` applies the constraining bijection, while plain arrays and
  floats pass through unchanged, so it needs no `isinstance` guard and is safe to
  apply anywhere. It resolves nested wrappers innermost-first, so a parameter frozen
  with `paramax.non_trainable` is read exactly like any other. This is the rule to
  follow when writing a kernel, mean function, likelihood or objective — see the
  "How does the parameter system work?" section of the Sharp Bits guide
  (`docs/sharp_bits.md`).

- **Dense joint predictive covariance for state-space GPs.**
  `StateSpacePrior.predict`, `StateSpaceConjugateModel.predict`, and
  `StateSpacePosterior.__call__` (`gpjax.state_space`) now accept
  `covariance="dense"`, returning the full joint covariance across test
  points rather than marginal variances only
  ([#651](https://github.com/QuantClimate/GPJax/issues/651)). For the
  unconditioned prior this is just the kernel's own dense gram — the
  state-space SDE is an exact representation with no training data to
  marginalise out. For the conditioned (smoothed) posterior it is built from
  the RTS smoother's cross-covariance recursion (Särkkä & Solin 2019 §12.2):
  `rts_smoother` gains an opt-in `return_gains=True` that exposes its
  already-computed per-step smoother gains, and a new
  `gpjax.state_space.prediction._dense_smoothed_test_covariance` chains them
  into the `M x M` test-point covariance. This keeps the state-space
  formulation's linear-in-`N` cost — no `N x N` gram over the training set is
  ever formed — with the cross-covariance work landing at `O(M^2 d^3)`,
  independent of `N` and no larger than the `O(M^2)` already required to
  store the dense output. `StateSpacePosterior.filtered` /
  `StateSpaceConjugateModel.predict_filter` (the *causal* predictive) keep
  raising `NotImplementedError` for `covariance="dense"`: each of their test
  points conditions on a different information set (observations up to its
  own timestamp), so a "joint" filtered covariance is not the dense
  conjugate-predictive-shaped object the smoothed path now matches, and
  extending it is left to a future issue if there is demand for it.

- **`gpjax.fit_natgrads` and the `gpjax.natural_gradients` module.** Trains a variational
  family by alternating one natural-gradient step on the variational distribution with
  one step of a supplied Optax optimiser on everything else — kernel and likelihood
  hyperparameters, the mean function, and the inducing inputs — following Salimbeni,
  Eleftheriadis and Hensman (2018), [arXiv:1803.09151](https://arxiv.org/abs/1803.09151).
  `VariationalGaussian` and `WhitenedVariationalGaussian` are supported. For a conjugate
  model on the full batch, `natgrad_lr=1.0` reaches the optimal `q` in one iteration.

- **`DualVariationalGaussian` and `gpjax.objectives.dual_elbo`.** The dual (t-SVGP)
  parameterisation of Adam, Chang, Khan and Solin (2021),
  [arXiv:2111.03412](https://arxiv.org/abs/2111.03412). Instead of the moments of
  `q(u)`, the family stores an unnormalised Gaussian *site* on the centred inducing
  outputs — `dual_vector` is the site's first natural parameter and `dual_matrix` its
  precision — from which `q(u)` is recovered through the working matrix
  `R = Kzz + Kzz Lambda_2 Kzz`. Because the stored coordinates are an affine image of
  the natural parameters, a natural-gradient step is a convex combination of the
  current sites with a closed-form target, so no expectation-to-natural round trip is
  needed and the KL is never differentiated. `fit_natgrads` dispatches on the family
  and takes that step; the step size means the same thing in both branches, and from
  the same starting `q` the dual and Salimbeni E-steps produce identical iterates —
  provided the dual branch's computed per-point curvature stays non-negative, so that
  its `beta_floor` never engages. That holds for a genuinely log-concave likelihood;
  GPJax's `inv_probit` clips its probabilities away from 0 and 1, which breaks
  log-concavity in the far tails, and there the two branches diverge. The
  dual branch restricts the step size to the interval from zero to one, since the
  update is a convex combination.
  `dual_elbo` has the same *value* as `elbo` at the implied moments, for any sites and
  any hyperparameters, but a different *hyperparameter gradient from `elbo` evaluated on
  the matched `VariationalGaussian`*: the prior part of `q` tracks the kernel while the
  data-dependent sites stay frozen. The difference is between the two
  parameterisations, not between the two functions — calling `elbo` directly on a
  `DualVariationalGaussian` returns the same value and the same gradients as
  `dual_elbo`, which is simply the batched-marginals fast path for that family. That
  frozen-site gradient is what gives the M-step its reported behaviour, so `Kzz` must
  not be detached and the implied moments must not be cached on the family.
  `DualVariationalGaussian` also works with plain `gpjax.fit`, where it is ordinary
  gradient descent in the dual coordinates. The `VariationalParametrisationSuite` ASV
  benchmark gains a `dual` axis value.

### Changed (breaking)

- **Models are no longer unwrapped before objectives, losses and predictions run.**
  `fit`, `fit_scipy`, `fit_lbfgs` and `fit_natgrads` no longer call
  `paramax.unwrap(model)` inside their loss functions, and the `gpjax.state_space`
  objectives and prediction entry points no longer unwrap on the way in. A model is
  now *always* in its wrapped form — the model you build, and the model a fitter
  hands back — and the constraining bijection is applied at the point of use by
  `gpjax.parameters.val`. Previously the same function could receive either form
  depending on how it was reached, which made "is this model wrapped?" an
  undocumented precondition on every user-facing extension point: an objective called
  directly received a wrapped model, while the identical function called through
  `fit` received an unwrapped one.

  Ordinary use is unaffected. `gpx.fit(...)` with built-in kernels, and calls such as
  `conjugate_mll(posterior, D)` or `posterior(X, train_data=D)`, behave exactly as
  before — `val` is idempotent, so GPJax's own functions still accept a model in
  either form. What changes is code *inside* the call path: a third-party kernel,
  mean function, likelihood or objective that reads a parameter directly
  (`model.prior.kernel.lengthscale` rather than
  `val(model.prior.kernel.lengthscale)`) now raises `TypeError: unsupported operand
  type(s) ... 'PositiveReal'` when reached through a fitter. Such code was already
  inconsistent rather than correct: because GPJax's own kernels have always resolved
  parameters at the point of use, a bare read failed on a direct `kernel.gram(x)`
  call before this change too, and worked only when something upstream happened to
  unwrap first. The fix is to wrap each parameter read in `val`, which returns the
  constrained value of a parameter and returns plain arrays and floats unchanged.
  `paramax.unwrap` is untouched and still resolves a whole model tree.

  Gradients are unchanged. The bijection sits on the differentiated path either way,
  so the chain rule accounts for it identically whether it is applied once or at each
  of several use sites, and parameters frozen with `paramax.non_trainable` still
  receive exactly zero gradient — `val` resolves nested wrappers innermost-first, so
  `lax.stop_gradient` lands on every path back to the underlying array rather than on
  one. Under `jit` there is no runtime cost: the bijection is loop-invariant under
  `vmap` and repeated reads of the same parameter are common-subexpression-eliminated,
  so it is computed once per loss evaluation however many call sites read it.

- **`OILMMPosterior.__call__`/`.predict` default to `covariance="diagonal"`**
  (was `"dense"`) ([#682](https://github.com/JaxGaussianProcesses/GPJax/issues/682)).
  The dense joint covariance costs `O(m n^2 p^2)` — an `np x np` matrix built
  from `m` dense `n x n` latent covariances — which forfeits the
  `O(mn^3 + nmp)` scaling that OILMM exists to provide, and is unaffordable
  well before the mean/marginal-variance query is. Marginal variances are the
  common case and are unaffected by the mixing matrix's off-diagonal
  structure, so the cheap path is now the default; pass
  `covariance="dense"` explicitly for the joint covariance.

### Removed

- **`NaturalVariationalGaussian` and `ExpectationVariationalGaussian`.** These were
  parameterisation-only classes with no optimiser attached: they stored the natural or
  expectation coordinates of `q(u)` but offered no way to take a natural-gradient step
  in them. Natural-gradient geometry belongs to the optimiser — the Fisher matrix *is*
  the Jacobian dη/dθ, so a natural-gradient step in the natural parameters θ is exactly
  an ordinary gradient step in the expectation parameters η, and either coordinate system
  can be recovered on the fly from whatever the family happens to store. `fit_natgrads`
  therefore operates directly on `VariationalGaussian` and `WhitenedVariationalGaussian`,
  which store constraint-respecting coordinates. Users of the removed classes should
  switch to `VariationalGaussian` with `gpjax.fit_natgrads`.
  The `VariationalParametrisationSuite` ASV benchmark loses its `natural` and
  `expectation` axis values; previously recorded results for those two arms are orphaned.

### Fixed

- **`gpx.kernels.RBF()` was a type error, and `White()` carried a phantom
  trainable lengthscale**
  ([#695](https://github.com/JaxGaussianProcesses/GPJax/issues/695)). Pyright
  synthesises `__init__` signatures from dataclass fields for any kernel that
  inherits its `__init__` (e.g. `RBF`, `Matern12/32/52`), and those fields had
  no defaults, so the canonical `RBF()` call was flagged as missing arguments
  while the nonsensical `RBF(name="xyz")` type-checked cleanly (raising
  `TypeError` at runtime). Kernel fields now carry real defaults matching
  their `__init__` defaults, and `name` is a `ClassVar` rather than a
  dataclass field, so the synthesised and hand-written signatures agree.
  Separately, `White` hardcoded `lengthscale=1.0` into
  `StationaryKernel.__init__` even though `White.__call__` never reads it,
  so every `White` kernel carried a real, trainable `PositiveReal` leaf with
  zero gradient that showed up in optimiser state and MCMC traces; `White`
  now has its own minimal `__init__` and no longer carries a lengthscale at
  all (`White().lengthscale is None`, and it is absent from
  `jax.tree_util.tree_flatten`). The stale `_compute_base_init` workaround in
  `kernels/base.py`, whose docstring claimed "equinox modules are frozen
  after `super().__init__()`" -- no longer true under the pinned Equinox
  version -- was removed in favour of a plain `super().__init__(...)` call.

- **`Zero` mean function is trainable and drifts away from zero.** Fitting a
  model with the default `Zero()` mean function moved its constant towards the
  data mean (0.0 → 5.09 on a dataset with mean 5), silently changing the
  posterior mean of every such model
  ([#712](https://github.com/JaxGaussianProcesses/GPJax/issues/712)). This is a
  regression of [#330](https://github.com/JaxGaussianProcesses/GPJax/issues/330),
  fixed once in #500 and reintroduced by the Equinox migration. The cause is a
  changed trainability contract rather than a lost line: under `nnx`, `fit`
  optimised only `Parameter` instances, so `Zero`'s bare array was inert by
  construction; under Equinox, `fit` partitions on `eqx.is_array`, which makes
  every array leaf trainable. `Zero` now wraps its constant in
  `paramax.non_trainable`, so it stays at zero by construction. `Constant` is
  unchanged and remains trainable.

  `Zero().constant` is now a `NonTrainable` wrapper rather than a bare array.
  Read it with `paramax.unwrap(mean_function).constant` (or `val`) if you were
  accessing it directly; evaluating the mean function is unaffected.

## [0.18.0] — 2026-07-26

### Removed

- **`tensorstore` runtime dependency** (macOS only). It was declared in
  `pyproject.toml` but imported nowhere in the package, forcing a ~14 MB wheel
  onto every macOS install
  ([#675](https://github.com/JaxGaussianProcesses/GPJax/issues/675)). Nothing
  else in the dependency graph requires it, so it was not a resolver workaround.
  A regression test now asserts every declared runtime dependency is actually
  imported.

### Fixed

- **`distributions._kl_divergence`**: now computes both log-determinants from the
  Cholesky factors it has already built, halving the O(N³) work on the hot path
  of every ELBO step. It previously factorised `Σq` and `Σp` for the trace and
  Mahalanobis terms, then called `logdet(Σq)` and `logdet(Σp)`, whose generic
  implementation factorised both covariances a *second* time — four
  factorisations where two suffice
  ([#664](https://github.com/JaxGaussianProcesses/GPJax/issues/664)). A new
  `linalg.logdet_from_factor(L)` helper computes `log|L Lᵀ| = 2 Σᵢ log Lᵢᵢ` and
  is itself `singledispatch`ed, so diagonal, block-diagonal, Kronecker and
  identity operators keep their structure-exploiting fast paths instead of being
  densified. Returned values are unchanged.
- **`prior_kl` for `VariationalGaussian` and `WhitenedVariationalGaussian`**: both
  now evaluate in closed form from the stored triangular root, instead of
  densifying `S = sqrt sqrtᵀ` and handing it to the generic Gaussian KL, which
  re-factorised a matrix whose Cholesky factor was already to hand
  ([#665](https://github.com/JaxGaussianProcesses/GPJax/issues/665)). The
  whitened KL against `N(0, I)` needs no factorisation at all and went from two
  dense Choleskys per call to zero; `VariationalGaussian` went from four to one,
  the unavoidable factorisation of `Kzz`. Values and gradients are unchanged.
  Every variational training step pays this cost, so `grad(prior_kl)` is roughly
  13× faster for the whitened family and 3× faster for `VariationalGaussian` at
  1024 inducing points. `GraphVariationalGaussian` and
  `HeteroscedasticVariationalFamily` inherit the improvement.
- **`HeteroscedasticGaussian.link_function`**: now requires the noise latent `g`
  and returns the conditional `N(y | f, σ²(g))`. It previously evaluated the
  noise transform at `g = 0` and returned the *prior-noise* density
  `N(y | f, σ²(0))` — silently, and independently of the noise process the
  likelihood exists to model
  ([#670](https://github.com/JaxGaussianProcesses/GPJax/issues/670)). Callers
  passing only `f` now get a `ValueError` pointing at the correct API instead of
  a wrong number.
- **`StationaryKernel.spectral_density`**: now returns the correctly
  parameterised spectral measure. It previously returned a *standardised*
  distribution (`Normal(0, 1)` for RBF, `StudentT(2ν, 0, 1)` for Matérn) that
  ignored the lengthscale and was hard-coded to one dimension, so
  `kernel.spectral_density.log_prob(ω)` gave the same curve for every ℓ
  ([#612](https://github.com/JaxGaussianProcesses/GPJax/issues/612)). The
  measure is now `D`-dimensional and carries `diag(ℓ)⁻¹` as its scale (ARD
  lengthscales included), satisfying Bochner's theorem
  `k(τ) = σ²·E_p(ω)[exp(i ωᵀτ)]`.

### Changed

- **`HeteroscedasticGaussian.link_function`** signature is now `(f, g=None)`. The
  `g` argument is mandatory in practice; omitting it raises rather than silently
  substituting `g = 0`. A heteroscedastic subclass that inherits
  `AbstractLikelihood.expected_log_likelihood` (which calls `link_function(f)`)
  now fails loudly instead of integrating against the prior-noise density.
  `HeteroscedasticGaussian` itself overrides that method, so the sanctioned
  variational path is unchanged.

- **`StationaryKernel.spectral_density`** return type is now
  `MultivariateNormal` / `MultivariateStudentT` with `event_shape == (D,)`,
  where it was previously the univariate `Normal` / `StudentT`. Code calling
  `.sample(key, (M, D))` should now call `.sample(key, (M,))`.
- **`kernels.approximations.RFF`** with an explicitly supplied `frequencies=`
  argument now treats those values as the spectral frequencies ω directly.
  They were previously divided by the lengthscale inside
  `BasisFunctionComputation.compute_features`, silently rescaling user-supplied
  frequencies. RFF Gram/cross-covariance values with *sampled* frequencies are
  bit-identical to `0.17.0` — the lengthscale simply moved from the feature map
  into the measure it is drawn from.

#### Migration

- **`StationaryKernel.spectral_density`**: the return type is now
  `MultivariateNormal` / `MultivariateStudentT` with `event_shape == (D,)`.
  Replace `kernel.spectral_density.sample(key, (M, D))` with
  `kernel.spectral_density.sample(key, (M,))`. Values from `log_prob` now vary
  with the lengthscale, as they always should have — there is no restore path,
  the previous output was incorrect.
- **`kernels.approximations.RFF(..., frequencies=...)`**: supplied frequencies
  are now used as the spectral frequencies ω directly. If you were
  pre-compensating for the old division by the lengthscale, remove that
  compensation. RFF with *sampled* frequencies is bit-identical to `0.17.0`.
- **`HeteroscedasticGaussian.link_function`**: now requires the noise latent,
  `link_function(f, g)`. Calls passing only `f` raise `ValueError`; they
  previously returned `N(y | f, σ²(0))`, which was wrong. Use
  `expected_log_likelihood(..., mean_g=, variance_g=)` or
  `predict(dist, noise_dist)` if you want the noise handled for you.
- **`tensorstore`**: removed as a runtime dependency. If you imported it
  transitively via GPJax on macOS, depend on it explicitly.
- **KL divergences**: `distributions._kl_divergence` and the variational
  `prior_kl` methods return the same values as `0.17.0` (bit-identical, or to
  ~1e-16 where floating-point reassociation applies). No action needed.

## [0.17.0] — 2026-07-04

### Fixed

- **`Prior.sample_approx` / `ConjugatePosterior.sample_approx`**: split the PRNG
  key before drawing RFF frequencies, Fourier weights, and (posterior) noise.
  They previously shared one key, making the pathwise-sample cross-covariance
  biased (Wilson et al. 2020 §3 requires independence). Marginals were
  unaffected.
- **`kernels.approximations.RFF`**: multi-dimensional Matérn frequencies are now
  drawn from a `MultivariateStudentT` (shared inverse-χ² per row), matching the
  isotropic Matérn spectral density. Previously iid-per-dim `StudentT` draws
  approximated a *tensor-product* Matérn in `d > 1`. `d == 1` is unchanged.
- **`objectives.conjugate_loocv`**: now routes noise and targets through the
  likelihood protocol, so multi-output (`MultiOutputGaussian`) LOOCV is correct
  (leave-one-scalar-out on the flattened NP system). Also removed a redundant
  `jnp.linalg.inv` (audit #662).
- **`kernels.nonstationary.ArcCosine`**: `weight_variance` and `bias_variance`
  are now wrapped in `NonNegativeReal`. They were previously assigned raw —
  silently frozen (Python-float default, excluded by `eqx.partition`) or
  trainable-but-unconstrained (array value, driveable negative → NaN).
- **`models.create_oilmm_from_data`**: now performs the documented PCA
  eigen-initialisation of the mixing matrix (top-M eigenvectors/eigenvalues of
  the empirical output covariance) instead of returning the random default. Raises
  `ValueError` for `N < 2` (empirical covariance is undefined).

### Changed

- Pathwise samples from `sample_approx` with a fixed `key` now differ from
  `0.16.0` (the old draws were biased — see Fixed above). No restore path; the
  old outputs were incorrect.
- Multi-dimensional Matérn `RFF` Gram matrices now converge to the isotropic
  Matérn. `d > 1` outputs differ from `0.16.0`. No restore path.
- Multi-output `conjugate_loocv` values differ from `0.16.0`. No restore path.
- **`models.OILMMPosterior.predict(..., return_full_cov=False)`** now returns a
  `lineax.DiagonalLinearOperator` scale (was a densified `MatrixLinearOperator`).
  Values are unchanged.
- **`Prior.predict` / `ConjugatePosterior.predict` / `NonConjugatePosterior.predict`
  with `return_covariance_type="diagonal"`** now return a
  `lineax.DiagonalLinearOperator` scale (was a densified `MatrixLinearOperator`).
  Values are unchanged; construction drops from O(M²) to O(M) memory and the
  `predict(...,"diagonal") → likelihood()` chain now uses the O(M) fast path.

#### Migration

- **ArcCosine**: if you relied on the accidental freeze of `weight_variance` /
  `bias_variance`, restore it explicitly with `paramax.non_trainable(...)`.
- **`sample_approx` / multi-dim Matérn RFF / multi-output LOOCV**: no restore
  path — the previous outputs were incorrect.
- **OILMM diagonal predict**: `predict(..., return_full_cov=False).scale` is now a
  `DiagonalLinearOperator`. Use `.as_matrix()` / `.diagonal` rather than assuming a
  dense matrix.
- **Diagonal predict**: `predict(..., "diagonal").scale` is now a
  `DiagonalLinearOperator`. Use `.as_matrix()` / `.diagonal`, not `.matrix`.

## [0.16.0] — 2026-06-28

### Added

- **`gpjax.summarise(model)`**: render any GPJax pytree (kernel, prior,
  posterior, likelihood, variational family, ...) as a flat `rich` table —
  one row per parameter — showing parameter path, class, constrained value,
  bijector, trainability, shape, and dtype. User-facing abstract bases gain
  `__rich__` and `_repr_mimebundle_` hooks for `rich.print(model)` and
  Jupyter auto-rendering; `repr()` is unchanged. `rich` is now a core
  runtime dependency. An optional `priors` mapping (parameter name → prior,
  e.g. a NumPyro distribution) populates the otherwise-empty Prior column.

## [0.15.0] — 2026-06-03

### Added

- **`gpjax.state_space` sub-package**: state-space (Markovian) Gaussian
  processes with a square-root Kalman filter (chunked checkpointed scan
  for `O(sqrt(N) · d²)` reverse-mode AD memory) and an RTS smoother.
  Public API:
  - `StateSpacePrior`, `StateSpaceConjugatePosterior`
  - `TruncatedPeriodic` (Solin & Särkkä 2014 truncated-Fourier kernel)
  - `to_sde` singledispatch dispatching kernels to closed-form SDEs
  - `state_space_mll` MLL objective
  - `fit`, `fit_scipy`, `fit_lbfgs` thin wrappers around the
    corresponding `gpjax.fit_*` optimisers
  - Supports `Matern12`, `Matern32`, `Matern52`, `TruncatedPeriodic`,
    and their `SumKernel` compositions on 1-D temporal data.
- **New poe task**: `all-tests-slow` runs long-running numerical and
  memory stress tests opt-in via the `slow` pytest marker.

### Changed (breaking)

- **`gpjax.likelihoods.Gaussian.predict` and
  `gpjax.likelihoods.HeteroscedasticGaussian.predict`** now return
  `gpjax.distributions.GaussianDistribution` (was
  `numpyro.distributions.MultivariateNormal`). For `Gaussian.predict`, the
  diagonal fast path preserves a `lineax.DiagonalLinearOperator` scale; the
  dense path wraps a `lineax.MatrixLinearOperator`.
  `HeteroscedasticGaussian.predict` wraps its (always-dense) covariance in a
  `lineax.MatrixLinearOperator`. Callers relying on
  `MultivariateNormal`-specific attributes (`scale_tril`,
  `precision_matrix`, etc.) must update; callers using `mean`,
  `variance`, `covariance_matrix` are unaffected.

## [0.14.0] — 2026-04-21

See [`docs/migrations/0.14.0.md`](docs/migrations/0.14.0.md) for full upgrade
instructions.

### Changed (breaking)

- **Module backend**: `flax.nnx.Module` has been replaced with
  `equinox.Module`. Custom kernels, mean functions, likelihoods, and
  variational families that subclassed `nnx.Module` must now subclass
  `eqx.Module` and declare class-level type annotations for every field.
- **Parameter system**: `PositiveReal`, `NonNegativeReal`, `Real`,
  `SigmoidBounded`, and `LowerTriangular` now inherit from
  `paramax.AbstractUnwrappable`. Read a parameter value with
  `param.unwrap()`, or resolve an entire model tree with
  `paramax.unwrap(model)`. `param.value` no longer exists.
- **Fitting API**: `gpx.fit`, `gpx.fit_scipy`, and `gpx.fit_lbfgs` no
  longer accept `params_bijection=` or `trainable=`. Bijection handling
  is automatic; freeze parameters by wrapping them with
  `paramax.non_trainable` via `eqx.tree_at`.
- **`LowerTriangular`** now requires a strictly positive diagonal
  (parameterised by `softplus_lower_cholesky`). Passing a matrix with
  zero or negative diagonal entries produces `NaN` during construction.
- **Linear algebra**: `gpjax.linalg` has been rewritten on top of
  [Lineax](https://docs.kidger.site/lineax/). Kernel `gram()` returns a
  `lineax.AbstractLinearOperator` instead of a `cola.LinearOperator`.
  `PSD`, `psd`, `Dense`, `Diagonal`, `Identity`, `Triangular`,
  `LinearOperator`, `diag`, `solve`, and `lower_cholesky` are removed;
  see the migration guide for Lineax equivalents.
- **Bijectors**: the custom bijector stack has been removed.
  `Parameter` (the old generic base class), `DEFAULT_BIJECTION`,
  `transform(...)`, and `FillTriangularTransform` are gone. Custom
  parameter classes should use `numpyro.distributions.biject_to` with a
  `numpyro.distributions.constraints` object.
- **`register_parameters` removed**. With the Equinox backend, GPJax
  identifies parameter classes through `isinstance` checks on
  `AbstractUnwrappable`, so the decorator is no longer needed.

### Added

- `gpjax.linalg.cholesky_factor` — single-dispatch Cholesky factoriser
  that returns a lower-triangular Lineax operator.
- `gpjax.linalg.add_jitter` — helper for adding a jitter term to a
  covariance operator.
- [`docs/migrations/0.14.0.md`](docs/migrations/0.14.0.md) — upgrade guide from
  0.13.x.

### Removed

- Flax is no longer a runtime dependency. The package `__description__`
  has been updated from "Gaussian processes in JAX and Flax" to
  "Gaussian processes in JAX".
- `cola-ml` is no longer a runtime dependency.

### Dependencies

- Added: `equinox>=0.11`, `paramax>=0.0.5`, `lineax`.
- Removed: `flax`, `cola-ml`.

[1.0.0]: https://github.com/QuantClimate/GPJax/releases/tag/v1.0.0
[0.18.0]: https://github.com/QuantClimate/GPJax/releases/tag/v0.18.0
[0.16.0]: https://github.com/QuantClimate/GPJax/releases/tag/v0.16.0
[0.17.0]: https://github.com/QuantClimate/GPJax/releases/tag/v0.17.0
[0.15.0]: https://github.com/QuantClimate/GPJax/releases/tag/v0.15.0
[0.14.0]: https://github.com/QuantClimate/GPJax/releases/tag/v0.14.0
