# ADR-0002: A random key in objectives that use Monte Carlo

- **Status:** accepted
- **Date:** 2026-11-24
- **Deciders:** Thomas Pinder (design settled in a grilling session)

## Context

The `Categorical` likelihood, p(y = c | f) = softmax(f)_c, needs the expected
log likelihood E_q[log softmax(f)_y] in `elbo`. This is a C-dimensional
Gaussian integral. Gauss–Hermite quadrature on a full grid needs J^C points,
so it is not practical for more than approximately four classes. GPflow
(`Softmax`, 100 samples) and GPyTorch (`SoftmaxLikelihood`, 10 samples) use
Monte Carlo with new samples from a global random number generator at each
loss evaluation. JAX has no global random number generator. Until now, every
objective had the signature `(model, Dataset) -> ScalarFloat`, and no random
key reached an objective or a likelihood.

## Decision

The user supplies the random key. `elbo` gets an optional keyword argument,
`elbo(q, data, *, key=None)`. `elbo` passes the key to
`likelihood.expected_log_likelihood(..., key=key)`, and the likelihood passes
it to its integrator. A new generic `MonteCarloIntegrator(num_samples=100)`
draws the samples. It is the default integrator of `Categorical`. A likelihood
with a `MonteCarloIntegrator` raises `ValueError` when it gets `key=None`;
there is no hidden default key. Deterministic likelihoods ignore the key, so
existing calls do not change. `fit`, `fit_scipy` and `fit_lbfgs` do not
change: the user closes over one key for each run,
`objective = lambda q, d: -elbo(q, d, key=elbo_key)`. The Monte Carlo samples
are then the same at each step of the run, so the loss is deterministic and
L-BFGS works.

## Key splitting

- The integrator draws all standard-normal samples for one evaluation in one
  call of shape (S, N, C) from the key it gets. Thus the samples are
  independent across samples, data points and classes, and no key is used
  for two draws.
- A likelihood that needs more than one draw splits its key first. The GEV
  return-level methods already do this (`key_f, key_g`).
- `elbo` passes the key on once and does not use it for anything else.
- The user splits the objective key from every other key in the run, for
  example the key that `fit` uses for mini-batches:
  `key, fit_key, elbo_key = jr.split(key, 3)`. Each new run uses a new split
  key.

## Considered options

- **A fixed integer seed on the integrator.** It does not change the objective
  signature, but the user cannot control the samples, and a seed that never
  changes hides the Monte Carlo error.
- **The closed-form Jensen bound** E[logsumexp f] ≤ log Σ_c exp(μ_c + σ²_c/2).
  It is deterministic, but it is loose when the variances are large.
- **Symmetric cubature (2C + 1 points, as in BayesNewton).** It is
  deterministic, but less accurate than Monte Carlo with many samples.
- **New keys at each step from `fit`.** `fit` would need a way to know which
  objectives take a key. This is left for a later change.

## Consequences

The glossary definition of an objective now has an optional `key`. With a
fixed key and mini-batches, the noise at batch position i is the same at each
step, although the data point at that position changes. This is acceptable
for the first version.
