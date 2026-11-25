# GPJax domain glossary

The ubiquitous language of this codebase. Code, docs, tests, and reviews use
these terms exactly; when a concept is missing, add it here in the same PR
that introduces it.

## Design principle

**Maths-first, comprehensible to the maths-familiar non-expert.** API names
follow the textbook equation unless the textbook word is jargon a
practitioner would not own. When the two conflict, choose the word a
scikit-learn/PyMC user already speaks and state the maths once in the
docstring (e.g. the class is `JointModel`, and its docstring says "the joint
distribution p(f, y)").

## Terms

**Prior** — the Gaussian process prior p(f), pairing a kernel with a mean
function. Queryable at any inputs: `prior(x)` returns the prior predictive.
Owns the model's single numerical-stabilisation knob, `jitter`.

**Likelihood** — the conditional distribution p(y | f). A *pure* conditional:
it holds no priors, no dataset facts, and no data sizes. (Both were tried and
removed at v1.0 — see ADR-0001.)

**Categorical likelihood** — the multi-class classification likelihood
p(y = c | f) = softmax(f)_c, with one latent function f_c per class and C
classes. A label is the class index c ∈ {0, …, C − 1}. The number of classes
C is the one structural fact it holds; it holds no other state.

**JointModel** — the joint distribution p(f, y) = p(y | f) · p(f), created by
`prior * likelihood`. The *trainable* object: `gpx.fit` optimises its
hyperparameters. It carries state that must exist before conditioning
(hyperparameters; the non-conjugate latent; the heteroscedastic noise-process
model) and no derived quantities. Concrete kinds: `ConjugateModel`,
`NonConjugateModel`, `HeteroscedasticModel` — each holds state the others
cannot.

**condition** — the operation p(f, y) + 𝒟 → p(f | 𝒟), spelled
`model.condition(D)` or the operator form `model | D` (read: "f given D").

The signature is `condition(train_data)` uniformly, on every conditionable
object, with `train_data` required. It is universal across the exact, latent,
sparse, collapsed and state-space modes. Where the maths does not consume the
data — the uncollapsed variational families, which already carry the fitted
q(u) — the argument is still accepted, for interface uniformity, and the
docstring says so plainly. A signature that varied by object would be a
worse API than one argument occasionally ignored.

The one documented exclusion is the heteroscedastic path
(`HeteroscedasticModel` and `HeteroscedasticVariationalFamily`), which has no
closed-form conditioned process: it carries two latent processes, signal and
noise, so there is no single p(f | 𝒟) to return. Both raise
`NotImplementedError` naming the alternative — inference runs through
`HeteroscedasticVariationalFamily` and the `heteroscedastic_elbo` objective,
and prediction through `predict` / `predict_latents`, or by conditioning the
`signal_variational` and `noise_variational` components individually.

`prior_kl` is deliberately *not* part of this contract: it keeps a per-family
signature, because only the collapsed family's KL is a function of the data.

**Posterior** — the conditioned process p(f | 𝒟) returned by `condition`. An
*immutable* pytree: the training-covariance factorisation is computed once and
cached; every query is a view of it. The uniform query surface:
`posterior(xtest, covariance="dense"|"diagonal")`, plus per-mode views —
`log_marginal_likelihood` / `loo` / `sample_approx` on the exact mode,
`log_posterior_density` on the latent mode. Users never name the concrete
implementations behind the interface.

**variational family** — a trainable approximate posterior over inducing
values: to sparse GPs what JointModel is to exact ones. It carries the joint
model in its `model` field, and `.condition(D)` yields a Posterior like any
other; `elbo`-style objectives are its training criteria. The model's
`Prior.jitter` is the only stabilisation knob — families carry none of their
own.

**evidence / log marginal likelihood** — p(𝒟), the normalising constant of
conditioning, exposed as `posterior.log_marginal_likelihood`. "Evidence" and
"marginal likelihood" are the same quantity; the attribute uses the
GP-community's term.

**sugar** — a documented one-line composition kept for ergonomics, never a
second implementation. `model.predict(x, D)` and `model(x, D)` are sugar for
`model.condition(D)(x)`; `model | D` is sugar for `model.condition(D)`.

**objective** — a scalar function `(model, Dataset) -> ScalarFloat` consumed
by `gpx.fit`. An objective that uses Monte Carlo also takes an optional
keyword `key`; the user supplies it (see ADR-0002). Objectives are thin:
`conjugate_mll` is the evidence view of the conditioned posterior, not a
second derivation.

**Dataset** — the data container. `n_total` records the full-dataset size
when the object is a minibatch view (stamped by `get_batch`); the minibatch
ELBO scale is derived from it, never supplied by hand. Read it through
`full_size`, which falls back to `n` for a whole dataset — production code
uses `data.full_size / data.n` and never re-spells the fallback inline.

**GridSpec** — the record of how a labelled xarray grid was flattened into a
**Dataset** by `gpjax.xarray.from_xarray`: which inputs fill which columns, the
grid's dims and coordinates, which cells were dropped for NaN, and the time
origin of each datetime input. It lives next to the model, never inside it:
`inputs_for` builds prediction inputs on a new grid, and `to_xarray` maps a
predictive distribution back onto the labelled grid. The **Dataset** itself
carries no grid metadata.
