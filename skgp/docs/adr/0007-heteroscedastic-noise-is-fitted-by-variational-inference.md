# Heteroscedastic noise is fitted by variational inference, with an inducing point at every row in a GPRegressor

A regressor's `learned_noise` setting (called `noise` before known noise was added) is `"constant"` (the default) or `"heteroscedastic"`, on `GPRegressor` and `SparseGPRegressor` alike, not a class of its own. With heteroscedastic noise, the logarithm of the noise variance is a second Gaussian process, the noise process, with a `"matern52"` kernel, the noise kernel, and a learned constant mean, and the noise variance is its exponential (GPJax's `HeteroscedasticGaussian` likelihood with the log-normal transform). GPJax 1.1 has no exact fit of this model, only its `HeteroscedasticVariationalFamily` and `heteroscedastic_elbo`. So a `GPRegressor` with heteroscedastic noise is fitted as a `GPClassifier` is ([ADR 0006](0006-the-classifier-uses-variational-inference-at-every-row.md)): by L-BFGS-B, with an inducing point at every training row for the curve and for the noise process, held there. A `SparseGPRegressor` gives each of the two `num_inducing` inducing points, which move. We chose this because the noise is a choice about the model, not about its size, so it belongs with the kernel as a setting of both classes; and because a `GPRegressor` then still summarises nothing, as with constant noise, while a `SparseGPRegressor` stays the one that does.

Each process's distribution over its inducing points is fitted in GPJax's whitened form, in place of the form GPJax's family uses. On 200 rows with noise growing from 0.1 to 3.1, the fit at every row in GPJax's own form stopped at a flat curve with all the variation as noise. In the whitened form it learned the curve and noise of 0.43, 1.77 and 2.71 where the true noise was 0.4, 1.6 and 2.8, the same as the sparse fit.

## Considered Options

- **A class of its own**, such as `HeteroscedasticGPRegressor`. Rejected: it would double the regression classes for one setting.
- **Noise given in a column**, a known measurement error per row. Not chosen: the request was noise learned from the data, and a column says nothing about the noise at new rows.
- **Softplus in place of the exponential**, GPJax's default transform. Not chosen: it needs quadrature, where the log-normal gives the fit's expected likelihood in closed form, and with the exponential a change in the target's units moves only the noise process's mean.
- **The curve's kernel for the noise too**, a copy learned on its own. Not chosen: a kernel chosen for the curve, such as a periodic one, need not suit the noise; a `"matern52"` kernel with one length scale per input is the default for the curve too.

## Consequences

- A `GPRegressor` with heteroscedastic noise fits about N² numbers for N rows, twice a classifier's, so it gets slow as rows grow: about 15 seconds at 200 rows with one input on a CPU. It warns above `skgp.config.size_warning_rows`, naming `SparseGPRegressor`.
- The fit on minibatches uses Adam alone, because GPJax's natural-gradient step does not support the heteroscedastic family. On 200 rows in minibatches of 50 it settled where the noise was 0.4 at 0.6 to 0.75, looser than a fit on all rows.
- `learned.noise` is the noise far from the training data, the exponential of half the learned mean, and `learned.noise_kernel` gives the noise kernel's values; the variance of the noise kernel has no units.
- `gpjax_model` is a `HeteroscedasticVariationalFamily`, which GPJax does not condition as a whole; `docs/advanced.md` says how to use it.
