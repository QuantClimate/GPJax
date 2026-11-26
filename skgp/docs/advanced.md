# Advanced use

skgp fits its models with [GPJax](https://github.com/JaxGaussianProcesses/GPJax), a library built on JAX. You never need to touch it for ordinary regression. Most of this page is for the cases skgp does not cover, such as fitting a model GPJax offers and skgp does not. Its last section is about sending skgp's log records to Python's `logging`.

You do not need it for the underlying curve without measurement noise. `fitted.predict(new, include_noise=False)` gives that directly, in your own units: the same mean as a prediction for a new observation, with a spread that leaves out the noise.

## The attribute

A fitted model has one attribute, `gpjax_model`. It returns the fitted GPJax model that skgp itself predicts with.

```python
fitted = model.fit(train)
gpjax_model = fitted.gpjax_model
```

A `FittedGPRegressor` and a `FittedGPClassifier` have it. A `GPRegressor`, `SparseGPRegressor`, `GPClassifier` or `SparseGPClassifier` is only a description and has no GPJax model until it is fitted.

## What you get

What the attribute returns depends on which class made the fitted model.

For a `GPRegressor` it returns a GPJax conjugate model: a Gaussian process prior with a Gaussian noise likelihood, with the hyperparameters skgp learned. With known noise (`noise_column`) the likelihood is skgp's own subclass of GPJax's `Gaussian`, which holds the variance of each training row's known noise, rescaled as the target was, and adds it to the learned noise at the training rows. GPJax's own prediction with noise then adds only the learned noise at new rows; skgp's `predict` adds each new row's known noise too. It holds no data. To predict with it you condition it on a GPJax `Dataset` that you build yourself from the data it was fitted on.

For a `SparseGPRegressor` it returns a GPJax variational family, the approximation skgp fitted. It holds the inducing points, and the same kind of conjugate model sits inside it at `fitted.gpjax_model.model`, so the noise likelihood is `fitted.gpjax_model.model.likelihood`. Conditioning works the same way: `fitted.gpjax_model.condition(training)(x_new)`. What differs is the data. A fit on all rows (no `minibatch_size`) needs the `Dataset`, built as above. A fit on minibatches keeps none of the training data, and its family already carries the fitted distribution over the inducing points, so it does not use the `Dataset` you pass. Either way the inputs are rescaled as described below, and the sparse model is not the exact one: it gives close but not identical answers to a `GPRegressor` on the same data.

For a `GPClassifier` it returns a GPJax variational family with GPJax's `Bernoulli` likelihood at `fitted.gpjax_model.model.likelihood`. Its inducing points are the rescaled training inputs, held fixed, and it carries the fitted distribution over the latent values there, so it needs no data: `fitted.gpjax_model.condition(None)(x_new)` gives the latent curve, and the likelihood turns that into the probability of the class coded 1 (`fitted.classes[1]`). The target was not rescaled; its two values were coded 0 and 1. For a `SparseGPClassifier` it is the same kind of family, but its inducing points are the fitted ones, fewer than the rows.

For a regressor with `learned_noise="heteroscedastic"` it returns a GPJax `HeteroscedasticVariationalFamily`. Its model, `fitted.gpjax_model.model`, is a `HeteroscedasticModel` with two Gaussian processes: the curve, at `.prior`, and the noise process, the logarithm of the noise variance, at `.noise_prior`, with a constant mean and GPJax's `Matern52` kernel. Its likelihood is GPJax's `HeteroscedasticGaussian` with the log-normal transform. GPJax conditions this family on nothing as a whole. Instead, `fitted.gpjax_model.predict(x_new)` gives the mean and variance of both processes at each row, and `fitted.gpjax_model.signal_variational.condition(None)(x_new)` gives the curve alone. skgp fitted each process's distribution over its inducing points in GPJax's whitened form, as it does for a classifier. A `GPRegressor` has an inducing point at every training row for each process, held there. A `SparseGPRegressor` has `num_inducing` for each, fitted.

The model works in the units skgp fitted it in, not your original ones. Before fitting, skgp shifts every input column and the target to mean 0 and scales each to standard deviation 1 (the population standard deviation, `ddof=0`; a column that never varies is only shifted). Your kernel still sees your units: the model's kernel, `prior.kernel` (at `fitted.gpjax_model.model.prior.kernel` for a variational family), is skgp's own kernel that turns the rescaled inputs back before it applies yours, which sits at its `.kernel`. The prior's mean function, `prior.mean_function`, is skgp's own too, and works on the rescaled inputs and target. For a GPJax mean function you gave, it turns the rescaled inputs back before it applies yours, which sits at its `.mean_function`, and rescales what yours gives as the target was. Build the `Dataset` and the new inputs the same way, using the means and standard deviations of the data the model was fitted on. To return to your units, multiply a predicted mean by the target's standard deviation and add the target's mean; multiply a predicted variance by the target's standard deviation squared.

## Use the 64-bit mode

skgp runs all its JAX work in 64-bit floating point, which Gaussian process fits need to stay stable. JAX defaults to 32-bit, so wrap your own calls in the same scope. This is the same conditioning skgp does for `predict`, written out by hand:

```python
import gpjax as gpx
import jax
import jax.numpy as jnp

with jax.enable_x64(True):
    training = gpx.Dataset(
        X=jnp.asarray(x_train),  # shape (n, number of features)
        y=jnp.asarray(y_train),  # shape (n, 1)
    )
    posterior = fitted.gpjax_model.condition(training)
    # The curve, without measurement noise: what predict(..., include_noise=False) gives.
    latent = posterior(jnp.asarray(x_new))
    # A new observation, noise included: what predict gives by default.
    observation = fitted.gpjax_model.likelihood(latent)
    mean = observation.mean
```

For a model fitted by a `SparseGPRegressor`, the one line that changes is where the likelihood lives:

```python
    observation = fitted.gpjax_model.model.likelihood(latent)
```

Keep everything that touches JAX inside the `with` block. The setting is thread-local and only covers operations run inside it, so an operation outside the block runs in 32-bit even on 64-bit arrays.

## Things to know

- This is GPJax's own API, and it can change when GPJax does. skgp pins a minimum GPJax version, not a maximum.
- skgp does not know what you do with the returned model. Anything you build from it is separate from the fitted model, and `fitted.predict` keeps giving the same answers.

## Sending the log records to `logging`

With `verbose=True`, skgp writes its records through [loguru](https://loguru.readthedocs.io/), which prints them to standard error unless you tell it otherwise. If the rest of your program uses Python's own `logging`, pass the records on to it once, before you fit:

```python
import logging

from loguru import logger


class ToLogging(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


logger.remove()  # stop loguru's own printing, so each record appears once
logger.add(ToLogging(), format="{message}")
```

Each record then arrives under the name of the skgp module that wrote it, such as `skgp._regressor`, so a handler or a level set on the `skgp` logger covers them all. `logger.remove()` takes away every loguru output, your own included, so leave that line out if you use loguru elsewhere.
