# skgp

Fit Gaussian process models from dataframes, without learning JAX.

A Gaussian process is a flexible way to learn a smooth curve from data and to say how uncertain you are about it. The libraries that fit them well ask you to build several pieces by hand. skgp does that setup for you: you describe the model with your column names, call `.fit()` and `.predict()`, and get answers back as dataframes. It is built for QuantClimate's own work first, and for anyone else who finds it useful.

## Status

The code in this repository does what this README says, and the examples below run as part of the test suite. skgp is in early development on the 0.x line, so the API can change between minor versions (0.1 to 0.2, for example).

Install it from PyPI:

```bash
pip install skgp
```

## Quickstart

This example makes up a small table of temperatures, fits a model to it, and predicts for two new places.

```python
import numpy as np
import polars as pl
from skgp import GPRegressor

# Made-up data: temperature falls with latitude and altitude, plus some noise.
rng = np.random.default_rng(0)
latitude = rng.uniform(35, 60, 40)
altitude = rng.uniform(0, 2000, 40)
temperature = 30 - 0.4 * (latitude - 35) - 0.006 * altitude + rng.normal(0, 0.5, 40)
train = pl.DataFrame(
    {"latitude": latitude, "altitude": altitude, "temperature": temperature}
)
new = pl.DataFrame({"latitude": [45.0, 52.0], "altitude": [100.0, 800.0]})

model = GPRegressor(features=["latitude", "altitude"], target="temperature")
fitted = model.fit(train)

prediction = fitted.predict(new)
print(
    prediction.to_frame()
)  # your rows, plus the predicted mean and standard deviation
print(prediction.interval(0.95))  # lower and upper bounds for each row
```

`prediction.to_frame()` gives back every column you passed, followed by `mean` and `std`, and `prediction.to_frame(interval=0.95)` adds `lower` and `upper` too. `prediction.interval()` uses a 95% level if you give none. Pass pandas dataframes instead of Polars ones and you get pandas dataframes back.

If your rows already have a column called `mean`, `std`, `lower` or `upper`, pass `suffix=` to `predict`, for example `fitted.predict(new, suffix="_predicted")`. Every column skgp adds then gets that ending (`mean_predicted` and so on).

skgp has no save function of its own. To keep a fitted model, save it with Python's `pickle`, as you would a scikit-learn model; [docs/saving.md](docs/saving.md) shows how, and what to watch for.

To see what skgp checks and chooses, add `verbose=True` to `check`, `fit` or `predict`:

```python
fitted = model.fit(train, verbose=True)
```

## Large datasets

`GPRegressor` fits an exact model when its learned noise is constant, the default, or it has known noise. The work it does grows with the cube of the number of rows and the memory it needs grows with the square, so it becomes impractical somewhere past ten thousand rows. For a table that large, use `SparseGPRegressor`. It takes the same `features`, `target`, `kernel`, `learned_noise` and `schema`, and you call the same `check`, `fit` and `predict`. The difference is how it learns: it summarises your data with a smaller set of points (called inducing points) and fits that summary. That is an approximation, so it trades some accuracy for speed and memory. skgp never chooses it for you: a `GPRegressor` never summarises your data, and a `SparseGPRegressor` always does. Noise is always Gaussian.

This example makes up 1,000 places instead of 40, and fits them in minibatches of 250 rows:

```python
from skgp import SparseGPRegressor

# The same made-up temperatures as above, for 1,000 places.
big_latitude = rng.uniform(35, 60, 1000)
big_altitude = rng.uniform(0, 2000, 1000)
big_temperature = (
    30 - 0.4 * (big_latitude - 35) - 0.006 * big_altitude + rng.normal(0, 0.5, 1000)
)
big_train = pl.DataFrame(
    {"latitude": big_latitude, "altitude": big_altitude, "temperature": big_temperature}
)

sparse = SparseGPRegressor(
    features=["latitude", "altitude"], target="temperature", minibatch_size=250
)
sparse_fitted = sparse.fit(big_train)
print(sparse_fitted.predict(new).to_frame())  # the same kind of result as before
```

`fit` returns the same `FittedGPRegressor` as before, so `.predict()` and `.to_json()` work in the same way. Besides the settings it shares with `GPRegressor`, a `SparseGPRegressor` has three:

- `minibatch_size`: how the fit goes through your data. Leave it out (the default) and each step looks at all the rows at once, which gives the most accurate sparse fit and no randomness from batching. Give a whole number above zero and each step looks at a random batch of that many rows, which suits data too large to process at once. Zero or a negative number is refused when you define the model.
- `num_inducing`: how many points summarise the data, a whole number above zero. Leave it out and skgp uses 10% of the rows, but never fewer than 100 or more than 1,000. If that number, or yours, is more than the number of rows, every row is used. The points start at the centres of a k-means clustering of your rescaled inputs and are moved during the fit.
- `random_state`: a whole number from 0 to 9,223,372,036,854,775,807, 0 unless you change it, that seeds everything random in the fit (where the points start and the order of the minibatches). The same data and settings always give the same model; change it to see how much a fit depends on it.

How the fit ends depends on `minibatch_size`. Without one, it stops the way an exact fit does, and a fit that stops early gives the same `skgp.ConvergenceWarning`. With one, the fit runs exactly `iterations` iterations, each on one minibatch: 2,000 unless you pass another number, for example `sparse.fit(big_train, iterations=5_000)`. skgp does not judge whether such a fit converged: it issues no `skgp.ConvergenceWarning`, and `to_json()` gives the iterations it ran and its final objective, with no `finished_properly`.

`fitted.to_json()` of a sparse model also has a `model` section: the class that made it, the number of inducing points used, the minibatch size (null when there is none) and the `random_state`. With `verbose=True`, `fit` also logs whether the fit is on all rows or on minibatches, how many inducing points and whether you chose that number, how they start (before it searches for them), the `random_state`, and, for a fit on all rows, why fitting stopped. A fit on minibatches also shows GPJax's progress bar.

A `GPRegressor` that is given a lot of rows warns you. Its `fit` issues a `skgp.LargeDataWarning` (a `UserWarning`) when the data has more rows than `skgp.config.size_warning_rows`, which is 10,000 to start. The fit then carries on and gives the same model it always would. Assign to the setting to change the limit, for example `skgp.config.size_warning_rows = 50_000`. It controls only the warning, never which model is fitted, and `SparseGPRegressor` never issues it. The reasons for this design are in [ADR 0004](docs/adr/0004-sparse-models-are-a-separate-class.md).

## Classification

`GPClassifier` predicts which of two classes each row belongs to, and how likely each class is. It takes the same `features`, `kernel` and `schema` as `GPRegressor`, and you call the same `check`, `fit` and `predict`. Its `target` may hold any two different values: numbers, text, or true and false.

This example makes up whether it rained at 40 places, more often further north, and predicts the chance of rain at two new places:

```python
from skgp import GPClassifier

# Made-up data: rain is more likely further north.
rain_latitude = rng.uniform(35, 60, 40)
rained = rng.uniform(size=40) < 1 / (1 + np.exp(-(rain_latitude - 47.5) / 3))
weather = pl.DataFrame(
    {"latitude": rain_latitude, "weather": np.where(rained, "rain", "dry")}
)

classifier = GPClassifier(features="latitude", target="weather")
classifier_fitted = classifier.fit(weather)
rain_prediction = classifier_fitted.predict(pl.DataFrame({"latitude": [40.0, 55.0]}))
print(
    rain_prediction.to_frame()
)  # your rows, each class's probability, the likelier class
```

`rain_prediction.to_frame()` gives your rows, then `probability_dry` and `probability_rain`, then `most_likely_class`. `rain_prediction.probabilities()` gives the same probabilities as a numpy array with one column per class, in the order of `rain_prediction.classes`, and `rain_prediction.most_likely()` gives the likelier class of each row. skgp codes your two values as 0 and 1 in sorted order (the smaller number, false, or the text that comes first alphabetically is 0), and with `verbose=True`, `fit` logs which is which.

The model is a smooth curve over the inputs, and the probability of the second class at a row is the probit (the standard normal distribution function) of the curve there. A row far from the training data gets probabilities set by the mean function, which is learned from the data unless you choose otherwise (see [Far from the data](#far-from-the-data)). The model is fitted by variational inference with an inducing point at every training row, so, like an exact regression model, it gets slow as rows grow (about 20 seconds at 500 rows and two and a half minutes at 1,000), and `fit` issues a `skgp.LargeDataWarning` above `skgp.config.size_warning_rows`. The fit stops once it has finished, after at most `iterations` iterations: 2,000 unless you pass another number. `classifier_fitted.to_json()` lists the two classes and what the fit learned: the kernel's values, such as a length scale along each input and a variance, and no noise. The reasons for this design are in [ADR 0006](docs/adr/0006-the-classifier-uses-variational-inference-at-every-row.md).

For a large table, use `SparseGPClassifier`. It is to `GPClassifier` what `SparseGPRegressor` is to `GPRegressor`: it takes the same settings plus `minibatch_size`, `num_inducing` and `random_state`, which work as described under [Large datasets](#large-datasets), and its `fit` returns the same kind of fitted model and prediction. Without a `minibatch_size` it fits on all rows at once with the same optimiser as `GPClassifier`, stopping once it has finished. With one, it runs exactly `iterations` iterations, each on one minibatch, as a `SparseGPRegressor` does. Its `to_json()` has the same `model` section as a sparse regressor's, and it never issues a `LargeDataWarning`. skgp never switches to it for you.

## Your own kernel

`kernel` takes a name or a GPJax kernel. The names are `"matern52"` (the default) for a smooth curve, `"matern32"` for a rougher one and `"rbf"` for a very smooth one, each with one length scale per input. For any other shape of curve, such as one that repeats, pass a kernel from GPJax's `gpjax.kernels`, alone or added and multiplied together. This example fits three years of weekly sales with a slow trend plus a season that repeats every year:

```python
import gpjax as gpx

# Made-up data: weekly sales that rise slowly and repeat every 52 weeks.
week = np.arange(156.0)
season = 80 * np.sin(2 * np.pi * week / 52)
sales = 1000 + 0.5 * week + season + rng.normal(0, 10, 156)
weekly = pl.DataFrame({"week": week, "sales": sales})

# A slow trend plus a season, written in weeks, the units of the week column.
kernel = gpx.kernels.RBF(lengthscale=100.0) + gpx.kernels.Periodic(period=50.0)
seasonal = GPRegressor(features="week", target="sales", kernel=kernel)
seasonal_fitted = seasonal.fit(weekly)
print(seasonal_fitted)  # the learned period, kernels[1].period, is close to 52 weeks
```

Every kernel sees your inputs in the units of your columns, so the length scale and the period above are in weeks, both as you give them and as the fit reports them. The values you give are where the fit starts. The fit learns all of them, except any you hold with `paramax.non_trainable`, as in GPJax. A kernel is taken as you give it: `gpx.kernels.RBF()` has one length scale for all its inputs, and `gpx.kernels.RBF(lengthscale=jnp.array([1.0, 500.0]))` one for each of two. A kernel's `active_dims` are positions in `features`, counting from 0. A named kernel's length scales start at each input's standard deviation.

A regressor's kernel variance is relative to the target's variance: 1 means the curve varies as much as the target does. Printing a fitted model and `to_json()` name each learned value by its place in the kernel, as GPJax names it: `lengthscale` and `variance` for a kernel by name, and `kernels[1].period` for the period of the second kernel of a sum.

## Far from the data

Near the training data, the curve follows the data. Far from it, where the data says nothing, the curve goes to its mean function, which `mean_function` sets:

- `"constant"` (the default): one number, learned from the data.
- `"zero"`: zero in the target's units, held there, for a target that goes back to 0 far from the data, such as a difference from a long-term average.
- `"linear"`: a number plus a slope along each input, all learned, so that a prediction keeps rising or falling as the data does.
- A GPJax mean function, from `gpjax.mean_functions`, which sees your inputs and target in their own units. Its values are where the fit starts, and the fit learns them, except any you hold with `paramax.non_trainable`. On a fit on minibatches, each step moves a value by about a twentieth of the target's standard deviation, or by about a twentieth of itself if it must be positive and is above 1, so start it near where it belongs, such as `gpx.mean_functions.Constant(1000.0)` for a target near 1,000: from 0, it would get there slowly. The names start at the target's average.

```python
# The same weekly sales, predicted two years after the last week.
trending = GPRegressor(features="week", target="sales", mean_function="linear")
trending_fitted = trending.fit(weekly)
print(trending_fitted.predict(pl.DataFrame({"week": [260.0]})))  # the rise carries on
```

With the default `"constant"`, the same prediction stays at about the average of the sales. Printing a fitted model and `to_json()` give what the mean function learned in your units: a `"constant"` mean function's `constant`, in the target's units, and a `"linear"` one's `intercept`, the target where every input is 0, and its `slope` along each input, in the target's units per unit of the input's column. Along a date input, the slope is per day and the intercept is at 1970-01-01.

The slope is one learned value, so a prediction's spread does not include any doubt about it, and that matters more the further ahead you predict. A GPJax `Linear` kernel added to the kernel treats the slope as uncertain instead, but it measures from 0 in each column's units, not from your data, and a fit can turn it off when the other kernel already explains the rise: on the weekly sales above, `gpx.kernels.Matern52() + gpx.kernels.Linear()` learns a `Linear` variance near 0 and predicts no rise.

With `learned_noise="heteroscedastic"`, the mean function is the curve's; the noise process keeps its own learned constant mean. A classifier takes `mean_function` too. Far from the data, its probabilities go towards those its mean function gives: with `"zero"`, 0.5 for each class.

## Date inputs

An input column may hold dates, or dates and times, as Polars or pandas stores them: a date input. skgp counts them in days since 1970-01-01, with the time of day as a fraction of a day, so a length scale or a period along a date input is in days: `gpx.kernels.Periodic(period=365.25)` repeats every year. A date and time with a time zone is converted to UTC first; one without a time zone is taken as it is. Printing a fitted model and `to_json()` name the unit, `days since 1970-01-01`, beside each date input. A date input of the training data must be a date input in the rows to predict, and the target must hold numbers. Durations and times of day are refused, with a note on how to turn them into numbers.

## Noise

By default a regressor learns one noise level for every row: `learned_noise="constant"`. Two settings change that: noise that varies with the inputs, and known noise.

### Noise that varies

With `learned_noise="heteroscedastic"`, on a `GPRegressor` or a `SparseGPRegressor`, it learns noise that varies with the inputs, for a measurement that gets noisier in some places than in others:

```python
# Made-up data: a measurement that gets noisier as x grows.
x = np.sort(rng.uniform(0, 10, 150))
y = 3 * np.sin(x) + (0.1 + 0.3 * x) * rng.normal(size=150)
noisy = pl.DataFrame({"x": x, "y": y})

varying = GPRegressor(features="x", target="y", learned_noise="heteroscedastic")
varying_fitted = varying.fit(noisy)

ends = pl.DataFrame({"x": [1.0, 9.0]})
observation = varying_fitted.predict(ends).std()
curve = varying_fitted.predict(ends, include_noise=False).std()
print(
    (observation**2 - curve**2) ** 0.5
)  # the noise at x = 1, then the larger noise at x = 9
```

A prediction's spread includes the noise the model expects at each row, unless you ask for the curve alone with `include_noise=False`; the difference of the two variances is the noise variance at that row. The logarithm of the noise variance is a second Gaussian process, the noise process, with a `"matern52"` kernel of its own, the noise kernel, learned together with the curve. Printing a fitted model and `to_json()` give `noise`, the median noise far from the training data, and `noise_kernel`, the values of the noise kernel, such as its length scale along each input in your units.

GPJax has no exact fit of this model. A `GPRegressor` fits it by variational inference with an inducing point at every training row for the curve and for the noise process, as a `GPClassifier` is fitted, so it gets slow as rows grow and warns above `skgp.config.size_warning_rows`. A `SparseGPRegressor` gives each process `num_inducing` inducing points; on minibatches it fits with Adam alone, as GPJax takes no natural-gradient step on this model. The reasons for this design are in [ADR 0007](docs/adr/0007-heteroscedastic-noise-is-fitted-by-variational-inference.md).

### Known noise

When each observation comes with its own known noise, such as an instrument's stated error, name the column that holds it with `noise_column`. It holds a standard deviation in the target's units for every row:

```python
# Made-up data: each reading comes with the stated error of the instrument that took it.
error = rng.choice([0.1, 1.0], 60)
x = np.sort(rng.uniform(0, 10, 60))
readings = pl.DataFrame(
    {"x": x, "y": np.sin(x) + error * rng.normal(size=60), "error": error}
)

known = GPRegressor(features="x", target="y", noise_column="error")
known_fitted = known.fit(readings)

# The same place, read by the precise instrument and by the rough one.
new_readings = pl.DataFrame({"x": [2.0, 2.0], "error": [0.1, 1.0]})
print(known_fitted.predict(new_readings).std())  # the second is much wider
```

The model trusts a row less the larger its known noise. It still learns one noise level on top, for scatter the stated errors leave out: the noise variance at a row is its known noise squared plus the learned noise squared. With `learned_noise=None` it learns none, and the known noise is all the noise there is. A prediction with noise, the default, adds each new row's own known noise, so the rows to predict need the column; `include_noise=False` predicts the curve alone and does not. Only a `GPRegressor` takes `noise_column`, with `learned_noise` `"constant"` or `None`: GPJax's sparse fits, and its fit of heteroscedastic noise, take one noise level for every row. A negative or missing known noise is refused.

## How good is a fit?

`fitted.score(held_out)` scores the predictions against held-out data with known targets and returns a `Score`, which maps each scorer's name to its value:

```python
held_out = pl.DataFrame(
    {"latitude": [40.0, 50.0], "altitude": [500.0, 1500.0], "temperature": [24.0, 17.0]}
)
print(fitted.score(held_out))  # the mean CRPS, sign flipped: closer to 0 is better
print(fitted.score(held_out, scoring=["neg_crps", "r2", "neg_mean_absolute_error"]))
```

`scoring` takes what scikit-learn's `cross_val_score` takes (a scorer's name, a scorer, a list of them or a dict), plus skgp's `"neg_crps"`. A regressor is scored by `"neg_crps"` unless you say otherwise: the mean continuous ranked probability score (CRPS), which compares the whole predicted distribution of each row with what was observed, with its sign flipped. The CRPS is in the target's units and equals the absolute error when a prediction has no spread, so it rewards a mean close to the observation and a spread that is honest about how close. A classifier is scored by `"neg_brier_score"`, the mean squared difference between the predicted probability of a class and whether it happened, with its sign flipped; the Brier loss has no unit and runs from 0 (perfect) to 1. Every scikit-learn classification scorer works whatever your two classes are: the scorers see them coded 0 and 1, the second class as 1. Following scikit-learn, every value is one where higher is better. The predictions scored include measurement noise, so a model with known noise needs its `noise_column` in the held-out data.

## With scikit-learn

`skgp.sklearn` has a scikit-learn estimator for each model: `GaussianProcessRegressor`, `SparseGaussianProcessRegressor`, `GaussianProcessClassifier` and `SparseGaussianProcessClassifier`. Each takes skgp's settings, its inputs as `X` and its target as `y`, and works in a `Pipeline`, in `cross_val_score` and in `GridSearchCV`. A regressor's `score` is the mean CRPS with its sign flipped, and a classifier's the Brier loss with its sign flipped. [docs/sklearn.md](docs/sklearn.md) shows how, and how the estimators differ from scikit-learn's own Gaussian processes.

## How it behaves

- **Dataframes in, dataframes out.** You never import JAX. Polars is the main dataframe, and any common dataframe, pandas included, works through [narwhals](https://narwhals-dev.github.io/narwhals/). Results come back in the type you passed in.
- **Nothing changes in place.** `GPRegressor` describes a model: the input columns (`features`), the `target` column, and optionally a `kernel`, a `mean_function`, the noise it learns (`learned_noise`), a column of known noise (`noise_column`) and a `schema`. `.fit()` returns a separate `FittedGPRegressor` (a `FittedGPClassifier` for a `GPClassifier`), and only that has `.predict()`. `model.replace(kernel="rbf")` returns a new model with that change. You can give the inputs and the target by position, `GPRegressor(["latitude", "altitude"], "temperature")`, and a single input by its name alone.
- **Mistakes in settings are caught when you define the model.** An unknown kernel or mean function name, a kernel whose `active_dims` point past the last feature, a misspelt setting, a column named twice, the target among the features or an empty column name raises a `ValueError` straight away, with one plain line per problem that names the setting and what you gave.
- **Your data is checked first.** Missing or non-numeric columns, missing values (empty or NaN), infinite values, fewer than 5 training rows, and prediction columns that do not match each stop with a clear error that names the column or the number of rows. A classifier's target may hold values of any type, but exactly two different ones. An input column may hold dates, or dates and times: a date input (see [Date inputs](#date-inputs)); a duration or a time of day is refused with a note on how to turn it into numbers. You can run the checks ahead of a long fit with `model.check(df)`, which returns a `CheckedData` that `fit` and `predict` accept without checking again. It is valid for any model of the same task with the same `features`, `target`, `schema` and `noise_column`, whatever its kernel or class, so you can check once and then fit several kernels, or both an exact and a sparse model when neither has known noise. Data checked by a regressor is refused by a classifier, and the other way round, because they check the target differently. Rows to predict can be checked ahead with `fitted.check(rows)`, which also checks that each date input of the training data is a date input in the rows, and every other input holds numbers; `predict` repeats that one check, which is quick. They do not need a target column, and a target column among them is ignored (handy when you predict on a held-out set that includes the true values). Any other column the model does not use is ignored too, and with `verbose=True` a log record names it.
- **You can add your own checks.** Pass `schema=` an object with a `.validate(frame)` method, such as a pandera schema. It runs first, on your dataframe as you passed it, and skgp's own checks always run after it.
- **skgp handles the details.** Inputs and target are rescaled before fitting and predictions are converted back to your units. The kernel sees your inputs in their own units, so its length scales are in them. The kernel defaults to `"matern52"`; the other names are `"matern32"` and `"rbf"`, and you can pass a GPJax kernel instead (see [Your own kernel](#your-own-kernel)). Far from the data, a prediction goes to a constant learned from the data unless you choose another `mean_function` (see [Far from the data](#far-from-the-data)).
- **A fit that does not finish warns you.** If fitting does not finish properly, `fit` still returns a model but issues a `skgp.ConvergenceWarning` (a `UserWarning`) that says in plain words why it stopped, and the model may be less reliable. skgp judges this from where the fit ended, not from what the optimiser reported: a fit counts as finished when further steps would no longer change it by much, even if the optimiser gave up or ran out of iterations. Filter or escalate the warning with the `warnings` module. Only a fit on all rows at once is judged: a fit on minibatches runs the `iterations` you give it, and skgp does not say whether it converged.
- **You set how long a fit may run.** Every `fit` takes `iterations`, 2,000 unless you give another whole number above zero. For a `GPRegressor`, a `SparseGPRegressor` without a `minibatch_size`, or a `GPClassifier`, it is the most iterations the optimiser may take, and it stops sooner once the fit has finished. For a fit on minibatches, it is exactly the number of iterations run. When a fit on all rows used them all without finishing, the `ConvergenceWarning` suggests a larger `iterations`.
- **Predictions are for a new observation unless you ask for the curve.** `fitted.predict()` returns a `Prediction`. Its `.mean()` and `.std()` are numpy arrays in your original units, and by default the spread includes measurement noise. `fitted.predict(new, include_noise=False)` gives the underlying curve without the noise: the same mean, with a smaller spread. Printing a prediction says which it is and shows its table, and a notebook shows that table as a table. `.to_frame()` and `.interval()` give the results as dataframes of your type. `.covariance()` gives the n-by-n covariance between the rows as a numpy array, which tells you how the predictions move together; it costs memory that grows with the square of the number of rows, so ask for it only when you need it. `.sample(n_samples, random_state=0)` gives random samples, in the shape scikit-learn's `sample_y` gives: an array with a row for each row and a column for each sample, each sample chosen at random from the joint distribution over all the rows, so nearby rows hang together. A `ClassPrediction`'s `.sample()` gives classes the same way: one possible underlying curve per sample, then a class at each row.
- **Repeated predictions are quick.** The first `predict` on a fitted model prepares the training data once and keeps the result, so later predictions reuse it. For a `GPRegressor` that kept result takes memory that grows with the square of the training rows (about 72 MB at 3,000 rows and 800 MB at 10,000). `predict` works through your rows 1,000 at a time; change that with `batch_size=`.
- **You can see inside the black box.** With `verbose=True`, `check`, `fit` and `predict` log each check and each choice skgp makes, as structured records through [loguru](https://loguru.readthedocs.io/). A long fit logs its iteration and objective every few seconds, and its last record gives how long the fit took and what it learned. Without `verbose`, skgp logs nothing. To send the records to Python's own `logging`, see [docs/advanced.md](docs/advanced.md#sending-the-log-records-to-logging). Printing a fitted model shows a short summary: the class that made it, its columns and kernel, the number of rows, whether the fit finished properly, and what it learned: the kernel's values, with length scales in your columns' units, the mean function's values and the noise, in your target's. `fitted.to_json()` gives the same as JSON, with the skgp version, for inspection. You cannot rebuild a model from it.
- **Saving a model is up to you.** skgp has no save function of its own. Save a fitted model with Python's `pickle`, as you would a scikit-learn model. Only load files you trust, because loading a pickle file can run code. Load them with the same versions of skgp, GPJax and JAX that saved them, and treat the file as you treat your training data, which it holds. See [docs/saving.md](docs/saving.md).
- **GPJax stays out of the way.** skgp is built on [GPJax](https://github.com/QuantClimate/GPJax), which does the fitting, and the underlying GPJax model is reachable from a fitted model for anyone who goes looking.

For advanced use, see [docs/advanced.md](docs/advanced.md).

## Not supported yet

Models for data on graphs, extreme-value models, and fully Bayesian fitting.

Classification is offered through `GPClassifier` and `SparseGPClassifier` (see [Classification](#classification)), with these limits:

- Two classes only.

Sparse approximations for very large datasets are offered through `SparseGPRegressor` (see [Large datasets](#large-datasets)), with these limits:

- No automatic choice. skgp does not pick between an exact and a sparse model for you, or between the two ways of fitting a sparse one ([ADR 0004](docs/adr/0004-sparse-models-are-a-separate-class.md)).
- No non-Gaussian noise in regression. A `SparseGPRegressor` takes Gaussian noise only.
- No chunked fits. Fitting exact models on pieces of the data and combining their predictions is not offered.

## Design notes

The project's vocabulary is in [GLOSSARY.md](GLOSSARY.md). Decisions that are hard to undo are recorded in [docs/adr](docs/adr).
