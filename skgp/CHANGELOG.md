# Changelog

All notable changes to skgp are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

## [0.1.0] - 2026-11-26

skgp's code moved into the [GPJax repository](https://github.com/QuantClimate/GPJax), in the folder `skgp/`. Install skgp from PyPI as before. Report issues at https://github.com/QuantClimate/GPJax/issues.

### Changed

- skgp needs `gpjax>=1.1.0,<2`. A future GPJax 2.0 does not install under this release.
- The documentation links names from GPJax, such as `gpjax.gps.Prior`, to GPJax's documentation.

## [0.0.4] - 2026-11-26

`mean_function` on every model: `"constant"`, now learned and the default, `"zero"`, `"linear"` or a GPJax mean function. The learned default changes every fit slightly near the data and predictions far from the data the most: a classifier no longer goes to a probability of 0.5 for each class there, which `mean_function="zero"` gives back. A model saved with pickle by 0.0.3 cannot be printed or described with `to_json()` by 0.0.4; load it with the version that saved it.

### Added

- `mean_function` on every model and every scikit-learn estimator: what the curve is far from the training data, where the data says nothing. `"constant"` (the default) is one number learned from the data; `"zero"` is zero in the target's units, held there; `"linear"` is a number plus a slope along each input, all learned, so a prediction far from the data keeps rising or falling along each input as the data does; and any GPJax mean function from `gpjax.mean_functions` works too, seeing the inputs and the target in their own units. With heteroscedastic noise, the mean function is the curve's; the noise process keeps its own learned constant mean. On a fit on minibatches, each step moves a value of a GPJax mean function by about a twentieth of the target's standard deviation (of 1 for a classifier), or by about a twentieth of the value itself if it must be positive and is above 1, as it moves skgp's own values, so a value that starts far from where it belongs gets there slowly. `to_json()` gives the setting under `setup.mean_function` and what it learned under `learned.mean_function`, and a printed fitted model and the `fit finished` log record give them after the kernel, as `mean_function` and `mean_function.<value>`, in your units: a `"constant"` mean function's `constant`, in the target's units, and a `"linear"` one's `intercept`, the target where every input is 0, and `slope` along each input, in the target's units per unit of the input's column (per day along a date input). A classifier's mean function is on the scale of the curve it turns into probabilities, with no units. With `verbose=True`, `fit` logs a `mean function chosen` record. `fitted.gpjax_model`'s prior holds skgp's own mean function, on the rescaled inputs and target, in place of GPJax's `Zero` (`docs/advanced.md` says how it is built).

### Changed

- A model's mean far from the data is now learned by default. Before, a regressor's was held at the training target's average, and a classifier's curve at 0, which gave each class a probability of 0.5 far from the data; `mean_function="zero"` gives a classifier that behaviour back. Fits, and so predictions, change a little near the data too.

## [0.0.3] - 2026-11-26

Date inputs, known noise, samples, scikit-learn estimators, and scores on held-out data. The regressors' `noise` setting is renamed `learned_noise`, so code that passes `noise=` must change.

### Added

- Date inputs: an input column may hold dates, or dates and times. skgp counts them in days since 1970-01-01 in UTC, with the time of day as a fraction of a day, so a length scale or a period along a date input is in days. A date and time with a time zone is converted to UTC; one without a time zone is taken as it is. Printing a fitted model names the unit beside each date input, and `to_json()` lists them under `setup.date_features`. A date input of the training data must be a date input in the rows to predict (`fitted.check` and `predict` check this), and an input that held numbers must hold numbers. The target must still hold numbers, and durations and times of day are still refused.
- Known noise: `noise_column` on a `GPRegressor` names the column that holds each observation's known noise, such as an instrument's stated error, as a standard deviation in the target's units. The model trusts a row less the larger its known noise, and learns a constant noise on top of it (the variances add), or none with `learned_noise=None`. A prediction with noise adds each new row's own known noise, so the rows to predict need the column unless `include_noise=False`. A negative or missing known noise is refused. A `SparseGPRegressor` takes no known noise, because GPJax's sparse fits take the same noise at every row, and known noise cannot be combined with `learned_noise="heteroscedastic"`.
- Samples: `Prediction.sample(n_samples=1, random_state=0)` gives random samples of what the model expects, in the shape scikit-learn's `sample_y` gives: a numpy array with a row for each row and a column for each sample, in the target's units. Each sample is drawn from the joint normal distribution over all the rows, so nearby rows hang together, and includes measurement noise when the prediction does; with heteroscedastic noise, samples of new observations come from a normal distribution with the right mean and covariance. `ClassPrediction.sample(...)` gives your classes the same way: one possible underlying curve per sample, the one the model turns into probabilities, then a class at each row with the probability that curve gives there.
- scikit-learn estimators: `skgp.sklearn` has `GaussianProcessRegressor`, `SparseGaussianProcessRegressor`, `GaussianProcessClassifier` and `SparseGaussianProcessClassifier`, one for each model. Each takes the model's settings except `features`, `target` and `schema`, plus `iterations`, its inputs as `X` (a Polars or pandas dataframe, whose columns keep their names, or an array, whose columns are `x0`, `x1`, ...) and its target as `y`, and `fit` changes it and returns it, so it works in a `Pipeline`, `cross_val_score` and `GridSearchCV`. The methods are scikit-learn's: `predict` with `return_std` and `return_cov`, `predict_proba`, and `sample_y` on every estimator, with `include_noise` on a regressor's. A regressor's `score` is the mean continuous ranked probability score (CRPS) with its sign flipped, and a classifier's the Brier loss with its sign flipped. `skgp.sklearn.neg_crps` is the CRPS as a scorer for `scoring=`. `fitted_` is the fitted skgp model. scikit-learn's own estimator checks pass, except where skgp refuses fewer than 5 rows and where the CRPS is not an R². scikit-learn and scoringrules are new dependencies (decision record 0009).
- A page, "Using skgp with scikit-learn" (`docs/sklearn.md`).
- Scores: `fitted.score(data, scoring=...)` scores a fitted model's predictions against held-out data with known targets, and returns a `Score`, which maps each scorer's name to its value and prints as a short table. `scoring` takes what `cross_val_score` takes (a scikit-learn scorer's name, a scorer, a list of them or a dict), and skgp's `"neg_crps"`. A regressor is scored by `"neg_crps"`, the mean CRPS with its sign flipped (in the target's units, computed with scoringrules), and a classifier by `"neg_brier_score"`, the Brier loss with its sign flipped (no unit, from 0 to 1), unless you say otherwise. A classifier's scorers see its classes coded 0 and 1, so scorers such as `f1` work whatever the two classes are. Every value is one where higher is better.

### Changed

- The regressors' `noise` setting is now `learned_noise`: `"constant"` (the default), `"heteroscedastic"`, or `None` for a `GPRegressor` whose noise is all known noise. `to_json()` names it `setup.learned_noise`, gives `setup.noise_column` for a model with known noise, and leaves out `learned.noise` for a model that learns no noise.

## [0.0.2] - 2026-11-25

The first release with working code: 0.0.1 only reserved the name on PyPI.

### Added

- `GPRegressor`, `FittedGPRegressor` and `Prediction`: fit an exact Gaussian process regression on a dataframe and predict new observations.
- `SparseGPRegressor`: a regression model for datasets too large for an exact one, with the same `features`, `target`, `kernel`, `noise` and `schema` and the same `check`, `fit` and `predict` as `GPRegressor`. It fits a sparse variational approximation: the data is summarised by a smaller set of inducing points, which start at k-means centres of the rescaled inputs (of a random subset of 10,000 rows when there are more) and are tuned during the fit. `fit` returns the same `FittedGPRegressor`, with the same `Prediction`. skgp never picks it for you, and `GPRegressor` always fits an exact model (decision record 0004, `docs/adr/0004-sparse-models-are-a-separate-class.md`).
- The settings of the sparse models: `minibatch_size` (none by default, which fits on all rows at once with L-BFGS-B, as an exact fit does; a whole number above zero fits on random minibatches of that size, with natural-gradient steps alternating with Adam steps), `num_inducing` (none by default, which means 10% of the rows but no fewer than 100 and no more than 1,000; every row is used if there are fewer rows than that, or than the number you give) and `random_state` (a whole number from 0 to 9,223,372,036,854,775,807, 0 by default, seeding the k-means starting points and the minibatch order, so the same data and settings give the same model).
- `GPClassifier`, `FittedGPClassifier` and `ClassPrediction`: Gaussian process classification of a target with two classes of any type (numbers, text, or true and false), from the same `features`, `kernel` and `schema` as `GPRegressor`. It is fitted by variational inference with an inducing point at every training row, with L-BFGS-B, as an exact fit does (decision record 0006, `docs/adr/0006-the-classifier-uses-variational-inference-at-every-row.md`). `predict` gives the probability of each class at each row and the likelier class, and `to_frame()` adds `probability_<class>` for each class and `most_likely_class` to your rows. `ClassPrediction.probabilities()` and `most_likely()` give the same as arrays. Inside the fit the two classes are coded 0 and 1 in sorted order; `fitted.classes` and `to_json()` list them in that order, and `verbose=True` logs which is which.
- `SparseGPClassifier`: a classifier for datasets too large for a `GPClassifier`, with the same settings plus `minibatch_size`, `num_inducing` and `random_state`, which work as for `SparseGPRegressor`. `fit` returns the same `FittedGPClassifier`.
- `kernel` on every model: `"rbf"`, `"matern32"` or `"matern52"`, or any kernel from `gpjax.kernels`, alone or added and multiplied together, such as `gpjax.kernels.RBF() + gpjax.kernels.Periodic(period=52.0)`. A given kernel's values are where the fit starts, and one held with `paramax.non_trainable` stays where it is. Every kernel sees the inputs in the units of your columns, so its length scales and periods are in those units; a named kernel's length scales start at each input's standard deviation. A kernel whose `active_dims` point past the last feature is refused when the model is defined.
- `noise` on `GPRegressor` and `SparseGPRegressor`: `"constant"` (the default) or `"heteroscedastic"`, noise that varies with the inputs. The logarithm of the noise variance is a second Gaussian process, the noise process, with a `"matern52"` kernel and a learned constant mean. Both regressors fit it by variational inference: a `GPRegressor` with an inducing point at every row for the curve and for the noise process, a `SparseGPRegressor` with `num_inducing` for each, and on minibatches with Adam alone (decision record 0007, `docs/adr/0007-heteroscedastic-noise-is-fitted-by-variational-inference.md`). A prediction's spread includes each row's own noise unless `include_noise=False`.
- `iterations` on every model's `fit`, 2,000 unless you give another whole number above zero. For a fit on all rows at once it is the most iterations the optimiser may take; for a fit on minibatches it is exactly the number of iterations run.
- Any dataframe in, the same type out: `fit` and `predict` accept Polars and pandas dataframes (through narwhals), and `to_frame()` returns the type passed to `predict`.
- Inputs, and a regressor's target, are rescaled before fitting, and predictions are returned in your original units.
- `Prediction.to_frame()` returns every column you passed to `predict`, followed by `mean` and `std`, and `to_frame(interval=0.95)` adds `lower` and `upper`. `Prediction.interval(level=0.95)` gives just the two bounds at each row: for a new observation, or for the underlying curve if the prediction was made with `include_noise=False`. `Prediction.covariance()` gives the covariance between every pair of predicted rows, as a numpy array with one row and one column per predicted row, built only when asked.
- `predict(..., include_noise=False)` predicts the underlying curve without the noise on each observation: the same mean, with a smaller spread. `Prediction.includes_noise` and the printed form say which a prediction is.
- `predict(..., suffix="_predicted")` adds that text to the name of every column skgp adds, for rows that already have a column with one of those names. Without a suffix such a clash raises a `ValueError` that names the column.
- `predict(..., batch_size=1_000)`: how many rows are predicted at a time, so memory does not grow with training rows times prediction rows. A fitted model prepares its training data for prediction once, on the first `predict`, and keeps the result.
- `to_json()` on a fitted model: a read-only JSON description of it, with `setup` (columns, kernel, and a regressor's `noise`), `learned` (every value of the fitted kernel by its place in the kernel as GPJax names it, such as `lengthscale` per input column and `variance`, and a regressor's noise as a standard deviation in the target's units), `fitting` (`iterations` and `final_objective`, and for a fit on all rows at once `finished_properly` and `stop_reason`), the number of training rows and the skgp version. A sparse model's also has a `model` section: `class`, `num_inducing`, `minibatch_size` and `random_state`.
- Data checks and `CheckedData`: `fit` and `predict` refuse missing columns, non-numeric input columns, missing or infinite values, and training data with fewer than 5 rows, with errors that name the column and counts. A date or time column is refused with a note on converting it to numbers. `model.check(df)` runs the checks once and returns a `CheckedData` that `fit` and `predict` accept without checking again, on any model of the same task with the same `features`, `target` and `schema`; data checked by a regressor is refused by a classifier, and the other way round. `check(df, for_prediction=True)` needs no target column, and `fitted.check(rows)` does the same from a fitted model.
- Columns given to `predict` or `check` that are neither inputs nor the target are ignored, and named by a `columns ignored` log record with `verbose=True`; a target column among rows to predict is ignored too, so you can predict on a held-out set that includes the true values.
- `schema=...` on every model: your own data checker, any object with a `.validate(frame)` method (such as a pandera schema). It runs first, on your dataframe as you passed it; skgp's own checks still run after it. skgp does not depend on pandera.
- `verbose=True` on `check`, `fit` and `predict`: structured log records through `loguru` showing the data checks, the settings you gave, what skgp chose (model, likelihood, kernel, rescaling, optimiser, inducing points), during a long fit on all rows at once the iteration and objective every few seconds, and during a fit on minibatches GPJax's progress bar, how long the fit took, what it learned, and, for a fit on all rows at once, whether it finished properly. With `verbose` off skgp emits nothing. `docs/advanced.md` shows how to send the records to Python's `logging`.
- `skgp.ConvergenceWarning` (a `UserWarning`): `fit` issues it once when a fit on all rows at once did not finish properly, judged from where the fit ended rather than from the optimiser's report. The message says why the fit stopped, and suggests a larger `iterations` when the fit used all of them. A fit on minibatches is not judged and never issues it (decision record 0008, `docs/adr/0008-only-a-fit-by-l-bfgs-b-is-judged-to-have-converged.md`).
- `skgp.LargeDataWarning` (a `UserWarning`): `fit` on a `GPRegressor`, with either kind of noise, or on a `GPClassifier` issues it once when the training data has more rows than `skgp.config.size_warning_rows` (10,000 to start). The message names the sparse model to use instead. The sparse models never issue it.
- Settings are easy to give and to get right. `features` and `target` can be given by position, and `features` can be a single column name. `model.replace(**changes)` returns a new, checked model. Editors and type checkers see every setting. A mistake in a setting raises a plain `ValueError` when the model is defined, with one line per problem, and an unknown setting name suggests the closest real one.
- Printing shows results, not internals. A model prints its settings; a fitted model prints the class that made it, its columns and kernel, the number of rows, how the fit ended if it was a fit on all rows at once, and what it learned; a `Prediction` prints the table `to_frame()` gives, and notebooks show it as an HTML table.
- A fitted model and its predictions are equal only to themselves and can be hashed. Two models are equal when their settings are, with kernels compared by GPJax.
- A fitted model can be saved with Python's `pickle`, as a scikit-learn model is, and predicts the same after loading. skgp has no save or load function of its own (decision record 0005, `docs/adr/0005-skgp-does-not-save-models.md`); `docs/saving.md` shows how, and what to watch for.
- `fitted.model`: the model description the fitted model was made by, which `CheckedData.model` also gives for checked data.
- Counts of 1,000 or more in errors, warnings, log records and printed predictions are written with a comma between the thousands.
- `fitted.gpjax_model`: the fitted GPJax model skgp predicts with, for cases skgp does not cover. `docs/advanced.md` says how to use it.

### Removed

- The placeholder `hello()` function.
- Support for Python 3.11. skgp now needs Python 3.12 or newer, as the newest NumPy and JAX releases do.


## [0.0.1]

### Added

- The name `skgp` on PyPI, with a placeholder `hello()` function.

[Unreleased]: https://github.com/QuantClimate/GPJax/compare/skgp-v0.1.0...HEAD
[0.1.0]: https://github.com/QuantClimate/GPJax/releases/tag/skgp-v0.1.0
[0.0.4]: https://pypi.org/project/skgp/0.0.4/
[0.0.3]: https://pypi.org/project/skgp/0.0.3/
[0.0.2]: https://pypi.org/project/skgp/0.0.2/
[0.0.1]: https://pypi.org/project/skgp/0.0.1/
