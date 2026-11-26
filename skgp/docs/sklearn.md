# Using skgp with scikit-learn

`skgp.sklearn` has a scikit-learn estimator for each of skgp's four models. An estimator follows scikit-learn's rules: it takes its inputs as `X` and its target as `y`, and `fit` changes it and returns it. So it works anywhere scikit-learn expects an estimator: in a `Pipeline`, in `cross_val_score` and in `GridSearchCV`.

| skgp model | scikit-learn estimator |
|---|---|
| `GPRegressor` | `skgp.sklearn.GaussianProcessRegressor` |
| `SparseGPRegressor` | `skgp.sklearn.SparseGaussianProcessRegressor` |
| `GPClassifier` | `skgp.sklearn.GaussianProcessClassifier` |
| `SparseGPClassifier` | `skgp.sklearn.SparseGaussianProcessClassifier` |

## Switching from scikit-learn's Gaussian processes

Change the import. The names, and the methods you call, are scikit-learn's:

```python
import numpy as np
from skgp.sklearn import GaussianProcessRegressor

# Made-up data: a smooth curve along two inputs, plus noise.
rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, (60, 2))
y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.1, 60)

estimator = GaussianProcessRegressor().fit(X, y)
mean, std = estimator.predict(X[:3], return_std=True)
samples = estimator.sample_y(X[:3], n_samples=5)
```

The settings are skgp's, not scikit-learn's. There is no kernel object to build and no `alpha`, `normalize_y` or `n_restarts_optimizer` to tune:

- `kernel`: `"matern52"` (the default), `"matern32"`, `"rbf"`, or any GPJax kernel. scikit-learn's kernel objects do not work. skgp rescales the inputs and the target itself, and its length scales are in the units of the columns of `X`.
- `mean_function`: what the curve is far from the data: `"constant"` (the default), `"zero"`, `"linear"`, or any GPJax mean function.
- `learned_noise`: `"constant"` (the default), `"heteroscedastic"`, or `None` with known noise.
- `noise_column`: the column of `X` that holds each row's known noise (see below).
- `iterations`: the most iterations the fit may take, 2,000 by default.
- The sparse estimators also take `num_inducing`, `minibatch_size` and `random_state`, as `SparseGPRegressor` and `SparseGPClassifier` do.

The settings are checked when you call `fit`, as scikit-learn asks, and a mistake raises `ValueError` with a plain message. After `fit`, `fitted_` is the fitted skgp model, so `print(estimator.fitted_)` shows what it learned and `estimator.fitted_.to_json()` gives the same as JSON.

## How the estimators differ from scikit-learn's own

- **The spread includes measurement noise.** `predict(X, return_std=True)` gives the spread of a new observation, noise included, unless you pass `include_noise=False` for the underlying curve alone. `sample_y` takes `include_noise` too. scikit-learn's `GaussianProcessRegressor` includes noise only when its kernel has a `WhiteKernel`.
- **`score` judges the whole prediction.** A regressor's `score` is the mean continuous ranked probability score (CRPS) with its sign flipped, not R². The CRPS compares the predicted distribution of each row with what was observed. It is in the target's units, it equals the absolute error when the prediction has no spread, and lower is better, so higher is better once its sign is flipped. A classifier's `score` is the Brier loss with its sign flipped, as scikit-learn's `neg_brier_score` scorer gives it, not accuracy. `cross_val_score` and `GridSearchCV` use `score` when you give no `scoring`.
- **Columns have names.** A Polars or pandas `X` keeps its column names, which appear in `fitted_` and in `feature_names_in_`. A numpy array's columns are called `x0`, `x1`, and so on. Fit on a dataframe and predict on an array, or the other way round, and scikit-learn warns, and the columns are matched by position. A column of dates is a date input, counted in days since 1970-01-01.
- **Two classes only.** The classifiers take exactly two classes, of any type, and refuse more. scikit-learn's classifiers have no `sample_y`; skgp's do, so that every estimator samples the same way.
- **At least 5 rows.** skgp refuses to fit fewer.

## In a pipeline

A `Pipeline` transforms `X` before the estimator sees it. skgp rescales the inputs itself, so a scaler is not needed, but any transformer works:

```python
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(
    ColumnTransformer([("scale", StandardScaler(), [0, 1])]),
    GaussianProcessRegressor(),
)
scores = cross_val_score(
    pipeline, X, y, cv=3
)  # the mean CRPS of each fold, sign flipped
```

## Choosing settings with a grid search

`GridSearchCV` tries each setting you list and keeps the best. Pass `skgp.sklearn.neg_crps` as `scoring` to compare by the CRPS alongside other scorers, or give no `scoring` to use the estimator's `score`, which is the same:

```python
from sklearn.model_selection import GridSearchCV
from skgp.sklearn import neg_crps

search = GridSearchCV(
    GaussianProcessRegressor(),
    {"kernel": ["rbf", "matern32", "matern52"]},
    cv=3,
    scoring={"crps": neg_crps, "r2": "r2"},
    refit="crps",
).fit(X, y)
print(search.best_params_)
```

`neg_crps` scores any regressor whose `predict` takes `return_std`, scikit-learn's own `GaussianProcessRegressor` included, so you can compare the two on the same folds.

## Known noise

When each row comes with its own known noise, such as an instrument's stated error, keep it as a column of `X` and name it with `noise_column`. It is a standard deviation in the target's units, it is not an input, and cross-validation splits it with its rows. Name an array's column as skgp names it: `"x2"` for the third.

A transformer earlier in a pipeline would change the noise column too, so pass it through unchanged with a `ColumnTransformer`. Its output columns are named after the transformers, so keep the names with `set_output(transform="pandas")` and `verbose_feature_names_out=False`:

```python
import pandas as pd

readings = pd.DataFrame(
    {"x": X[:, 0], "z": X[:, 1], "error": rng.choice([0.05, 0.3], 60)}
)
noisy_y = y + readings["error"] * rng.normal(size=60)

keep_the_noise = ColumnTransformer(
    [("scale", StandardScaler(), ["x", "z"]), ("noise", "passthrough", ["error"])],
    verbose_feature_names_out=False,
).set_output(transform="pandas")
known = make_pipeline(keep_the_noise, GaussianProcessRegressor(noise_column="error"))
known.fit(readings, noisy_y)
known_mean, known_std = known.predict(readings.iloc[:3], return_std=True)
```

A prediction with noise adds each row's own known noise. scikit-learn wants the columns `X` had at `fit`, so the rows you predict have the noise column even for `include_noise=False`, which does not use it. `SparseGaussianProcessRegressor` takes no known noise, as `SparseGPRegressor` does not.

## skgp's own models

The estimators are a second way in, for scikit-learn's tools; the models fitted are the same. skgp's own models, such as `GPRegressor`, differ in these ways:

- **They never change.** `GPRegressor(...).fit(df)` returns a separate `FittedGPRegressor`, and only that can predict. An estimator's `fit` changes the estimator, and the fitted skgp model is its `fitted_`.
- **They take one dataframe with your column names.** You name the inputs (`features`) and the `target` among its columns. An estimator takes `X` and `y` apart, and inside `fitted_` the target is called `y`.
- **They return richer results.** `fitted.predict(rows)` returns a `Prediction` (a `ClassPrediction` for a classifier) that puts the predictions next to your rows with `to_frame()`, and gives intervals, the covariance and samples. An estimator's `predict` returns numpy arrays, as scikit-learn's do.
- **They take a `schema`,** your own data checker, which runs before skgp's checks. An estimator takes none, because a pipeline has changed the data before it gets there; check the data before the pipeline instead.
- **Rows to predict need only the columns they use.** A `GPRegressor` with known noise needs the noise column only for a prediction with noise; an estimator needs every column it was fitted on.

skgp's fitted models score themselves with the same scorers: `fitted.score(held_out, scoring=["neg_crps", "r2"])` returns a `Score`.

[Decision record 0009](https://github.com/QuantClimate/GPJax/blob/main/skgp/docs/adr/0009-skgp-offers-scikit-learn-estimators-beside-its-frozen-models.md) records why skgp has both.
