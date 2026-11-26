# skgp offers scikit-learn estimators beside its frozen models

skgp now has a scikit-learn estimator for each of its four models, in `skgp.sklearn`: `GaussianProcessRegressor`, `SparseGaussianProcessRegressor`, `GaussianProcessClassifier` and `SparseGaussianProcessClassifier`. Each takes skgp's settings, its inputs as `X` and its target as `y`, and `fit` changes it and returns it, so that it works in a `Pipeline`, `cross_val_score` and `GridSearchCV`. Inside, it describes the matching skgp model and keeps the fitted model as `fitted_`. skgp's own models stay as [ADR 0001](0001-fit-returns-a-new-object.md) made them: frozen, with `fit` returning a separate fitted model. This supersedes ADR 0001's consequence that skgp does not work in scikit-learn pipelines. We chose this because skgp is meant to be a better choice than scikit-learn's Gaussian processes for people who use scikit-learn's tools, and those tools call `fit` on an object and then `predict` on the same object, which a frozen model cannot allow. The estimators carry scikit-learn's fitted-or-not state; the dataframe API keeps its guarantee that an unfitted model cannot predict.

## Considered Options

- **Make the models themselves scikit-learn estimators:** one class per model instead of two, but `fit` would change the model, `FittedGPRegressor` would go, and an unfitted model could be asked to predict. Rejected for the reasons in ADR 0001.
- **A wrapper that takes a skgp model, such as `Estimator(GPRegressor(...))`:** fewer classes, but a grid search would have to name settings through the wrapper, and `features` and `target` would mean nothing, as `X` and `y` give them. Rejected in favour of estimators that take the settings directly.
- **scikit-learn as an optional extra:** a lighter install, but the estimators, and scoring with scikit-learn's metrics, are part of what skgp offers, so scikit-learn is a dependency.

## Consequences

- The estimators keep scikit-learn's method names and shapes, with skgp's meanings where they differ: a regressor's spread includes measurement noise unless `include_noise=False`, a regressor's `score` is the mean CRPS with its sign flipped rather than R², and a classifier's is the Brier loss with its sign flipped rather than accuracy.
- scikit-learn's own estimator checks run against all four in the tests. Two checks are expected to fail, each listed with the reason: one fits on fewer than 5 rows, which skgp refuses, and one expects a regressor's `score` to be above 0.5, as an R² of a good fit is, which a CRPS with its sign flipped never is.
- A setting added to a model has to be added to its estimator too.
