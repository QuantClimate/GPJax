# `fit` returns a new fitted object instead of changing the model

`GPRegressor` is a frozen description of a model, and `.fit(df)` returns a separate `FittedGPRegressor`; neither is ever changed in place. The name "skgp" and the `.fit()` / `.predict()` methods suggest scikit-learn, where `fit` changes the estimator and returns it, and a reader will expect that. We chose the other shape because an object that is either fitted or not is a state users can get wrong, and because a frozen definition can be shared, compared and reused safely. We are not following scikit-learn's conventions, so skgp models do not work in scikit-learn pipelines. (Superseded in part by [ADR 0009](0009-skgp-offers-scikit-learn-estimators-beside-its-frozen-models.md): skgp now also offers scikit-learn estimators, beside these models, which stay as they are.)

## Considered Options

- **scikit-learn style (`fit` changes the model and returns it):** familiar, and would work in scikit-learn pipelines. Rejected because of the unfitted-versus-fitted state described above.

## Consequences

- Calling `predict` on an unfitted model is impossible: only a `FittedGPRegressor` has `predict`.
- Fitting the same `GPRegressor` to two datasets gives two independent fitted models.
