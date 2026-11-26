# Sparse models are a separate class, and skgp never switches to them on its own

For datasets too large for an exact Gaussian process, skgp offers `SparseGPRegressor` as its own class, and `GPRegressor` always fits an exact model when its noise is constant or known. (With heteroscedastic noise there is no exact fit; [ADR 0007](0007-heteroscedastic-noise-is-fitted-by-variational-inference.md) records how a `GPRegressor` fits it.) We first planned to let `GPRegressor` pick its approximation from the number of rows, so users would not have to choose, and reversed that. A sparse model is an approximation, and the same `GPRegressor` would have given a different model on 9,900 rows than on 10,100, with limits that could only be guessed and no way to compare the approximation with the exact fit on one's own data.

## Considered Options

- **Pick the approximation from the number of training rows:** spares the user the choice. Rejected because the model would change at a data size the user never chose, and the limits would be arbitrary numbers (the paper we looked at only says "roughly 10⁴ to 10⁵" rows).
- **Chunked exact fits with combined predictions:** a baseline the same paper calls hard to beat. Not chosen; we use sparse variational models.

## Consequences

- `GPRegressor` warns when given more rows than a limit (`size_warning_rows` in skgp's global settings, 10,000 to start), names `SparseGPRegressor`, and still fits. The limit controls only the warning, never which model is fitted.
- A single setting, `minibatch_size`, decides how a sparse model is fitted: none means the collapsed fit on the full data, a number means the uncollapsed fit on minibatches. The collapsed fit needs the training data to predict, so its fitted model keeps the data; the uncollapsed fitted model does not.
- Both classes return the same `FittedGPRegressor`.
- The uncollapsed fit uses random numbers, so `SparseGPRegressor` takes an integer `random_state` with a fixed default. The same data and settings give the same model, in the spirit of [ADR 0003](0003-the-fit-is-not-constrained-or-restarted.md).
- The same holds for classification: `SparseGPClassifier` is separate from `GPClassifier`, which always fits an inducing point at every row.
- Anyone adding automatic selection later is reopening this decision; read this record first.
