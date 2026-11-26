# API reference

Everything below is importable from `skgp` itself, for example `from skgp import GPRegressor`. Each page is rendered from the docstrings in the source.

## Describe a model

A description names the columns and settings. It is never fitted in place: `fit` returns a separate fitted model.

```{eval-rst}
.. currentmodule:: skgp

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GPRegressor
   SparseGPRegressor
   GPClassifier
   SparseGPClassifier
```

## Fitted models, predictions, scores and checked data

```{eval-rst}
.. currentmodule:: skgp

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FittedGPRegressor
   Prediction
   FittedGPClassifier
   ClassPrediction
   Score
   CheckedData
```

## scikit-learn estimators

Importable from `skgp.sklearn`, for example `from skgp.sklearn import GaussianProcessRegressor`. See [Using skgp with scikit-learn](../sklearn.md).

```{eval-rst}
.. currentmodule:: skgp.sklearn

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GaussianProcessRegressor
   SparseGaussianProcessRegressor
   GaussianProcessClassifier
   SparseGaussianProcessClassifier
   neg_crps
```

## Global settings

`skgp.config` is the one object of the class below. Change a setting for the rest of the session by assigning to it, for example `skgp.config.size_warning_rows = 50_000`.

```{eval-rst}
.. autoclass:: skgp._config.Config
```

## Warnings

Both are subclasses of `UserWarning`, so the {py:mod}`warnings` module can filter them or turn them into errors.

```{eval-rst}
.. currentmodule:: skgp

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ConvergenceWarning
   LargeDataWarning
```
