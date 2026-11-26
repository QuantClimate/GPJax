"""scikit-learn estimators: skgp's four models with scikit-learn's rules, so that they work in a
`Pipeline`, in `cross_val_score` and in `GridSearchCV`.

Each estimator takes its inputs as `X` and its target as `y`, and `fit` changes it and returns
it, as scikit-learn requires. Inside, it describes the matching skgp model, fits it and keeps
the fitted model as `fitted_`. skgp's own models stay as they are: frozen, with `fit` returning
a separate fitted model (decision record 0009).

    from skgp.sklearn import GaussianProcessRegressor

    estimator = GaussianProcessRegressor(kernel="rbf").fit(X, y)
    mean, std = estimator.predict(X_new, return_std=True)
"""

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Self

import narwhals.stable.v2 as nw
import numpy as np
import polars as pl
import scoringrules
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import brier_score_loss, get_scorer
from sklearn.utils import Tags
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    column_or_1d,
    validate_data,
)

from skgp import _frames
from skgp._classifier import GPClassifier, SparseGPClassifier
from skgp._prediction import ClassPrediction, Prediction
from skgp._regressor import GPRegressor, SparseGPRegressor
from skgp._types import Frame, Kernel, LearnedNoise, MeanFunction, Scoring

__all__ = [
    "GaussianProcessClassifier",
    "GaussianProcessRegressor",
    "SparseGaussianProcessClassifier",
    "SparseGaussianProcessRegressor",
    "neg_crps",
]

_SkgpModel = GPRegressor | SparseGPRegressor | GPClassifier | SparseGPClassifier

TARGET = "y"
"""The name the target gets in the skgp model inside an estimator, as scikit-learn calls it."""


def neg_crps(estimator: Any, X: Any, y: Any) -> float:
    """The mean continuous ranked probability score (CRPS) of a regressor's predictions of
    `y` at the rows `X`, with its sign flipped, so that higher is better, as scikit-learn's
    scorers are: a scorer for `cross_val_score(scoring=neg_crps)` or `GridSearchCV`.

    The CRPS of one row compares the whole predicted distribution of a new observation with
    the observed value: for a normal distribution it is in the target's units, it is the
    absolute error when the prediction has no spread, and lower is better. It rewards a
    mean close to the observation and a spread that is honest about how close. The
    estimator's `predict(X, return_std=True)` gives the mean and standard deviation of
    each row; any scikit-learn regressor that takes `return_std`, such as scikit-learn's own
    `GaussianProcessRegressor`, can be scored. The scores come from the scoringrules
    package.
    """
    mean, std = estimator.predict(X, return_std=True)
    observed = column_or_1d(y, dtype=np.float64)
    return -float(np.mean(scoringrules.crps_normal(observed, mean, std)))


def _refuse_sample_weight(sample_weight: Any) -> None:
    """Raise `ValueError` if `score` is given weights for its rows, which skgp does not
    take."""
    if sample_weight is not None:
        raise ValueError("score takes no sample_weight: every row counts the same.")


def _neg_brier_score(estimator: Any, X: Any, y: Any) -> float:
    """The Brier loss of a two-class classifier's probabilities of `y` at the rows `X`,
    with its sign flipped, as scikit-learn's `neg_brier_score` scorer gives it, but with the
    estimator's second class, `classes_[1]`, as the one whose probability is scored, so that
    classes of any type work, text included, where scikit-learn's scorer needs 0 and 1."""
    probability = estimator.predict_proba(X)[:, 1]
    observed = column_or_1d(y)
    return -float(
        brier_score_loss(observed, probability, pos_label=estimator.classes_[1])
    )


_SKGP_SCORERS: dict[str, Callable[[Any, Any, Any], float]] = {
    "neg_crps": neg_crps,
    "neg_brier_score": _neg_brier_score,
}
"""The scorers skgp gives itself, in place of scikit-learn's of the same name, if any."""


def _named_scorers(scoring: Scoring) -> dict[str, Callable[..., float]]:
    """Each scorer of `scoring`, by the name its value gets: a name stands for skgp's own
    scorer of that name (`neg_crps` and `neg_brier_score`) or scikit-learn's, and a scorer
    given itself is named after its function."""
    pairs: list[tuple[str, Any]]
    if isinstance(scoring, str):
        pairs = [(scoring, scoring)]
    elif isinstance(scoring, Mapping):
        pairs = [(str(name), given) for name, given in scoring.items()]
    elif callable(scoring):
        pairs = [(_name_of(scoring), scoring)]
    else:
        pairs = [(_name_of(given), given) for given in scoring]
    if not pairs:
        raise ValueError("scoring names no scorer; give at least one.")
    return {
        name: _SKGP_SCORERS.get(given) or get_scorer(given)
        if isinstance(given, str)
        else given
        for name, given in pairs
    }


def _name_of(given: Any) -> str:
    """The name the value of a scorer gets: the name given, or a scorer's function name."""
    return given if isinstance(given, str) else getattr(given, "__name__", repr(given))


class _ScoredRegressor(RegressorMixin, BaseEstimator):
    """A fitted regressor's prediction of held-out rows, as scikit-learn's scorers see an
    estimator: `predict` gives the prediction's mean, and its standard deviation with
    `return_std`, whatever `X` is, so the rows are checked and predicted once for every
    scorer."""

    def __init__(self, prediction: Prediction | None = None) -> None:
        self.prediction = prediction

    def predict(self, X: Any, return_std: bool = False) -> Any:
        assert self.prediction is not None
        if return_std:
            return self.prediction.mean(), self.prediction.std()
        return self.prediction.mean()


class _ScoredClassifier(ClassifierMixin, BaseEstimator):
    """A fitted classifier's prediction of held-out rows, as scikit-learn's scorers see an
    estimator, with the classes coded 0 and 1 as the model coded them, so that every
    scorer works whatever the classes are (scikit-learn's `f1` and `neg_brier_score`, for
    two, need 0 and 1 to know which class counts as the positive one). Like
    `_ScoredRegressor`, it gives the same prediction whatever `X` is."""

    classes_ = np.array([0, 1])

    def __init__(self, prediction: ClassPrediction | None = None) -> None:
        self.prediction = prediction

    def predict(self, X: Any) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: Any) -> np.ndarray:
        assert self.prediction is not None
        return self.prediction.probabilities()


def _scores(
    prediction: Prediction | ClassPrediction, observed: np.ndarray, scoring: Scoring
) -> dict[str, float]:
    """The value of each scorer of `scoring` for `prediction` of the `observed` targets
    (for a classifier, the classes coded 0 and 1), by the name its value gets."""
    scorers = _named_scorers(scoring)
    if isinstance(prediction, ClassPrediction):
        if any(scorer is neg_crps for scorer in scorers.values()):
            raise ValueError(
                "neg_crps scores a predicted distribution of numbers, which a classifier "
                "does not give; score it by neg_brier_score or neg_log_loss instead."
            )
        estimator: BaseEstimator = _ScoredClassifier(prediction)
    else:
        estimator = _ScoredRegressor(prediction)
    return {
        name: float(scorer(estimator, None, observed))
        for name, scorer in scorers.items()
    }


class _Estimator(BaseEstimator):
    """What every skgp estimator shares: turning `X` into a dataframe with named columns,
    describing the skgp model and fitting it. The settings are stored as given, and checked
    by the skgp model at `fit`, as scikit-learn asks."""

    _model_class: ClassVar[type[_SkgpModel]]

    iterations: int
    # scikit-learn's record of `X` at `fit`: how many columns, and their names when `X`
    # named them.
    n_features_in_: int
    feature_names_in_: np.ndarray
    # The fitted skgp model, a `FittedGPRegressor` or a `FittedGPClassifier`, after `fit`.
    fitted_: Any

    def _skgp_model(self, columns: list[str]) -> _SkgpModel:
        """The skgp model this estimator describes, with every setting given to it but
        `iterations`, which belongs to `fit`, and the inputs among `columns`: all but the
        `noise_column`, if there is one."""
        settings = {
            name: value
            for name, value in self.get_params(deep=False).items()
            if name != "iterations"
        }
        noise_column = settings.get("noise_column")
        features = [name for name in columns if name != noise_column]
        return self._model_class(features, TARGET, **settings)

    def _as_frame(self, X: Any, *, reset: bool) -> Frame:
        """`X` as a dataframe with named columns, after scikit-learn's check of its columns.

        In `fit` (with `reset`), a Polars or pandas dataframe keeps its column names, which
        scikit-learn records as `feature_names_in_`, and the columns of anything else, a
        numeric array, are called `x0`, `x1`, .... Later, `X` is checked to have as many
        columns as at `fit`, and the same names if both were named; scikit-learn warns when
        only one of them was, and the columns are then matched by position."""
        try:
            frame = _frames.to_frame(X)
        except TypeError:
            array = check_array(X, dtype=np.float64)
            validate_data(self, array, reset=reset, skip_check_array=True)
            names = self._column_names(array.shape[1])
            return _frames.to_frame(
                pl.DataFrame({name: array[:, i] for i, name in enumerate(names)})
            )
        validate_data(self, X, reset=reset, skip_check_array=True)
        named = all(isinstance(name, str) for name in frame.columns)
        if named and (reset or hasattr(self, "feature_names_in_")):
            return frame
        names = self._column_names(len(frame.columns))
        return frame.rename(dict(zip(frame.columns, names, strict=True)))

    def _column_names(self, n_columns: int) -> list[str]:
        """The names of the columns of an `X` that does not name them: those `X` had at
        `fit`, if it named them, and otherwise `x0`, `x1`, ...."""
        if hasattr(self, "feature_names_in_"):
            return [str(name) for name in self.feature_names_in_]
        return [f"x{i}" for i in range(n_columns)]

    def _fit_skgp_model(self, X: Any, y: Any) -> None:
        """Fit the skgp model to `X` and `y` and keep it as `fitted_`."""
        frame = self._as_frame(X, reset=True)
        if TARGET in frame.columns:
            raise ValueError(
                f"X has a column named '{TARGET}', the name an estimator gives the target. "
                "Rename that column."
            )
        target = nw.new_series(
            TARGET,
            self._target(y),
            backend=nw.get_native_namespace(frame),
        )
        if len(target) != len(frame):
            raise ValueError(
                f"X has {len(frame):,} rows and y has {len(target):,} values; they must be "
                "the same."
            )
        data = frame.with_columns(target)
        model = self._skgp_model(list(frame.columns))
        self.fitted_ = model.fit(_frames.to_native(data), iterations=self.iterations)

    def _target(self, y: Any) -> np.ndarray:
        """`y` as a one-dimensional numpy array."""
        raise NotImplementedError

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False  # skgp refuses missing values
        return tags

    def _predict_rows(self, X: Any, **arguments: Any) -> Prediction | ClassPrediction:
        """The fitted model's prediction at the rows `X`."""
        check_is_fitted(self, "fitted_")
        rows = _frames.to_native(self._as_frame(X, reset=False))
        return self.fitted_.predict(rows, **arguments)


class _Regressor(RegressorMixin, _Estimator):
    """What both regression estimators share: `predict` with scikit-learn's `return_std` and
    `return_cov`, `sample_y`, and `score` by the CRPS."""

    def fit(self, X: Any, y: Any) -> Self:
        """Fit to the inputs `X` and the target `y`, and return this estimator, fitted.

        `X` is a Polars or pandas dataframe, whose columns keep their names, or a numeric
        array, whose columns are called `x0`, `x1`, ...; `y` holds one number per row.
        The skgp model checks the data and the settings, and raises `ValueError` naming
        what is wrong. The fitted skgp model is `fitted_`.
        """
        self._fit_skgp_model(X, y)
        return self

    def _target(self, y: Any) -> np.ndarray:
        return column_or_1d(y, dtype=np.float64, warn=True)

    def predict(
        self,
        X: Any,
        return_std: bool = False,
        return_cov: bool = False,
        include_noise: bool = True,
    ) -> Any:
        """The predicted mean of each row of `X`, in the target's units, as a numpy array;
        with `return_std=True` also the standard deviation of each row, and with
        `return_cov=True` the covariance between the rows instead, as scikit-learn's
        `GaussianProcessRegressor.predict` gives them.

        Unlike scikit-learn's, the spread is of a new observation, measurement noise
        included, unless `include_noise=False`, which gives the spread of the underlying
        curve. `X` has the columns it had at `fit`, as scikit-learn requires, so a model
        with known noise has its `noise_column` there even for the curve alone, whose
        prediction does not use it.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std and return_cov can be True: the standard "
                "deviations are the square roots of the covariance's diagonal."
            )
        prediction = self._predict_rows(X, include_noise=include_noise)
        assert isinstance(prediction, Prediction)
        if return_std:
            return prediction.mean(), prediction.std()
        if return_cov:
            return prediction.mean(), prediction.covariance()
        return prediction.mean()

    def sample_y(
        self,
        X: Any,
        n_samples: int = 1,
        random_state: int = 0,
        include_noise: bool = True,
    ) -> np.ndarray:
        """Random samples of the prediction at the rows of `X`: a numpy array of a row for
        each row and a column for each sample, as scikit-learn's `sample_y` gives (see
        `skgp.Prediction.sample`). Each sample includes measurement noise unless
        `include_noise=False`. `random_state` is a whole number."""
        prediction = self._predict_rows(X, include_noise=include_noise)
        return prediction.sample(n_samples, random_state)

    def score(self, X: Any, y: Any, sample_weight: None = None) -> float:
        """The mean CRPS of the predictions of `y` at the rows `X`, with its sign flipped,
        so that higher is better (see `neg_crps`). It is what `cross_val_score` and
        `GridSearchCV` use when given no `scoring`, in place of the R² scikit-learn's
        regressors give. Rows cannot be weighted."""
        _refuse_sample_weight(sample_weight)
        return neg_crps(self, X, y)


class _Classifier(ClassifierMixin, _Estimator):
    """What both classification estimators share: `predict` and `predict_proba` as
    scikit-learn's classifiers give them, `sample_y`, and `score` by the Brier loss."""

    classes_: np.ndarray

    def fit(self, X: Any, y: Any) -> Self:
        """Fit to the inputs `X` and the classes `y`, and return this estimator, fitted.

        `X` is a Polars or pandas dataframe, whose columns keep their names, or a numeric
        array, whose columns are called `x0`, `x1`, ...; `y` holds one of two classes per
        row, of any type. The two classes, in sorted order, are `classes_`. The fitted skgp
        model is `fitted_`.
        """
        self._fit_skgp_model(X, y)
        self.classes_ = np.asarray(self.fitted_.classes)
        return self

    def _target(self, y: Any) -> np.ndarray:
        """`y` as a one-dimensional numpy array, after checking, as scikit-learn's
        classifiers do, that it holds classes, not continuous numbers, and two of them."""
        observed = column_or_1d(y, warn=True)
        check_classification_targets(observed)
        kind = type_of_target(observed, input_name="y", raise_unknown=True)
        if kind != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is "
                f"{kind}: y holds more than two classes, and a {type(self).__name__} "
                "needs exactly two."
            )
        return observed

    def predict(self, X: Any) -> np.ndarray:
        """The more likely class at each row of `X`, as a numpy array of the classes."""
        prediction = self._predict_rows(X)
        assert isinstance(prediction, ClassPrediction)
        return prediction.most_likely()

    def predict_proba(self, X: Any) -> np.ndarray:
        """The probability of each class at each row of `X`: a numpy array with a row for
        each row and a column for each class, in the order of `classes_`."""
        prediction = self._predict_rows(X)
        assert isinstance(prediction, ClassPrediction)
        return prediction.probabilities()

    def sample_y(self, X: Any, n_samples: int = 1, random_state: int = 0) -> np.ndarray:
        """Random samples of the class at each row of `X`: a numpy array of a row for each
        row and a column for each sample, holding the classes (see
        `skgp.ClassPrediction.sample`). scikit-learn's classifiers have no `sample_y`; skgp's
        have one so that every estimator samples the same way. `random_state` is a whole
        number."""
        prediction = self._predict_rows(X)
        return prediction.sample(n_samples, random_state)

    def score(self, X: Any, y: Any, sample_weight: None = None) -> float:
        """The Brier loss of the predicted probabilities of `y` at the rows `X`, with its
        sign flipped, so that higher is better, as scikit-learn's `neg_brier_score` scorer
        gives it: the mean, over the rows, of the squared difference between the
        probability of `classes_[1]` and 1 where it is the class, 0 where it is not. It is
        what `cross_val_score` and `GridSearchCV` use when given no `scoring`, in place of
        the accuracy scikit-learn's classifiers give. Rows cannot be weighted."""
        _refuse_sample_weight(sample_weight)
        return _neg_brier_score(self, X, y)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        if tags.classifier_tags is not None:
            tags.classifier_tags.multi_class = False
        return tags


class GaussianProcessRegressor(_Regressor):
    """skgp's `GPRegressor` as a scikit-learn estimator: an exact Gaussian process regression
    model, with skgp's settings.

    The inputs come from `X` and the target from `y`, so there are no `features` or
    `target`; and no `schema`, because a `Pipeline` has changed the data before it gets
    here. Every other setting is a `GPRegressor`'s, checked at `fit`:

    - `kernel`: `"matern52"` (the default), `"matern32"`, `"rbf"` or a GPJax kernel. Its
      length scales are in the units of the columns of `X`.
    - `mean_function`: what the curve is far from the training data: `"constant"` (the
      default), `"zero"`, `"linear"` or a GPJax mean function.
    - `learned_noise`: `"constant"` (the default), `"heteroscedastic"`, or `None`, which
      needs a `noise_column`.
    - `noise_column`: the column of `X` that holds each row's known noise, as a standard
      deviation in the target's units, or `None` (the default). It is not an input. For an
      array, name it as its column is named: `"x3"` for the fourth.
    - `iterations`: the most iterations the fit may take, 2,000 by default.

    After `fit`: `fitted_` is the fitted skgp model (`print(estimator.fitted_)`,
    `estimator.fitted_.to_json()`), `n_features_in_` is the number of columns of `X`, and
    `feature_names_in_` their names when `X` named them.
    """

    _model_class = GPRegressor

    def __init__(
        self,
        *,
        kernel: Kernel = "matern52",
        mean_function: MeanFunction = "constant",
        learned_noise: LearnedNoise | None = "constant",
        noise_column: str | None = None,
        iterations: int = 2000,
    ) -> None:
        self.kernel = kernel
        self.mean_function = mean_function
        self.learned_noise = learned_noise
        self.noise_column = noise_column
        self.iterations = iterations


class SparseGaussianProcessRegressor(_Regressor):
    """skgp's `SparseGPRegressor` as a scikit-learn estimator: a sparse Gaussian process
    regression model, for data with too many rows for an exact one, with skgp's settings.

    There are no `features`, `target` or `schema` (see `GaussianProcessRegressor`), and no
    `noise_column`, which a `SparseGPRegressor` does not take. The settings, checked at
    `fit`, are a `SparseGPRegressor`'s:

    - `kernel` and `mean_function`: as for `GaussianProcessRegressor`.
    - `learned_noise`: `"constant"` (the default) or `"heteroscedastic"`.
    - `num_inducing`: how many inducing points summarise the data, or `None` (the default)
      to let skgp choose.
    - `minibatch_size`: the rows each iteration looks at, or `None` (the default) for all.
    - `random_state`: the seed of everything random in the fit, 0 by default.
    - `iterations`: 2,000 by default: the most iterations a fit on all rows may take, and
      exactly the number a fit on minibatches runs.

    After `fit`, `fitted_`, `n_features_in_` and `feature_names_in_` are as for
    `GaussianProcessRegressor`.
    """

    _model_class = SparseGPRegressor

    def __init__(
        self,
        *,
        kernel: Kernel = "matern52",
        mean_function: MeanFunction = "constant",
        learned_noise: LearnedNoise = "constant",
        num_inducing: int | None = None,
        minibatch_size: int | None = None,
        random_state: int = 0,
        iterations: int = 2000,
    ) -> None:
        self.kernel = kernel
        self.mean_function = mean_function
        self.learned_noise = learned_noise
        self.num_inducing = num_inducing
        self.minibatch_size = minibatch_size
        self.random_state = random_state
        self.iterations = iterations


class GaussianProcessClassifier(_Classifier):
    """skgp's `GPClassifier` as a scikit-learn estimator: Gaussian process classification of
    two classes, with skgp's settings.

    There are no `features`, `target` or `schema` (see `GaussianProcessRegressor`). The
    settings, checked at `fit`, are a `GPClassifier`'s:

    - `kernel` and `mean_function`: as for `GaussianProcessRegressor`.
    - `iterations`: the most iterations the fit may take, 2,000 by default.

    After `fit`: `classes_` holds the two classes in sorted order, `fitted_` is the fitted
    skgp model, and `n_features_in_` and `feature_names_in_` are as for
    `GaussianProcessRegressor`. More than two classes are not supported yet.
    """

    _model_class = GPClassifier

    def __init__(
        self,
        *,
        kernel: Kernel = "matern52",
        mean_function: MeanFunction = "constant",
        iterations: int = 2000,
    ) -> None:
        self.kernel = kernel
        self.mean_function = mean_function
        self.iterations = iterations


class SparseGaussianProcessClassifier(_Classifier):
    """skgp's `SparseGPClassifier` as a scikit-learn estimator: Gaussian process
    classification of two classes for data with too many rows for a
    `GaussianProcessClassifier`, with skgp's settings.

    There are no `features`, `target` or `schema` (see `GaussianProcessRegressor`). The
    settings, checked at `fit`, are a `SparseGPClassifier`'s: `kernel`, `mean_function`, `num_inducing`,
    `minibatch_size`, `random_state` and `iterations`, as for
    `SparseGaussianProcessRegressor`.

    After `fit`, `classes_`, `fitted_`, `n_features_in_` and `feature_names_in_` are as for
    `GaussianProcessClassifier`.
    """

    _model_class = SparseGPClassifier

    def __init__(
        self,
        *,
        kernel: Kernel = "matern52",
        mean_function: MeanFunction = "constant",
        num_inducing: int | None = None,
        minibatch_size: int | None = None,
        random_state: int = 0,
        iterations: int = 2000,
    ) -> None:
        self.kernel = kernel
        self.mean_function = mean_function
        self.num_inducing = num_inducing
        self.minibatch_size = minibatch_size
        self.random_state = random_state
        self.iterations = iterations
