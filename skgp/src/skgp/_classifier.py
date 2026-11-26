"""The classification models: `GPClassifier`, `SparseGPClassifier`, and the
`FittedGPClassifier` that fitting either of them returns."""

from collections.abc import Iterable
from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING, Any, ClassVar

from narwhals.stable.v2.typing import IntoDataFrame
import numpy as np

from skgp import _fit_kinds, _frames, _gp
from skgp._checked import CheckedData
from skgp._fit_kinds import FitKind
from skgp._log import Reporter
from skgp._model import _Fitted, _Model, _SparseModel
from skgp._prediction import ClassPrediction
from skgp._score import Score
from skgp._text import listed
from skgp._types import Frame, Kernel, Likelihood, MeanFunction, Scoring, Task


class _Classifier(_Model):
    """What every kind of classifier description shares. A `FittedGPClassifier` accepts any
    of them."""

    task: ClassVar[Task] = "classification"

    @property
    def likelihood(self) -> Likelihood:
        return "Bernoulli"

    def fit(
        self,
        data: IntoDataFrame | CheckedData,
        *,
        iterations: int = 2000,
        verbose: bool = False,
    ) -> "FittedGPClassifier":
        """Fit to `data`, returning a separate fitted model.

        `data` is a Polars, pandas or other dataframe, which is checked first, or a
        `CheckedData` from the `check` of this model or of any classifier with the same
        features, target and schema, which is not checked again. Its target column must
        hold exactly two different values. `self` is unchanged.

        The two values are coded 0 and 1 in sorted order: the smaller number, false, or
        the text that comes first alphabetically is 0. `iterations` is a whole number
        above zero. For a `GPClassifier`, or a `SparseGPClassifier` without a
        `minibatch_size`, it is the most iterations the optimiser may take; it stops
        sooner once the fit has finished. For a `SparseGPClassifier` with a
        `minibatch_size`, it is exactly the number of iterations run, each on one
        minibatch.

        With `verbose=True`, skgp emits log records through `loguru`, each a short
        message with named fields (in `record["extra"]`), so you choose where they go:
        among them, which of your values was coded 0 and which 1. A long fit reports its
        progress every few seconds, except a `SparseGPClassifier` with a
        `minibatch_size`, which shows GPJax's progress bar on standard error instead.
        With the default `False` it emits nothing.

        If a fit on all rows at once does not finish properly (the same condition
        `to_json()` reports as `finished_properly: false`), `fit` issues one
        `skgp.ConvergenceWarning`, a `UserWarning`, whatever `verbose` is, and still
        returns the fitted model, which may be less reliable. Filter or escalate it with
        the `warnings` module. If the fit used all its iterations, the warning suggests a
        larger `iterations`. Fits that finish properly are silent. A `SparseGPClassifier`
        with a `minibatch_size` runs exactly its `iterations`, and skgp does not judge
        whether it converged, so it never issues this warning.
        """
        return self._fit(FittedGPClassifier, data, iterations, verbose)

    def _prepared_target(
        self, frame: Frame, report: Reporter
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
        # The checks have made sure that the target holds exactly two values. Classes are
        # not rescaled.
        first, second = _frames.distinct_values(frame, self.target)
        report.info("classes coded", target=self.target, class_0=first, class_1=second)
        return (
            _frames.is_value(frame, self.target, second),
            {"classes": (first, second)},
            {},
        )


class GPClassifier(_Classifier):
    """A description of a Gaussian process classifier, for a target with two classes.

    It names the dataframe columns that are inputs, the column that is the target, the
    kernel, the mean function, and optionally a `schema` that checks your data first (see the `schema`
    setting). The target may hold any two different values: numbers, text, or true and
    false. The inputs and the target may be given by position:
    `GPClassifier(["latitude", "altitude"], "rain")`. A mistake in any setting raises
    `ValueError` when the model is defined. It never changes: fitting it returns a
    separate fitted model, and `replace` returns a new model with other settings.

    The model is a smooth curve over the inputs, and the probability of the second class
    at a row is the probit (the standard normal distribution function) of the curve
    there. It is fitted by variational inference with an inducing point at every training
    row, so it gets slow and needs a lot of memory as rows grow, as an exact regressor
    does. When `fit` is given more rows than `skgp.config.size_warning_rows` (10,000
    unless you change it), it issues one `skgp.LargeDataWarning` naming
    `SparseGPClassifier`, then fits as usual.
    """

    # The settings, for editors and type checkers. At run time `_Model.__init__` takes
    # them, passing on only those given, so the others keep their defaults.
    if TYPE_CHECKING:

        def __init__(
            self,
            features: str | Iterable[str],
            target: str,
            *,
            kernel: Kernel = ...,
            mean_function: MeanFunction = ...,
            schema: Any = ...,
        ) -> None: ...

    @property
    def fit_kind(self) -> FitKind[Any]:
        return _fit_kinds.CLASSIFICATION


class SparseGPClassifier(_Classifier, _SparseModel):
    """A description of a sparse Gaussian process classifier, for data with too many rows
    for a `GPClassifier`.

    It takes the same inputs, target, kernel, mean function and schema as a `GPClassifier`,
    and fitting it
    returns the same kind of fitted model. The data is summarised by `num_inducing` points,
    so the model is an approximation: it is never chosen for you. It never changes: fitting
    it returns a separate fitted model.

    Without a `minibatch_size` the fit looks at all the rows at once, with the optimiser a
    `GPClassifier` uses. With one, it works through the rows in random minibatches, which
    suits data too large to process at once. The inducing points start at the centres of a
    k-means clustering of the rescaled inputs and are tuned during the fit together with the
    other settings. The same data and settings always give the same model.
    """

    # The settings, for editors and type checkers. At run time `_Model.__init__` takes
    # them, passing on only those given, so the others keep their defaults.
    if TYPE_CHECKING:

        def __init__(
            self,
            features: str | Iterable[str],
            target: str,
            *,
            kernel: Kernel = ...,
            mean_function: MeanFunction = ...,
            schema: Any = ...,
            minibatch_size: int | np.integer | None = ...,
            num_inducing: int | np.integer | None = ...,
            random_state: int | np.integer = ...,
        ) -> None: ...

    @property
    def fit_kind(self) -> FitKind[Any]:
        return (
            _fit_kinds.SPARSE_CLASSIFICATION_ON_ALL_ROWS
            if self.minibatch_size is None
            else _fit_kinds.SPARSE_ON_MINIBATCHES
        )


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class FittedGPClassifier(_Fitted):
    """A model fitted to data by a `GPClassifier` or a `SparseGPClassifier`. It never
    changes."""

    model: "GPClassifier | SparseGPClassifier"
    """The model description that was fitted, with every setting it was fitted with."""

    classes: tuple[Any, Any]
    """The two values of the target column: the one coded 0 first, then the one coded 1."""

    def predict(
        self,
        data: IntoDataFrame | CheckedData,
        *,
        suffix: str = "",
        batch_size: int = 1_000,
        verbose: bool = False,
    ) -> ClassPrediction:
        """What the model expects at each row of `data`, as a `ClassPrediction`: the
        probability of each class, and the more likely one.

        `data` needs the feature columns but no target. It is a dataframe, which is
        checked first, or a `CheckedData` from the `check` of this model or of any
        classifier with the same features, target and schema (for training or
        prediction data), which is not checked again. Columns the model does not use are
        ignored, and a verbose log record names them. Every column of `data` comes back
        in `ClassPrediction.to_frame()`, in the same kind of dataframe as `data`; a
        missing input column raises.

        - `suffix`: text added to the names of the columns skgp adds (the probability
          of each class and `most_likely_class`), for rows that already have a column
          with one of those names. Empty by default.
        - `batch_size`: how many rows are predicted at a time, a whole number above
          zero. Memory grows with it; the answers do not change.

        A row far from the training data, where the model knows little, gets
        probabilities nearer 0.5. No probability is below 0.001 or above 0.999.

        With `verbose=True`, skgp emits log records through `loguru`, each a short
        message with named fields (in `record["extra"]`). With the default `False` it
        emits nothing.
        """
        report, frame, x_new = self._rows_to_predict(
            data, verbose=verbose, suffix=suffix, batch_size=batch_size
        )
        # A new observation is a class, so its mean is the probability of the class coded 1.
        probability_of_second, _ = self._fitted_gp.mean_and_variance(
            x_new, batch_size=int(batch_size), include_noise=True
        )
        report.info("prediction made", n_rows=len(probability_of_second))
        return ClassPrediction(
            frame,
            suffix,
            self.classes,
            probability_of_second,
            functools.partial(self._curve_mean_and_covariance, x_new, int(batch_size)),
            _gp.probability_given_curve,
        )

    def _curve_mean_and_covariance(
        self, x_new: np.ndarray, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The mean and the joint covariance of the underlying curve at the rescaled rows
        `x_new`, which a `ClassPrediction` samples from."""
        mean, _ = self._fitted_gp.mean_and_variance(
            x_new, batch_size=batch_size, include_noise=False
        )
        return mean, self._fitted_gp.covariance(x_new, include_noise=False)

    def score(
        self,
        data: IntoDataFrame | CheckedData,
        *,
        scoring: Scoring = "neg_brier_score",
    ) -> Score:
        """How well the model predicts the known classes of held-out `data`, as a `Score`:
        the value of each scorer, where higher is better.

        `data` holds the feature columns and the target, a class the model knows at every
        row. It is a dataframe, which is checked first, or a `CheckedData` of this model,
        which is not checked again but for the target. Other columns are ignored, as in
        `predict`.

        `scoring` takes what `cross_val_score(scoring=...)` takes: a scikit-learn scorer's
        name, such as `"accuracy"`, `"neg_log_loss"`, `"roc_auc"` or `"f1"`
        (`sklearn.metrics.get_scorer_names()` lists them), a scorer itself, a list of
        either, or a dict from the name each value gets to its scorer. The default,
        `"neg_brier_score"`, is the Brier loss with its sign flipped: the mean squared
        difference between the probability of `classes[1]` and 1 where it is the class,
        0 where it is not, from 0 (perfect) to 1, no unit. The scorers see the classes
        coded 0 and 1, as the model coded them, so `classes[1]` is the positive class of
        `f1` and the like, whatever the classes are. An unknown name raises
        scikit-learn's `ValueError`.
        """
        return self._score(data, scoring)

    def _observed(self, target: np.ndarray) -> np.ndarray:
        return (target == self.classes[1]).astype(int)

    def _require_known_classes(self, frame: Frame) -> None:
        unknown = sorted(
            {value for value in frame[self.model.target].to_list()} - set(self.classes),
            key=repr,
        )
        if unknown:
            first, second = self.classes
            raise ValueError(
                f"Column '{self.model.target}' holds {listed([repr(v) for v in unknown])}, "
                f"which the model was not fitted to; it knows only {first!r} and "
                f"{second!r}."
            )

    def _target_shown(self) -> str:
        first, second = self.classes
        return f"{self.model.target} (classes {first!r} and {second!r})"

    def _target_section(self) -> dict[str, Any]:
        return {"classes": list(self.classes)}

    def to_json(self) -> str:
        """A read-only JSON description of the fitted model, for inspection and audit.

        It cannot be used to rebuild the model. The output is always strict JSON: any
        number that is not finite (NaN or infinite, for example the `final_objective`
        of a fit that diverged) is written as `null`, so it round-trips through any
        standard JSON parser. The keys are:

        - `skgp_version`: the version of skgp that fitted the model. A model saved
          with pickle and loaded elsewhere keeps the version that fitted it, which may
          differ from the installed one.
        - `training_rows`: how many rows of data the model was fitted to.
        - `setup`: what you asked for. `features` (the input columns, in order),
          `date_features` (only when there are date inputs: each of them with its
          unit, `"days since 1970-01-01"`), `target` (the target column),
          `kernel`: its name, or for a GPJax kernel
          its class, with a sum or product of kernels written out, such as
          `"RBF + Periodic"`, and `mean_function`: its name, or for a GPJax mean function its class, written
          out as a kernel's is.
        - `classes`: the two values of the target column, the one coded 0 first.
        - `learned`: what fitting found out about the data.

          - `kernel`: every value of the fitted kernel, by its place in the kernel, as
            GPJax names it: `lengthscale` and `variance` for a kernel by name, and
            `kernels[1].period` for the period of the second kernel of a sum. A value
            with one number per input is given for each input column, by its name; a
            single number for all of them is one number. A length scale is how far you
            have to move along an input, in the column's own units, before the
            probabilities change appreciably: larger means smoother along it. A period
            is in the column's units too; the length scale of a `Periodic` kernel
            measures a fraction of the period instead, and has no units. The variance is that of the curve whose
            probit is the probability of the second class, which has no units: larger
            means the probabilities can come nearer 0 and 1.
          - `mean_function`: only for a mean function with values, so not for
            `"zero"`. Its values are on the scale of the same curve, with no
            units: `constant` for `"constant"`; for `"linear"`, `intercept`, the curve
            where every input is 0, and `slope`, for each input column, by its name,
            how much the curve rises per unit of the column; for a GPJax mean
            function, every value of it, by its place in it, as for `kernel`.
        - `fitting`: how the search for the learned values ended.

          - `finished_properly`: only for a fit on all rows. True if the fit finished,
            false if it did not, in which case the learned values may be poor. It is
            judged from where the fit ended, not from what the optimiser reported: true
            means further steps would no longer change the fit by much, whether the
            optimiser stopped by itself or used all its `iterations`. A
            `SparseGPClassifier` fitted on minibatches runs every one of its
            `iterations`, and skgp does not judge whether it converged.
          - `iterations`: how many iterations the fit took: at most the `iterations`
            given to `fit` for a fit on all rows, and exactly that many for a fit on
            minibatches, where an iteration looks at one minibatch.
          - `final_objective`: how well the fit explains the classes at the end, as a
            number where lower is better: the negative of a lower bound on the log
            marginal likelihood. For a fit on minibatches it is scaled to all the rows
            and averaged over the iterations of the last pass, so it is noisy. It is
            useful for comparing fits of the same data and settings.
          - `stop_reason`: only for a fit on all rows. Why the fit stopped, as a
            sentence in plain words, the same one the `ConvergenceWarning` gives.
        - `model`: only for a model made by a `SparseGPClassifier`.

          - `class`: the name of the description that made the model,
            `"SparseGPClassifier"`.
          - `num_inducing`: how many inducing points the fit used (never more than the
            number of rows).
          - `minibatch_size`: the minibatch size, or null for the fit on all rows at
            once.
          - `random_state`: the whole number that seeded the fit.
        """
        return self._json_text()
