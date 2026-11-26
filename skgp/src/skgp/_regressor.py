"""The regression models: `GPRegressor`, `SparseGPRegressor`, and the `FittedGPRegressor`
that fitting either of them returns."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

from narwhals.stable.v2.typing import IntoDataFrame
import numpy as np
from pydantic import field_validator, model_validator

from skgp import _fit_kinds, _frames
from skgp._checked import CheckedData, missing_columns_message
from skgp._fit_kinds import FitKind
from skgp._log import Reporter
from skgp._model import _Fitted, _Model, _SparseModel
from skgp._prediction import Prediction
from skgp._scaling import Standardiser
from skgp._score import Score
from skgp._types import (
    Frame,
    Kernel,
    LearnedNoise,
    Likelihood,
    MeanFunction,
    Scoring,
    Task,
)


class _Regressor(_Model):
    """What every kind of regressor description shares. A `FittedGPRegressor` accepts any of
    them."""

    task: ClassVar[Task] = "regression"

    learned_noise: LearnedNoise | None = "constant"
    """The noise on each observation that the model learns from the data, and how it varies.

    - `"constant"` (the default): the same at every row, one number learned from the data.
      With a `GPRegressor`'s `noise_column`, it is added to each row's known noise: the noise variance at a
      row is its known noise squared plus the learned noise squared.
    - `"heteroscedastic"`: varying with the inputs, as where a measurement gets noisier as
      it gets larger. The logarithm of the noise variance is a second Gaussian process,
      the noise process, with a `"matern52"` kernel of its own, the noise kernel, learned
      from the data together with the curve. GPJax has no exact fit of this model, so a
      `GPRegressor` fits it by variational inference with an inducing point at every row
      for the curve and for the noise process, as a `GPClassifier` is fitted, and a
      `SparseGPRegressor` with its inducing points for each of the two. It cannot be
      combined with a `noise_column`.
    - `None`: no learned noise; the noise is the known noise in a `GPRegressor`'s
      `noise_column` alone, which must then be given.

    A prediction's spread includes the noise at each row, unless you ask for the curve
    alone with `include_noise=False`."""

    @property
    def likelihood(self) -> Likelihood:
        return (
            "HeteroscedasticGaussian"
            if self.learned_noise == "heteroscedastic"
            else "Gaussian"
        )

    def fit(
        self,
        data: IntoDataFrame | CheckedData,
        *,
        iterations: int = 2000,
        verbose: bool = False,
    ) -> "FittedGPRegressor":
        """Fit to `data`, returning a separate fitted model.

        `data` is a Polars, pandas or other dataframe, which is checked first,
        or a `CheckedData` from the `check` of this model or of any regressor with the
        same features, target, schema and column of known noise, which is not checked
        again. `self` is unchanged.

        `iterations` is a whole number above zero. For a `GPRegressor`, or a
        `SparseGPRegressor` without a `minibatch_size`, it is the most iterations the
        optimiser may take; it stops sooner once the fit has finished. For a
        `SparseGPRegressor` with a `minibatch_size`, it is exactly the number of
        iterations run, each on one minibatch.

        With `verbose=True`, skgp emits log records through `loguru`, each a short
        message with named fields (in `record["extra"]`), so you choose where they
        go. A long fit reports its progress every few seconds, except a
        `SparseGPRegressor` with a `minibatch_size`, which shows GPJax's progress
        bar on standard error instead. With the default `False` it emits nothing.

        If a fit on all rows at once does not finish properly (the same condition
        `to_json()` reports as `finished_properly: false`), `fit` issues one
        `skgp.ConvergenceWarning`, a `UserWarning`, whatever `verbose` is, and
        still returns the fitted model, which may be less reliable. Filter or
        escalate it with the `warnings` module. The warning says what to check: the
        scale of your data and whether it is noise-free, and for a fit that used all
        its iterations without settling, a larger `iterations`. Fits that finish
        properly are silent. A `SparseGPRegressor` with a `minibatch_size` runs
        exactly its `iterations`, and skgp does not judge whether it converged, so
        it never issues this warning.
        """
        return self._fit(FittedGPRegressor, data, iterations, verbose)

    def _target_scaler(self, target_fields: dict[str, Any]) -> Standardiser | None:
        return target_fields["_y_scaler"]

    def _prepared_target(
        self, frame: Frame, report: Reporter
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
        raw_y = _frames.columns_to_array(frame, (self.target,))
        y_scaler = Standardiser.fit(raw_y)
        return (
            y_scaler.apply(raw_y)[:, 0],
            {"_y_scaler": y_scaler},
            _target_rescaling(y_scaler),
        )


class GPRegressor(_Regressor):
    """A description of a Gaussian process regression model that summarises nothing: exact
    with constant or known noise.

    It names the dataframe columns that are inputs, the column that is the
    target, the kernel, the mean function, the noise it learns, optionally a column of
    known noise, and
    optionally a `schema` that checks your data first (see the `schema` setting). With
    constant learned noise (the default), known noise, or both, the fit is exact; with
    heteroscedastic noise it is variational, with an inducing point at every row (see the
    `learned_noise` setting). The inputs and the target may be given by position:
    `GPRegressor(["latitude", "altitude"], "temperature")`. A mistake in any setting
    raises `ValueError` when the model is defined. It never changes: fitting it returns
    a separate fitted model, and `replace` returns a new model with other settings.

    It gets slow and needs a lot of memory as rows grow, with either noise. When `fit` is
    given more rows than `skgp.config.size_warning_rows` (10,000 unless you change
    it), it issues one `skgp.LargeDataWarning` naming `SparseGPRegressor`, then fits
    as usual. The limit only decides whether you are warned; it never changes the
    model fitted.
    """

    noise_column: str | None = None
    """The column of your dataframe that holds the known noise on each observation, such
    as an instrument's stated error: a standard deviation, in the target's units, for every
    row. Zero is allowed; a negative or missing value is refused. The model learns the
    rest of the noise, as `learned_noise` says, or none.

    It goes with `learned_noise` `"constant"` or `None`. A `SparseGPRegressor` takes no
    known noise: GPJax's sparse fits take the same noise at every row. Rows to predict need the column when the
    prediction includes noise (`include_noise=True`, the default), because a new
    observation has its own known noise; without noise they do not. It is neither an
    input nor the target. Leave it unset (the default) when no noise is known."""

    @field_validator("noise_column")
    @classmethod
    def _noise_column_named(cls, noise_column: str | None) -> str | None:
        if noise_column == "":
            raise ValueError(
                "noise_column is an empty column name ''; give the name of the column "
                "that holds the known noise, or leave it unset."
            )
        return noise_column

    @model_validator(mode="after")
    def _known_noise_fits_the_other_settings(self) -> Self:
        column = self.noise_column
        if column is None and self.learned_noise is None:
            raise ValueError(
                "learned_noise is None and there is no noise_column, so the model would "
                "have no noise at all. Give noise_column, the column that holds the known "
                'noise, or let the model learn the noise with learned_noise="constant".'
            )
        if column is not None and self.learned_noise == "heteroscedastic":
            raise ValueError(
                f"noise_column '{column}' cannot be combined with "
                'learned_noise="heteroscedastic". Use learned_noise="constant" to learn '
                "noise on top of the known noise, or None to use the known noise alone."
            )
        if column is not None and column in self.features:
            raise ValueError(
                f"noise_column '{column}' is also in features; a column cannot be both an "
                "input and the known noise."
            )
        if column is not None and column == self.target:
            raise ValueError(
                f"noise_column '{column}' is also the target; a column cannot be both "
                "the target and the known noise."
            )
        return self

    def _known_noise_column(self) -> str | None:
        return self.noise_column

    def _known_noise_variance(
        self, frame: Frame, target_fields: dict[str, Any]
    ) -> np.ndarray | None:
        if self.noise_column is None:
            return None
        return _known_noise_variance(
            frame, self.noise_column, target_fields["_y_scaler"]
        )

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
            learned_noise: LearnedNoise | None = ...,
            noise_column: str | None = ...,
            schema: Any = ...,
        ) -> None: ...

    @property
    def fit_kind(self) -> FitKind[Any]:
        if self.learned_noise == "heteroscedastic":
            return _fit_kinds.HETEROSCEDASTIC_AT_EVERY_ROW
        return _fit_kinds.EXACT


# `_SparseModel` first, so that `learned_noise` comes after `kernel` in the settings, not
# after `random_state`.
class SparseGPRegressor(_SparseModel, _Regressor):
    """A description of a sparse Gaussian process regression model, for data with too many
    rows for an exact one.

    It takes the same inputs, target, kernel, mean function, learned noise and schema as a
    `GPRegressor`,
    but no known noise, and fitting it returns the same kind of fitted model. The data is summarised by
    `num_inducing` points, so the model is an approximation: it is never chosen for
    you. Noise is Gaussian. It never changes: fitting it returns a separate fitted model.

    Without a `minibatch_size` the fit looks at all the rows at once. With one, it works
    through the rows in random minibatches, which suits data too large to process at
    once. The inducing points start at the centres of a k-means clustering of the
    rescaled inputs and are tuned during the fit together with the other settings. The
    same data and settings always give the same model.
    """

    learned_noise: LearnedNoise = "constant"
    """The noise on each observation that the model learns from the data, and how it
    varies: `"constant"` (the default), the same at every row, or `"heteroscedastic"`,
    varying with the inputs (see `GPRegressor`'s `learned_noise`). A `SparseGPRegressor`
    takes no known noise, so it always learns its noise."""

    settings_not_taken: ClassVar[dict[str, str]] = {
        "noise_column": (
            "GPJax's sparse fits take the same noise at every row. Use a GPRegressor for "
            "known noise."
        )
    }

    @property
    def fit_kind(self) -> FitKind[Any]:
        if self.learned_noise == "heteroscedastic":
            if self.minibatch_size is None:
                return _fit_kinds.HETEROSCEDASTIC_SPARSE_ON_ALL_ROWS
            return _fit_kinds.HETEROSCEDASTIC_SPARSE_ON_MINIBATCHES
        if self.minibatch_size is None:
            return _fit_kinds.SPARSE_REGRESSION_ON_ALL_ROWS
        return _fit_kinds.SPARSE_ON_MINIBATCHES

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
            learned_noise: LearnedNoise = ...,
            schema: Any = ...,
            minibatch_size: int | np.integer | None = ...,
            num_inducing: int | np.integer | None = ...,
            random_state: int | np.integer = ...,
        ) -> None: ...


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class FittedGPRegressor(_Fitted):
    """A model fitted to data by a `GPRegressor` or a `SparseGPRegressor`. It never changes."""

    model: "GPRegressor | SparseGPRegressor"
    """The model description that was fitted, with every setting it was fitted with."""

    # The shift and scale used on the target before fitting.
    _y_scaler: Standardiser

    def predict(
        self,
        data: IntoDataFrame | CheckedData,
        *,
        include_noise: bool = True,
        suffix: str = "",
        batch_size: int = 1_000,
        verbose: bool = False,
    ) -> Prediction:
        """What the model expects at each row of `data`, as a `Prediction`.

        `data` needs the feature columns but no target. It is a dataframe,
        which is checked first, or a `CheckedData` from the `check` of this model
        or of any regressor with the same features, target, schema and column of
        known noise (for training or prediction data), which is not checked again. Columns the model does
        not use are ignored, and a verbose log record names them. Every column of
        `data` comes back in `Prediction.to_frame()`, in the same kind of
        dataframe as `data`; a missing input column raises.

        - `include_noise`: True (the default) predicts a new observation, so the
          spread includes measurement noise: with heteroscedastic noise, the noise the
          model expects at that row; with known noise, each row's own known noise, from
          the `noise_column` of `data`, which must then have it, plus any learned noise.
          False predicts the underlying curve without the noise: the same mean, with a
          smaller spread, and needs no `noise_column`.
        - `suffix`: text added to the names of the columns skgp adds (`mean`, `std`,
          `lower` and `upper`), for rows that already have a column with one of
          those names. With `suffix="_predicted"` they are `mean_predicted` and so
          on. Empty by default.
        - `batch_size`: how many rows are predicted at a time, a whole number above
          zero. Memory grows with it; the answers do not change.

        The first prediction prepares the training data and keeps the result on
        the fitted model, so later predictions, and `covariance()` of any of them,
        reuse it. For a `GPRegressor` it takes memory that grows with the square of
        the training rows (about 72 MB at 3,000 rows and 800 MB at 10,000, twice as
        much with heteroscedastic noise).

        With `verbose=True`, skgp emits log records through `loguru`, each a short
        message with named fields (in `record["extra"]`). With the default `False`
        it emits nothing.
        """
        report, frame, x_new = self._rows_to_predict(
            data, verbose=verbose, suffix=suffix, batch_size=batch_size
        )
        known_noise_variance = self._known_noise_variance_of_rows(frame, include_noise)
        scaled_mean, scaled_variance = self._fitted_gp.mean_and_variance(
            x_new,
            batch_size=int(batch_size),
            include_noise=include_noise,
            known_noise_variance=known_noise_variance,
        )
        mean = self._y_scaler.undo(scaled_mean.reshape(-1, 1))[:, 0]
        variance = self._y_scaler.undo_variance(scaled_variance.reshape(-1, 1))[:, 0]
        report.info("prediction made", n_rows=int(mean.shape[0]))

        def joint_covariance() -> np.ndarray:
            scaled_covariance = self._fitted_gp.covariance(
                x_new,
                include_noise=include_noise,
                known_noise_variance=known_noise_variance,
            )
            return self._y_scaler.undo_variance(scaled_covariance)

        return Prediction(
            frame, suffix, bool(include_noise), mean, variance, joint_covariance
        )

    def score(
        self,
        data: IntoDataFrame | CheckedData,
        *,
        scoring: Scoring = "neg_crps",
    ) -> Score:
        """How well the model predicts the known targets of held-out `data`, as a `Score`:
        the value of each scorer, where higher is better.

        `data` holds the feature columns and the target, a known number at every row, and,
        for a model with known noise, its `noise_column`. It is a dataframe, which is
        checked first, or a `CheckedData` of this model, which is not checked again but for
        the target. Other columns are ignored, as in `predict`.

        `scoring` takes what `cross_val_score(scoring=...)` takes: a scikit-learn scorer's
        name, such as `"r2"` or `"neg_mean_absolute_error"`
        (`sklearn.metrics.get_scorer_names()` lists them), a scorer itself, a list of
        either, or a dict from the name each value gets to its scorer. The default,
        `"neg_crps"`, is skgp's: the mean continuous ranked probability score (CRPS) with
        its sign flipped (see `skgp.sklearn.neg_crps`), in the target's units. An unknown
        name raises scikit-learn's `ValueError`.

        The prediction scored is of new observations, noise included, because the targets
        are observations. A scorer that needs a point prediction, such as `r2`, gets the
        mean.
        """
        return self._score(data, scoring)

    def _known_noise_variance_of_rows(
        self, frame: Frame, include_noise: bool
    ) -> np.ndarray | None:
        """The variance of the known noise of each row to predict, rescaled as the target
        was, when the prediction includes noise and the model has a `noise_column`; None
        otherwise. Raises `ValueError` when the rows need the column and lack it."""
        column = self.model._known_noise_column()
        if column is None or not include_noise:
            return None
        if column not in frame.columns:
            raise ValueError(
                f"{missing_columns_message([column])} It holds the known noise on each "
                "new observation, which a prediction with noise adds. Predict with "
                "include_noise=False for the curve alone, without it."
            )
        return _known_noise_variance(frame, column, self._y_scaler)

    def _target_rescaling_fields(self) -> dict[str, float]:
        return _target_rescaling(self._y_scaler)

    def _learned(self) -> dict[str, Any]:
        """What fitting learned, with the learned noise, if any, in the target's units: the
        `learned` section of `to_json()` (which describes it), also shown by the printed
        model and the `fit finished` log record."""
        learned = super()._learned()
        if "noise" not in learned:
            return learned
        return {**learned, "noise": learned["noise"] * float(self._y_scaler.scale[0])}

    def _setup_after_kernel(self) -> dict[str, Any]:
        column = self.model._known_noise_column()
        known_noise = {} if column is None else {"noise_column": column}
        return {"learned_noise": self.model.learned_noise, **known_noise}

    def to_json(self) -> str:
        """A read-only JSON description of the fitted model, for inspection and audit.

        It cannot be used to rebuild the model. The output is always strict
        JSON: any number that is not finite (NaN or infinite, for example the
        `final_objective` of a fit that diverged) is written as `null`, so it
        round-trips through any standard JSON parser. The keys are:

        - `skgp_version`: the version of skgp that fitted the model. A model
          saved with pickle and loaded elsewhere keeps the version that fitted
          it, which may differ from the installed one.
        - `training_rows`: how many rows of data the model was fitted to.
        - `setup`: what you asked for. `features` (the input columns, in
          order), `date_features` (only when there are date inputs: each of
          them with its unit, `"days since 1970-01-01"`), `target` (the target
          column), `kernel`: its name, or for a
          GPJax kernel its class, with a sum or product of kernels written out,
          such as `"RBF + Periodic"`, `mean_function`: its name, or for a GPJax mean function its class, written
          out as a kernel's is, `learned_noise`: `"constant"`,
          `"heteroscedastic"` or null, and `noise_column` (only for a model with
          known noise: the column that holds it).
        - `learned`: what fitting found out about the data.

          - `kernel`: every value of the fitted kernel, by its place in the
            kernel, as GPJax names it: `lengthscale` and `variance` for a kernel
            by name, and `kernels[1].period` for the period of the second kernel
            of a sum. A value with one number per input is given for each input
            column, by its name; a single number for all of them is one number.
            A length scale is how far you have to move along an input, in the
            column's own units, before the curve changes appreciably: larger
            means a smoother curve along it. A period is in the column's units
            too; the length scale of a `Periodic` kernel measures a fraction of
            the period instead, and has no units. A variance is relative to the
            target's variance: 1 means the curve varies as much as the target
            does. A value with other units, such as the variance of a `Linear`
            kernel, is in the column's units for the input and relative to the
            target's variance for the curve.
          - `mean_function`: only for a mean function with values, so not for
            `"zero"`. For `"constant"`, `constant`: the curve far from the
            data, in the target's units. For `"linear"`, `intercept`: the curve where
            every input is 0, in the target's units (at 1970-01-01 along a date
            input), and `slope`: for each input column, by its name, how much the
            curve rises per unit of the column (per day along a date input), in the
            target's units. For a GPJax mean function, every value of it, by its place
            in it, as for `kernel`, in the units it sees: the columns' and the
            target's.
          - `noise`: only for a model that learns noise. The standard deviation of
            the learned noise on each observation, in the target's units (how noisy
            the data looks). With known noise, it is the noise the fit learned on top
            of each row's known noise. With heteroscedastic
            noise, it is the median noise far from the training data, where the
            noise process (the logarithm of the noise variance) has its learned
            mean: the exponential of half that mean, times the target's standard
            deviation. A prediction there adds more noise than this, because it
            also counts how unsure the noise process is.
          - `noise_kernel`: only with heteroscedastic noise. Every value of the
            noise kernel, the `"matern52"` kernel of the noise process, named as in
            `kernel`. A length scale is how far you have to move along an input, in
            the column's own units, before the noise changes appreciably. The
            variance is that of the noise process, which has no units: larger
            means the noise varies more.
        - `fitting`: how the search for the learned values ended.

          - `finished_properly`: only for a model fitted on all rows at once. True if
            the fit finished, false if it did not, in which case the learned values
            may be poor. It is judged from where the fit ended, not from what the
            optimiser reported: true means further steps would no longer change the
            fit by much, whether the optimiser stopped by itself or used all its
            `iterations`. A sparse model fitted on minibatches runs every one of its
            `iterations`, and skgp does not judge whether it converged.
          - `iterations`: how many iterations the fit took: at most the `iterations`
            given to `fit` for a model fitted on all rows at once, and exactly that
            many for a sparse model fitted on minibatches. There an
            iteration looks at one minibatch, and a pass through the data is as many
            iterations as it takes to see every row once on average.
          - `final_objective`: how well the fit explains the rescaled data at the
            end, as a number where lower is better. For an exact model it is the
            negative log marginal likelihood. For a sparse model fitted on all rows,
            or any model with heteroscedastic noise, it is the negative of a lower
            bound on that likelihood, so a little above what an exact fit would give.
            For a sparse model fitted on minibatches it is the negative of the same kind of bound, scaled to all
            the rows and averaged over the iterations of the last pass; each one sees
            only its minibatch, so it is noisy. It is on the rescaled data, so it is
            useful for comparing fits of the same data and settings, not as a value
            in your units.
          - `stop_reason`: only for a model fitted on all rows at once. Why the fit
            stopped, as a sentence in plain words, the same one the
            `ConvergenceWarning` gives. The optimiser's own text is only in the
            verbose log, in the `optimiser_message` field of `fit finished`.
        - `model`: only for a model made by a `SparseGPRegressor`.

          - `class`: `"SparseGPRegressor"`.
          - `num_inducing`: how many inducing points the fit used (never more
            than the number of rows).
          - `minibatch_size`: the minibatch size, or null for the fit on all rows
            at once.
          - `random_state`: the whole number that seeded the fit.
        """
        return self._json_text()


def _known_noise_variance(
    frame: Frame, column: str, y_scaler: Standardiser
) -> np.ndarray:
    """The variance of the known noise in `column` at each row of `frame`, rescaled as the
    target was by `y_scaler`: the standard deviation divided by the target's scale,
    squared."""
    standard_deviation = _frames.columns_to_array(frame, (column,))[:, 0]
    return (standard_deviation / y_scaler.scale[0]) ** 2


def _target_rescaling(y_scaler: Standardiser) -> dict[str, float]:
    """The target's part of the `rescaling chosen` and `rescaling applied` log records."""
    return {
        "target_shift": float(y_scaler.shift[0]),
        "target_scale": float(y_scaler.scale[0]),
    }
