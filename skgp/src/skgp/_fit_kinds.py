"""The ways a model is fitted, and what each one needs from the rest of skgp.

A regressor with constant noise is fitted in one of three ways: an exact fit (`GPRegressor`),
a sparse fit on all rows at once, by the collapsed bound (`SparseGPRegressor` without a
`minibatch_size`), and a sparse fit on minibatches (`SparseGPRegressor` with one). A regressor
with heteroscedastic noise is fitted in one of three as well: on all rows with an inducing
point at every row (`GPRegressor`), a sparse fit on all rows at once (`SparseGPRegressor`
without a `minibatch_size`), and a sparse fit on minibatches by Adam alone
(`SparseGPRegressor` with one). A classifier is fitted in one of three too: on all rows with
an inducing point at every row (`GPClassifier`), a sparse fit on all rows at once
(`SparseGPClassifier` without a `minibatch_size`), and the same sparse fit on minibatches as a
regressor's with constant noise (`SparseGPClassifier` with one). Fitting, the `model chosen`
log record, the warning when there are many rows and the warning when a fit did not finish all
differ between them, so each question is answered here once, by the kind, instead of by a test
of the settings at every place it comes up. `model.fit_kind` gives the kind of a model.
"""

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from skgp import _gp, _how_a_fit_ended, _inducing
from skgp._constants import Constant as C
from skgp._log import Reporter
from skgp._text import count
from skgp._types import OnIteration

if TYPE_CHECKING:
    from skgp._classifier import GPClassifier, SparseGPClassifier
    from skgp._model import _Model, _SparseModel
    from skgp._regressor import GPRegressor, SparseGPRegressor


class FitKind[M: _Model]:
    """One way of fitting a model. Its methods take the model whose settings it uses."""

    did_not_finish_message = staticmethod(_how_a_fit_ended.warning_for_a_regressor)
    """What the `ConvergenceWarning` says when a fit of this kind did not finish, given how
    it ended. The default is for a regressor. A fit on minibatches is not judged, so it
    never warns (decision record 0008)."""

    def description(self, model: M) -> str:
        """What kind of model is fitted, as the `model chosen` log record says it."""
        raise NotImplementedError

    def large_data_message(self, rows: int) -> str | None:
        """What the `LargeDataWarning` says when the training data has `rows` rows, more than
        `skgp.config.size_warning_rows`; None for a kind that never warns."""
        return None

    def fit(
        self,
        model: M,
        prior: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        iterations: int,
        report: Reporter,
        known_noise_variance: np.ndarray | None = None,
    ) -> tuple[_gp.FittedGP, _how_a_fit_ended.Optimisation]:
        """Fit the `prior`, from `_gp.prior_to_fit`, to the rescaled data, with
        the `iterations` given to `fit`, reporting the optimiser chosen. Returns the fitted
        Gaussian process (a `_gp.FittedGP`) and how the search ended.

        `known_noise_variance` is the variance of each training row's known noise, rescaled as the
        target was, or None for a model with no `noise_column`. Only an exact fit takes it:
        the settings refuse a `noise_column` on every other kind of model."""
        raise NotImplementedError


class _Exact(FitKind["GPRegressor"]):
    def description(self, model: "GPRegressor") -> str:
        return "exact Gaussian process"

    def large_data_message(self, rows: int) -> str:
        return (
            f"{count(rows, 'row')} is a lot for an exact model, which gets slow and needs a lot of "
            "memory as rows grow. Consider SparseGPRegressor, which takes the same settings. "
            "The fit carries on."
        )

    def fit(
        self,
        model: "GPRegressor",
        prior: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        iterations: int,
        report: Reporter,
        known_noise_variance: np.ndarray | None = None,
    ) -> tuple[_gp.FittedGP, _how_a_fit_ended.Optimisation]:
        on_iteration = _start_optimiser(report, "marginal likelihood", iterations)
        return _gp.fit_exact(
            prior,
            x,
            y,
            known_noise_variance=known_noise_variance,
            learns_noise=model.learned_noise is not None,
            max_iterations=iterations,
            on_iteration=on_iteration,
        )


class _SparseRegressionOnAllRows(FitKind["SparseGPRegressor"]):
    def description(self, model: "SparseGPRegressor") -> str:
        return "sparse Gaussian process (fitted on all rows at once)"

    def fit(
        self,
        model: "SparseGPRegressor",
        prior: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        iterations: int,
        report: Reporter,
        known_noise_variance: np.ndarray | None = None,
    ) -> tuple[_gp.FittedGP, _how_a_fit_ended.Optimisation]:
        inducing = _start_inducing_points(model, x, report)
        on_iteration = _start_optimiser(
            report, "approximate marginal likelihood", iterations
        )
        return _gp.fit_sparse_regression_on_all_rows(
            prior,
            x,
            y,
            inducing,
            max_iterations=iterations,
            on_iteration=on_iteration,
        )


class _SparseOnMinibatches(FitKind["_SparseModel"]):
    """A sparse fit on minibatches, of a regressor or of a classifier, with the model's own
    likelihood."""

    optimiser = C.OPTIMISER_ON_MINIBATCHES
    """The optimiser, as the `optimiser chosen` record names it."""

    def description(self, model: "_SparseModel") -> str:
        assert model.minibatch_size is not None  # this kind is for fits on minibatches
        rows = count(model.minibatch_size, "row")
        return f"sparse Gaussian process (fitted on minibatches of {rows})"

    def fit(
        self,
        model: "_SparseModel",
        prior: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        iterations: int,
        report: Reporter,
        known_noise_variance: np.ndarray | None = None,
    ) -> tuple[_gp.FittedGP, _how_a_fit_ended.Optimisation]:
        assert model.minibatch_size is not None  # this kind is for fits on minibatches
        inducing = _start_inducing_points(model, x, report)
        report.info(
            "optimiser chosen",
            optimiser=self.optimiser,
            objective="approximate marginal likelihood",
            iterations=iterations,
        )
        return _gp.fit_sparse_on_minibatches(
            prior,
            x,
            y,
            inducing,
            model.minibatch_size,
            iterations=iterations,
            likelihood=model.likelihood,
            random_state=model.random_state,
            show_progress=report.verbose,
        )


class _HeteroscedasticSparseOnMinibatches(_SparseOnMinibatches):
    """A `SparseGPRegressor` with heteroscedastic noise fitted on minibatches."""

    optimiser = C.OPTIMISER_ON_MINIBATCHES_WITH_HETEROSCEDASTIC_NOISE


class _ClassificationOnAllRows[M: _Model](FitKind[M]):
    """A classifier's fit on all rows at once, by the optimiser the exact fit uses."""

    did_not_finish_message = staticmethod(_how_a_fit_ended.warning_for_a_classifier)


class _Classification(_ClassificationOnAllRows["GPClassifier"]):
    def description(self, model: "GPClassifier") -> str:
        return "Gaussian process (variational, with an inducing point at every row)"

    def large_data_message(self, rows: int) -> str:
        return (
            f"{count(rows, 'row')} is a lot for a GPClassifier, which gets slow and needs a lot of "
            "memory as rows grow. Consider SparseGPClassifier, which takes the same "
            "settings. The fit carries on."
        )

    def fit(
        self,
        model: "GPClassifier",
        prior: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        iterations: int,
        report: Reporter,
        known_noise_variance: np.ndarray | None = None,
    ) -> tuple[_gp.FittedGP, _how_a_fit_ended.Optimisation]:
        on_iteration = _start_optimiser(
            report, "approximate marginal likelihood", iterations
        )
        return _gp.fit_classifier_on_all_rows(
            prior,
            x,
            y,
            max_iterations=iterations,
            on_iteration=on_iteration,
        )


class _SparseClassificationOnAllRows(_ClassificationOnAllRows["SparseGPClassifier"]):
    def description(self, model: "SparseGPClassifier") -> str:
        return "sparse Gaussian process (fitted on all rows at once)"

    def fit(
        self,
        model: "SparseGPClassifier",
        prior: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        iterations: int,
        report: Reporter,
        known_noise_variance: np.ndarray | None = None,
    ) -> tuple[_gp.FittedGP, _how_a_fit_ended.Optimisation]:
        inducing = _start_inducing_points(model, x, report)
        on_iteration = _start_optimiser(
            report, "approximate marginal likelihood", iterations
        )
        return _gp.fit_classifier_on_all_rows(
            prior,
            x,
            y,
            max_iterations=iterations,
            on_iteration=on_iteration,
            inducing_inputs=inducing,
        )


class _HeteroscedasticAtEveryRow(FitKind["GPRegressor"]):
    """A `GPRegressor` with heteroscedastic noise, which has no exact fit: variational, with
    an inducing point at every row for the curve and for the noise, as a `GPClassifier`."""

    def description(self, model: "GPRegressor") -> str:
        return "Gaussian process (variational, with an inducing point at every row)"

    def large_data_message(self, rows: int) -> str:
        return (
            f"{count(rows, 'row')} is a lot for a GPRegressor with heteroscedastic noise, "
            "which gets slow and needs a lot of memory as rows grow. Consider "
            "SparseGPRegressor, which takes the same settings. The fit carries on."
        )

    def fit(
        self,
        model: "GPRegressor",
        prior: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        iterations: int,
        report: Reporter,
        known_noise_variance: np.ndarray | None = None,
    ) -> tuple[_gp.FittedGP, _how_a_fit_ended.Optimisation]:
        on_iteration = _start_optimiser(
            report, "approximate marginal likelihood", iterations
        )
        return _gp.fit_heteroscedastic_on_all_rows(
            prior, x, y, max_iterations=iterations, on_iteration=on_iteration
        )


class _HeteroscedasticSparseOnAllRows(FitKind["SparseGPRegressor"]):
    """A `SparseGPRegressor` with heteroscedastic noise fitted on all rows at once."""

    def description(self, model: "SparseGPRegressor") -> str:
        return "sparse Gaussian process (fitted on all rows at once)"

    def fit(
        self,
        model: "SparseGPRegressor",
        prior: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        iterations: int,
        report: Reporter,
        known_noise_variance: np.ndarray | None = None,
    ) -> tuple[_gp.FittedGP, _how_a_fit_ended.Optimisation]:
        inducing = _start_inducing_points(model, x, report)
        on_iteration = _start_optimiser(
            report, "approximate marginal likelihood", iterations
        )
        return _gp.fit_heteroscedastic_on_all_rows(
            prior,
            x,
            y,
            max_iterations=iterations,
            on_iteration=on_iteration,
            inducing_inputs=inducing,
        )


EXACT = _Exact()
HETEROSCEDASTIC_AT_EVERY_ROW = _HeteroscedasticAtEveryRow()
HETEROSCEDASTIC_SPARSE_ON_ALL_ROWS = _HeteroscedasticSparseOnAllRows()
HETEROSCEDASTIC_SPARSE_ON_MINIBATCHES = _HeteroscedasticSparseOnMinibatches()
SPARSE_REGRESSION_ON_ALL_ROWS = _SparseRegressionOnAllRows()
SPARSE_ON_MINIBATCHES = _SparseOnMinibatches()
CLASSIFICATION = _Classification()
SPARSE_CLASSIFICATION_ON_ALL_ROWS = _SparseClassificationOnAllRows()


def _start_inducing_points(
    model: "_SparseModel", x: np.ndarray, report: Reporter
) -> np.ndarray:
    """Where the inducing points of a sparse fit start, logging how they are chosen before
    the search for them, which can take a while."""
    num_inducing = (
        _inducing.default_count(len(x))
        if model.num_inducing is None
        else model.num_inducing
    )
    report.info("random state", random_state=model.random_state)
    subset_rows = _inducing.k_means_subset_rows(len(x), num_inducing)
    report.info(
        "inducing points chosen",
        count=min(num_inducing, len(x)),
        is_default=model.num_inducing is None,
        starting="k-means centres of the rescaled inputs"
        if subset_rows is None
        else f"k-means centres of a random subset of {subset_rows:,} of the rescaled input rows",
    )
    return _inducing.k_means_centres(x, num_inducing, model.random_state)


def _start_optimiser(report: Reporter, objective: str, iterations: int) -> OnIteration:
    """Report the optimiser that fits on all rows at once, which takes at most `iterations`
    iterations on the `objective`, and return what to call after each of its iterations: it
    reports the iteration and the objective, at most once every `C.PROGRESS_SECONDS`
    seconds."""
    report.info(
        "optimiser chosen",
        optimiser=C.OPTIMISER,
        objective=objective,
        max_iterations=iterations,
    )
    last = time.monotonic()

    def on_iteration(iteration: int, value: float) -> None:
        nonlocal last
        if time.monotonic() - last >= C.PROGRESS_SECONDS:
            last = time.monotonic()
            report.info("fit in progress", iteration=iteration, objective=value)

    return on_iteration
