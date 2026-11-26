from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw

from skgp import _frames
from skgp._constants import Constant as C
from skgp._log import Reporter
from skgp._text import count, joined, listed, quoted
from skgp._types import Frame

if TYPE_CHECKING:
    from skgp._model import _Model


def _what_decides_the_checks(model: "_Model") -> dict[str, Any]:
    """The settings that decide what the checks are, so the only ones a `CheckedData` is tied
    to: the columns, the schema and the column of known noise, if any. Whether the model is
    a regressor or a classifier decides them too (`task`)."""
    return {
        "features": model.features,
        "target": model.target,
        "schema": model.schema,
        "noise_column": model._known_noise_column(),
    }


@dataclass(frozen=True, eq=False)
class CheckedData:
    """A dataframe that has passed every data check of a model (a GPRegressor, a
    SparseGPRegressor, a GPClassifier or a SparseGPClassifier).

    Create it with `check(data)` on the model, or with
    `check(data, for_prediction=True)` for rows to predict, which
    need no target column. Rows to predict may have a target column (for
    example a held-out set with its true values); it is ignored. Any column that
    is neither an input, the target nor the column of known noise is ignored too.
    Passing it to `fit` or `predict` skips the checks, so checking once and using the result several times costs one check.

    It is valid for any model of the same task (regression or classification) with the same
    features, target, schema and column of known noise (`noise_column`, if any) as the model
    that made it, whatever its class, kernel or other settings, because only those decide
    what the checks are. A regressor checks that
    the target holds numbers and a classifier that it holds two classes, so data checked by
    one is refused by the other. Any other model refuses it with an error that names what
    differs. Data checked by a model with a different schema (or none) has not passed this
    model's schema, so check it again with this model's `check`.

    Training data (the default) must have at least 5 rows.

    It is an immutable snapshot of the data as it was when it was checked: the
    checked data is copied (or, for an immutable dataframe such as Polars,
    shared, which is free), so changing your own dataframe afterwards never
    changes what `fit` or `predict` sees from a `CheckedData`.
    """

    model: "_Model"
    """The model whose checks the data passed."""
    _frame: Frame
    _for_prediction: bool

    def require_valid_for(self, model: "_Model", *, for_prediction: bool) -> None:
        """Raise `ValueError` unless `model` may use this data for the given purpose."""
        if self.model.task != model.task:
            raise ValueError(
                f"This data was checked by a {type(self.model).__name__}, which checks it for "
                f"{self.model.task}, and this model is a {type(model).__name__}, which "
                f"checks it for {model.task}. Run check again on this model: model.check(df)."
            )
        theirs, ours = (
            _what_decides_the_checks(self.model),
            _what_decides_the_checks(model),
        )
        differing = [name for name in theirs if theirs[name] != ours[name]]
        if differing:
            raise ValueError(_refusal(differing))
        if self._for_prediction and not for_prediction:
            raise ValueError(
                "This CheckedData was made with for_prediction=True, so it has no checked "
                "target column and cannot be used to fit. Call check() without "
                "for_prediction to check training data."
            )


def _refusal(differing: list[str]) -> str:
    """Why a model cannot take data checked by one whose `differing` settings are not its own."""
    verb = "differs" if len(differing) == 1 and differing[0] != "features" else "differ"
    message = f"This data was checked by a model whose {joined(differing)} {verb} from this model's."
    if "schema" in differing:
        message += (
            " It was checked against a different schema (or with none), so it has not "
            "passed this model's schema."
        )
    return f"{message} Run check again on this model: model.check(df)."


def check(
    model: "_Model", frame: Frame, *, for_prediction: bool, report: Reporter
) -> CheckedData:
    """Run every data check of `model`, in order, on `frame`.

    Training data needs the target column and at least `C.MIN_TRAINING_ROWS`
    rows; a classifier's target must hold exactly two classes. Inputs hold numbers, or
    dates, or dates and times. Rows to predict
    need only the feature columns, and a target column among them is ignored.
    Any column that is neither an input, the target nor the column of known noise is
    ignored too.

    `report` receives the checks that ran and whether they passed, and the
    columns that were ignored.
    """
    try:
        frame, checks = _run_checks(model, frame, for_prediction=for_prediction)
    except Exception as error:
        report.warning("data checks failed", passed=False, error=str(error))
        raise
    report.info(
        "data checks passed",
        checks=checks,
        n_rows=len(frame),
        passed=True,
    )
    used = {*model.features, model.target, model._known_noise_column()}
    ignored = [name for name in frame.columns if name not in used]
    if ignored:
        report.info("columns ignored", columns=ignored)
    return CheckedData(model, _frames.snapshot(frame), for_prediction)


def _run_checks(
    model: "_Model", frame: Frame, *, for_prediction: bool
) -> tuple[Frame, list[str]]:
    """`frame` as the checks saw it (your schema may return a new one) and the names of the
    checks that ran, in order; or the first problem found, raised."""
    checks = []
    if model.schema is not None:
        frame = _check_user_schema(model, frame)
        checks.append("schema")

    needed = model.features if for_prediction else (*model.features, model.target)
    # Rows to predict need the known noise only for a prediction with noise, which `predict`
    # asks for itself; here it is checked whenever it is there.
    noise_column = model._known_noise_column()
    if noise_column is not None and (
        not for_prediction or noise_column in frame.columns
    ):
        needed = (*needed, noise_column)
    # A classifier's target holds classes, which may be text or true/false, not numbers.
    classifying = model.task == "classification"
    numbers = model.features if classifying else needed

    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise ValueError(missing_columns_message(missing))
    checks.append("columns_present")

    n_rows = count(len(frame), "row")
    if not for_prediction:
        if len(frame) < C.MIN_TRAINING_ROWS:
            raise ValueError(
                f"The data has {n_rows}, but at least {C.MIN_TRAINING_ROWS} rows "
                "are needed to fit a model."
            )
        checks.append("enough_rows")

    _check_numeric(model, frame, numbers)
    checks.append("numeric")

    counts = _frames.count_missing_and_infinite(frame, needed)
    _refuse_missing_values(counts, len(frame))
    checks.append("no_missing_values")
    _refuse_infinite_values(counts, len(frame))
    checks.append("no_infinite_values")

    if noise_column is not None and noise_column in needed:
        _check_known_noise_not_negative(frame, noise_column)
        checks.append("no_negative_known_noise")

    if classifying and not for_prediction:
        classes = _frames.distinct_values(frame, model.target)
        if len(classes) != C.CLASSIFIER_CLASSES:
            raise ValueError(_not_two_classes(model, classes))
        checks.append("two_classes")
    return frame, checks


def check_known_targets(model: "_Model", frame: Frame) -> None:
    """Raise `ValueError` unless `frame`, already checked as rows to predict, has the
    target column with a known value at every row, a number for a regressor, as scoring
    predictions against it needs."""
    target = model.target
    if target not in frame.columns:
        raise ValueError(missing_columns_message([target]))
    if model.task == "regression":
        _check_numeric(model, frame, (target,))
    counts = _frames.count_missing_and_infinite(frame, (target,))
    _refuse_missing_values(counts, len(frame))
    _refuse_infinite_values(counts, len(frame))


def _refuse_missing_values(counts: dict[str, tuple[int, int]], rows: int) -> None:
    """Raise `ValueError` for the first column that `counts`, from
    `_frames.count_missing_and_infinite` on `rows` rows, says has a missing value."""
    for name, (n_missing, _) in counts.items():
        if n_missing:
            raise ValueError(
                f"Column '{name}' has {count(n_missing, 'missing value')} (empty or NaN) "
                f"out of {count(rows, 'row')}. Remove or fill them first."
            )


def _refuse_infinite_values(counts: dict[str, tuple[int, int]], rows: int) -> None:
    """Raise `ValueError` for the first column that `counts`, from
    `_frames.count_missing_and_infinite` on `rows` rows, says has an infinite value."""
    for name, (_, n_infinite) in counts.items():
        if n_infinite:
            raise ValueError(
                f"Column '{name}' has {count(n_infinite, 'infinite value')} "
                f"out of {count(rows, 'row')}. Remove or replace them first."
            )


def missing_columns_message(missing: list[str]) -> str:
    """What the error says when the data lacks the `missing` columns."""
    return (
        f"The data is missing {count(len(missing), 'column')} the model needs: "
        f"{quoted(missing)}."
    )


def _check_known_noise_not_negative(frame: Frame, column: str) -> None:
    """Raise `ValueError` if the column of known noise has a value below zero."""
    n_negative = _frames.count_below_zero(frame, column)
    if n_negative:
        raise ValueError(
            f"Column '{column}' has {count(n_negative, 'negative value')} out of "
            f"{count(len(frame), 'row')}. It holds the known noise on each observation, a "
            "standard deviation, which cannot be below zero."
        )


def _check_numeric(model: "_Model", frame: Frame, columns: tuple[str, ...]) -> None:
    """Raise `ValueError` for the first of `columns` that does not hold numbers. An input
    may hold dates, or dates and times, instead: a date input, which the model counts in
    days since 1970-01-01."""
    for name in columns:
        dtype = frame.schema[name]
        if _frames.holds_dates_or_times(frame, name):
            if name in model.features:
                continue
            raise ValueError(
                f"Column '{name}' holds dates, not numbers. Only inputs may hold dates; "
                "a regressor's target must be numbers."
            )
        if isinstance(dtype, nw.Duration):
            raise ValueError(
                f"Column '{name}' holds durations, not numbers. Convert it to numbers "
                "first, for example a number of hours."
            )
        if dtype.is_temporal():
            raise ValueError(
                f"Column '{name}' holds times of day, not numbers. Convert it to numbers "
                "first, for example the number of hours since midnight."
            )
        if not dtype.is_numeric():
            raise ValueError(
                f"Column '{name}' is not numeric (it holds {dtype} values). "
                + (
                    f"A {type(model).__name__}'s inputs must be numbers; its target may "
                    "hold any two values."
                    if model.task == "classification"
                    else "Inputs, the target and the known noise must be numbers."
                )
            )


def _not_two_classes(model: "_Model", classes: list[Any]) -> str:
    """Why a classifier cannot fit a target holding `classes`, which are not two."""
    target, name = model.target, type(model).__name__
    if len(classes) == 1:
        return (
            f"Column '{target}' holds only one value, {classes[0]!r}; a {name} needs "
            "exactly two different values in its target, one for each class."
        )
    shown = listed([repr(value) for value in classes])
    return (
        f"Column '{target}' holds {len(classes):,} different values ({shown}); a "
        f"{name} needs exactly two, one for each class. More than two classes are not "
        "supported yet."
    )


def checked(
    model: "_Model", data: Any, *, for_prediction: bool, report: Reporter
) -> CheckedData:
    """`data` as `CheckedData` for `model`: already-checked data is only verified to suit
    this model, never checked again; a raw dataframe is checked now.

    `report` receives a record of what happened: that the data was already checked, or the
    checks that ran and whether they passed.
    """
    if isinstance(data, CheckedData):
        data.require_valid_for(model, for_prediction=for_prediction)
        report.info("data already checked", n_rows=len(data._frame))
        return data
    return check(
        model,
        _frames.to_frame(data),
        for_prediction=for_prediction,
        report=report,
    )


def _check_user_schema(model: "_Model", frame: Frame) -> Frame:
    """The user's own schema runs here, before every built-in check.

    It gets the frame in the caller's own type. Whatever it raises propagates.
    A frame it returns replaces `frame`; returning None keeps `frame`.
    """
    returned = model.schema.validate(_frames.to_native(frame))
    return frame if returned is None else _frames.to_frame(returned)
