"""The data checks, and what `CheckedData` is valid for, for both classes of regressor."""

from datetime import date, time, timedelta
import re
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from helpers import Recorder, both_libraries, good_frame
from skgp import GPRegressor, SparseGPRegressor


def model():
    return GPRegressor(features=["x"], target="y")


def sparse(**settings):
    return SparseGPRegressor(features=["x"], target="y", num_inducing=4, **settings)


@pytest.fixture(params=[model, sparse], ids=["exact", "sparse"])
def either_model(request):
    """An unfitted model of each class: the data checks are the same for both."""
    return request.param()


# --- what the checks refuse, and what they say ---


def test_check_refuses_too_few_rows_and_says_the_minimum(either_model):
    with pytest.raises(ValueError, match=r"4 rows.*at least 5"):
        either_model.check(good_frame(n=4))

    either_model.check(good_frame(n=5))  # exactly the minimum is enough


def test_check_names_every_missing_column_and_prediction_data_needs_no_target(
    either_model,
):
    features_z_and_w = either_model.replace(features=["x", "z", "w"])
    with pytest.raises(ValueError, match=r"missing 2 columns.*: 'z' and 'w'"):
        features_z_and_w.check(good_frame())
    with pytest.raises(ValueError, match=r"missing 1 column.*: 'y'"):
        either_model.check(good_frame().select("x"))
    with pytest.raises(ValueError, match=r"missing 1 column.*: 'x'"):
        either_model.check(pl.DataFrame({"other": [1.0, 2.0]}), for_prediction=True)

    either_model.check(
        pl.DataFrame({"x": [1.0]}), for_prediction=True
    )  # no target and one row: fine


@pytest.mark.parametrize("bad_column", ["x", "y"])
def test_check_names_a_column_that_is_not_numeric_and_refuses_true_false_columns(
    either_model, bad_column
):
    with pytest.raises(ValueError, match=rf"'{bad_column}'.*not numeric"):
        either_model.check(good_frame().with_columns(pl.Series(bad_column, ["a"] * 10)))
    with pytest.raises(ValueError, match=rf"'{bad_column}'.*not numeric"):
        either_model.check(
            good_frame().with_columns(pl.Series(bad_column, [True, False] * 5))
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            [timedelta(hours=hours) for hours in range(10)],
            (
                "Column 'x' holds durations, not numbers. Convert it to numbers first, "
                "for example a number of hours."
            ),
        ),
        (
            [time(hour) for hour in range(10)],
            (
                "Column 'x' holds times of day, not numbers. Convert it to numbers first, "
                "for example the number of hours since midnight."
            ),
        ),
    ],
    ids=["duration", "time of day"],
)
def test_a_duration_or_time_of_day_input_is_refused_saying_how_to_make_it_numbers(
    values, message
):
    with pytest.raises(ValueError, match=re.escape(message)):
        model().check(good_frame().with_columns(pl.Series("x", values)))


def test_a_target_of_dates_is_refused_because_only_inputs_may_hold_dates():
    days = [date(2026, 1, day) for day in range(1, 11)]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Column 'y' holds dates, not numbers. Only inputs may hold dates; a "
            "regressor's target must be numbers."
        ),
    ):
        model().check(good_frame().with_columns(pl.Series("y", days)))


@pytest.mark.parametrize("bad_column", ["x", "y"])
def test_check_counts_missing_values_per_column_whether_null_or_nan_and_in_whole_number_columns(
    either_model, bad_column
):
    good = [float(i) for i in range(10)]
    values = [1.0, float("nan"), 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    frames = both_libraries({"x": good, "y": good, bad_column: values})
    whole = [1, None, 3, None, 5, 6, 7, 8, 9, 10]
    other = {"x": list(range(10)), "y": list(range(10))}
    frames += [
        pl.DataFrame(
            {**other, bad_column: whole}, schema={"x": pl.Int64, "y": pl.Int64}
        ),
        pd.DataFrame(
            {
                name: pd.array(column, dtype="Int64")
                for name, column in {**other, bad_column: whole}.items()
            }
        ),
    ]

    for frame in frames:
        with pytest.raises(ValueError, match=rf"'{bad_column}'.*2 missing"):
            either_model.check(frame)


@pytest.mark.parametrize("bad_column", ["x", "y"])
def test_check_counts_infinite_values_per_column(either_model, bad_column):
    columns = {"x": [float(i) for i in range(10)], "y": [float(i) for i in range(10)]}
    columns[bad_column] = [float("inf"), float("-inf"), float("inf")] + [1.0] * 7

    for frame in both_libraries(columns):
        with pytest.raises(ValueError, match=rf"'{bad_column}'.*3 infinite"):
            either_model.check(frame)


def test_the_first_problem_reported_is_the_first_in_the_order_of_the_checks():
    # columns present, enough rows, numeric, no missing values, no infinite values
    cases = [
        (
            pl.DataFrame({"x": [1.0, 2.0]}),
            r"missing.*'y'",
        ),  # a missing column before too few rows
        (
            good_frame(n=4).with_columns(pl.Series("x", ["a"] * 4)),
            r"4 rows.*at least 5",
        ),
        (pl.DataFrame({"x": ["a"] * 10, "y": [None] + [1.0] * 9}), r"'x'.*not numeric"),
        (
            pl.DataFrame({"x": [float("inf")] + [1.0] * 9, "y": [None] + [1.0] * 9}),
            r"'y'.*1 missing",
        ),
    ]
    for frame, message in cases:
        with pytest.raises(ValueError, match=message):
            model().check(frame)


def test_messages_use_the_singular_for_a_count_of_one():
    with pytest.raises(
        ValueError, match=r"1 missing value \(empty or NaN\) out of 2 rows"
    ):
        model().check(
            pl.DataFrame({"x": [1.0, None], "y": [1.0, 2.0]}), for_prediction=True
        )
    with pytest.raises(ValueError, match=r"1 infinite value out of 2 rows"):
        model().check(pl.DataFrame({"x": [1.0, float("inf")]}), for_prediction=True)


@pytest.mark.parametrize(
    ("not_an_eager_dataframe", "type_named"),
    [
        (pl.LazyFrame({"x": [1.0, 2.0]}), "LazyFrame"),
        ({"x": [1.0, 2.0]}, "dict"),
        (np.zeros((2, 1)), "numpy.ndarray"),
        ([1.0, 2.0], "list"),
    ],
    ids=["lazy-frame", "dict", "numpy-array", "list"],
)
def test_anything_but_an_eager_dataframe_is_refused_with_a_type_error_naming_what_was_given(
    sine_fit, not_an_eager_dataframe, type_named
):
    for call in (model().check, model().fit, sine_fit.predict):
        with pytest.raises(TypeError, match=type_named):
            call(not_an_eager_dataframe)


def test_fit_and_predict_run_the_checks_on_a_raw_dataframe(sine_fit):
    with pytest.raises(ValueError, match=r"missing.*'y'"):
        model().fit(good_frame().select("x"))
    with pytest.raises(ValueError, match=r"missing.*'x'"):
        sine_fit.predict(pl.DataFrame({"other": [1.0]}))
    with pytest.raises(ValueError, match=r"'x'.*1 missing"):
        sine_fit.predict(pl.DataFrame({"x": [1.0, None]}))


# --- columns that are not the model's, and what a verbose check says ---


def messages(records):
    return [record["message"] for record in records]


def test_a_verbose_check_says_what_it_checked_and_names_the_columns_it_ignored(records):
    model().check(good_frame(n=5).with_columns(station=pl.lit("a")), verbose=True)

    assert messages(records) == [
        "settings given: features x, target y; set by you: features, target; no schema supplied",
        "data checks passed: columns_present, enough_rows, numeric, no_missing_values, no_infinite_values (5 rows)",
        "columns ignored: station (the model does not use them)",
    ]


def test_other_columns_never_warn_and_a_target_among_rows_to_predict_is_not_called_ignored(
    records,
):
    rows = pl.DataFrame(
        {
            "x": [1.0, 2.0],
            "station": ["a", "b"],
            "y": ["not", "numbers"],
            "notes": [1, 2],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model().check(rows, for_prediction=True, verbose=True)

    assert [
        line for line in messages(records) if line.startswith("columns ignored")
    ] == ["columns ignored: station, notes (the model does not use them)"]


def test_predict_ignores_a_target_and_any_other_column_among_the_rows(sine_fit):
    rows = pl.DataFrame(
        {
            "x": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "y": [999.0] * 5,  # far from the truth, so it would show if it were used
            "station": ["a", "b", "c", "d", "e"],
        }
    )  # the data is sin(3x) plus noise: -0.14, -1.0, 0, 1.0, 0.14

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prediction = sine_fit.predict(rows)

    assert prediction.mean() == pytest.approx([-0.14, -1.0, 0.0, 1.0, 0.14], abs=0.3)


# --- CheckedData: what it is valid for ---


@pytest.mark.slow
def test_fit_and_predict_use_checked_data_as_it_was_when_checked_and_do_not_check_it_again():
    train = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]}
    )
    rows = pd.DataFrame({"x": [2.0, 4.0]})
    checked_train = model().check(train)
    checked_rows = model().check(rows, for_prediction=True)
    train.loc[0, "y"] = float(
        "nan"
    )  # changes after the check would fail the checks, if they ran again
    rows.loc[0, "x"] = float("nan")

    prediction = model().fit(checked_train).predict(checked_rows)

    assert prediction.to_frame()["x"].tolist() == [2.0, 4.0]
    assert prediction.mean() == pytest.approx(
        [4.0, 16.0], abs=1.5
    )  # near the training points y = x squared


def test_predict_accepts_checked_data_of_either_kind_from_any_model_with_the_same_features_target_and_schema(
    sine_fit,
):
    rows = pl.DataFrame(
        {"x": [-1.0, -0.5, 0.0, 0.5, 1.0]}
    )  # the data is sin(3x) plus noise: -0.14, -1.0, 0, 1.0, 0.14
    # `sine_fit` is an exact model with the default kernel: the class, the kernel and the
    # sparse settings play no part in the checks.
    other_class_and_kernel = SparseGPRegressor(
        features=["x"], target="y", kernel="rbf", num_inducing=9, random_state=1
    )

    for_prediction = model().check(rows, for_prediction=True)
    training_shaped = model().check(rows.with_columns(y=pl.col("x")))
    from_other = other_class_and_kernel.check(rows, for_prediction=True)

    for checked in (for_prediction, training_shaped, from_other):
        assert sine_fit.predict(checked).mean() == pytest.approx(
            [-0.14, -1.0, 0.0, 1.0, 0.14], abs=0.3
        )


def test_checked_prediction_data_cannot_be_used_to_fit():
    checked = model().check(pl.DataFrame({"x": [1.0]}), for_prediction=True)

    with pytest.raises(
        ValueError,
        match=r"made with for_prediction=True, so it has no checked target column",
    ):
        model().fit(checked)


def refusal(made_by, used_by):
    with pytest.raises(ValueError, match=r"This data was checked by") as raised:
        used_by.fit(made_by.check(good_frame().with_columns(z=pl.col("x"))))
    message = str(raised.value)
    assert "model.check(df)" in message
    assert "gpjax" not in message.lower()
    assert "for_prediction" not in message
    return message


@pytest.mark.parametrize(
    ("used_by", "said", "not_said"),
    [
        (
            GPRegressor(features=["x"], target="z"),
            "whose target differs from this model's",
            ("features", "kernel", "schema"),
        ),
        (
            GPRegressor(features=["x", "z"], target="y"),
            "whose features differ from this model's",
            ("kernel", "target", "schema"),
        ),
        (
            SparseGPRegressor(
                features=["x"], target="z", kernel="rbf", schema=Recorder()
            ),
            "whose target and schema differ from this model's",
            ("features", "kernel", "SparseGPRegressor"),
        ),
    ],
    ids=["target", "features", "several settings"],
)
def test_checked_data_from_a_model_that_differs_is_refused_naming_only_the_settings_that_differ(
    used_by, said, not_said
):
    message = refusal(GPRegressor(features=["x"], target="y"), used_by)

    assert said in message
    for unrelated in not_said:
        assert unrelated not in message


def test_the_refusal_for_a_different_schema_explains_what_that_means():
    plain, schema = GPRegressor(features=["x"], target="y"), Recorder()

    message = refusal(plain, GPRegressor(features=["x"], target="y", schema=schema))
    again = refusal(
        GPRegressor(features=["x"], target="y", schema=Recorder()),
        GPRegressor(features=["x"], target="y", schema=Recorder()),
    )

    assert "different schema (or with none)" in message
    assert "has not passed this model's schema" in message
    assert "different schema (or with none)" in again


def test_a_fitted_model_refuses_prediction_data_checked_by_a_model_with_other_features(
    sine_fit,
):
    other = GPRegressor(features=["z"], target="y")
    checked = other.check(pl.DataFrame({"z": [3.0]}), for_prediction=True)

    with pytest.raises(ValueError, match=r"whose features differ from this model's"):
        sine_fit.predict(checked)
