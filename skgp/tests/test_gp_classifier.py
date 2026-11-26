"""`GPClassifier`: what it accepts, what it predicts, and how it describes itself."""

import json
import re
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from helpers import (
    WEATHER_GROUPS as GROUPS,
    sine_data,
)
import skgp
from skgp import GPClassifier, GPRegressor, SparseGPRegressor


@pytest.fixture(scope="module")
def groups_fit():
    return GPClassifier("x", "weather").fit(GROUPS)


def test_each_training_row_is_more_likely_its_own_class_and_the_classes_are_your_values(
    groups_fit,
):
    prediction = groups_fit.predict(GROUPS)
    frame = prediction.to_frame()

    assert groups_fit.classes == prediction.classes == ("dry", "rain")
    assert list(prediction.most_likely()) == GROUPS["weather"].to_list()
    np.testing.assert_allclose(prediction.probabilities().sum(axis=1), 1.0)
    assert frame.columns == [
        "x",
        "weather",
        "probability_dry",
        "probability_rain",
        "most_likely_class",
    ]
    assert (frame["probability_dry"] > 0.5).to_list() == [True] * 15 + [False] * 15


def test_a_row_far_from_the_data_is_less_certain_than_one_inside_a_group(groups_fit):
    rows = pl.DataFrame({"x": [2.0, 30.0]})

    inside, far = groups_fit.predict(rows).probabilities()[:, 1]

    assert inside > 0.9
    assert abs(far - 0.5) < abs(inside - 0.5)


def test_to_json_lists_the_classes_and_learns_no_noise(groups_fit):
    description = json.loads(groups_fit.to_json())

    assert description["classes"] == ["dry", "rain"]
    assert set(description["learned"]) == {"kernel", "mean_function"}
    assert description["fitting"]["finished_properly"] is True
    assert (
        description["fitting"]["iterations"] < 2000
    )  # it stopped once it had finished


def test_true_and_false_are_coded_in_sorted_order_the_coding_is_logged_and_pandas_comes_back(
    records,
):
    rows = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 8), "wet": [True, False] * 4})
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", skgp.ConvergenceWarning
        )  # 3 is too few to finish
        fitted = GPClassifier("x", "wet").fit(rows, iterations=3, verbose=True)

    frame = fitted.predict(rows).to_frame()

    assert fitted.classes == (False, True)
    assert "classes coded: in wet, False is 0 and True is 1" in [
        record["message"] for record in records
    ]
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns[-3:]) == [
        "probability_False",
        "probability_True",
        "most_likely_class",
    ]


@pytest.mark.parametrize(
    ("target", "said"),
    [
        (["dry"] * 6, "Column 'weather' holds only one value, 'dry'"),
        (
            ["dry", "rain", "snow"] * 2,
            "Column 'weather' holds 3 different values ('dry', 'rain', 'snow')",
        ),
        (["dry", None] * 3, "Column 'weather' has 3 missing values"),
    ],
    ids=["one value", "three values", "missing values"],
)
def test_a_target_without_exactly_two_classes_is_refused_naming_the_column(
    target, said
):
    rows = pl.DataFrame({"x": np.arange(6.0), "weather": target})

    with pytest.raises(ValueError, match=f"^{re.escape(said)}"):
        GPClassifier("x", "weather").check(rows)


def test_rows_to_predict_need_no_target_and_inputs_must_still_be_numbers():
    classifier = GPClassifier("x", "weather")

    classifier.check(pl.DataFrame({"x": [0.5]}), for_prediction=True)
    with pytest.raises(ValueError, match="A GPClassifier's inputs must be numbers"):
        classifier.check(GROUPS.with_columns(pl.col("x").cast(pl.String)))


def test_data_checked_by_a_regressor_is_refused_by_a_classifier_and_the_other_way_round():
    rows = pl.DataFrame({"x": np.arange(6.0), "y": [0.0, 1.0] * 3})
    regressor, classifier = GPRegressor("x", "y"), GPClassifier("x", "y")

    with pytest.raises(
        ValueError,
        match=r"^This data was checked by a GPRegressor, which checks it for regression, and "
        r"this model is a GPClassifier, which checks it for classification\. Run check again",
    ):
        classifier.fit(regressor.check(rows))
    with pytest.raises(ValueError, match="checked by a GPClassifier"):
        regressor.fit(classifier.check(rows))


@pytest.mark.parametrize(
    "model",
    [
        GPRegressor("x", "y"),
        SparseGPRegressor("x", "y", minibatch_size=10),
        GPClassifier("x", "y"),
    ],
    ids=["exact", "minibatches", "classifier"],
)
@pytest.mark.parametrize("iterations", [0, -3, 2.5, True, "10"])
def test_every_model_refuses_iterations_that_are_not_a_whole_number_above_zero(
    model, iterations
):
    with pytest.raises(ValueError, match=r"^iterations is how many iterations"):
        model.fit(sine_data(), iterations=iterations)


def test_a_classifier_that_runs_out_of_iterations_warns_and_suggests_more():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = GPClassifier("x", "weather").fit(GROUPS, iterations=2)

    fitting = json.loads(fitted.to_json())["fitting"]
    (warning,) = [w for w in caught if issubclass(w.category, skgp.ConvergenceWarning)]

    assert fitting["iterations"] == 2
    assert fitting["finished_properly"] is False
    # No advice about the scale of the target or noise-free data: those are for regression.
    assert str(warning.message) == (
        "Fitting did not finish properly after 2 iterations. It reached its limit of "
        "iterations while the fit was still improving. The fitted model may be less "
        "reliable. To let it run longer, pass a larger iterations to fit. Predictions are "
        "still available."
    )


def test_a_classifier_on_more_rows_than_the_limit_warns_naming_the_sparse_classifier(
    monkeypatch,
):
    monkeypatch.setattr(skgp.config, "size_warning_rows", 20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GPClassifier("x", "weather").fit(GROUPS, iterations=1)

    (warning,) = [w for w in caught if issubclass(w.category, skgp.LargeDataWarning)]

    assert str(warning.message) == (
        "30 rows is a lot for a GPClassifier, which gets slow and needs a lot of memory as "
        "rows grow. Consider SparseGPClassifier, which takes the same settings. The fit "
        "carries on."
    )
    assert warning.filename == __file__  # points at the call to fit
