"""`fitted.score(data)`: how well a fitted model predicts held-out data, as a `Score`."""

import re

import polars as pl
import pytest

from helpers import Recorder
from skgp import GPRegressor

# Ten readings of y = x, from x = 0 to 9, each known to within 0.01, so the fitted curve is
# the line itself.
LINE = pl.DataFrame(
    {
        "x": [float(x) for x in range(10)],
        "y": [float(x) for x in range(10)],
        "error": [0.01] * 10,
    }
)
# Three new readings on the line, between the old ones, each known only to within 1. A
# prediction of each is then a normal distribution with its mean on the reading and a
# standard deviation of 1, whose CRPS is 2 phi(0) - 1 / sqrt(pi) = 0.7979 - 0.5642 = 0.2337.
ROUGH_READINGS = pl.DataFrame(
    {"x": [2.5, 4.5, 6.5], "y": [2.5, 4.5, 6.5], "error": [1.0, 1.0, 1.0]}
)


@pytest.fixture(scope="module")
def line_fit():
    fitted = GPRegressor(
        "x", "y", noise_column="error", learned_noise=None, kernel="rbf"
    ).fit(LINE)
    # JAX compiles on the first prediction; pay for it here, once, not in a test.
    fitted.predict(ROUGH_READINGS)
    return fitted


def test_a_regressor_is_scored_by_the_mean_crps_sign_flipped_by_default(line_fit):
    score = line_fit.score(ROUGH_READINGS)

    assert dict(score) == {"neg_crps": pytest.approx(-0.2337, abs=0.005)}
    assert score.number_of_rows == 3


def test_scikit_learn_scorers_score_the_mean_in_the_order_given(line_fit):
    score = line_fit.score(ROUGH_READINGS, scoring=["neg_mean_absolute_error", "r2"])

    assert list(score) == ["neg_mean_absolute_error", "r2"]
    assert score["neg_mean_absolute_error"] == pytest.approx(0.0, abs=0.01)
    assert score["r2"] == pytest.approx(1.0, abs=0.001)


def test_a_scorer_given_itself_or_in_a_dict_is_named_as_given(line_fit):
    def always_one(estimator, X, y):
        return 1.0

    assert list(line_fit.score(ROUGH_READINGS, scoring=always_one)) == ["always_one"]
    named = line_fit.score(ROUGH_READINGS, scoring={"crps": "neg_crps"})
    assert list(named) == ["crps"]


def test_a_score_is_a_mapping_that_never_changes_and_prints_as_a_table(line_fit):
    score = line_fit.score(ROUGH_READINGS, scoring=["r2", "neg_crps"])

    assert score == {"r2": score["r2"], "neg_crps": score["neg_crps"]}
    with pytest.raises(TypeError):
        score["r2"] = 0.0  # changing it is the point
    title, r2, crps = repr(score).split("\n")
    assert title == "Score on 3 rows (higher is better):"
    assert r2.startswith("  r2        ")
    assert crps.startswith("  neg_crps  -0.23")


# This test checks what the schema sees, not the fit, which need not settle.
@pytest.mark.filterwarnings("ignore::skgp.ConvergenceWarning")
def test_rows_are_checked_once_by_your_schema_and_may_hold_the_target(line_fit):
    schema = Recorder()
    with_schema = GPRegressor(
        "x", "y", noise_column="error", learned_noise=None, schema=schema
    ).fit(LINE)
    schema.seen.clear()

    with_schema.score(ROUGH_READINGS)

    assert len(schema.seen) == 1
    assert "y" in schema.seen[0].columns


def test_checked_rows_are_scored_too(line_fit):
    checked = line_fit.check(ROUGH_READINGS)

    assert line_fit.score(checked)["neg_crps"] == pytest.approx(-0.2337, abs=0.005)


def test_a_classifier_is_scored_by_the_brier_loss_sign_flipped_by_default(weather_fit):
    # Rows far from the boundary, each predicted with a probability within 0.05 of its own
    # class, so the squared differences, and their mean, are below 0.05 squared.
    far = pl.DataFrame({"x": [-3.0, 3.0], "weather": ["dry", "rain"]})

    score = weather_fit.score(far)

    assert list(score) == ["neg_brier_score"]
    assert -0.0025 < score["neg_brier_score"] <= 0.0


def test_a_classifier_s_scorers_take_text_classes_with_the_second_as_positive(
    weather_fit,
):
    # Dry below zero and rain above, with each row right.
    far = pl.DataFrame(
        {"x": [-3.0, -2.0, 2.0, 3.0], "weather": ["dry", "dry", "rain", "rain"]}
    )

    score = weather_fit.score(far, scoring=["accuracy", "f1", "roc_auc"])

    assert dict(score) == {"accuracy": 1.0, "f1": 1.0, "roc_auc": 1.0}


def test_a_classifier_is_not_scored_by_the_crps(weather_fit):
    far = pl.DataFrame({"x": [-3.0], "weather": ["dry"]})

    with pytest.raises(ValueError, match=r"^neg_crps scores a predicted distribution"):
        weather_fit.score(far, scoring="neg_crps")


def test_score_needs_the_target_column(line_fit):
    with pytest.raises(
        ValueError,
        match=re.escape("The data is missing 1 column the model needs: 'y'."),
    ):
        line_fit.score(ROUGH_READINGS.drop("y"))


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (
            [2.5, None, 6.5],
            "Column 'y' has 1 missing value (empty or NaN) out of 3 rows.",
        ),
        ([2.5, float("inf"), 6.5], "Column 'y' has 1 infinite value out of 3 rows."),
        (["a", "b", "c"], "Column 'y' is not numeric"),
    ],
    ids=["missing", "infinite", "text"],
)
def test_score_needs_a_known_number_at_every_row_of_a_regressor_s_target(
    line_fit, target, message
):
    with pytest.raises(ValueError, match=re.escape(message)):
        line_fit.score(ROUGH_READINGS.with_columns(pl.Series("y", target)))


def test_score_refuses_a_class_the_model_was_not_fitted_to(weather_fit):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Column 'weather' holds 'snow', which the model was not fitted to; it knows "
            "only 'dry' and 'rain'."
        ),
    ):
        weather_fit.score(pl.DataFrame({"x": [0.0], "weather": ["snow"]}))


def test_score_refuses_data_with_no_rows(line_fit):
    with pytest.raises(
        ValueError, match=r"^The data has no rows; scoring needs at least"
    ):
        line_fit.score(ROUGH_READINGS.head(0))


def test_score_refuses_an_unknown_scorer_and_an_empty_list(line_fit):
    with pytest.raises(ValueError, match="'nope'"):
        line_fit.score(ROUGH_READINGS, scoring="nope")
    with pytest.raises(ValueError, match=r"^scoring names no scorer"):
        line_fit.score(ROUGH_READINGS, scoring=[])


def test_a_model_with_known_noise_needs_its_column_to_be_scored(line_fit):
    with pytest.raises(ValueError, match=r"missing 1 column the model needs: 'error'"):
        line_fit.score(ROUGH_READINGS.drop("error"))
