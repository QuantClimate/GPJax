"""A `GPRegressor` with a `noise_column`: each observation's known noise, as a standard
deviation in the target's units, with learned noise on top or none."""

import json
import pickle
import re

import numpy as np
import polars as pl
import pytest

from skgp import GPRegressor, SparseGPRegressor

# Ten readings of y = x, each known to within 0.05, except the one at x = 5, which reads 15
# and is known only to within 100.
LINE_WITH_A_ROUGH_READING = pl.DataFrame(
    {
        "x": [float(x) for x in range(10)],
        "y": [0.0, 1.0, 2.0, 3.0, 4.0, 15.0, 6.0, 7.0, 8.0, 9.0],
        "error": [0.05] * 5 + [100.0] + [0.05] * 4,
    }
)


@pytest.fixture(scope="module")
def known_only():
    fitted = GPRegressor(
        "x", "y", noise_column="error", learned_noise=None, kernel="rbf"
    ).fit(LINE_WITH_A_ROUGH_READING)
    # JAX compiles on the first prediction; pay for it here, once, not in a test.
    fitted.predict(pl.DataFrame({"x": [5.0], "error": [0.0]}))
    return fitted


def test_a_row_with_large_known_noise_is_trusted_less(known_only):
    at_the_rough_reading = pl.DataFrame({"x": [5.0]})

    mean = known_only.predict(at_the_rough_reading, include_noise=False).mean()

    assert mean[0] == pytest.approx(5.0, abs=0.3)


def test_a_prediction_with_noise_adds_each_new_rows_own_known_noise(known_only):
    rows = pl.DataFrame({"x": [2.0, 2.0], "error": [0.5, 0.0]})

    observation = known_only.predict(rows).std() ** 2
    curve = known_only.predict(rows, include_noise=False).std() ** 2

    np.testing.assert_allclose(observation - curve, [0.25, 0.0], atol=1e-9)


def test_the_covariance_of_a_prediction_with_noise_adds_known_noise_to_each_row_alone(
    known_only,
):
    rows = pl.DataFrame({"x": [2.0, 2.0], "error": [0.5, 0.0]})

    observation = known_only.predict(rows).covariance()
    curve = known_only.predict(rows, include_noise=False).covariance()

    np.testing.assert_allclose(
        observation - curve, [[0.25, 0.0], [0.0, 0.0]], atol=1e-9
    )


def test_learned_noise_is_the_scatter_the_known_noise_leaves_out():
    # Scatter of 0.5 at every row, of which the instrument states 0.4, so the rest is
    # 0.3: 0.4 squared plus 0.3 squared is 0.5 squared.
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 10.0, 300)
    data = pl.DataFrame(
        {"x": x, "y": np.sin(x) + 0.5 * rng.normal(size=300), "error": 0.4}
    )

    fitted = GPRegressor("x", "y", noise_column="error").fit(data)

    learned = json.loads(fitted.to_json())["learned"]["noise"]
    assert learned == pytest.approx(0.3, abs=0.1)


def test_rows_to_predict_need_the_known_noise_only_for_a_prediction_with_noise(
    known_only,
):
    without_it = pl.DataFrame({"x": [2.0]})

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The data is missing 1 column the model needs: 'error'. It holds the known "
            "noise on each new observation, which a prediction with noise adds. Predict "
            "with include_noise=False for the curve alone, without it."
        ),
    ):
        known_only.predict(without_it)
    assert known_only.predict(without_it, include_noise=False).mean()[
        0
    ] == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"learned_noise": None},
            (
                "learned_noise is None and there is no noise_column, so the model "
                "would have no noise at all. Give noise_column"
            ),
        ),
        (
            {"noise_column": "error", "learned_noise": "heteroscedastic"},
            (
                "noise_column 'error' cannot be combined with "
                'learned_noise="heteroscedastic"'
            ),
        ),
        (
            {"noise_column": "x"},
            (
                "noise_column 'x' is also in features; a column cannot be both an input "
                "and the known noise."
            ),
        ),
        (
            {"noise_column": "y"},
            (
                "noise_column 'y' is also the target; a column cannot be both the target "
                "and the known noise."
            ),
        ),
    ],
    ids=["no noise at all", "with heteroscedastic", "an input", "the target"],
)
def test_known_noise_that_cannot_work_is_refused_when_the_model_is_defined(
    settings, message
):
    with pytest.raises(ValueError, match=re.escape(message)):
        GPRegressor("x", "y", **settings)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"noise_column": "error"},
            (
                "SparseGPRegressor has no setting 'noise_column': GPJax's sparse fits take "
                "the same noise at every row. Use a GPRegressor for known noise."
            ),
        ),
        (
            {"learned_noise": None},
            "learned_noise must be 'constant' or 'heteroscedastic'; got None.",
        ),
    ],
    ids=["known noise", "no learned noise"],
)
def test_a_sparse_regressor_takes_no_known_noise_and_says_why(settings, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        SparseGPRegressor("x", "y", **settings)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            [0.1] * 8 + [-0.1, -0.2],
            (
                "Column 'error' has 2 negative values out of 10 rows. It holds the known "
                "noise on each observation, a standard deviation, which cannot be below "
                "zero."
            ),
        ),
        ([0.1] * 9 + [None], "Column 'error' has 1 missing value"),
        (["a"] * 10, "Column 'error' is not numeric"),
    ],
    ids=["negative", "missing", "text"],
)
def test_known_noise_that_is_not_a_standard_deviation_is_refused(error, message):
    data = pl.DataFrame(
        {"x": [float(x) for x in range(10)], "y": [0.0] * 10, "error": error}
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        GPRegressor("x", "y", noise_column="error").check(data)


def test_training_data_needs_the_known_noise():
    with pytest.raises(
        ValueError, match=re.escape("missing 1 column the model needs: 'error'")
    ):
        GPRegressor("x", "y", noise_column="error").check(
            LINE_WITH_A_ROUGH_READING.drop("error")
        )


def test_data_checked_without_known_noise_is_refused_by_a_model_with_it():
    checked = GPRegressor("x", "y").check(LINE_WITH_A_ROUGH_READING)

    with pytest.raises(
        ValueError, match=r"whose noise_column differs from this model's"
    ):
        GPRegressor("x", "y", noise_column="error").fit(checked)


def test_a_fitted_model_with_known_noise_alone_names_its_column_and_learned_no_noise(
    known_only,
):
    description = json.loads(known_only.to_json())

    assert description["setup"]["learned_noise"] is None
    assert description["setup"]["noise_column"] == "error"
    assert "noise" not in description["learned"]
    assert "learned_noise None; noise_column error" in repr(known_only)


def test_a_pickled_model_with_known_noise_predicts_the_same(known_only):
    rows = pl.DataFrame({"x": [2.0, 5.0], "error": [0.5, 0.0]})

    loaded = pickle.loads(pickle.dumps(known_only))  # nosec B301: loads what this test just pickled

    np.testing.assert_allclose(
        loaded.predict(rows).std(), known_only.predict(rows).std(), rtol=1e-9
    )
