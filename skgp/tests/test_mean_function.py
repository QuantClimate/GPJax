"""The mean function: what the curve is far from the training data, where the data says
nothing. Near the data the curve follows the data whatever it is."""

from datetime import date, timedelta
import json
import pickle

import gpjax as gpx
import jax.numpy as jnp
from loguru import logger
import numpy as np
import paramax
import polars as pl
import pytest

from skgp import GPClassifier, GPRegressor, SparseGPClassifier, SparseGPRegressor
from skgp.sklearn import GaussianProcessRegressor

# A straight line, 100 + 2x, with a small wiggle, on x from 0 to 10, with a known noise of
# 0.1 at every row for the models that take it.
X = np.linspace(0.0, 10.0, 30)
LINE = pl.DataFrame(
    {"x": X, "y": 100 + 2 * X + 0.3 * np.sin(3 * X), "error": np.full(30, 0.1)}
)
# Far beyond the data, where the line is at 200.
FAR = pl.DataFrame({"x": [50.0]})


def mean_far_away(model):
    return model.fit(LINE).predict(FAR).mean()[0]


def learned_mean_function(fitted):
    return json.loads(fitted.to_json())["learned"].get("mean_function")


@pytest.fixture(scope="module")
def linear_fit():
    fitted = GPRegressor("x", "y", mean_function="linear").fit(LINE)
    # JAX compiles on the first prediction; pay for it here, once, not in a test.
    fitted.predict(FAR)
    return fitted


def test_a_linear_mean_function_carries_on_the_trend_far_from_the_data(linear_fit):
    assert linear_fit.predict(FAR).mean()[0] == pytest.approx(200, abs=2)


def test_a_linear_mean_function_learns_its_intercept_and_slope_in_your_units(
    linear_fit,
):
    learned = learned_mean_function(linear_fit)

    assert learned["intercept"] == pytest.approx(100, abs=1)  # y where x is 0
    assert learned["slope"] == {"x": pytest.approx(2, abs=0.05)}  # y per unit of x


def test_the_default_constant_mean_function_is_learned_in_the_targets_units():
    fitted = GPRegressor("x", "y").fit(LINE)
    constant = learned_mean_function(fitted)["constant"]

    # Far from the data the prediction is the learned constant, somewhere in y's range.
    assert 100 < constant < 125
    assert fitted.predict(FAR).mean()[0] == pytest.approx(constant, abs=0.5)


def test_a_zero_mean_function_goes_back_to_zero_in_the_targets_units_and_learns_nothing():
    fitted = GPRegressor("x", "y", mean_function="zero").fit(LINE)
    far = pl.DataFrame({"x": [1_000.0]})

    assert fitted.predict(far).mean()[0] == pytest.approx(0, abs=0.01)
    assert learned_mean_function(fitted) is None
    assert "mean_function." not in repr(fitted)


def test_a_gpjax_mean_function_sees_your_units_and_keeps_a_held_value():
    # paramax holds arrays only: a plain 80.0 would be learned.
    held_at_80 = gpx.mean_functions.Constant(paramax.non_trainable(jnp.asarray(80.0)))
    fitted = GPRegressor("x", "y", mean_function=held_at_80).fit(LINE)
    far = pl.DataFrame({"x": [1_000.0]})

    assert fitted.predict(far).mean()[0] == pytest.approx(80, abs=0.01)
    assert learned_mean_function(fitted) == {"constant": pytest.approx(80)}


# Only the sum of the two constants can be learned, so the fit does not settle.
# This test checks only how the sum is named and reported.
@pytest.mark.filterwarnings("ignore::skgp.ConvergenceWarning")
def test_a_sum_of_gpjax_mean_functions_is_named_and_reported_part_by_part():
    model = GPRegressor(
        "x",
        "y",
        mean_function=gpx.mean_functions.Constant(1.0)
        + gpx.mean_functions.Constant(2.0),
    )
    fitted = model.fit(LINE)

    assert "mean_function=Constant + Constant" in repr(model)
    assert json.loads(fitted.to_json())["setup"]["mean_function"] == (
        "Constant + Constant"
    )
    assert set(learned_mean_function(fitted)) == {
        "means[0].constant",
        "means[1].constant",
    }
    assert hash(model) == hash(model.replace())


@pytest.mark.slow
@pytest.mark.parametrize(
    "model",
    [
        GPRegressor("x", "y", mean_function="linear", noise_column="error"),
        SparseGPRegressor("x", "y", mean_function="linear", num_inducing=10),
        SparseGPRegressor(
            "x",
            "y",
            mean_function="linear",
            num_inducing=10,
            minibatch_size=10,
        ),
        GPRegressor("x", "y", mean_function="linear", learned_noise="heteroscedastic"),
    ],
    ids=["known noise", "sparse", "sparse on minibatches", "heteroscedastic"],
)
def test_every_fit_of_a_regressor_takes_the_mean_function(model):
    iterations = 3_000 if getattr(model, "minibatch_size", None) else 2_000
    fitted = model.fit(LINE, iterations=iterations)

    # The curve alone, so that the rows need no column of known noise.
    assert fitted.predict(FAR, include_noise=False).mean()[0] == pytest.approx(
        200, abs=10
    )


def test_a_linear_slope_along_a_date_input_is_per_day():
    days = [date(2026, 1, 1) + timedelta(days=day) for day in range(30)]
    rising = pl.DataFrame({"day": days, "level": [5 + 0.5 * day for day in range(30)]})

    fitted = GPRegressor("day", "level", mean_function="linear").fit(rising)

    assert learned_mean_function(fitted)["slope"] == {
        "day": pytest.approx(0.5, abs=0.01)
    }


# Rain where x is above 3, on x from 0 to 10.
RAIN = pl.DataFrame({"x": X, "rain": np.where(X > 3, "rain", "dry")})


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_class", [GPClassifier, SparseGPClassifier], ids=["exact", "sparse"]
)
def test_a_classifiers_zero_mean_function_gives_each_class_one_half_far_away(
    model_class,
):
    fitted = model_class("x", "rain", mean_function="zero").fit(RAIN)

    far = fitted.predict(pl.DataFrame({"x": [-1_000.0]})).probabilities()[0]

    assert far == pytest.approx([0.5, 0.5], abs=0.001)
    assert learned_mean_function(fitted) is None


@pytest.mark.slow
def test_a_classifiers_default_constant_mean_function_is_learned():
    fitted = GPClassifier("x", "rain").fit(RAIN)

    assert set(learned_mean_function(fitted)) == {"constant"}
    assert json.loads(fitted.to_json())["setup"]["mean_function"] == "constant"


@pytest.mark.slow
def test_a_gpjax_mean_function_on_minibatches_steps_in_the_targets_units():
    # A target of about 0.002 that varies by about 0.0005. Each step on minibatches moves a
    # value by a few hundredths of the target's spread, so a constant that starts at the
    # target's average stays near it, instead of wandering off by several times the target.
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 300)
    small = pl.DataFrame(
        {"x": x, "y": 0.002 + 0.0005 * np.sin(x) + 0.0001 * rng.normal(size=300)}
    )
    model = SparseGPRegressor(
        "x",
        "y",
        mean_function=gpx.mean_functions.Constant(0.002),
        minibatch_size=50,
        num_inducing=20,
    )

    fitted = model.fit(small, iterations=500)

    assert learned_mean_function(fitted)["constant"] == pytest.approx(0.002, abs=0.0005)


def test_any_other_mean_function_is_refused_when_the_model_is_defined():
    with pytest.raises(ValueError, match="mean_function must be 'zero', 'constant' or"):
        GPRegressor("x", "y", mean_function="quadratic")  # ty: ignore[invalid-argument-type]  # the wrong value is the point


def test_a_fitted_model_with_a_linear_mean_function_predicts_the_same_after_pickling(
    linear_fit,
):
    loaded = pickle.loads(pickle.dumps(linear_fit))  # nosec B301: loads what this test just pickled

    np.testing.assert_allclose(
        loaded.predict(FAR).mean(), linear_fit.predict(FAR).mean(), rtol=1e-9
    )


def test_a_verbose_fit_says_which_mean_function_it_chose():
    records = []
    sink = logger.add(lambda message: records.append(message.record["message"]))
    try:
        GPRegressor("x", "y", mean_function="linear").fit(LINE, verbose=True)
    finally:
        logger.remove(sink)

    assert "mean function chosen: linear (set by you)" in records


def test_the_scikit_learn_estimator_takes_the_mean_function():
    estimator = GaussianProcessRegressor(mean_function="linear")

    estimator.fit(LINE.select("x").to_numpy(), LINE["y"].to_numpy())

    assert estimator.get_params()["mean_function"] == "linear"
    assert estimator.predict(np.array([[50.0]]))[0] == pytest.approx(200, abs=2)
