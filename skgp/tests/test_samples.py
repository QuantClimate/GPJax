"""Random samples from a prediction: `Prediction.sample` and `ClassPrediction.sample`."""

import pickle
import re

import numpy as np
import polars as pl
import pytest

from helpers import NOISE_SD


def test_a_prediction_of_no_rows_has_no_samples(sine_fit):
    no_rows = pl.DataFrame({"x": []}, schema={"x": pl.Float64})

    assert sine_fit.predict(no_rows).sample(3).size == 0


def test_the_same_random_state_gives_the_same_samples(sine_fit):
    prediction = sine_fit.predict(pl.DataFrame({"x": [0.0, 0.5]}))

    np.testing.assert_array_equal(
        prediction.sample(3, random_state=7), prediction.sample(3, random_state=7)
    )
    assert not np.array_equal(
        prediction.sample(3, random_state=7), prediction.sample(3, random_state=8)
    )


def test_samples_of_a_new_observation_spread_by_the_measurement_noise(sine_fit):
    # The training data is sin(3x) with noise of standard deviation 0.1 (`NOISE_SD`), and
    # the curve is well known at x = 0, so new observations there scatter by about 0.1
    # around 0.
    samples = sine_fit.predict(pl.DataFrame({"x": [0.0]})).sample(4_000)[0]

    assert samples.mean() == pytest.approx(0.0, abs=0.05)
    assert samples.std() == pytest.approx(NOISE_SD, rel=0.15)


def test_samples_of_the_curve_hang_together_across_rows(sine_fit):
    # One curve has one value at a place, so two rows at the same place get the same value in
    # every sample of the curve, to within the small jitter GPJax adds to keep the
    # covariance invertible; new observations there each have their own noise.
    same_place = pl.DataFrame({"x": [0.3, 0.3]})

    curve = sine_fit.predict(same_place, include_noise=False).sample(50)
    observations = sine_fit.predict(same_place).sample(50)

    np.testing.assert_allclose(curve[0], curve[1], atol=0.005)
    assert np.abs(observations[0] - observations[1]).max() > 0.1


@pytest.fixture(scope="module")
def weather_prediction(weather_fit):
    # Dry below zero and rain above: halfway, at 0, either class is as likely; at -3 it is
    # almost always dry, and at 3 almost always rain.
    return weather_fit.predict(pl.DataFrame({"x": [0.0, 0.0, -3.0, 3.0]}))


def test_class_samples_hold_your_classes_as_often_as_their_probabilities_say(
    weather_prediction,
):
    samples = weather_prediction.sample(2_000)

    assert set(np.unique(samples)) == {"dry", "rain"}
    share_of_rain = (samples == "rain").mean(axis=1)
    np.testing.assert_allclose(share_of_rain[:2], [0.5, 0.5], atol=0.04)
    assert share_of_rain[2] < 0.05
    assert share_of_rain[3] > 0.95


def test_class_samples_hang_together_across_rows(weather_prediction):
    # Two rows at the same place, where either class is as likely: drawn on their own they
    # would agree in half the samples. Drawn from one curve, they agree far more often.
    samples = weather_prediction.sample(2_000)

    assert (samples[0] == samples[1]).mean() > 0.75


@pytest.mark.parametrize(
    ("n_samples", "random_state", "message"),
    [
        (
            0,
            0,
            "n_samples is how many samples to give, so it must be a whole number above zero; got 0.",
        ),
        (
            2.0,
            0,
            "n_samples is how many samples to give, so it must be a whole number above zero; got 2.0.",
        ),
        (
            1,
            -1,
            (
                "random_state is the seed of the samples, so it must be a whole number from 0 "
                "to 9,223,372,036,854,775,807; got -1."
            ),
        ),
    ],
    ids=["no samples", "not whole", "negative seed"],
)
def test_sample_refuses_a_number_of_samples_or_a_seed_that_is_not_a_whole_number(
    sine_fit, weather_prediction, n_samples, random_state, message
):
    prediction = sine_fit.predict(pl.DataFrame({"x": [0.0]}))

    for regression_or_class_prediction in (prediction, weather_prediction):
        with pytest.raises(ValueError, match=re.escape(message)):
            regression_or_class_prediction.sample(n_samples, random_state)


def test_a_pickled_class_prediction_gives_the_same_samples(weather_prediction):
    loaded = pickle.loads(pickle.dumps(weather_prediction))

    np.testing.assert_array_equal(loaded.sample(5), weather_prediction.sample(5))
