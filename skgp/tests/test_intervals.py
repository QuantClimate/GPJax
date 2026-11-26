import numpy as np
import polars as pl
import pytest

from helpers import sine_data
from skgp import GPRegressor


def test_default_interval_is_95_percent_around_the_mean_in_original_units(sine_fit):
    # sine_fit is a fit to y = sin(3x) with noise of standard deviation 0.1. At x = 0 the curve
    # is sin(0) = 0, so a new observation is within 0 ± 1.96 * 0.1 95% of the time, and within
    # 0 ± 1.28 * 0.1 80% of the time. In rescaled units the bounds would be about 0.28.
    prediction = sine_fit.predict(pl.DataFrame({"x": [0.0]}))

    interval = prediction.interval()

    assert interval.columns == ["lower", "upper"]
    assert interval.row(0) == pytest.approx((-0.196, 0.196), abs=0.04)
    assert prediction.interval(0.8).row(0) == pytest.approx((-0.128, 0.128), abs=0.04)


@pytest.mark.parametrize("level", [-0.5, 0, 1, 1.5, float("nan")])
def test_a_level_outside_zero_to_one_is_rejected_with_a_clear_error(sine_fit, level):
    prediction = sine_fit.predict(pl.DataFrame({"x": [0.0]}))

    with pytest.raises(
        ValueError, match=r"level must be between 0 and 1 \(exclusive\)"
    ):
        prediction.interval(level)
    with pytest.raises(
        ValueError, match=r"interval must be between 0 and 1 \(exclusive\)"
    ):
        prediction.to_frame(interval=level)


@pytest.mark.slow
def test_the_spread_is_for_a_new_observation_so_it_includes_the_noise():
    # With 200 points the curve itself is known to well under 0.1, so a spread near the noise level
    # (0.3) can only come from adding the noise.
    noise_sd = 0.3
    fitted = GPRegressor(features=["x"], target="y").fit(
        sine_data(n=200, noise_sd=noise_sd, seed=1)
    )

    std = fitted.predict(pl.DataFrame({"x": np.linspace(-1.5, 1.5, 50)})).std()

    assert std.min() > 0.8 * noise_sd
    assert std.max() < 1.3 * noise_sd
