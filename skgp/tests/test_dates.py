"""Input columns of dates, or of dates and times: counted in days since 1970-01-01."""

from datetime import UTC, date, datetime, timedelta
import json
import re

import gpjax as gpx
import numpy as np
import pandas as pd
import polars as pl
import pytest

from skgp import GPRegressor

# Twenty days from 2026-01-01, with a target that rises by 1 a day from 0, wobbling by about
# 0.01: on 2026-01-05 it is about 4, and at noon that day about 4.5.
DAYS = [date(2026, 1, 1) + timedelta(days=day) for day in range(20)]
RISING = [day + 0.01 * (-1) ** day for day in range(20)]


@pytest.fixture(scope="module")
def daily_fit():
    fitted = GPRegressor(features="day", target="level").fit(
        pl.DataFrame({"day": DAYS, "level": RISING})
    )
    # JAX compiles on the first prediction; pay for it here, once, not in a test.
    fitted.predict(pl.DataFrame({"day": [date(2026, 1, 5)]}))
    return fitted


def test_a_time_of_day_counts_as_a_fraction_of_a_day(daily_fit):
    noon = pl.DataFrame({"day": [datetime(2026, 1, 5, 12)]})

    assert daily_fit.predict(noon).mean()[0] == pytest.approx(4.5, abs=0.05)


def test_a_date_and_time_with_a_time_zone_counts_from_that_instant_in_utc(daily_fit):
    # One in the morning in Amsterdam in winter is midnight in UTC, so 2026-01-05 itself.
    amsterdam = pl.DataFrame({"day": [datetime(2026, 1, 5, 1)]}).with_columns(
        pl.col("day").dt.replace_time_zone("Europe/Amsterdam")
    )

    assert daily_fit.predict(amsterdam).mean()[0] == pytest.approx(4.0, abs=0.05)


def test_pandas_dates_and_times_are_inputs_too(daily_fit):
    rows = pd.DataFrame({"day": pd.to_datetime(["2026-01-05 12:00"])})

    assert daily_fit.predict(rows).mean()[0] == pytest.approx(4.5, abs=0.05)


def test_pandas_dates_and_times_in_any_time_zone_count_from_that_instant_in_utc(
    daily_fit,
):
    # Seven in the evening of 2026-01-04 in New York, in winter, is midnight on 2026-01-05
    # in UTC.
    new_york = pd.to_datetime(["2026-01-04 19:00"]).tz_localize("America/New_York")

    prediction = daily_fit.predict(pd.DataFrame({"day": new_york}))

    assert prediction.mean()[0] == pytest.approx(4.0, abs=0.05)


@pytest.mark.slow
def test_a_period_along_a_date_column_is_in_days():
    # A cycle that repeats every 7 days, measured each day for 8 weeks with a little noise.
    days = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day) for day in range(56)]
    noise = 0.1 * np.random.default_rng(0).normal(size=56)
    weekly = np.sin(2 * np.pi * np.arange(56) / 7) + noise
    kernel = gpx.kernels.Periodic(period=6.5)

    fitted = GPRegressor(features="day", target="level", kernel=kernel).fit(
        pl.DataFrame({"day": days, "level": weekly})
    )

    learned = json.loads(fitted.to_json())["learned"]["kernel"]
    assert learned["period"] == pytest.approx(7.0, rel=0.02)


def test_a_printed_fitted_model_and_its_json_name_the_unit_of_a_date_column(
    daily_fit,
):
    assert "features day (days since 1970-01-01);" in repr(daily_fit)
    setup = json.loads(daily_fit.to_json())["setup"]
    assert setup["date_features"] == {"day": "days since 1970-01-01"}


def test_a_column_that_held_dates_must_hold_dates_in_the_rows_to_predict(daily_fit):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Column 'day' held dates when the model was fitted, so the rows to predict "
            "must hold dates there too; it holds Float64 values."
        ),
    ):
        daily_fit.predict(pl.DataFrame({"day": [20_458.0]}))  # 2026-01-05, in days
    with pytest.raises(ValueError, match=r"^Column 'day' held dates when the model"):
        daily_fit.check(pl.DataFrame({"day": [20_458.0]}))


def test_a_column_that_held_numbers_must_hold_numbers_in_the_rows_to_predict(
    sine_fit,
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Column 'x' held numbers when the model was fitted, so the rows to predict "
            "must hold numbers there too, not dates."
        ),
    ):
        sine_fit.predict(pl.DataFrame({"x": [date(2026, 1, 5)]}))
