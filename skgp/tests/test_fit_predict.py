import numpy as np
import pandas as pd
import polars as pl
import pytest

from skgp import GPRegressor

# Seven points of y = x squared: a smooth curve that every kernel recovers between the points.
SQUARES = pl.DataFrame(
    {
        "x": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        "y": [9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0],
    }
)
# Rows to predict that already have columns with the names skgp gives its predictions.
CLASHING = pl.DataFrame(
    {"x": [0.0], "mean": [123.0], "std": [1.0], "lower": [122.0], "upper": [124.0]}
)


@pytest.mark.parametrize("kernel", ["rbf", "matern32", "matern52"])
def test_each_kernel_name_fits_a_curve_through_the_data(kernel):
    fitted = GPRegressor(features=["x"], target="y", kernel=kernel).fit(SQUARES)

    prediction = fitted.predict(pl.DataFrame({"x": [1.5]}))

    assert prediction.mean() == pytest.approx([2.25], abs=0.3)  # 1.5 squared


def test_to_frame_is_every_column_passed_followed_by_mean_and_std(sine_fit):
    # sine_fit is a fit to y = sin(3x) with noise of standard deviation 0.1.
    rows = pl.DataFrame(
        {
            "station": ["a", "b", "c"],
            "x": [-1.0, 0.0, 1.0],
            "y": [-0.14, 0.0, 0.14],  # the true values
        }
    )

    frame = sine_fit.predict(rows).to_frame()

    assert frame.columns == ["station", "x", "y", "mean", "std"]
    assert frame["station"].to_list() == ["a", "b", "c"]
    assert frame["y"].to_list() == [-0.14, 0.0, 0.14]
    assert frame["mean"][1] == pytest.approx(0.0, abs=0.2)  # sin(0)
    assert 0.05 < frame["std"][1] < 0.3  # about the noise, 0.1


def test_to_frame_with_an_interval_adds_lower_and_upper_columns(sine_fit):
    frame = sine_fit.predict(pl.DataFrame({"x": [0.0]})).to_frame(interval=0.95)

    assert frame.columns == ["x", "mean", "std", "lower", "upper"]
    # sin(0) is 0 and the noise is 0.1: a new observation is within 0 ± 1.96 * 0.1, 95% of the time.
    assert frame["lower"][0] == pytest.approx(-0.196, abs=0.04)
    assert frame["upper"][0] == pytest.approx(0.196, abs=0.04)


def test_without_noise_the_spread_is_of_the_curve_alone_and_the_mean_is_the_same(
    sine_fit,
):
    # 80 points with noise of standard deviation 0.1 pin the curve down to well under the noise.
    rows = pl.DataFrame({"x": [0.0]})

    observation = sine_fit.predict(rows)
    curve = sine_fit.predict(rows, include_noise=False)

    assert observation.std()[0] == pytest.approx(0.1, abs=0.02)
    assert curve.std()[0] < 0.07
    assert curve.mean() == pytest.approx(observation.mean())


def test_changing_an_array_a_prediction_gave_does_not_change_the_prediction(sine_fit):
    prediction = sine_fit.predict(pl.DataFrame({"x": [0.0, 0.5, 1.0]}))
    original = prediction.mean().copy()

    prediction.mean()[:] = -999.0

    np.testing.assert_array_equal(prediction.mean(), original)
    np.testing.assert_array_equal(prediction.to_frame()["mean"].to_numpy(), original)


def test_predicting_zero_rows_gives_empty_results(sine_fit):
    prediction = sine_fit.predict(pl.DataFrame({"x": []}, schema={"x": pl.Float64}))

    assert prediction.mean().shape == (0,)
    assert prediction.std().shape == (0,)
    assert prediction.interval().shape == (0, 2)


def test_a_fit_and_a_prediction_equal_only_themselves_and_can_be_dictionary_keys(
    sine_fit, pandas_fit
):
    rows = pl.DataFrame({"x": [0.0]})
    first, second = sine_fit.predict(rows), sine_fit.predict(rows)

    kept = {sine_fit: "sine", pandas_fit: "pandas", first: "first", second: "second"}

    assert sine_fit != pandas_fit  # the same numbers, fitted twice
    assert first != second
    assert [kept[sine_fit], kept[first]] == ["sine", "first"]
    assert len(kept) == 4


def test_printing_a_prediction_shows_its_table_and_whether_it_includes_the_noise(
    sine_fit,
):
    rows = pl.DataFrame({"station": ["Utrecht"], "x": [0.0]})

    observation = sine_fit.predict(rows)
    curve = sine_fit.predict(rows, include_noise=False)

    assert "measurement noise included" in repr(observation)
    assert "without measurement noise" in repr(curve)
    for shown in (repr(observation), str(curve), observation._repr_html_()):
        assert "Utrecht" in shown
        assert "mean" in shown
        assert "std" in shown
    assert "<table" in observation._repr_html_()


def test_printing_a_prediction_whose_columns_clash_shows_only_the_predictions(
    sine_fit,
):
    shown = repr(sine_fit.predict(CLASHING))

    assert "123.0" not in shown  # the rows' own `mean` column
    assert "pass suffix= to predict" in shown


def test_a_fitted_model_checks_rows_to_predict_which_need_no_target(sine_fit):
    checked = sine_fit.check(pl.DataFrame({"x": [0.0]}))

    assert sine_fit.predict(checked).mean() == pytest.approx([0.0], abs=0.2)  # sin(0)
    with pytest.raises(ValueError, match=r"missing 1 column the model needs: 'x'"):
        sine_fit.check(pl.DataFrame({"y": [0.0]}))


@pytest.mark.parametrize("batch_size", [0, -1, 2.5, True, "3", None])
def test_a_batch_size_that_is_not_a_whole_number_above_zero_is_refused(
    sine_fit, batch_size
):
    with pytest.raises(
        ValueError,
        match=r"batch_size is how many rows are predicted at a time, so it must be a "
        r"whole number above zero",
    ):
        sine_fit.predict(pl.DataFrame({"x": [0.0]}), batch_size=batch_size)


@pytest.mark.parametrize("suffix", [None, 1])
def test_a_suffix_that_is_not_text_is_refused(sine_fit, suffix):
    with pytest.raises(
        ValueError, match=r"suffix is text added to the names .* must be a string"
    ):
        sine_fit.predict(pl.DataFrame({"x": [0.0]}), suffix=suffix)


@pytest.mark.slow
def test_prediction_in_several_batches_keeps_the_rows_in_order_and_the_pandas_index():
    fitted = GPRegressor(features=["x"], target="y").fit(
        pd.DataFrame(SQUARES.to_dict(as_series=False))
    )
    x = [
        2.0,
        -2.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        0.0,
    ]  # not sorted, so a wrong row order shows
    rows = pd.DataFrame({"x": x}, index=[50 - 3 * i for i in range(8)])  # not 0..n

    prediction = fitted.predict(rows, batch_size=3)  # 8 rows: batches of 3, 3 and 2
    interval = prediction.interval()

    assert prediction.mean() == pytest.approx(
        [4.0, 4.0, 1.0, 1.0, 0.25, 0.25, 2.25, 0.0], abs=0.3
    )  # x squared
    assert (interval["lower"].to_numpy() < prediction.mean()).all()
    assert (prediction.mean() < interval["upper"].to_numpy()).all()
    assert interval.index.tolist() == [50, 47, 44, 41, 38, 35, 32, 29]
    assert prediction.to_frame()["x"].tolist() == x


@pytest.fixture(scope="module")
def lapse_fit():
    """Temperature is 20,000 plus 0.1 per metre of altitude, give or take 5, and does not depend on latitude."""
    data = pl.DataFrame(
        {
            "latitude": [50.0, 50.4, 50.8, 51.2, 51.6, 52.0],
            "altitude": [0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0],
            "temperature": [20005.0, 20055.0, 20125.0, 20175.0, 20245.0, 20295.0],
        }
    )
    return GPRegressor(
        features=["latitude", "altitude"], target="temperature", kernel="matern32"
    ).fit(data)


@pytest.mark.slow
def test_columns_on_very_different_scales_are_predicted_in_original_units(lapse_fit):

    prediction = lapse_fit.predict(
        pl.DataFrame({"latitude": [51.0], "altitude": [1500.0]})
    )

    assert prediction.mean() == pytest.approx(
        [20150.0], abs=10.0
    )  # 20,000 + 0.1 * 1500
    # The spread is in degrees too: a spread left in rescaled units would be about 0.04.
    assert 1.0 < prediction.std()[0] < 30.0


@pytest.mark.slow
def test_covariance_of_the_same_row_twice_is_in_original_units_with_the_noise_on_the_diagonal_only(
    lapse_fit,
):

    rows = pl.DataFrame({"latitude": [51.0, 51.0], "altitude": [1500.0, 1500.0]})

    matrix = lapse_fit.predict(rows).covariance()
    curve = lapse_fit.predict(rows, include_noise=False).covariance()

    # A covariance left in rescaled units would be about 0.001, as the spread would.
    assert matrix[0, 0] > 1.0
    assert matrix[0, 0] == pytest.approx(matrix[1, 1])
    # The same row twice moves together, but each observation also has its own noise.
    assert 0.0 < matrix[0, 1] < matrix[0, 0]
    # Without the noise both rows are the same point of the same curve.
    assert curve[0, 1] == pytest.approx(curve[0, 0], rel=1e-3)
    assert curve[0, 1] == pytest.approx(matrix[0, 1], rel=1e-3)


def test_columns_may_come_in_any_order_when_fitting_and_predicting():
    # y is a plus ten times b, so swapping the two columns would give 23 below, not 32.
    data = pl.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [0.0, 2.0, 1.0, 4.0, 3.0, 5.0],
            "y": [0.0, 21.0, 12.0, 43.0, 34.0, 55.0],
        }
    )
    model = GPRegressor(features=["a", "b"], target="y")

    for columns in (["a", "b", "y"], ["y", "b", "a"]):
        fitted = model.fit(data.select(columns))
        for new in (
            pl.DataFrame({"a": [2.0], "b": [3.0]}),
            pl.DataFrame({"b": [3.0], "a": [2.0]}),
            pl.DataFrame(
                {"b": [3.0], "y": [0.0], "a": [2.0]}
            ),  # a target column in between takes no part
        ):
            assert fitted.predict(new).mean() == pytest.approx([32.0], abs=1.0)


def test_a_column_that_never_varies_still_fits_and_predicts_sensibly():
    x = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    data = pl.DataFrame({"x": x, "site": [12.5] * 6, "y": [2 * value for value in x]})

    prediction = (
        GPRegressor(features=["x", "site"], target="y")
        .fit(data)
        .predict(pl.DataFrame({"x": [1.0], "site": [12.5]}))
    )

    assert prediction.mean() == pytest.approx([2.0], abs=0.2)
    assert prediction.std()[0] < 1.0


@pytest.mark.slow
def test_a_target_that_never_varies_is_predicted_as_that_constant():
    data = pl.DataFrame({"x": np.linspace(0.0, 1.0, 20), "y": np.full(20, 42.0)})

    prediction = (
        GPRegressor(features=["x"], target="y")
        .fit(data)
        .predict(pl.DataFrame({"x": [0.25, 0.75]}))
    )

    assert prediction.mean() == pytest.approx([42.0, 42.0], abs=1e-3)
    assert np.isfinite(prediction.std()).all()


def test_to_frame_refuses_to_repeat_a_column_name_but_the_other_results_still_work(
    sine_fit,
):
    prediction = sine_fit.predict(CLASHING)

    # Only the names it would add count: `lower` and `upper` only with an interval.
    with pytest.raises(
        ValueError,
        match=r"columns named 'mean' and 'std', which skgp would add for the predictions\. "
        r"Pass suffix= to predict",
    ):
        prediction.to_frame()
    with pytest.raises(ValueError, match=r"'mean', 'std', 'lower' and 'upper', which"):
        prediction.to_frame(interval=0.95)
    with pytest.raises(
        ValueError, match=r"'mean_p', .*even with suffix '_p'\. Pass a different suffix"
    ):
        sine_fit.predict(
            pl.DataFrame({"x": [0.0], "mean_p": [1.0]}), suffix="_p"
        ).to_frame()
    assert prediction.mean() == pytest.approx([0.0], abs=0.2)  # sin(0)
    assert prediction.interval().columns == ["lower", "upper"]


def test_a_suffix_goes_on_every_column_skgp_adds_whether_or_not_it_clashes(sine_fit):
    for rows in (pl.DataFrame({"x": [0.0]}), CLASHING):
        prediction = sine_fit.predict(rows, suffix="_predicted")

        assert prediction.to_frame(interval=0.95).columns == [
            *rows.columns,
            "mean_predicted",
            "std_predicted",
            "lower_predicted",
            "upper_predicted",
        ]
        assert prediction.interval().columns == ["lower_predicted", "upper_predicted"]
