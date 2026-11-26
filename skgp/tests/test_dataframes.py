import numpy as np
import pandas as pd
import polars as pl
import pytest

from skgp import GPRegressor


def test_pandas_in_gives_pandas_out_with_rows_matched_by_position(pandas_fit):
    # pandas_fit is a fit to y = sin(3x) with noise of standard deviation 0.1.
    new = pd.DataFrame({"x": [-1.0, 0.0, 1.0]}, index=[10, 20, 30])

    prediction = pandas_fit.predict(new)
    frame = prediction.to_frame()
    interval = prediction.interval()

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["x", "mean", "std"]
    assert frame.index.tolist() == [10, 20, 30]
    assert frame["x"].tolist() == [-1.0, 0.0, 1.0]
    assert frame["mean"].tolist() == pytest.approx(
        [-0.14, 0.0, 0.14], abs=0.2
    )  # sin(3x)
    assert isinstance(interval, pd.DataFrame)
    assert list(interval.columns) == ["lower", "upper"]
    assert interval.index.tolist() == [10, 20, 30]
    # Each row's interval surrounds that row's own mean (a mismatch of index would leave gaps).
    assert (interval["lower"] < frame["mean"]).all()
    assert (frame["mean"] < interval["upper"]).all()


@pytest.mark.slow
def test_whole_number_inputs_are_fitted_and_predicted_from_either_dataframe_type():
    # y is about twice x, and x is a whole number: 5 is between the rows 4 and 6.
    rows = {"x": [0, 2, 4, 6, 8, 10], "y": [0.1, 3.9, 8.1, 11.9, 16.1, 19.9]}
    model = GPRegressor(features=["x"], target="y")

    from_polars = model.fit(pl.DataFrame(rows)).predict(pl.DataFrame({"x": [5]}))
    from_pandas = model.fit(pd.DataFrame(rows)).predict(pd.DataFrame({"x": [5]}))

    assert from_polars.mean() == pytest.approx([10.0], abs=0.5)
    assert from_pandas.mean() == pytest.approx([10.0], abs=0.5)


def test_single_precision_columns_are_not_mangled():
    # Single precision holds x near 1000 to about 0.00006, so y = 2 * (x - 1000) is still recoverable.
    x = np.array([1000.0, 1000.5, 1001.0, 1001.5, 1002.0, 1002.5], dtype=np.float32)
    data = pl.DataFrame({"x": x, "y": 2 * (x - 1000)})

    prediction = (
        GPRegressor(features=["x"], target="y")
        .fit(data)
        .predict(pl.DataFrame({"x": [1001.25]}))
    )

    assert prediction.mean() == pytest.approx([2.5], abs=0.2)
