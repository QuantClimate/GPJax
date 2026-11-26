import polars as pl
import pytest

from skgp import GPRegressor


def test_a_column_with_a_tiny_but_real_spread_is_scaled_by_its_own_spread():
    # A level of a million with a spread of 0.001: relative spread 1e-9, far above rounding noise, so
    # this column does vary. If it were treated as constant, the model could not tell the rows apart
    # and would predict the mean of y, 2.5, between the first two rows instead of 0.5.
    data = pl.DataFrame(
        {"x": [1e6 + 0.001 * i for i in range(6)], "y": [float(i) for i in range(6)]}
    )

    prediction = (
        GPRegressor(features=["x"], target="y")
        .fit(data)
        .predict(pl.DataFrame({"x": [1e6 + 0.0005]}))
    )

    assert prediction.mean() == pytest.approx([0.5], abs=0.1)
