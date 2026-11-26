"""Fixtures shared by the tests. The plain helpers they use are in `helpers.py`.

A fit costs up to a second or two, much of it JAX compiling. A fitted model never changes,
so the models that many tests only read are fitted once per session here and shared.
"""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from helpers import (
    UNFINISHED_FITS,
    WEATHER_GROUPS,
    capturing_records,
    fit_and_watch,
    sine_data,
)
import skgp
from skgp import GPClassifier, GPRegressor, SparseGPClassifier


@pytest.fixture
def records():
    with capturing_records() as got:
        yield got


@pytest.fixture(scope="session")
def sine_fit():
    fitted = GPRegressor(features=["x"], target="y").fit(sine_data())
    # The first prediction of a session is slow, because JAX compiles as it goes. Pay for it
    # here, once, so that no test that only reads this model takes over a second for it.
    fitted.predict(pl.DataFrame({"x": [0.0]}))
    return fitted


@pytest.fixture(scope="session")
def pandas_fit():
    return GPRegressor(features=["x"], target="y").fit(
        pd.DataFrame(sine_data().to_dict(as_series=False))
    )


@pytest.fixture(scope="session")
def diverged_fit():
    """A fit whose objective is NaN: targets near the largest float, through the public fit alone."""
    rng = np.random.default_rng(0)
    data = pl.DataFrame(
        {"x": np.linspace(0.0, 1.0, 10), "y": rng.choice([-1.0, 1.0], 10) * 1.7e308}
    )
    # The values overflow on purpose: this fixture is a fit that fails.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", skgp.ConvergenceWarning)
        return GPRegressor(features=["x"], target="y").fit(data)


@pytest.fixture(scope="session")
def unfinished():
    """One fit of each kind in `UNFINISHED_FITS`, stopped early by a small `iterations`: with
    the fitted model, the warnings it issued and the verbose log records it emitted."""
    data = sine_data(40)
    classes = data.with_columns(above=pl.col("y") > 0)
    return {
        kind: fit_and_watch(
            model,
            classes if isinstance(model, (GPClassifier, SparseGPClassifier)) else data,
            iterations=iterations,
        )
        for kind, (model, iterations) in UNFINISHED_FITS.items()
    }


@pytest.fixture(scope="session")
def weather_fit():
    """A `GPClassifier` fitted to `WEATHER_GROUPS`: dry below zero and rain above."""
    fitted = GPClassifier("x", "weather").fit(WEATHER_GROUPS)
    # JAX compiles on the first prediction; pay for it here, once, not in a test.
    fitted.predict(WEATHER_GROUPS.head(1))
    return fitted
