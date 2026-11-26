"""Plain helpers shared by the tests: made-up data, a recording schema, and fits cut short.

The fixtures are in `conftest.py`. These are ordinary functions, imported by name.
"""

from contextlib import contextmanager
from types import SimpleNamespace
import warnings

from loguru import logger
import numpy as np
import pandas as pd
import polars as pl

from skgp import GPClassifier, GPRegressor, SparseGPClassifier, SparseGPRegressor

NOISE_SD = 0.1

# One fit of each kind, cut short by a small `iterations`: the model and its iterations. The
# `unfinished` fixture in conftest.py fits each to `sine_data(40)`, and each classifier to
# whether y is above zero there. Forty rows in minibatches of 20 make one pass in 2 iterations.
UNFINISHED_FITS = {
    "exact": (GPRegressor(features=["x"], target="y"), 1),
    "sparse on all rows": (
        SparseGPRegressor(features=["x"], target="y", num_inducing=10),
        1,
    ),
    "sparse on minibatches": (
        SparseGPRegressor(
            features=["x"], target="y", num_inducing=10, minibatch_size=20
        ),
        2,
    ),
    "heteroscedastic at every row": (
        GPRegressor(features=["x"], target="y", learned_noise="heteroscedastic"),
        1,
    ),
    "heteroscedastic sparse on all rows": (
        SparseGPRegressor(
            features=["x"], target="y", learned_noise="heteroscedastic", num_inducing=10
        ),
        1,
    ),
    "heteroscedastic sparse on minibatches": (
        SparseGPRegressor(
            features=["x"],
            target="y",
            learned_noise="heteroscedastic",
            num_inducing=10,
            minibatch_size=20,
        ),
        2,
    ),
    "classifier": (GPClassifier(features=["x"], target="above"), 1),
    "sparse classifier on all rows": (
        SparseGPClassifier(features=["x"], target="above", num_inducing=10),
        1,
    ),
    "sparse classifier on minibatches": (
        SparseGPClassifier(
            features=["x"], target="above", num_inducing=10, minibatch_size=20
        ),
        2,
    ),
}
KINDS = list(UNFINISHED_FITS)

# Two groups of 15 rows each, far apart along x: below zero it is dry, above zero it rains.
WEATHER_GROUPS = pl.DataFrame(
    {
        "x": np.concatenate([np.linspace(-3.0, -1.0, 15), np.linspace(1.0, 3.0, 15)]),
        "weather": ["dry"] * 15 + ["rain"] * 15,
    }
)


def sine_data(n=80, noise_sd=NOISE_SD, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    return pl.DataFrame({"x": x, "y": np.sin(3 * x) + noise_sd * rng.normal(size=n)})


SCALES_NOISE_SD = 5.0
SCALES_FEATURES = ["latitude", "altitude"]


def scales_truth(latitude, altitude):
    return 20_000.0 + 400.0 * np.sin(3 * (latitude - 51.0)) + 0.1 * altitude


def scales_data(n=150, seed=1):
    """Two inputs on very different scales and a target far from zero, so a value left in
    rescaled units would be visibly wrong."""
    rng = np.random.default_rng(seed)
    latitude = rng.uniform(50.0, 52.0, n)
    altitude = rng.uniform(0.0, 3000.0, n)
    temperature = scales_truth(latitude, altitude) + SCALES_NOISE_SD * rng.normal(
        size=n
    )
    return pl.DataFrame(
        {"latitude": latitude, "altitude": altitude, "temperature": temperature}
    )


def scales_rows():
    return pl.DataFrame(
        {"latitude": [50.2, 51.0, 51.9], "altitude": [10.0, 1500.0, 2900.0]}
    )


class Recorder:
    """A plain object that satisfies the schema contract and records what it was given.
    Setting `raises` later makes it refuse every frame from then on."""

    def __init__(self, returns=None, raises=None):
        self.seen = []
        self.returns = returns
        self.raises = raises

    def validate(self, frame):
        self.seen.append(frame)
        if self.raises is not None:
            raise self.raises
        return self.returns


def good_frame(n=10):
    """`n` rows of y = x squared, for x = 0, 1, 2, ..."""
    return pl.DataFrame(
        {"x": [float(i) for i in range(n)], "y": [float(i) ** 2 for i in range(n)]}
    )


# A line rising 2 per unit from 1000, with wobble of about 5 either way: 12 rows, in units far
# from zero, so a number left in rescaled units would be visibly wrong. At x = 35 it is about 1070.
NOISY_LINE = pl.DataFrame(
    {
        "x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0],
        "y": [
            1004.0,
            1015.0,
            1046.0,
            1057.0,
            1074.0,
            1105.0,
            1122.0,
            1134.0,
            1165.0,
            1176.0,
            1203.0,
            1218.0,
        ],
    }
)


def both_libraries(columns):
    return [pl.DataFrame(columns), pd.DataFrame(columns)]


@contextmanager
def capturing_records():
    got = []
    handler = logger.add(
        lambda message: got.append(message.record), level=0, format="{message}"
    )
    try:
        yield got
    finally:
        logger.remove(handler)


def fit_and_watch(model, data, iterations=None, verbose=True):
    """Fit, cut short to `iterations` when given, and keep what it emitted: a namespace of the
    fitted model, the log records and the warnings."""
    cut_short = {} if iterations is None else {"iterations": iterations}
    with warnings.catch_warnings(record=True) as caught, capturing_records() as got:
        warnings.simplefilter("always")
        fitted = model.fit(data, verbose=verbose, **cut_short)
    return SimpleNamespace(fitted=fitted, records=got, warnings=caught)
