"""Every type alias in skgp: the kinds of value a setting, an argument or a log field may take.

A type alias lives here and nowhere else; classes live with the code that uses them.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Annotated, Any, Literal

from gpjax.kernels import AbstractKernel
from gpjax.mean_functions import AbstractMeanFunction
import narwhals.stable.v2 as nw
import numpy as np
from pydantic import BeforeValidator, Field


def _as_python_int(value: object) -> object:
    return int(value) if isinstance(value, np.integer) else value


WholeNumber = Annotated[int, BeforeValidator(_as_python_int), Field(strict=True)]
"""A whole number. A numpy integer is accepted; a float, a string and a bool are not, even
when they look like one (`1.0`, `"3"`, `True`)."""

Task = Literal["regression", "classification"]
"""What a model predicts: a number (regression) or one of two classes (classification)."""

KernelName = Literal["rbf", "matern32", "matern52"]
"""The kernels a model may use by name."""

Kernel = KernelName | AbstractKernel
"""The kernel of a model: one of the names, or any GPJax kernel object."""

MeanFunctionName = Literal["zero", "constant", "linear"]
"""The mean functions a model may use by name."""

MeanFunction = MeanFunctionName | AbstractMeanFunction
"""The mean function of a model: one of the names, or any GPJax mean function object."""

LearnedNoise = Literal["constant", "heteroscedastic"]
"""How the noise a regressor learns varies: the same at every row, or with the inputs."""

Likelihood = Literal["Gaussian", "HeteroscedasticGaussian", "Bernoulli"]
"""How the target is modelled given the curve: with Gaussian noise, the same at every row or
varying with the inputs (regression), or as one of two classes through the probit
(classification)."""

Frame = nw.DataFrame[Any]
"""A dataframe of any library narwhals supports, as skgp handles it inside (see `_frames`)."""

Scoring = (
    str
    | Callable[..., float]
    | Iterable[str | Callable[..., float]]
    | Mapping[str, str | Callable[..., float]]
)
"""What `fitted.score` takes as `scoring`, as `cross_val_score` does: a scorer's name, a
scorer (called with an estimator, `X` and `y`), a list of either, or a dict from the name a
`Score` gives each value to its scorer."""

OnIteration = Callable[[int, float], None]
"""Called after each iteration of the optimiser with its number (from 1) and the objective."""
