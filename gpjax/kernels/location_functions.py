# Copyright 2026 The thomaspinder Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Location functions: kernel parameters that change with input location.

A location function gives one value at each input location, for example the
standard deviation of a :class:`~gpjax.kernels.VaryingAmplitude` kernel or the
lengthscale of a :class:`~gpjax.kernels.Gibbs` kernel.

A location function is not a mean function. A mean function describes the
Gaussian process itself, but a location function describes its covariance.
The two types also have different contracts:

- A location function evaluates one input point of shape `(D,)` and returns a
  scalar. JAX's `vmap` then evaluates it once for each row of a kernel matrix.
- A location function selects its own input columns with `active_dims`. This
  lets a kernel measure distance over spatial columns while its location
  function reads covariate columns, such as elevation.
- A location function returns a value on the log scale. The kernel applies
  `exp`, so the parameter it controls is always positive and a coefficient
  multiplies that parameter: a weight $\beta$ multiplies the parameter by
  $e^{\beta}$ for each unit of its column.
"""

import abc

import beartype.typing as tp
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)
from paramax import AbstractUnwrappable

from gpjax.parameters import (
    Real,
    val,
)
from gpjax.summary import _SummaryMixin
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class AbstractLocationFunction(_SummaryMixin, eqx.Module):
    r"""Base class for location functions.

    A subclass implements `__call__`, which maps one input point to a scalar on
    the log scale. Use `slice_input` to read only the columns in
    `active_dims`.
    """

    active_dims: tp.Union[list[int], slice] = eqx.field(
        static=True, default_factory=lambda: slice(None)
    )

    @abc.abstractmethod
    def __call__(self, x: Num[Array, " D"]) -> ScalarFloat:
        r"""Evaluate the location function at one input point.

        Args:
            x: one input point, with all columns of the data.

        Returns:
            The log of the parameter value at `x`.
        """
        ...

    def slice_input(self, x: Num[Array, " D"]) -> Num[Array, " Q"]:
        r"""Select the columns in `active_dims` from one input point.

        Args:
            x: one input point, with all columns of the data.

        Returns:
            The selected columns.
        """
        return x[..., self.active_dims]


class Constant(AbstractLocationFunction):
    r"""A location function with the same value at all locations.

    $$\log g(x) = c$$

    With this function, a nonstationary kernel is equal to its stationary base
    kernel with its scale multiplied by $e^{c}$. The default $c = 0$ gives the
    base kernel exactly.
    """

    value: tp.Any

    def __init__(self, value: tp.Union[ScalarFloat, AbstractUnwrappable] = 0.0):
        """Initialise the location function.

        Args:
            value: the log value $c$. A float is wrapped as a trainable `Real`.
        """
        if isinstance(value, AbstractUnwrappable):
            self.value = value
        else:
            self.value = Real(jnp.asarray(value, dtype=float))
        self.active_dims = slice(None)

    def __call__(self, x: Num[Array, " D"]) -> ScalarFloat:
        return jnp.asarray(val(self.value)).squeeze()


class Linear(AbstractLocationFunction):
    r"""A location function that is log-linear in selected input columns.

    $$\log g(x) = \beta^{\top} x_{\mathcal{A}} + b$$

    Here $x_{\mathcal{A}}$ are the columns in `active_dims`. The weights
    $\beta$ start at zero, so a model starts as its stationary base kernel and
    moves away from it only as far as the data supports.

    By default the function has no intercept ($b = 0$): the base kernel holds
    the overall scale, and the location function holds only the change over
    location. An intercept would duplicate the variance or lengthscale of the
    base kernel, and the data could not separate the two.

    Standardise the covariate columns before fitting, so that the weights have
    similar scales.
    """

    weights: tp.Any
    bias: tp.Any

    def __init__(
        self,
        active_dims: list[int],
        weights: tp.Union[
            Float[Array, " Q"], list[float], AbstractUnwrappable, None
        ] = None,
        intercept: bool = False,
    ):
        r"""Initialise the location function.

        Args:
            active_dims: the indices of the covariate columns. This argument is
                required, so that the function cannot read the spatial columns
                by mistake.
            weights: the initial weights $\beta$, one for each column in
                `active_dims`. Defaults to zeros.
            intercept: whether to add a trainable intercept $b$, which starts
                at zero.
        """
        if not isinstance(active_dims, list) or not active_dims:
            raise TypeError(
                "Expected `active_dims` to be a non-empty list of column indices. "
                f"Got {active_dims!r}."
            )

        if weights is None:
            weights = jnp.zeros(len(active_dims))
        if not isinstance(weights, AbstractUnwrappable):
            weights = jnp.asarray(weights, dtype=float)
            if weights.shape != (len(active_dims),):
                raise ValueError(
                    f"Expected one weight for each of the {len(active_dims)} "
                    f"columns in `active_dims`. Got weights of shape "
                    f"{weights.shape}."
                )
            weights = Real(weights)

        self.active_dims = active_dims
        self.weights = weights
        self.bias = Real(jnp.array(0.0)) if intercept else None

    def __call__(self, x: Num[Array, " D"]) -> ScalarFloat:
        value = jnp.dot(self.slice_input(x), val(self.weights))
        if self.bias is not None:
            value = value + val(self.bias)
        return value.squeeze()


__all__ = [
    "AbstractLocationFunction",
    "Constant",
    "Linear",
]
