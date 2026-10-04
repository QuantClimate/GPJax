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

from typing import ClassVar

import beartype.typing as tp
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from paramax import AbstractUnwrappable

from gpjax.kernels.base import (
    AbstractKernel,
    val,
)
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.parameters import (
    NonNegativeReal,
    PositiveReal,
    SigmoidBounded,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class Gneiting(AbstractKernel):
    r"""The Gneiting nonseparable space–time kernel.

    Computes the covariance for a pair of inputs with spatial separation $h$
    over the space columns and time lag $u$ over the time column (Gneiting,
    2002, eq. 14):
    $$
    k(h, u) = \frac{\sigma^2}{\psi(u)^{d/2}}
    \exp\!\left(-\frac{(\lVert h\rVert/\ell_s)^{2\gamma}}
    {\psi(u)^{\beta\gamma}}\right),
    \qquad \psi(u) = \left(\frac{\lvert u\rvert}{\ell_t}\right)^{2\alpha} + 1,
    $$
    where $d$ is the number of space columns.

    The kernel is stationary, but it is not separable: as the time lag grows,
    $\psi(u)$ grows and the spatial correlation decays more slowly. The
    interaction parameter $\beta \in [0, 1]$ controls this effect, and
    $\beta = 0$ gives the separable product of a powered exponential kernel in
    space and a generalised Cauchy kernel in time. $\alpha \in (0, 1]$ and
    $\gamma \in (0, 1]$ set the smoothness in time and in space.

    The trainable parameters $\alpha$, $\beta$ and $\gamma$ are bounded to the
    open interval $(0, 1)$. To fix one of them at a bound, pass a
    non-trainable value, for example `paramax.non_trainable(jnp.array(1.0))`.

    The kernel has two lengthscales and no closed-form spectral density, so it
    is not a :class:`StationaryKernel` subclass and it does not support random
    Fourier features.
    """

    name: ClassVar[str] = "Gneiting"
    space_dims: list[int] = eqx.field(static=True)
    time_dim: int = eqx.field(static=True)
    variance: tp.Any
    space_lengthscale: tp.Any
    time_lengthscale: tp.Any
    alpha: tp.Any
    beta: tp.Any
    gamma: tp.Any

    def __init__(
        self,
        space_dims: list[int],
        time_dim: int,
        variance: tp.Union[ScalarFloat, AbstractUnwrappable] = 1.0,
        space_lengthscale: tp.Union[ScalarFloat, AbstractUnwrappable] = 1.0,
        time_lengthscale: tp.Union[ScalarFloat, AbstractUnwrappable] = 1.0,
        alpha: tp.Union[ScalarFloat, AbstractUnwrappable] = 0.5,
        beta: tp.Union[ScalarFloat, AbstractUnwrappable] = 0.5,
        gamma: tp.Union[ScalarFloat, AbstractUnwrappable] = 0.5,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        r"""Initialise the kernel.

        Args:
            space_dims: the indices of the space columns.
            time_dim: the index of the time column.
            variance: the variance $\sigma^2$.
            space_lengthscale: the spatial lengthscale $\ell_s$.
            time_lengthscale: the temporal lengthscale $\ell_t$.
            alpha: the smoothness in time, $\alpha \in (0, 1]$.
            beta: the space–time interaction, $\beta \in [0, 1]$.
            gamma: the smoothness in space, $\gamma \in (0, 1]$.
            compute_engine: the computation engine that the kernel uses to
                compute its covariance matrices.

        Raises:
            ValueError: if the columns are not valid, or if a float value of
                `alpha`, `beta` or `gamma` is not in the open interval
                $(0, 1)$.
        """
        _check_columns(space_dims, time_dim)
        self.space_dims = list(space_dims)
        self.time_dim = time_dim
        self.variance = _wrap(variance, NonNegativeReal)
        self.space_lengthscale = _wrap(space_lengthscale, PositiveReal)
        self.time_lengthscale = _wrap(time_lengthscale, PositiveReal)
        self.alpha = _wrap_unit(alpha, "alpha")
        self.beta = _wrap_unit(beta, "beta")
        self.gamma = _wrap_unit(gamma, "gamma")
        super().__init__(compute_engine=compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        h = (x[..., self.space_dims] - y[..., self.space_dims]) / val(
            self.space_lengthscale
        )
        u = (x[..., self.time_dim] - y[..., self.time_dim]) / val(self.time_lengthscale)
        gamma = val(self.gamma)
        psi = _power(u**2, val(self.alpha)) + 1.0
        space_term = _power(jnp.sum(h**2), gamma) / psi ** (val(self.beta) * gamma)
        dims = len(self.space_dims)
        K = val(self.variance) * psi ** (-0.5 * dims) * jnp.exp(-space_term)
        return K.squeeze()


def _power(t: Float[Array, ""], p: Float[Array, ""]) -> Float[Array, ""]:
    # t**p with t >= 0. The gradient of 0**p with respect to p, or of t**p at
    # t = 0 for p < 1, is not finite, so the zero case takes a separate branch.
    positive = t > 0
    safe = jnp.where(positive, t, 1.0)
    return jnp.where(positive, safe**p, 0.0)


def _check_columns(space_dims: tp.Any, time_dim: tp.Any) -> None:
    if (
        not isinstance(space_dims, (list, tuple))
        or not space_dims
        or not all(isinstance(i, int) for i in space_dims)
    ):
        raise ValueError(
            "Expected `space_dims` to be a non-empty list of column indices. "
            f"Got {space_dims!r}."
        )
    if len(set(space_dims)) != len(space_dims):
        raise ValueError(f"`space_dims` has repeated columns: {space_dims!r}.")
    if not isinstance(time_dim, int):
        raise ValueError(
            f"Expected `time_dim` to be one column index. Got {time_dim!r}."
        )
    if time_dim in space_dims:
        raise ValueError(
            f"Column {time_dim} is both a space column and the time column."
        )


def _wrap(value: tp.Any, parameter: type) -> tp.Any:
    if isinstance(value, AbstractUnwrappable):
        return value
    return parameter(jnp.asarray(value, dtype=float))


def _wrap_unit(value: tp.Any, label: str) -> tp.Any:
    if isinstance(value, AbstractUnwrappable):
        return value
    value = jnp.asarray(value, dtype=float)
    if not 0.0 < float(value) < 1.0:
        raise ValueError(
            f"Expected `{label}` in the open interval (0, 1), so that it can be "
            f"trained. Got {float(value)}. To fix it at a bound, pass "
            "`paramax.non_trainable(jnp.array(value))`."
        )
    return SigmoidBounded(value, low=0.0, high=1.0)


__all__ = ["Gneiting"]
