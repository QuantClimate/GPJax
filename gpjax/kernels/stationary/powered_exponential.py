# Copyright 2022 The thomaspinder Contributors. All Rights Reserved.
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
import jax.numpy as jnp
from jaxtyping import Float
import paramax
from paramax import AbstractUnwrappable

from gpjax.kernels.base import val
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.stationary.utils import euclidean_distance
from gpjax.parameters import SigmoidBounded
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class PoweredExponential(StationaryKernel):
    r"""The powered exponential family of kernels.

    Computes the covariance for pairs of inputs $(x, y)$ with length-scale parameter
    $\ell$, variance $\sigma^2$ and power $\kappa$.
    $$
    k(x, y)=\sigma^2\exp\Bigg(-\Big(\frac{\lVert x-y\rVert_2}{\ell}\Big)^\kappa\Bigg)
    $$

    This also equivalent to the symmetric generalized normal distribution.
    See Diggle and Ribeiro (2007) - "Model-based Geostatistics".
    and
    https://en.wikipedia.org/wiki/Generalized_normal_distribution#Symmetric_version

    The kernel is positive definite in every dimension only for
    $0 < \kappa \le 2$. A float `power` in $(0, 2)$ becomes a trainable
    parameter that the optimiser keeps inside $(0, 2)$. `power=2.0` gives the
    RBF kernel and is held fixed, because a bounded parameter cannot sit on its
    bound; to learn the power, start it inside the interval, for example at
    1.9. A parameter that you pass yourself must keep the power in $(0, 2]$.
    """

    name: ClassVar[str] = "Powered Exponential"
    power: tp.Any

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[LengthscaleCompatible, AbstractUnwrappable] = 1.0,
        variance: tp.Union[ScalarFloat, AbstractUnwrappable] = 1.0,
        power: tp.Union[ScalarFloat, AbstractUnwrappable] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        r"""Initializes the kernel.

        Args:
            active_dims: the indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            power: the power of the kernel κ, in $(0, 2]$.
            n_dims: the number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: the computation engine that the kernel uses to compute the
                covariance matrix.

        Raises:
            ValueError: if the power is not in $(0, 2]$.
        """
        self.power = _wrap_power(power)

        super().__init__(active_dims, lengthscale, variance, n_dims, compute_engine)

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        x = self.slice_input(x) / val(self.lengthscale)
        y = self.slice_input(y) / val(self.lengthscale)
        power_val = val(self.power)
        K = val(self.variance) * jnp.exp(-(euclidean_distance(x, y) ** power_val))
        return K.squeeze()


def _wrap_power(power: tp.Any) -> tp.Any:
    value = float(val(power))
    if not 0.0 < value <= 2.0:
        raise ValueError(
            "Expected `power` in (0, 2], where the powered exponential kernel is "
            f"positive definite. Got {value}."
        )
    if isinstance(power, AbstractUnwrappable):
        return power
    power = jnp.asarray(power, dtype=float)
    if value == 2.0:
        return paramax.non_trainable(power)
    return SigmoidBounded(power, low=0.0, high=2.0)
