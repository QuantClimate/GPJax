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

import typing as tp
from typing import ClassVar

import jax.numpy as jnp
from jaxtyping import Float
from paramax import AbstractUnwrappable

from gpjax.kernels.base import val
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.parameters import PositiveReal
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = Float[Array, " D"] | ScalarArray
LengthscaleCompatible = ScalarFloat | list[float] | Lengthscale


class Periodic(StationaryKernel):
    r"""The periodic kernel.

    Computes the covariance for pairs of inputs $(x, y)$ with length-scale
    parameter $\ell$, variance $\sigma^2$ and period $p$.
    $$
    k(x, y) = \sigma^2 \exp \left( -\frac{1}{2} \sum_{i=1}^{D} \left(\frac{\sin (\pi (x_i - y_i)/p)}{\ell}\right)^2 \right)
    $$
    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    name: ClassVar[str] = "Periodic"
    period: tp.Any

    def __init__(
        self,
        active_dims: list[int] | slice | None = None,
        lengthscale: LengthscaleCompatible | AbstractUnwrappable = 1.0,
        variance: ScalarFloat | AbstractUnwrappable = 1.0,
        period: ScalarFloat | AbstractUnwrappable = 1.0,
        n_dims: int | None = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: the indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            period: the period of the kernel p.
            n_dims: the number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: the computation engine that the kernel uses to compute the
                covariance matrix.
        """

        if isinstance(period, AbstractUnwrappable):
            self.period = period
        else:
            self.period = PositiveReal(period)

        super().__init__(active_dims, lengthscale, variance, n_dims, compute_engine)

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        x = self.slice_input(x)
        y = self.slice_input(y)
        period_val = val(self.period)
        sine_squared = (
            jnp.sin(jnp.pi * (x - y) / period_val) / val(self.lengthscale)
        ) ** 2
        K = val(self.variance) * jnp.exp(-0.5 * jnp.sum(sine_squared, axis=0))
        return K.squeeze()
