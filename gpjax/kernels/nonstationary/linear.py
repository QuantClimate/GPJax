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

import jax.numpy as jnp
from jaxtyping import Float
from paramax import AbstractUnwrappable

from gpjax.kernels.base import AbstractKernel, val
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.parameters import NonNegativeReal
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class Linear(AbstractKernel):
    r"""The linear kernel.

    Computes the covariance for pairs of inputs $(x, y)$ with variance $\sigma^2$:
    $$
    k(x, y) = \sigma^2 x^{\top}y
    $$
    """

    name: str = "Linear"
    variance: tp.Any

    def __init__(
        self,
        active_dims: list[int] | slice | None = None,
        variance: ScalarFloat | AbstractUnwrappable = 1.0,
        n_dims: int | None = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            variance: the variance of the kernel σ.
            n_dims: The number of input dimensions.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """

        if isinstance(variance, AbstractUnwrappable):
            self.variance = variance
        else:
            self.variance = NonNegativeReal(variance)

        super().__init__(active_dims, n_dims, compute_engine)

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        K = val(self.variance) * jnp.matmul(x.T, y)
        return K.squeeze()
