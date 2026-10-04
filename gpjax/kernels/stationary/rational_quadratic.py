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
from jaxtyping import Float
from paramax import AbstractUnwrappable

from gpjax.kernels.base import val
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.stationary.utils import squared_distance
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class RationalQuadratic(StationaryKernel):
    r"""The Rational Quadratic kernel.

    Computes the covariance for pairs of inputs $(x, y)$ with lengthscale parameter
    $\ell$, variance $\sigma^2$ and shape parameter $\alpha$.
    $$
    k(x,y)=\sigma^2\Bigg(1+\frac{\lVert x-y\rVert^2_2}{2\alpha\ell^2}\Bigg)^{-\alpha}
    $$

    As $\alpha \to \infty$ this recovers the :class:`~gpjax.kernels.RBF` kernel; it is
    equivalently a scale mixture of RBF kernels with a Gamma-distributed inverse
    squared lengthscale.
    """

    name: ClassVar[str] = "Rational Quadratic"
    isotropic_radial: ClassVar[bool] = True
    alpha: tp.Any

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[LengthscaleCompatible, AbstractUnwrappable] = 1.0,
        variance: tp.Union[ScalarFloat, AbstractUnwrappable] = 1.0,
        alpha: tp.Union[ScalarFloat, AbstractUnwrappable] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            alpha: the alpha parameter of the kernel α.
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """
        self.alpha = alpha

        super().__init__(active_dims, lengthscale, variance, n_dims, compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / val(self.lengthscale)
        y = self.slice_input(y) / val(self.lengthscale)
        alpha_val = val(self.alpha)
        K = val(self.variance) * (1 + 0.5 * squared_distance(x, y) / alpha_val) ** (
            -alpha_val
        )
        return K.squeeze()
