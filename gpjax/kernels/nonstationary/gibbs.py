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

import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.location_functions import AbstractLocationFunction
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class Gibbs(AbstractKernel):
    r"""A base kernel whose lengthscale changes with location.

    The Gibbs kernel (Gibbs, 1997), also known as the Paciorek–Schervish
    kernel (Paciorek & Schervish, 2006). A location function $g$ gives
    $\log\ell(x)$, and $\ell(x)$ multiplies the lengthscale of an isotropic
    base kernel $k_0$ with correlation $\rho$ and variance $\sigma^2$:
    $$
    k(x, y) = \sigma^2
    \left(\frac{2\,\ell(x)\,\ell(y)}{\ell(x)^2 + \ell(y)^2}\right)^{d/2}
    \rho\!\left(\sqrt{\frac{2}{\ell(x)^2 + \ell(y)^2}}\,
    \lVert x - y\rVert\right),
    \qquad \ell(x) = \exp g(x).
    $$

    Here $d$ is the number of columns over which the base kernel measures
    distance, and $\lVert\cdot\rVert$ uses the base kernel's lengthscales, so
    an ARD base kernel keeps its shape and $\ell(x)$ scales it. Correlation
    decays faster where $\ell(x)$ is small, for example over mountains, and
    more slowly where it is large. The marginal variance is $\sigma^2$ at every
    location.

    The kernel is positive definite when $\rho$ is positive definite in every
    dimension. Only base kernels with `isotropic_radial = True` meet this
    condition: RBF, the Matérn kernels, RationalQuadratic and
    PoweredExponential.

    The base kernel selects the columns over which it measures distance with
    its own `active_dims`, and the location function selects its covariate
    columns. The wrapper itself always receives every column.
    """

    name: ClassVar[str] = "Gibbs"
    base_kernel: AbstractKernel
    lengthscale: AbstractLocationFunction

    def __init__(
        self,
        base_kernel: AbstractKernel,
        lengthscale: AbstractLocationFunction,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        r"""Initialise the kernel.

        Args:
            base_kernel: the isotropic kernel $k_0$ whose lengthscale changes.
            lengthscale: the location function that gives $\log\ell(x)$.
            compute_engine: the computation engine that the kernel uses to
                compute its covariance matrices.

        Raises:
            TypeError: if `base_kernel` is not an isotropic radial kernel.
        """
        if not getattr(base_kernel, "isotropic_radial", False):
            raise TypeError(
                "Gibbs needs an isotropic radial base kernel: RBF, Matern12, "
                "Matern32, Matern52, RationalQuadratic or PoweredExponential. "
                f"Got {type(base_kernel).__name__}, for which the Gibbs "
                "construction is not guaranteed to be positive definite."
            )
        self.base_kernel = base_kernel
        self.lengthscale = lengthscale
        super().__init__(compute_engine=compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        log_lx = self.lengthscale(x)
        log_ly = self.lengthscale(y)
        # log(ℓ(x)² + ℓ(y)²), computed stably for large or small lengthscales.
        log_sum = jnp.logaddexp(2.0 * log_lx, 2.0 * log_ly)
        log_ratio = jnp.log(2.0) + log_lx + log_ly - log_sum
        dims = self.base_kernel.slice_input(x).shape[-1]
        prefactor = jnp.exp(0.5 * dims * log_ratio)
        # Scaling both inputs by the same factor scales their distance, so the
        # base kernel evaluates ρ at the Gibbs distance with its own variance.
        scale = jnp.exp(0.5 * (jnp.log(2.0) - log_sum))
        return (prefactor * self.base_kernel(scale * x, scale * y)).squeeze()


__all__ = ["Gibbs"]
