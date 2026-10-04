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

from gpjax.kernels.approximations import RFF
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


class VaryingAmplitude(AbstractKernel):
    r"""A base kernel whose standard deviation changes with location.

    Computes the covariance for a pair of inputs $(x, y)$ from a base kernel
    $k_0$ and a location function $g$ that gives $\log\sigma(x)$:
    $$
    k(x, y) = \sigma(x)\,\sigma(y)\,k_0(x, y), \qquad \sigma(x) = \exp g(x).
    $$

    The kernel is positive definite for every base kernel that is positive
    definite. The marginal variance at $x$ is $\sigma(x)^2 k_0(x, x)$, so the
    base kernel's variance holds the overall scale and $\sigma(x)$ holds the
    change over location. Use it when variability is larger in some regions
    than in others, for example over land than over sea.

    The base kernel selects the columns over which it measures distance with
    its own `active_dims`, and the location function selects its covariate
    columns. The wrapper itself always receives every column.
    """

    name: ClassVar[str] = "Varying amplitude"
    base_kernel: AbstractKernel
    amplitude: AbstractLocationFunction

    def __init__(
        self,
        base_kernel: AbstractKernel,
        amplitude: AbstractLocationFunction,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        r"""Initialise the kernel.

        Args:
            base_kernel: the kernel $k_0$ whose amplitude changes. It must
                evaluate one pair of points at a time.
            amplitude: the location function that gives $\log\sigma(x)$.
            compute_engine: the computation engine that the kernel uses to
                compute its covariance matrices.
        """
        _check_pointwise(base_kernel, type(self).__name__)
        self.base_kernel = base_kernel
        self.amplitude = amplitude
        super().__init__(compute_engine=compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        scale = jnp.exp(self.amplitude(x) + self.amplitude(y))
        return (scale * self.base_kernel(x, y)).squeeze()


def _check_pointwise(base_kernel: AbstractKernel, wrapper: str) -> None:
    # RFF returns None from __call__: it builds its matrices from features.
    if isinstance(base_kernel, RFF):
        raise TypeError(
            f"{wrapper} needs a base kernel that evaluates one pair of points at "
            "a time, but RFF builds its matrices from random features. Use the "
            "kernel that RFF approximates as the base kernel."
        )


__all__ = ["VaryingAmplitude"]
