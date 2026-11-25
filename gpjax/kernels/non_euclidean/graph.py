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
from jaxtyping import (
    Float,
    Integer,
    Num,
)
import paramax
from paramax import AbstractUnwrappable

from gpjax.kernels.base import val
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    EigenKernelComputation,
)
from gpjax.kernels.non_euclidean.utils import (
    calculate_heat_semigroup,
    jax_gather_nd,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.parameters import PositiveReal
from gpjax.typing import (
    Array,
    ScalarFloat,
    ScalarInt,
)


class GraphKernel(StationaryKernel):
    r"""The Matérn graph kernel defined on the vertex set of a graph.

    A Matérn graph kernel defined through the graph Laplacian spectrum.

    The kernel evaluates a Matérn spectral filter on each Laplacian eigenvalue
    $\lambda$:
    $$
    \Phi(\lambda) = \left(\frac{2\nu}{\ell^2} + \lambda\right)^{-\nu},
    $$
    where $\ell$ is the lengthscale parameter and $\nu$ is the smoothness
    parameter. The resulting spectral weights are normalised and scaled by the
    variance parameter.

    The key reference for this object is Borovitskiy et al. (2021).

    .. seealso::

        :doc:`/examples/graph_kernels` fits one to a signal on a barbell graph.
    """

    smoothness: tp.Any
    num_vertex: ScalarInt | None
    laplacian: Float[Array, "N N"]
    eigenvalues: Float[Array, "N 1"]
    eigenvectors: Float[Array, "N N"]
    name: str = "Graph Matérn"

    def __init__(
        self,
        laplacian: Num[Array, "N N"],
        active_dims: list[int] | slice | None = None,
        lengthscale: ScalarFloat | Float[Array, " D"] | AbstractUnwrappable = 1.0,
        variance: ScalarFloat | AbstractUnwrappable = 1.0,
        smoothness: ScalarFloat = 1.0,
        n_dims: int | None = None,
        compute_engine: AbstractKernelComputation = EigenKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            laplacian: the Laplacian matrix of the graph.
            active_dims: The indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            smoothness: the smoothness parameter of the Matérn kernel.
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """
        if isinstance(smoothness, AbstractUnwrappable):
            self.smoothness = smoothness
        else:
            self.smoothness = PositiveReal(smoothness)

        laplacian = jnp.asarray(laplacian, dtype=jnp.float64)
        evals, evecs = jnp.linalg.eigh(laplacian)
        self.laplacian = paramax.non_trainable(laplacian)
        self.eigenvectors = paramax.non_trainable(evecs)
        self.eigenvalues = paramax.non_trainable(evals.reshape(-1, 1))
        self.num_vertex = evals.shape[0]

        super().__init__(active_dims, lengthscale, variance, n_dims, compute_engine)

    def __call__(
        self,
        x: ScalarInt | Integer[Array, " N"] | Integer[Array, "N 1"],
        y: ScalarInt | Integer[Array, " M"] | Integer[Array, "M 1"],
    ):
        x_idx = self._prepare_indices(x)
        y_idx = self._prepare_indices(y)
        S = calculate_heat_semigroup(self)
        eigenvectors = val(self.eigenvectors)
        Kxx = (jax_gather_nd(eigenvectors, x_idx) * S.squeeze()) @ jnp.transpose(
            jax_gather_nd(eigenvectors, y_idx)
        )  # shape (n,n)
        return Kxx.squeeze()

    def _prepare_indices(
        self,
        indices: ScalarInt | Integer[Array, " N"] | Integer[Array, "N 1"],
    ) -> Integer[Array, "N 1"]:
        """Ensure index arrays are integer column vectors regardless of caller shape."""

        idx = jnp.asarray(indices, dtype=jnp.int32)
        idx = jnp.atleast_1d(idx)
        return idx.reshape(-1, 1)
