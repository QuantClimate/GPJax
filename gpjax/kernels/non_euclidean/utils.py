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

from __future__ import annotations

import typing as tp

import jax.numpy as jnp
from jaxtyping import (
    Float,
    Int,
)

from gpjax.kernels.base import val
from gpjax.typing import Array

if tp.TYPE_CHECKING:
    from gpjax.kernels.non_euclidean.graph import GraphKernel


def jax_gather_nd(
    params: Float[Array, " N *rest"], indices: Int[Array, " M 1"]
) -> Float[Array, " M *rest"]:
    r"""Slice a `params` array at a set of `indices`.

    This is a reimplementation of TensorFlow's `gather_nd` function:
    [link](https://www.tensorflow.org/api_docs/python/tf/gather_nd)

    Args:
        params: an arbitrary array with leading axes of length $N$ upon
            which we shall slice.
        indices: an integer array of length $M$ with values in the range
            $[0, N)$ whose value at index $i$ will be used to slice `params` at
            index $i$.

    Returns:
        An arbitrary array with leading axes of length $M$.
    """
    tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    return params[tuple_indices]


def calculate_heat_semigroup(kernel: GraphKernel) -> Float[Array, "N M"]:
    r"""Returns the rescaled heat semigroup, S

    Args:
        kernel: instance of the graph kernel

    Returns:
        S
    """
    smoothness = val(kernel.smoothness)
    lengthscale = val(kernel.lengthscale)
    variance = val(kernel.variance)
    eigenvalues = val(kernel.eigenvalues)
    S = jnp.power(
        eigenvalues + 2 * smoothness / lengthscale / lengthscale,
        -smoothness,
    )
    S = jnp.multiply(S, kernel.num_vertex / jnp.sum(S))
    # Scale the transform eigenvalues by the kernel variance
    S = jnp.multiply(S, variance)
    return S
