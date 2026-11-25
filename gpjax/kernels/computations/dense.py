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

from jax import vmap
from jaxtyping import Float

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.typing import Array

if tp.TYPE_CHECKING:
    from gpjax.kernels.base import AbstractKernel

K = tp.TypeVar("K", bound="AbstractKernel")


class DenseKernelComputation(AbstractKernelComputation):
    r"""Dense kernel computation class. Operations with the kernel assume
    a dense gram matrix structure.
    """

    def _cross_covariance(
        self, kernel: K, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        cross_cov = vmap(lambda x: vmap(lambda y: kernel(x, y))(y))(x)
        return cross_cov
