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

from jaxtyping import (
    Float,
    Num,
)

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.typing import Array

if tp.TYPE_CHECKING:
    from gpjax.kernels.non_euclidean.graph import GraphKernel

Kernel = tp.TypeVar("Kernel", bound="GraphKernel")


class EigenKernelComputation(AbstractKernelComputation):
    r"""Eigen kernel computation class. Kernels who operate on an
    eigen-decomposed structure should use this computation object.
    """

    def _cross_covariance(
        self, kernel: Kernel, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "N M"]:
        return kernel(x, y)
