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

"""JaxKern."""

from gpjax.kernels import (
    location_functions,
    stationary,
)
from gpjax.kernels.additive import (
    OrthogonalAdditiveKernel,
)
from gpjax.kernels.approximations import RFF
from gpjax.kernels.base import (
    AbstractKernel,
    Constant,
    ProductKernel,
    SumKernel,
)
from gpjax.kernels.computations import (
    BasisFunctionComputation,
    ConstantDiagonalKernelComputation,
    DenseKernelComputation,
    DiagonalKernelComputation,
    EigenKernelComputation,
)
from gpjax.kernels.multioutput import (
    ICMKernel,
    LCMKernel,
    MultiOutputKernel,
    MultiOutputKernelComputation,
)
from gpjax.kernels.non_euclidean import GraphKernel
from gpjax.kernels.nonstationary import (
    ArcCosine,
    Gibbs,
    Linear,
    Polynomial,
    VaryingAmplitude,
)
from gpjax.kernels.stationary import (
    RBF,
    Gneiting,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    PoweredExponential,
    RationalQuadratic,
    White,
)

__all__ = [
    "RBF",
    "RFF",
    "AbstractKernel",
    "ArcCosine",
    "BasisFunctionComputation",
    "Constant",
    "ConstantDiagonalKernelComputation",
    "DenseKernelComputation",
    "DiagonalKernelComputation",
    "EigenKernelComputation",
    "Gibbs",
    "Gneiting",
    "GraphKernel",
    "ICMKernel",
    "LCMKernel",
    "Linear",
    "Matern12",
    "Matern32",
    "Matern52",
    "MultiOutputKernel",
    "MultiOutputKernelComputation",
    "OrthogonalAdditiveKernel",
    "Periodic",
    "Polynomial",
    "PoweredExponential",
    "ProductKernel",
    "RationalQuadratic",
    "SumKernel",
    "VaryingAmplitude",
    "White",
    "location_functions",
    "stationary",
]
