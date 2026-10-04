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

import jax.numpy as jnp
from jaxtyping import Float
import numpyro.distributions as npd

from gpjax.kernels.base import val
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.stationary.utils import (
    build_student_t_distribution,
    euclidean_distance,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class Matern12(StationaryKernel):
    r"""The Matérn kernel with smoothness parameter fixed at 0.5.

    Computes the covariance on a pair of inputs $(x, y)$ with
    lengthscale parameter $\ell$ and variance $\sigma^2$.

    $$
    k(x, y) = \sigma^2\exp\Bigg(-\frac{\lvert x-y \rvert}{\ell}\Bigg)
    $$
    """

    name: ClassVar[str] = "Matérn12"
    isotropic_radial: ClassVar[bool] = True

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / val(self.lengthscale)
        y = self.slice_input(y) / val(self.lengthscale)
        K = val(self.variance) * jnp.exp(-euclidean_distance(x, y))
        return K.squeeze()

    @property
    def spectral_density(self) -> npd.MultivariateStudentT:
        r"""The spectral measure of the Matérn-1/2 kernel: a multivariate
        Student's t with 1 degree of freedom and scale $\mathrm{diag}(\ell)^{-1}$."""
        return build_student_t_distribution(
            nu=1, scale_tril=self._spectral_scale_tril()
        )
