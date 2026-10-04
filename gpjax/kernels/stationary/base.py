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
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
import numpyro.distributions as npd
from paramax import AbstractUnwrappable

from gpjax.kernels.base import AbstractKernel, val
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.parameters import (
    NonNegativeReal,
    PositiveReal,
)
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class StationaryKernel(AbstractKernel):
    """Base class for stationary kernels.

    Stationary kernels are a class of kernels that are invariant to translations
    in the input space. They can be isotropic or anisotropic, meaning that they
    can have a single lengthscale for all input dimensions or a different lengthscale
    for each input dimension.
    """

    # True when the kernel is a function of the lengthscale-scaled Euclidean
    # distance only and is positive definite in every dimension. Gibbs accepts
    # only such base kernels.
    isotropic_radial: ClassVar[bool] = False
    lengthscale: AbstractUnwrappable = eqx.field(
        default_factory=lambda: PositiveReal(1.0)
    )
    variance: AbstractUnwrappable = eqx.field(
        default_factory=lambda: NonNegativeReal(1.0)
    )

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[LengthscaleCompatible, AbstractUnwrappable] = 1.0,
        variance: tp.Union[ScalarFloat, AbstractUnwrappable] = 1.0,
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
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """
        super().__init__(
            active_dims=active_dims, n_dims=n_dims, compute_engine=compute_engine
        )
        # active_dims/n_dims are validated by AbstractKernel.__init__ above; a
        # vector lengthscale may further pin down n_dims (e.g. ARD with
        # active_dims=slice(None)), so re-derive it from the lengthscale here.
        self.n_dims = _validate_lengthscale(lengthscale, self.n_dims)

        if isinstance(lengthscale, AbstractUnwrappable):
            self.lengthscale = lengthscale
        else:
            self.lengthscale = PositiveReal(lengthscale)

        if isinstance(variance, AbstractUnwrappable):
            self.variance = variance
        else:
            self.variance = NonNegativeReal(variance)

    def _spectral_scale_tril(self) -> Float[Array, "D D"]:
        r"""The scale matrix $\mathrm{diag}(1/\ell)$ of the spectral measure.

        The spectral measure of a stationary kernel scales inversely with the
        lengthscale: short lengthscales give wide spectra. Broadcasts a scalar
        (isotropic) lengthscale across all $D$ dimensions and uses an ARD
        lengthscale vector elementwise.
        """
        if self.n_dims is None:
            raise ValueError(
                f"Expected the number of dimensions to be specified for {self.name} "
                "in order to construct its spectral measure. Please specify the "
                "n_dims argument for the kernel."
            )
        return jnp.diag(jnp.ones(self.n_dims) / val(self.lengthscale))

    @property
    def spectral_density(self) -> npd.MultivariateNormal | npd.MultivariateStudentT:
        r"""The normalised spectral measure $p(\boldsymbol{\omega})$ of the kernel.

        By Bochner's theorem, a stationary kernel is the Fourier transform of a
        finite measure. This property returns that measure *normalised to a
        probability distribution over* $\mathbb{R}^D$, so that

        $$
        k(\boldsymbol{\tau}) = \sigma^2 \,
            \mathbb{E}_{p(\boldsymbol{\omega})}
            \big[e^{i \boldsymbol{\omega}^\top \boldsymbol{\tau}}\big].
        $$

        The measure depends on the lengthscale $\ell$ (as an inverse scale) but
        **not** on the variance $\sigma^2$: the variance is the measure's total
        mass, which normalisation divides out, and it re-enters as the explicit
        prefactor above. The unnormalised spectral density of Rasmussen &
        Williams (2006, §4.2.1) is recovered as
        $S(\boldsymbol{\omega}) = \sigma^2 (2\pi)^D p(\boldsymbol{\omega})$
        under the convention
        $k(\boldsymbol{\tau}) = (2\pi)^{-D}\int S(\boldsymbol{\omega})
        e^{i\boldsymbol{\omega}^\top\boldsymbol{\tau}}\,d\boldsymbol{\omega}$.

        Returns:
            The spectral measure as a $D$-dimensional numpyro distribution.
        """
        raise NotImplementedError(
            f"Kernel {self.name} does not have a spectral density."
        )


def _validate_lengthscale(
    lengthscale: tp.Union[LengthscaleCompatible, AbstractUnwrappable],
    n_dims: tp.Union[int, None],
):
    # Check that the lengthscale is a valid value.
    _check_lengthscale(lengthscale)

    n_dims = _check_lengthscale_dims_compat(lengthscale, n_dims)
    return n_dims


def _check_lengthscale_dims_compat(
    lengthscale: tp.Union[LengthscaleCompatible, AbstractUnwrappable],
    n_dims: tp.Union[int, None],
):
    r"""Check that the lengthscale is compatible with n_dims.

    If possible, infer the number of input dimensions from the lengthscale.
    """

    if isinstance(lengthscale, AbstractUnwrappable):
        return _check_lengthscale_dims_compat(val(lengthscale), n_dims)

    lengthscale = jnp.asarray(lengthscale)
    ls_shape = jnp.shape(lengthscale)

    if ls_shape == ():
        return n_dims
    elif ls_shape != () and n_dims is None:
        return ls_shape[0]
    elif ls_shape != () and n_dims is not None:
        if ls_shape != (n_dims,):
            raise ValueError(
                "Expected `lengthscale` to be compatible with the number "
                f"of input dimensions. Got `lengthscale` with shape {ls_shape}, "
                f"but the number of input dimensions is {n_dims}."
            )
        return n_dims


def _check_lengthscale(lengthscale: tp.Any):
    """Check that the lengthscale is a valid value."""

    if isinstance(lengthscale, AbstractUnwrappable):
        _check_lengthscale(val(lengthscale))
        return

    if not isinstance(lengthscale, (int, float, jnp.ndarray, list, tuple)):
        raise TypeError(
            f"Expected `lengthscale` to be a array-like. Got {lengthscale}."
        )

    if isinstance(lengthscale, (jnp.ndarray, list)):
        ls_shape = jnp.shape(jnp.asarray(lengthscale))

        if len(ls_shape) > 1:
            raise ValueError(
                f"Expected `lengthscale` to be a scalar or 1D array. "
                f"Got `lengthscale` with shape {ls_shape}."
            )


__all__ = [
    "StationaryKernel",
]
