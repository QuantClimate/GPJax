"""Kernels specific to state-space GPs (e.g. TruncatedPeriodic) and the to_sde dispatcher."""

from __future__ import annotations

from functools import singledispatch

import equinox as eqx
import jax.numpy as jnp
import paramax

from gpjax.kernels.base import ProductKernel, SumKernel
from gpjax.kernels.computations.dense import DenseKernelComputation
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.stationary.matern12 import Matern12
from gpjax.kernels.stationary.matern32 import Matern32
from gpjax.kernels.stationary.matern52 import Matern52
from gpjax.kernels.stationary.periodic import Periodic
from gpjax.parameters import PositiveReal, val
from gpjax.state_space._bessel import _stable_scaled_ive
from gpjax.state_space._validation import _validate_temporal_kernel
from gpjax.state_space.sde import (
    LinearSDE,
    Matern12SDE,
    Matern32SDE,
    Matern52SDE,
    ProductSDE,
    SumSDE,
    TruncatedPeriodicSDE,
)

_STATIONARY_MATERN_KERNELS = (Matern12, Matern32, Matern52)


class TruncatedPeriodic(StationaryKernel):
    """Truncated-Fourier approximation of the periodic kernel.

    See Solin & Särkkä (2014). The kernel is the truncated cosine series

        k(τ) = σ² · (Ĩ_0(c) + 2 Σ_{k=1}^{K} Ĩ_k(c) cos(2π k τ / period))

    where Ĩ_k(c) = e^{-c} I_k(c) is the scaled modified Bessel function and
    c = 1/(4ℓ²). Acceptance of a 1-D temporal input is enforced by
    ``_validate_temporal_kernel`` at ``to_sde`` time, not here.

    Example:
        >>> from gpjax.state_space import TruncatedPeriodic
        >>> kernel = TruncatedPeriodic(
        ...     lengthscale=1.0, variance=1.0, period=1.0, truncation_order=6,
        ... )
        >>> kernel.truncation_order
        6
    """

    period: paramax.AbstractUnwrappable
    truncation_order: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        active_dims=None,
        lengthscale=1.0,
        variance=1.0,
        period=1.0,
        truncation_order=6,
        n_dims=None,
        compute_engine=None,
    ):
        if isinstance(period, paramax.AbstractUnwrappable):
            self.period = period
        else:
            self.period = PositiveReal(period)
        self.truncation_order = truncation_order
        if compute_engine is None:
            compute_engine = DenseKernelComputation()
        super().__init__(
            active_dims=active_dims,
            lengthscale=lengthscale,
            variance=variance,
            n_dims=n_dims,
            compute_engine=compute_engine,
        )

    def __call__(self, x, y):
        tau = self.slice_input(x) - self.slice_input(y)
        tau_scalar = jnp.squeeze(tau)
        lengthscale = val(self.lengthscale)
        variance = val(self.variance)
        period = val(self.period)
        truncation_order = self.truncation_order

        bessel_argument = 1.0 / (4.0 * lengthscale**2)
        scaled_bessels = _stable_scaled_ive(bessel_argument, truncation_order)
        harmonic_indices = jnp.arange(1, truncation_order + 1)
        cos_terms = jnp.cos(2.0 * jnp.pi * harmonic_indices * tau_scalar / period)
        return variance * (
            scaled_bessels[0] + 2.0 * jnp.sum(scaled_bessels[1:] * cos_terms)
        )


@singledispatch
def to_sde(kernel) -> LinearSDE:
    """Convert a kernel to its state-space (linear SDE) representation.

    Default handler raises ``NotImplementedError`` listing supported kernels.

    Example:
        >>> import gpjax as gpx
        >>> from gpjax.state_space import to_sde
        >>> sde = to_sde(gpx.kernels.Matern32(lengthscale=1.0, variance=1.0))
        >>> type(sde).__name__
        'Matern32SDE'
    """
    raise NotImplementedError(
        f"State-space conversion not implemented for {type(kernel).__name__}. "
        "Supported kernels: Matern12, Matern32, Matern52, TruncatedPeriodic, "
        "and SumKernel of the above."
    )


@to_sde.register
def _to_sde_matern12(kernel: Matern12) -> Matern12SDE:
    _validate_temporal_kernel(kernel)
    return Matern12SDE(
        lengthscale=val(kernel.lengthscale),
        variance=val(kernel.variance),
    )


@to_sde.register
def _to_sde_matern32(kernel: Matern32) -> Matern32SDE:
    _validate_temporal_kernel(kernel)
    return Matern32SDE(
        lengthscale=val(kernel.lengthscale),
        variance=val(kernel.variance),
    )


@to_sde.register
def _to_sde_matern52(kernel: Matern52) -> Matern52SDE:
    _validate_temporal_kernel(kernel)
    return Matern52SDE(
        lengthscale=val(kernel.lengthscale),
        variance=val(kernel.variance),
    )


@to_sde.register
def _to_sde_truncated_periodic(kernel: TruncatedPeriodic) -> TruncatedPeriodicSDE:
    _validate_temporal_kernel(kernel)
    return TruncatedPeriodicSDE(
        lengthscale=val(kernel.lengthscale),
        variance=val(kernel.variance),
        period=val(kernel.period),
        truncation_order=kernel.truncation_order,
    )


@to_sde.register
def _to_sde_sum_kernel(kernel: SumKernel) -> SumSDE:
    components = tuple(to_sde(component) for component in kernel.kernels)
    return SumSDE(components=components)


@to_sde.register
def _to_sde_periodic(kernel: Periodic) -> LinearSDE:
    raise NotImplementedError(
        "Periodic kernel does not have a finite-dimensional state-space representation. "
        "Use TruncatedPeriodic for the truncated-Fourier approximation."
    )


def _find_quasi_periodic_factors(
    kernels: tuple,
) -> tuple[TruncatedPeriodic, StationaryKernel] | None:
    """Return ``(periodic_factor, matern_factor)`` if ``kernels`` is exactly a
    ``TruncatedPeriodic`` and a stationary Matérn factor, else ``None``.
    """
    if len(kernels) != 2:
        return None
    periodic_factor, matern_factor = None, None
    for factor in kernels:
        if isinstance(factor, TruncatedPeriodic):
            periodic_factor = factor
        elif isinstance(factor, _STATIONARY_MATERN_KERNELS):
            matern_factor = factor
    if periodic_factor is None or matern_factor is None:
        return None
    return periodic_factor, matern_factor


@to_sde.register
def _to_sde_product_kernel(kernel: ProductKernel) -> LinearSDE:
    quasi_periodic_factors = _find_quasi_periodic_factors(kernel.kernels)
    if quasi_periodic_factors is not None:
        periodic_factor, matern_factor = quasi_periodic_factors
        return ProductSDE(components=(to_sde(periodic_factor), to_sde(matern_factor)))
    raise NotImplementedError(
        "ProductKernel state-space conversion is only supported for the "
        "quasi-periodic TruncatedPeriodic x {Matern12, Matern32, Matern52} case "
        "(Solin & Sarkka 2014 sec. 3), whose Kronecker state dimension is "
        "(2K+1)*d. Generic ProductKernel conversion is not supported in v1: state "
        "dimension blows up Kronecker-style as the product of component state "
        "dimensions, which defeats the O(N · d³) advantage of the state-space "
        "approach."
    )
