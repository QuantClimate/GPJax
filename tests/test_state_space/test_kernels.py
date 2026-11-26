"""Tests for state-space kernels and the to_sde dispatcher."""

import gpjax as gpx
from gpjax.state_space.kernels import TruncatedPeriodic, to_sde
from gpjax.state_space.sde import (
    Matern12SDE,
    Matern32SDE,
    Matern52SDE,
    ProductSDE,
    SumSDE,
    TruncatedPeriodicSDE,
)
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special


def test_truncated_periodic_call_matches_bessel_formula():
    lengthscale, variance, period, K = 0.4, 1.0, 1.0, 6
    kernel = TruncatedPeriodic(
        lengthscale=lengthscale,
        variance=variance,
        period=period,
        truncation_order=K,
    )
    taus = jnp.array([0.0, 0.1, 0.25, 0.5, 0.9])
    bessel_argument = 1.0 / (4.0 * lengthscale**2)
    expected = (
        variance
        * np.exp(-bessel_argument)
        * (
            scipy.special.iv(0, bessel_argument)
            + 2.0
            * sum(
                scipy.special.iv(k, bessel_argument)
                * np.cos(2 * np.pi * k * np.asarray(taus) / period)
                for k in range(1, K + 1)
            )
        )
    )
    actual = np.array([kernel(jnp.array([float(t)]), jnp.array([0.0])) for t in taus])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_truncated_periodic_accepts_parent_config_space():
    """The kernel itself does no 1-D-temporal enforcement; that is `to_sde`'s job."""
    kernel = TruncatedPeriodic(
        lengthscale=jnp.ones(3),
        variance=1.0,
        period=1.0,
        truncation_order=4,
        n_dims=3,
    )
    assert kernel.truncation_order == 4


def test_to_sde_matern12_returns_matern12sde():
    kernel = gpx.kernels.Matern12(lengthscale=2.0, variance=3.0)
    sde = to_sde(kernel)
    assert isinstance(sde, Matern12SDE)
    np.testing.assert_allclose(np.asarray(sde.lengthscale), 2.0)
    np.testing.assert_allclose(np.asarray(sde.variance), 3.0)


def test_to_sde_default_handler_raises_for_unsupported_kernel():
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
    with pytest.raises(
        NotImplementedError, match=r"Matern12|Matern32|Matern52|TruncatedPeriodic"
    ):
        to_sde(kernel)


def test_to_sde_matern32_returns_matern32sde():
    kernel = gpx.kernels.Matern32(lengthscale=1.5, variance=2.0)
    sde = to_sde(kernel)
    assert isinstance(sde, Matern32SDE)
    np.testing.assert_allclose(np.asarray(sde.lengthscale), 1.5)
    np.testing.assert_allclose(np.asarray(sde.variance), 2.0)


def test_to_sde_matern52_returns_matern52sde():
    kernel = gpx.kernels.Matern52(lengthscale=0.7, variance=1.2)
    sde = to_sde(kernel)
    assert isinstance(sde, Matern52SDE)
    np.testing.assert_allclose(np.asarray(sde.lengthscale), 0.7)
    np.testing.assert_allclose(np.asarray(sde.variance), 1.2)


def test_to_sde_truncated_periodic_returns_truncated_periodic_sde():
    kernel = TruncatedPeriodic(
        lengthscale=0.5, variance=2.5, period=3.0, truncation_order=4
    )
    sde = to_sde(kernel)
    assert isinstance(sde, TruncatedPeriodicSDE)
    assert sde.state_dim == 1 + 2 * 4
    np.testing.assert_allclose(np.asarray(sde.lengthscale), 0.5)
    np.testing.assert_allclose(np.asarray(sde.variance), 2.5)
    np.testing.assert_allclose(np.asarray(sde.period), 3.0)


def test_to_sde_sum_kernel_returns_sumsde():
    kernel = gpx.kernels.Matern32(lengthscale=1.0, variance=1.0) + TruncatedPeriodic(
        lengthscale=0.5,
        variance=1.0,
        period=1.0,
        truncation_order=4,
    )
    sde = to_sde(kernel)
    assert isinstance(sde, SumSDE)
    assert len(sde.components) == 2
    assert isinstance(sde.components[0], Matern32SDE)
    assert isinstance(sde.components[1], TruncatedPeriodicSDE)
    expected_state_dim = 2 + (1 + 2 * 4)
    assert sde.state_dim == expected_state_dim


@pytest.mark.parametrize(
    ("kernel_factory", "error_match"),
    [
        (
            lambda: gpx.kernels.Periodic(lengthscale=1.0, variance=1.0),
            "TruncatedPeriodic",
        ),
        (
            lambda: gpx.kernels.RBF(lengthscale=1.0, variance=1.0),
            r"Matern12|Matern32|Matern52|TruncatedPeriodic",
        ),
        (
            lambda: gpx.kernels.Linear(),
            r"Matern12|Matern32|Matern52|TruncatedPeriodic|state-space conversion not implemented",
        ),
    ],
)
def test_to_sde_rejects_unsupported_kernels(kernel_factory, error_match):
    kernel = kernel_factory()
    with pytest.raises(NotImplementedError, match=error_match):
        to_sde(kernel)


def test_to_sde_rejects_product_kernel_with_kronecker_blowup_message():
    kernel = gpx.kernels.Matern32(lengthscale=1.0, variance=1.0) * gpx.kernels.Matern32(
        lengthscale=0.5, variance=1.0
    )
    with pytest.raises(NotImplementedError, match=r"Kronecker|product|state space"):
        to_sde(kernel)


@pytest.mark.parametrize(
    ("kernel_factory", "error_match"),
    [
        (
            # Three-factor product: genuinely unsupported even though a
            # TruncatedPeriodic x Matern32 pair is embedded within it.
            lambda: (
                gpx.kernels.Matern32(lengthscale=1.0, variance=1.0)
                * TruncatedPeriodic(lengthscale=0.5, variance=1.0, period=1.0)
                * gpx.kernels.Matern12(lengthscale=1.0, variance=1.0)
            ),
            r"Kronecker|product|state space",
        ),
        (
            # Two periodics, no Matern factor.
            lambda: (
                TruncatedPeriodic(lengthscale=0.5, variance=1.0, period=1.0)
                * TruncatedPeriodic(lengthscale=0.3, variance=1.0, period=2.0)
            ),
            r"Kronecker|product|state space",
        ),
        (
            # Non-truncated Periodic x Matern: the periodic factor itself has
            # no finite-dimensional state-space representation.
            lambda: (
                gpx.kernels.Periodic(lengthscale=1.0, variance=1.0)
                * gpx.kernels.Matern32(lengthscale=1.0, variance=1.0)
            ),
            r"Kronecker|product|state space",
        ),
    ],
)
def test_to_sde_rejects_genuinely_unsupported_product_kernels(
    kernel_factory, error_match
):
    kernel = kernel_factory()
    with pytest.raises(NotImplementedError, match=error_match):
        to_sde(kernel)


@pytest.mark.parametrize(
    ("matern_cls", "matern_sde_cls", "matern_state_dim"),
    [
        (gpx.kernels.Matern12, Matern12SDE, 1),
        (gpx.kernels.Matern32, Matern32SDE, 2),
        (gpx.kernels.Matern52, Matern52SDE, 3),
    ],
)
@pytest.mark.parametrize("truncation_order", [1, 4])
def test_to_sde_periodic_times_matern_returns_product_sde_with_expected_state_dim(
    matern_cls, matern_sde_cls, matern_state_dim, truncation_order
):
    periodic_kernel = TruncatedPeriodic(
        lengthscale=0.5, variance=1.3, period=1.5, truncation_order=truncation_order
    )
    matern_kernel = matern_cls(lengthscale=2.0, variance=0.7)

    for kernel in (periodic_kernel * matern_kernel, matern_kernel * periodic_kernel):
        sde = to_sde(kernel)
        assert isinstance(sde, ProductSDE)
        periodic_state_dim = 1 + 2 * truncation_order
        assert sde.state_dim == periodic_state_dim * matern_state_dim
        factor_types = {type(component) for component in sde.components}
        assert factor_types == {TruncatedPeriodicSDE, matern_sde_cls}


def test_to_sde_periodic_times_matern_kernel_value_matches_dense_product():
    lengthscale_periodic, variance_periodic, period, K = 0.5, 1.3, 1.5, 5
    lengthscale_matern, variance_matern = 2.0, 0.7

    periodic_kernel = TruncatedPeriodic(
        lengthscale=lengthscale_periodic,
        variance=variance_periodic,
        period=period,
        truncation_order=K,
    )
    matern_kernel = gpx.kernels.Matern32(
        lengthscale=lengthscale_matern, variance=variance_matern
    )
    product_kernel = periodic_kernel * matern_kernel
    sde = to_sde(product_kernel)

    for tau in (0.0, 0.1, 0.37, 1.0, 2.5):
        H = np.asarray(sde.observation_matrix)
        A, _ = sde.discretise(jnp.asarray(tau))
        P_inf = np.asarray(
            sde.stationary_state_cov_sqrt @ sde.stationary_state_cov_sqrt.T
        )
        # k(τ) = H A(τ) P_∞ Hᵀ for a stationary LTI state-space GP.
        sde_kernel_value = (H @ np.asarray(A) @ P_inf @ H.T)[0, 0]

        dense_kernel_value = product_kernel(jnp.array([tau]), jnp.array([0.0]))
        expected = periodic_kernel(jnp.array([tau]), jnp.array([0.0])) * matern_kernel(
            jnp.array([tau]), jnp.array([0.0])
        )
        np.testing.assert_allclose(dense_kernel_value, expected, atol=1e-10)
        np.testing.assert_allclose(sde_kernel_value, expected, atol=1e-8)
