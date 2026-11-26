"""Tests for LinearSDE and its per-kernel discretisations.

See plans/2026-04-21-state-space-gps-design.md §Core types.
"""

import equinox as eqx
from gpjax.state_space.sde import (
    LinearSDE,
    Matern12SDE,
    Matern32SDE,
    Matern52SDE,
    ProductSDE,
    SumSDE,
    TruncatedPeriodicSDE,
)
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import pytest
import scipy.linalg
import scipy.special


def _trivial_matern12_sde(lengthscale=1.0, variance=1.0):
    F = jnp.array([[-1.0 / lengthscale]])
    L = jnp.array([[1.0]])
    Qc = jnp.array([[2.0 * variance / lengthscale]])
    H = jnp.array([[1.0]])
    L_inf = jnp.array([[jnp.sqrt(variance)]])
    return LinearSDE(
        drift_matrix=F,
        diffusion_matrix=L,
        process_noise_spectral_density=Qc,
        observation_matrix=H,
        stationary_state_cov_sqrt=L_inf,
        state_dim=1,
    )


@pytest.mark.parametrize("nu_name", ["Matern12SDE", "Matern32SDE", "Matern52SDE"])
def test_discrete_transition_equals_matrix_exponential(nu_name):
    import gpjax.state_space.sde as sde_module

    sde = getattr(sde_module, nu_name)(lengthscale=0.7, variance=1.3)
    time_step = jnp.asarray(0.37)
    transition_matrix, _ = sde.discretise(time_step)
    expected = jsl.expm(sde.drift_matrix * time_step)
    assert jnp.allclose(transition_matrix, expected, atol=1e-8)


def test_linear_sde_discretise_zero_dt_is_identity_transition():
    sde = _trivial_matern12_sde()
    A, L_Q = sde.discretise(jnp.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(A), np.eye(1))
    np.testing.assert_array_equal(np.asarray(L_Q), np.zeros((1, 1)))


@pytest.mark.parametrize("dt", [1e-6, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("lengthscale", [0.3, 1.0, 5.0])
def test_matern12_A_matches_scipy_expm(dt, lengthscale):
    sde = Matern12SDE(lengthscale=lengthscale, variance=1.5)
    A, _ = sde.discretise(jnp.asarray(dt))
    expected = scipy.linalg.expm(np.asarray(sde.drift_matrix) * dt)
    np.testing.assert_allclose(np.asarray(A), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dt", [1e-6, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("lengthscale", [0.3, 1.0, 5.0])
def test_matern12_Q_equals_stationary_identity(dt, lengthscale):
    variance = 1.5
    sde = Matern12SDE(lengthscale=lengthscale, variance=variance)
    A, L_Q = sde.discretise(jnp.asarray(dt))
    Q = np.asarray(L_Q @ L_Q.T)
    P_inf = np.array([[variance]])
    expected = P_inf - np.asarray(A) @ P_inf @ np.asarray(A).T
    np.testing.assert_allclose(Q, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dt", [1e-6, 0.01, 0.5, 5.0])
@pytest.mark.parametrize("lengthscale", [0.2, 1.0, 3.0])
def test_matern32_A_matches_scipy_expm(dt, lengthscale):
    sde = Matern32SDE(lengthscale=lengthscale, variance=2.0)
    A, _ = sde.discretise(jnp.asarray(dt))
    expected = scipy.linalg.expm(np.asarray(sde.drift_matrix) * dt)
    np.testing.assert_allclose(np.asarray(A), expected, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("dt", [1e-3, 0.01, 0.5, 5.0])
@pytest.mark.parametrize("lengthscale", [0.2, 1.0, 3.0])
def test_matern32_Q_equals_stationary_identity(dt, lengthscale):
    variance = 2.0
    sde = Matern32SDE(lengthscale=lengthscale, variance=variance)
    A, L_Q = sde.discretise(jnp.asarray(dt))
    Q = np.asarray(L_Q @ L_Q.T)
    P_inf = np.diag([variance, 3.0 * variance / lengthscale**2])
    expected = P_inf - np.asarray(A) @ P_inf @ np.asarray(A).T
    np.testing.assert_allclose(Q, expected, atol=1e-9, rtol=1e-8)


def test_matern32_lyapunov_identity():
    sde = Matern32SDE(lengthscale=1.5, variance=1.2)
    F = np.asarray(sde.drift_matrix)
    L = np.asarray(sde.diffusion_matrix)
    Qc = np.asarray(sde.process_noise_spectral_density)
    L_inf = np.asarray(sde.stationary_state_cov_sqrt)
    P_inf = L_inf @ L_inf.T
    residual = F @ P_inf + P_inf @ F.T + L @ Qc @ L.T
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_matern32_zero_dt_returns_exact_identity_and_zero_noise():
    sde = Matern32SDE(lengthscale=1.0, variance=2.0)
    A, L_Q = sde.discretise(jnp.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(A), np.eye(2))
    np.testing.assert_array_equal(np.asarray(L_Q), np.zeros((2, 2)))


@pytest.mark.parametrize("dt", [1e-6, 1e-4])
def test_matern32_small_dt_process_noise_is_finite_and_near_psd(dt):
    """At small dt, `Pinf - A Pinf A.T` is dominated by floating-point error.
    Require: finite, symmetric, min eigenvalue tolerably small-negative."""
    sde = Matern32SDE(lengthscale=1.0, variance=2.0)
    _, L_Q = sde.discretise(jnp.asarray(dt))
    Q = np.asarray(L_Q @ L_Q.T)
    assert np.isfinite(Q).all()
    np.testing.assert_allclose(Q, Q.T, atol=1e-14)
    assert np.linalg.eigvalsh(Q).min() >= -1e-10


@pytest.mark.parametrize("dt", [1e-6, 0.01, 0.5, 5.0])
@pytest.mark.parametrize("lengthscale", [0.2, 1.0, 3.0])
def test_matern52_A_matches_scipy_expm(dt, lengthscale):
    sde = Matern52SDE(lengthscale=lengthscale, variance=2.0)
    A, _ = sde.discretise(jnp.asarray(dt))
    expected = scipy.linalg.expm(np.asarray(sde.drift_matrix) * dt)
    np.testing.assert_allclose(np.asarray(A), expected, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("dt", [1e-3, 0.01, 0.5, 5.0])
@pytest.mark.parametrize("lengthscale", [0.2, 1.0, 3.0])
def test_matern52_Q_equals_stationary_identity(dt, lengthscale):
    variance = 2.0
    sde = Matern52SDE(lengthscale=lengthscale, variance=variance)
    A, L_Q = sde.discretise(jnp.asarray(dt))
    Q = np.asarray(L_Q @ L_Q.T)
    kappa = 5.0 / 3.0 * variance / lengthscale**2
    P_inf = np.array(
        [
            [variance, 0.0, -kappa],
            [0.0, kappa, 0.0],
            [-kappa, 0.0, 25.0 * variance / lengthscale**4],
        ]
    )
    expected = P_inf - np.asarray(A) @ P_inf @ np.asarray(A).T
    np.testing.assert_allclose(Q, expected, atol=1e-9, rtol=1e-8)


def test_matern52_lyapunov_identity():
    sde = Matern52SDE(lengthscale=1.5, variance=1.2)
    F = np.asarray(sde.drift_matrix)
    L = np.asarray(sde.diffusion_matrix)
    Qc = np.asarray(sde.process_noise_spectral_density)
    L_inf = np.asarray(sde.stationary_state_cov_sqrt)
    P_inf = L_inf @ L_inf.T
    residual = F @ P_inf + P_inf @ F.T + L @ Qc @ L.T
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_matern52_zero_dt_returns_exact_identity_and_zero_noise():
    sde = Matern52SDE(lengthscale=1.0, variance=2.0)
    A, L_Q = sde.discretise(jnp.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(A), np.eye(3))
    np.testing.assert_array_equal(np.asarray(L_Q), np.zeros((3, 3)))


@pytest.mark.parametrize("dt", [1e-6, 1e-4])
def test_matern52_small_dt_process_noise_is_finite_and_near_psd(dt):
    sde = Matern52SDE(lengthscale=1.0, variance=2.0)
    _, L_Q = sde.discretise(jnp.asarray(dt))
    Q = np.asarray(L_Q @ L_Q.T)
    assert np.isfinite(Q).all()
    np.testing.assert_allclose(Q, Q.T, atol=1e-14)
    assert np.linalg.eigvalsh(Q).min() >= -1e-10


@pytest.mark.parametrize(
    ("sde_cls", "state_dim"),
    [(Matern12SDE, 1), (Matern32SDE, 2), (Matern52SDE, 3)],
)
def test_matern_sde_is_jittable_and_vmap_compatible(sde_cls, state_dim):
    sde = sde_cls(lengthscale=1.0, variance=1.0)
    discretise_jitted = eqx.filter_jit(sde.discretise)
    A, L_Q = discretise_jitted(jnp.asarray(0.5))
    assert A.shape == (state_dim, state_dim)
    assert L_Q.shape == (state_dim, state_dim)

    dts = jnp.array([0.1, 0.5, 1.0])
    A_batched, L_Q_batched = jax.vmap(sde.discretise)(dts)
    assert A_batched.shape == (3, state_dim, state_dim)
    assert L_Q_batched.shape == (3, state_dim, state_dim)


@pytest.mark.parametrize("sde_cls", [Matern12SDE, Matern32SDE, Matern52SDE])
def test_matern_sde_gradient_through_lengthscale(sde_cls):
    def loss(lengthscale):
        sde = sde_cls(lengthscale=lengthscale, variance=1.0)
        A, _ = sde.discretise(jnp.asarray(0.5))
        return jnp.sum(A)

    grad_value = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(grad_value)


@pytest.mark.parametrize("sde_cls", [Matern12SDE, Matern32SDE, Matern52SDE])
def test_matern_sde_gradient_through_variance(sde_cls):
    def loss(variance):
        sde = sde_cls(lengthscale=1.0, variance=variance)
        _, L_Q = sde.discretise(jnp.asarray(0.5))
        return jnp.sum(L_Q @ L_Q.T)

    grad_value = jax.grad(loss)(jnp.asarray(1.5))
    assert jnp.isfinite(grad_value)


@pytest.mark.parametrize("truncation_order", [1, 4, 8])
def test_truncated_periodic_state_dim_is_one_plus_2k(truncation_order):
    sde = TruncatedPeriodicSDE(
        lengthscale=1.0,
        variance=1.0,
        period=1.0,
        truncation_order=truncation_order,
    )
    assert sde.state_dim == 1 + 2 * truncation_order


def test_truncated_periodic_L_Q_is_zero():
    sde = TruncatedPeriodicSDE(
        lengthscale=1.0, variance=1.0, period=1.0, truncation_order=3
    )
    _, L_Q = sde.discretise(jnp.asarray(0.7))
    np.testing.assert_array_equal(np.asarray(L_Q), 0.0)


def test_truncated_periodic_A_is_block_diagonal_rotation():
    sde = TruncatedPeriodicSDE(
        lengthscale=1.0,
        variance=1.0,
        period=2.0,
        truncation_order=2,
    )
    dt = 0.3
    A, _ = sde.discretise(jnp.asarray(dt))
    A = np.asarray(A)
    # DC block: 1×1 identity
    np.testing.assert_allclose(A[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(A[0, 1:], 0.0, atol=1e-12)
    # Harmonic k=1 block: 2×2 rotation with ω_1 = 2π/2 = π
    omega_1 = np.pi
    expected_k1 = np.array(
        [
            [np.cos(omega_1 * dt), np.sin(omega_1 * dt)],
            [-np.sin(omega_1 * dt), np.cos(omega_1 * dt)],
        ]
    )
    np.testing.assert_allclose(A[1:3, 1:3], expected_k1, atol=1e-10)


def test_truncated_periodic_stationary_variances_match_bessel_coefficients():
    lengthscale = 0.7
    variance = 1.3
    K = 4
    sde = TruncatedPeriodicSDE(
        lengthscale=lengthscale,
        variance=variance,
        period=1.0,
        truncation_order=K,
    )
    P_inf = (
        np.asarray(sde.stationary_state_cov_sqrt)
        @ np.asarray(sde.stationary_state_cov_sqrt).T
    )

    bessel_argument = 1.0 / (4.0 * lengthscale**2)
    expected_dc = variance * scipy.special.ive(0, bessel_argument)
    np.testing.assert_allclose(P_inf[0, 0], expected_dc, atol=1e-12)

    for k in range(1, K + 1):
        expected_k = 2.0 * variance * scipy.special.ive(k, bessel_argument)
        np.testing.assert_allclose(
            P_inf[1 + 2 * (k - 1), 1 + 2 * (k - 1)], expected_k, atol=1e-12
        )
        np.testing.assert_allclose(
            P_inf[1 + 2 * (k - 1) + 1, 1 + 2 * (k - 1) + 1],
            expected_k,
            atol=1e-12,
        )


def test_truncated_periodic_matches_2d_dc_reference_at_tau_zero():
    """v5.1's 1-D DC and BayesNewton's 2-D DC must produce the same observable process.

    Specifically the kernel k(τ) = H P_∞ Hᵀ at τ=0 must match.
    """
    from tests._reference.two_d_dc_periodic import (
        two_d_dc_periodic_stationary_variance_diag,
    )

    lengthscale, variance, period, K = 0.5, 1.0, 2.0, 4
    sde = TruncatedPeriodicSDE(
        lengthscale=lengthscale,
        variance=variance,
        period=period,
        truncation_order=K,
    )
    P_inf_v51 = np.asarray(
        sde.stationary_state_cov_sqrt @ sde.stationary_state_cov_sqrt.T
    )
    H_v51 = np.asarray(sde.observation_matrix)
    k_tau0_v51 = (H_v51 @ P_inf_v51 @ H_v51.T)[0, 0]

    ref_diag = two_d_dc_periodic_stationary_variance_diag(
        lengthscale, variance, period, K
    )
    H_ref = np.zeros((1, 2 * (K + 1)))
    for k in range(K + 1):
        H_ref[0, 2 * k] = 1.0
    k_tau0_ref = (H_ref @ np.diag(ref_diag) @ H_ref.T)[0, 0]

    np.testing.assert_allclose(k_tau0_v51, k_tau0_ref, atol=1e-12)


def test_sum_sde_state_dim_is_sum_of_components():
    sde_a = Matern12SDE(lengthscale=1.0, variance=1.0)
    sde_b = Matern32SDE(lengthscale=0.5, variance=2.0)
    sde_sum = SumSDE(components=(sde_a, sde_b))
    assert sde_sum.state_dim == 1 + 2


def test_sum_sde_A_is_block_diagonal():
    sde_a = Matern12SDE(lengthscale=1.0, variance=1.0)
    sde_b = Matern32SDE(lengthscale=0.5, variance=2.0)
    sde_sum = SumSDE(components=(sde_a, sde_b))
    A_sum, _ = sde_sum.discretise(jnp.asarray(0.3))
    A_a, _ = sde_a.discretise(jnp.asarray(0.3))
    A_b, _ = sde_b.discretise(jnp.asarray(0.3))
    np.testing.assert_allclose(np.asarray(A_sum[:1, :1]), np.asarray(A_a), atol=1e-12)
    np.testing.assert_allclose(np.asarray(A_sum[1:, 1:]), np.asarray(A_b), atol=1e-12)
    np.testing.assert_allclose(np.asarray(A_sum[:1, 1:]), 0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(A_sum[1:, :1]), 0, atol=1e-12)


def test_sum_sde_components_remain_differentiable_pytree_leaves():
    def loss(lengthscale):
        sde_a = Matern12SDE(lengthscale=lengthscale, variance=1.0)
        sde_b = Matern32SDE(lengthscale=0.5, variance=2.0)
        sde_sum = SumSDE(components=(sde_a, sde_b))
        A_sum, _ = sde_sum.discretise(jnp.asarray(0.3))
        return jnp.sum(A_sum)

    grad_value = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(grad_value)
    assert grad_value != 0.0


def test_sum_sde_observation_matrix_is_row_concatenation():
    sde_a = Matern12SDE(lengthscale=1.0, variance=1.0)
    sde_b = Matern32SDE(lengthscale=0.5, variance=2.0)
    sde_sum = SumSDE(components=(sde_a, sde_b))
    sum_state_dim = sde_a.state_dim + sde_b.state_dim
    assert sde_sum.observation_matrix.shape == (1, sum_state_dim)
    np.testing.assert_allclose(
        np.asarray(sde_sum.observation_matrix[:, : sde_a.state_dim]),
        np.asarray(sde_a.observation_matrix),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(sde_sum.observation_matrix[:, sde_a.state_dim :]),
        np.asarray(sde_b.observation_matrix),
        atol=1e-14,
    )


def test_sum_sde_zero_dt_returns_identity_and_zero_noise():
    sde_a = Matern12SDE(lengthscale=1.0, variance=1.0)
    sde_b = Matern32SDE(lengthscale=0.5, variance=2.0)
    sde_sum = SumSDE(components=(sde_a, sde_b))
    A, L_Q = sde_sum.discretise(jnp.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(A), np.eye(sde_sum.state_dim))
    np.testing.assert_array_equal(
        np.asarray(L_Q), np.zeros((sde_sum.state_dim, sde_sum.state_dim))
    )


@pytest.mark.parametrize("sde_cls", [Matern32SDE, Matern52SDE])
@pytest.mark.parametrize("n", [50, 200, 1000, 2000])
def test_matern_sde_gradient_through_lengthscale_at_realistic_N(sde_cls, n):
    """Gradient through the Matern SDE must be finite for time-densely-sampled
    data, where Q can pick up round-off-induced tiny negative eigenvalues."""
    times = jnp.linspace(0.0, 20.0, n)

    def loss(lengthscale):
        sde = sde_cls(lengthscale=lengthscale, variance=1.0)

        def step_loss(time_step):
            A, L_Q = sde.discretise(time_step)
            return jnp.sum(A) + jnp.sum(L_Q @ L_Q.T)

        time_steps = jnp.diff(times)
        return jax.vmap(step_loss)(time_steps).sum()

    grad_value = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(grad_value)


@pytest.mark.parametrize("sde_name", ["Matern12SDE", "Matern32SDE", "Matern52SDE"])
def test_continuous_lyapunov_stationarity(sde_name):
    """F P∞ + P∞ Fᵀ + L Qc Lᵀ = 0 ties the stored Qc to the drift and P∞."""
    import gpjax.state_space.sde as sde_module

    sde = getattr(sde_module, sde_name)(lengthscale=0.7, variance=1.3)
    drift = sde.drift_matrix
    diffusion = sde.diffusion_matrix
    spectral_density = sde.process_noise_spectral_density
    stationary_cov = sde.stationary_state_cov_sqrt @ sde.stationary_state_cov_sqrt.T

    residual = (
        drift @ stationary_cov
        + stationary_cov @ drift.T
        + diffusion @ spectral_density @ diffusion.T
    )
    assert jnp.allclose(residual, 0.0, atol=1e-9)


# --- ProductSDE (quasi-periodic TruncatedPeriodic x Matern) -----------------


@pytest.mark.parametrize(
    ("matern_cls", "matern_state_dim"),
    [(Matern12SDE, 1), (Matern32SDE, 2), (Matern52SDE, 3)],
)
@pytest.mark.parametrize("truncation_order", [1, 3])
def test_product_sde_state_dim_is_product_of_component_state_dims(
    matern_cls, matern_state_dim, truncation_order
):
    periodic = TruncatedPeriodicSDE(
        lengthscale=0.6, variance=1.1, period=1.5, truncation_order=truncation_order
    )
    matern = matern_cls(lengthscale=1.2, variance=0.8)
    product = ProductSDE(components=(periodic, matern))
    expected_state_dim = (1 + 2 * truncation_order) * matern_state_dim
    assert product.state_dim == expected_state_dim
    assert product.drift_matrix.shape == (expected_state_dim, expected_state_dim)
    assert product.observation_matrix.shape == (1, expected_state_dim)


@pytest.mark.parametrize("matern_cls", [Matern12SDE, Matern32SDE, Matern52SDE])
def test_product_sde_lyapunov_identity(matern_cls):
    """F P∞ + P∞ Fᵀ + L Qc Lᵀ = 0 must hold for the Kronecker-composed product."""
    periodic = TruncatedPeriodicSDE(
        lengthscale=0.6, variance=1.1, period=1.5, truncation_order=3
    )
    matern = matern_cls(lengthscale=1.2, variance=0.8)
    product = ProductSDE(components=(periodic, matern))

    drift = np.asarray(product.drift_matrix)
    diffusion = np.asarray(product.diffusion_matrix)
    spectral_density = np.asarray(product.process_noise_spectral_density)
    stationary_cov = np.asarray(
        product.stationary_state_cov_sqrt @ product.stationary_state_cov_sqrt.T
    )

    residual = (
        drift @ stationary_cov
        + stationary_cov @ drift.T
        + diffusion @ spectral_density @ diffusion.T
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


@pytest.mark.parametrize("matern_cls", [Matern12SDE, Matern32SDE, Matern52SDE])
def test_product_sde_A_equals_kron_of_component_transitions(matern_cls):
    periodic = TruncatedPeriodicSDE(
        lengthscale=0.6, variance=1.1, period=1.5, truncation_order=2
    )
    matern = matern_cls(lengthscale=1.2, variance=0.8)
    product = ProductSDE(components=(periodic, matern))

    dt = jnp.asarray(0.37)
    A_product, _ = product.discretise(dt)
    A_periodic, _ = periodic.discretise(dt)
    A_matern, _ = matern.discretise(dt)
    expected = jnp.kron(A_periodic, A_matern)
    np.testing.assert_allclose(np.asarray(A_product), np.asarray(expected), atol=1e-10)


@pytest.mark.parametrize("matern_cls", [Matern12SDE, Matern32SDE, Matern52SDE])
def test_product_sde_A_matches_matrix_exponential_of_kronecker_sum_drift(matern_cls):
    """A(Δt) = expm(FΔt) must hold at the product level too, not just per-factor."""
    periodic = TruncatedPeriodicSDE(
        lengthscale=0.6, variance=1.1, period=1.5, truncation_order=2
    )
    matern = matern_cls(lengthscale=1.2, variance=0.8)
    product = ProductSDE(components=(periodic, matern))

    dt = 0.29
    A_product, _ = product.discretise(jnp.asarray(dt))
    expected = scipy.linalg.expm(np.asarray(product.drift_matrix) * dt)
    np.testing.assert_allclose(np.asarray(A_product), expected, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("matern_cls", [Matern12SDE, Matern32SDE, Matern52SDE])
def test_product_sde_Q_equals_stationary_identity(matern_cls):
    periodic = TruncatedPeriodicSDE(
        lengthscale=0.6, variance=1.1, period=1.5, truncation_order=2
    )
    matern = matern_cls(lengthscale=1.2, variance=0.8)
    product = ProductSDE(components=(periodic, matern))

    dt = jnp.asarray(0.8)
    A, L_Q = product.discretise(dt)
    Q = np.asarray(L_Q @ L_Q.T)
    P_inf = np.asarray(
        product.stationary_state_cov_sqrt @ product.stationary_state_cov_sqrt.T
    )
    expected = P_inf - np.asarray(A) @ P_inf @ np.asarray(A).T
    np.testing.assert_allclose(Q, expected, atol=1e-8, rtol=1e-7)


def test_product_sde_zero_dt_returns_identity_and_zero_noise():
    periodic = TruncatedPeriodicSDE(
        lengthscale=0.6, variance=1.1, period=1.5, truncation_order=2
    )
    matern = Matern32SDE(lengthscale=1.2, variance=0.8)
    product = ProductSDE(components=(periodic, matern))

    A, L_Q = product.discretise(jnp.asarray(0.0))
    np.testing.assert_allclose(np.asarray(A), np.eye(product.state_dim), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(L_Q), np.zeros((product.state_dim, product.state_dim)), atol=1e-12
    )


def test_product_sde_observation_matrix_is_kron_of_components():
    periodic = TruncatedPeriodicSDE(
        lengthscale=0.6, variance=1.1, period=1.5, truncation_order=2
    )
    matern = Matern32SDE(lengthscale=1.2, variance=0.8)
    product = ProductSDE(components=(periodic, matern))
    expected = jnp.kron(periodic.observation_matrix, matern.observation_matrix)
    np.testing.assert_allclose(
        np.asarray(product.observation_matrix), np.asarray(expected), atol=1e-14
    )


def test_product_sde_is_jittable_and_vmap_compatible():
    periodic = TruncatedPeriodicSDE(
        lengthscale=0.6, variance=1.1, period=1.5, truncation_order=2
    )
    matern = Matern32SDE(lengthscale=1.2, variance=0.8)
    product = ProductSDE(components=(periodic, matern))
    state_dim = product.state_dim

    discretise_jitted = eqx.filter_jit(product.discretise)
    A, L_Q = discretise_jitted(jnp.asarray(0.5))
    assert A.shape == (state_dim, state_dim)
    assert L_Q.shape == (state_dim, state_dim)

    dts = jnp.array([0.1, 0.5, 1.0])
    A_batched, L_Q_batched = jax.vmap(product.discretise)(dts)
    assert A_batched.shape == (3, state_dim, state_dim)
    assert L_Q_batched.shape == (3, state_dim, state_dim)


def test_product_sde_gradient_through_matern_lengthscale():
    def loss(lengthscale):
        periodic = TruncatedPeriodicSDE(
            lengthscale=0.6, variance=1.1, period=1.5, truncation_order=2
        )
        matern = Matern32SDE(lengthscale=lengthscale, variance=0.8)
        product = ProductSDE(components=(periodic, matern))
        A, L_Q = product.discretise(jnp.asarray(0.5))
        return jnp.sum(A) + jnp.sum(L_Q @ L_Q.T)

    grad_value = jax.grad(loss)(jnp.asarray(1.2))
    assert jnp.isfinite(grad_value)
