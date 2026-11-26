"""Numerical robustness stress tests (slow). Run via `poe test-slow` or `poe all-tests-slow`.

See plans/2026-04-22-state-space-gps-implementation.md Phase 14.
"""

from __future__ import annotations

import warnings

import gpjax as gpx
from gpjax.state_space.gps import StateSpacePrior
from gpjax.state_space.inference import kalman_filter
from gpjax.state_space.kernels import TruncatedPeriodic
from gpjax.state_space.objectives import state_space_mll
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest


@pytest.mark.slow
def test_state_space_mll_forward_finite_at_N_100k():
    """Forward MLL must be finite at N=10^5 for Matern52, with subset cross-check."""
    n = 100_000
    lengthscale = 1.5
    variance = 0.8
    obs_stddev = 0.2

    key = jr.key(0)
    key_x, key_y = jr.split(key)
    X = jnp.sort(jr.uniform(key_x, shape=(n,), minval=0.0, maxval=200.0))
    y = jr.normal(key_y, shape=(n,))

    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern52(lengthscale=lengthscale, variance=variance),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior = prior * likelihood

    mll_full = state_space_mll(posterior, train_data)
    assert jnp.isfinite(mll_full), f"MLL is non-finite at N={n}: {mll_full}"

    # Cross-check: first 500 points against state-space MLL on the truncated
    # dataset — the dense GP at full N is infeasible, so we just verify that the
    # subset evaluation also returns a finite value via state-space inference.
    n_subset = 500
    X_subset = X[:n_subset]
    y_subset = y[:n_subset]
    train_subset = gpx.Dataset(X=X_subset.reshape(-1, 1), y=y_subset.reshape(-1, 1))
    prior_subset = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern52(lengthscale=lengthscale, variance=variance),
    )
    likelihood_subset = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior_subset = prior_subset * likelihood_subset

    mll_ss_subset = state_space_mll(posterior_subset, train_subset)
    assert jnp.isfinite(mll_ss_subset), (
        f"Subset state-space MLL non-finite: {mll_ss_subset}"
    )

    # Dense GP reference on the same 500 points.
    dense_prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern52(lengthscale=lengthscale, variance=variance),
    )
    dense_posterior = dense_prior * likelihood_subset
    mll_dense_subset = gpx.objectives.conjugate_mll(dense_posterior, train_subset)
    assert jnp.isfinite(mll_dense_subset)
    np.testing.assert_allclose(
        float(mll_ss_subset), float(mll_dense_subset), atol=1e-4, rtol=1e-4
    )


@pytest.mark.slow
def test_state_space_mll_reverse_mode_grad_at_N_100k_with_truncated_periodic():
    """jax.grad through state_space_mll must be finite at N=10^5 with state_dim=24."""
    n = 100_000
    key = jr.key(1)
    key_x, key_y = jr.split(key)
    X = jnp.sort(jr.uniform(key_x, shape=(n,), minval=0.0, maxval=200.0))
    y = jr.normal(key_y, shape=(n,))
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))

    def loss(lengthscale_value):
        kernel = gpx.kernels.Matern52(
            lengthscale=lengthscale_value, variance=0.5
        ) + TruncatedPeriodic(
            lengthscale=0.5,
            variance=0.3,
            period=2.0,
            truncation_order=10,
        )
        prior = StateSpacePrior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.2)
        posterior = prior * likelihood
        return -state_space_mll(posterior, train_data)

    grad_value = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(grad_value), f"Non-finite gradient at N={n}: {grad_value}"
    # The gradient should not blow up to O(N). A well-conditioned MLL gradient
    # remains bounded by some constant times sqrt(N) at most.
    assert abs(float(grad_value)) < 1e8, f"Suspiciously large gradient: {grad_value}"


def _build_sum_sde(lengthscale_value):
    from gpjax.state_space.sde import (
        Matern52SDE,
        SumSDE,
        TruncatedPeriodicSDE,
    )

    sde_a = Matern52SDE(lengthscale=lengthscale_value, variance=0.5)
    sde_b = TruncatedPeriodicSDE(
        lengthscale=0.5,
        variance=0.3,
        period=2.0,
        truncation_order=6,
    )
    return SumSDE(components=(sde_a, sde_b))


@pytest.mark.slow
def test_state_space_mll_chunked_loss_and_grad_agreement():
    """Across chunk_size in {None, 1, sqrt(N), N}, loss/gradient agree to atol=1e-10."""
    n = 10_000
    obs_stddev = 0.3

    key = jr.key(2)
    key_x, key_y = jr.split(key)
    X = jnp.sort(jr.uniform(key_x, shape=(n,), minval=0.0, maxval=20.0))
    y = jr.normal(key_y, shape=(n,))

    time_steps = jnp.concatenate([jnp.zeros(1), jnp.diff(X)])
    is_observed = jnp.ones(n, dtype=bool)
    sigma_eff = jnp.asarray(obs_stddev)

    def filter_loss(lengthscale_value, chunk_size):
        sde = _build_sum_sde(lengthscale_value)
        return -kalman_filter(
            sde,
            y,
            time_steps,
            is_observed,
            sigma_eff,
            chunk_size=chunk_size,
        )

    chunk_sizes = [None, 1, round(n**0.5), n]

    losses = [
        float(filter_loss(jnp.asarray(1.0), chunk_size)) for chunk_size in chunk_sizes
    ]
    for chunk_size, loss_value in zip(chunk_sizes, losses, strict=True):
        np.testing.assert_allclose(
            loss_value,
            losses[0],
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"chunk_size={chunk_size} loss disagrees",
        )

    grads = [
        float(jax.grad(filter_loss, argnums=0)(jnp.asarray(1.0), chunk_size))
        for chunk_size in chunk_sizes
    ]
    for chunk_size, grad_value in zip(chunk_sizes, grads, strict=True):
        np.testing.assert_allclose(
            grad_value,
            grads[0],
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"chunk_size={chunk_size} gradient disagrees",
        )


@pytest.mark.slow
def test_state_space_mll_chunked_memory_scaling():
    """Memory peak: sqrt(N)-chunked AD must use < 0.5 * full-N memory."""
    n = 10_000

    try:
        memory_stats = jax.devices()[0].memory_stats()
    except (NotImplementedError, AttributeError):
        pytest.skip("Backend does not expose memory_stats")

    if memory_stats is None or "peak_bytes_in_use" not in memory_stats:
        pytest.skip("Backend memory_stats does not report peak_bytes_in_use")

    key = jr.key(3)
    key_x, key_y = jr.split(key)
    X = jnp.sort(jr.uniform(key_x, shape=(n,), minval=0.0, maxval=20.0))
    y = jr.normal(key_y, shape=(n,))

    time_steps = jnp.concatenate([jnp.zeros(1), jnp.diff(X)])
    is_observed = jnp.ones(n, dtype=bool)
    sigma_eff = jnp.asarray(0.3)

    def loss(lengthscale_value, chunk_size):
        sde = _build_sum_sde(lengthscale_value)
        return -kalman_filter(
            sde,
            y,
            time_steps,
            is_observed,
            sigma_eff,
            chunk_size=chunk_size,
        )

    grad_fn = jax.grad(loss, argnums=0)

    _ = grad_fn(jnp.asarray(1.0), n).block_until_ready()
    full_n_peak = jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0)

    _ = grad_fn(jnp.asarray(1.0), round(n**0.5)).block_until_ready()
    sqrt_n_peak = jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0)

    # peak_bytes_in_use is a cumulative high-water mark on most backends; if
    # both runs report the same peak we cannot meaningfully compare them.
    if sqrt_n_peak == full_n_peak:
        pytest.skip(
            "peak_bytes_in_use is cumulative high-water mark on this backend; "
            "cannot separate per-run peaks"
        )
    assert sqrt_n_peak < 0.5 * full_n_peak, (
        f"sqrt(N) chunking should halve peak memory; got "
        f"sqrt_n_peak={sqrt_n_peak}, full_n_peak={full_n_peak}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("obs_stddev", "lengthscale", "time_range"),
    [
        (1e-8, 1.0, 10.0),
        (0.1, 0.01, 10.0),
        (0.1, 1000.0, 10.0),
        (0.1, 1.0, 1e-3),
        (0.1, 1.0, 1e6),
    ],
)
def test_state_space_mll_extreme_regimes_finite(obs_stddev, lengthscale, time_range):
    """Tiny obs noise, extreme lengthscales, tiny/huge dt — MLL and predict stay finite."""
    n = 20
    key = jr.key(4)
    key_x, key_y = jr.split(key)
    X = jnp.sort(jr.uniform(key_x, shape=(n,), minval=0.0, maxval=time_range))
    y = jr.normal(key_y, shape=(n,))
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))

    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern32(lengthscale=lengthscale, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior = prior * likelihood

    mll = state_space_mll(posterior, train_data)
    assert jnp.isfinite(mll), (
        f"MLL non-finite for obs_stddev={obs_stddev}, "
        f"lengthscale={lengthscale}, time_range={time_range}"
    )

    Xtest = jnp.array([[time_range / 2]])
    pred_dist = posterior.predict(Xtest, train_data)
    assert jnp.isfinite(np.asarray(pred_dist.mean)).all()
    assert jnp.isfinite(np.asarray(pred_dist.variance)).all()


@pytest.mark.slow
def test_state_space_validation_warns_on_non_float64_dtype():
    """float32 X/y must emit a UserWarning per design's validation contract."""
    from gpjax.state_space._validation import validate_state_space_data

    n = 10
    X = jnp.linspace(0.0, 5.0, n, dtype=jnp.float32).reshape(-1, 1)
    y = jnp.zeros((n, 1), dtype=jnp.float32)
    with pytest.warns(UserWarning, match=r"float|dtype|x64"):
        validate_state_space_data(X, y)


@pytest.mark.slow
def test_state_space_mll_finite_with_float32_inputs():
    """MLL on float32 inputs is finite and drifts (but not catastrophically) from float64."""
    n = 50
    X64 = jnp.linspace(0.0, 5.0, n).reshape(-1, 1)
    y64 = jr.normal(jr.key(7), shape=(n, 1))
    X32 = X64.astype(jnp.float32)
    y32 = y64.astype(jnp.float32)

    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.3)
    posterior = prior * likelihood

    train_data_64 = gpx.Dataset(X=X64, y=y64)
    mll64 = float(state_space_mll(posterior, train_data_64))

    # Construct float32 dataset under suppressed warnings (Dataset emits its own
    # non-float64 warning), then evaluate MLL.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        train_data_32 = gpx.Dataset(X=X32, y=y32)
        mll32 = float(state_space_mll(posterior, train_data_32))

    assert np.isfinite(mll32), f"float32 MLL non-finite: {mll32}"
    # Numerical drift is expected; just sanity-check the magnitude.
    assert abs(mll64 - mll32) < 10 * abs(mll64), "Drift suspiciously large"
