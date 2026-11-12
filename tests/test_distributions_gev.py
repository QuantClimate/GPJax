# Copyright 2026 The GPJax Contributors. All Rights Reserved.
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
"""Tests for the generalised extreme value (GEV) distribution.

SciPy's ``genextreme`` uses the shape ``c = -xi``. The tests use the climate
convention ``xi`` (``xi > 0`` is the heavy-tailed Frechet type). A reference in
50-digit decimal arithmetic checks values where SciPy is not exact.
"""

from decimal import (
    Decimal,
    localcontext,
)
import itertools

import gpjax.distributions as gpd
from gpjax.distributions import (
    GeneralisedExtremeValue,
    gev_return_level,
    gev_return_period,
)
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy import (
    integrate,
    stats,
)

config.update("jax_enable_x64", True)

XI_GRID = [-0.5, -0.3, -0.1, -1e-4, -1e-8, 0.0, 1e-8, 1e-4, 0.1, 0.3, 0.5]
# The documented bound on |mass - 1| of the continued pseudo-density for
# -0.5 <= xi <= 0.5.
CONTINUATION_MASS_BOUND = 3e-5
NEAR_GUMBEL = [-1e-4, -1e-8, 0.0, 1e-8, 1e-4]
LOC, SCALE = 1.5, 2.0
PROBS = np.array([1e-8, 1e-3, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-6])


def _gev(xi, *, continuation=False, loc=LOC, scale=SCALE):
    return GeneralisedExtremeValue(
        loc=jnp.asarray(loc),
        scale=jnp.asarray(scale),
        concentration=jnp.asarray(xi),
        support_continuation=continuation,
    )


def _scipy(xi):
    return stats.genextreme(c=-xi, loc=LOC, scale=SCALE)


def _points(xi):
    """Points inside the support, at fixed probabilities."""
    return _scipy(xi).ppf(PROBS)


# ---------------------------------------------------------------------------
# High-precision reference
# ---------------------------------------------------------------------------


def _decimal(value) -> Decimal:
    return Decimal(float(value))


def _reference(y, xi, loc=LOC, scale=SCALE):
    """Log density, cdf and sf in 50-digit arithmetic; None outside the support."""
    with localcontext() as ctx:
        ctx.prec = 50
        y, xi, loc, scale = map(_decimal, (y, xi, loc, scale))
        z = (y - loc) / scale
        if xi == 0:
            u = z
        else:
            t = 1 + xi * z
            if t <= 0:
                return None
            u = t.ln() / xi
        e = (-u).exp()
        log_pdf = -scale.ln() - (1 + xi) * u - e
        cdf = (-e).exp()
        return float(log_pdf), float(cdf), float(1 - cdf)


def _reference_log_pdf(y, xi, loc=LOC, scale=SCALE) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 50
        y, xi, loc, scale = map(Decimal, (y, xi, loc, scale))
        z = (y - loc) / scale
        u = z if xi == 0 else (1 + xi * z).ln() / xi
        return -scale.ln() - (1 + xi) * u - (-u).exp()


def _reference_quantile(log_neg_log, xi, loc=LOC, scale=SCALE) -> float:
    """Quantile from L = log(-log F) in 50-digit arithmetic."""
    with localcontext() as ctx:
        ctx.prec = 50
        xi, loc, scale = map(_decimal, (xi, loc, scale))
        z = -log_neg_log if xi == 0 else ((-xi * log_neg_log).exp() - 1) / xi
        return float(loc + scale * z)


def _reference_icdf(q, xi):
    with localcontext() as ctx:
        ctx.prec = 50
        return _reference_quantile((-(_decimal(q).ln())).ln(), xi)


def _reference_isf(p, xi):
    with localcontext() as ctx:
        ctx.prec = 50
        return _reference_quantile((-((1 - _decimal(p)).ln())).ln(), xi)


# ---------------------------------------------------------------------------
# Agreement with SciPy and with the high-precision reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("xi", XI_GRID)
def test_log_prob_matches_scipy(xi):
    y = _points(xi)
    ours = _gev(xi).log_prob(jnp.asarray(y))
    np.testing.assert_allclose(ours, _scipy(xi).logpdf(y), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("xi", XI_GRID)
def test_log_prob_cdf_and_sf_match_the_high_precision_reference(xi):
    y = _points(xi)
    dist = _gev(xi)
    reference = np.array([_reference(value, xi) for value in y])
    np.testing.assert_allclose(
        dist.log_prob(jnp.asarray(y)), reference[:, 0], rtol=1e-12
    )
    np.testing.assert_allclose(dist.cdf(jnp.asarray(y)), reference[:, 1], rtol=1e-12)
    np.testing.assert_allclose(dist.sf(jnp.asarray(y)), reference[:, 2], rtol=1e-12)


@pytest.mark.parametrize("xi", XI_GRID)
def test_cdf_and_sf_match_scipy(xi):
    y = _points(xi)
    dist = _gev(xi)
    np.testing.assert_allclose(dist.cdf(jnp.asarray(y)), _scipy(xi).cdf(y), rtol=1e-9)
    np.testing.assert_allclose(dist.sf(jnp.asarray(y)), _scipy(xi).sf(y), rtol=1e-9)


@pytest.mark.parametrize("xi", XI_GRID)
@pytest.mark.parametrize("probability", [1e-15, 1e-12, 1e-6])
def test_sf_keeps_the_digits_of_small_probabilities(xi, probability):
    y = _reference_isf(probability, xi)
    sf = _gev(xi).sf(jnp.asarray(y))
    np.testing.assert_allclose(sf, _reference(y, xi)[2], rtol=1e-9)
    np.testing.assert_allclose(sf, _scipy(xi).sf(y), rtol=1e-8)


@pytest.mark.parametrize("xi", XI_GRID)
def test_icdf_matches_scipy_and_the_reference(xi):
    q = jnp.asarray(PROBS)
    ours = _gev(xi).icdf(q)
    np.testing.assert_allclose(ours, _scipy(xi).ppf(PROBS), rtol=1e-9, atol=1e-12)
    reference = [_reference_icdf(value, xi) for value in PROBS]
    np.testing.assert_allclose(ours, reference, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("xi", XI_GRID)
def test_isf_matches_scipy_and_the_reference(xi):
    p = np.array([1e-15, 1e-12, 1e-6, 0.01, 0.5, 0.9])
    ours = _gev(xi).isf(jnp.asarray(p))
    np.testing.assert_allclose(ours, _scipy(xi).isf(p), rtol=1e-9, atol=1e-12)
    reference = [_reference_isf(value, xi) for value in p]
    np.testing.assert_allclose(ours, reference, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("xi", XI_GRID)
def test_icdf_inverts_cdf(xi):
    y = jnp.asarray(_points(xi)[:-1])
    dist = _gev(xi)
    np.testing.assert_allclose(dist.icdf(dist.cdf(y)), y, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(dist.isf(dist.sf(y)), y, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("continuation", [False, True])
def test_quantiles_at_zero_and_one_are_the_end_points(continuation):
    q = jnp.array([0.0, 1.0])
    heavy = _gev(0.25, continuation=continuation).icdf(q)
    bounded = _gev(-0.25, continuation=continuation).icdf(q)
    gumbel = _gev(0.0, continuation=continuation).icdf(q)
    np.testing.assert_allclose(heavy, [LOC - SCALE / 0.25, jnp.inf])
    np.testing.assert_allclose(bounded, [-jnp.inf, LOC + SCALE / 0.25])
    np.testing.assert_allclose(gumbel, [-jnp.inf, jnp.inf])
    assert jnp.isnan(_gev(0.1).icdf(jnp.array(1.5)))


# ---------------------------------------------------------------------------
# Exact tail probabilities outside the support
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("continuation", [False, True])
def test_cdf_and_sf_are_exact_outside_the_support(continuation):
    heavy = _gev(0.3, continuation=continuation)
    lower_end = LOC - SCALE / 0.3
    below = jnp.array([lower_end - 1e3, lower_end - 1.0, lower_end - 1e-9, lower_end])
    assert jnp.all(heavy.cdf(below) == 0.0)
    assert jnp.all(heavy.sf(below) == 1.0)

    bounded = _gev(-0.3, continuation=continuation)
    upper_end = LOC + SCALE / 0.3
    above = jnp.array([upper_end, upper_end + 1e-9, upper_end + 1.0, upper_end + 1e3])
    assert jnp.all(bounded.cdf(above) == 1.0)
    assert jnp.all(bounded.sf(above) == 0.0)


def test_cdf_has_no_overflow_near_the_lower_end_point():
    heavy = _gev(0.5)
    lower_end = LOC - SCALE / 0.5
    y = jnp.array([lower_end + 1e-300, lower_end + 1e-12, lower_end + 1e-3])
    assert jnp.all(heavy.cdf(y) == 0.0)
    assert jnp.all(heavy.sf(y) == 1.0)
    grads = jax.vmap(jax.grad(heavy.cdf))(y)
    assert jnp.all(jnp.isfinite(grads))


def test_return_period_is_infinite_where_the_exceedance_probability_is_zero():
    bounded = _gev(-0.3)
    upper_end = LOC + SCALE / 0.3
    level = jnp.array([upper_end, upper_end + 5.0])
    periods = bounded.return_period(level)
    assert jnp.all(jnp.isposinf(periods))
    grads = jax.vmap(jax.grad(bounded.return_period))(level)
    assert jnp.all(jnp.isfinite(grads))
    # Inside the support, the return period is finite.
    assert jnp.isfinite(bounded.return_period(jnp.array(upper_end - 0.5)))


def test_log_prob_is_minus_infinity_outside_the_support_without_continuation():
    heavy = _gev(0.3)
    bounded = _gev(-0.3)
    assert jnp.isneginf(heavy.log_prob(jnp.array(LOC - SCALE / 0.3 - 1.0)))
    assert jnp.isneginf(bounded.log_prob(jnp.array(LOC + SCALE / 0.3 + 1.0)))
    grad = jax.grad(heavy.log_prob)(jnp.array(LOC - SCALE / 0.3 - 1.0))
    assert not jnp.isnan(grad)


# ---------------------------------------------------------------------------
# Return levels and return periods
# ---------------------------------------------------------------------------

RETURN_PERIODS = np.array([1.5, 2.0, 10.0, 100.0, 1e3, 1e6, 1e12])


@pytest.mark.parametrize("xi", XI_GRID)
def test_return_level_matches_the_closed_form(xi):
    ours = _gev(xi).return_level(jnp.asarray(RETURN_PERIODS))
    reference = [_reference_isf(1.0 / period, xi) for period in RETURN_PERIODS]
    np.testing.assert_allclose(ours, reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        ours, _scipy(xi).isf(1.0 / RETURN_PERIODS), rtol=1e-9, atol=1e-12
    )
    if xi not in (0.0, 1e-8, -1e-8, 1e-4, -1e-4):
        # The textbook form, which is accurate away from the Gumbel limit.
        s = -np.log1p(-1.0 / RETURN_PERIODS)
        textbook = LOC - SCALE / xi * (1.0 - s ** (-xi))
        np.testing.assert_allclose(ours, textbook, rtol=1e-10)


def test_return_level_in_the_gumbel_limit():
    gumbel = LOC - SCALE * np.log(-np.log1p(-1.0 / RETURN_PERIODS))
    np.testing.assert_allclose(
        _gev(0.0).return_level(jnp.asarray(RETURN_PERIODS)), gumbel, rtol=1e-14
    )
    for xi in (-1e-8, 1e-8):
        near = _gev(xi).return_level(jnp.asarray(RETURN_PERIODS))
        # The first-order term is -scale * xi * L**2 / 2, with |L| < 28 here.
        np.testing.assert_allclose(near, gumbel, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("xi", XI_GRID)
def test_return_level_increases_with_the_return_period(xi):
    periods = jnp.geomspace(1.01, 1e9, 200)
    levels = _gev(xi).return_level(periods)
    assert jnp.all(jnp.diff(levels) > 0.0)


@pytest.mark.parametrize("xi", XI_GRID)
def test_return_period_inverts_return_level(xi):
    periods = jnp.asarray(RETURN_PERIODS)
    dist = _gev(xi)
    np.testing.assert_allclose(
        dist.return_period(dist.return_level(periods)), periods, rtol=1e-6
    )


def test_return_level_outside_the_valid_range():
    dist = _gev(0.2)
    assert jnp.isnan(dist.return_level(jnp.array(0.5)))
    np.testing.assert_allclose(dist.return_level(jnp.array(1.0)), LOC - SCALE / 0.2)
    np.testing.assert_allclose(_gev(-0.2).return_level(jnp.inf), LOC + SCALE / 0.2)


def test_pure_functions_match_the_methods():
    xi = jnp.array([-0.2, 0.0, 0.3])
    dist = GeneralisedExtremeValue(LOC, SCALE, xi)
    periods = jnp.array([[10.0], [100.0]])
    np.testing.assert_allclose(
        gev_return_level(periods, LOC, SCALE, xi), dist.return_level(periods)
    )
    levels = jnp.array([[3.0], [8.0]])
    np.testing.assert_allclose(
        gev_return_period(levels, LOC, SCALE, xi), dist.return_period(levels)
    )
    assert gev_return_level(periods, LOC, SCALE, xi).shape == (2, 3)


def test_pure_functions_broadcast_over_parameter_samples():
    key_loc, key_scale, key_shape = jr.split(jr.key(0), 3)
    loc = jr.normal(key_loc, (50, 4))
    scale = jnp.exp(0.1 * jr.normal(key_scale, (50, 1)))
    shape = 0.1 * jr.normal(key_shape, (50, 1))
    levels = gev_return_level(jnp.array(100.0), loc, scale, shape)
    assert levels.shape == (50, 4)
    expected = GeneralisedExtremeValue(loc, scale, shape).isf(jnp.array(0.01))
    np.testing.assert_allclose(levels, expected)


# ---------------------------------------------------------------------------
# Sampling, mean and variance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.3])
def test_samples_pass_a_kolmogorov_smirnov_test(xi):
    samples = _gev(xi).sample(jr.key(7), (4000,))
    result = stats.kstest(np.asarray(samples), _scipy(xi).cdf)
    assert result.pvalue > 1e-3


def test_sample_shape():
    dist = GeneralisedExtremeValue(
        jnp.zeros(2), jnp.ones(2), jnp.array([-0.2, 0.2]), support_continuation=True
    )
    assert dist.batch_shape == (2,)
    assert dist.sample(jr.key(0)).shape == (2,)
    assert dist.sample(jr.key(0), (3, 4)).shape == (3, 4, 2)


@pytest.mark.parametrize("xi", [x for x in XI_GRID if x < 0.5])
def test_mean_and_variance_match_scipy(xi):
    dist = _gev(xi)
    mean, variance = _scipy(xi).stats(moments="mv")
    # SciPy's variance loses about 1e-8 of relative accuracy near xi = 0; the
    # Gumbel-limit test below checks that region against the Taylor series.
    variance_rtol = 1e-8 if abs(xi) >= 0.1 or xi == 0.0 else 1e-7
    np.testing.assert_allclose(dist.mean, mean, rtol=1e-8)
    np.testing.assert_allclose(dist.variance, variance, rtol=variance_rtol)


@pytest.mark.parametrize("xi", [-1e-8, 0.0, 1e-8])
def test_mean_and_variance_in_the_gumbel_limit(xi):
    # First-order Taylor series in xi:
    # mean: gamma + xi (gamma^2 + zeta(2)) / 2;
    # variance: zeta(2) + xi (2 zeta(3) + 2 gamma zeta(2)).
    gamma, zeta2, zeta3 = np.euler_gamma, np.pi**2 / 6.0, 1.2020569031595942
    mean = LOC + SCALE * (gamma + xi * 0.5 * (gamma**2 + zeta2))
    variance = SCALE**2 * (zeta2 + xi * (2.0 * zeta3 + 2.0 * gamma * zeta2))
    np.testing.assert_allclose(_gev(xi).mean, mean, rtol=1e-14)
    np.testing.assert_allclose(_gev(xi).variance, variance, rtol=1e-14)


def test_mean_and_variance_are_infinite_where_they_do_not_exist():
    assert jnp.isposinf(_gev(1.0).mean)
    assert jnp.isposinf(_gev(1.5).mean)
    assert jnp.isposinf(_gev(0.5).variance)
    assert jnp.isposinf(_gev(0.7).variance)
    assert jnp.isfinite(_gev(0.7).mean)


# ---------------------------------------------------------------------------
# The Gumbel limit: finite, correct and continuous values and gradients
# ---------------------------------------------------------------------------


def _all_methods(continuation):
    def evaluate(name, y, loc, scale, xi):
        dist = GeneralisedExtremeValue(
            loc, scale, xi, support_continuation=continuation
        )
        if name in ("icdf", "isf"):
            return getattr(dist, name)(jax.nn.sigmoid(y))
        if name == "return_level":
            return dist.return_level(2.0 + jnp.exp(y))
        return getattr(dist, name)(y)

    return evaluate


METHODS = ["log_prob", "cdf", "sf", "icdf", "isf", "return_level", "return_period"]


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("xi", NEAR_GUMBEL)
def test_values_and_gradients_are_finite_near_the_gumbel_limit(continuation, name, xi):
    evaluate = _all_methods(continuation)
    for y in (-3.0, -0.5, 0.0, 0.7, 4.0, 15.0):
        args = tuple(jnp.asarray(v) for v in (y, LOC, SCALE, xi))
        value = evaluate(name, *args)
        grads = jax.grad(lambda *a: evaluate(name, *a), argnums=(0, 1, 2, 3))(*args)
        assert jnp.isfinite(value), (name, y)
        assert all(jnp.isfinite(g) for g in grads), (name, y, grads)


@pytest.mark.parametrize("continuation", [False, True])
def test_log_prob_gradient_at_zero_shape_matches_the_gumbel_derivatives(continuation):
    z = jnp.linspace(-3.0, 6.0, 19)
    y = LOC + SCALE * z

    def log_prob(y, loc, scale, xi):
        return GeneralisedExtremeValue(
            loc, scale, xi, support_continuation=continuation
        ).log_prob(y)

    grads = jax.vmap(jax.grad(log_prob, argnums=(0, 1, 2, 3)), (0, None, None, None))(
        y, jnp.asarray(LOC), jnp.asarray(SCALE), jnp.asarray(0.0)
    )
    d_dz = -1.0 + jnp.exp(-z)
    np.testing.assert_allclose(grads[0], d_dz / SCALE, rtol=1e-12)
    np.testing.assert_allclose(grads[1], -d_dz / SCALE, rtol=1e-12)
    np.testing.assert_allclose(grads[2], (-1.0 - z * d_dz) / SCALE, rtol=1e-12)
    # d/dxi at xi = 0: u = z - xi z**2 / 2 + O(xi**2).
    d_dxi = -z + (1.0 - jnp.exp(-z)) * z**2 / 2.0
    np.testing.assert_allclose(grads[3], d_dxi, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", METHODS)
def test_values_and_gradients_are_continuous_in_the_shape_across_zero(name):
    evaluate = _all_methods(False)
    for y in (-2.0, 0.3, 5.0):
        values, grads = [], []
        for xi in (-1e-7, 0.0, 1e-7):
            args = tuple(jnp.asarray(v) for v in (y, LOC, SCALE, xi))
            values.append(evaluate(name, *args))
            grads.append(
                jnp.stack(jax.grad(lambda *a: evaluate(name, *a), (0, 1, 2, 3))(*args))
            )
        np.testing.assert_allclose(values[0], values[1], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(values[2], values[1], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grads[0], grads[1], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(grads[2], grads[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("xi", [-0.4, -0.1, -1e-4, -1e-8, 1e-8, 1e-4, 0.1, 0.4])
def test_shape_gradient_matches_high_precision_finite_differences(xi):
    step = Decimal("1e-15")
    for q in (0.01, 0.4, 0.95):
        y = float(_scipy(xi).ppf(q))
        with localcontext() as ctx:
            ctx.prec = 50
            xi_d = Decimal(xi)
            y_d = Decimal(y)
            reference = (
                _reference_log_pdf(y_d, xi_d + step)
                - _reference_log_pdf(y_d, xi_d - step)
            ) / (2 * step)
        grad = jax.grad(lambda x, y=y: _gev(x).log_prob(jnp.asarray(y)))(
            jnp.asarray(xi)
        )
        np.testing.assert_allclose(grad, float(reference), rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# The support continuation
# ---------------------------------------------------------------------------


def _upper_t(xi):
    return gpd._CONTINUATION_UPPER_T * (1.0 + xi)


def _exact_region(z, xi):
    """The points where the continued log density equals the exact one."""
    t = 1.0 + xi * z
    u = np.where(xi == 0.0, z, np.log(np.abs(t)) / (xi if xi != 0.0 else 1.0))
    lower_ok = u >= gpd._CONTINUATION_LOWER_U
    upper_ok = (xi >= 0.0) | (t >= _upper_t(xi))
    return (t > 0.0) & lower_ok & upper_ok


@pytest.mark.parametrize("xi", XI_GRID)
def test_continuation_equals_the_exact_log_density_inside_its_region(xi):
    z = np.linspace(-12.0, 40.0, 2001)
    keep = _exact_region(z, xi)
    y = jnp.asarray(LOC + SCALE * z[keep])
    np.testing.assert_allclose(
        _gev(xi, continuation=True).log_prob(y), _gev(xi).log_prob(y), rtol=1e-14
    )
    # The probabilities of the agreement tests above are all in the region.
    assert np.all(_exact_region((_points(xi) - LOC) / SCALE, xi)[1:-1])


@pytest.mark.parametrize("xi", [-0.5, -0.3, -1e-8, 0.0, 1e-8, 0.3, 0.5])
def test_continuation_is_finite_with_finite_gradients_everywhere(xi):
    y = jnp.linspace(-1e3, 1e3, 4001)

    def log_prob(y, loc, scale, xi):
        return _gev(xi, continuation=True, loc=loc, scale=scale).log_prob(y)

    values = log_prob(y, LOC, SCALE, xi)
    grads = jax.vmap(jax.grad(log_prob, argnums=(0, 1, 2, 3)), (0, None, None, None))(
        y, jnp.asarray(LOC), jnp.asarray(SCALE), jnp.asarray(xi)
    )
    assert jnp.all(jnp.isfinite(values))
    for grad in grads:
        assert jnp.all(jnp.isfinite(grad))


def _junctions(xi):
    """The standardised points where the continuation starts."""
    lower = gpd._CONTINUATION_LOWER_U * float(
        np.expm1(xi * gpd._CONTINUATION_LOWER_U) / (xi * gpd._CONTINUATION_LOWER_U)
        if xi != 0.0
        else 1.0
    )
    junctions = [lower]
    if xi < 0.0:
        junctions.append((_upper_t(xi) - 1.0) / xi)
    return junctions


@pytest.mark.parametrize("xi", [-0.5, -0.3, -0.1, -1e-3, 0.0, 1e-3, 0.1, 0.3, 0.5])
def test_continuation_is_continuously_differentiable_at_the_junctions(xi):
    def log_prob(y, loc, scale, xi):
        return _gev(xi, continuation=True, loc=loc, scale=scale).log_prob(y)

    gradient = jax.grad(log_prob, argnums=(0, 1, 2, 3))
    params = (jnp.asarray(LOC), jnp.asarray(SCALE), jnp.asarray(xi))
    for junction in _junctions(xi):
        y_j = LOC + SCALE * junction
        step = 1e-9 * max(1.0, abs(y_j))
        left, right = jnp.asarray(y_j - step), jnp.asarray(y_j + step)
        value_left, value_right = log_prob(left, *params), log_prob(right, *params)
        grad_left = jnp.stack(gradient(left, *params))
        grad_right = jnp.stack(gradient(right, *params))
        slope = jnp.abs(grad_left[0])
        assert jnp.abs(value_left - value_right) <= 3.0 * step * slope + 1e-10
        np.testing.assert_allclose(grad_left, grad_right, rtol=1e-5, atol=1e-6)


def test_continuation_pushes_the_location_back_into_the_support():
    def gradients(y, xi):
        def log_prob(y, loc, scale, xi):
            return _gev(xi, continuation=True, loc=loc, scale=scale).log_prob(y)

        return jax.grad(log_prob, argnums=(0, 1, 2, 3))(
            jnp.asarray(y), jnp.asarray(LOC), jnp.asarray(SCALE), jnp.asarray(xi)
        )

    # Heavy tail: the observation is below the lower end point.
    lower_end = LOC - SCALE / 0.3
    for y in (lower_end - 0.1, lower_end - 10.0, lower_end - 1e3):
        d_y, d_loc, d_scale, d_xi = gradients(y, 0.3)
        assert d_y > 0.0 and d_loc < 0.0 and d_scale > 0.0 and d_xi < 0.0
    # Bounded tail: the observation is above the upper end point.
    upper_end = LOC + SCALE / 0.3
    for y in (upper_end + 0.1, upper_end + 10.0, upper_end + 1e3):
        d_y, d_loc, d_scale, d_xi = gradients(y, -0.3)
        assert d_y < 0.0 and d_loc > 0.0 and d_scale > 0.0 and d_xi > 0.0
    # Gumbel: an observation far below the location.
    d_y, d_loc, d_scale, _ = gradients(LOC - 50.0 * SCALE, 0.0)
    assert d_y > 0.0 and d_loc < 0.0 and d_scale > 0.0


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.3])
def test_continuation_decreases_monotonically_away_from_the_support(xi):
    dist = _gev(xi, continuation=True)
    below = LOC + SCALE * jnp.linspace(-1e3, -20.0, 200)
    assert jnp.all(jnp.diff(dist.log_prob(below)) > 0.0)
    if xi < 0.0:
        above = LOC + SCALE * jnp.linspace(1.0 / abs(xi), 1e3, 200)
        assert jnp.all(jnp.diff(dist.log_prob(above)) < 0.0)


@pytest.mark.parametrize("xi", [-0.5, -0.4, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
def test_continuation_adds_a_negligible_mass(xi):
    dist = _gev(xi, continuation=True)
    log_prob = jax.jit(lambda z: dist.log_prob(LOC + SCALE * z))

    def density(z):
        return SCALE * float(jnp.exp(log_prob(jnp.asarray(z))))

    # Integrate the pseudo-density in pieces between the junctions and the end
    # point; above z = 400 there is no continuation, so the exact tail is added.
    upper = 1.0 / -xi + 1.0 if xi < 0.0 else 400.0
    edges = sorted({-60.0, upper, *_junctions(xi), *([1.0 / -xi] if xi < 0 else [])})
    mass = sum(
        integrate.quad(density, start, stop, epsabs=1e-13, epsrel=1e-12, limit=500)[0]
        for start, stop in itertools.pairwise(edges)
    )
    if xi >= 0.0:
        mass += float(dist.sf(jnp.asarray(LOC + SCALE * upper)))
    assert abs(mass - 1.0) <= CONTINUATION_MASS_BOUND


def test_continuation_does_not_change_cdf_quantiles_or_samples():
    y = jnp.linspace(-20.0, 20.0, 101)
    q = jnp.linspace(0.0, 1.0, 101)
    for xi in (-0.3, 0.0, 0.3):
        exact, continued = _gev(xi), _gev(xi, continuation=True)
        np.testing.assert_array_equal(exact.cdf(y), continued.cdf(y))
        np.testing.assert_array_equal(exact.sf(y), continued.sf(y))
        np.testing.assert_array_equal(exact.icdf(q), continued.icdf(q))
        np.testing.assert_array_equal(
            exact.sample(jr.key(1), (10,)), continued.sample(jr.key(1), (10,))
        )


def test_distribution_is_a_pytree_that_keeps_the_continuation_setting():
    dist = _gev(0.3, continuation=True)
    y = jnp.array(LOC - SCALE / 0.3 - 1.0)
    jitted = jax.jit(lambda d, y: d.log_prob(y))(dist, y)
    assert jnp.isfinite(jitted)
    np.testing.assert_allclose(jitted, dist.log_prob(y))
    leaves, treedef = jax.tree_util.tree_flatten(dist)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.support_continuation is True
    assert len(leaves) == 3


def test_vmap_over_parameters():
    xi = jnp.linspace(-0.4, 0.4, 9)
    values = jax.vmap(lambda x: _gev(x, continuation=True).log_prob(jnp.array(2.0)))(xi)
    expected = jnp.stack([_gev(x).log_prob(jnp.array(2.0)) for x in xi])
    np.testing.assert_allclose(values, expected, rtol=1e-12)
