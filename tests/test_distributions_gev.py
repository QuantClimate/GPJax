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
convention ``xi`` (``xi > 0`` is the heavy-tailed Frechet type).
"""

import functools
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
XI_AWAY_FROM_GUMBEL = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
NEAR_GUMBEL = [-1e-4, -1e-8, 0.0, 1e-8, 1e-4]
LOC, SCALE = 1.5, 2.0
PROBS = np.array([1e-8, 1e-3, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-6])
SMALL_PROBS = np.array([1e-15, 1e-12, 1e-6])
RETURN_PERIODS = np.array([1.5, 2.0, 10.0, 100.0, 1e3, 1e6, 1e12])
DOCUMENTED_MASS_ERROR = 3e-5
METHODS = ["log_prob", "cdf", "sf", "icdf", "isf", "return_level", "return_period"]


def _gev(xi, *, continuation=False, loc=LOC, scale=SCALE):
    return GeneralisedExtremeValue(
        loc=jnp.asarray(loc),
        scale=jnp.asarray(scale),
        concentration=jnp.asarray(xi),
        support_continuation=continuation,
    )


def _scipy(xi):
    return stats.genextreme(c=-xi, loc=LOC, scale=SCALE)


def _continued_log_prob(y, loc, scale, xi):
    return _gev(xi, continuation=True, loc=loc, scale=scale).log_prob(y)


def _evaluate(method, continuation, unconstrained, loc, scale, xi):
    dist = GeneralisedExtremeValue(loc, scale, xi, support_continuation=continuation)
    if method in ("icdf", "isf"):
        return getattr(dist, method)(jax.nn.sigmoid(unconstrained))
    if method == "return_level":
        return dist.return_level(2.0 + jnp.exp(unconstrained))
    return getattr(dist, method)(unconstrained)


def _upper_junction_t(xi):
    return gpd._CONTINUATION_UPPER_T * (1.0 + xi)


def _is_between_the_junctions(z, xi):
    t = 1.0 + xi * z
    u = z if xi == 0.0 else np.log(np.abs(t)) / xi
    above_lower_junction = u >= gpd._CONTINUATION_LOWER_U
    below_upper_junction = (xi >= 0.0) | (t >= _upper_junction_t(xi))
    return (t > 0.0) & above_lower_junction & below_upper_junction


def _junctions(xi):
    lower_u = gpd._CONTINUATION_LOWER_U
    lower = lower_u if xi == 0.0 else float(np.expm1(xi * lower_u) / xi)
    return [lower, (_upper_junction_t(xi) - 1.0) / xi] if xi < 0.0 else [lower]


@pytest.mark.parametrize("xi", XI_GRID)
def test_log_prob_cdf_and_sf_match_scipy(xi):
    y = _scipy(xi).ppf(PROBS)
    dist = _gev(xi)
    np.testing.assert_allclose(
        dist.log_prob(jnp.asarray(y)), _scipy(xi).logpdf(y), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(dist.cdf(jnp.asarray(y)), _scipy(xi).cdf(y), rtol=1e-9)
    np.testing.assert_allclose(dist.sf(jnp.asarray(y)), _scipy(xi).sf(y), rtol=1e-9)


@pytest.mark.parametrize("xi", XI_GRID)
def test_sf_matches_scipy_for_small_probabilities(xi):
    y = _scipy(xi).isf(SMALL_PROBS)
    np.testing.assert_allclose(_gev(xi).sf(jnp.asarray(y)), _scipy(xi).sf(y), rtol=1e-8)


@pytest.mark.parametrize("xi", XI_GRID)
def test_icdf_and_isf_match_scipy(xi):
    dist = _gev(xi)
    np.testing.assert_allclose(
        dist.icdf(jnp.asarray(PROBS)), _scipy(xi).ppf(PROBS), rtol=1e-9, atol=1e-12
    )
    exceedance = np.concatenate([SMALL_PROBS, [0.01, 0.5, 0.9]])
    np.testing.assert_allclose(
        dist.isf(jnp.asarray(exceedance)),
        _scipy(xi).isf(exceedance),
        rtol=1e-9,
        atol=1e-12,
    )


def test_quantiles_at_zero_and_one_are_the_end_points():
    q = jnp.array([0.0, 1.0])
    np.testing.assert_allclose(_gev(0.25).icdf(q), [LOC - SCALE / 0.25, jnp.inf])
    np.testing.assert_allclose(_gev(-0.25).icdf(q), [-jnp.inf, LOC + SCALE / 0.25])
    np.testing.assert_allclose(_gev(0.0).icdf(q), [-jnp.inf, jnp.inf])
    assert jnp.isnan(_gev(0.1).icdf(jnp.array(1.5)))


def test_cdf_and_sf_are_exact_outside_the_support():
    heavy = _gev(0.3)
    lower_end = LOC - SCALE / 0.3
    below = jnp.array([lower_end - 1e3, lower_end - 1.0, lower_end - 1e-9, lower_end])
    assert jnp.all(heavy.cdf(below) == 0.0)
    assert jnp.all(heavy.sf(below) == 1.0)

    bounded = _gev(-0.3)
    upper_end = LOC + SCALE / 0.3
    above = jnp.array([upper_end, upper_end + 1e-9, upper_end + 1.0, upper_end + 1e3])
    assert jnp.all(bounded.cdf(above) == 1.0)
    assert jnp.all(bounded.sf(above) == 0.0)


def test_cdf_has_no_overflow_near_the_lower_end_point():
    heavy = _gev(0.5)
    lower_end = LOC - SCALE / 0.5
    just_inside = jnp.array([lower_end + 1e-12, lower_end + 1e-3])
    assert jnp.all(heavy.cdf(just_inside) == 0.0)
    assert jnp.all(heavy.sf(just_inside) == 1.0)
    assert jnp.all(jnp.isfinite(jax.vmap(jax.grad(heavy.cdf))(just_inside)))


def test_return_period_is_infinite_where_the_exceedance_probability_is_zero():
    bounded = _gev(-0.3)
    upper_end = LOC + SCALE / 0.3
    impossible_levels = jnp.array([upper_end, upper_end + 5.0])
    assert jnp.all(jnp.isposinf(bounded.return_period(impossible_levels)))
    gradients = jax.vmap(jax.grad(bounded.return_period))(impossible_levels)
    assert jnp.all(jnp.isfinite(gradients))
    assert jnp.isfinite(bounded.return_period(jnp.array(upper_end - 0.5)))


def test_log_prob_is_minus_infinity_outside_the_support_without_continuation():
    below_lower_end = jnp.array(LOC - SCALE / 0.3 - 1.0)
    above_upper_end = jnp.array(LOC + SCALE / 0.3 + 1.0)
    assert jnp.isneginf(_gev(0.3).log_prob(below_lower_end))
    assert jnp.isneginf(_gev(-0.3).log_prob(above_upper_end))
    assert not jnp.isnan(jax.grad(_gev(0.3).log_prob)(below_lower_end))


@pytest.mark.parametrize("xi", XI_AWAY_FROM_GUMBEL)
def test_return_level_matches_the_textbook_form(xi):
    exceedance_rate = -np.log1p(-1.0 / RETURN_PERIODS)
    if xi == 0.0:
        textbook = LOC - SCALE * np.log(exceedance_rate)
    else:
        textbook = LOC - SCALE / xi * (1.0 - exceedance_rate ** (-xi))
    levels = _gev(xi).return_level(jnp.asarray(RETURN_PERIODS))
    np.testing.assert_allclose(levels, textbook, rtol=1e-10)


def test_pure_functions_broadcast_over_parameter_draws():
    key_loc, key_scale, key_shape = jr.split(jr.key(0), 3)
    loc = jr.normal(key_loc, (50, 4))
    scale = jnp.exp(0.1 * jr.normal(key_scale, (50, 1)))
    shape = 0.1 * jr.normal(key_shape, (50, 1))
    levels = gev_return_level(100.0, loc, scale, shape)
    assert levels.shape == (50, 4)
    np.testing.assert_allclose(
        levels, GeneralisedExtremeValue(loc, scale, shape).return_level(100.0)
    )
    np.testing.assert_allclose(
        gev_return_period(levels, loc, scale, shape), 100.0, rtol=1e-6
    )


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.3])
def test_samples_pass_a_kolmogorov_smirnov_test(xi):
    samples = _gev(xi).sample(jr.key(7), (4000,))
    assert stats.kstest(np.asarray(samples), _scipy(xi).cdf).pvalue > 1e-3


@pytest.mark.parametrize("xi", [xi for xi in XI_GRID if xi < 0.5])
def test_mean_and_variance_match_scipy(xi):
    mean, variance = _scipy(xi).stats(moments="mv")
    np.testing.assert_allclose(_gev(xi).mean, mean, rtol=1e-8)
    np.testing.assert_allclose(_gev(xi).variance, variance, rtol=1e-7)


def test_mean_and_variance_are_infinite_where_they_do_not_exist():
    assert jnp.isposinf(_gev(1.0).mean)
    assert jnp.isposinf(_gev(0.5).variance)
    assert jnp.isfinite(_gev(0.7).mean)


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("xi", NEAR_GUMBEL)
def test_values_and_gradients_are_finite_near_the_gumbel_limit(
    continuation, method, xi
):
    evaluate = functools.partial(_evaluate, method, continuation)
    for argument in (-3.0, -0.5, 0.0, 0.7, 4.0, 15.0):
        inputs = tuple(jnp.asarray(value) for value in (argument, LOC, SCALE, xi))
        value, gradients = jax.value_and_grad(evaluate, argnums=(0, 1, 2, 3))(*inputs)
        assert jnp.isfinite(value)
        assert all(jnp.isfinite(gradient) for gradient in gradients)


@pytest.mark.parametrize("continuation", [False, True])
def test_log_prob_gradient_at_zero_shape_matches_the_gumbel_derivatives(continuation):
    z = jnp.linspace(-3.0, 6.0, 19)

    def log_prob(y, loc, scale, xi):
        return GeneralisedExtremeValue(
            loc, scale, xi, support_continuation=continuation
        ).log_prob(y)

    d_y, d_loc, d_scale, d_xi = jax.vmap(
        jax.grad(log_prob, argnums=(0, 1, 2, 3)), (0, None, None, None)
    )(LOC + SCALE * z, jnp.asarray(LOC), jnp.asarray(SCALE), jnp.asarray(0.0))

    d_log_prob_d_z = -1.0 + jnp.exp(-z)
    d_reduced_variate_d_xi = -(z**2) / 2.0
    np.testing.assert_allclose(d_y, d_log_prob_d_z / SCALE, rtol=1e-12)
    np.testing.assert_allclose(d_loc, -d_log_prob_d_z / SCALE, rtol=1e-12)
    np.testing.assert_allclose(d_scale, (-1.0 - z * d_log_prob_d_z) / SCALE, rtol=1e-12)
    np.testing.assert_allclose(
        d_xi, -z + d_log_prob_d_z * d_reduced_variate_d_xi, rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("xi", XI_GRID)
def test_continuation_equals_the_exact_log_density_between_the_junctions(xi):
    z = np.linspace(-12.0, 40.0, 2001)
    y = jnp.asarray(LOC + SCALE * z[_is_between_the_junctions(z, xi)])
    np.testing.assert_allclose(
        _gev(xi, continuation=True).log_prob(y), _gev(xi).log_prob(y), rtol=1e-14
    )


@pytest.mark.parametrize("xi", [-0.5, -0.3, -1e-8, 0.0, 1e-8, 0.3, 0.5])
def test_continuation_is_finite_with_finite_gradients_everywhere(xi):
    y = jnp.linspace(-1e3, 1e3, 4001)
    parameters = (jnp.asarray(LOC), jnp.asarray(SCALE), jnp.asarray(xi))
    values = _continued_log_prob(y, *parameters)
    gradients = jax.vmap(
        jax.grad(_continued_log_prob, argnums=(0, 1, 2, 3)), (0, None, None, None)
    )(y, *parameters)
    assert jnp.all(jnp.isfinite(values))
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)


@pytest.mark.parametrize("xi", [-0.5, -0.3, -0.1, -1e-3, 0.0, 1e-3, 0.1, 0.3, 0.5])
def test_continuation_is_continuously_differentiable_at_the_junctions(xi):
    gradient = jax.grad(_continued_log_prob, argnums=(0, 1, 2, 3))
    parameters = (jnp.asarray(LOC), jnp.asarray(SCALE), jnp.asarray(xi))
    for junction in _junctions(xi):
        y_junction = LOC + SCALE * junction
        step = 1e-9 * max(1.0, abs(y_junction))
        left, right = jnp.asarray(y_junction - step), jnp.asarray(y_junction + step)
        gradient_left = jnp.stack(gradient(left, *parameters))
        gradient_right = jnp.stack(gradient(right, *parameters))
        value_jump = jnp.abs(
            _continued_log_prob(left, *parameters)
            - _continued_log_prob(right, *parameters)
        )
        assert value_jump <= 3.0 * step * jnp.abs(gradient_left[0]) + 1e-10
        np.testing.assert_allclose(gradient_left, gradient_right, rtol=1e-5, atol=1e-6)


def test_continuation_gradient_points_into_the_support():
    gradient = jax.grad(_continued_log_prob, argnums=(0, 1, 2, 3))

    lower_end = LOC - SCALE / 0.3
    for distance_below in (0.1, 10.0, 1e3):
        d_y, d_loc, d_scale, d_xi = gradient(
            lower_end - distance_below, LOC, SCALE, 0.3
        )
        assert d_y > 0.0 and d_loc < 0.0 and d_scale > 0.0 and d_xi < 0.0

    upper_end = LOC + SCALE / 0.3
    for distance_above in (0.1, 10.0, 1e3):
        d_y, d_loc, d_scale, d_xi = gradient(
            upper_end + distance_above, LOC, SCALE, -0.3
        )
        assert d_y < 0.0 and d_loc > 0.0 and d_scale > 0.0 and d_xi > 0.0

    d_y, d_loc, d_scale, _ = gradient(LOC - 50.0 * SCALE, LOC, SCALE, 0.0)
    assert d_y > 0.0 and d_loc < 0.0 and d_scale > 0.0


@pytest.mark.parametrize("xi", XI_AWAY_FROM_GUMBEL)
def test_continuation_adds_a_negligible_mass(xi):
    dist = _gev(xi, continuation=True)
    log_prob = jax.jit(lambda z: dist.log_prob(LOC + SCALE * z))

    def standardised_density(z):
        return SCALE * float(jnp.exp(log_prob(jnp.asarray(z))))

    upper_end = [-1.0 / xi] if xi < 0.0 else []
    integration_end = upper_end[0] + 1.0 if upper_end else 400.0
    edges = sorted({-60.0, integration_end, *_junctions(xi), *upper_end})
    mass = sum(
        integrate.quad(
            standardised_density, start, stop, epsabs=1e-13, epsrel=1e-12, limit=500
        )[0]
        for start, stop in itertools.pairwise(edges)
    )
    if not upper_end:
        mass += float(dist.sf(jnp.asarray(LOC + SCALE * integration_end)))
    assert abs(mass - 1.0) <= DOCUMENTED_MASS_ERROR


def test_jit_keeps_the_continuation_setting():
    dist = _gev(0.3, continuation=True)
    below_lower_end = jnp.array(LOC - SCALE / 0.3 - 1.0)
    jitted = jax.jit(lambda dist, y: dist.log_prob(y))(dist, below_lower_end)
    assert jnp.isfinite(jitted)
    np.testing.assert_allclose(jitted, dist.log_prob(below_lower_end))
