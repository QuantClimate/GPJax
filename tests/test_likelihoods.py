# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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

from collections.abc import Callable

import equinox as eqx
import gpjax as gpx
from gpjax.distributions import (
    GaussianDistribution,
    GeneralisedExtremeValue,
    gev_return_level,
)
from gpjax.likelihoods import (
    GEV,
    Bernoulli,
    Gaussian,
    Poisson,
    StudentT,
    inv_probit,
)
from gpjax.parameters import (
    SigmoidBounded,
    val,
)
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
)
import lineax as lx
import numpy as np
import numpyro.distributions as npd
import optax as ox
import paramax
import pytest
from scipy import stats

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.key(123)


def _compute_latent_dist(
    n: int,
) -> tuple[npd.MultivariateNormal, Float[Array, " N"], Float[Array, "N N"]]:
    k1, k2 = jr.split(_initialise_key)
    latent_mean = jr.uniform(k1, shape=(n,))
    latent_sqrt = jr.uniform(k2, shape=(n, n))
    latent_cov = jnp.matmul(latent_sqrt, latent_sqrt.T)
    latent_dist = npd.MultivariateNormal(loc=latent_mean, covariance_matrix=latent_cov)
    return latent_dist, latent_mean, latent_cov


@pytest.mark.parametrize("n", [1, 2, 10, 3])
@pytest.mark.parametrize("obs_stddev", [0.1, 0.5, 1.0, 0.0])
def test_gaussian_likelihood(n: int, obs_stddev: float):
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    likelihood = Gaussian(obs_stddev=obs_stddev)

    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(x), npd.Normal)

    # Construct latent function distribution.
    latent_dist, latent_mean, latent_cov = _compute_latent_dist(n)
    pred_dist = likelihood(latent_dist)
    assert isinstance(pred_dist, GaussianDistribution)

    # Check predictive mean and variance.
    assert (pred_dist.mean == latent_mean).all()
    noise_matrix = jnp.eye(n) * likelihood.obs_stddev.unwrap() ** 2
    assert np.allclose(pred_dist.covariance_matrix, latent_cov + noise_matrix)


@pytest.mark.parametrize("n", [1, 2, 10])
def test_bernoulli_likelihood(n: int):
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    likelihood = Bernoulli()

    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(x), npd.BernoulliProbs)

    # Construct latent function distribution.
    latent_dist, latent_mean, latent_cov = _compute_latent_dist(n)
    pred_dist = likelihood(latent_dist)
    assert isinstance(pred_dist, npd.BernoulliProbs)

    # Check predictive mean and variance.
    p = inv_probit(latent_mean / jnp.sqrt(1.0 + jnp.diagonal(latent_cov)))
    assert (pred_dist.mean == p).all()
    assert (pred_dist.variance == p * (1.0 - p)).all()


@pytest.mark.parametrize("n", [1, 2, 10])
def test_poisson_likelihood(n: int):
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    likelihood = Poisson()

    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(x), npd.Poisson)

    # Construct latent function distribution.
    latent_dist, latent_mean, _latent_cov = _compute_latent_dist(n)
    pred_dist = likelihood(latent_dist)
    assert isinstance(pred_dist, npd.Poisson)

    # Check predictive mean and variance.
    rate = jnp.exp(latent_mean)
    assert (pred_dist.mean == rate).all()


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("degrees_of_freedom", [2.5, 4.0, 30.0])
def test_studentt_likelihood(n: int, degrees_of_freedom: float):
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    likelihood = StudentT(degrees_of_freedom=degrees_of_freedom)

    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(x), npd.StudentT)
    assert jnp.allclose(likelihood.degrees_of_freedom.unwrap(), degrees_of_freedom)
    assert jnp.allclose(likelihood.scale.unwrap(), 1.0)

    # Construct latent function distribution.
    latent_dist, latent_mean, _latent_cov = _compute_latent_dist(n)
    pred_dist = likelihood(latent_dist)
    assert isinstance(pred_dist, npd.StudentT)

    # Check predictive mean matches the latent mean, pushed through the link.
    assert (pred_dist.mean == latent_mean).all()


def test_studentt_likelihood_scale():
    likelihood = StudentT(degrees_of_freedom=3.0, scale=0.5)
    assert jnp.allclose(likelihood.degrees_of_freedom.unwrap(), 3.0)
    assert jnp.allclose(likelihood.scale.unwrap(), 0.5)

    f = jnp.zeros((5,))
    dist = likelihood.link_function(f)
    assert jnp.allclose(dist.scale, 0.5)


class TestMultiOutputGaussian:
    def test_init_scalar_noise(self):
        """Scalar obs_stddev broadcasts to [P] vector."""
        from gpjax.likelihoods import MultiOutputGaussian

        lik = MultiOutputGaussian(num_outputs=3, obs_stddev=0.5)
        assert lik.obs_stddev.unwrap().shape == (3,)
        assert jnp.allclose(lik.obs_stddev.unwrap(), jnp.full(3, 0.5))

    def test_init_vector_noise(self):
        """Vector obs_stddev is accepted directly."""
        from gpjax.likelihoods import MultiOutputGaussian

        noise = jnp.array([0.1, 0.2, 0.3])
        lik = MultiOutputGaussian(num_outputs=3, obs_stddev=noise)
        assert jnp.allclose(lik.obs_stddev.unwrap(), noise)

    def test_noise_vector_shape(self):
        """noise_vector() returns [NP] in output-major order."""
        from gpjax.likelihoods import MultiOutputGaussian

        lik = MultiOutputGaussian(num_outputs=3, obs_stddev=1.0)
        nv = lik.noise_vector(10)
        assert nv.shape == (30,)

    def test_noise_vector_ordering(self):
        """noise_vector() repeats per-output variance in output-major order."""
        from gpjax.likelihoods import MultiOutputGaussian

        noise = jnp.array([1.0, 2.0])
        lik = MultiOutputGaussian(num_outputs=2, obs_stddev=noise)
        nv = lik.noise_vector(3)
        # Output-major: [σ₁² repeated N, σ₂² repeated N]
        expected = jnp.array([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
        assert jnp.allclose(nv, expected)

    def test_is_gaussian_subclass(self):
        """MultiOutputGaussian is a Gaussian subclass for posterior dispatch."""
        from gpjax.likelihoods import MultiOutputGaussian

        lik = MultiOutputGaussian(num_outputs=2)
        assert isinstance(lik, Gaussian)


def _gaussian(mean, cov):
    return GaussianDistribution(jnp.asarray(mean), lx.MatrixLinearOperator(cov))


def _correlated(mean, sd, correlation):
    n = len(mean)
    corr = jnp.full((n, n), correlation) + (1.0 - correlation) * jnp.eye(n)
    cov = corr * jnp.outer(jnp.asarray(sd), jnp.asarray(sd))
    return _gaussian(jnp.asarray(mean), cov)


def _brute_force_expected_log_likelihood(likelihood, y, mean, variance):
    sd = np.sqrt(variance)
    f = mean + sd * np.linspace(-12.0, 12.0, 200001)
    log_prob = np.asarray(likelihood.link_function(jnp.asarray(f)).log_prob(y))
    return float(np.trapezoid(stats.norm.pdf(f, mean, sd) * log_prob, f))


def _gumbel_moment_start(y):
    sigma0 = float(jnp.std(y)) * np.sqrt(6.0) / np.pi
    return float(jnp.mean(y)) - 0.5772 * sigma0, sigma0


def _fit_gev_variational(x, y, inducing_inputs, kernel, num_iters):
    mu0, sigma0 = _gumbel_moment_start(y)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Constant(mu0), kernel=kernel)
    q = gpx.variational_families.WhitenedVariationalGaussian(
        model=prior * GEV(scale=sigma0, shape=0.0), inducing_inputs=inducing_inputs
    )
    q_with_fixed_inducing_inputs = eqx.tree_at(
        lambda q: q.inducing_inputs, q, paramax.non_trainable(q.inducing_inputs)
    )
    data = gpx.Dataset(X=x, y=y)
    fitted, history = gpx.fit(
        model=q_with_fixed_inducing_inputs,
        objective=lambda q, d: -gpx.objectives.elbo(q, d),
        train_data=data,
        optim=ox.adam(0.05),
        num_iters=num_iters,
        verbose=False,
    )
    assert jnp.all(jnp.isfinite(history))
    return fitted, data


def test_gev_link_function_uses_the_continuation_and_predict_is_exact():
    likelihood = GEV(scale=2.0, shape=-0.2)
    f = jnp.linspace(-1.0, 1.0, 3)
    link = likelihood.link_function(f)
    assert link.support_continuation is True
    np.testing.assert_allclose(link.loc, f)
    np.testing.assert_allclose(link.scale, 2.0)
    np.testing.assert_allclose(link.concentration, -0.2)

    latent_dist, latent_mean, _ = _compute_latent_dist(3)
    predictive = likelihood(latent_dist)
    assert isinstance(predictive, GeneralisedExtremeValue)
    assert predictive.support_continuation is False
    np.testing.assert_allclose(predictive.loc, latent_mean)


def test_gev_with_zero_shape_is_the_gumbel_likelihood():
    y = jnp.linspace(-3.0, 6.0, 7)
    np.testing.assert_allclose(
        GEV(shape=0.0).link_function(jnp.zeros(7)).log_prob(y),
        npd.Gumbel(0.0, 1.0).log_prob(y),
        rtol=1e-12,
    )


def test_gev_keeps_the_parameters_of_the_user():
    wide = GEV(shape=SigmoidBounded(0.7, low=-1.0, high=1.0))
    fixed = GEV(
        scale=paramax.non_trainable(jnp.array(3.0)),
        shape=paramax.non_trainable(jnp.array(0.2)),
    )
    np.testing.assert_allclose(val(wide.shape), 0.7)
    np.testing.assert_allclose(val(fixed.shape), 0.2)
    np.testing.assert_allclose(val(fixed.scale), 3.0)


def test_gev_accepts_traced_parameters_under_jit():
    def scale_and_shape(scale, shape):
        likelihood = GEV(scale=scale, shape=shape)
        return val(likelihood.scale), val(likelihood.shape)

    np.testing.assert_allclose(
        jax.jit(scale_and_shape)(jnp.array(3.0), jnp.array(0.2)), (3.0, 0.2)
    )


def test_gev_rejects_parameters_outside_their_range():
    with pytest.raises(ValueError, match="shape"):
        GEV(shape=0.5)
    with pytest.raises(ValueError, match="shape"):
        GEV(shape=-0.7)
    with pytest.raises(ValueError, match="scale"):
        GEV(scale=0.0)


def test_prior_times_gev_is_a_non_conjugate_model():
    prior = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(), mean_function=gpx.mean_functions.Constant()
    )
    assert isinstance(prior * GEV(), gpx.gps.NonConjugateModel)
    assert isinstance(GEV() * prior, gpx.gps.NonConjugateModel)


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.3])
def test_gev_expected_log_likelihood_matches_a_brute_force_integral(shape):
    likelihood = GEV(scale=1.5, shape=shape)
    y = jnp.array([[0.5], [2.0], [4.0], [-0.5]])
    mean = jnp.array([[0.0], [1.0], [2.5], [0.3]])
    variance = jnp.array([[0.05], [0.2], [0.4], [0.1]])
    brute_force = [
        _brute_force_expected_log_likelihood(
            likelihood, float(y[i, 0]), float(mean[i, 0]), float(variance[i, 0])
        )
        for i in range(4)
    ]
    np.testing.assert_allclose(
        likelihood.expected_log_likelihood(y, mean, variance), brute_force, rtol=1e-6
    )


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.3])
def test_gev_expected_log_likelihood_is_finite_with_nodes_outside_the_support(shape):
    y = jnp.array([[0.0], [3.0], [-2.0]])
    mean = jnp.zeros((3, 1))
    ten_scales_wide = jnp.full((3, 1), 100.0)

    def objective(mean, variance, likelihood):
        return jnp.sum(likelihood.expected_log_likelihood(y, mean, variance))

    likelihood = GEV(scale=1.0, shape=shape)
    value, (d_mean, d_variance, d_likelihood) = jax.value_and_grad(
        objective, (0, 1, 2)
    )(mean, ten_scales_wide, likelihood)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(d_mean)) and jnp.all(jnp.isfinite(d_variance))
    assert jnp.isfinite(d_likelihood.scale._unconstrained)
    assert jnp.isfinite(d_likelihood.shape._unconstrained)
    assert jnp.all(d_variance < 0.0)


def test_gev_return_level_samples_match_the_closed_form():
    scale, shape, num_samples = 2.0, -0.15, 20000
    likelihood = GEV(scale=scale, shape=shape)
    mean, sd = jnp.array([10.0, 20.0]), jnp.array([0.5, 1.5])
    latent = _correlated(mean, sd, 0.5)
    periods = jnp.array([10.0, 100.0])
    levels = likelihood.return_level_samples(
        latent, periods, key=jr.key(1), num_samples=num_samples
    )
    level_above_location = gev_return_level(periods, 0.0, scale, shape)
    expected_mean = mean[None, :] + level_above_location[:, None]
    monte_carlo_error = 4.0 * sd / np.sqrt(num_samples)

    assert levels.shape == (num_samples, 2, 2)
    assert jnp.all(jnp.abs(levels.mean(0) - expected_mean) < monte_carlo_error)
    np.testing.assert_allclose(levels.std(0), jnp.stack([sd, sd]), rtol=0.03)
    np.testing.assert_allclose(
        levels[:, 1] - levels[:, 0],
        level_above_location[1] - level_above_location[0],
        rtol=1e-10,
    )
    one_period = likelihood.return_level_samples(
        latent, 100.0, key=jr.key(1), num_samples=5
    )
    assert one_period.shape == (5, 2)


def test_gev_return_period_samples_invert_the_return_level():
    likelihood = GEV(scale=3.0, shape=0.2)
    mean = jnp.array([5.0, 8.0])
    almost_certain_latent = _correlated(mean, [1e-6, 1e-6], 0.0)
    periods = jnp.array([20.0, 500.0])
    samples = likelihood.return_period_samples(
        almost_certain_latent,
        gev_return_level(periods, mean, 3.0, 0.2),
        key=jr.key(3),
        num_samples=100,
    )
    np.testing.assert_allclose(samples, jnp.broadcast_to(periods, (100, 2)), rtol=1e-4)


def test_gev_return_period_samples_keep_infinity():
    scale, shape, num_samples = 1.0, -0.25, 20000
    mean, sd, level = 0.0, 1.0, 4.5
    samples = GEV(scale=scale, shape=shape).return_period_samples(
        _correlated([mean], [sd], 0.0), level, key=jr.key(4), num_samples=num_samples
    )
    share_impossible = float(jnp.mean(jnp.isposinf(samples)))
    probability_end_point_below_level = stats.norm.cdf(
        level - scale / abs(shape), mean, sd
    )
    monte_carlo_error = 4.0 * np.sqrt(
        probability_end_point_below_level
        * (1 - probability_end_point_below_level)
        / num_samples
    )
    assert not jnp.any(jnp.isnan(samples))
    assert abs(share_impossible - probability_end_point_below_level) < monte_carlo_error
    assert jnp.all((1.0 / samples)[jnp.isposinf(samples)] == 0.0)


def test_gev_probability_ratio_from_paired_samples():
    likelihood = GEV(scale=1.0, shape=0.1)
    locations = jnp.array([0.0, 0.5])
    periods = likelihood.return_period_samples(
        _correlated(locations, [0.3, 0.3], 0.95), 4.0, key=jr.key(5), num_samples=4000
    )
    ratio = periods[:, 0] / periods[:, 1]
    at_the_latent_mean = gpx.distributions.gev_return_period(4.0, locations, 1.0, 0.1)
    assert float(jnp.mean(ratio > 1.0)) > 0.95
    np.testing.assert_allclose(
        jnp.median(ratio), at_the_latent_mean[0] / at_the_latent_mean[1], rtol=0.05
    )


@pytest.mark.parametrize("shape", [-0.2, 0.2])
def test_gev_variational_recovery(shape):
    num_data = 600
    key_x, key_y = jr.split(jr.key(int(1000 * shape) + num_data))
    x = jnp.sort(jr.uniform(key_x, (num_data, 1)), axis=0)
    location = 3.0 * jnp.sin(2.0 * jnp.pi * x)
    y = GeneralisedExtremeValue(location, 1.0, shape).sample(key_y)
    fitted, data = _fit_gev_variational(
        x,
        y,
        jnp.linspace(0.0, 1.0, 20)[:, None],
        gpx.kernels.RBF(lengthscale=0.3),
        num_iters=500,
    )
    likelihood = fitted.model.likelihood
    latent = fitted.condition(data)(x, covariance="diagonal").mean
    assert abs(float(val(likelihood.shape)) - shape) < 0.08
    assert abs(float(val(likelihood.scale)) - 1.0) < 0.1
    assert float(jnp.sqrt(jnp.mean((latent - location[:, 0]) ** 2))) < 0.2


@pytest.mark.slow
@pytest.mark.parametrize("shape", [-0.2, 0.2])
def test_gev_map_recovery_with_repeated_sites(shape):
    num_sites, num_years = 10, 30
    sites = jnp.linspace(0.0, 1.0, num_sites)
    x = jnp.repeat(sites, num_years)[:, None]
    location = 3.0 * jnp.sin(2.0 * jnp.pi * x)
    y = GeneralisedExtremeValue(location, 1.0, shape).sample(
        jr.key(int(1000 * shape) + num_sites)
    )
    data = gpx.Dataset(X=x, y=y)
    mu0, sigma0 = _gumbel_moment_start(y)
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(mu0),
        kernel=gpx.kernels.RBF(lengthscale=0.3),
    )
    fitted, history = gpx.fit_scipy(
        model=prior * GEV(scale=sigma0, shape=0.0),
        objective=lambda m, d: -gpx.objectives.log_posterior_density(m, d),
        train_data=data,
        max_iters=1000,
        verbose=False,
    )
    latent = fitted.condition(data)(sites[:, None], covariance="diagonal").mean
    location_at_sites = 3.0 * jnp.sin(2.0 * jnp.pi * sites)
    assert history[-1] < history[0]
    assert abs(float(val(fitted.likelihood.shape)) - shape) < 0.15
    assert abs(float(val(fitted.likelihood.scale)) - 1.0) < 0.12
    assert float(jnp.sqrt(jnp.mean((latent - location_at_sites) ** 2))) < 0.3


def test_gev_covariate_in_the_location_shifts_the_return_level():
    trend = 2.0
    sites, gmst = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, 15), jnp.linspace(0.0, 1.0, 20), indexing="ij"
    )
    x = jnp.stack([sites.ravel(), gmst.ravel()], axis=1)
    location = 10.0 + jnp.sin(2.0 * jnp.pi * x[:, :1]) + trend * x[:, 1:]
    y = GeneralisedExtremeValue(location, 1.0, 0.1).sample(jr.key(5))
    inducing_sites, inducing_gmst = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, 8), jnp.linspace(0.0, 1.0, 3), indexing="ij"
    )
    fitted, data = _fit_gev_variational(
        x,
        y,
        jnp.stack([inducing_sites.ravel(), inducing_gmst.ravel()], axis=1),
        gpx.kernels.RBF(active_dims=[0], lengthscale=0.3)
        + gpx.kernels.Linear(active_dims=[1]),
        num_iters=500,
    )
    likelihood = fitted.model.likelihood

    one_site_cool_then_warm = jnp.array([[0.25, 0.0], [0.25, 1.0]])
    paired_latent = fitted.condition(data)(one_site_cool_then_warm)
    levels = likelihood.return_level_samples(
        paired_latent, 100.0, key=jr.key(0), num_samples=4000
    )
    shift = levels[:, 1] - levels[:, 0]
    assert abs(float(jnp.mean(shift)) - trend) < 0.5
    assert float(jnp.mean(shift > 0.0)) > 0.99

    cool_100_year_level = float(jnp.median(levels[:, 0]))
    periods = likelihood.return_period_samples(
        paired_latent, cool_100_year_level, key=jr.key(1), num_samples=4000
    )
    probability_ratio = periods[:, 0] / periods[:, 1]
    assert float(jnp.mean(probability_ratio > 1.0)) > 0.99
    assert float(jnp.median(periods[:, 0])) == pytest.approx(100.0, rel=0.3)
