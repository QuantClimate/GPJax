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
    PositiveReal,
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


# ---------------------------------------------------------------------------
# GEV likelihood
# ---------------------------------------------------------------------------


def _gaussian(mean, cov):
    return GaussianDistribution(jnp.asarray(mean), lx.MatrixLinearOperator(cov))


def _brute_force_expected_log_likelihood(likelihood, y, mean, variance):
    """E_q[log p(y | f)] by the trapezoid rule on a fine grid of f."""
    sd = np.sqrt(variance)
    f = mean + sd * np.linspace(-12.0, 12.0, 200001)
    weights = stats.norm.pdf(f, mean, sd)
    log_prob = np.asarray(likelihood.link_function(jnp.asarray(f)).log_prob(y))
    return float(np.trapezoid(weights * log_prob, f))


def test_gev_default_parameters():
    likelihood = GEV()
    assert isinstance(likelihood.scale, PositiveReal)
    assert isinstance(likelihood.shape, SigmoidBounded)
    assert (likelihood.shape.low, likelihood.shape.high) == (-0.5, 0.5)
    np.testing.assert_allclose(val(likelihood.scale), 1.0)
    np.testing.assert_allclose(val(likelihood.shape), 0.1)
    assert likelihood.integrator.num_points == 20


@pytest.mark.parametrize("n", [1, 2, 10])
def test_gev_link_function_and_predict(n: int):
    likelihood = GEV(scale=2.0, shape=-0.2)
    f = jnp.linspace(-1.0, 1.0, n)
    link = likelihood.link_function(f)
    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(link, GeneralisedExtremeValue)
    assert link.support_continuation is True
    np.testing.assert_allclose(link.loc, f)
    np.testing.assert_allclose(link.scale, 2.0)
    np.testing.assert_allclose(link.concentration, -0.2)

    latent_dist, latent_mean, latent_cov = _compute_latent_dist(n)
    for dist in (latent_dist, _gaussian(latent_mean, latent_cov)):
        predictive = likelihood(dist)
        assert isinstance(predictive, GeneralisedExtremeValue)
        # The plug-in predictive is exact, at the latent mean.
        assert predictive.support_continuation is False
        np.testing.assert_allclose(predictive.loc, latent_mean)
        assert predictive.batch_shape == (n,)


def test_gev_link_function_keeps_float32():
    f = jnp.zeros(3, dtype=jnp.float32)
    link = GEV(scale=2.0, shape=0.1).link_function(f)
    assert link.scale.dtype == jnp.float32
    assert link.concentration.dtype == jnp.float32
    assert link.log_prob(jnp.ones(3, dtype=jnp.float32)).dtype == jnp.float32


def test_gev_shape_can_be_fixed_zero_or_a_parameter_of_the_user():
    fixed = GEV(shape=paramax.non_trainable(jnp.array(0.2)))
    np.testing.assert_allclose(val(fixed.shape), 0.2)

    zero = GEV(shape=0.0)
    np.testing.assert_allclose(val(zero.shape), 0.0)
    y = jnp.linspace(-3.0, 6.0, 7)
    np.testing.assert_allclose(
        zero.link_function(jnp.zeros(7)).log_prob(y),
        npd.Gumbel(0.0, 1.0).log_prob(y),
        rtol=1e-12,
    )

    wide = GEV(shape=SigmoidBounded(0.7, low=-1.0, high=1.0))
    np.testing.assert_allclose(val(wide.shape), 0.7)


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
    ours = likelihood.expected_log_likelihood(y, mean, variance)
    assert ours.shape == (4,)
    reference = [
        _brute_force_expected_log_likelihood(
            likelihood, float(y[i, 0]), float(mean[i, 0]), float(variance[i, 0])
        )
        for i in range(4)
    ]
    np.testing.assert_allclose(ours, reference, rtol=1e-6)


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.3])
def test_gev_expected_log_likelihood_is_finite_for_a_wide_variational_distribution(
    shape,
):
    # A standard deviation of 10 scales puts Gauss-Hermite nodes far outside the
    # support of the GEV, where the exact log density is -inf.
    y = jnp.array([[0.0], [3.0], [-2.0]])
    mean = jnp.zeros((3, 1))
    variance = jnp.full((3, 1), 100.0)

    def objective(mean, variance, likelihood):
        return jnp.sum(likelihood.expected_log_likelihood(y, mean, variance))

    likelihood = GEV(scale=1.0, shape=shape)
    value = objective(mean, variance, likelihood)
    grad_mean, grad_variance, grad_likelihood = jax.grad(objective, (0, 1, 2))(
        mean, variance, likelihood
    )
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad_mean)) and jnp.all(jnp.isfinite(grad_variance))
    assert jnp.isfinite(grad_likelihood.scale._unconstrained)
    assert jnp.isfinite(grad_likelihood.shape._unconstrained)
    # A larger variational variance reduces the expected log-likelihood.
    assert jnp.all(grad_variance < 0.0)


def test_gev_expected_log_likelihood_under_jit_and_vmap():
    likelihood = GEV(scale=1.2, shape=0.15)
    key_y, key_mean = jr.split(jr.key(3))
    y = jr.normal(key_y, (6, 1))
    means = jr.normal(key_mean, (4, 6, 1))
    variance = jnp.full((6, 1), 0.3)
    direct = jnp.stack(
        [likelihood.expected_log_likelihood(y, m, variance) for m in means]
    )
    jitted = jax.jit(lambda lik, y, m, v: lik.expected_log_likelihood(y, m, v))(
        likelihood, y, means[0], variance
    )
    mapped = jax.vmap(lambda m: likelihood.expected_log_likelihood(y, m, variance))(
        means
    )
    np.testing.assert_allclose(jitted, direct[0], rtol=1e-12)
    np.testing.assert_allclose(mapped, direct, rtol=1e-12)


# ---------------------------------------------------------------------------
# GEV return levels with uncertainty
# ---------------------------------------------------------------------------


def _correlated_dist(mean, sd, correlation):
    n = len(mean)
    corr = jnp.full((n, n), correlation) + (1.0 - correlation) * jnp.eye(n)
    cov = corr * jnp.outer(jnp.asarray(sd), jnp.asarray(sd))
    return _gaussian(jnp.asarray(mean), cov)


def test_gev_return_level_samples_have_the_documented_shape():
    likelihood = GEV(scale=2.0, shape=0.1)
    dist = _correlated_dist([0.0, 1.0, 2.0, 3.0], [0.5] * 4, 0.3)
    key = jr.key(0)
    single = likelihood.return_level_samples(dist, 100.0, key=key, num_samples=50)
    several = likelihood.return_level_samples(
        dist, jnp.array([10.0, 50.0, 100.0]), key=key, num_samples=50
    )
    assert single.shape == (50, 4)
    assert several.shape == (50, 3, 4)
    np.testing.assert_allclose(several[:, 2, :], single, rtol=1e-12)


def test_gev_return_level_samples_match_the_closed_form():
    # The return level is the latent location plus a constant, so its
    # distribution is Gaussian with the latent variance.
    scale, shape, period = 2.0, -0.15, 100.0
    likelihood = GEV(scale=scale, shape=shape)
    mean, sd = jnp.array([10.0, 20.0]), jnp.array([0.5, 1.5])
    dist = _correlated_dist(mean, sd, 0.5)
    samples = likelihood.return_level_samples(
        dist, period, key=jr.key(1), num_samples=20000
    )
    offset = gev_return_level(period, 0.0, scale, shape)
    standard_error = sd / np.sqrt(20000)
    assert jnp.all(jnp.abs(samples.mean(0) - (mean + offset)) < 4.0 * standard_error)
    np.testing.assert_allclose(samples.std(0), sd, rtol=0.03)
    # Two return periods differ by a constant in every sample.
    pair = likelihood.return_level_samples(
        dist, jnp.array([10.0, 100.0]), key=jr.key(1), num_samples=10
    )
    difference = gev_return_level(100.0, 0.0, scale, shape) - gev_return_level(
        10.0, 0.0, scale, shape
    )
    np.testing.assert_allclose(pair[:, 1] - pair[:, 0], difference, rtol=1e-10)


def test_gev_return_level_samples_keep_the_joint_draw():
    likelihood = GEV(scale=1.0, shape=0.0)
    joint = _correlated_dist([0.0, 0.0], [1.0, 1.0], 0.999999)
    samples = likelihood.return_level_samples(
        joint, 10.0, key=jr.key(2), num_samples=2000
    )
    np.testing.assert_allclose(samples[:, 0], samples[:, 1], atol=1e-2)
    independent = GaussianDistribution(
        jnp.zeros(2), lx.DiagonalLinearOperator(jnp.ones(2))
    )
    samples = likelihood.return_level_samples(
        independent, 10.0, key=jr.key(2), num_samples=2000
    )
    assert abs(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]) < 0.1


def test_gev_return_period_samples_invert_the_return_level():
    likelihood = GEV(scale=3.0, shape=0.2)
    mean = jnp.array([5.0, 8.0])
    dist = _correlated_dist(mean, [1e-6, 1e-6], 0.0)
    periods = jnp.array([20.0, 500.0])
    levels = gev_return_level(periods, mean, 3.0, 0.2)
    samples = likelihood.return_period_samples(
        dist, levels, key=jr.key(3), num_samples=100
    )
    assert samples.shape == (100, 2)
    np.testing.assert_allclose(samples, jnp.broadcast_to(periods, (100, 2)), rtol=1e-4)


def test_gev_return_period_samples_keep_infinity():
    # Bounded tail: the level is impossible in samples whose upper end point,
    # f + scale / |shape|, is below it.
    scale, shape = 1.0, -0.25
    likelihood = GEV(scale=scale, shape=shape)
    mean, sd, level = 0.0, 1.0, 4.5
    dist = _correlated_dist([mean], [sd], 0.0)
    samples = likelihood.return_period_samples(
        dist, level, key=jr.key(4), num_samples=20000
    )
    assert not jnp.any(jnp.isnan(samples))
    share_impossible = float(jnp.mean(jnp.isposinf(samples)))
    expected = stats.norm.cdf(level - scale / abs(shape), mean, sd)
    assert abs(share_impossible - expected) < 4.0 * np.sqrt(
        expected * (1 - expected) / 20000
    )
    # The exceedance probability 1 / T is exactly 0 in those samples.
    assert jnp.all((1.0 / samples)[jnp.isposinf(samples)] == 0.0)


def test_gev_probability_ratio_from_paired_samples():
    # The same latent draw at two covariate values: the second has a higher
    # location, so the level is more likely there in almost every sample.
    likelihood = GEV(scale=1.0, shape=0.1)
    dist = _correlated_dist([0.0, 0.5], [0.3, 0.3], 0.95)
    periods = likelihood.return_period_samples(
        dist, 4.0, key=jr.key(5), num_samples=4000
    )
    probability = 1.0 / periods
    ratio = probability[:, 1] / probability[:, 0]
    assert ratio.shape == (4000,)
    assert float(jnp.mean(ratio > 1.0)) > 0.95
    at_mean = gpx.distributions.gev_return_period(4.0, jnp.array([0.0, 0.5]), 1.0, 0.1)
    np.testing.assert_allclose(jnp.median(ratio), at_mean[0] / at_mean[1], rtol=0.05)


# ---------------------------------------------------------------------------
# GEV: recovery of known parameters, and a covariate in the location
# ---------------------------------------------------------------------------


def _gumbel_moment_start(y):
    sigma0 = float(jnp.std(y)) * np.sqrt(6.0) / np.pi
    return float(jnp.mean(y)) - 0.5772 * sigma0, sigma0


def _fit_gev_variational(x, y, inducing_inputs, kernel, num_iters):
    mu0, sigma0 = _gumbel_moment_start(y)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Constant(mu0), kernel=kernel)
    q = gpx.variational_families.WhitenedVariationalGaussian(
        model=prior * GEV(scale=sigma0, shape=0.0), inducing_inputs=inducing_inputs
    )
    data = gpx.Dataset(X=x, y=y)
    fitted, history = gpx.fit(
        model=q,
        objective=lambda q, d: -gpx.objectives.elbo(q, d),
        train_data=data,
        optim=ox.adam(0.05),
        num_iters=num_iters,
        verbose=False,
    )
    assert jnp.all(jnp.isfinite(history))
    return fitted, data


def _check_variational_recovery(shape, num_data, num_iters, tolerance):
    key_x, key_y = jr.split(jr.key(int(1000 * shape) + num_data))
    x = jnp.sort(jr.uniform(key_x, (num_data, 1)), axis=0)
    location = 3.0 * jnp.sin(2.0 * jnp.pi * x)
    y = GeneralisedExtremeValue(location, 1.0, shape).sample(key_y)
    fitted, data = _fit_gev_variational(
        x,
        y,
        jnp.linspace(0.0, 1.0, 20)[:, None],
        gpx.kernels.RBF(lengthscale=0.3),
        num_iters,
    )
    likelihood = fitted.model.likelihood
    latent = fitted.condition(data)(x, covariance="diagonal").mean
    assert abs(float(val(likelihood.shape)) - shape) < tolerance["shape"]
    assert abs(float(val(likelihood.scale)) - 1.0) < tolerance["scale"]
    assert float(jnp.sqrt(jnp.mean((latent - location[:, 0]) ** 2))) < tolerance["rmse"]


def test_gev_variational_recovery_quick():
    _check_variational_recovery(
        0.2, 200, 1000, {"shape": 0.1, "scale": 0.15, "rmse": 0.35}
    )


@pytest.mark.slow
@pytest.mark.parametrize("shape", [-0.2, 0.2])
def test_gev_variational_recovery(shape):
    _check_variational_recovery(
        shape, 600, 1500, {"shape": 0.08, "scale": 0.1, "rmse": 0.2}
    )


def _check_map_recovery(shape, num_sites, num_years, tolerance):
    # Many years at each site: all the years of a site share one latent value,
    # so the latent cannot interpolate the data and the mode exists.
    key = jr.key(int(1000 * shape) + num_sites)
    sites = jnp.linspace(0.0, 1.0, num_sites)
    x = jnp.repeat(sites, num_years)[:, None]
    location = 3.0 * jnp.sin(2.0 * jnp.pi * x)
    y = GeneralisedExtremeValue(location, 1.0, shape).sample(key)
    data = gpx.Dataset(X=x, y=y)
    mu0, sigma0 = _gumbel_moment_start(y)
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(mu0),
        kernel=gpx.kernels.RBF(lengthscale=0.3),
    )
    model = prior * GEV(scale=sigma0, shape=0.0)
    fitted, history = gpx.fit_scipy(
        model=model,
        objective=lambda m, d: -gpx.objectives.log_posterior_density(m, d),
        train_data=data,
        max_iters=1000,
        verbose=False,
    )
    assert history[-1] < history[0]
    latent = fitted.condition(data)(sites[:, None], covariance="diagonal").mean
    assert abs(float(val(fitted.likelihood.shape)) - shape) < tolerance["shape"]
    assert abs(float(val(fitted.likelihood.scale)) - 1.0) < tolerance["scale"]
    truth = 3.0 * jnp.sin(2.0 * jnp.pi * sites)
    assert float(jnp.sqrt(jnp.mean((latent - truth) ** 2))) < tolerance["rmse"]


def test_gev_map_recovery_with_repeated_sites_quick():
    _check_map_recovery(0.2, 4, 20, {"shape": 0.2, "scale": 0.25, "rmse": 0.5})


@pytest.mark.slow
@pytest.mark.parametrize("shape", [-0.2, 0.2])
def test_gev_map_recovery_with_repeated_sites(shape):
    _check_map_recovery(shape, 10, 30, {"shape": 0.15, "scale": 0.12, "rmse": 0.3})


def test_gev_covariate_in_the_location_shifts_the_return_level():
    # mu = mu0(site) + beta * gmst: the covariate is the second input column,
    # with a Linear kernel on it.
    beta = 2.0
    sites, gmst = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, 15), jnp.linspace(0.0, 1.0, 20), indexing="ij"
    )
    x = jnp.stack([sites.ravel(), gmst.ravel()], axis=1)
    location = 10.0 + jnp.sin(2.0 * jnp.pi * x[:, :1]) + beta * x[:, 1:]
    y = GeneralisedExtremeValue(location, 1.0, 0.1).sample(jr.key(5))
    z_sites, z_gmst = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, 8), jnp.linspace(0.0, 1.0, 3), indexing="ij"
    )
    kernel = gpx.kernels.RBF(active_dims=[0], lengthscale=0.3) + gpx.kernels.Linear(
        active_dims=[1]
    )
    fitted, data = _fit_gev_variational(
        x,
        y,
        jnp.stack([z_sites.ravel(), z_gmst.ravel()], axis=1),
        kernel,
        1000,
    )
    likelihood = fitted.model.likelihood

    # The same site in a cool climate (gmst = 0) and a warm one (gmst = 1),
    # with a dense covariance so that the two columns are paired draws.
    pair = jnp.array([[0.25, 0.0], [0.25, 1.0]])
    latent = fitted.condition(data)(pair)
    levels = likelihood.return_level_samples(
        latent, 100.0, key=jr.key(0), num_samples=4000
    )
    shift = levels[:, 1] - levels[:, 0]
    assert abs(float(jnp.mean(shift)) - beta) < 0.5
    assert float(jnp.mean(shift > 0.0)) > 0.99

    # The probability ratio of the cool-climate 100-year level.
    level = float(jnp.median(levels[:, 0]))
    periods = likelihood.return_period_samples(
        latent, level, key=jr.key(1), num_samples=4000
    )
    probability_ratio = periods[:, 0] / periods[:, 1]
    assert float(jnp.mean(probability_ratio > 1.0)) > 0.99
    assert float(jnp.median(periods[:, 0])) == pytest.approx(100.0, rel=0.3)


def test_gev_shape_stays_in_its_bounds_and_a_fixed_shape_stays_fixed():
    # Heavy-tailed data (shape 0.8), and a start of the shape near its bound.
    key_x, key_y = jr.split(jr.key(9))
    x = jnp.sort(jr.uniform(key_x, (80, 1)), axis=0)
    y = GeneralisedExtremeValue(jnp.zeros_like(x), 1.0, 0.8).sample(key_y)
    data = gpx.Dataset(X=x, y=y)
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(), kernel=gpx.kernels.RBF()
    )
    z = jnp.linspace(0.0, 1.0, 5)[:, None]

    def fit(likelihood):
        q = gpx.variational_families.WhitenedVariationalGaussian(
            model=prior * likelihood, inducing_inputs=z
        )
        fitted, history = gpx.fit(
            model=q,
            objective=lambda q, d: -gpx.objectives.elbo(q, d),
            train_data=data,
            optim=ox.adam(0.1),
            num_iters=300,
            verbose=False,
        )
        assert jnp.all(jnp.isfinite(history))
        return fitted.model.likelihood

    free = fit(GEV(shape=0.49))
    assert -0.5 < float(val(free.shape)) < 0.5
    assert abs(float(val(free.shape)) - 0.49) > 1e-3
    assert jnp.isfinite(free.shape._unconstrained)

    fixed = fit(GEV(shape=paramax.non_trainable(jnp.array(0.1))))
    assert float(val(fixed.shape)) == pytest.approx(0.1, abs=1e-12)
