"""Tests for the predictively oriented posterior."""

import gpjax as gpx
from gpjax.experimentation import predictively_oriented
from gpjax.parameters import val
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm
import numpy as np
import pytest


def _data(num_data: int = 30) -> gpx.Dataset:
    x = jnp.linspace(-2.0, 2.0, num_data).reshape(-1, 1)
    y = jnp.sin(2.0 * x) + 0.1 * jr.normal(jr.key(1), x.shape)
    return gpx.Dataset(X=x, y=y)


def _model(obs_stddev: float = 0.2) -> gpx.ConjugateModel:
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(jnp.array([0.3])),
        kernel=gpx.kernels.RBF(lengthscale=0.7),
    )
    return prior * gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)


def test_one_particle_without_adaptation_gives_the_bayesian_posterior():
    # With one particle and one copy of each observation, the allocation step
    # gives every observation to the particle. Each sweep is then an exact
    # draw from the Bayesian posterior.
    data, model = _data(12), _model()
    posterior = predictively_oriented.condition(
        model,
        data,
        jr.key(0),
        num_particles=1,
        num_adaptation_sweeps=0,
        num_final_sweeps=4000,
    )
    test_inputs = jnp.linspace(-2.5, 2.5, 7).reshape(-1, 1)
    latent = posterior(test_inputs)
    exact = model.condition(data)(test_inputs, covariance="diagonal")

    np.testing.assert_allclose(latent.mean, exact.mean, atol=0.05)
    np.testing.assert_allclose(latent.variance, exact.variance, rtol=0.1)
    assert val(posterior.likelihood.obs_stddev) == pytest.approx(0.2)


def test_condition_fits_the_noise_and_keeps_the_model():
    data, model = _data(), _model()
    posterior = predictively_oriented.condition(
        model,
        data,
        jr.key(0),
        num_particles=8,
        num_adaptation_sweeps=15,
        num_final_sweeps=3,
    )

    assert posterior.noise_trace.shape == (15,)
    assert val(posterior.likelihood.obs_stddev) == pytest.approx(
        float(posterior.noise_trace[-1])
    )
    assert val(model.likelihood.obs_stddev) == pytest.approx(0.2)
    assert posterior.prior is model.prior
    # The final sweeps use every observation.
    assert posterior.particles.shape == (24, data.n)
    np.testing.assert_array_equal(
        jnp.sort(posterior.particle_inputs[:, 0]), data.X[:, 0]
    )


def test_observation_predictive_adds_the_noise_variance():
    data, model = _data(), _model()
    posterior = predictively_oriented.condition(
        model,
        data,
        jr.key(0),
        num_particles=8,
        num_adaptation_sweeps=10,
        num_final_sweeps=3,
    )
    test_inputs = jnp.linspace(-2.5, 2.5, 7).reshape(-1, 1)
    latent = posterior(test_inputs)
    observed = posterior.predict_observations(test_inputs)
    noise_variance = val(posterior.likelihood.obs_stddev) ** 2

    assert observed.batch_shape == (7,)
    np.testing.assert_allclose(observed.mean, latent.mean)
    np.testing.assert_allclose(observed.variance, latent.variance + noise_variance)
    assert jnp.all(jnp.isfinite(observed.log_prob(jnp.zeros(7))))


def test_inducing_inputs_hold_the_particles():
    data, model = _data(), _model()
    inducing_inputs = jnp.linspace(-2.0, 2.0, 6).reshape(-1, 1)
    posterior = predictively_oriented.condition(
        model,
        data,
        jr.key(0),
        num_particles=8,
        num_adaptation_sweeps=10,
        num_final_sweeps=3,
        inducing_inputs=inducing_inputs,
    )

    np.testing.assert_array_equal(posterior.particle_inputs, inducing_inputs)
    assert posterior.particles.shape == (24, 6)
    assert jnp.all(jnp.isfinite(posterior(data.X).mean))


def test_two_branches_score_better_than_the_bayesian_posterior():
    # Each observation is on one of two curves, 1.5 above or below sin(x).
    # The Bayesian posterior can only widen one Gaussian; the particles can
    # follow both curves.
    x = jr.uniform(jr.key(2), (160, 1), minval=-2.0, maxval=2.0)
    branch = jnp.where(jr.bernoulli(jr.key(3), 0.5, x.shape), 1.5, -1.5)
    y = jnp.sin(x) + branch + 0.1 * jr.normal(jr.key(4), x.shape)
    train = gpx.Dataset(X=x[:120], y=y[:120])
    test = gpx.Dataset(X=x[120:], y=y[120:])
    model = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.RBF(lengthscale=1.0),
    ) * gpx.likelihoods.Gaussian(obs_stddev=1.5)

    posterior = predictively_oriented.condition(
        model,
        train,
        jr.key(0),
        num_particles=20,
        num_adaptation_sweeps=50,
        num_final_sweeps=10,
    )
    score = -jnp.mean(posterior.predict_observations(test.X).log_prob(test.y[:, 0]))
    bayesian = model.likelihood(model.condition(train)(test.X, covariance="diagonal"))
    # `bayesian.log_prob` is the joint density of all test points; the score
    # is the mean over test points of each marginal density.
    bayesian_score = -jnp.mean(
        norm.logpdf(test.y[:, 0], bayesian.mean, jnp.sqrt(bayesian.variance))
    )

    assert score < bayesian_score - 0.5


def test_condition_rejects_a_non_gaussian_likelihood():
    data = _data()
    model = (
        gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=gpx.kernels.RBF())
        * gpx.likelihoods.Bernoulli()
    )
    # In the tests, the beartype import hook rejects the model first; without
    # beartype, `condition` raises its own TypeError.
    with pytest.raises(TypeError):
        predictively_oriented.condition(model, data, jr.key(0))


@pytest.mark.parametrize("validation_fraction", [0.0, 1.0])
def test_condition_rejects_an_empty_part(validation_fraction):
    with pytest.raises(ValueError, match="at least one observation"):
        predictively_oriented.condition(
            _model(),
            _data(),
            jr.key(0),
            validation_fraction=validation_fraction,
        )


def test_condition_rejects_one_noise_value_for_each_observation():
    data = _data()
    model = _model(obs_stddev=jnp.full((data.n,), 0.2))
    with pytest.raises(ValueError, match="one noise standard deviation"):
        predictively_oriented.condition(model, data, jr.key(0))


def test_condition_rejects_several_outputs():
    x = jnp.linspace(-2.0, 2.0, 10).reshape(-1, 1)
    data = gpx.Dataset(X=x, y=jnp.hstack([x, x]))
    with pytest.raises(ValueError, match="one output"):
        predictively_oriented.condition(_model(), data, jr.key(0))
