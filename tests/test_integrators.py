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


import equinox as eqx
from gpjax.integrators import (
    AnalyticalGaussianIntegrator,
    GHQuadratureIntegrator,
    MonteCarloIntegrator,
)
from gpjax.likelihoods import (
    Bernoulli,
    Categorical,
    Gaussian,
    HeteroscedasticGEV,
)
import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("num_points", [10, 20, 30])
def test_quadrature(jit: bool, num_points: int):
    def test():
        def fun(x, y):
            """In practice, the first argument will be the latent function values"""
            return x**2 + y

        mean = jnp.array([[2.0]])
        variance = jnp.array([[1.0]])
        fn_val = GHQuadratureIntegrator(num_points=num_points).integrate(
            fun=fun,
            mean=mean,
            variance=variance,
            y=jnp.ones_like(mean),
            likelihood=None,
        )
        return fn_val.squeeze().round(1)

    if jit:
        test = jax.jit(test)
    assert test() == 6.0


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize(
    "params", [(0.5, -4.22579135), (1.0, -1.91893853), (0.01, -9996.31376835)]
)
def test_analytical_gaussian(jit: bool, params: tuple[float, float]):
    obs_stddev, expected = params
    likelihood = Gaussian(obs_stddev=jnp.array([obs_stddev]))
    mu = jnp.array([[0.0]])
    variance = jnp.array([[1.0]])
    y = jnp.array([[1.0]])

    @jax.jit
    def _ell(lik, y, mean, variance):
        return lik.expected_log_likelihood(y=y, mean=mean, variance=variance)

    if jit:
        ell = _ell(likelihood, y=y, mean=mu, variance=variance)
    else:
        ell = likelihood.expected_log_likelihood(y=y, mean=mu, variance=variance)
    np.testing.assert_almost_equal(ell, expected)


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("params", [(0.25, 0.5, -0.65437282), (0.5, 1.0, -0.61716802)])
def test_bernoulli_quadrature(jit: bool, params: tuple[float, float]):
    mu, variance, expected = params
    mu = jnp.atleast_2d(mu)
    variance = jnp.atleast_2d(variance)
    likelihood = Bernoulli()
    y = jnp.array([[1.0]])

    @jax.jit
    def _ell(lik, y, mean, variance):
        return lik.expected_log_likelihood(y=y, mean=mean, variance=variance)

    if jit:
        ell = _ell(likelihood, y=y, mean=mu, variance=variance)
    else:
        ell = likelihood.expected_log_likelihood(y=y, mean=mu, variance=variance)
    np.testing.assert_almost_equal(ell, expected)


def test_integrators_compare_by_type_and_settings():
    assert GHQuadratureIntegrator(10) == GHQuadratureIntegrator(10)
    assert hash(GHQuadratureIntegrator(10)) == hash(GHQuadratureIntegrator(10))
    assert GHQuadratureIntegrator(10) != GHQuadratureIntegrator(20)
    assert GHQuadratureIntegrator() != AnalyticalGaussianIntegrator()


def test_integrators_with_array_settings_compare_by_identity():
    class WithArraySetting(GHQuadratureIntegrator):
        def __init__(self):
            super().__init__()
            self.nodes = jnp.arange(3.0)

    integrator = WithArraySetting()
    the_same_integrator = integrator
    assert integrator == the_same_integrator
    assert integrator != WithArraySetting()


@pytest.mark.parametrize(
    "build",
    [
        lambda: HeteroscedasticGEV(shape=0.1, num_points=3),
        lambda: Bernoulli(integrator=GHQuadratureIntegrator(10)),
    ],
)
def test_likelihoods_with_equal_integrators_are_equal_pytrees(build):
    first, second = build(), build()
    stacked = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), first, second)
    assert jax.tree_util.tree_structure(stacked) == jax.tree_util.tree_structure(first)

    trace_count = 0

    @eqx.filter_jit
    def traced(likelihood):
        nonlocal trace_count
        trace_count += 1
        return jnp.zeros(())

    traced(first)
    traced(second)
    assert trace_count == 1


# --- Monte Carlo integration ---


@pytest.mark.parametrize("jit", [True, False])
def test_monte_carlo_matches_the_closed_form_gaussian_value(jit: bool):
    # E[log N(y | f, 1)] with f ~ N(0, 1) and y = 1 is -1.91893853 (closed form).
    likelihood = Gaussian(
        obs_stddev=jnp.array([1.0]),
        integrator=MonteCarloIntegrator(num_samples=20_000),
    )
    y, mean, variance = jnp.array([[1.0]]), jnp.array([[0.0]]), jnp.array([[1.0]])

    def expected_log_likelihood(likelihood, key):
        return likelihood.expected_log_likelihood(y, mean, variance, key=key)

    if jit:
        expected_log_likelihood = jax.jit(expected_log_likelihood)
    estimate = expected_log_likelihood(likelihood, jax.random.key(0))
    assert estimate.shape == (1,)
    np.testing.assert_allclose(estimate, -1.91893853, atol=0.04)


def test_monte_carlo_two_class_softmax_matches_one_dimensional_quadrature():
    # With two classes, log softmax(f)_1 = log sigmoid(f_1 - f_0), and
    # f_1 - f_0 ~ N(mu_1 - mu_0, v_0 + v_1): a 1-D Gauss-Hermite integral.
    mean = jnp.array([[0.3, -0.5], [1.0, 2.0]])
    variance = jnp.array([[0.4, 0.9], [1.5, 0.2]])
    y = jnp.array([[1.0], [0.0]])
    likelihood = Categorical(
        num_classes=2, integrator=MonteCarloIntegrator(num_samples=20_000)
    )
    estimate = likelihood.expected_log_likelihood(
        y, mean, variance, key=jax.random.key(1)
    )

    points, weights = np.polynomial.hermite.hermgauss(60)
    expected = []
    for n in range(2):
        label = int(y[n, 0])
        other = 1 - label
        difference_mean = float(mean[n, label] - mean[n, other])
        difference_stddev = float(np.sqrt(variance[n, 0] + variance[n, 1]))
        difference = difference_mean + np.sqrt(2.0) * difference_stddev * points
        log_sigmoid = -np.logaddexp(0.0, -difference)
        expected.append(np.sum(weights * log_sigmoid) / np.sqrt(np.pi))
    assert estimate.shape == (2,)
    np.testing.assert_allclose(estimate, np.array(expected), atol=0.02)


def test_monte_carlo_is_deterministic_for_a_fixed_key():
    likelihood = Categorical(num_classes=3, integrator=MonteCarloIntegrator())
    mean = jnp.zeros((4, 3))
    variance = jnp.ones((4, 3))
    y = jnp.array([[0.0], [1.0], [2.0], [1.0]])
    first = likelihood.expected_log_likelihood(y, mean, variance, key=jax.random.key(3))
    second = likelihood.expected_log_likelihood(
        y, mean, variance, key=jax.random.key(3)
    )
    other = likelihood.expected_log_likelihood(y, mean, variance, key=jax.random.key(4))
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, other)


def test_monte_carlo_without_a_key_raises():
    likelihood = Categorical(num_classes=3, integrator=MonteCarloIntegrator())
    with pytest.raises(ValueError, match="key="):
        likelihood.expected_log_likelihood(
            jnp.zeros((2, 1)), jnp.zeros((2, 3)), jnp.ones((2, 3))
        )


def test_monte_carlo_integrators_with_equal_settings_are_equal():
    assert MonteCarloIntegrator(num_samples=10) == MonteCarloIntegrator(num_samples=10)
    assert MonteCarloIntegrator(num_samples=10) != MonteCarloIntegrator(num_samples=20)


@pytest.mark.parametrize(
    "likelihood",
    [Bernoulli(), Gaussian(obs_stddev=jnp.array([0.5]))],
    ids=["quadrature", "analytical"],
)
def test_deterministic_integrators_ignore_a_supplied_key(likelihood):
    # The objective can pass a key whatever the likelihood; a deterministic
    # integrator gives the same value with and without it.
    y, mean, variance = jnp.array([[1.0]]), jnp.array([[0.3]]), jnp.array([[0.7]])
    without_key = likelihood.expected_log_likelihood(y, mean, variance)
    with_key = likelihood.expected_log_likelihood(
        y, mean, variance, key=jax.random.key(0)
    )
    np.testing.assert_array_equal(with_key, without_key)


def test_quadrature_with_more_than_one_latent_function_raises():
    likelihood = Categorical(num_classes=3, integrator=GHQuadratureIntegrator())
    with pytest.raises(ValueError, match="MonteCarloIntegrator"):
        likelihood.expected_log_likelihood(
            jnp.zeros((2, 1)), jnp.zeros((2, 3)), jnp.ones((2, 3))
        )
