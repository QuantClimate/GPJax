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
)
from gpjax.likelihoods import (
    Bernoulli,
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
