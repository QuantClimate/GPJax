# Copyright 2024 The GPJax Contributors. All Rights Reserved.
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

"""Tests that core GPJax operations produce correct dtypes under float32 and
float64.  The global ``jax_enable_x64`` flag is toggled per-test via the
*enable_x64* fixture so these tests exercise both precision modes.
"""

from gpjax.dataset import Dataset
from gpjax.distributions import GeneralisedExtremeValue
from gpjax.gps import Prior
from gpjax.kernels.stationary import RBF, Matern32
from gpjax.likelihoods import (
    GEV,
    Gaussian,
    HeteroscedasticGEV,
)
from gpjax.mean_functions import Constant
from gpjax.objectives import (
    conjugate_mll,
    log_posterior_density,
)
from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest

config.update("jax_enable_x64", True)

KERNELS = [RBF, Matern32]


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["float32", "float64"])
def enable_x64(request):
    """Toggle x64 mode and yield the target dtype."""
    dtype = request.param
    use_x64 = dtype == jnp.float64
    prev = config.jax_enable_x64
    config.update("jax_enable_x64", use_x64)
    try:
        yield dtype
    finally:
        config.update("jax_enable_x64", prev)


# --------------------------------------------------------------------------- #
#  Kernel dtype tests                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64:UserWarning")
@pytest.mark.parametrize("KernelClass", KERNELS)
def test_kernel_gram_dtype(KernelClass, enable_x64):
    dtype = enable_x64
    x = jnp.linspace(0, 1, 10, dtype=dtype)[:, None]
    kernel = KernelClass()
    Kxx = kernel.gram(x).as_matrix()
    assert Kxx.dtype == dtype, f"Expected {dtype}, got {Kxx.dtype}"


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64:UserWarning")
@pytest.mark.parametrize("KernelClass", KERNELS)
def test_kernel_cross_covariance_dtype(KernelClass, enable_x64):
    dtype = enable_x64
    x = jnp.linspace(0, 1, 10, dtype=dtype)[:, None]
    y = jnp.linspace(0.5, 1.5, 5, dtype=dtype)[:, None]
    kernel = KernelClass()
    Kxy = kernel.cross_covariance(x, y)
    assert Kxy.dtype == dtype, f"Expected {dtype}, got {Kxy.dtype}"


# --------------------------------------------------------------------------- #
#  GP prior dtype tests                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64:UserWarning")
def test_prior_predict_dtype(enable_x64):
    dtype = enable_x64
    x = jnp.linspace(0, 1, 10, dtype=dtype)[:, None]
    prior = Prior(mean_function=Constant(), kernel=RBF())
    dist = prior.predict(x)
    assert dist.mean.dtype == dtype, (
        f"Mean dtype: expected {dtype}, got {dist.mean.dtype}"
    )


# --------------------------------------------------------------------------- #
#  GP posterior dtype tests                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64:UserWarning")
@pytest.mark.filterwarnings("ignore:X is not of type float64:UserWarning")
@pytest.mark.filterwarnings("ignore:y is not of type float64:UserWarning")
def test_posterior_predict_dtype(enable_x64):
    dtype = enable_x64
    x_train = jnp.linspace(0, 1, 20, dtype=dtype)[:, None]
    y_train = jnp.sin(x_train)
    D = Dataset(X=x_train, y=y_train)

    x_test = jnp.linspace(0, 1, 5, dtype=dtype)[:, None]
    prior = Prior(mean_function=Constant(), kernel=RBF())
    likelihood = Gaussian()
    posterior = prior * likelihood

    dist = posterior.predict(x_test, D)
    assert dist.mean.dtype == dtype, (
        f"Mean dtype: expected {dtype}, got {dist.mean.dtype}"
    )


# --------------------------------------------------------------------------- #
#  Objective dtype tests                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64:UserWarning")
@pytest.mark.filterwarnings("ignore:X is not of type float64:UserWarning")
@pytest.mark.filterwarnings("ignore:y is not of type float64:UserWarning")
def test_conjugate_mll_dtype(enable_x64):
    dtype = enable_x64
    x = jnp.linspace(0, 1, 20, dtype=dtype)[:, None]
    y = jnp.sin(x)
    D = Dataset(X=x, y=y)

    prior = Prior(mean_function=Constant(), kernel=RBF())
    likelihood = Gaussian()
    posterior = prior * likelihood

    mll = conjugate_mll(posterior, D)
    assert mll.dtype == dtype, f"MLL dtype: expected {dtype}, got {mll.dtype}"


# --------------------------------------------------------------------------- #
#  GEV dtype tests                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64:UserWarning")
@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("shape", [-0.2, 0.0, 1e-6, 0.2])
def test_gev_distribution_dtype(enable_x64, continuation, shape):
    dtype = enable_x64
    dist = GeneralisedExtremeValue(
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(2.0, dtype=dtype),
        jnp.asarray(shape, dtype=dtype),
        support_continuation=continuation,
    )
    y = jnp.linspace(-20.0, 30.0, 11, dtype=dtype)
    q = jnp.linspace(0.01, 0.99, 5, dtype=dtype)
    outputs = {
        "log_prob": dist.log_prob(y),
        "cdf": dist.cdf(y),
        "sf": dist.sf(y),
        "icdf": dist.icdf(q),
        "isf": dist.isf(q),
        "return_level": dist.return_level(jnp.asarray([10.0, 1e4], dtype=dtype)),
        "return_level_of_a_float": dist.return_level(100.0),
        "return_period": dist.return_period(y),
        "sample": dist.sample(jr.key(0), (5,)),
        "mean": dist.mean,
        "variance": dist.variance,
    }
    for name, output in outputs.items():
        assert output.dtype == dtype, f"{name}: expected {dtype}, got {output.dtype}"
        if name != "log_prob" or continuation:
            assert jnp.all(~jnp.isnan(output)), name
    if continuation:
        assert jnp.all(jnp.isfinite(outputs["log_prob"]))


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64:UserWarning")
def test_gev_float32_matches_float64():
    def evaluate(dtype):
        dist = GeneralisedExtremeValue(
            jnp.asarray(1.0, dtype=dtype),
            jnp.asarray(2.0, dtype=dtype),
            jnp.asarray(1e-5, dtype=dtype),
            support_continuation=True,
        )
        y = jnp.linspace(-5.0, 30.0, 11, dtype=dtype)
        return dist.log_prob(y), dist.return_level(jnp.asarray([10.0, 1e3], dtype))

    for low, high in zip(evaluate(jnp.float32), evaluate(jnp.float64), strict=True):
        assert low.dtype == jnp.float32
        assert jnp.allclose(low, high, rtol=1e-5, atol=1e-5)


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64:UserWarning")
@pytest.mark.filterwarnings("ignore:X is not of type float64:UserWarning")
@pytest.mark.filterwarnings("ignore:y is not of type float64:UserWarning")
def test_gev_likelihood_dtype(enable_x64):
    dtype = enable_x64
    y = jnp.linspace(0.0, 3.0, 6, dtype=dtype)[:, None]
    mean = jnp.zeros((6, 1), dtype=dtype)
    variance = jnp.full((6, 1), 0.5, dtype=dtype)

    expected = GEV(scale=1.5, shape=0.1).expected_log_likelihood(y, mean, variance)
    assert expected.dtype == dtype
    expected = HeteroscedasticGEV(shape=0.1).expected_log_likelihood(
        y, mean, variance, mean_g=mean, variance_g=variance
    )
    assert expected.dtype == dtype

    data = Dataset(X=jnp.linspace(0.0, 1.0, 6, dtype=dtype)[:, None], y=y)
    prior = Prior(mean_function=Constant(), kernel=RBF())
    model = (prior * GEV(scale=1.5, shape=0.1)).init_latent(data.n)
    model = model.__class__(
        prior=model.prior,
        likelihood=model.likelihood,
        latent=jnp.zeros((6, 1), dtype=dtype),
    )
    density = log_posterior_density(model, data)
    assert density.dtype == dtype
