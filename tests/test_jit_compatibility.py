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

"""JIT compatibility smoke tests.

Verifies that core GPJax operations (kernel evaluation, GP prediction,
linalg operators) produce identical results with and without
``jax.jit``.
"""

from gpjax.dataset import Dataset
from gpjax.gps import Prior
from gpjax.kernels.stationary import RBF, Matern32
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant
import jax
from jax import config
import jax.numpy as jnp
import lineax as lx
import pytest

config.update("jax_enable_x64", True)

# Shared fixtures ----------------------------------------------------------- #

N_TRAIN = 20
N_TEST = 5
D = 1


@pytest.fixture
def train_data():
    x = jnp.linspace(0, 1, N_TRAIN)[:, None]
    y = jnp.sin(x)
    return Dataset(X=x, y=y)


@pytest.fixture
def test_inputs():
    return jnp.linspace(0, 1, N_TEST)[:, None]


# --------------------------------------------------------------------------- #
#  Kernel JIT tests                                                            #
# --------------------------------------------------------------------------- #

KERNELS = [RBF, Matern32]


@pytest.mark.parametrize("KernelClass", KERNELS)
def test_kernel_gram_jit(KernelClass):
    x = jnp.linspace(0, 1, N_TRAIN)[:, None]
    kernel = KernelClass()

    def gram_fn(x):
        return kernel.gram(x).as_matrix()

    result = gram_fn(x)
    result_jit = jax.jit(gram_fn)(x)
    assert jnp.allclose(result, result_jit, atol=1e-12)


@pytest.mark.parametrize("KernelClass", KERNELS)
def test_kernel_cross_covariance_jit(KernelClass):
    x = jnp.linspace(0, 1, N_TRAIN)[:, None]
    y = jnp.linspace(0.5, 1.5, N_TEST)[:, None]
    kernel = KernelClass()

    def cross_cov_fn(x, y):
        return kernel.cross_covariance(x, y)

    result = cross_cov_fn(x, y)
    result_jit = jax.jit(cross_cov_fn)(x, y)
    assert jnp.allclose(result, result_jit, atol=1e-12)


# --------------------------------------------------------------------------- #
#  GP Prior JIT tests                                                          #
# --------------------------------------------------------------------------- #


def test_prior_predict_jit(test_inputs):
    prior = Prior(mean_function=Constant(), kernel=RBF())

    def predict_fn(x):
        dist = prior.predict(x)
        return dist.mean, dist.covariance()

    mean, cov = predict_fn(test_inputs)
    mean_jit, cov_jit = jax.jit(predict_fn)(test_inputs)

    assert jnp.allclose(mean, mean_jit, atol=1e-12)
    assert jnp.allclose(cov, cov_jit, atol=1e-12)


# --------------------------------------------------------------------------- #
#  GP Posterior JIT tests                                                      #
# --------------------------------------------------------------------------- #


def test_conjugate_posterior_predict_jit(train_data, test_inputs):
    prior = Prior(mean_function=Constant(), kernel=RBF())
    likelihood = Gaussian()
    posterior = prior * likelihood

    def predict_fn(x_test, data):
        dist = posterior.predict(x_test, data)
        return dist.mean, dist.covariance()

    mean, cov = predict_fn(test_inputs, train_data)
    mean_jit, cov_jit = jax.jit(predict_fn)(test_inputs, train_data)

    assert jnp.allclose(mean, mean_jit, atol=1e-12)
    assert jnp.allclose(cov, cov_jit, atol=1e-12)


# --------------------------------------------------------------------------- #
#  Linalg operator JIT tests                                                   #
# --------------------------------------------------------------------------- #


def test_dense_matmul_jit():
    A = lx.MatrixLinearOperator(jnp.eye(3))
    v = jnp.ones(3)

    def matmul_fn(v):
        return A.mv(v)

    result = matmul_fn(v)
    result_jit = jax.jit(matmul_fn)(v)
    assert jnp.allclose(result, result_jit, atol=1e-12)


def test_diagonal_as_matrix_jit():
    d = jnp.array([1.0, 2.0, 3.0])

    def as_matrix_fn(d):
        return lx.DiagonalLinearOperator(d).as_matrix()

    result = as_matrix_fn(d)
    result_jit = jax.jit(as_matrix_fn)(d)
    assert jnp.allclose(result, result_jit, atol=1e-12)


def test_identity_as_matrix_jit():
    def as_matrix_fn():
        return lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct((3,), jnp.float64)
        ).as_matrix()

    result = as_matrix_fn()
    result_jit = jax.jit(as_matrix_fn)()
    assert jnp.allclose(result, result_jit, atol=1e-12)
