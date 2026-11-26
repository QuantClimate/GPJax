# Copyright 2022 The thomaspinder Contributors. All Rights Reserved.
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

from itertools import product
from typing import Any

import equinox as eqx
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.kernels.nonstationary import (
    ArcCosine,
    Linear,
    Polynomial,
)
from gpjax.parameters import NonNegativeReal, val
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from paramax import AbstractUnwrappable
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def params_product(params: dict[str, list]) -> list[dict[str, Any]]:
    return [
        dict(zip(params.keys(), values, strict=False))
        for values in product(*params.values())
    ]


TESTED_KERNELS = [
    (
        ArcCosine,
        params_product(
            {
                "order": [0, 1, 2],
                "weight_variance": [0.1, 1.0],
                "bias_variance": [0.1, 1.0],
            }
        ),
    ),
    (Linear, [{}]),
    (Polynomial, params_product({"degree": [1, 2, 3], "shift": [1e-6, 0.1, 1.0]})),
]

VARIANCES = [0.1]


@pytest.fixture
def kernel_request(
    kernel,
    params,
    variance,
):
    return kernel, params, variance


# NOTE: no marks here. `kernel`, `params` and `variance` are supplied by the
# parametrize marks on each consuming test. Marking a fixture is a no-op and is
# a hard error from pytest 9.1.
@pytest.fixture
def test_init(kernel_request):
    kernel, params, variance = kernel_request
    return kernel(**params, variance=variance)


@pytest.mark.parametrize(
    ("kernel", "params"), [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("variance", VARIANCES)
def test_init_override_paramtype(kernel_request):
    kernel, params, variance = kernel_request

    new_params = {}  # otherwise we change the fixture and next test fails
    for param, value in params.items():
        if param in ("degree", "order"):
            continue
        new_params[param] = value

    k = kernel(**new_params, variance=NonNegativeReal(variance))
    assert isinstance(k.variance, NonNegativeReal)

    for param in params:
        if param in ("degree", "order", "weight_variance", "bias_variance"):
            continue
        # Parameter is now a raw value, not a Static object
        assert not isinstance(getattr(k, param), AbstractUnwrappable)


@pytest.mark.parametrize("kernel", [k[0] for k in TESTED_KERNELS])
def test_init_defaults(kernel: type[AbstractKernel]):
    # Initialise kernel
    k = kernel()

    # Check that the parameters are set correctly
    assert isinstance(k.compute_engine, type(AbstractKernelComputation()))
    assert isinstance(k.variance, NonNegativeReal)


@pytest.mark.parametrize("kernel", [k[0] for k in TESTED_KERNELS])
@pytest.mark.parametrize("variance", VARIANCES)
def test_init_variances(kernel: type[AbstractKernel], variance):
    # Initialise kernel
    k = kernel(variance=variance)

    # Check that the parameters are set correctly
    assert isinstance(k.variance, NonNegativeReal)
    assert jnp.allclose(k.variance.unwrap(), jnp.asarray(variance))


@pytest.mark.parametrize(
    ("kernel", "params"), [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("variance", VARIANCES)
@pytest.mark.parametrize("n", [1, 2, 5], ids=lambda x: f"n={x}")
def test_gram(test_init: AbstractKernel, n: int):
    # kernel is initialized in the test_init fixture
    k = test_init
    n_dims = k.n_dims or 1

    # Inputs
    x = jnp.linspace(0.0, 1.0, n * n_dims).reshape(n, n_dims)

    # Test gram matrix
    Kxx = k.gram(x)
    assert isinstance(Kxx, lx.AbstractLinearOperator)
    assert Kxx.as_matrix().shape == (n, n)
    assert jnp.all(jnp.linalg.eigvalsh(Kxx.as_matrix() + jnp.eye(n) * 1e-6) > 0.0)


@pytest.mark.parametrize(
    ("kernel", "params"), [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("variance", VARIANCES)
@pytest.mark.parametrize("n_a", [1, 2, 5], ids=lambda x: f"n_a={x}")
@pytest.mark.parametrize("n_b", [1, 2, 5], ids=lambda x: f"n_b={x}")
def test_cross_covariance(test_init: AbstractKernel, n_a: int, n_b: int):
    # kernel is initialized in the test_init fixture
    k = test_init
    n_dims = k.n_dims or 1

    # Inputs
    x = jnp.linspace(0.0, 1.0, n_a * n_dims).reshape(n_a, n_dims)
    y = jnp.linspace(0.0, 1.0, n_b * n_dims).reshape(n_b, n_dims)

    # Test cross covariance matrix
    Kxy = k.cross_covariance(x, y)
    assert isinstance(Kxy, jax.Array)
    assert Kxy.shape == (n_a, n_b)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_arccosine_special_case(order: int):
    """For certain values of weight variance (1.0) and bias variance (0.0), we can test
    our calculations using the Monte Carlo expansion of the arccosine kernel, e.g.
    see Eq. (1) of https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf.
    """
    kernel = ArcCosine(
        weight_variance=jnp.array([1.0, 1.0]), bias_variance=1e-25, order=order
    )

    # Inputs close(ish) together
    a = jnp.array([[0.0, 0.0]])
    b = jnp.array([[2.0, 2.0]])

    # calc cross-covariance exactly
    Kab_exact = kernel.cross_covariance(a, b)

    # calc cross-covariance using samples
    weights = jax.random.normal(jr.key(123), (10_000, 2))  # [S, d]
    weights_a = jnp.matmul(weights, a.T)  # [S, 1]
    weights_b = jnp.matmul(weights, b.T)  # [S, 1]
    H_a = jnp.heaviside(weights_a, 0.5)
    H_b = jnp.heaviside(weights_b, 0.5)
    integrands = H_a * H_b * (weights_a**order) * (weights_b**order)
    Kab_approx = 2.0 * jnp.mean(integrands)

    assert jnp.max(Kab_approx - Kab_exact) < 1e-4


def test_arccosine_variances_wrapped_and_trainable():
    """All three ArcCosine variances must be NonNegativeReal so they are
    trainable (not silently frozen) and constrained (cannot go negative)."""
    kernel = ArcCosine(
        order=1,
        n_dims=2,
        weight_variance=1.0,
        bias_variance=1.0,
        variance=1.0,
    )
    assert isinstance(kernel.weight_variance, NonNegativeReal)
    assert isinstance(kernel.bias_variance, NonNegativeReal)
    assert isinstance(kernel.variance, NonNegativeReal)

    # Every variance must land in the trainable (array) partition, not static.
    params, _ = eqx.partition(kernel, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    assert len(leaves) >= 3  # weight, bias, variance all present as arrays


def test_arccosine_variance_stays_positive_under_optimisation():
    """A gradient step that would push weight_variance negative must instead
    leave the constrained value positive and the Gram finite/PSD."""
    x = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    kernel = ArcCosine(order=1, n_dims=1, weight_variance=jnp.array(0.05))
    params, static = eqx.partition(kernel, eqx.is_array)

    def loss(p):
        k = eqx.combine(p, static)
        return jnp.sum(k.gram(x).as_matrix())  # gradient pushes variances down

    grads = jax.grad(loss)(params)
    stepped = jax.tree_util.tree_map(lambda leaf, g: leaf - 100.0 * g, params, grads)
    k_new = eqx.combine(stepped, static)
    assert val(k_new.weight_variance) > 0.0
    gram = k_new.gram(x).as_matrix()
    assert jnp.all(jnp.isfinite(gram))
