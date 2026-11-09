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

import dataclasses
from itertools import product
from typing import Any

from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    PoweredExponential,
    RationalQuadratic,
    White,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.parameters import (
    NonNegativeReal,
    PositiveReal,
    SigmoidBounded,
    val,
)
import jax
from jax import config
import jax.numpy as jnp
import lineax as lx
import paramax
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
    (RBF, [{}]),
    (Matern12, [{}]),
    (Matern32, [{}]),
    (Matern52, [{}]),
    (White, [{}]),
    (Periodic, params_product({"period": [0.1, 1.0]})),
    (PoweredExponential, params_product({"power": [0.1, 0.9]})),
    (RationalQuadratic, params_product({"alpha": [0.1, 1.0]})),
]

LENGTHSCALES = [
    0.1,
    jnp.array(0.1),
    [0.1, 0.2],
    jnp.array([0.1, 0.2]),
]

VARIANCES = [0.1]


@pytest.fixture
def kernel_request(
    kernel,
    params,
    lengthscale,
    variance,
):
    return kernel, params, lengthscale, variance


@pytest.fixture
def test_init(kernel_request):
    kernel, params, lengthscale, variance = kernel_request

    # Initialise kernel
    if kernel == White:
        k = kernel(variance=variance, **params)
    else:
        k = kernel(lengthscale=lengthscale, variance=variance, **params)

    return k


@pytest.mark.parametrize(
    "kernel, params", [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("lengthscale", LENGTHSCALES)
@pytest.mark.parametrize("variance", VARIANCES)
def test_init_override_paramtype(kernel_request):
    kernel, params, lengthscale, variance = kernel_request

    new_params = {}  # otherwise we change the fixture and next test fails
    for param, value in params.items():
        new_params[param] = value

    kwargs = {**new_params, "variance": NonNegativeReal(variance)}
    if kernel != White:
        kwargs["lengthscale"] = PositiveReal(lengthscale)

    k = kernel(**kwargs)
    assert isinstance(k.variance, NonNegativeReal)

    for param in params:
        # Extra kernel params (like period, power, alpha) are now wrapped
        # as PositiveReal when a raw scalar is provided, ensuring they
        # participate correctly in gradient-based optimisation.
        attr = getattr(k, param)
        if isinstance(attr, AbstractUnwrappable):
            assert jnp.allclose(attr.unwrap(), jnp.asarray(params[param]))


@pytest.mark.parametrize("kernel", [k[0] for k in TESTED_KERNELS])
def test_init_defaults(kernel: type[StationaryKernel]):
    # Initialise kernel
    k = kernel()

    # Check that the parameters are set correctly
    assert isinstance(k.compute_engine, type(AbstractKernelComputation()))
    assert isinstance(k.variance, NonNegativeReal)

    # White has no lengthscale: __call__ never reads it, so it is not a
    # real (trainable) field -- see test_white_kernel_has_no_lengthscale_leaf.
    if kernel == White:
        assert k.lengthscale is None
    else:
        assert isinstance(k.lengthscale, PositiveReal)


@pytest.mark.parametrize("kernel", [k[0] for k in TESTED_KERNELS])
@pytest.mark.parametrize("lengthscale", LENGTHSCALES)
def test_init_lengthscales(kernel: type[StationaryKernel], lengthscale):
    # We can skip the White kernel as it does not have a lengthscale
    if kernel == White:
        return

    # Initialise kernel
    k = kernel(lengthscale=lengthscale)

    # Check that the parameters are set correctly
    assert isinstance(k.lengthscale, PositiveReal)
    assert jnp.allclose(k.lengthscale.unwrap(), jnp.asarray(lengthscale))

    # Check that error is raised if lengthscale shape is invalid
    with pytest.raises((ValueError, TypeError)):
        k = kernel(lengthscale=jnp.ones((2, 2)))

    with pytest.raises(TypeError):
        k = kernel(lengthscale="invalid type")

    # Check that error is raised if lengthscale is not compatible with n_dims
    with pytest.raises(ValueError):
        k = kernel(lengthscale=jnp.ones(2), n_dims=1)


@pytest.mark.parametrize("kernel", [k[0] for k in TESTED_KERNELS])
@pytest.mark.parametrize("variance", VARIANCES)
def test_init_variances(kernel: type[StationaryKernel], variance):
    # Initialise kernel
    k = kernel(variance=variance)

    # Check that the parameters are set correctly
    assert isinstance(k.variance, NonNegativeReal)
    assert jnp.allclose(k.variance.unwrap(), jnp.asarray(variance))

    # Check that error is raised if variance is not valid
    with pytest.raises((ValueError, TypeError)):
        k = kernel(variance="invalid type")


@pytest.mark.parametrize(
    "kernel, params", [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("lengthscale", LENGTHSCALES)
@pytest.mark.parametrize("variance", VARIANCES)
@pytest.mark.parametrize("n", [1, 2, 5], ids=lambda x: f"n={x}")
def test_gram(test_init: StationaryKernel, n: int):
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
    "kernel, params", [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("lengthscale", LENGTHSCALES)
@pytest.mark.parametrize("variance", VARIANCES)
@pytest.mark.parametrize("n_a", [1, 2, 5], ids=lambda x: f"n_a={x}")
@pytest.mark.parametrize("n_b", [1, 2, 5], ids=lambda x: f"n_b={x}")
def test_cross_covariance(test_init: StationaryKernel, n_a: int, n_b: int):
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


def test_white_kernel_has_no_lengthscale_leaf():
    """Regression test for issue #695.

    `White.__call__` never reads a lengthscale, so `StationaryKernel`'s
    lengthscale must not survive as a real (trainable) pytree leaf on
    `White` -- previously it was hardcoded to 1.0 and silently optimised.
    """
    kernel = White()
    leaves = jax.tree_util.tree_leaves(kernel)

    # Only the (unconstrained) variance leaf should remain.
    assert len(leaves) == 1
    assert jnp.allclose(leaves[0], kernel.variance._unconstrained)


@pytest.mark.parametrize("kernel", [k[0] for k in TESTED_KERNELS])
def test_name_is_not_a_constructor_argument(kernel: type[StationaryKernel]):
    """Regression test for issue #695.

    `name` is a `ClassVar`, not a dataclass field: it must not appear in the
    pytree, and passing it as a keyword argument must fail at runtime just as
    it already did before the fix (this pins the *runtime* contract; the
    corresponding *static* contract -- that a type checker also rejects
    `kernel(name=...)` -- is exercised via pyright in issue #695's fix).
    """
    kernel_name = kernel().name
    assert isinstance(kernel_name, str)
    assert "name" not in [f.name for f in dataclasses.fields(kernel)]
    with pytest.raises(TypeError):
        kernel(name="renamed")


# ---------------------------------------------------------------------------
# PoweredExponential power is constrained to (0, 2] (#813).
# ---------------------------------------------------------------------------


def test_powered_exponential_power_is_trainable_and_bounded():
    kernel = PoweredExponential(power=1.5)
    assert isinstance(kernel.power, SigmoidBounded)
    assert jnp.allclose(val(kernel.power), 1.5)
    # A large step in the unconstrained space cannot leave the interval.
    stepped = jax.tree_util.tree_map(lambda leaf: leaf + 100.0, kernel)
    assert 0.0 < float(val(stepped.power)) <= 2.0


def test_powered_exponential_power_two_is_fixed():
    kernel = PoweredExponential(power=2.0)
    assert isinstance(kernel.power, paramax.NonTrainable)
    assert jnp.allclose(val(kernel.power), 2.0)


@pytest.mark.parametrize("power", [-1.0, 0.0, 2.5, 3.0])
def test_powered_exponential_rejects_power_outside_the_interval(power):
    with pytest.raises(ValueError, match=r"\(0, 2\]"):
        PoweredExponential(power=power)


def test_powered_exponential_rejects_an_invalid_parameter():
    with pytest.raises(ValueError, match=r"\(0, 2\]"):
        PoweredExponential(power=PositiveReal(3.0))


def test_powered_exponential_accepts_a_valid_parameter():
    power = PositiveReal(1.2)
    assert PoweredExponential(power=power).power is power


@pytest.mark.parametrize("power", [0.3, 1.0, 1.7, 2.0])
def test_powered_exponential_is_positive_definite(power):
    X = jnp.linspace(0.0, 5.0, 200)[:, None]
    gram = PoweredExponential(power=power).gram(X).as_matrix()
    assert float(jnp.linalg.eigvalsh(gram).min()) > -1e-8
