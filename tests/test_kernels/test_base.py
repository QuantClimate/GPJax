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

import equinox as eqx
from gpjax.kernels.base import (
    AbstractKernel,
    CombinationKernel,
    ProductKernel,
    SumKernel,
)
from gpjax.kernels.nonstationary import (
    Linear,
    Polynomial,
)
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
)
from gpjax.parameters import (
    PositiveReal,
    Real,
)
from jax import config
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

TESTED_KERNELS = [
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Polynomial,
    Linear,
    RationalQuadratic,
]


@pytest.mark.parametrize("kernel", TESTED_KERNELS)
@pytest.mark.parametrize(
    ("active_dims", "n_dims"),
    (p := [([3], 1), ([2, 3, 4], 3), (slice(1, 3), 2), (None, None)]),
    ids=[f"active_dims={x[0]}-n_dims={x[1]}" for x in p],
)
def test_init_dims(kernel: type[AbstractKernel], active_dims, n_dims):
    # initialize with active_dims, check if n_dims is inferred correctly
    k = kernel(active_dims=active_dims)
    assert k.active_dims == active_dims or slice(None)
    assert k.n_dims == n_dims

    # initialize with n_dims, check that active_dims is set to full slice
    k = kernel(n_dims=n_dims)
    assert k.active_dims == slice(None)
    assert k.n_dims == n_dims

    # initialize with both, no errors should be raised for mismatch
    k = kernel(active_dims=active_dims, n_dims=n_dims)
    assert k.active_dims == active_dims or slice(None)
    assert k.n_dims == n_dims

    # test that error is raised if they are incompatible
    with pytest.raises(ValueError, match="length of active_dims"):
        kernel(active_dims=[3], n_dims=2)

    with pytest.raises(ValueError, match="length of active_dims"):
        kernel(active_dims=slice(2), n_dims=1)

    # test that error is raised if types are wrong
    with pytest.raises(TypeError):
        kernel(active_dims="3", n_dims=2)

    with pytest.raises(TypeError):
        kernel(active_dims=[3], n_dims="2")


@pytest.mark.parametrize("combination_type", [SumKernel, ProductKernel])
@pytest.mark.parametrize("kernel", TESTED_KERNELS)
@pytest.mark.parametrize("n_kerns", [2, 3, 4])
def test_combination_kernel(
    combination_type: type[CombinationKernel],
    kernel: type[AbstractKernel],
    n_kerns: int,
) -> None:
    # Create inputs
    n = 20
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)

    # Create list of kernels
    kernels = [kernel() for _ in range(n_kerns)]

    # Create combination kernel
    combination_kernel = combination_type(kernels=kernels)

    # Check kernels stored as tuple
    assert isinstance(combination_kernel.kernels, tuple)
    assert len(combination_kernel.kernels) == n_kerns
    assert isinstance(combination_kernel.kernels[0], AbstractKernel)

    # Compute gram matrix
    Kxx = combination_kernel.gram(x)

    # Check shapes
    assert Kxx.as_matrix().shape[0] == Kxx.as_matrix().shape[1]
    assert Kxx.as_matrix().shape[1] == n

    # Check positive definiteness
    jitter = 1e-6
    eigen_values = jnp.linalg.eigvalsh(Kxx.as_matrix() + jnp.eye(n) * jitter)
    assert (eigen_values > 0).all()


@pytest.mark.parametrize("k1", TESTED_KERNELS)
@pytest.mark.parametrize("k2", TESTED_KERNELS)
def test_sum_kern_value(k1: type[AbstractKernel], k2: type[AbstractKernel]) -> None:
    k1 = k1()
    k2 = k2()

    # Create inputs
    n = 10
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)

    # Create sum kernel
    sum_kernel = SumKernel(kernels=[k1, k2])

    # Compute gram matrix
    Kxx = sum_kernel.gram(x)

    # Compute gram matrix
    Kxx_k1 = k1.gram(x)
    Kxx_k2 = k2.gram(x)

    # Check manual and automatic gram matrices are equal
    assert jnp.all(Kxx.as_matrix() == Kxx_k1.as_matrix() + Kxx_k2.as_matrix())


@pytest.mark.parametrize("k1", TESTED_KERNELS)
@pytest.mark.parametrize("k2", TESTED_KERNELS)
def test_prod_kern_value(k1: AbstractKernel, k2: AbstractKernel) -> None:
    k1 = k1()
    k2 = k2()

    # Create inputs
    n = 10
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)

    # Create product kernel
    prod_kernel = ProductKernel(kernels=[k1, k2])

    # Compute gram matrix
    Kxx = prod_kernel.gram(x)

    # Compute gram matrix
    Kxx_k1 = k1.gram(x)
    Kxx_k2 = k2.gram(x)

    # Check manual and automatic gram matrices are equal
    assert jnp.all(Kxx.as_matrix() == Kxx_k1.as_matrix() * Kxx_k2.as_matrix())


def test_kernel_subclassing():
    # Test initialising abstract kernel raises TypeError with unimplemented __call__ method:
    with pytest.raises(TypeError):
        AbstractKernel()

    # Create a dummy kernel class with __call__ implemented:
    class DummyKernel(AbstractKernel):
        test_a: Real
        test_b: PositiveReal

        def __init__(
            self,
            active_dims=None,
            test_a: Float[Array, "1"] = jnp.array([1.0]),
            test_b: Float[Array, "1"] = jnp.array([2.0]),
        ):
            self.test_a = Real(test_a)
            self.test_b = PositiveReal(test_b)
            super().__init__(active_dims)

        def __call__(
            self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
        ) -> Float[Array, "1"]:
            return x * self.test_b.unwrap() * y

    # Initialise dummy kernel class and test __call__ method:
    dummy_kernel = DummyKernel()
    assert dummy_kernel.test_a.unwrap() == jnp.array([1.0])
    assert dummy_kernel.test_b.unwrap() == jnp.array([2.0])
    assert dummy_kernel(jnp.array([1.0]), jnp.array([2.0])) == 4.0


def test_constant_kernel_is_eqx_module():
    from gpjax.kernels.base import Constant

    k = Constant()
    assert isinstance(k, eqx.Module)


def test_sum_kernel_is_concrete_class():
    """SumKernel should be a class, not ft.partial."""
    from gpjax.kernels.base import Constant

    assert isinstance(SumKernel, type)
    k1 = Constant(constant=1.0)
    k2 = Constant(constant=2.0)
    sk = k1 + k2
    assert isinstance(sk, SumKernel)


def test_product_kernel_is_concrete_class():
    from gpjax.kernels.base import Constant

    assert isinstance(ProductKernel, type)
    k1 = Constant(constant=1.0)
    k2 = Constant(constant=2.0)
    pk = k1 * k2
    assert isinstance(pk, ProductKernel)


def test_combination_kernel_flattens_same_type():
    """(a + b) + c should flatten to [a, b, c], not nest."""
    from gpjax.kernels.base import Constant

    k1 = Constant(constant=1.0)
    k2 = Constant(constant=2.0)
    k3 = Constant(constant=3.0)
    sk = (k1 + k2) + k3
    assert len(sk.kernels) == 3


def test_combination_kernel_uses_tuples():
    from gpjax.kernels.base import Constant

    k1 = Constant(constant=1.0)
    k2 = Constant(constant=2.0)
    sk = k1 + k2
    assert isinstance(sk.kernels, tuple)


def test_nested_sum_of_product_value() -> None:
    """Test that SumKernel preserves a nested ProductKernel."""
    x = jnp.array([1.0, 2.0])

    k1 = RBF()
    k2 = Linear()
    k3 = RBF()

    # Build nested combination: k1 + (k2 * k3)
    k_prod = ProductKernel(kernels=[k2, k3])
    k_sum = SumKernel(kernels=[k1, k_prod])

    # The product kernel should NOT be flattened into the sum
    expected = k1(x, x) + k2(x, x) * k3(x, x)
    assert jnp.isclose(k_sum(x, x), expected)


def test_nested_product_of_sum_value() -> None:
    """Test that ProductKernel preserves a nested SumKernel."""
    x = jnp.array([1.0, 2.0])

    k1 = RBF()
    k2 = Linear()
    k3 = RBF()

    # Build nested combination: k1 * (k2 + k3)
    k_sum = SumKernel(kernels=[k2, k3])
    k_prod = ProductKernel(kernels=[k1, k_sum])

    # The sum kernel should NOT be flattened into the product
    expected = k1(x, x) * (k2(x, x) + k3(x, x))
    assert jnp.isclose(k_prod(x, x), expected)


def test_same_type_flattening_preserved() -> None:
    """Test that same-type nesting is still flattened correctly."""
    k1 = RBF()
    k2 = Linear()
    k3 = Matern12()

    # SumKernel([SumKernel([k1, k2]), k3]) should flatten to SumKernel([k1, k2, k3])
    k_inner_sum = SumKernel(kernels=[k1, k2])
    k_outer_sum = SumKernel(kernels=[k_inner_sum, k3])
    assert len(k_outer_sum.kernels) == 3

    # ProductKernel([ProductKernel([k1, k2]), k3]) should flatten to ProductKernel([k1, k2, k3])
    k_inner_prod = ProductKernel(kernels=[k1, k2])
    k_outer_prod = ProductKernel(kernels=[k_inner_prod, k3])
    assert len(k_outer_prod.kernels) == 3


def test_nested_combination_kernel_structure() -> None:
    """Test that nested combination kernels of different type are not flattened."""
    k1 = RBF()
    k2 = Linear()
    k3 = Matern12()

    # SumKernel([k1, ProductKernel([k2, k3])]) should keep the ProductKernel intact
    k_prod = ProductKernel(kernels=[k2, k3])
    k_sum = SumKernel(kernels=[k1, k_prod])
    assert len(k_sum.kernels) == 2
    assert isinstance(k_sum.kernels[1], CombinationKernel)

    # ProductKernel([k1, SumKernel([k2, k3])]) should keep the SumKernel intact
    k_sum2 = SumKernel(kernels=[k2, k3])
    k_prod2 = ProductKernel(kernels=[k1, k_sum2])
    assert len(k_prod2.kernels) == 2
    assert isinstance(k_prod2.kernels[1], CombinationKernel)


def test_nested_combination_via_operators() -> None:
    """Test that operator overloading preserves nested combination structure."""
    x = jnp.array([1.0, 2.0])

    k1 = RBF()
    k2 = Linear()
    k3 = RBF()

    # k1 + k2 * k3 should compute k1(x,x) + k2(x,x)*k3(x,x)
    k_combined = k1 + k2 * k3
    expected = k1(x, x) + k2(x, x) * k3(x, x)
    assert jnp.isclose(k_combined(x, x), expected)


@pytest.mark.parametrize("k1", TESTED_KERNELS)
@pytest.mark.parametrize("k2", TESTED_KERNELS)
def test_nested_sum_of_product_gram(
    k1: type[AbstractKernel], k2: type[AbstractKernel]
) -> None:
    k1 = k1()
    k2 = k2()
    k3 = RBF()

    # Create inputs
    n = 10
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)

    # Create nested kernel: k1 + (k2 * k3)
    k_prod = ProductKernel(kernels=[k2, k3])
    k_sum = SumKernel(kernels=[k1, k_prod])

    # Compute gram matrix
    Kxx = k_sum.gram(x)

    # Compute expected gram matrix manually
    Kxx_k1 = k1.gram(x).as_matrix()
    Kxx_k2 = k2.gram(x).as_matrix()
    Kxx_k3 = k3.gram(x).as_matrix()
    Kxx_expected = Kxx_k1 + Kxx_k2 * Kxx_k3

    assert jnp.allclose(Kxx.as_matrix(), Kxx_expected)
