# Copyright 2026 The thomaspinder Contributors. All Rights Reserved.
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

from gpjax.kernels.location_functions import (
    AbstractLocationFunction,
    Constant,
    Linear,
)
from gpjax.parameters import Real
import jax
from jax import config
import jax.numpy as jnp
import paramax
import pytest

config.update("jax_enable_x64", True)

X = jnp.array([0.3, -1.2, 2.0, 0.5])


def test_constant_returns_its_value():
    assert jnp.allclose(Constant(0.7)(X), 0.7)


def test_constant_defaults_to_zero():
    assert jnp.allclose(Constant()(X), 0.0)


def test_constant_accepts_a_parameter():
    frozen = paramax.non_trainable(Real(jnp.array(1.5)))
    assert jnp.allclose(Constant(frozen)(X), 1.5)


def test_linear_starts_at_zero():
    fn = Linear(active_dims=[1, 3])
    assert jnp.allclose(fn(X), 0.0)
    assert fn.bias is None


def test_linear_reads_only_its_columns():
    fn = Linear(active_dims=[1, 3], weights=[2.0, -1.0])
    assert jnp.allclose(fn(X), 2.0 * X[1] - 1.0 * X[3])


def test_linear_intercept_starts_at_zero_and_is_trainable():
    fn = Linear(active_dims=[0], weights=[1.0], intercept=True)
    assert jnp.allclose(fn(X), X[0])
    shifted = jax.tree_util.tree_map(lambda leaf: leaf + 0.5, fn)
    assert jnp.allclose(shifted(X), 1.5 * X[0] + 0.5)


def test_linear_requires_active_dims():
    with pytest.raises(TypeError):
        Linear()  # type: ignore[call-arg]


def test_linear_rejects_empty_active_dims():
    with pytest.raises(TypeError, match="non-empty list"):
        Linear(active_dims=[])


@pytest.mark.parametrize("active_dims", [slice(None), (0, 1)])
def test_linear_rejects_active_dims_that_are_not_a_list(active_dims):
    with pytest.raises(TypeError):
        Linear(active_dims=active_dims)


def test_linear_rejects_wrong_number_of_weights():
    with pytest.raises(ValueError, match="one weight for each"):
        Linear(active_dims=[0, 1], weights=[1.0])


def test_linear_gradient_is_the_selected_columns():
    fn = Linear(active_dims=[1, 2], weights=[0.1, 0.2])
    grads = jax.grad(lambda f: f(X))(fn)
    (gradient,) = jax.tree_util.tree_leaves(grads.weights)
    assert jnp.allclose(gradient, X[jnp.array([1, 2])])


def test_location_functions_work_under_jit_and_vmap():
    fn = Linear(active_dims=[0], weights=[2.0])
    rows = jnp.stack([X, 2 * X])
    values = jax.jit(jax.vmap(fn))(rows)
    assert jnp.allclose(values, 2.0 * rows[:, 0])


def test_custom_location_function():
    class Quadratic(AbstractLocationFunction):
        def __call__(self, x):
            return jnp.sum(self.slice_input(x) ** 2)

    fn = Quadratic(active_dims=[0, 1])
    assert jnp.allclose(fn(X), X[0] ** 2 + X[1] ** 2)
