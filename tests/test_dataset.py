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

from dataclasses import is_dataclass

try:
    from beartype.roar import BeartypeCallHintParamViolation
    from jaxtyping import TypeCheckError

    ValidationErrors = (TypeError, BeartypeCallHintParamViolation, TypeCheckError)
except ImportError:
    ValidationErrors = ValueError

from gpjax.dataset import Dataset
from jax import config
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_dataset_init(n: int, in_dim: int) -> None:
    # Create dataset
    x = jnp.ones((n, in_dim))
    y = jnp.ones((n, 1))

    D = Dataset(X=x, y=y)

    # Test dataset shapes
    assert D.n == n
    assert D.in_dim == in_dim

    # Test representation
    assert (
        D.__repr__()
        == f"Dataset(Number of observations: {n} - Input dimension: {in_dim})"
    )

    # Ensure dataclass
    assert is_dataclass(D)

    # Test supervised and unsupervised
    assert Dataset(X=x, y=y).is_supervised() is True
    assert Dataset(y=y).is_unsupervised() is True

    # Check tree flatten
    assert jtu.tree_leaves(D) == [x, y]


@pytest.mark.parametrize("n", [0, 1, 3])
@pytest.mark.parametrize("n_outputs", [1, 2])
def test_unsupervised_dataset_observation_count(n, n_outputs):
    data = Dataset(y=jnp.ones((n, n_outputs)))
    assert data.is_unsupervised()
    assert data.n == n
    assert data.full_size == n
    leaves, treedef = jtu.tree_flatten(data)
    assert jtu.tree_unflatten(treedef, leaves).n == n


@pytest.mark.parametrize("n1", [1, 2, 10])
@pytest.mark.parametrize("n2", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_dataset_add(n1: int, n2: int, in_dim: int) -> None:
    # Create first dataset
    x1 = jnp.ones((n1, in_dim))
    y1 = jnp.ones((n1, 1))
    D1 = Dataset(X=x1, y=y1)

    # Create second dataset
    x2 = 2 * jnp.ones((n2, in_dim))
    y2 = 2 * jnp.ones((n2, 1))
    D2 = Dataset(X=x2, y=y2)

    # Add datasets
    D = D1 + D2

    # Test shapes
    assert D.n == n1 + n2
    assert D.in_dim == in_dim

    # Test representation
    assert (
        D.__repr__()
        == f"Dataset(Number of observations: {n1 + n2} - Input dimension: {in_dim})"
    )

    # Ensure dataclass
    assert is_dataclass(D)

    # Test supervised and unsupervised
    assert (Dataset(X=x1, y=y1) + Dataset(X=x2, y=y2)).is_supervised() is True
    assert (Dataset(y=y1) + Dataset(y=y2)).is_unsupervised() is True

    # Check tree flatten
    x = jnp.concatenate((x1, x2))
    y = jnp.concatenate((y1, y2))
    (jtu.tree_leaves(D)[0] == x).all()
    (jtu.tree_leaves(D)[1] == y).all()


# @pytest.mark.parametrize(("nx", "ny"), [(1, 2), (2, 1), (10, 5), (5, 10)])
# @pytest.mark.parametrize("in_dim", [1, 2, 10])
# def test_dataset_incorrect_lengths(nx: int, ny: int, in_dim: int) -> None:
#     # Create input and output pairs of different lengths
#     x = jnp.ones((nx, in_dim))
#     y = jnp.ones((ny, 1))

#     # Ensure error is raised upon dataset creation
#     with pytest.raises(ValidationErrors):
#         Dataset(X=x, y=y)


# @pytest.mark.parametrize("n", [1, 2, 10])
# @pytest.mark.parametrize("in_dim", [1, 2, 10])
# def test_2d_inputs(n: int, in_dim: int) -> None:
#     # Create dataset where output dimension is incorrectly not 2D
#     x = jnp.ones((n, in_dim))
#     y = jnp.ones((n,))

#     # Ensure error is raised upon dataset creation
#     with pytest.raises(ValidationErrors):
#         Dataset(X=x, y=y)

#     # Create dataset where input dimension is incorrectly not 2D
#     x = jnp.ones((n,))
#     y = jnp.ones((n, 1))

# # Ensure error is raised upon dataset creation
# with pytest.raises(ValidationErrors):
#     Dataset(X=x, y=y)


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_y_none(n: int, in_dim: int) -> None:
    # Create a dataset with no output
    x = jnp.ones((n, in_dim))
    D = Dataset(X=x)

    # Ensure is dataclass
    assert is_dataclass(D)

    # Ensure output is None
    assert D.y is None

    # Check tree flatten
    assert jtu.tree_leaves(D) == [x]


@pytest.mark.parametrize(
    ("prec_x", "prec_y"),
    [
        (jnp.float32, jnp.float64),
        (jnp.float64, jnp.float32),
        (jnp.float32, jnp.float32),
    ],
)
@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_precision_warning(
    n: int, in_dim: int, prec_x: jnp.dtype, prec_y: jnp.dtype
) -> None:
    # Create dataset
    x = jnp.ones((n, in_dim)).astype(prec_x)
    y = jnp.ones((n, 1)).astype(prec_y)

    # Check for warnings if dtypes are not float64
    expected_warnings = 0
    if prec_x != jnp.float64:
        expected_warnings += 1
    if prec_y != jnp.float64:
        expected_warnings += 1

    with pytest.warns(UserWarning, match=".* is not of type float64.*") as record:
        Dataset(X=x, y=y)

    assert len(record) == expected_warnings


def test_dataset_single_output_properties():
    """Single-output dataset has multi_output=False, num_outputs=1."""
    X = jnp.linspace(0, 1, 10).reshape(-1, 1)
    y = jnp.sin(X)
    D = Dataset(X=X, y=y)
    assert D.multi_output is False
    assert D.num_outputs == 1


def test_dataset_multi_output_properties():
    """Multi-output dataset has multi_output=True, num_outputs=P."""
    X = jnp.linspace(0, 1, 10).reshape(-1, 1)
    y = jnp.column_stack([jnp.sin(X), jnp.cos(X), X**2])
    D = Dataset(X=X, y=y)
    assert D.multi_output is True
    assert D.num_outputs == 3


def test_n_total_defaults_to_none():
    x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)
    dataset = Dataset(X=x, y=jnp.sin(x))
    assert dataset.n_total is None


def test_get_batch_stamps_n_total():
    from gpjax.fit import get_batch
    import jax
    import jax.random as jr

    x = jnp.linspace(0.0, 1.0, 25).reshape(-1, 1)
    dataset = Dataset(X=x, y=jnp.sin(x))
    batch = get_batch(dataset, batch_size=4, key=jr.key(0))
    assert batch.n == 4
    assert batch.n_total == 25

    # The stamp is static aux_data: it survives pytree operations.
    mapped = jax.tree_util.tree_map(lambda leaf: leaf, batch)
    assert mapped.n_total == 25

    # Batching a batch preserves the original full size.
    rebatch = get_batch(batch, batch_size=2, key=jr.key(1))
    assert rebatch.n_total == 25
