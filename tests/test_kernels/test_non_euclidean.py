# # Copyright 2022 The thomaspinder Contributors. All Rights Reserved.
# #
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

from gpjax.kernels.non_euclidean import GraphKernel
from gpjax.kernels.non_euclidean.utils import (
    calculate_heat_semigroup,
    jax_gather_nd,
)
from jax import (
    config,
    jit,
    vmap,
)
import jax.numpy as jnp
import networkx as nx
import pytest

# # Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_graph_kernel():
    # Create a random graph, G, and verice labels, x,
    n_verticies = 20
    n_edges = 40
    G = nx.gnm_random_graph(n_verticies, n_edges, seed=123)
    x = jnp.arange(n_verticies).reshape(-1, 1)

    # Compute graph laplacian
    L = nx.laplacian_matrix(G).toarray() + jnp.eye(n_verticies) * 1e-12

    # Create graph kernel
    kern = GraphKernel(
        laplacian=L,
    )
    assert isinstance(kern, GraphKernel)
    assert kern.num_vertex == n_verticies
    assert kern.eigenvalues.unwrap().shape == (n_verticies, 1)
    assert kern.eigenvectors.unwrap().shape == (n_verticies, n_verticies)

    # Compute gram matrix
    Kxx = kern.gram(x)
    assert Kxx.as_matrix().shape == (n_verticies, n_verticies)

    # Check positive definiteness
    Kxx_dense = Kxx.as_matrix() + jnp.eye(n_verticies) * 1e-6
    eigen_values = jnp.linalg.eigvalsh(Kxx_dense)
    assert all(eigen_values > 0)


def _build_test_kernel(n_vertices: int = 10) -> GraphKernel:
    graph = nx.path_graph(n_vertices)
    laplacian = nx.laplacian_matrix(graph).toarray() + jnp.eye(n_vertices) * 1e-12
    return GraphKernel(laplacian=laplacian)


def test_graph_kernel_accepts_vector_indices():
    kernel = _build_test_kernel()
    col_indices = jnp.arange(5).reshape(-1, 1)
    vector_indices = col_indices.squeeze()

    matrix_eval = kernel(col_indices, col_indices)
    vector_eval = kernel(vector_indices, vector_indices)

    assert matrix_eval.shape == (5, 5)
    assert vector_eval.shape == (5, 5)
    assert jnp.allclose(matrix_eval, vector_eval)


def test_graph_kernel_vmappable_over_single_indices():
    kernel = _build_test_kernel()
    idx = jnp.arange(4)

    diag_entries = vmap(lambda z: kernel(z, z))(idx)
    assert diag_entries.shape == (4,)
    assert jnp.all(diag_entries >= 0.0)


def test_graph_kernel_vmappable_over_pairs():
    kernel = _build_test_kernel()
    x = jnp.arange(5)
    y = jnp.array([4, 3, 2, 1, 0])

    vectorised_eval = vmap(lambda a, b: kernel(a, b))(x, y)
    baseline = jnp.asarray([kernel(a, b) for a, b in zip(x, y, strict=False)])

    assert vectorised_eval.shape == (5,)
    assert jnp.allclose(vectorised_eval, baseline)


def test_graph_kernel_is_jittable():
    kernel = _build_test_kernel()
    jit_kernel = jit(lambda a, b: kernel(a, b))

    column = jnp.arange(5).reshape(-1, 1)
    vector = jnp.arange(5)
    pairs = [
        (0, 0),
        (0, column),
        (vector, 1),
        (vector, vector),
        (column, column),
    ]

    for x, y in pairs:
        expected = kernel(x, y)
        result = jit_kernel(x, y)
        assert result.shape == expected.shape
        assert jnp.allclose(result, expected)


# ---------------------------------------------------------------------------
# jax_gather_nd utility
# ---------------------------------------------------------------------------


class TestJaxGatherNd:
    """Tests for jax_gather_nd utility function."""

    def test_basic_1d_gather(self):
        params = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        indices = jnp.array([[0], [2], [4]])
        result = jax_gather_nd(params, indices)
        assert jnp.allclose(result, jnp.array([10.0, 30.0, 50.0]))

    def test_2d_gather_rows(self):
        params = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        indices = jnp.array([[0], [2]])
        result = jax_gather_nd(params, indices)
        expected = jnp.array([[1.0, 2.0], [5.0, 6.0]])
        assert jnp.allclose(result, expected)

    def test_single_index(self):
        params = jnp.arange(10.0)
        indices = jnp.array([[3]])
        result = jax_gather_nd(params, indices)
        assert jnp.allclose(result, jnp.array([3.0]))

    def test_all_indices(self):
        params = jnp.array([100.0, 200.0, 300.0])
        indices = jnp.arange(3).reshape(-1, 1)
        result = jax_gather_nd(params, indices)
        assert jnp.allclose(result, params)

    def test_repeated_indices(self):
        params = jnp.array([10.0, 20.0, 30.0])
        indices = jnp.array([[1], [1], [1]])
        result = jax_gather_nd(params, indices)
        assert jnp.allclose(result, jnp.array([20.0, 20.0, 20.0]))


# ---------------------------------------------------------------------------
# calculate_heat_semigroup utility
# ---------------------------------------------------------------------------


class TestCalculateHeatSemigroup:
    """Tests for calculate_heat_semigroup utility function."""

    @pytest.fixture
    def graph_kernel(self):
        n_vertices = 10
        graph = nx.path_graph(n_vertices)
        laplacian = nx.laplacian_matrix(graph).toarray() + jnp.eye(n_vertices) * 1e-12
        return GraphKernel(laplacian=laplacian)

    def test_output_shape(self, graph_kernel):
        S = calculate_heat_semigroup(graph_kernel)
        assert S.shape == (graph_kernel.num_vertex, 1)

    def test_finite_values(self, graph_kernel):
        S = calculate_heat_semigroup(graph_kernel)
        assert jnp.all(jnp.isfinite(S))

    def test_positive_values(self, graph_kernel):
        S = calculate_heat_semigroup(graph_kernel)
        assert jnp.all(S > 0.0)

    def test_scales_with_variance(self):
        """Doubling kernel variance should double the semigroup values."""
        n_vertices = 8
        graph = nx.path_graph(n_vertices)
        laplacian = nx.laplacian_matrix(graph).toarray() + jnp.eye(n_vertices) * 1e-12

        k1 = GraphKernel(laplacian=laplacian, variance=1.0)
        k2 = GraphKernel(laplacian=laplacian, variance=2.0)

        S1 = calculate_heat_semigroup(k1)
        S2 = calculate_heat_semigroup(k2)
        assert jnp.allclose(S2, 2.0 * S1, atol=1e-6)

    def test_sums_to_num_vertex_times_variance(self, graph_kernel):
        """The sum of S should equal num_vertex * variance (by construction)."""
        S = calculate_heat_semigroup(graph_kernel)
        expected_sum = graph_kernel.num_vertex * graph_kernel.variance.unwrap()
        assert jnp.allclose(jnp.sum(S), expected_sum, atol=1e-5)
