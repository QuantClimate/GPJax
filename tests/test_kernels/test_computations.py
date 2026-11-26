"""Tests for kernel computation engines: BasisFunctionComputation and EigenKernelComputation."""

from gpjax.kernels.approximations import RFF
from gpjax.kernels.computations import (
    EigenKernelComputation,
)
from gpjax.kernels.non_euclidean import GraphKernel
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
)
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# BasisFunctionComputation
# ---------------------------------------------------------------------------


class TestBasisFunctionComputation:
    """Tests for the BasisFunctionComputation engine (used by RFF kernels)."""

    @pytest.fixture(params=[RBF, Matern12, Matern32, Matern52])
    def rff_kernel(self, request):
        """Create an RFF kernel with a given base kernel."""
        base_kernel = request.param(n_dims=2)
        return RFF(base_kernel=base_kernel, num_basis_fns=50, key=jr.key(0))

    def test_compute_features_shape(self, rff_kernel):
        """Features should have shape (N, 2*num_basis_fns)."""
        x = jr.normal(jr.key(1), (10, 2))
        features = rff_kernel.compute_engine.compute_features(rff_kernel, x)
        assert features.shape == (10, 100)  # 2 * 50

    def test_compute_features_cos_sin_structure(self):
        """Features should be [cos(z), sin(z)]."""
        base = RBF(n_dims=1)
        rff = RFF(base_kernel=base, num_basis_fns=5, key=jr.key(0))
        x = jnp.array([[1.0], [2.0], [3.0]])

        features = rff.compute_engine.compute_features(rff, x)
        assert features.shape == (3, 10)  # 2 * 5

        # First half should be cos, second half should be sin
        # They should satisfy cos^2 + sin^2 = 1 for each feature pair
        cos_part = features[:, :5]
        sin_part = features[:, 5:]
        sum_sq = cos_part**2 + sin_part**2
        assert jnp.allclose(sum_sq, 1.0)

    def test_scaling(self, rff_kernel):
        """Scaling should be variance / num_basis_fns."""
        expected = rff_kernel.base_kernel.variance.unwrap() / rff_kernel.num_basis_fns
        actual = rff_kernel.compute_engine.scaling(rff_kernel)
        assert jnp.allclose(actual, expected)

    def test_gram_shape(self, rff_kernel):
        """Gram matrix should be (N, N)."""
        x = jr.normal(jr.key(1), (10, 2))
        gram = rff_kernel.gram(x)
        assert isinstance(gram, lx.AbstractLinearOperator)
        assert lx.is_positive_semidefinite(gram)
        assert gram.as_matrix().shape == (10, 10)

    def test_gram_symmetry(self, rff_kernel):
        """Gram matrix should be symmetric."""
        x = jr.normal(jr.key(1), (8, 2))
        gram = rff_kernel.gram(x).as_matrix()
        assert jnp.allclose(gram, gram.T, atol=1e-6)

    def test_gram_positive_diagonal(self, rff_kernel):
        """Diagonal of gram matrix should be non-negative."""
        x = jr.normal(jr.key(1), (8, 2))
        gram = rff_kernel.gram(x).as_matrix()
        assert jnp.all(jnp.diag(gram) >= 0.0)

    def test_cross_covariance_shape(self, rff_kernel):
        """Cross-covariance should be (N, M)."""
        x = jr.normal(jr.key(1), (10, 2))
        y = jr.normal(jr.key(2), (5, 2))
        cc = rff_kernel.compute_engine.cross_covariance(rff_kernel, x, y)
        assert cc.shape == (10, 5)

    def test_cross_covariance_self_equals_gram(self, rff_kernel):
        """cross_covariance(x, x) should equal gram matrix entries."""
        x = jr.normal(jr.key(1), (8, 2))
        gram = rff_kernel.gram(x).as_matrix()
        cc = rff_kernel.compute_engine.cross_covariance(rff_kernel, x, x)
        assert jnp.allclose(gram, cc, atol=1e-5)

    def test_gram_computes_features_exactly_once(self, rff_kernel, monkeypatch):
        """gram(x) should compute the feature matrix a single time.

        `cross_covariance(x, x)` legitimately computes the feature matrix
        twice (once per argument), but `gram` has its own single-pass
        `_gram` override precisely to avoid that redundant work. If `gram`
        ever regresses to routing through `cross_covariance`, this test
        catches it.
        """
        x = jr.normal(jr.key(1), (8, 2))
        engine = rff_kernel.compute_engine
        original_compute_features = engine.compute_features
        call_count = 0

        def counting_compute_features(kernel, x):
            nonlocal call_count
            call_count += 1
            return original_compute_features(kernel, x)

        monkeypatch.setattr(engine, "compute_features", counting_compute_features)

        rff_kernel.gram(x)

        assert call_count == 1

    def test_gram_matches_manual_feature_computation(self, rff_kernel):
        """gram(x) should equal scaling * features @ features.T exactly."""
        x = jr.normal(jr.key(1), (8, 2))
        engine = rff_kernel.compute_engine
        features = engine.compute_features(rff_kernel, x)
        expected = engine.scaling(rff_kernel) * jnp.matmul(features, features.T)

        gram = rff_kernel.gram(x).as_matrix()

        assert jnp.allclose(gram, expected)

    def test_diagonal(self, rff_kernel):
        """Diagonal should return a linear operator."""
        x = jr.normal(jr.key(1), (8, 2))
        diag = rff_kernel.compute_engine.diagonal(rff_kernel, x)
        assert isinstance(diag, lx.AbstractLinearOperator)
        assert lx.is_positive_semidefinite(diag)

    @pytest.mark.parametrize("n_points", [1, 5, 20])
    def test_varying_input_sizes(self, n_points):
        """BasisFunctionComputation should handle different input sizes."""
        base = RBF(n_dims=3)
        rff = RFF(base_kernel=base, num_basis_fns=20, key=jr.key(0))
        x = jr.normal(jr.key(1), (n_points, 3))
        gram = rff.gram(x)
        assert gram.as_matrix().shape == (n_points, n_points)

    @pytest.mark.parametrize("n_dims", [1, 3, 10])
    def test_varying_dimensions(self, n_dims):
        """BasisFunctionComputation should handle different input dimensions."""
        base = RBF(n_dims=n_dims)
        rff = RFF(base_kernel=base, num_basis_fns=20, key=jr.key(0))
        x = jr.normal(jr.key(1), (5, n_dims))
        features = rff.compute_engine.compute_features(rff, x)
        assert features.shape == (5, 40)  # 2 * 20


# ---------------------------------------------------------------------------
# EigenKernelComputation
# ---------------------------------------------------------------------------


class TestEigenKernelComputation:
    """Tests for the EigenKernelComputation engine (used by graph kernels)."""

    @pytest.fixture
    def graph_kernel(self):
        """Create a simple GraphKernel for testing."""
        n_vertices = 10
        graph = nx.path_graph(n_vertices)
        laplacian = nx.laplacian_matrix(graph).toarray() + jnp.eye(n_vertices) * 1e-12
        return GraphKernel(laplacian=laplacian)

    def test_engine_type(self, graph_kernel):
        """GraphKernel should use EigenKernelComputation."""
        assert isinstance(graph_kernel.compute_engine, EigenKernelComputation)

    def test_cross_covariance_delegates_to_kernel(self, graph_kernel):
        """EigenKernelComputation._cross_covariance delegates to kernel.__call__."""
        x = jnp.arange(5).reshape(-1, 1)
        y = jnp.arange(3).reshape(-1, 1)

        cc = graph_kernel.compute_engine.cross_covariance(graph_kernel, x, y)
        direct = graph_kernel(x, y)

        assert cc.shape == (5, 3)
        assert jnp.allclose(cc, direct)

    def test_gram_returns_psd(self, graph_kernel):
        """Gram should return PSD linear operator."""
        x = jnp.arange(10).reshape(-1, 1)
        gram = graph_kernel.gram(x)
        assert isinstance(gram, lx.AbstractLinearOperator)
        assert lx.is_positive_semidefinite(gram)

    def test_gram_symmetry(self, graph_kernel):
        """Gram matrix should be symmetric."""
        x = jnp.arange(10).reshape(-1, 1)
        gram = graph_kernel.gram(x).as_matrix()
        assert jnp.allclose(gram, gram.T, atol=1e-6)
