"""Tests for the Lineax-based linear algebra module."""

from gpjax.linalg import add_jitter, cholesky_factor, logdet, logdet_from_factor
from gpjax.linalg.custom_operators import BlockDiag, Kronecker
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

# --- cholesky_factor tests ---


def test_cholesky_factor_dense():
    A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
    op = lx.MatrixLinearOperator(A)
    L_op = cholesky_factor(op)
    L = L_op.as_matrix()
    assert jnp.allclose(L @ L.T, A, atol=1e-5)
    assert jnp.allclose(L, jnp.tril(L))


def test_cholesky_factor_diagonal():
    d = jnp.array([4.0, 9.0, 16.0])
    op = lx.DiagonalLinearOperator(d)
    L_op = cholesky_factor(op)
    assert isinstance(L_op, lx.DiagonalLinearOperator)
    assert jnp.allclose(lx.diagonal(L_op), jnp.sqrt(d))


def test_cholesky_factor_identity():
    op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float64))
    L_op = cholesky_factor(op)
    assert isinstance(L_op, lx.IdentityLinearOperator)


# --- logdet tests ---


def test_logdet_dense():
    A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
    op = lx.MatrixLinearOperator(A)
    expected = jnp.log(jnp.linalg.det(A))
    assert jnp.allclose(logdet(op), expected, atol=1e-5)


def test_logdet_diagonal():
    d = jnp.array([2.0, 3.0, 5.0])
    op = lx.DiagonalLinearOperator(d)
    assert jnp.allclose(logdet(op), jnp.sum(jnp.log(d)))


def test_logdet_identity():
    op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((4,), jnp.float64))
    assert jnp.allclose(logdet(op), 0.0)


# --- logdet_from_factor tests ---
#
# `logdet_from_factor(L)` computes log|A| for A = L Lᵀ, reusing a Cholesky
# factor the caller already holds (issue #664). It must agree with `logdet`
# and must not densify structured operators.


def _blockdiag_case():
    b1 = jnp.array(
        [
            [5.88278148942421, 1.8045229252228703],
            [1.8045229252228703, 4.877985143235627],
        ]
    )
    b2 = jnp.array(
        [
            [6.8994720881809375, -1.2155970227187396, -0.7818214380109865],
            [-1.2155970227187396, 3.727962808803854, 0.4565422704125992],
            [-0.7818214380109865, 0.4565422704125992, 3.355979468413311],
        ]
    )
    return b1, b2


def test_logdet_from_factor_dense():
    A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
    op = lx.MatrixLinearOperator(A)
    result = logdet_from_factor(cholesky_factor(op))
    assert jnp.allclose(result, jnp.log(jnp.linalg.det(A)), atol=1e-10)
    assert jnp.allclose(result, logdet(op), atol=1e-10)


def test_logdet_from_factor_diagonal():
    d = jnp.array([2.0, 3.0, 5.0])
    op = lx.DiagonalLinearOperator(d)
    result = logdet_from_factor(cholesky_factor(op))
    assert jnp.allclose(result, jnp.sum(jnp.log(d)), atol=1e-10)
    assert jnp.allclose(result, logdet(op), atol=1e-10)


def test_logdet_from_factor_identity():
    op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((4,), jnp.float64))
    assert jnp.allclose(logdet_from_factor(cholesky_factor(op)), 0.0)


def test_logdet_from_factor_blockdiag():
    b1, b2 = _blockdiag_case()
    op = BlockDiag([lx.MatrixLinearOperator(b1), lx.MatrixLinearOperator(b2)])
    result = logdet_from_factor(cholesky_factor(op))
    assert jnp.allclose(result, 7.599554710816069, atol=1e-10)
    assert jnp.allclose(result, logdet(op), atol=1e-10)


def test_logdet_from_factor_kronecker():
    b1, b2 = _blockdiag_case()
    op = Kronecker(lx.MatrixLinearOperator(b1), lx.MatrixLinearOperator(b2))
    result = logdet_from_factor(cholesky_factor(op))
    assert jnp.allclose(result, 18.435424994931296, atol=1e-10)
    assert jnp.allclose(result, logdet(op), atol=1e-10)


def test_logdet_from_factor_diagonal_does_not_densify(monkeypatch):
    """The diagonal fast path must never materialise an N×N matrix."""

    def forbidden(self):
        raise AssertionError("as_matrix() called: diagonal fast path densified")

    monkeypatch.setattr(lx.DiagonalLinearOperator, "as_matrix", forbidden)
    factor = lx.DiagonalLinearOperator(jnp.array([2.0, 3.0, 5.0]))
    result = logdet_from_factor(factor)
    assert jnp.allclose(result, 2.0 * jnp.sum(jnp.log(jnp.array([2.0, 3.0, 5.0]))))


def test_logdet_from_factor_blockdiag_does_not_densify(monkeypatch):
    """Block-diagonal structure must be exploited, not flattened."""

    def forbidden(self):
        raise AssertionError("as_matrix() called: block-diagonal structure lost")

    monkeypatch.setattr(BlockDiag, "as_matrix", forbidden)
    b1, b2 = _blockdiag_case()
    factor = cholesky_factor(
        BlockDiag([lx.MatrixLinearOperator(b1), lx.MatrixLinearOperator(b2)])
    )
    assert jnp.allclose(logdet_from_factor(factor), 7.599554710816069, atol=1e-10)


def test_logdet_from_factor_kronecker_does_not_densify(monkeypatch):
    """Kronecker structure must be exploited, not expanded to the full product."""

    def forbidden(self):
        raise AssertionError("as_matrix() called: Kronecker structure lost")

    monkeypatch.setattr(Kronecker, "as_matrix", forbidden)
    b1, b2 = _blockdiag_case()
    factor = cholesky_factor(
        Kronecker(lx.MatrixLinearOperator(b1), lx.MatrixLinearOperator(b2))
    )
    assert jnp.allclose(logdet_from_factor(factor), 18.435424994931296, atol=1e-10)


def test_logdet_diagonal_does_not_densify(monkeypatch):
    """`logdet`'s own diagonal fast path is unchanged by the #664 refactor."""

    def forbidden(self):
        raise AssertionError("as_matrix() called: logdet diagonal fast path densified")

    monkeypatch.setattr(lx.DiagonalLinearOperator, "as_matrix", forbidden)
    d = jnp.array([2.0, 3.0, 5.0])
    assert jnp.allclose(logdet(lx.DiagonalLinearOperator(d)), jnp.sum(jnp.log(d)))


def test_logdet_from_factor_is_jittable():
    A = jnp.array([[4.0, 2.0], [2.0, 3.0]])

    def fn(matrix):
        return logdet_from_factor(cholesky_factor(lx.MatrixLinearOperator(matrix)))

    assert jnp.allclose(jax.jit(fn)(A), jnp.log(jnp.linalg.det(A)), atol=1e-10)
    assert jnp.all(jnp.isfinite(jax.grad(fn)(A)))


# --- add_jitter tests ---


def test_add_jitter():
    m = jnp.eye(3)
    result = add_jitter(m, 0.1)
    assert jnp.allclose(jnp.diag(result), 1.1)


def test_add_jitter_non_square_raises():
    with pytest.raises(ValueError, match="square"):
        add_jitter(jnp.ones((2, 3)))


def test_add_jitter_non_2d_raises():
    with pytest.raises(ValueError, match="2D"):
        add_jitter(jnp.ones((2,)))


# --- BlockDiag tests ---


def test_block_diag_mv():
    A = lx.MatrixLinearOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    B = lx.MatrixLinearOperator(jnp.array([[5.0]]))
    bd = BlockDiag(blocks=(A, B))
    x = jnp.array([1.0, 0.0, 2.0])
    result = bd.mv(x)
    expected = jnp.array([1.0, 3.0, 10.0])
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "block_shapes",
    [((2, 2), (3, 3)), ((2, 3), (3, 2)), ((2, 3),), ((1, 2), (3, 1), (2, 4))],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("use_jit", [False, True])
def test_block_diag_mv_matches_dense_value_and_gradients(block_shapes, dtype, use_jit):
    matrices = tuple(
        jnp.arange(rows * cols, dtype=dtype).reshape(rows, cols) + index + 1
        for index, (rows, cols) in enumerate(block_shapes)
    )
    x = jnp.arange(sum(cols for _, cols in block_shapes), dtype=dtype) + 0.5
    weights = jnp.arange(sum(rows for rows, _ in block_shapes), dtype=dtype) + 1

    def apply(blocks, vector):
        operator = BlockDiag(tuple(lx.MatrixLinearOperator(block) for block in blocks))
        return operator.mv(vector)

    def loss(blocks, vector):
        return jnp.dot(weights, apply(blocks, vector))

    def reference_loss(blocks, vector):
        return jnp.dot(weights, jax.scipy.linalg.block_diag(*blocks) @ vector)

    evaluate = jax.jit(apply) if use_jit else apply
    result = evaluate(matrices, x)
    expected = jax.scipy.linalg.block_diag(*matrices) @ x
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    differentiate = jax.grad(loss, argnums=(0, 1))
    if use_jit:
        differentiate = jax.jit(differentiate)
    actual_gradients = differentiate(matrices, x)
    expected_gradients = jax.grad(reference_loss, argnums=(0, 1))(matrices, x)
    for actual, expected in zip(
        jax.tree.leaves(actual_gradients),
        jax.tree.leaves(expected_gradients),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_block_diag_as_matrix():
    A = lx.MatrixLinearOperator(jnp.eye(2))
    B = lx.MatrixLinearOperator(2.0 * jnp.eye(3))
    bd = BlockDiag(blocks=(A, B))
    mat = bd.as_matrix()
    assert mat.shape == (5, 5)
    expected = jax.scipy.linalg.block_diag(jnp.eye(2), 2.0 * jnp.eye(3))
    assert jnp.allclose(mat, expected)


def test_block_diag_structures():
    A = lx.MatrixLinearOperator(jnp.eye(2))
    B = lx.MatrixLinearOperator(jnp.eye(3))
    bd = BlockDiag(blocks=(A, B))
    assert bd.in_structure().shape == (5,)
    assert bd.out_structure().shape == (5,)


# --- Kronecker tests ---


def test_kronecker_mv():
    A = lx.MatrixLinearOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    B = lx.MatrixLinearOperator(jnp.array([[5.0, 6.0], [7.0, 8.0]]))
    kron = Kronecker(A=A, B=B)
    x = jnp.ones(4)
    expected = jnp.kron(A.as_matrix(), B.as_matrix()) @ x
    result = kron.mv(x)
    assert jnp.allclose(result, expected, atol=1e-5)


def test_kronecker_mv_rectangular():
    A = lx.MatrixLinearOperator(jnp.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))
    B = lx.MatrixLinearOperator(jnp.array([[5.0, 6.0], [7.0, 8.0]]))
    kron = Kronecker(A=A, B=B)
    x = jnp.ones(6)
    expected = jnp.kron(A.as_matrix(), B.as_matrix()) @ x
    result = kron.mv(x)
    assert jnp.allclose(result, expected, atol=1e-5)


def test_kronecker_as_matrix():
    A = lx.MatrixLinearOperator(jnp.array([[1.0, 0.0], [0.0, 2.0]]))
    B = lx.MatrixLinearOperator(jnp.eye(3))
    kron = Kronecker(A=A, B=B)
    mat = kron.as_matrix()
    expected = jnp.kron(A.as_matrix(), B.as_matrix())
    assert jnp.allclose(mat, expected)


def test_kronecker_structures():
    A = lx.MatrixLinearOperator(jnp.eye(2))
    B = lx.MatrixLinearOperator(jnp.eye(3))
    kron = Kronecker(A=A, B=B)
    assert kron.in_structure().shape == (6,)
    assert kron.out_structure().shape == (6,)


def test_stabilised_cholesky_identity():
    from gpjax.linalg.utils import stabilised_cholesky

    factor = stabilised_cholesky(jnp.eye(3), 1e-2)
    assert jnp.allclose(factor, jnp.sqrt(1.01) * jnp.eye(3), atol=1e-12)


def test_stabilised_cholesky_reconstructs():
    from gpjax.linalg.utils import stabilised_cholesky

    root = jnp.array([[1.0, 0.0], [0.4, 0.8]])
    psd = root @ root.T
    factor = stabilised_cholesky(psd, 1e-3)
    assert jnp.allclose(factor @ factor.T, psd + 1e-3 * jnp.eye(2), atol=1e-10)
