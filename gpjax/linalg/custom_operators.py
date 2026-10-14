"""Custom Lineax operators for GPJax."""

from itertools import accumulate

import jax
import jax.numpy as jnp
import lineax as lx


class BlockDiag(lx.AbstractLinearOperator):
    """Block diagonal linear operator."""

    blocks: tuple[lx.AbstractLinearOperator, ...]

    def __init__(self, blocks):
        self.blocks = tuple(blocks)

    def mv(self, x):
        sizes = [b.in_structure().shape[0] for b in self.blocks]
        splits = tuple(accumulate(sizes[:-1]))
        xs = jnp.split(x, splits)
        ys = [b.mv(xi) for b, xi in zip(self.blocks, xs, strict=False)]
        return jnp.concatenate(ys)

    def as_matrix(self):
        return jax.scipy.linalg.block_diag(*[b.as_matrix() for b in self.blocks])

    def transpose(self):
        return BlockDiag(tuple(b.transpose() for b in self.blocks))

    def in_structure(self):
        n = sum(b.in_structure().shape[0] for b in self.blocks)
        dtype = self.blocks[0].in_structure().dtype
        return jax.ShapeDtypeStruct((n,), dtype)

    def out_structure(self):
        n = sum(b.out_structure().shape[0] for b in self.blocks)
        dtype = self.blocks[0].out_structure().dtype
        return jax.ShapeDtypeStruct((n,), dtype)


class Kronecker(lx.AbstractLinearOperator):
    """Kronecker product linear operator with efficient mv via the vec trick."""

    A: lx.AbstractLinearOperator
    B: lx.AbstractLinearOperator

    def __init__(self, A, B):
        self.A = A
        self.B = B

    def mv(self, x):
        # C-order vec trick: (A kron B)x = vec_C(A @ X @ B^T)
        # where X = x.reshape(n, m) in C (row-major) order.
        # Note: row @ B^T = B @ row (as 1D vectors), so we use B.mv on rows.
        n = self.A.in_structure().shape[0]
        m = self.B.in_structure().shape[0]
        X = x.reshape(n, m)  # n x m, C-order
        # Compute A @ X by applying A.mv to each column of X
        AX = jax.vmap(self.A.mv, in_axes=1, out_axes=1)(X)
        # Compute (A @ X) @ B^T by applying B.mv to each row of AX
        AXBt = jax.vmap(self.B.mv, in_axes=0, out_axes=0)(AX)
        return AXBt.ravel()

    def as_matrix(self):
        return jnp.kron(self.A.as_matrix(), self.B.as_matrix())

    def transpose(self):
        return Kronecker(self.A.transpose(), self.B.transpose())

    def in_structure(self):
        na = self.A.in_structure().shape[0]
        nb = self.B.in_structure().shape[0]
        dtype = self.A.in_structure().dtype
        return jax.ShapeDtypeStruct((na * nb,), dtype)

    def out_structure(self):
        na = self.A.out_structure().shape[0]
        nb = self.B.out_structure().shape[0]
        dtype = self.A.out_structure().dtype
        return jax.ShapeDtypeStruct((na * nb,), dtype)


# Register tag queries for custom operators.
# Lineax uses singledispatch for is_symmetric, is_diagonal, etc.
# These must be registered so __check_init__ can run.


@lx.is_symmetric.register(BlockDiag)
def _is_symmetric_blockdiag(op):
    return all(lx.is_symmetric(b) for b in op.blocks)


@lx.is_symmetric.register(Kronecker)
def _is_symmetric_kronecker(op):
    return lx.is_symmetric(op.A) and lx.is_symmetric(op.B)


@lx.is_diagonal.register(BlockDiag)
def _is_diagonal_blockdiag(op):
    return all(lx.is_diagonal(b) for b in op.blocks)


@lx.is_diagonal.register(Kronecker)
def _is_diagonal_kronecker(op):
    return lx.is_diagonal(op.A) and lx.is_diagonal(op.B)


@lx.is_tridiagonal.register(BlockDiag)
def _is_tridiagonal_blockdiag(op):
    return all(lx.is_tridiagonal(b) for b in op.blocks)


@lx.is_tridiagonal.register(Kronecker)
def _is_tridiagonal_kronecker(op):
    return False


@lx.is_lower_triangular.register(BlockDiag)
def _is_lower_triangular_blockdiag(op):
    return all(lx.is_lower_triangular(b) for b in op.blocks)


@lx.is_lower_triangular.register(Kronecker)
def _is_lower_triangular_kronecker(op):
    return lx.is_lower_triangular(op.A) and lx.is_lower_triangular(op.B)


@lx.is_upper_triangular.register(BlockDiag)
def _is_upper_triangular_blockdiag(op):
    return all(lx.is_upper_triangular(b) for b in op.blocks)


@lx.is_upper_triangular.register(Kronecker)
def _is_upper_triangular_kronecker(op):
    return lx.is_upper_triangular(op.A) and lx.is_upper_triangular(op.B)


@lx.is_positive_semidefinite.register(BlockDiag)
def _is_psd_blockdiag(op):
    return all(lx.is_positive_semidefinite(b) for b in op.blocks)


@lx.is_positive_semidefinite.register(Kronecker)
def _is_psd_kronecker(op):
    return lx.is_positive_semidefinite(op.A) and lx.is_positive_semidefinite(op.B)


@lx.is_negative_semidefinite.register(BlockDiag)
def _is_nsd_blockdiag(op):
    return all(lx.is_negative_semidefinite(b) for b in op.blocks)


@lx.is_negative_semidefinite.register(Kronecker)
def _is_nsd_kronecker(op):
    return False


@lx.has_unit_diagonal.register(BlockDiag)
def _has_unit_diagonal_blockdiag(op):
    # Neither operator's blocks are guaranteed to have unit diagonal in
    # general, so report `False` rather than inspecting the blocks: this is
    # the same conservative default lineax's own composite operators use.
    return False


@lx.has_unit_diagonal.register(Kronecker)
def _has_unit_diagonal_kronecker(op):
    return False
