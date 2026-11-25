"""Orthogonal Additive Kernel (OAK).

Reference:
    Lu, X., Boukouvalas, A., & Hensman, J. (2022).
    Additive Gaussian Processes Revisited. ICML.
"""

import typing as tp

import equinox as eqx
import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.base import AbstractKernel, val
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.parameters import NonNegativeReal
from gpjax.typing import Array, ScalarFloat


def _constrained_se_kernel(
    x: Float[Array, ""],
    y: Float[Array, ""],
    lengthscale: Float[Array, ""],
    variance: Float[Array, ""],
) -> ScalarFloat:
    """Compute the constrained SE kernel under standard normal input density.

    Given base SE kernel k(x,y) = sigma^2 exp(-(x-y)^2 / (2l^2)),
    the constrained kernel is k_tilde(x,y) = k(x,y) - k_hat(x,y)
    where k_hat is the projection term ensuring orthogonality
    w.r.t. N(0,1) input density.

    Args:
        x: First scalar input.
        y: Second scalar input.
        lengthscale: Kernel lengthscale l.
        variance: Kernel variance sigma^2.

    Returns:
        Scalar constrained kernel value.
    """
    ell_sq = jnp.square(lengthscale)

    # Base SE kernel: k(x, y) = sigma^2 * exp(-(x - y)^2 / (2 * l^2))
    k_base = variance * jnp.exp(-0.5 * jnp.square(x - y) / ell_sq)

    # Projection term (Eq. 10 of Lu et al. 2022 with mu=0, delta^2=1):
    # k_hat(x, y) = sigma^2 * l * sqrt(l^2 + 2) / (l^2 + 1)
    #               * exp(-(x^2 + y^2) / (2(l^2 + 1)))
    projection_coeff = variance * lengthscale * jnp.sqrt(ell_sq + 2.0) / (ell_sq + 1.0)
    k_hat = projection_coeff * jnp.exp(
        -(jnp.square(x) + jnp.square(y)) / (2.0 * (ell_sq + 1.0))
    )

    return k_base - k_hat


def _newton_girard(
    z: Float[Array, " D *trailing"],
    max_order: int,
) -> Float[Array, " D_tilde_plus_1 *trailing"]:
    """Compute elementary symmetric polynomials via Newton-Girard recursion.

    Given values z_1, ..., z_D, computes e_0, e_1, ..., e_{max_order} where:
    - e_0 = 1
    - e_1 = sum(z_i)
    - e_2 = sum_{i<j} z_i * z_j
    - e_n = (1/n) sum_{k=1}^{n} (-1)^{k-1} e_{n-k} s_k

    and s_k = sum(z_i^k) are power sums.

    Each z_i may itself be an array of arbitrary trailing shape (e.g. a
    scalar per dimension for OAK's kernel evaluation, or an (N, N) matrix
    per dimension for Sobol index computation): all arithmetic broadcasts
    element-wise over the trailing dimensions, so the same recursion serves
    both call sites.

    Uses jax.lax.fori_loop for JAX compatibility (no Python for loops).

    Args:
        z: Array of shape (D, *trailing) -- one value per dimension, each
            value itself an array of arbitrary (possibly empty) trailing
            shape.
        max_order: Maximum order of elementary symmetric polynomial to compute.

    Returns:
        Array of shape (max_order + 1, *trailing) containing e_0 through
        e_{max_order}.
    """
    trailing_ndim = z.ndim - 1

    # Power sums: s[k, ...] = sum_{d=1}^D z_d[...]^{k+1}  (vectorised over k)
    exponents = jnp.arange(1, max_order + 1).reshape((-1,) + (1,) * z.ndim)
    power_sums = jnp.sum(z[None] ** exponents, axis=1)

    # Precompute sign-alternating power sums: (-1)^{k-1} * s_k
    signs = (-1.0) ** jnp.arange(max_order)  # [+1, -1, +1, -1, ...]
    signed_power_sums = signs.reshape((-1,) + (1,) * trailing_ndim) * power_sums

    elem_sym = jnp.zeros((max_order + 1, *z.shape[1:]))
    elem_sym = elem_sym.at[0].set(1.0)

    def _recursion_step(order, elem_sym):
        # e_order = (1/order) * sum_{k=1}^{order} (-1)^{k-1} * e[order-k] * s[k-1]
        k_indices = jnp.arange(max_order)
        lookback_indices = (order - 1 - k_indices).clip(0)
        mask = (k_indices < order).reshape((-1,) + (1,) * trailing_ndim)
        previous_values = jnp.where(mask, elem_sym[lookback_indices], 0.0)
        value = jnp.sum(previous_values * signed_power_sums, axis=0) / order
        return elem_sym.at[order].set(value)

    elem_sym = lax.fori_loop(1, max_order + 1, _recursion_step, elem_sym)
    return elem_sym


class OrthogonalAdditiveKernel(AbstractKernel):
    r"""Orthogonal Additive Kernel (OAK).

    Wraps D one-dimensional SE base kernels with an orthogonality constraint
    (under standard normal input density) and combines them via Newton-Girard
    into an additive kernel with configurable maximum interaction order.

    The kernel decomposes as:
        K = sum_{l=0}^{D_tilde} sigma^2_l * E_l

    where E_l is the l-th elementary symmetric polynomial of the D constrained
    base kernel evaluations, and sigma^2_l are learnable order variances.

    Reference:
        Lu, X., Boukouvalas, A., & Hensman, J. (2022).
        Additive Gaussian Processes Revisited. ICML.

    Args:
        base_kernels: List of D one-dimensional base kernels (typically RBF
            with active_dims=[i] for each dimension i). Each must have
            lengthscale and variance attributes.
        max_order: Maximum interaction order (D_tilde). Defaults to D.
            Must be <= D.
        order_variances: Initial order variances of shape (max_order + 1,).
            Entry 0 is the offset variance, entry d is the d-th order
            interaction variance. Defaults to ones.
        fix_base_variance: If True (default), pin every base-kernel
            variance to 1 so that ``order_variances`` alone control
            per-order scaling.  This avoids over-parameterisation and
            matches the reference (Lu et al. 2022, S3.2).
        compute_engine: Kernel computation engine. Defaults to
            DenseKernelComputation.

    .. seealso::

        :doc:`/examples/oak` decomposes a fitted model into its per-order and
        per-dimension contributions.
    """

    name: str = "Orthogonal Additive"
    base_kernels: tuple
    max_order: int = eqx.field(static=True)
    fix_base_variance: bool = eqx.field(static=True)
    order_variances: tp.Any

    def __init__(
        self,
        base_kernels: list[AbstractKernel],
        max_order: int | None = None,
        order_variances: Float[Array, " D_tilde_plus_1"] | None = None,
        fix_base_variance: bool = True,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        num_dimensions = len(base_kernels)

        if num_dimensions == 0:
            raise ValueError("Must provide at least one base kernel.")

        if max_order is None:
            max_order = num_dimensions
        if max_order > num_dimensions:
            raise ValueError(
                f"max_order ({max_order}) must be <= number of base kernels "
                f"({num_dimensions})."
            )

        self.base_kernels = tuple(base_kernels)
        self.max_order = max_order
        self.fix_base_variance = fix_base_variance

        if order_variances is None:
            order_variances = jnp.ones(max_order + 1)

        if isinstance(order_variances, NonNegativeReal):
            self.order_variances = order_variances
        else:
            self.order_variances = NonNegativeReal(order_variances)

        super().__init__(compute_engine=compute_engine)

    @property
    def _lengthscales(self) -> Float[Array, " D"]:
        """Stack base kernel lengthscales into a single array."""
        return jnp.stack([val(k.lengthscale).squeeze() for k in self.base_kernels])

    @property
    def _variances(self) -> Float[Array, " D"]:
        """Per-dimension base-kernel variances.

        Returns ones when ``fix_base_variance`` is ``True`` (default),
        otherwise stacks the learnable base-kernel variance parameters.
        """
        if self.fix_base_variance:
            return jnp.ones(len(self.base_kernels))
        return jnp.stack([val(k.variance).squeeze() for k in self.base_kernels])

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        r"""Evaluate the OAK kernel on a pair of inputs.

        Computes constrained kernel values per dimension via vmap, then
        combines via Newton-Girard recursion weighted by order variances.

        Args:
            x: Left input of shape (D,).
            y: Right input of shape (D,).

        Returns:
            Scalar kernel value.
        """
        # Constrained kernel per dimension (vmapped over all D simultaneously)
        per_dim_values = jax.vmap(_constrained_se_kernel)(
            x, y, self._lengthscales, self._variances
        )

        # Elementary symmetric polynomials via Newton-Girard recursion
        elem_sym = _newton_girard(per_dim_values, self.max_order)

        # Weighted sum: K(x,y) = sum_d sigma^2_d * e_d
        return jnp.dot(val(self.order_variances), elem_sym)
