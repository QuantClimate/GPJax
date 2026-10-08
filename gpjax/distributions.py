# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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


import math

from beartype.typing import (
    Optional,
)
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Float
import lineax as lx
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import is_prng_key

from gpjax.linalg import cholesky_factor, logdet, logdet_from_factor
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class GaussianDistribution(Distribution):
    r"""Multivariate Gaussian distribution for GP predictions.

    This is the return type of all ``predict()`` methods in GPJax. It wraps a
    mean vector and a covariance ``lx.AbstractLinearOperator``,
    providing methods for sampling, computing log-probabilities, and evaluating
    KL divergences.

    The distribution is parameterised as

    .. math::

        p(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \mathbf{\Sigma})

    where :math:`\boldsymbol{\mu}` is the ``loc`` (mean) vector and
    :math:`\mathbf{\Sigma}` is represented by the ``scale``
    ``lx.AbstractLinearOperator``.

    Args:
        loc (Float[Array, " N"]): Mean vector of the distribution.
        scale (lx.AbstractLinearOperator): Covariance matrix represented as a
            Lineax linear operator (e.g. ``lx.MatrixLinearOperator`` or
            ``lx.DiagonalLinearOperator``).

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gpjax.distributions import GaussianDistribution
        >>> mu = jnp.array([0.0, 1.0])
        >>> cov = lx.MatrixLinearOperator(jnp.eye(2))
        >>> dist = GaussianDistribution(loc=mu, scale=cov)
        >>> dist.mean  # doctest: +SKIP
        Array([0., 1.], dtype=float32)
        >>> dist.variance  # doctest: +SKIP
        Array([1., 1.], dtype=float32)
    """

    support = constraints.real_vector

    def __init__(
        self,
        loc: Optional[Float[Array, " N"]],
        scale: Optional[lx.AbstractLinearOperator],
        validate_args=None,
    ):
        self.loc = loc
        self.scale = scale
        batch_shape = ()
        event_shape = jnp.shape(self.loc)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        r"""Draw samples from the distribution.

        Generates samples via the reparameterisation trick:

        .. math::

            \mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z},
            \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})

        where :math:`\mathbf{L}` is the lower Cholesky factor of the covariance.

        Args:
            key (KeyArray): JAX PRNG key.
            sample_shape (tuple of int, optional): Leading batch dimensions for
                the samples. Defaults to ``()``, returning a single sample.

        Returns:
            Float[Array, "... N"]: Array of samples with shape
                ``(*sample_shape, N)``.
        """
        assert is_prng_key(key)
        # Obtain covariance root.
        covariance_root = cholesky_factor(self.scale)

        # Gather n samples from standard normal distribution Z = [z₁, ..., zₙ]ᵀ.
        white_noise = jr.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )

        # xᵢ ~ N(loc, cov) <=> xᵢ = loc + sqrt zᵢ, where zᵢ ~ N(0, I).
        def affine_transformation(_x):
            return self.loc + covariance_root.mv(_x)

        if not sample_shape:
            return affine_transformation(white_noise)

        flat_noise = white_noise.reshape((math.prod(sample_shape), self.event_shape[0]))
        return vmap(affine_transformation)(flat_noise).reshape(white_noise.shape)

    @property
    def mean(self) -> Float[Array, " N"]:
        r"""Calculates the mean."""
        return self.loc

    @property
    def variance(self) -> Float[Array, " N"]:
        r"""Calculates the marginal variance (diagonal of the covariance)."""
        return lx.diagonal(self.scale)

    def entropy(self) -> ScalarFloat:
        r"""Calculates the differential entropy of the distribution.

        .. math::

            H[p] = \tfrac{1}{2}\bigl(N(1 + \ln 2\pi) + \ln|\mathbf{\Sigma}|\bigr)

        Returns:
            ScalarFloat: Entropy in nats.
        """
        return 0.5 * (
            self.event_shape[0] * (1.0 + jnp.log(2.0 * jnp.pi)) + logdet(self.scale)
        )

    def median(self) -> Float[Array, " N"]:
        r"""Calculates the median (equal to the mean for a Gaussian)."""
        return self.loc

    def mode(self) -> Float[Array, " N"]:
        r"""Calculates the mode (equal to the mean for a Gaussian)."""
        return self.loc

    def covariance(self) -> Float[Array, "N N"]:
        r"""Materialises the full covariance matrix as a dense array.

        Returns:
            Float[Array, "N N"]: Dense covariance matrix.
        """
        return self.scale.as_matrix()

    @property
    def covariance_matrix(self) -> Float[Array, "N N"]:
        r"""Property alias for :meth:`covariance`."""
        return self.covariance()

    def stddev(self) -> Float[Array, " N"]:
        r"""Calculates the marginal standard deviation."""
        return jnp.sqrt(lx.diagonal(self.scale))

    def log_prob(self, y: Float[Array, " N"]) -> ScalarFloat:
        r"""Calculates the log pdf of the multivariate Gaussian.

        .. math::

            \log p(\mathbf{y}) = -\tfrac{1}{2}\bigl[
                N\ln 2\pi + \ln|\mathbf{\Sigma}|
                + (\mathbf{y} - \boldsymbol{\mu})^\top
                  \mathbf{\Sigma}^{-1}
                  (\mathbf{y} - \boldsymbol{\mu})
            \bigr]

        Args:
            y (Float[Array, " N"]): Point at which to evaluate the log-density.

        Returns:
            ScalarFloat: Log probability.
        """
        mu = self.loc
        sigma = self.scale
        n = mu.shape[-1]

        # diff, y - µ
        diff = y - mu

        # compute the pdf, -1/2[ n log(2π) + log|Σ| + (y - µ)ᵀΣ⁻¹(y - µ) ]
        # Use the Cholesky factor L for both the log-determinant and the solve.
        # This is more efficient (single Cholesky) and propagates NaN rather
        # than raising an error when the matrix is singular, which is essential
        # for MCMC samplers that must evaluate the density at rejected proposals.
        L = cholesky_factor(sigma)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L.as_matrix())))
        L_inv_diff = jsp.linalg.solve_triangular(L.as_matrix(), diff, lower=True)
        return -0.5 * (
            n * jnp.log(2.0 * jnp.pi) + log_det + jnp.dot(L_inv_diff, L_inv_diff)
        )

    def kl_divergence(self, other: "GaussianDistribution") -> ScalarFloat:
        r"""KL divergence from ``self`` to ``other``.

        Computes :math:`\operatorname{KL}[q \| p]` where ``self`` is *q* and
        ``other`` is *p*.

        Args:
            other (GaussianDistribution): The reference distribution *p*.

        Returns:
            ScalarFloat: KL divergence in nats.
        """
        return _kl_divergence(self, other)


def _check_and_return_dimension(
    q: GaussianDistribution, p: GaussianDistribution
) -> int:
    r"""Checks that the dimensions of the distributions are compatible."""
    if q.event_shape != p.event_shape:
        raise ValueError(
            "Distribution event shapes are not compatible: `q.event_shape ="
            f" {q.event_shape}` and `p.event_shape = {p.event_shape}`. Please check"
            " your mean and covariance shapes."
        )

    return q.event_shape[-1]


def _frobenius_norm_squared(matrix: Float[Array, "N N"]) -> ScalarFloat:
    r"""Calculates the squared Frobenius norm of a matrix."""
    return jnp.sum(jnp.square(matrix))


def _kl_divergence(q: GaussianDistribution, p: GaussianDistribution) -> ScalarFloat:
    r"""KL-divergence between two Gaussians.

    Computes the KL divergence, $\operatorname{KL}[q\mid\mid p]$, between two
    multivariate Gaussian distributions $q(x) = \mathcal{N}(x; \mu_q, \Sigma_q)$
    and $p(x) = \mathcal{N}(x; \mu_p, \Sigma_p)$.

    Args:
        q: a multivariate Gaussian distribution.
        p: another multivariate Gaussian distribution.

    Returns:
        ScalarFloat: The KL divergence between q and p.
    """
    n_dim = _check_and_return_dimension(q, p)

    # Extract q mean and covariance.
    mu_q = q.loc
    sigma_q = q.scale

    # Extract p mean and covariance.
    mu_p = p.loc
    sigma_p = p.scale

    # Find covariance roots.
    sqrt_p = cholesky_factor(sigma_p)
    sqrt_q = cholesky_factor(sigma_q)

    # diff, μp - μq
    diff = mu_p - mu_q

    # trace term, tr[Σp⁻¹ Σq] = tr[(LpLpᵀ)⁻¹(LqLqᵀ)] = tr[(Lp⁻¹Lq)(Lp⁻¹Lq)ᵀ] = (fr[LqLp⁻¹])²
    # Use jsp.linalg.solve_triangular for matrix RHS since lx.linear_solve only handles vectors.

    trace = _frobenius_norm_squared(
        jsp.linalg.solve_triangular(sqrt_p.as_matrix(), sqrt_q.as_matrix(), lower=True)
    )  # TODO: Not most efficient, given the `as_matrix()` call (e.g., consider diagonal p and q). Need to abstract solving linear operator against another linear operator.

    # Mahalanobis term, (μp - μq)ᵀ Σp⁻¹ (μp - μq) = tr [(μp - μq)ᵀ [LpLpᵀ]⁻¹ (μp - μq)] = (fr[Lp⁻¹(μp - μq)])²
    mahalanobis = jnp.sum(
        jnp.square(lx.linear_solve(sqrt_p, diff, solver=lx.Triangular()).value)
    )

    # Log-determinants, log|Σ| = 2 Σᵢ log Lᵢᵢ, reusing the factors computed above.
    # Calling `logdet(Σ)` here would re-factorise both covariances, doubling the
    # O(N³) work on the hot path of every ELBO step (issue #664).
    logdet_q = logdet_from_factor(sqrt_q)
    logdet_p = logdet_from_factor(sqrt_p)

    # KL[q(x)||p(x)] = [ [(μp - μq)ᵀ Σp⁻¹ (μp - μq)] - n - log|Σq| + log|Σp| + tr[Σp⁻¹ Σq] ] / 2
    return (mahalanobis - n_dim - logdet_q + logdet_p + trace) / 2.0


__all__ = [
    "GaussianDistribution",
]
