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
    Union,
)
from jax import (
    lax,
    vmap,
)
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import (
    Float,
    Int,
)
import lineax as lx
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    is_prng_key,
    promote_shapes,
)
from scipy.special import zeta

from gpjax.linalg import cholesky_factor, logdet, logdet_from_factor
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)

RealLike = Union[float, int, Float[Array, "..."], Int[Array, "..."]]


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


_SERIES_THRESHOLD = 0.05
_LOG1P_RATIO_COEFFICIENTS = tuple((-1.0) ** k / (k + 1) for k in range(15))
_EXPM1_RATIO_COEFFICIENTS = tuple(1.0 / math.factorial(k + 1) for k in range(11))
_EULER_GAMMA = 0.5772156649015329
_ZETA_FROM_2 = tuple(float(zeta(k)) for k in range(2, 21))
_LOG_GAMMA_RATIO_COEFFICIENTS = (
    _EULER_GAMMA,
    *(z / k for k, z in enumerate(_ZETA_FROM_2, start=2)),
)
_LOG_GAMMA_DIFFERENCE_COEFFICIENTS = tuple(
    z * (2.0**k - 2.0) / k for k, z in enumerate(_ZETA_FROM_2, start=2)
)

_CONTINUATION_LOWER_U = -5.0
_CONTINUATION_UPPER_T = 1e-2
_CDF_UNDERFLOW_U = -8.0


def _as_arrays(*values: RealLike) -> tuple[Array, ...]:
    r"""Convert numbers to JAX arrays; a Python number stays weakly typed."""
    return tuple(jnp.asarray(value) for value in values)


def _horner(x: Array, coefficients: tuple[float, ...]) -> Array:
    r"""Evaluate the polynomial with the given coefficients (lowest first)."""
    result = jnp.zeros_like(x) + coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = result * x + coefficient
    return result


def _log1p_ratio(w: Array) -> Array:
    r"""Return ``log1p(w) / w``, with the value 1 at ``w = 0``. Requires ``w > -1``.

    The value and every derivative are finite at ``w = 0``: a Taylor series
    replaces the direct form near zero, and the two-sided ``where`` keeps the
    branch that is not used out of the gradient.
    """
    small = jnp.abs(w) < _SERIES_THRESHOLD
    w_small = jnp.where(small, w, 0.0)
    w_large = jnp.where(small, 1.0, w)
    return jnp.where(
        small,
        _horner(w_small, _LOG1P_RATIO_COEFFICIENTS),
        jnp.log1p(w_large) / w_large,
    )


def _expm1_ratio(a: Array) -> Array:
    r"""Return ``expm1(a) / a``, with the value 1 at ``a = 0``.

    As for :func:`_log1p_ratio`, a Taylor series replaces the direct form near
    zero, so the value and the gradient are finite at ``a = 0``.
    """
    small = jnp.abs(a) < _SERIES_THRESHOLD
    a_small = jnp.where(small, a, 0.0)
    a_large = jnp.where(small, 1.0, a)
    return jnp.where(
        small,
        _horner(a_small, _EXPM1_RATIO_COEFFICIENTS),
        jnp.expm1(a_large) / a_large,
    )


def _reduced_variate(z: Array, xi: Array) -> Array:
    r"""Return ``u = log(1 + xi z) / xi``, which is ``z`` at ``xi = 0``.

    Requires ``1 + xi z > 0``.
    """
    return z * _log1p_ratio(xi * z)


def _standard_log_density(u: Array, xi: Array) -> Array:
    r"""The log density of the standardised GEV at the reduced variate ``u``."""
    return -(1.0 + xi) * u - jnp.exp(-u)


def _exact_log_density(z: Array, xi: Array) -> Array:
    r"""The exact standardised log density; ``-inf`` outside the support."""
    inside = 1.0 + xi * z > 0.0
    u = _reduced_variate(jnp.where(inside, z, 0.0), xi)
    return jnp.where(inside, _standard_log_density(u, xi), -jnp.inf)


def _upper_junction_t(xi: Array) -> Array:
    r"""The value of ``t = 1 + xi z`` at the upper junction (used for ``xi < 0``)."""
    return _CONTINUATION_UPPER_T * jnp.maximum(1.0 + xi, _CONTINUATION_UPPER_T)


def _continued_log_density(z: Array, xi: Array) -> Array:
    r"""The standardised log density with the support continuation.

    The exact log density, except in two regions where it continues as the
    straight line (in ``z``) that has the same value and slope at the junction:

    * below ``z_lo``, where ``u = -5``, for every ``xi``;
    * above ``z_hi``, where ``t = 1e-2 (1 + xi)``, for ``xi < 0`` only.

    At a junction, the value and the first derivatives with respect to ``z``
    and ``xi`` agree on the two sides.
    """
    u_lo = _CONTINUATION_LOWER_U
    z_lo = u_lo * _expm1_ratio(xi * u_lo)
    value_lo = -(1.0 + xi) * u_lo - math.exp(-u_lo)
    slope_lo = (math.exp(-u_lo) - (1.0 + xi)) * jnp.exp(-xi * u_lo)
    below_lower_junction = z < z_lo

    above_upper_junction = (xi < 0.0) & (1.0 + xi * z < _upper_junction_t(xi))
    xi_negative = jnp.where(above_upper_junction, xi, -1.0)
    t_hi = _upper_junction_t(xi_negative)
    u_hi = jnp.log(t_hi) / xi_negative
    z_hi = (t_hi - 1.0) / xi_negative
    value_hi = -(1.0 + xi_negative) * u_hi - jnp.exp(-u_hi)
    slope_hi = (jnp.exp(-u_hi) - (1.0 + xi_negative)) / t_hi

    between_junctions = ~(below_lower_junction | above_upper_junction)
    u = _reduced_variate(jnp.where(between_junctions, z, 0.0), xi)
    return jnp.where(
        below_lower_junction,
        value_lo + slope_lo * (z - z_lo),
        jnp.where(
            above_upper_junction,
            value_hi + slope_hi * (z - z_hi),
            _standard_log_density(u, xi),
        ),
    )


def _tail_probabilities(z: Array, xi: Array) -> tuple[Array, Array]:
    r"""Return ``F`` and ``1 - F`` of the standardised GEV at ``z``.

    Both are exact outside the support: ``F = 0`` below the lower end point
    (``xi > 0``) and ``F = 1`` at and above the upper end point (``xi < 0``).
    ``1 - F`` is computed as ``-expm1(-exp(-u))``, so small exceedance
    probabilities keep their relative accuracy.
    """
    z_underflow = _CDF_UNDERFLOW_U * _expm1_ratio(xi * _CDF_UNDERFLOW_U)
    below = z < z_underflow
    above = (xi < 0.0) & (1.0 + xi * z <= 0.0)
    inside = ~(below | above)
    u = _reduced_variate(jnp.where(inside, z, 0.0), xi)
    neg_log_cdf = jnp.exp(-u)
    cdf = jnp.where(below, 0.0, jnp.where(above, 1.0, jnp.exp(-neg_log_cdf)))
    sf = jnp.where(below, 1.0, jnp.where(above, 0.0, -jnp.expm1(-neg_log_cdf)))
    return cdf, sf


def _reduced_quantile(log_neg_log: Array, xi: Array) -> Array:
    r"""Return the standardised quantile ``z`` with ``log(-log F(z)) = L``.

    ``z = (exp(-xi L) - 1) / xi = -L expm1(-xi L) / (-xi L)``, which is ``-L``
    at ``xi = 0``. ``L = +inf`` gives the lower end point and ``L = -inf`` the
    upper end point.
    """
    finite = jnp.isfinite(log_neg_log)
    log_neg_log_finite = jnp.where(finite, log_neg_log, 0.0)
    z = -log_neg_log_finite * _expm1_ratio(-xi * log_neg_log_finite)
    xi_safe = jnp.where(xi == 0.0, 1.0, xi)
    lower_end = jnp.where(xi > 0.0, -1.0 / xi_safe, -jnp.inf)
    upper_end = jnp.where(xi < 0.0, -1.0 / xi_safe, jnp.inf)
    end_point = jnp.where(log_neg_log > 0.0, lower_end, upper_end)
    return jnp.where(finite, z, jnp.where(jnp.isnan(log_neg_log), jnp.nan, end_point))


def _standard_mean(xi: Array) -> Array:
    r"""Return ``(Gamma(1 - xi) - 1) / xi``; ``gamma`` at 0 and ``inf`` for ``xi >= 1``."""
    exists = xi < 1.0
    xi = jnp.where(exists, xi, 0.0)
    small = jnp.abs(xi) < _SERIES_THRESHOLD
    xi_small = jnp.where(small, xi, 0.0)
    xi_large = jnp.where(small, 0.5, xi)
    log_gamma_ratio = _horner(xi_small, _LOG_GAMMA_RATIO_COEFFICIENTS)
    series = log_gamma_ratio * _expm1_ratio(xi_small * log_gamma_ratio)
    direct = jnp.expm1(jsp.special.gammaln(1.0 - xi_large)) / xi_large
    return jnp.where(exists, jnp.where(small, series, direct), jnp.inf)


def _standard_variance(xi: Array) -> Array:
    r"""Return ``(Gamma(1 - 2 xi) - Gamma(1 - xi)^2) / xi^2``.

    The value is ``pi^2 / 6`` at 0 and ``inf`` for ``xi >= 1/2``.
    """
    exists = xi < 0.5
    xi = jnp.where(exists, xi, 0.0)
    small = jnp.abs(xi) < _SERIES_THRESHOLD
    xi_small = jnp.where(small, xi, 0.0)
    xi_large = jnp.where(small, 0.25, xi)
    log_gamma = xi_small * _horner(xi_small, _LOG_GAMMA_RATIO_COEFFICIENTS)
    difference_ratio = _horner(xi_small, _LOG_GAMMA_DIFFERENCE_COEFFICIENTS)
    series = (
        jnp.exp(2.0 * log_gamma)
        * difference_ratio
        * _expm1_ratio(xi_small**2 * difference_ratio)
    )
    log_gamma_1 = jsp.special.gammaln(1.0 - xi_large)
    log_gamma_2 = jsp.special.gammaln(1.0 - 2.0 * xi_large)
    direct = (
        jnp.exp(2.0 * log_gamma_1)
        * jnp.expm1(log_gamma_2 - 2.0 * log_gamma_1)
        / xi_large**2
    )
    return jnp.where(exists, jnp.where(small, series, direct), jnp.inf)


def gev_return_level(
    return_period: RealLike,
    loc: RealLike,
    scale: RealLike,
    concentration: RealLike,
) -> Float[Array, "..."]:
    r"""The GEV return level for a return period.

    The return level $z_T$ is the level that the annual maximum exceeds with
    probability $1/T$, that is, on average once in $T$ years:

    .. math::

        z_T = \mu - \frac{\sigma}{\xi}\left[1 - \{-\log(1 - 1/T)\}^{-\xi}\right],
        \qquad
        z_T = \mu - \sigma \log\{-\log(1 - 1/T)\} \quad (\xi = 0).

    The function is accurate and has finite gradients for all $\xi$,
    including $\xi = 0$, and for large $T$. All arguments broadcast with the
    NumPy rules. Thus a user who has samples of $(\mu, \sigma, \xi)$ from any
    source gets samples of the return level.

    Example:
        >>> import jax.numpy as jnp
        >>> from gpjax.distributions import gev_return_level
        >>> periods = jnp.array([10.0, 100.0])
        >>> levels = gev_return_level(periods, loc=40.0, scale=12.0,
        ...                           concentration=0.1)
        >>> levels.shape
        (2,)

    Args:
        return_period: The return period $T > 1$, in blocks (for example, years
            for annual maxima). $T = 1$ gives the lower end point and
            $T = \infty$ the upper end point; $T < 1$ gives NaN.
        loc: The location $\mu$.
        scale: The scale $\sigma > 0$.
        concentration: The shape $\xi$. $\xi > 0$ is the heavy-tailed (Fréchet)
            type and $\xi < 0$ the bounded (Weibull) type. This is the climate
            convention; ``scipy.stats.genextreme`` uses $c = -\xi$.

    Returns:
        The return levels, with the broadcast shape of the arguments.
    """
    loc, scale, concentration = _as_arrays(loc, scale, concentration)
    exceedance = 1.0 / jnp.asarray(return_period)
    log_neg_log = jnp.log(-jnp.log1p(-exceedance))
    return loc + scale * _reduced_quantile(log_neg_log, concentration)


def gev_return_period(
    level: RealLike,
    loc: RealLike,
    scale: RealLike,
    concentration: RealLike,
) -> Float[Array, "..."]:
    r"""The GEV return period of a level: $T = 1 / P(Y > \text{level})$.

    The inverse of :func:`gev_return_level`. The exceedance probability comes
    from :meth:`GeneralisedExtremeValue.sf`, so very large return periods keep
    their accuracy. Where the exceedance probability is 0, that is, at or
    above the upper end point when $\xi < 0$, the return period is ``inf``
    (not NaN), and its gradient is 0. All arguments broadcast with the NumPy
    rules.

    Example:
        >>> import jax.numpy as jnp
        >>> from gpjax.distributions import gev_return_period
        >>> period = gev_return_period(100.0, loc=40.0, scale=12.0,
        ...                            concentration=0.1)

    Args:
        level: The level, in the units of the data.
        loc: The location $\mu$.
        scale: The scale $\sigma > 0$.
        concentration: The shape $\xi$ (climate convention, see
            :func:`gev_return_level`).

    Returns:
        The return periods, with the broadcast shape of the arguments.
    """
    level, loc, scale, concentration = _as_arrays(level, loc, scale, concentration)
    _, sf = _tail_probabilities((level - loc) / scale, concentration)
    positive = sf > 0.0
    return jnp.where(positive, 1.0 / jnp.where(positive, sf, 1.0), jnp.inf)


class GeneralisedExtremeValue(Distribution):
    r"""The generalised extreme value (GEV) distribution.

    The GEV distribution is the limit distribution of block maxima, for example
    annual maxima of daily rainfall. With $t = 1 + \xi (y - \mu) / \sigma$, its
    cumulative distribution function is

    .. math::

        F(y) = \exp\left(-t^{-1/\xi}\right), \qquad
        F(y) = \exp\left(-e^{-(y - \mu)/\sigma}\right) \quad (\xi = 0),

    on the support $t > 0$. The shape $\xi$ (``concentration``) sets the tail:

    * $\xi > 0$: heavy upper tail (Fréchet type), with the lower end point
      $\mu - \sigma/\xi$;
    * $\xi = 0$: the Gumbel distribution, with support on the real line;
    * $\xi < 0$: bounded upper tail (Weibull type), with the upper end point
      $\mu - \sigma/\xi$.

    This is the climate convention. ``scipy.stats.genextreme`` uses $c = -\xi$.
    The shape is called ``concentration``, as in TensorFlow Probability,
    because ``shape`` is a method of every NumPyro distribution.

    All methods are accurate and have finite gradients with respect to the
    value and to the three parameters at $\xi = 0$ and near it: the ratios
    $\log(1 + w)/w$ and $(e^a - 1)/a$ that cause $0/0$ are Taylor series near
    zero.

    **Support continuation.** Outside the support the exact log density is
    $-\infty$, and it has no useful gradient. A Gaussian distribution over the
    location always has mass where an observation is outside the support (for
    $\xi \neq 0$), so an expected log-likelihood is $-\infty$, and
    Gauss–Hermite nodes can fall outside the support. With
    ``support_continuation=True``, ``log_prob`` is a pseudo-log-density that is
    finite everywhere:

    * Exact where the reduced variate $u = \log(t)/\xi$ is at least $-5$
      (there $F \geq \exp(-e^5) \approx 10^{-64}$), and, when $\xi < 0$,
      where $t \geq 10^{-2}(1 + \xi)$.
    * Below the point where $u = -5$, for every $\xi$, the log density
      continues as the straight line in $y$ with the same value and slope.
      This covers the region below the lower end point when $\xi > 0$. It also
      bounds the slope of the double-exponential lower tail when
      $\xi \leq 0$.
    * Above the point where $t = 10^{-2}(1 + \xi)$, for $\xi < 0$ only, the
      log density continues in the same way. This covers the region above the
      upper end point. The factor $1 + \xi$ makes the slope there about
      $-100/\sigma$ for every $\xi$.

    The continuation is continuously differentiable: at each junction, the
    value and the first derivatives with respect to $y$, $\mu$, $\sigma$ and
    $\xi$ agree on the two sides. The junctions move smoothly with $\xi$, and
    the function is continuous in $\xi$ through 0. Outside the support the
    gradient always points into the support: a gradient step increases the
    log density by a move of the location, the scale or the shape that brings
    the end point past the observation. The continuation is designed for
    $\xi > -1$.

    The continued function is a pseudo-density. It puts a small mass where the
    exact density is zero, and it changes the density in the two regions. For
    $-0.5 \leq \xi \leq 0.5$ its total mass differs from 1 by at most
    $3 \times 10^{-5}$, at $\xi = -0.5$; there the exact probability of the
    upper region is $2.5 \times 10^{-5}$. The difference is less than
    $10^{-7}$ for $\xi \geq -0.3$ and less than $10^{-10}$ for
    $\xi \geq -0.2$, and the lower region contributes about $10^{-64}$.
    ``cdf``, ``sf``, ``icdf``, ``isf``, ``sample``, ``mean`` and ``variance``
    are always exact; only ``log_prob`` changes.

    Example:
        >>> import jax.numpy as jnp
        >>> from gpjax.distributions import GeneralisedExtremeValue
        >>> dist = GeneralisedExtremeValue(loc=40.0, scale=12.0, concentration=0.1)
        >>> level_100 = dist.return_level(100.0)
        >>> probability = dist.sf(level_100)  # 0.01

    Args:
        loc: The location $\mu$.
        scale: The scale $\sigma > 0$.
        concentration: The shape $\xi$.
        support_continuation: If ``True``, ``log_prob`` uses the support
            continuation above. If ``False`` (the default), ``log_prob`` is
            exact and is ``-inf`` outside the support.
        validate_args: Passed to :class:`numpyro.distributions.Distribution`.
    """

    arg_constraints = {  # noqa: RUF012
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    support = constraints.real
    reparametrized_params = ["loc", "scale", "concentration"]  # noqa: RUF012
    pytree_aux_fields = ("support_continuation",)

    def __init__(
        self,
        loc: RealLike = 0.0,
        scale: RealLike = 1.0,
        concentration: RealLike = 0.0,
        *,
        support_continuation: bool = False,
        validate_args: Optional[bool] = None,
    ):
        loc, scale, concentration = _as_arrays(loc, scale, concentration)
        self.loc, self.scale, self.concentration = promote_shapes(
            loc, scale, concentration
        )
        self.support_continuation = bool(support_continuation)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc), jnp.shape(scale), jnp.shape(concentration)
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _standardise(self, value: RealLike) -> Array:
        return (jnp.asarray(value) - self.loc) / self.scale

    def log_prob(self, value: RealLike) -> Float[Array, "..."]:
        r"""The log density at ``value``.

        Exact (``-inf`` outside the support) by default. With
        ``support_continuation=True``, the continued pseudo-log-density that is
        described in the class docstring.

        Args:
            value: The point at which to evaluate the log density.

        Returns:
            The log density, with the broadcast shape of ``value`` and the
            batch shape.
        """
        z = self._standardise(value)
        if self.support_continuation:
            standard = _continued_log_density(z, self.concentration)
        else:
            standard = _exact_log_density(z, self.concentration)
        return standard - jnp.log(self.scale)

    def cdf(self, value: RealLike) -> Float[Array, "..."]:
        r"""The cumulative distribution function $P(Y \leq y)$.

        Exact everywhere: 0 below the lower end point ($\xi > 0$), and 1 at
        and above the upper end point ($\xi < 0$).

        Args:
            value: The point $y$.

        Returns:
            The probability $P(Y \leq y)$.
        """
        return _tail_probabilities(self._standardise(value), self.concentration)[0]

    def sf(self, value: RealLike) -> Float[Array, "..."]:
        r"""The survival function, or exceedance probability, $P(Y > y)$.

        The name follows ``scipy.stats`` (NumPyro has no survival function).
        It is computed as $-\mathrm{expm1}(-e^{-u})$, not as $1 - F(y)$, so
        very small probabilities (for example $10^{-12}$) keep their relative
        accuracy. It is exactly 1 below the lower end point ($\xi > 0$) and
        exactly 0 at and above the upper end point ($\xi < 0$).

        Args:
            value: The point $y$.

        Returns:
            The probability $P(Y > y)$.
        """
        return _tail_probabilities(self._standardise(value), self.concentration)[1]

    def icdf(self, q: RealLike) -> Float[Array, "..."]:
        r"""The quantile function, the inverse of :meth:`cdf`.

        $q = 0$ gives the lower end point ($-\infty$ if $\xi \leq 0$) and
        $q = 1$ gives the upper end point ($+\infty$ if $\xi \geq 0$).

        Args:
            q: The probability $q = P(Y \leq y)$.

        Returns:
            The quantile $y$.
        """
        log_neg_log = jnp.log(-jnp.log(jnp.asarray(q)))
        return self.loc + self.scale * _reduced_quantile(
            log_neg_log, self.concentration
        )

    def isf(self, p: RealLike) -> Float[Array, "..."]:
        r"""The inverse survival function, the inverse of :meth:`sf`.

        The name follows ``scipy.stats``. It is accurate for very small
        exceedance probabilities $p$, where ``icdf(1 - p)`` loses digits.

        Args:
            p: The exceedance probability $p = P(Y > y)$.

        Returns:
            The level $y$.
        """
        log_neg_log = jnp.log(-jnp.log1p(-jnp.asarray(p)))
        return self.loc + self.scale * _reduced_quantile(
            log_neg_log, self.concentration
        )

    def sample(
        self, key: KeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Float[Array, "..."]:
        r"""Draw samples by the inverse of the cumulative distribution function.

        A standard Gumbel draw $G$ gives $y = \mu + \sigma (e^{\xi G} - 1)/\xi$.

        Args:
            key: A JAX PRNG key.
            sample_shape: Leading dimensions of the samples.

        Returns:
            Samples of shape ``sample_shape + batch_shape``.
        """
        assert is_prng_key(key)
        dtype = jnp.result_type(self.loc, self.scale, self.concentration)
        gumbel = jr.gumbel(key, sample_shape + self.batch_shape, dtype=dtype)
        return self.loc + self.scale * _reduced_quantile(-gumbel, self.concentration)

    @property
    def mean(self) -> Float[Array, "..."]:
        r"""The mean, $\mu + \sigma\{\Gamma(1 - \xi) - 1\}/\xi$.

        The mean is $\mu + \sigma\gamma$ at $\xi = 0$, and ``inf`` for
        $\xi \geq 1$.
        """
        mean = self.loc + self.scale * _standard_mean(self.concentration)
        return jnp.broadcast_to(mean, self.batch_shape)

    @property
    def variance(self) -> Float[Array, "..."]:
        r"""The variance, $\sigma^2\{\Gamma(1 - 2\xi) - \Gamma(1 - \xi)^2\}/\xi^2$.

        The variance is $\sigma^2\pi^2/6$ at $\xi = 0$, and ``inf`` for
        $\xi \geq 1/2$.
        """
        variance = self.scale**2 * _standard_variance(self.concentration)
        return jnp.broadcast_to(variance, self.batch_shape)

    def return_level(self, return_period: RealLike) -> Float[Array, "..."]:
        r"""The return level for ``return_period``; see :func:`gev_return_level`.

        Args:
            return_period: The return period $T > 1$, in blocks.

        Returns:
            The level that is exceeded with probability $1/T$.
        """
        return gev_return_level(return_period, self.loc, self.scale, self.concentration)

    def return_period(self, level: RealLike) -> Float[Array, "..."]:
        r"""The return period of ``level``; see :func:`gev_return_period`.

        Args:
            level: The level, in the units of the data.

        Returns:
            $1 / P(Y > \text{level})$, which is ``inf`` where the exceedance
            probability is 0.
        """
        return gev_return_period(level, self.loc, self.scale, self.concentration)


__all__ = [
    "GaussianDistribution",
    "GeneralisedExtremeValue",
    "gev_return_level",
    "gev_return_period",
]
