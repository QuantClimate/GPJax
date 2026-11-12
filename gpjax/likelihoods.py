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
from __future__ import annotations

import abc
from dataclasses import dataclass

import beartype.typing as tp
import equinox as eqx
import jax
from jax import vmap
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Float
import lineax as lx
import numpy as np
import numpyro.distributions as npd
from paramax import AbstractUnwrappable

from gpjax.distributions import (
    GaussianDistribution,
    GeneralisedExtremeValue,
    RealLike,
    gev_return_level,
    gev_return_period,
)
from gpjax.integrators import (
    AbstractIntegrator,
    AnalyticalGaussianIntegrator,
    GHQuadratureIntegrator,
)
from gpjax.parameters import (
    NonNegativeReal,
    PositiveReal,
    SigmoidBounded,
    val,
)
from gpjax.summary import _SummaryMixin
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)


def _diagonal_scale(op):
    """Unwrap a single TaggedLinearOperator layer and return the inner diagonal, or None.

    Returns the input itself if it is already a DiagonalLinearOperator; returns the
    inner DiagonalLinearOperator if `op` is a TaggedLinearOperator wrapping one;
    returns None otherwise (e.g. dense MatrixLinearOperator, nested tags).
    """
    if isinstance(op, lx.DiagonalLinearOperator):
        return op
    if isinstance(op, lx.TaggedLinearOperator) and isinstance(
        op.operator, lx.DiagonalLinearOperator
    ):
        return op.operator
    return None


@dataclass(slots=True)
class NoiseMoments:
    log_variance: Array
    inv_variance: Array
    variance: Array


jax.tree_util.register_pytree_node(
    NoiseMoments,
    lambda x: ((x.log_variance, x.inv_variance, x.variance), None),
    lambda _, x: NoiseMoments(*x),
)


class AbstractLikelihood(_SummaryMixin, eqx.Module):
    r"""Abstract base class for likelihoods.

    All likelihoods must inherit from this class and implement the `predict` and
    `link_function` methods.
    """

    integrator: AbstractIntegrator = eqx.field(static=True)

    def __init__(
        self,
        integrator: AbstractIntegrator = GHQuadratureIntegrator(),
    ):
        """Initializes the likelihood.

        Args:
            integrator (AbstractIntegrator): The integrator to be used for computing expected log
                likelihoods. Must be an instance of `AbstractIntegrator`.
        """
        self.integrator = integrator

    def __call__(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.Distribution:
        r"""Evaluate the likelihood function at a given predictive distribution.

        Args:
            dist: The predictive distribution to evaluate the likelihood at.

        Returns:
            The predictive distribution.
        """
        return self.predict(dist)

    @abc.abstractmethod
    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.Distribution:
        r"""Evaluate the likelihood function at a given predictive distribution.

        Args:
            dist: The predictive distribution to evaluate the likelihood at.

        Returns:
            npd.Distribution: The predictive distribution.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def link_function(self, f: Float[Array, ...]) -> npd.Distribution:
        r"""Return the link function of the likelihood function.

        Args:
            f (Float[Array, "..."]): the latent Gaussian process values.

        Returns:
            npd.Distribution: The distribution of observations, y, given values of the
                Gaussian process, f.
        """
        raise NotImplementedError

    def expected_log_likelihood(
        self,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        mean_g: tp.Optional[Float[Array, "N D"]] = None,
        variance_g: tp.Optional[Float[Array, "N D"]] = None,
        **_: tp.Any,
    ) -> Float[Array, " N"]:
        r"""Compute the expected log likelihood.

        For a variational distribution $q(f)\sim\mathcal{N}(m, s)$ and a likelihood
        $p(y|f)$, compute the expected log likelihood:

        .. math::

            \mathbb{E}_{q(f)}\left[\log p(y|f)\right]

        Args:
            y (Float[Array, 'N D']): The observed response variable.
            mean (Float[Array, 'N D']): The variational mean.
            variance (Float[Array, 'N D']): The variational variance.
            mean_g (Float[Array, 'N D']): Optional moments of the latent noise
                process for heteroscedastic likelihoods.
            variance_g (Float[Array, 'N D']): Optional moments of the latent noise
                process for heteroscedastic likelihoods.
            **_: Unused extra arguments for compatibility with specialised
                likelihoods.

        Returns:
            ScalarFloat: The expected log likelihood.
        """
        log_prob = vmap(lambda f, y: self.link_function(f).log_prob(y))
        return self.integrator(
            fun=log_prob, y=y, mean=mean, variance=variance, likelihood=self
        )


class AbstractNoiseTransform(eqx.Module):
    """Abstract base class for noise transformations."""

    @abc.abstractmethod
    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        """Transform the input noise signal."""
        raise NotImplementedError

    @abc.abstractmethod
    def moments(
        self, mean: Float[Array, ...], variance: Float[Array, ...]
    ) -> NoiseMoments:
        """Compute the moments of the transformed noise signal."""
        raise NotImplementedError


class LogNormalTransform(AbstractNoiseTransform):
    """Log-normal noise transformation."""

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        return jnp.exp(x)

    def moments(
        self, mean: Float[Array, ...], variance: Float[Array, ...]
    ) -> NoiseMoments:
        expected_variance = jnp.exp(mean + 0.5 * variance)
        expected_log_variance = mean
        expected_inv_variance = jnp.exp(-mean + 0.5 * variance)
        return NoiseMoments(
            log_variance=expected_log_variance,
            inv_variance=expected_inv_variance,
            variance=expected_variance,
        )


class SoftplusTransform(AbstractNoiseTransform):
    """Softplus noise transformation."""

    num_points: int = eqx.field(static=True, default=20)

    def __init__(self, num_points: int = 20):
        self.num_points = num_points

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        return jnn.softplus(x)

    def moments(
        self, mean: Float[Array, ...], variance: Float[Array, ...]
    ) -> NoiseMoments:
        quad_x, quad_w = np.polynomial.hermite.hermgauss(self.num_points)
        quad_w = jnp.asarray(quad_w / jnp.sqrt(jnp.pi))
        quad_x = jnp.asarray(quad_x)

        std = jnp.sqrt(variance)
        samples = mean[..., None] + jnp.sqrt(2.0) * std[..., None] * quad_x
        sigma2 = self(samples)
        log_sigma2 = jnp.log(sigma2)
        inv_sigma2 = 1.0 / sigma2

        expected_variance = jnp.sum(sigma2 * quad_w, axis=-1)
        expected_log_variance = jnp.sum(log_sigma2 * quad_w, axis=-1)
        expected_inv_variance = jnp.sum(inv_sigma2 * quad_w, axis=-1)

        return NoiseMoments(
            log_variance=expected_log_variance,
            inv_variance=expected_inv_variance,
            variance=expected_variance,
        )


class AbstractHeteroscedasticLikelihood(AbstractLikelihood):
    r"""Base class for heteroscedastic likelihoods with latent noise processes."""

    noise_transform: AbstractNoiseTransform

    def __init__(
        self,
        noise_transform: tp.Union[
            AbstractNoiseTransform,
            tp.Callable[[Float[Array, ...]], Float[Array, ...]],
        ] = SoftplusTransform(),
        integrator: AbstractIntegrator = GHQuadratureIntegrator(),
    ):
        if isinstance(noise_transform, AbstractNoiseTransform):
            self.noise_transform = noise_transform
        else:
            transform_name = getattr(noise_transform, "__name__", "")
            if noise_transform is jnp.exp or transform_name == "exp":
                self.noise_transform = LogNormalTransform()
            else:
                # Default to SoftplusTransform for softplus or unknown callables (legacy behavior used quadrature)
                # Note: If an unknown callable is passed, we technically use SoftplusTransform which applies softplus.
                # Users should implement AbstractNoiseTransform for custom transforms.
                self.noise_transform = SoftplusTransform()

        super().__init__(integrator=integrator)

    def __call__(
        self,
        dist: tp.Union[npd.MultivariateNormal, GaussianDistribution],
        noise_dist: tp.Optional[
            tp.Union[npd.MultivariateNormal, GaussianDistribution]
        ] = None,
    ) -> npd.Distribution:
        return self.predict(dist, noise_dist)

    def supports_tight_bound(self) -> bool:
        """Return whether the tighter bound from Lazaro-Gredilla & Titsias (2011)
        is applicable."""
        return False

    def noise_statistics(
        self, mean: Float[Array, "N D"], variance: Float[Array, "N D"]
    ) -> NoiseMoments:
        r"""Moment matching of the transformed noise process.

        Args:
            mean: Mean of the latent noise GP.
            variance: Variance of the latent noise GP.

        Returns:
            NoiseMoments: Expected log variance, inverse variance, and variance.
        """
        return self.noise_transform.moments(mean, variance)

    def expected_log_likelihood(
        self,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        mean_g: tp.Optional[Float[Array, "N D"]] = None,
        variance_g: tp.Optional[Float[Array, "N D"]] = None,
        **kwargs: tp.Any,
    ) -> Float[Array, " N"]:
        raise NotImplementedError


class Gaussian(AbstractLikelihood):
    r"""Gaussian likelihood object."""

    obs_stddev: tp.Any
    num_outputs: int = eqx.field(static=True, default=1)

    def __init__(
        self,
        obs_stddev: tp.Union[ScalarFloat, Float[Array, "#N"], NonNegativeReal] = 1.0,
        integrator: AbstractIntegrator = AnalyticalGaussianIntegrator(),
    ):
        r"""Initializes the Gaussian likelihood.

        Args:
            obs_stddev (Union[ScalarFloat, Float[Array, "#N"]]): the standard deviation
                of the Gaussian observation noise.
            integrator (AbstractIntegrator): The integrator to be used for computing expected log
                likelihoods. Must be an instance of `AbstractIntegrator`. For the Gaussian likelihood, this defaults to
                the `AnalyticalGaussianIntegrator`, as the expected log likelihood can be computed analytically.
        """
        if not isinstance(obs_stddev, NonNegativeReal):
            obs_stddev = NonNegativeReal(jnp.asarray(obs_stddev))
        self.obs_stddev = obs_stddev
        self.num_outputs = 1

        super().__init__(integrator)

    def link_function(self, f: Float[Array, ...]) -> npd.Normal:
        r"""The link function of the Gaussian likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            npd.Normal: The likelihood function.
        """
        # obs_stddev is non-negative, and zero (noise-free) is valid here; numpyro's
        # argument validation, on by default from 0.22, would reject it as scale <= 0.
        return npd.Normal(
            loc=f,
            scale=val(self.obs_stddev).astype(f.dtype),
            validate_args=False,
        )

    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> GaussianDistribution:
        r"""Evaluate the Gaussian likelihood at a predictive distribution.

        Preserves diagonal scale when the input carries a
        ``lineax.DiagonalLinearOperator`` (including when wrapped in
        ``lx.TaggedLinearOperator`` as emitted by
        ``DiagonalKernelComputation`` / ``ConstantDiagonalKernelComputation``).
        Always returns ``GaussianDistribution``. This widens the previous return
        type from ``numpyro.distributions.MultivariateNormal`` — see CHANGELOG v0.15.

        Args:
            dist: The Gaussian process posterior at a finite set of test points.

        Returns:
            GaussianDistribution: The predictive distribution with observation
            noise added to the diagonal of the covariance.
        """
        obs_var = val(self.obs_stddev) ** 2

        if isinstance(dist, GaussianDistribution):
            diag = _diagonal_scale(dist.scale)
            if diag is not None:
                noisy_scale = lx.DiagonalLinearOperator(lx.diagonal(diag) + obs_var)
                return GaussianDistribution(dist.mean, noisy_scale)

        # Dense fallback — widen return type to GaussianDistribution for API
        # consistency.
        n_data = dist.event_shape[0]
        cov = dist.covariance_matrix
        noisy_cov = cov.at[jnp.diag_indices(n_data)].add(obs_var)
        return GaussianDistribution(dist.mean, lx.MatrixLinearOperator(noisy_cov))

    def noise_vector(self, n: int) -> Float[Array, " N"]:
        """Per-observation noise variance vector (scalar broadcast for single-output)."""
        return jnp.full(n, jnp.square(val(self.obs_stddev)))

    def prepare_targets(
        self, y: Float[Array, "N 1"], mx: Float[Array, "N 1"]
    ) -> tuple[Float[Array, "N 1"], Float[Array, "N 1"]]:
        """Return targets and mean in the format expected by the unified predict/MLL path."""
        return y, mx


class MultiOutputGaussian(Gaussian):
    """Gaussian likelihood with per-output noise variance.

    Args:
        num_outputs: Number of output dimensions (P).
        obs_stddev: Per-output noise standard deviation. Scalar broadcasts to [P].
    """

    def __init__(
        self,
        num_outputs: int,
        obs_stddev: tp.Union[float, Float[Array, " P"]] = 1.0,
    ):
        if isinstance(obs_stddev, (int, float)):
            obs_stddev = jnp.full(num_outputs, float(obs_stddev))
        super().__init__(
            obs_stddev=NonNegativeReal(jnp.asarray(obs_stddev)),
        )
        self.num_outputs = num_outputs

    def noise_vector(self, n: int) -> Float[Array, " NP"]:
        """Per-observation noise variance in output-major (Kronecker) order.

        Returns sigma_p^2 with each output's variance repeated N times,
        concatenated across outputs: [sigma_1^2...sigma_1^2, sigma_2^2...sigma_2^2, ...].
        """
        per_output_var = jnp.square(val(self.obs_stddev))  # [P]
        return jnp.repeat(per_output_var, n)  # [NP]

    def prepare_targets(
        self, y: Float[Array, "N P"], mx: Float[Array, "N 1"]
    ) -> tuple[Float[Array, "NP 1"], Float[Array, "NP 1"]]:
        """Reshape multi-output targets to output-major long format."""
        P = self.num_outputs
        y_flat = y.T.reshape(-1, 1)  # [N, P] -> [NP, 1]
        mx_flat = jnp.tile(mx, (P, 1))  # [N, 1] -> [NP, 1]
        return y_flat, mx_flat


class HeteroscedasticGaussian(AbstractHeteroscedasticLikelihood):
    def predict(
        self,
        dist: tp.Union[npd.MultivariateNormal, GaussianDistribution],
        noise_dist: tp.Optional[
            tp.Union[npd.MultivariateNormal, GaussianDistribution]
        ] = None,
    ) -> GaussianDistribution:
        if noise_dist is None:
            raise ValueError(
                "noise_dist must be provided for heteroscedastic prediction."
            )

        n_data = dist.event_shape[0]
        noise_mean = noise_dist.mean
        noise_variance = jnp.diag(noise_dist.covariance_matrix)
        noise_stats = self.noise_statistics(
            noise_mean[..., None], noise_variance[..., None]
        )

        cov = dist.covariance_matrix
        noisy_cov = cov.at[jnp.diag_indices(n_data)].add(noise_stats.variance.squeeze())

        return GaussianDistribution(dist.mean, lx.MatrixLinearOperator(noisy_cov))

    def link_function(
        self,
        f: Float[Array, ...],
        g: tp.Optional[Float[Array, ...]] = None,
    ) -> npd.Normal:
        r"""The conditional observation density $p(y \mid f, g)$.

        For a heteroscedastic likelihood the observation noise is itself a
        function of a second latent process $g$, so the conditional is
        $\mathcal{N}(y \mid f, \sigma^2(g))$ (Lázaro-Gredilla & Titsias, 2011).
        Unlike the homoscedastic likelihoods, $f$ alone does not determine the
        density, so `g` is required.

        Args:
            f (Float[Array, "..."]): the latent signal process values.
            g (Float[Array, "..."] | None): the latent noise process values. Required —
                there is no conditional density without it.

        Returns:
            npd.Normal: The observation density given both latent processes.

        Raises:
            ValueError: If `g` is not supplied.
        """
        if g is None:
            raise ValueError(
                f"{type(self).__name__}.link_function requires the noise latent `g` "
                "as well as the signal latent `f`: the observation noise is "
                "sigma^2(g), so p(y | f) alone is not defined. Pass "
                "`link_function(f, g)`, or use `expected_log_likelihood(..., "
                "mean_g=..., variance_g=...)` / `predict(dist, noise_dist)` which "
                "handle the noise process for you."
            )
        return npd.Normal(loc=f, scale=jnp.sqrt(self.noise_transform(g)))

    def expected_log_likelihood(
        self,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        mean_g: tp.Optional[Float[Array, "N D"]] = None,
        variance_g: tp.Optional[Float[Array, "N D"]] = None,
        noise_stats: tp.Optional[NoiseMoments] = None,
        return_parts: bool = False,
        **_: tp.Any,
    ) -> tp.Union[Float[Array, " N"], tuple[Float[Array, " N"], NoiseMoments]]:
        if mean_g is None or variance_g is None:
            raise ValueError(
                "mean_g and variance_g must be provided for heteroscedastic models."
            )

        if noise_stats is None:
            noise_stats = self.noise_statistics(mean_g, variance_g)
        sq_error = jnp.square(y - mean)
        log2pi = jnp.log(2.0 * jnp.pi)
        expected = -0.5 * (
            log2pi
            + noise_stats.log_variance
            + (sq_error + variance) * noise_stats.inv_variance
        )
        expected_sum = jnp.sum(expected, axis=1)
        if return_parts:
            return expected_sum, noise_stats
        return expected_sum

    def supports_tight_bound(self) -> bool:
        return True


class Bernoulli(AbstractLikelihood):
    def link_function(self, f: Float[Array, ...]) -> npd.BernoulliProbs:
        r"""The probit link function of the Bernoulli likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            npd.Bernoulli: The likelihood function.
        """
        return npd.Bernoulli(probs=inv_probit(f))

    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.BernoulliProbs:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist ([npd.MultivariateNormal, GaussianDistribution].): The Gaussian
                process posterior, evaluated at a finite set of test points.

        Returns:
            npd.Bernoulli: The pointwise predictive distribution.
        """
        variance = jnp.diag(dist.covariance_matrix)
        mean = dist.mean.ravel()
        return self.link_function(mean / jnp.sqrt(1.0 + variance))


class Poisson(AbstractLikelihood):
    def link_function(self, f: Float[Array, ...]) -> npd.Poisson:
        r"""The link function of the Poisson likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            npd.Poisson: The likelihood function.
        """
        return npd.Poisson(rate=jnp.exp(f))

    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.Poisson:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist (tp.Union[npd.MultivariateNormal, GaussianDistribution]): The Gaussian
                process posterior, evaluated at a finite set of test points.

        Returns:
            npd.Poisson: The pointwise predictive distribution.
        """
        return self.link_function(dist.mean)


class StudentT(AbstractLikelihood):
    r"""Student's t likelihood object for robust regression.

    Replaces the Gaussian likelihood's light-tailed noise model with a
    heavy-tailed Student's t distribution, so that outlying observations pull
    the posterior mean less strongly (Jylanki, Vanhatalo & Vehtari, 2011).
    Since the Student's t distribution is not conjugate to a Gaussian prior,
    the expected log likelihood has no closed form and is instead computed by
    Gauss-Hermite quadrature via `GHQuadratureIntegrator`.
    """

    degrees_of_freedom: tp.Any
    scale: tp.Any

    def __init__(
        self,
        degrees_of_freedom: tp.Union[ScalarFloat, PositiveReal] = 4.0,
        scale: tp.Union[ScalarFloat, PositiveReal] = 1.0,
        integrator: AbstractIntegrator = GHQuadratureIntegrator(num_points=20),
    ):
        r"""Initializes the Student's t likelihood.

        Args:
            degrees_of_freedom (Union[ScalarFloat, PositiveReal]): the degrees
                of freedom of the Student's t distribution. Lower values give
                heavier tails and hence more robustness to outliers; as the
                degrees of freedom grow large the distribution approaches a
                Gaussian.
            scale (Union[ScalarFloat, PositiveReal]): the scale of the
                Student's t observation noise. Note this is not the
                observation standard deviation: for `degrees_of_freedom > 2`
                the variance is `scale**2 * degrees_of_freedom /
                (degrees_of_freedom - 2)`.
            integrator (AbstractIntegrator): The integrator to be used for computing expected log
                likelihoods. Must be an instance of `AbstractIntegrator`. Defaults to the
                `GHQuadratureIntegrator`, as the Student's t expected log likelihood has no
                closed form.
        """
        if not isinstance(degrees_of_freedom, PositiveReal):
            degrees_of_freedom = PositiveReal(jnp.asarray(degrees_of_freedom))
        self.degrees_of_freedom = degrees_of_freedom

        if not isinstance(scale, PositiveReal):
            scale = PositiveReal(jnp.asarray(scale))
        self.scale = scale

        super().__init__(integrator)

    def link_function(self, f: Float[Array, ...]) -> npd.StudentT:
        r"""The link function of the Student's t likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            npd.StudentT: The likelihood function.
        """
        return npd.StudentT(
            df=val(self.degrees_of_freedom),
            loc=f,
            scale=val(self.scale).astype(f.dtype),
        )

    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.StudentT:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters. As with `Poisson`, the
        Student's t distribution is not conjugate to a Gaussian latent, so
        this evaluates the link function at the posterior mean rather than
        marginalising the latent uncertainty.

        Args:
            dist (tp.Union[npd.MultivariateNormal, GaussianDistribution]): The Gaussian
                process posterior, evaluated at a finite set of test points.

        Returns:
            npd.StudentT: The pointwise predictive distribution.
        """
        return self.link_function(dist.mean)


LatentDistribution = tp.Union[npd.MultivariateNormal, GaussianDistribution]
ParameterLike = tp.Union[float, int, Float[Array, ""], AbstractUnwrappable]

_GEV_SHAPE_BOUNDS = (-0.5, 0.5)


def _check_inside(
    value: tp.Any, low: float, high: tp.Optional[float], name: str, hint: str
) -> None:
    r"""Raise a ``ValueError`` if a concrete ``value`` is not in ``(low, high)``.

    A traced value (inside ``jax.jit``) is not checked.
    """
    try:
        concrete = np.asarray(value)
    except jax.errors.TracerArrayConversionError:
        return
    inside = concrete > low
    if high is not None:
        inside = inside & (concrete < high)
    if not np.all(inside):
        interval = f"({low}, {high})" if high is not None else f"> {low}"
        raise ValueError(
            f"The GEV {name} must be {interval}, not {concrete}. {hint}".rstrip()
        )


def _gev_scale_parameter(scale: ParameterLike) -> AbstractUnwrappable:
    r"""Wrap a GEV scale in a ``PositiveReal``, unless it is a parameter already."""
    if isinstance(scale, AbstractUnwrappable):
        return scale
    _check_inside(scale, 0.0, None, "scale", "")
    return PositiveReal(jnp.asarray(scale))


def _gev_shape_parameter(shape: ParameterLike) -> AbstractUnwrappable:
    r"""Wrap a GEV shape in a ``SigmoidBounded`` on ``(-0.5, 0.5)``.

    A parameter of the user (for example a ``SigmoidBounded`` with other
    bounds, or a fixed value in ``paramax.non_trainable``) is kept as it is.
    """
    if isinstance(shape, AbstractUnwrappable):
        return shape
    low, high = _GEV_SHAPE_BOUNDS
    _check_inside(
        shape,
        low,
        high,
        "shape",
        "For other bounds, pass a parameter, for example "
        "SigmoidBounded(0.7, low=-1.0, high=1.0), or a fixed value, for example "
        "paramax.non_trainable(jnp.array(0.7)).",
    )
    return SigmoidBounded(jnp.asarray(shape), low=low, high=high)


def _latent_draws(
    dist: LatentDistribution, key: KeyArray, num_samples: int
) -> Float[Array, "S N"]:
    r"""Draw ``num_samples`` latent vectors from ``dist``, shape ``(S, N)``."""
    draws = dist.sample(key, (num_samples,))
    return draws.reshape(num_samples, -1)


def _return_level_draws(
    loc: Float[Array, "S N"],
    scale: tp.Union[Float[Array, "S N"], Float[Array, ""]],
    shape: Float[Array, ""],
    return_period: RealLike,
) -> Float[Array, "S ..."]:
    r"""Return levels of shape ``(S, *jnp.shape(return_period), N)``."""
    period = jnp.asarray(return_period, dtype=loc.dtype)
    target = (loc.shape[0],) + (1,) * period.ndim + (loc.shape[1],)
    scale = jnp.reshape(scale, target) if jnp.ndim(scale) == 2 else scale
    return gev_return_level(period[..., None], loc.reshape(target), scale, shape)


class GEV(AbstractLikelihood):
    r"""Generalised extreme value (GEV) likelihood for block maxima.

    The latent Gaussian process gives the GEV location, $\mu(x) = f(x)$, and
    the scale $\sigma$ and shape $\xi$ are parameters that all inputs share:

    .. math::

        y \mid f(x) \sim \operatorname{GEV}(\mu = f(x), \sigma, \xi).

    This is the latent Gaussian model for spatial extremes (Cooley, Nychka &
    Naveau, 2007; Davison, Padoan & Ribatet, 2012). The data are block maxima,
    for example annual maxima of daily rainfall. The shape uses the climate
    convention: $\xi > 0$ is a heavy upper tail, $\xi = 0$ is the Gumbel
    distribution and $\xi < 0$ is a bounded upper tail. See
    :class:`~gpjax.distributions.GeneralisedExtremeValue`.

    ``prior * GEV()`` gives a :class:`~gpjax.gps.NonConjugateModel`. Fit it by
    MAP with :func:`~gpjax.objectives.log_posterior_density`, or fit a
    variational family with :func:`~gpjax.objectives.elbo`.

    **Support.** The GEV has a finite end point for $\xi \neq 0$, and the
    exact log density is $-\infty$ outside the support. Thus
    :meth:`link_function`, which ``log_posterior_density``, ``elbo`` and the
    Gauss–Hermite expected log-likelihood use, returns the distribution with
    ``support_continuation=True``: a pseudo-log-density that is finite and has
    a gradient that points into the support. :meth:`predict` returns the exact
    distribution.

    **Fitting.** Prefer a variational family with ``elbo``. When each latent
    value has one observation, the joint mode of the latent values, the scale
    and the shape does not exist: the latent can interpolate the data, and then
    the scale goes to 0 or the shape goes to its upper bound. Thus MAP with
    ``log_posterior_density`` gives useful estimates of the scale and the
    shape only when inputs repeat, for example many years at each station,
    where all the years of a station share one latent value.

    **Starting values.** The GEV log density has small gradients far from the
    data in the heavy tail, so start the latent location and the scale near
    the data. The Gumbel moment estimates are $\sigma_0 = s\sqrt{6}/\pi$ and
    $\mu_0 = \bar{y} - 0.5772\,\sigma_0$, where $\bar{y}$ and $s$ are the
    sample mean and standard deviation. Use a
    :class:`~gpjax.variational_families.WhitenedVariationalGaussian`, whose
    latent starts at the mean function, with a
    :class:`~gpjax.mean_functions.Constant` mean at $\mu_0$:

    .. code-block:: python

        sigma0 = jnp.std(y) * jnp.sqrt(6.0) / jnp.pi
        mu0 = jnp.mean(y) - 0.5772 * sigma0
        prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Constant(mu0), kernel=kernel
        )
        q = gpx.variational_families.WhitenedVariationalGaussian(
            model=prior * gpx.likelihoods.GEV(scale=sigma0), inducing_inputs=z
        )
        q = eqx.tree_at(
            lambda q: q.inducing_inputs, q, paramax.non_trainable(q.inducing_inputs)
        )

    Keep the inducing inputs fixed, as in the last statement. If they train
    with the step size of the other parameters, the fit can become unstable,
    and its result can then change with the platform.

    The unwhitened :class:`~gpjax.variational_families.VariationalGaussian`
    starts its latent near 0, not at the mean function; give it
    ``variational_mean`` at $\mu_0$.

    **Covariates.** A covariate in the location, for example
    $\mu = \mu_0(x) + \beta\,\mathrm{GMST}(t)$, is an input column: add a
    :class:`~gpjax.kernels.Linear` kernel with ``active_dims`` on that column.

    Example:
        >>> import jax.numpy as jnp
        >>> import gpjax as gpx
        >>> likelihood = gpx.likelihoods.GEV(scale=12.0, shape=0.1)
        >>> prior = gpx.gps.Prior(
        ...     mean_function=gpx.mean_functions.Constant(40.0),
        ...     kernel=gpx.kernels.RBF(),
        ... )
        >>> model = prior * likelihood
        >>> type(model).__name__
        'NonConjugateModel'
    """

    scale: tp.Any
    shape: tp.Any

    def __init__(
        self,
        scale: ParameterLike = 1.0,
        shape: ParameterLike = 0.1,
        integrator: AbstractIntegrator = GHQuadratureIntegrator(num_points=20),
    ):
        r"""Initialise the GEV likelihood.

        Args:
            scale: The GEV scale $\sigma > 0$. A number is wrapped in a
                ``PositiveReal``; a parameter is used as it is.
            shape: The GEV shape $\xi$. A number must be in $(-0.5, 0.5)$ and is
                wrapped in ``SigmoidBounded(shape, low=-0.5, high=0.5)``, so the
                shape stays in that interval during training. A parameter is
                used as it is: pass ``SigmoidBounded`` with other bounds, or a
                fixed value such as ``paramax.non_trainable(jnp.array(0.1))``.
                The value 0 (Gumbel) is valid and trains.
            integrator: The integrator for the expected log-likelihood.
                Defaults to Gauss–Hermite quadrature with 20 points.

        Raises:
            ValueError: If a number for ``scale`` is not positive, or a number
                for ``shape`` is not in $(-0.5, 0.5)$.
        """
        self.scale = _gev_scale_parameter(scale)
        self.shape = _gev_shape_parameter(shape)
        super().__init__(integrator)

    def link_function(self, f: Float[Array, ...]) -> GeneralisedExtremeValue:
        r"""The GEV distribution of the observations, given the location ``f``.

        The distribution uses the support continuation, so its ``log_prob`` is
        finite and has useful gradients for every ``f``; see
        :class:`~gpjax.distributions.GeneralisedExtremeValue`.

        Args:
            f: The latent location values.

        Returns:
            GeneralisedExtremeValue: The distribution of $y$ given $f$.
        """
        return GeneralisedExtremeValue(
            loc=f,
            scale=val(self.scale).astype(f.dtype),
            concentration=val(self.shape).astype(f.dtype),
            support_continuation=True,
            validate_args=False,
        )

    def predict(self, dist: LatentDistribution) -> GeneralisedExtremeValue:
        r"""The plug-in predictive distribution at the latent mean.

        As for :class:`Poisson` and :class:`StudentT`, the latent uncertainty
        is not integrated out: the result is the exact GEV distribution with
        the location at the latent mean. Use :meth:`return_level_samples` for
        return levels that include the latent uncertainty.

        Args:
            dist: The latent Gaussian process at a finite set of test inputs.

        Returns:
            GeneralisedExtremeValue: The exact GEV, one for each test input.
        """
        mean = dist.mean
        return GeneralisedExtremeValue(
            loc=mean,
            scale=val(self.scale).astype(mean.dtype),
            concentration=val(self.shape).astype(mean.dtype),
            validate_args=False,
        )

    def return_level_samples(
        self,
        dist: LatentDistribution,
        return_period: RealLike,
        *,
        key: KeyArray,
        num_samples: int = 1000,
    ) -> Float[Array, "S ..."]:
        r"""Posterior samples of the return level at each test input.

        Each sample is $z_T = f + \sigma c_T(\xi)$, with $f$ a draw of the
        latent location from ``dist`` and $c_T$ the standardised return level
        of :func:`~gpjax.distributions.gev_return_level`. Take quantiles over
        the first axis for credible intervals.

        The samples include the uncertainty of the latent location. They do
        not include the uncertainty of the scale and the shape, which are point
        estimates, or of the kernel hyperparameters. Here the return level is
        the latent location plus a constant, so its distribution is Gaussian,
        with the mean and variance of ``dist`` shifted by $\sigma c_T(\xi)$.
        The samples give the same result, with the same interface as
        :meth:`HeteroscedasticGEV.return_level_samples`.

        If ``dist`` has a dense covariance, every sample is one joint draw over
        the test inputs, so quantities over several inputs (for example a
        regional mean) are correct. A diagonal covariance (from
        ``covariance="diagonal"``) gives independent draws: the marginal
        intervals at each input are correct, but a joint quantity is not.

        Example:
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> import lineax as lx
            >>> import gpjax as gpx
            >>> likelihood = gpx.likelihoods.GEV(scale=12.0, shape=0.1)
            >>> latent = gpx.distributions.GaussianDistribution(
            ...     jnp.array([40.0, 45.0]), lx.MatrixLinearOperator(jnp.eye(2))
            ... )
            >>> levels = likelihood.return_level_samples(
            ...     latent, jnp.array([10.0, 100.0]), key=jr.key(0), num_samples=500
            ... )
            >>> levels.shape
            (500, 2, 2)
            >>> interval = jnp.quantile(levels, jnp.array([0.05, 0.95]), axis=0)

        Args:
            dist: The posterior of the latent location at $N$ test inputs, for
                example ``posterior(test_inputs)``.
            return_period: One return period $T > 1$, or an array of them, in
                blocks (years for annual maxima).
            key: The random key for the latent draws.
            num_samples: The number of samples $S$.

        Returns:
            The return levels, of shape ``(S, N)`` for one return period and
            ``(S, *jnp.shape(return_period), N)`` for an array of them.
        """
        draws = _latent_draws(dist, key, num_samples)
        return _return_level_draws(
            draws,
            val(self.scale).astype(draws.dtype),
            val(self.shape).astype(draws.dtype),
            return_period,
        )

    def return_period_samples(
        self,
        dist: LatentDistribution,
        level: RealLike,
        *,
        key: KeyArray,
        num_samples: int = 1000,
    ) -> Float[Array, "S N"]:
        r"""Posterior samples of the return period of ``level`` at each test input.

        Each sample is $T = 1/P(Y > \text{level})$ for one draw of the latent
        location; see :func:`~gpjax.distributions.gev_return_period`. Where the
        level is above the upper end point of a sample ($\xi < 0$), the
        exceedance probability is 0 and the sample is ``inf``. Keep these
        samples: ``jnp.mean(jnp.isinf(samples), axis=0)`` is the posterior
        probability that the level is impossible, and ``1 / samples`` gives
        exceedance probabilities with exact zeros. The uncertainty is as for
        :meth:`return_level_samples`.

        **Paired samples.** Every sample is one joint draw over the test
        inputs when ``dist`` has a dense covariance. Thus, to compare the same
        location in two climates, put the location twice in the test inputs,
        with the two values of the covariate (for example the global mean
        temperature), and predict with ``covariance="dense"``. The ratio of
        the exceedance probabilities of the two columns is the probability
        ratio, sample by sample:

        .. code-block:: python

            x_pair = jnp.array([[site, gmst_past], [site, gmst_now]])
            periods = likelihood.return_period_samples(
                posterior(x_pair), level=event_amount, key=key
            )
            p_past, p_now = 1.0 / periods[:, 0], 1.0 / periods[:, 1]
            probability_ratio = p_now / p_past  # inf where p_past == 0

        Args:
            dist: The posterior of the latent location at $N$ test inputs.
            level: The level, in the units of the data: a number, or an array
                that broadcasts to ``(N,)`` (for example the observed event
                amount at each location).
            key: The random key for the latent draws.
            num_samples: The number of samples $S$.

        Returns:
            The return periods, of shape ``(S, N)``.
        """
        draws = _latent_draws(dist, key, num_samples)
        return gev_return_period(
            jnp.asarray(level, dtype=draws.dtype),
            draws,
            val(self.scale).astype(draws.dtype),
            val(self.shape).astype(draws.dtype),
        )


class HeteroscedasticGEV(AbstractHeteroscedasticLikelihood):
    r"""GEV likelihood with a latent location and a latent scale.

    Two independent latent Gaussian processes give the GEV location and the
    transformed GEV scale; the shape $\xi$ is a parameter that all inputs
    share:

    .. math::

        y \mid f(x), g(x) \sim \operatorname{GEV}\big(\mu = f(x),\,
        \sigma = h(g(x)),\, \xi\big).

    The transform $h$ is ``noise_transform``. The default,
    :class:`LogNormalTransform`, gives $\sigma = e^{g}$, the log link that is
    usual for the GEV scale. Note the difference from
    :class:`HeteroscedasticGaussian`, where the transform gives a variance:
    here it gives the GEV scale itself. Only the transform is used, not its
    moments.

    Build the model with :class:`~gpjax.gps.HeteroscedasticModel` and fit a
    :class:`~gpjax.variational_families.HeteroscedasticVariationalFamily` with
    :func:`~gpjax.objectives.heteroscedastic_elbo`. This likelihood has no
    closed-form bound, so ``heteroscedastic_elbo`` uses the chained bound, with
    :meth:`expected_log_likelihood`.

    :meth:`link_function` uses the support continuation of
    :class:`~gpjax.distributions.GeneralisedExtremeValue` and :meth:`predict`
    is exact, as for :class:`GEV`.

    **Fitting.** ``HeteroscedasticVariationalFamily`` is not whitened, and its
    fit depends strongly on the start and on the step size. If you use it,
    start the variational means at the Gumbel moment estimates of
    :class:`GEV` (``variational_mean_f`` at $\mu_0$ and
    ``variational_mean_g`` at $\log\sigma_0$ for $\sigma = e^{g}$), with
    constant mean functions at the same values, and use a small step size.
    Two whitened families, whose latents start at their mean functions, are
    easier to fit. Give their diagonal predictions to
    :meth:`expected_log_likelihood`:

    .. code-block:: python

        def negative_elbo(model, data):  # model holds q_f, q_g and likelihood
            f = model.q_f.condition(data)(data.X, covariance="diagonal")
            g = model.q_g.condition(data)(data.X, covariance="diagonal")
            expected = model.likelihood.expected_log_likelihood(
                data.y, f.mean[:, None], f.variance[:, None],
                mean_g=g.mean[:, None], variance_g=g.variance[:, None],
            )
            kl = model.q_f.prior_kl() + model.q_g.prior_kl()
            return kl - jnp.sum(expected) * data.full_size / data.n

    Here ``q_f`` and ``q_g`` are
    :class:`~gpjax.variational_families.WhitenedVariationalGaussian`, with
    constant mean functions at $\mu_0$ and $\log\sigma_0$ and with fixed
    inducing inputs, as for :class:`GEV`. With many
    observations at each input (for example many years at each station),
    gather the moments of each station by a station index. With one
    observation at each input, a location and a scale that both vary are
    weakly identified.

    Example:
        >>> import gpjax as gpx
        >>> prior = gpx.gps.Prior(
        ...     mean_function=gpx.mean_functions.Constant(40.0),
        ...     kernel=gpx.kernels.RBF(),
        ... )
        >>> noise_prior = gpx.gps.Prior(
        ...     mean_function=gpx.mean_functions.Constant(2.5),
        ...     kernel=gpx.kernels.RBF(),
        ... )
        >>> model = gpx.gps.HeteroscedasticModel(
        ...     prior=prior,
        ...     likelihood=gpx.likelihoods.HeteroscedasticGEV(shape=0.1),
        ...     noise_prior=noise_prior,
        ... )
    """

    shape: tp.Any
    num_points: int = eqx.field(static=True, default=20)

    def __init__(
        self,
        shape: ParameterLike = 0.1,
        noise_transform: tp.Union[
            AbstractNoiseTransform,
            tp.Callable[[Float[Array, ...]], Float[Array, ...]],
        ] = LogNormalTransform(),
        num_points: int = 20,
    ):
        r"""Initialise the heteroscedastic GEV likelihood.

        Args:
            shape: The GEV shape $\xi$, as for :class:`GEV`.
            noise_transform: The map from the latent $g$ to the GEV scale.
                Defaults to :class:`LogNormalTransform`, $\sigma = e^{g}$.
            num_points: The number of Gauss–Hermite points in each of the two
                dimensions of the quadrature rule, which thus has
                ``num_points**2`` nodes.

        Raises:
            ValueError: If a number for ``shape`` is not in $(-0.5, 0.5)$.
        """
        self.shape = _gev_shape_parameter(shape)
        self.num_points = num_points
        super().__init__(
            noise_transform=noise_transform,
            integrator=GHQuadratureIntegrator(num_points=num_points),
        )

    def link_function(
        self,
        f: Float[Array, ...],
        g: tp.Optional[Float[Array, ...]] = None,
    ) -> GeneralisedExtremeValue:
        r"""The GEV distribution of the observations, given ``f`` and ``g``.

        Args:
            f: The latent location values.
            g: The latent scale values. Required.

        Returns:
            GeneralisedExtremeValue: The distribution of $y$ given $f$ and $g$,
                with the support continuation.

        Raises:
            ValueError: If ``g`` is not given.
        """
        if g is None:
            raise ValueError(
                f"{type(self).__name__}.link_function requires the latent scale "
                "`g` as well as the latent location `f`. Pass "
                "`link_function(f, g)`, or use `expected_log_likelihood(..., "
                "mean_g=..., variance_g=...)` or `predict(dist, noise_dist)`."
            )
        return GeneralisedExtremeValue(
            loc=f,
            scale=self.noise_transform(g),
            concentration=val(self.shape).astype(f.dtype),
            support_continuation=True,
            validate_args=False,
        )

    def predict(
        self,
        dist: LatentDistribution,
        noise_dist: tp.Optional[LatentDistribution] = None,
    ) -> GeneralisedExtremeValue:
        r"""The plug-in predictive distribution at the latent means.

        The result is the exact GEV with the location at the mean of ``dist``
        and the scale at the transform of the mean of ``noise_dist`` (for
        $\sigma = e^{g}$, this is the posterior median of $\sigma$). The latent
        uncertainty is not integrated out; use :meth:`return_level_samples`
        for that.

        Args:
            dist: The latent location at a finite set of test inputs.
            noise_dist: The latent scale process at the same inputs.

        Returns:
            GeneralisedExtremeValue: The exact GEV, one for each test input.

        Raises:
            ValueError: If ``noise_dist`` is not given.
        """
        if noise_dist is None:
            raise ValueError(
                "noise_dist must be provided for heteroscedastic prediction."
            )
        mean = dist.mean
        return GeneralisedExtremeValue(
            loc=mean,
            scale=self.noise_transform(noise_dist.mean),
            concentration=val(self.shape).astype(mean.dtype),
            validate_args=False,
        )

    def expected_log_likelihood(
        self,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        mean_g: tp.Optional[Float[Array, "N D"]] = None,
        variance_g: tp.Optional[Float[Array, "N D"]] = None,
        **_: tp.Any,
    ) -> Float[Array, " N"]:
        r"""The expected log-likelihood under independent Gaussians of f and g.

        .. math::

            \mathbb{E}_{q(f)\,q(g)}\left[\log p(y \mid f, g)\right],
            \quad q(f) = \mathcal{N}(m_f, s_f),\ q(g) = \mathcal{N}(m_g, s_g),

        by a tensor-product Gauss–Hermite rule with ``num_points**2`` nodes,
        with the continued log density. Other keyword arguments, for example
        the ``noise_stats`` that ``heteroscedastic_elbo`` passes, are ignored.

        The function takes the marginal moments directly, for any number $N$
        of observations, and it can be used under ``jax.jit`` and
        ``jax.grad``. For example, with many years of maxima at each station,
        gather the moments of each station by a station index:
        ``mean[station_index]``. The memory is of order
        ``N * D * num_points**2``.

        Args:
            y: The observations, shape ``(N, D)``.
            mean: The means of the latent location, shape ``(N, D)``.
            variance: The variances of the latent location, shape ``(N, D)``.
            mean_g: The means of the latent scale process, shape ``(N, D)``.
            variance_g: The variances of the latent scale process, shape
                ``(N, D)``.
            **_: Ignored.

        Returns:
            The expected log-likelihood of each observation, summed over the
            ``D`` columns, shape ``(N,)``.

        Raises:
            ValueError: If ``mean_g`` or ``variance_g`` is not given.
        """
        if mean_g is None or variance_g is None:
            raise ValueError(
                "mean_g and variance_g must be provided for heteroscedastic models."
            )
        nodes, weights = np.polynomial.hermite.hermgauss(self.num_points)
        nodes = jnp.asarray(np.sqrt(2.0) * nodes, dtype=mean.dtype)
        weights = jnp.asarray(weights / np.sqrt(np.pi), dtype=mean.dtype)
        f_nodes = mean[..., None] + jnp.sqrt(variance)[..., None] * nodes
        g_nodes = mean_g[..., None] + jnp.sqrt(variance_g)[..., None] * nodes
        log_prob_on_node_grid = self.link_function(
            f_nodes[..., :, None], g_nodes[..., None, :]
        ).log_prob(y[..., None, None])
        expected = jnp.einsum("ndfg,f,g->nd", log_prob_on_node_grid, weights, weights)
        return jnp.sum(expected, axis=1)

    def return_level_samples(
        self,
        dist: LatentDistribution,
        noise_dist: LatentDistribution,
        return_period: RealLike,
        *,
        key: KeyArray,
        num_samples: int = 1000,
    ) -> Float[Array, "S ..."]:
        r"""Posterior samples of the return level at each test input.

        As :meth:`GEV.return_level_samples`, but each sample draws the latent
        location from ``dist`` and, independently, the latent scale process
        from ``noise_dist``, and uses the scale $h(g)$ of that draw. The samples
        include the uncertainty of both latent processes, not of the shape,
        which is a point estimate, or of the kernel hyperparameters.

        With a fitted
        :class:`~gpjax.variational_families.HeteroscedasticVariationalFamily`
        ``q``, use the two predictive distributions of ``q`` and the likelihood
        of ``q.model``, which holds the fitted shape:

        .. code-block:: python

            dist_f, dist_g = q.predict_latents(test_inputs)
            levels = q.model.likelihood.return_level_samples(
                dist_f, dist_g, jnp.array([10.0, 100.0]), key=key
            )

        Args:
            dist: The posterior of the latent location at $N$ test inputs.
            noise_dist: The posterior of the latent scale process at the same
                inputs.
            return_period: One return period $T > 1$, or an array of them.
            key: The random key for the latent draws.
            num_samples: The number of samples $S$.

        Returns:
            The return levels, of shape ``(S, N)`` for one return period and
            ``(S, *jnp.shape(return_period), N)`` for an array of them.
        """
        key_f, key_g = jr.split(key)
        draws_f = _latent_draws(dist, key_f, num_samples)
        draws_g = _latent_draws(noise_dist, key_g, num_samples)
        return _return_level_draws(
            draws_f,
            self.noise_transform(draws_g),
            val(self.shape).astype(draws_f.dtype),
            return_period,
        )

    def return_period_samples(
        self,
        dist: LatentDistribution,
        noise_dist: LatentDistribution,
        level: RealLike,
        *,
        key: KeyArray,
        num_samples: int = 1000,
    ) -> Float[Array, "S N"]:
        r"""Posterior samples of the return period of ``level`` at each test input.

        As :meth:`GEV.return_period_samples` (including ``inf`` samples and
        paired samples), with the latent scale drawn as in
        :meth:`return_level_samples`.

        Args:
            dist: The posterior of the latent location at $N$ test inputs.
            noise_dist: The posterior of the latent scale process at the same
                inputs.
            level: The level: a number, or an array that broadcasts to ``(N,)``.
            key: The random key for the latent draws.
            num_samples: The number of samples $S$.

        Returns:
            The return periods, of shape ``(S, N)``.
        """
        key_f, key_g = jr.split(key)
        draws_f = _latent_draws(dist, key_f, num_samples)
        draws_g = _latent_draws(noise_dist, key_g, num_samples)
        return gev_return_period(
            jnp.asarray(level, dtype=draws_f.dtype),
            draws_f,
            self.noise_transform(draws_g),
            val(self.shape).astype(draws_f.dtype),
        )


def inv_probit(x: Float[Array, " *N"]) -> Float[Array, " *N"]:
    r"""Compute the inverse probit function.

    Args:
        x (``Float[Array, "*N"]``): A vector of values.

    Returns:
        ``Float[Array, "*N"]``: The inverse probit of the input vector.
    """
    jitter = 1e-3  # To ensure output is in interval (0, 1).
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter) + jitter


NonGaussian = tp.Union[Poisson, Bernoulli, StudentT, GEV]

__all__ = [
    "GEV",
    "AbstractHeteroscedasticLikelihood",
    "AbstractLikelihood",
    "AbstractNoiseTransform",
    "Bernoulli",
    "Gaussian",
    "HeteroscedasticGEV",
    "HeteroscedasticGaussian",
    "LogNormalTransform",
    "MultiOutputGaussian",
    "NoiseMoments",
    "NonGaussian",
    "Poisson",
    "SoftplusTransform",
    "StudentT",
    "inv_probit",
]
