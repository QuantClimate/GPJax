# Copyright 2026 The GPJax Contributors. All Rights Reserved.
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
r"""Predictively oriented posteriors for GPs with a Gaussian likelihood.

Lau, Knoblauch and Sharrock (2026), *Predictively Oriented Gaussian Process
Posteriors*, arXiv:2610.03201.

The Bayesian posterior of a GP minimises the expected log score of single
functions. The predictively oriented posterior keeps the prior and the
likelihood, but minimises the log score of the predictive distribution:

.. math::

    \tau_n = \arg\min_{\tau} \Big\{ -\sum_{i=1}^n \log \int
    \mathcal{N}(y_i;\, u_i,\, \sigma^2)\, \tau(\mathrm{d}\mathbf{u})
    + \mathrm{KL}\big(\tau \,\|\, \pi\big) \Big\}.

Here :math:`\mathbf{u}` holds the function values at the particle inputs,
:math:`\pi` is the GP prior at these inputs, and :math:`\sigma` is the
observation noise standard deviation. :func:`condition` represents
:math:`\tau_n` (tau) with particles from the paper's Replica-Gibbs sampler. The
predictive at a test input is thus an equal-weight mixture of Gaussians, one for
each particle.

:func:`condition` makes an existing GP robust to misspecification: it keeps the
kernel hyperparameters of the model and fits only :math:`\sigma` again.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import (
    cho_solve,
    solve_triangular,
)
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from jaxtyping import Float
import lineax as lx
import numpyro.distributions as npd
import optax as ox

from gpjax.dataset import Dataset
from gpjax.gps import (
    ConjugateModel,
    Prior,
)
from gpjax.likelihoods import Gaussian
from gpjax.linalg.utils import stabilised_cholesky
from gpjax.parameters import (
    NonNegativeReal,
    val,
)
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)

__all__ = ["PredictivelyOrientedPosterior", "condition"]


class PredictivelyOrientedPosterior(eqx.Module):
    r"""A predictively oriented GP posterior, represented by particles.

    Make one with :func:`condition`. Each particle is a vector of function
    values at ``particle_inputs``. Given particle :math:`k`, the function at a
    test input :math:`x` is Gaussian, with the mean and variance of the GP prior
    conditioned on the particle. The posterior is the equal-weight mixture of
    these Gaussians.

    Attributes:
        prior: The GP prior of the model, unchanged.
        likelihood: The Gaussian likelihood of the model, with the observation
            noise standard deviation that :func:`condition` fitted.
        particle_inputs: The inputs at which the particles hold function values:
            the training inputs, or the inducing inputs.
        particles: The function values of each particle at ``particle_inputs``.
            There is one row for each particle of each final sweep.
        noise_trace: The observation noise standard deviation after each
            adaptation sweep. Use it to check that the fit converged.
    """

    prior: Prior
    likelihood: Gaussian
    particle_inputs: Float[Array, "M D"]
    particles: Float[Array, "P M"]
    noise_trace: Float[Array, " T"]

    def __call__(self, test_inputs: Float[Array, "N D"]) -> npd.MixtureSameFamily:
        r"""Return the posterior of the latent function at ``test_inputs``.

        Args:
            test_inputs: The inputs at which to evaluate the posterior.

        Returns:
            npd.MixtureSameFamily: The marginal distribution of the function at
                each test input, with batch shape ``(N,)``. It is an equal-weight
                mixture of one Gaussian for each particle.
        """
        means, variance = self._component_moments(test_inputs)
        return _equal_weight_mixture(means, variance)

    def predict_observations(
        self, test_inputs: Float[Array, "N D"]
    ) -> npd.MixtureSameFamily:
        r"""Return the predictive distribution of observations at ``test_inputs``.

        This is the latent posterior with the observation noise variance added
        to each Gaussian of the mixture. The paper evaluates this distribution.

        Args:
            test_inputs: The inputs at which to predict.

        Returns:
            npd.MixtureSameFamily: The marginal predictive distribution at each
                test input, with batch shape ``(N,)``.
        """
        means, variance = self._component_moments(test_inputs)
        noise_variance = val(self.likelihood.obs_stddev) ** 2
        return _equal_weight_mixture(means, variance + noise_variance)

    def _component_moments(
        self, test_inputs: Float[Array, "N D"]
    ) -> tuple[Float[Array, "N P"], Float[Array, " N"]]:
        cholesky = _prior_cholesky(self.prior, self.particle_inputs)
        prior_mean = self.prior.mean_function(self.particle_inputs)[:, 0]
        whitened = solve_triangular(
            cholesky, (self.particles - prior_mean).T, lower=True
        ).T
        projection, variance = _conditional(
            self.prior, self.particle_inputs, cholesky, test_inputs
        )
        means = self.prior.mean_function(test_inputs) + (whitened @ projection).T
        return means, variance


def condition(
    model: ConjugateModel,
    train_data: Dataset,
    key: KeyArray,
    *,
    num_particles: int = 50,
    num_adaptation_sweeps: int = 100,
    num_final_sweeps: int = 30,
    validation_fraction: float = 0.3,
    noise_optimiser: ox.GradientTransformation | None = None,
    inducing_inputs: Float[Array, "M D"] | None = None,
) -> PredictivelyOrientedPosterior:
    r"""Condition a GP on data with the predictively oriented posterior.

    The function keeps the prior of ``model``, with its kernel
    hyperparameters, and fits the observation noise standard deviation
    :math:`\sigma` (sigma) again. It follows the paper's algorithm "PrO-GP
    hyperparameter optimisation", with no kernel updates:

    1. Split ``train_data`` at random into a fit part and a validation part.
    2. Adaptation: for ``num_adaptation_sweeps`` sweeps, run one Replica-Gibbs
       sweep on the fit part, then take one step of ``noise_optimiser`` on
       :math:`\log\sigma`. The step minimises the mean negative log predictive
       density of the validation part. A separate validation part is
       necessary: if the particles and :math:`\sigma` use the same data,
       :math:`\sigma` goes to zero.
    3. Final sweeps: with :math:`\sigma` fixed, run ``num_final_sweeps`` sweeps
       on all of ``train_data`` and keep the particles of each sweep.

    Each observation has ``num_particles`` copies in the allocation step. This
    is the paper's setting :math:`\lambda_n = n` (lambda, the weight of the data
    term), which gives the prior the same weight as in the Bayesian posterior.

    The paper does not say which data the final sweeps use. This function uses
    all of ``train_data``, so the posterior uses every observation. For the
    exact variant, it extends each particle from the fit inputs to all the
    training inputs with a draw from the GP prior, conditioned on the particle.

    Args:
        model: A GP with a Gaussian likelihood, usually with kernel
            hyperparameters fitted by maximum marginal likelihood. The noise
            standard deviation of the likelihood is the first value of
            :math:`\sigma`, and must be one scalar for all observations.
        train_data: The data, with one output.
        key: The random key for the split and the sampler.
        num_particles: The number of particles :math:`K`.
        num_adaptation_sweeps: The number of sweeps that also update
            :math:`\sigma`.
        num_final_sweeps: The number of sweeps with :math:`\sigma` fixed. The
            posterior keeps the particles of all of them, so it has
            ``num_particles * num_final_sweeps`` particles.
        validation_fraction: The fraction of ``train_data`` in the validation
            part.
        noise_optimiser: The Optax optimiser for :math:`\log\sigma`. The
            default is ``optax.adam(0.1)``.
        inducing_inputs: The inputs of the particles for the sparse variant. If
            ``None``, the particles are at the training inputs (the exact
            variant). Each sweep of the exact variant costs
            :math:`O(K n^3)` time for :math:`n` observations.

    Returns:
        PredictivelyOrientedPosterior: The posterior particles, the prior and
            the likelihood with the fitted :math:`\sigma`.

    Raises:
        TypeError: If ``model`` is not a GP with a Gaussian likelihood.
        ValueError: If ``train_data`` has more than one output, the likelihood
            has more than one noise standard deviation, or the validation part
            or the fit part would be empty.

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gpjax.experimentation import predictively_oriented
        >>> x = jnp.linspace(-2.0, 2.0, 40).reshape(-1, 1)
        >>> data = gpx.Dataset(X=x, y=jnp.sin(3.0 * x))
        >>> prior = gpx.gps.Prior(
        ...     mean_function=gpx.mean_functions.Zero(), kernel=gpx.kernels.RBF()
        ... )
        >>> model = prior * gpx.likelihoods.Gaussian(obs_stddev=0.1)
        >>> posterior = predictively_oriented.condition(
        ...     model, data, jr.key(0), num_particles=10,
        ...     num_adaptation_sweeps=20, num_final_sweeps=5,
        ... )
        >>> posterior.particles.shape
        (50, 40)
        >>> posterior.predict_observations(x).batch_shape
        (40,)
    """
    num_validation = _validation_size(model, train_data, validation_fraction)
    num_data = train_data.n
    initial_obs_stddev = val(model.likelihood.obs_stddev)
    if noise_optimiser is None:
        noise_optimiser = ox.adam(0.1)

    prior = model.prior
    inputs = train_data.X
    residual = (train_data.y - prior.mean_function(inputs))[:, 0]
    split_key, initial_key, adaptation_key, extension_key, final_key = jr.split(key, 5)

    # The fit part comes first in `order`. For the exact variant, the final
    # particle inputs are `inputs[order]`, so the Cholesky factor of the fit
    # part is the top-left block of the final Cholesky factor.
    order = jr.permutation(split_key, num_data)
    fit_index = order[: num_data - num_validation]
    validation_index = order[num_data - num_validation :]
    if inducing_inputs is None:
        adaptation_inputs = inputs[fit_index]
        final_inputs = inputs[order]
    else:
        adaptation_inputs = final_inputs = inducing_inputs

    # Adaptation: Replica-Gibbs sweeps on the fit part, noise steps on the
    # validation part.
    cholesky = _prior_cholesky(prior, adaptation_inputs)
    fit_projection, fit_variance = _conditional(
        prior, adaptation_inputs, cholesky, inputs[fit_index]
    )
    validation_projection, validation_variance = _conditional(
        prior, adaptation_inputs, cholesky, inputs[validation_index]
    )
    fit_residual = residual[fit_index]
    validation_residual = residual[validation_index]

    def validation_score(log_obs_stddev, particles):
        return _mean_negative_log_predictive_density(
            particles,
            validation_projection,
            validation_variance,
            validation_residual,
            jnp.exp(log_obs_stddev),
        )

    def adaptation_step(carry, step_key):
        particles, log_obs_stddev, optimiser_state = carry
        particles = _gibbs_sweep(
            step_key,
            particles,
            fit_projection,
            fit_variance,
            fit_residual,
            jnp.exp(log_obs_stddev),
            num_particles,
        )
        gradient = jax.grad(validation_score)(log_obs_stddev, particles)
        updates, optimiser_state = noise_optimiser.update(
            gradient, optimiser_state, log_obs_stddev
        )
        log_obs_stddev = ox.apply_updates(log_obs_stddev, updates)
        return (particles, log_obs_stddev, optimiser_state), jnp.exp(log_obs_stddev)

    initial_log_obs_stddev = jnp.log(initial_obs_stddev)
    initial_particles = jr.normal(
        initial_key, (num_particles, adaptation_inputs.shape[0])
    )
    (particles, log_obs_stddev, _), noise_trace = jax.lax.scan(
        adaptation_step,
        (
            initial_particles,
            initial_log_obs_stddev,
            noise_optimiser.init(initial_log_obs_stddev),
        ),
        jr.split(adaptation_key, num_adaptation_sweeps),
    )
    obs_stddev = jnp.exp(log_obs_stddev)

    # Final sweeps on all the data. In whitened coordinates, new standard normal
    # entries extend a particle to the validation inputs with a draw from the
    # prior conditioned on the particle.
    if inducing_inputs is None:
        extension = jr.normal(extension_key, (num_particles, num_validation))
        particles = jnp.concatenate([particles, extension], axis=1)
        cholesky = _prior_cholesky(prior, final_inputs)
    projection, variance = _conditional(prior, final_inputs, cholesky, inputs)

    def final_step(particles, step_key):
        particles = _gibbs_sweep(
            step_key,
            particles,
            projection,
            variance,
            residual,
            obs_stddev,
            num_particles,
        )
        return particles, particles

    _, kept = jax.lax.scan(final_step, particles, jr.split(final_key, num_final_sweeps))
    whitened = kept.reshape(-1, final_inputs.shape[0])
    function_values = prior.mean_function(final_inputs)[:, 0] + whitened @ cholesky.T

    likelihood = eqx.tree_at(
        lambda likelihood: likelihood.obs_stddev,
        model.likelihood,
        NonNegativeReal(obs_stddev),
    )
    return PredictivelyOrientedPosterior(
        prior=prior,
        likelihood=likelihood,
        particle_inputs=final_inputs,
        particles=function_values,
        noise_trace=noise_trace,
    )


def _validation_size(
    model: ConjugateModel, train_data: Dataset, validation_fraction: float
) -> int:
    """Check the arguments of :func:`condition` and return the validation size."""
    if not isinstance(model, ConjugateModel) or not isinstance(
        model.likelihood, Gaussian
    ):
        raise TypeError(
            "The predictively oriented posterior needs a GP with a Gaussian "
            f"likelihood; got {type(model).__name__} with "
            f"{type(model.likelihood).__name__}."
        )
    if train_data.y.shape[1] != 1:
        raise ValueError(f"The data must have one output; got {train_data.y.shape[1]}.")
    obs_stddev = val(model.likelihood.obs_stddev)
    if jnp.ndim(obs_stddev) != 0:
        raise ValueError(
            "The likelihood must have one noise standard deviation for all "
            f"observations; got shape {jnp.shape(obs_stddev)}."
        )
    num_data = train_data.n
    num_validation = round(validation_fraction * num_data)
    if not 0 < num_validation < num_data:
        raise ValueError(
            f"validation_fraction={validation_fraction} gives {num_validation} "
            f"of {num_data} observations for validation; the validation part "
            "and the fit part must both have at least one observation."
        )
    return num_validation


def _prior_cholesky(
    prior: Prior, particle_inputs: Float[Array, "M D"]
) -> Float[Array, "M M"]:
    """Return the lower Cholesky factor L of the prior covariance K = L Lᵀ."""
    gram = prior.kernel.gram(particle_inputs).as_matrix()
    return stabilised_cholesky(gram, prior.jitter)


def _conditional(
    prior: Prior,
    particle_inputs: Float[Array, "M D"],
    cholesky: Float[Array, "M M"],
    inputs: Float[Array, "N D"],
) -> tuple[Float[Array, "M N"], Float[Array, " N"]]:
    """Return the projection and the conditional variance at ``inputs``.

    The projection B = L⁻¹ K(particle_inputs, inputs) maps a whitened particle
    z to the conditional mean Bᵀz of the zero-mean prior at ``inputs``. The
    conditional variance is s(x, x) = k(x, x) - ‖Bₓ‖².
    """
    kernel = prior.kernel
    cross_covariance = kernel.cross_covariance(particle_inputs, inputs)
    projection = solve_triangular(cholesky, cross_covariance, lower=True)
    prior_variance = lx.diagonal(kernel.diagonal(inputs))
    variance = prior_variance - jnp.sum(projection**2, axis=0)
    return projection, jnp.maximum(variance, 0.0)


def _gibbs_sweep(
    key: KeyArray,
    particles: Float[Array, "K M"],
    projection: Float[Array, "M N"],
    conditional_variance: Float[Array, " N"],
    residual: Float[Array, " N"],
    obs_stddev: ScalarFloat,
    num_copies: int,
) -> Float[Array, "K M"]:
    """Run one Replica-Gibbs sweep on whitened particles.

    The allocation step shares ``num_copies`` copies of each observation among
    the particles, in proportion to the predictive density of each particle at
    the observation. Given these counts, the Gaussian step draws each particle
    from its exact conditional distribution.
    """
    noise_variance = conditional_variance + obs_stddev**2
    log_density = norm.logpdf(
        residual, particles @ projection, jnp.sqrt(noise_variance)
    )

    allocation_key, draw_key = jr.split(key)
    responsibilities = jax.nn.softmax(log_density, axis=0).T
    counts = jr.multinomial(allocation_key, num_copies, responsibilities)
    weights = counts / noise_variance[:, None]

    def draw(arguments):
        weight, standard_normal = arguments
        precision = jnp.eye(projection.shape[0]) + (projection * weight) @ projection.T
        factor = jnp.linalg.cholesky(precision)
        mean = cho_solve((factor, True), projection @ (weight * residual))
        return mean + solve_triangular(factor.T, standard_normal, lower=False)

    # `lax.map` draws one particle at a time, so the memory for the precision
    # matrices does not grow with the number of particles.
    standard_normal = jr.normal(draw_key, particles.shape, dtype=particles.dtype)
    return jax.lax.map(draw, (weights.T, standard_normal))


def _mean_negative_log_predictive_density(
    particles: Float[Array, "K M"],
    projection: Float[Array, "M N"],
    conditional_variance: Float[Array, " N"],
    residual: Float[Array, " N"],
    obs_stddev: ScalarFloat,
) -> ScalarFloat:
    """Return the mean of -log p(y) under the equal-weight particle mixture."""
    scale = jnp.sqrt(conditional_variance + obs_stddev**2)
    log_density = norm.logpdf(residual, particles @ projection, scale)
    num_particles = particles.shape[0]
    return -jnp.mean(logsumexp(log_density, axis=0) - jnp.log(num_particles))


def _equal_weight_mixture(
    means: Float[Array, "N P"], variance: Float[Array, " N"]
) -> npd.MixtureSameFamily:
    """Return the equal-weight mixture of Gaussians with these moments."""
    num_components = means.shape[1]
    weights = jnp.full((num_components,), 1.0 / num_components, dtype=means.dtype)
    # The latent variance is zero at a particle input of the exact variant.
    # NumPyro's argument validation would reject this scale of zero.
    components = npd.Normal(
        loc=means, scale=jnp.sqrt(variance)[:, None], validate_args=False
    )
    return npd.MixtureSameFamily(npd.Categorical(probs=weights), components)
