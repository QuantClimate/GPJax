# Copyright 2026 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""The deep conditioning module.

Conditioning a Gaussian process on data is one computation: stabilise and
factor the training covariance once, then derive every downstream quantity —
the predictive distribution, the log marginal likelihood (evidence),
leave-one-out densities, and pathwise samples — as views of that single
factorisation. This module is the one home of that algebra.

A :class:`Posterior` is an immutable pytree produced by
``model.condition(train_data)`` (equivalently ``model | train_data``). It
caches the Cholesky factor and representer weights, so repeated queries never
re-factorise.
"""

from abc import abstractmethod
import typing as tp
from typing import Literal
import warnings

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import (
    Float,
    Num,
)
import lineax as lx
import numpyro.distributions as npd

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.kernels import RFF
from gpjax.linalg.custom_operators import BlockDiag
from gpjax.linalg.utils import stabilised_cholesky
from gpjax.parameters import val
from gpjax.typing import (
    Array,
    FunctionalSample,
    KeyArray,
    ScalarFloat,
)


def _prior_covariance(
    kernel: tp.Any,
    test_inputs: Num[Array, "N D"],
    covariance: Literal["dense", "diagonal"],
) -> Float[Array, "..."]:
    r"""The prior gram :math:`K_{tt}`, shape ``(N, N)``, when ``covariance``
    is ``"dense"``; otherwise its diagonal, shape ``(N,)``.
    """
    if covariance == "dense":
        return kernel.gram(test_inputs).as_matrix()
    return lx.diagonal(kernel.diagonal(test_inputs))


def _assemble_predictive(
    mean: Float[Array, "N 1"],
    prior_covariance: Float[Array, "..."],
    *,
    covariance: Literal["dense", "diagonal"],
    subtracted_factor: Float[Array, "K N"],
    added_factor: Float[Array, "J N"] | None = None,
    jitter: ScalarFloat,
) -> GaussianDistribution:
    r"""Assemble a predictive distribution from a mean and covariance factors.

    Every conditioned process in this module ends the same way: the predictive
    covariance is the prior covariance at the test inputs, less the Gram of one
    factor and — for the sparse families — plus the Gram of another, with the
    model's jitter stabilising the diagonal. Only the factors differ between
    processes, so only they are passed in. A factor :math:`F` of shape
    ``(K, N)`` over the ``N`` test inputs contributes :math:`F^{\top}F`
    densely, or the squared column norms :math:`\sum_{k} F_{kn}^2` diagonally.

    Args:
        mean: The predictive mean, of shape ``(N, 1)``.
        prior_covariance: :math:`K_{tt}` or its diagonal, whichever matches
            ``covariance``; see :func:`_prior_covariance`.
        covariance: Whether to build the dense joint covariance over the test
            inputs or only the marginal (diagonal) variances.
        subtracted_factor: Factor whose Gram is subtracted — the uncertainty
            conditioning removes.
        added_factor: Optional factor whose Gram is added — the variational
            family's own covariance, propagated to the test inputs.
        jitter: Added to the diagonal for numerical stability.

    Returns:
        GaussianDistribution: The predictive distribution at the test inputs.
    """
    if covariance == "dense":
        predictive_cov = prior_covariance - jnp.matmul(
            subtracted_factor.T, subtracted_factor
        )
        if added_factor is not None:
            predictive_cov = predictive_cov + jnp.matmul(added_factor.T, added_factor)
        predictive_cov = predictive_cov + jitter * jnp.eye(predictive_cov.shape[0])
        scale = lx.MatrixLinearOperator(predictive_cov)
    else:
        marginal_var = prior_covariance - jnp.einsum(
            "ij,ij->j", subtracted_factor, subtracted_factor
        )
        if added_factor is not None:
            marginal_var = marginal_var + jnp.einsum(
                "ij,ij->j", added_factor, added_factor
            )
        marginal_var = marginal_var + jitter
        scale = lx.DiagonalLinearOperator(jnp.atleast_1d(marginal_var.squeeze()))

    return GaussianDistribution(loc=jnp.atleast_1d(mean.squeeze()), scale=scale)


def _stack_predictives(
    predictives: list[GaussianDistribution],
    *,
    covariance: Literal["dense", "diagonal"],
) -> GaussianDistribution:
    r"""Join the predictives of ``L`` independent latent functions.

    The joint mean is flattened in blocked order: index :math:`l N + n` holds
    latent function :math:`l` at test input :math:`n`. The latent functions
    are independent, so the joint covariance is block diagonal, with one
    :math:`N \times N` block for each latent function.

    Args:
        predictives: The predictive of each latent function, over the same
            ``N`` test inputs.
        covariance: Whether the predictives hold dense covariances or only the
            marginal variances.

    Returns:
        GaussianDistribution: The joint predictive over ``N * L`` values. With
            one latent function, this is ``predictives[0]``.
    """
    if len(predictives) == 1:
        return predictives[0]
    loc = jnp.concatenate([predictive.mean for predictive in predictives])
    if covariance == "dense":
        scale = BlockDiag([predictive.scale for predictive in predictives])
    else:
        scale = lx.DiagonalLinearOperator(
            jnp.concatenate([predictive.variance for predictive in predictives])
        )
    return GaussianDistribution(loc=loc, scale=scale)


class Posterior(eqx.Module):
    r"""A conditioned Gaussian process, :math:`p(f \mid \mathcal{D})`.

    The result of conditioning a joint model on data. Immutable: the
    factorisation of the training covariance is computed once at
    ``condition`` time and cached on this object; every query is a view of
    it. Query the process at test inputs by calling it::

        posterior = model.condition(train_data)   # or: model | train_data
        predictive = posterior(test_inputs)
    """

    @abstractmethod
    def __call__(
        self,
        test_inputs: Num[Array, "N D"],
        *,
        covariance: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Evaluate the conditioned process at the given test inputs.

        Args:
            test_inputs: Input locations at which to query the process.
            covariance: Whether to return the dense joint covariance over the
                test inputs or only the marginal (diagonal) variances.

        Returns:
            GaussianDistribution: The predictive distribution at the inputs.
        """
        raise NotImplementedError

    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: Dataset | None = None,
        *,
        covariance: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Sugar for calling the posterior: ``predict(t) == self(t)``.

        Retained for signature compatibility with the pre-v1.0 API;
        ``train_data`` is accepted and ignored — this process is already
        conditioned on its training set.
        """
        del train_data
        return self(test_inputs, covariance=covariance)


class ExactPosterior(Posterior):
    r"""Exactly conditioned GP: a Gaussian likelihood integrated analytically.

    Caches the lower Cholesky factor of
    :math:`\Sigma = K_{xx} + \texttt{jitter}\,\mathbf{I} + \mathrm{diag}(\sigma^2)`
    and the representer weights :math:`\alpha = \Sigma^{-1}(y - m(x))`. The
    predictive moments, the evidence, LOO densities, and pathwise samples are
    all views of these two objects.
    """

    prior: tp.Any
    likelihood: tp.Any
    train_data: Dataset
    cholesky_factor: Float[Array, "NP NP"]
    representer_weights: Float[Array, "NP 1"]
    residual: Float[Array, "NP 1"]
    log_marginal_likelihood: ScalarFloat

    def __init__(self, prior: tp.Any, likelihood: tp.Any, train_data: Dataset):
        from gpjax.kernels.multioutput.base import MultiOutputKernel

        kernel = prior.kernel
        if isinstance(kernel, MultiOutputKernel):
            if not train_data.multi_output:
                raise ValueError("MultiOutputKernel requires multi-output data.")
            if train_data.num_outputs != kernel.num_outputs:
                raise ValueError(
                    f"Dataset has {train_data.num_outputs} outputs "
                    f"but kernel expects {kernel.num_outputs}."
                )

        x, y = train_data.X, train_data.y
        mean_x = prior.mean_function(x)
        y_flat, mean_flat = likelihood.prepare_targets(y, mean_x)
        noise = likelihood.noise_vector(train_data.n)

        gram_plus_noise = kernel.gram(x).as_matrix() + jnp.diag(noise)
        factor = stabilised_cholesky(gram_plus_noise, prior.jitter)
        residual = (y_flat - mean_flat).reshape(-1, 1)
        weights = jsp.linalg.cho_solve((factor, True), residual)

        num_scalars = residual.shape[0]
        half_logdet = jnp.sum(jnp.log(jnp.diagonal(factor)))
        evidence = (
            -0.5 * (jnp.sum(residual * weights) + num_scalars * jnp.log(2.0 * jnp.pi))
            - half_logdet
        )

        self.prior = prior
        self.likelihood = likelihood
        self.train_data = train_data
        self.cholesky_factor = factor
        self.representer_weights = weights
        self.residual = residual
        self.log_marginal_likelihood = jnp.squeeze(evidence)

    def __call__(
        self,
        test_inputs: Num[Array, "N D"],
        *,
        covariance: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        kernel = self.prior.kernel
        x = self.train_data.X
        num_outputs = self.likelihood.num_outputs

        cross_cov = kernel.cross_covariance(x, test_inputs)
        solved_cross = jsp.linalg.solve_triangular(
            self.cholesky_factor, cross_cov, lower=True
        )

        mean_test_raw = self.prior.mean_function(test_inputs)
        mean_test = (
            jnp.tile(mean_test_raw, (num_outputs, 1))
            if num_outputs > 1
            else mean_test_raw
        )
        mean = mean_test + jnp.matmul(cross_cov.T, self.representer_weights)

        if covariance == "diagonal" and num_outputs > 1:
            warnings.warn(
                "Diagonal covariance is not yet supported for multi-output GPs. "
                "Returning full covariance.",
                stacklevel=2,
            )
            covariance = "dense"

        return _assemble_predictive(
            mean,
            _prior_covariance(kernel, test_inputs, covariance),
            covariance=covariance,
            subtracted_factor=solved_cross,
            jitter=self.prior.jitter,
        )

    def loo(self) -> Float[Array, " NP"]:
        r"""Per-point leave-one-out predictive log-densities.

        Computed from the cached factor via Rasmussen & Williams eq. 5.12 —
        no model is refit. Sum the result for the LOOCV objective.
        """
        factor = self.cholesky_factor
        num_scalars = factor.shape[0]
        factor_inv = jsp.linalg.solve_triangular(
            factor, jnp.eye(num_scalars), lower=True
        )
        precision_diag = jnp.sum(factor_inv**2, axis=0).reshape(-1, 1)

        loo_means = self.residual - self.representer_weights / precision_diag
        loo_vars = 1.0 / precision_diag
        loo_dist = npd.Normal(loc=loo_means, scale=jnp.sqrt(loo_vars))
        return loo_dist.log_prob(self.residual).squeeze(-1)

    def sample_approx(
        self,
        num_samples: int,
        key: KeyArray,
        num_features: int | None = 100,
    ) -> FunctionalSample:
        r"""Draw approximate posterior samples via pathwise conditioning.

        Decomposes each sample into Fourier features of the prior plus
        canonical features weighted through the cached training factor
        (Wilson et al., 2020).

        Args:
            num_samples: The desired number of samples.
            key: The random seed used for the sample(s).
            num_features: The number of Fourier features used to approximate
                the prior component of each sample.

        Returns:
            FunctionalSample: A function evaluating the sample draws at any
                inputs; the same draw is returned for all queries.
        """
        if (not isinstance(num_samples, int)) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if self.likelihood.num_outputs > 1:
            raise ValueError(
                "sample_approx does not support multi-output likelihoods yet."
            )

        freq_key, weight_key, noise_key = jr.split(key, 3)
        fourier_feature_fn = _build_fourier_features_fn(
            self.prior, num_features, freq_key
        )
        fourier_weights = jr.normal(weight_key, [num_samples, 2 * num_features])

        x = self.train_data.X
        obs_var = val(self.likelihood.obs_stddev) ** 2
        observation_noise = jnp.sqrt(obs_var) * jr.normal(
            noise_key, [self.train_data.n, num_samples]
        )
        prior_features = fourier_feature_fn(x)
        perturbed_residual = (
            self.residual
            + observation_noise
            - jnp.inner(prior_features, fourier_weights)
        )
        canonical_weights = jsp.linalg.cho_solve(
            (self.cholesky_factor, True), perturbed_residual
        )

        def sample_fn(test_inputs: Float[Array, "n D"]) -> Float[Array, "n B"]:
            fourier_features = fourier_feature_fn(test_inputs)
            weight_space_contribution = jnp.inner(fourier_features, fourier_weights)
            canonical_features = self.prior.kernel.cross_covariance(test_inputs, x)
            function_space_contribution = jnp.matmul(
                canonical_features, canonical_weights
            )
            return (
                self.prior.mean_function(test_inputs)
                + weight_space_contribution
                + function_space_contribution
            )

        return sample_fn


class LatentPosterior(Posterior):
    r"""Approximately conditioned GP for non-Gaussian likelihoods.

    Conditioning is on the model's whitened latent vector rather than on the
    observations directly: the cached factor is the prior gram's Cholesky, and
    the latent plays the role of the representer weights. The joint
    log-density (``log_posterior_density``) is the quantity MCMC or MAP
    optimisation targets.
    """

    prior: tp.Any
    likelihood: tp.Any
    train_data: Dataset
    cholesky_factor: Float[Array, "N N"]
    latent: Float[Array, "N L"]

    def __init__(
        self,
        prior: tp.Any,
        likelihood: tp.Any,
        latent: tp.Any,
        train_data: Dataset,
    ):
        self.prior = prior
        self.likelihood = likelihood
        self.train_data = train_data
        self.cholesky_factor = stabilised_cholesky(
            prior.kernel.gram(train_data.X).as_matrix(), prior.jitter
        )
        self.latent = val(latent)

    def __call__(
        self,
        test_inputs: Num[Array, "N D"],
        *,
        covariance: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        kernel = self.prior.kernel
        x = self.train_data.X

        cross_cov = kernel.cross_covariance(x, test_inputs)
        solved_cross = jsp.linalg.solve_triangular(
            self.cholesky_factor, cross_cov, lower=True
        )

        mean_test = self.prior.mean_function(test_inputs)
        mean = mean_test + jnp.matmul(solved_cross.T, self.latent)

        # The latent functions share one kernel, so they share one predictive
        # covariance. Build it once, from the first latent column.
        shared = _assemble_predictive(
            mean[:, :1],
            _prior_covariance(kernel, test_inputs, covariance),
            covariance=covariance,
            subtracted_factor=solved_cross,
            jitter=self.prior.jitter,
        )
        predictives = [
            GaussianDistribution(loc=mean[:, index], scale=shared.scale)
            for index in range(mean.shape[1])
        ]
        return _stack_predictives(predictives, covariance=covariance)

    @property
    def log_posterior_density(self) -> ScalarFloat:
        r"""Unnormalised log-posterior density of the whitened latent model.

        :math:`\log p(y \mid f(x)) + \log \mathcal{N}(w_x \mid 0, I)` where
        :math:`f(x) = m(x) + L_x w_x`.
        """
        mean_x = self.prior.mean_function(self.train_data.X)
        latent_function = mean_x + self.cholesky_factor @ self.latent
        observation_density = self.likelihood.link_function(latent_function)
        whitened_prior = npd.Normal(loc=0.0, scale=1.0)
        return (
            observation_density.log_prob(self.train_data.y).sum()
            + whitened_prior.log_prob(self.latent).sum()
        )


class SparsePosterior(Posterior):
    r"""A sparse variational GP conditioned through its inducing variables.

    The predictive :math:`q(f(\cdot)) = \int p(f(\cdot) \mid u)\,q(u)\,\mathrm{d}u`
    of a Gaussian-output variational family, derived once for both the
    whitened and unwhitened parameterisations. The cached factor is the lower
    Cholesky of the (stabilised) inducing gram :math:`K_{zz}`; the variational
    moments :math:`(\mu, \mathrm{sqrt})` with :math:`S = \mathrm{sqrt}\,
    \mathrm{sqrt}^{\top}` are supplied by the family at ``condition`` time —
    a site-parameterised family converts to moments first.

    Internal: constructed by ``AbstractVariationalGaussian.condition``, not
    by users directly. The family is retained for its shape hooks
    (``_fmt_Kzt_Ktt``, ``_fmt_inducing_inputs``), which adapt kernel matrices
    for non-Euclidean index sets.
    """

    family: tp.Any
    inducing_inputs: tp.Any
    cholesky_kzz: Float[Array, "M M"]
    variational_mean: Float[Array, "M L"]
    variational_root: Float[Array, "... M M"]
    whitened: bool = eqx.field(static=True)

    def __init__(
        self,
        family: tp.Any,
        variational_mean: Float[Array, "M L"],
        variational_root: Float[Array, "... M M"],
        *,
        whitened: bool,
    ):
        model = family.model
        inducing_inputs = family._fmt_inducing_inputs()
        kzz_dense = model.prior.kernel.gram(inducing_inputs).as_matrix()

        self.family = family
        self.inducing_inputs = inducing_inputs
        self.cholesky_kzz = stabilised_cholesky(kzz_dense, model.prior.jitter)
        self.variational_mean = variational_mean
        self.variational_root = variational_root
        self.whitened = whitened

    @property
    def model(self) -> tp.Any:
        """The joint model the family approximates the posterior of."""
        return self.family.model

    def __call__(
        self,
        test_inputs: Num[Array, "N D"],
        *,
        covariance: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Evaluate the sparse predictive at the given test inputs.

        With :math:`A = L_z^{-1} K_{zt}`, the whitened parameterisation gives
        mean :math:`m(t) + A^{\top}\mu` and root term
        :math:`\mathrm{sqrt}^{\top} A`; the unwhitened one gives mean
        :math:`m(t) + A^{\top} L_z^{-1}(\mu - m(z))` and root term
        :math:`\mathrm{sqrt}^{\top} K_{zz}^{-1} K_{zt}`. In both cases the
        covariance is
        :math:`K_{tt} - A^{\top}A + \mathrm{root}^{\top}\mathrm{root}` plus
        the model's jitter on the diagonal.

        Args:
            test_inputs: Input locations at which to query the process.
            covariance: Whether to return the dense joint covariance over the
                test inputs or only the marginal (diagonal) variances.

        Returns:
            GaussianDistribution: The predictive distribution at the inputs.
        """
        model = self.model
        kernel = model.prior.kernel
        mean_function = model.prior.mean_function
        inducing_inputs = self.inducing_inputs
        cholesky_kzz = self.cholesky_kzz

        cross_cov = kernel.cross_covariance(inducing_inputs, test_inputs)
        test_mean = mean_function(test_inputs)

        prior_cov = _prior_covariance(kernel, test_inputs, covariance)
        if covariance == "dense":
            cross_cov, prior_cov = self.family._fmt_Kzt_Ktt(cross_cov, prior_cov)
        else:
            # Only the cross-covariance needs the family's shape hook here;
            # the placeholder keeps the hook's two-argument contract and its
            # result is discarded.
            cross_cov, _ = self.family._fmt_Kzt_Ktt(cross_cov, jnp.empty((0, 0)))

        # A = Lz^{-1} Kzt
        projected_cross = jsp.linalg.solve_triangular(
            cholesky_kzz, cross_cov, lower=True
        )

        if self.whitened:
            # q(u) = N(m(z) + Lz mu, Lz S Lz^T): mean = m(t) + A^T mu.
            mean = test_mean + jnp.matmul(projected_cross.T, self.variational_mean)
            root_factor = projected_cross
        else:
            # q(u) = N(mu, S): mean = m(t) + Ktz Kzz^{-1} (mu - m(z)).
            inducing_mean = mean_function(inducing_inputs)
            centred_mean = jsp.linalg.solve_triangular(
                cholesky_kzz, self.variational_mean - inducing_mean, lower=True
            )
            mean = test_mean + jnp.matmul(projected_cross.T, centred_mean)
            root_factor = jsp.linalg.solve_triangular(
                cholesky_kzz.T, projected_cross, lower=False
            )

        # One root of shape (M, M) for each latent function: column l of the
        # mean and root l give the predictive of latent function l.
        num_latent_functions = self.variational_mean.shape[1]
        roots = jnp.reshape(
            self.variational_root, (num_latent_functions, *cholesky_kzz.shape)
        )

        # Ktt - A^T A + root^T root  [the S-term, S = sqrt sqrt^T]
        predictives = [
            _assemble_predictive(
                mean[:, [index]],
                prior_cov,
                covariance=covariance,
                subtracted_factor=projected_cross,
                added_factor=jnp.matmul(root.T, root_factor),
                jitter=model.prior.jitter,
            )
            for index, root in enumerate(roots)
        ]
        return _stack_predictives(predictives, covariance=covariance)

    def sample_approx(
        self,
        num_samples: int,
        key: KeyArray,
        num_features: int | None = 100,
    ) -> FunctionalSample:
        r"""Draw approximate posterior samples via decoupled (pathwise) sampling.

        The sparse analogue of :meth:`ExactPosterior.sample_approx`: a prior
        draw built from Fourier features of the kernel, plus a canonical
        correction that pulls the draw onto a draw of the variational
        distribution :math:`q(u)` at the inducing inputs (Wilson et al., 2020,
        section 5.2),

        .. math::

            \hat{f}(\cdot) = \hat{f}_{\text{prior}}(\cdot)
                + K_{\cdot z} K_{zz}^{-1}
                \big(u - \hat{f}_{\text{prior}}(z)\big),
                \qquad u \sim q(u).

        Reuses the cached :math:`K_{zz}` Cholesky factor and variational
        moments; no extra factorisation is performed. For the whitened
        parameterisation, the draw of :math:`u` is de-whitened through the
        same cached factor before entering the correction.

        Args:
            num_samples: The desired number of samples.
            key: The random seed used for the sample(s).
            num_features: The number of Fourier features used to approximate
                the prior component of each sample.

        Returns:
            FunctionalSample: A function evaluating the sample draws at any
                inputs; the same draw is returned for all queries.
        """
        if (not isinstance(num_samples, int)) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        model = self.model
        num_latent_functions = model.likelihood.num_latent_functions
        if num_latent_functions > 1:
            raise ValueError(
                "sample_approx supports one latent function only, but the "
                f"likelihood has {num_latent_functions}."
            )
        if model.likelihood.num_outputs > 1:
            raise ValueError(
                "sample_approx does not support multi-output likelihoods yet."
            )

        freq_key, weight_key, inducing_key = jr.split(key, 3)
        prior = model.prior
        inducing_inputs = self.inducing_inputs
        cholesky_kzz = self.cholesky_kzz
        num_inducing = cholesky_kzz.shape[0]

        fourier_feature_fn = _build_fourier_features_fn(prior, num_features, freq_key)
        fourier_weights = jr.normal(weight_key, [num_samples, 2 * num_features])

        # A draw of q(u) (or, whitened, of q(v)): mu + sqrt @ eps.
        standard_normal = jr.normal(inducing_key, [num_inducing, num_samples])
        site_draws = self.variational_mean + jnp.matmul(
            self.variational_root, standard_normal
        )
        if self.whitened:
            # u = m(z) + Lz v, so u - m(z) = Lz v -- the mean cancels without
            # ever being added, matching the de-meaned residual below.
            centred_inducing_draws = jnp.matmul(cholesky_kzz, site_draws)
        else:
            centred_inducing_draws = site_draws - prior.mean_function(inducing_inputs)

        # Kzz^{-1}(u - f_prior(z)), with f_prior(z) = m(z) + phi(z)^T w:
        # the mean cancels, leaving only the deviation phi(z)^T w to subtract.
        prior_features_z = fourier_feature_fn(inducing_inputs)
        perturbed_residual = centred_inducing_draws - jnp.inner(
            prior_features_z, fourier_weights
        )
        canonical_weights = jsp.linalg.cho_solve(
            (cholesky_kzz, True), perturbed_residual
        )

        def sample_fn(test_inputs: Float[Array, "n D"]) -> Float[Array, "n B"]:
            fourier_features = fourier_feature_fn(test_inputs)
            weight_space_contribution = jnp.inner(fourier_features, fourier_weights)
            canonical_features = prior.kernel.cross_covariance(
                test_inputs, inducing_inputs
            )
            function_space_contribution = jnp.matmul(
                canonical_features, canonical_weights
            )
            return (
                prior.mean_function(test_inputs)
                + weight_space_contribution
                + function_space_contribution
            )

        return sample_fn


class CollapsedPosterior(Posterior):
    r"""The Titsias (2009) collapsed sparse posterior.

    Conditioning solves for the optimal variational distribution over the
    inducing variables analytically, which requires a Gaussian likelihood and
    the full training set. Cached at ``condition`` time: the Cholesky factor
    of the stabilised :math:`K_{zz}`, the scaled projection
    :math:`A = L_z^{-1} K_{zx} / \sigma`, the factor of
    :math:`B = I + AA^{\top}`, the training residual, and the prior variances
    at the training inputs. The predictive and the collapsed evidence bound
    (:attr:`elbo_bound`) are views of these factors.

    Internal: constructed by
    ``CollapsedVariationalGaussian.condition(train_data)``, not by users
    directly.
    """

    family: tp.Any
    inducing_inputs: tp.Any
    cholesky_kzz: Float[Array, "M M"]
    scaled_cross: Float[Array, "M N"]
    cholesky_b: Float[Array, "M M"]
    residual: Float[Array, "N 1"]
    representer_weights: Float[Array, "M 1"]
    observation_variance: ScalarFloat
    kxx_diagonal: Float[Array, " N"]

    def __init__(self, family: tp.Any, train_data: Dataset):
        model = family.model
        kernel = model.prior.kernel
        mean_function = model.prior.mean_function
        obs_stddev = val(model.likelihood.obs_stddev)
        noise_var = obs_stddev**2

        inducing_inputs = family._fmt_inducing_inputs()
        x, y = train_data.X, train_data.y
        num_inducing = inducing_inputs.shape[0]

        kzz_dense = kernel.gram(inducing_inputs).as_matrix()
        cholesky_kzz = stabilised_cholesky(kzz_dense, model.prior.jitter)
        cross_cov = kernel.cross_covariance(inducing_inputs, x)

        # Lz^{-1} Kzx and its noise-scaled form A = Lz^{-1} Kzx / o.
        projected_cross = jsp.linalg.solve_triangular(
            cholesky_kzz, cross_cov, lower=True
        )
        scaled_cross = projected_cross / obs_stddev

        # LL^T = B = I + AA^T
        cholesky_b = jnp.linalg.cholesky(
            jnp.eye(num_inducing) + jnp.matmul(scaled_cross, scaled_cross.T)
        )

        residual = y - mean_function(x)

        # Kzz^{-1} Kzx B_z^{-1} (y - m(x)) / o^2, with B_z = Lz B Lz^T: the
        # representer weights of the collapsed predictive mean.
        projected_residual = jsp.linalg.cho_solve(
            (cholesky_b, True), jnp.matmul(projected_cross, residual)
        )
        representer_weights = (
            jsp.linalg.solve_triangular(cholesky_kzz.T, projected_residual, lower=False)
            / noise_var
        )

        self.family = family
        self.inducing_inputs = inducing_inputs
        self.cholesky_kzz = cholesky_kzz
        self.scaled_cross = scaled_cross
        self.cholesky_b = cholesky_b
        self.residual = residual
        self.representer_weights = representer_weights
        self.observation_variance = noise_var
        # The prior variances at the training inputs enter the bound's trace
        # term. `kernel.diagonal` is the one correct entry point: unlike a
        # vmap of `kernel.__call__`, it works for every kernel.
        self.kxx_diagonal = lx.diagonal(kernel.diagonal(x))

    @property
    def model(self) -> tp.Any:
        """The joint model the family approximates the posterior of."""
        return self.family.model

    def __call__(
        self,
        test_inputs: Num[Array, "N D"],
        *,
        covariance: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Evaluate the collapsed predictive at the given test inputs.

        Args:
            test_inputs: Input locations at which to query the process.
            covariance: Whether to return the dense joint covariance over the
                test inputs or only the marginal (diagonal) variances.

        Returns:
            GaussianDistribution: The predictive distribution at the inputs.
        """
        model = self.model
        kernel = model.prior.kernel
        mean_function = model.prior.mean_function
        cholesky_kzz = self.cholesky_kzz

        cross_cov = kernel.cross_covariance(self.inducing_inputs, test_inputs)
        test_mean = mean_function(test_inputs)

        # m(t) + 1/o^2 Ktz Kzz^{-1} Kzx B_z^{-1} (y - m(x))
        mean = test_mean + jnp.matmul(cross_cov.T, self.representer_weights)

        # Lz^{-1} Kzt and L^{-1} Lz^{-1} Kzt
        projected_cross = jsp.linalg.solve_triangular(
            cholesky_kzz, cross_cov, lower=True
        )
        site_projection = jsp.linalg.solve_triangular(
            self.cholesky_b, projected_cross, lower=True
        )

        # Ktt - Ktz Kzz^{-1} Kzt + Ktz Lz^{-T} B^{-1} Lz^{-1} Kzt
        return _assemble_predictive(
            mean,
            _prior_covariance(kernel, test_inputs, covariance),
            covariance=covariance,
            subtracted_factor=projected_cross,
            added_factor=site_projection,
            jitter=model.prior.jitter,
        )

    @property
    def elbo_bound(self) -> ScalarFloat:
        r"""The collapsed evidence lower bound of Titsias (2009).

        Notation and derivation:

        Let :math:`Q = K_{xz}K_{zz}^{-1}K_{zx}`; we must compute the log
        normal pdf

        .. math::

            \log \mathcal{N}(y; m_x, \sigma^2 I + Q)
            = -\tfrac{n}{2}\log 2\pi - \tfrac{1}{2}\log|\sigma^2 I + Q|
            - \tfrac{1}{2}(y - m_x)^{\top}(\sigma^2 I + Q)^{-1}(y - m_x).

        The log determinant is computed via the matrix determinant lemma,

        .. math::

            \log|\sigma^2 I + Q| = n\log\sigma^2 + \log|B|,
            \qquad B = I + AA^{\top},\quad A = L_z^{-1}K_{zx}/\sigma,

        and the matrix inversion lemma gives

        .. math::

            (\sigma^2 I + Q)^{-1} = \big(I - A^{\top}B^{-1}A\big)/\sigma^2,

        so the quadratic term is
        :math:`[(y-m_x)^{\top}(y-m_x) - (y-m_x)^{\top}A^{\top}B^{-1}A(y-m_x)]
        / \sigma^2`. The bound subtracts the trace correction
        :math:`\operatorname{tr}(K_{xx} - Q) / (2\sigma^2)`, with
        :math:`\operatorname{tr}(K_{zz}^{-1}K_{zx}K_{xz}) = \sigma^2
        \operatorname{tr}(AA^{\top})`.

        Returns:
            ScalarFloat: The collapsed evidence lower bound on the training
                data this posterior was conditioned on.
        """
        noise_var = self.observation_variance
        num_data = self.residual.shape[0]

        # log|B| = 2 sum_i log L_ii
        log_det_b = 2.0 * jnp.sum(jnp.log(jnp.diagonal(self.cholesky_b)))

        # L^{-1} A (y - m(x))
        projected_residual = jsp.linalg.solve_triangular(
            self.cholesky_b,
            jnp.matmul(self.scaled_cross, self.residual),
            lower=True,
        )

        # (y - m(x))^T (o^2 I + Q)^{-1} (y - m(x))
        quadratic = (
            jnp.sum(self.residual**2) - jnp.sum(projected_residual**2)
        ) / noise_var

        # 2 log N(y; m(x), o^2 I + Q)
        two_log_prob = (
            -num_data * jnp.log(2.0 * jnp.pi * noise_var) - log_det_b - quadratic
        )

        # 1/o^2 tr(Kxx - Q), using tr(AA^T) = sum(A * A)
        two_trace = jnp.sum(self.kxx_diagonal) / noise_var - jnp.sum(
            self.scaled_cross**2
        )

        # log N(y; m(x), o^2 I + Q) - 1/(2 o^2) tr(Kxx - Q)
        return (two_log_prob - two_trace).squeeze() / 2.0

    @property
    def prior_kl(self) -> ScalarFloat:
        r"""KL divergence from the optimal collapsed :math:`q^{\star}(u)` to
        the prior :math:`p(u)`.

        Titsias' optimal variational distribution has moments
        :math:`S^{\star} = L_z B^{-1} L_z^{\top}` and centred mean
        :math:`\tilde m^{\star} = \sigma^{-1} L_z B^{-1} A (y - m(x))`, so

        .. math::

            \operatorname{KL} = \tfrac{1}{2}\big(
                \operatorname{tr}(B^{-1}) - M
                + \sigma^{-2}\lVert B^{-1}A(y - m(x))\rVert^2
                + \log|B|
            \big).

        Returns:
            ScalarFloat: The KL divergence of the optimal collapsed
                variational distribution from the prior.
        """
        num_inducing = self.cholesky_b.shape[0]

        # tr(B^{-1}) = ||L^{-1}||_F^2
        inverse_factor = jsp.linalg.solve_triangular(
            self.cholesky_b, jnp.eye(num_inducing), lower=True
        )
        trace = jnp.sum(inverse_factor**2)

        # B^{-1} A (y - m(x))
        weighted_residual = jsp.linalg.cho_solve(
            (self.cholesky_b, True),
            jnp.matmul(self.scaled_cross, self.residual),
        )
        mahalanobis = jnp.sum(weighted_residual**2) / self.observation_variance

        log_det_b = 2.0 * jnp.sum(jnp.log(jnp.diagonal(self.cholesky_b)))
        return 0.5 * (trace - num_inducing + mahalanobis + log_det_b)


def _build_fourier_features_fn(
    prior: tp.Any, num_features: int, key: KeyArray
) -> tp.Callable[[Float[Array, "N D"]], Float[Array, "N L"]]:
    r"""Return a function evaluating features sampled from the Fourier feature
    decomposition of the prior's kernel.

    Args:
        prior (Prior): The Prior distribution.
        num_features (int): The number of feature functions to be sampled.
        key (KeyArray): The random seed used.

    Returns:
        Callable: A callable function evaluating the sampled feature functions.
    """
    if (not isinstance(num_features, int)) or num_features <= 0:
        raise ValueError("num_features must be a positive integer")

    approximate_kernel = RFF(
        base_kernel=prior.kernel, num_basis_fns=num_features, key=key
    )

    def eval_fourier_features(test_inputs: Float[Array, "N D"]) -> Float[Array, "N L"]:
        feature_matrix = approximate_kernel.compute_features(x=test_inputs)
        feature_matrix *= jnp.sqrt(val(prior.kernel.variance) / num_features)
        return feature_matrix

    return eval_fourier_features


__all__ = [
    "CollapsedPosterior",
    "ExactPosterior",
    "LatentPosterior",
    "Posterior",
    "SparsePosterior",
]
