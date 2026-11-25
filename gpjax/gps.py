# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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
r"""Gaussian process priors and joint models.

The API mirrors the mathematics:

.. code-block:: python

    prior = gpx.Prior(mean_function=meanf, kernel=kernel)   # p(f)
    likelihood = gpx.likelihoods.Gaussian()                 # p(y | f)
    model = prior * likelihood                              # p(f, y)
    posterior = model.condition(train_data)                 # p(f | D)
    predictive = posterior(test_inputs)

A :class:`JointModel` is the trainable object — ``gpx.fit`` optimises its
hyperparameters. Conditioning it on data returns an immutable
:class:`gpjax.conditioning.Posterior` that caches the factorisation.
"""

import typing as tp
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Float,
    Num,
)
import lineax as lx
from paramax import AbstractUnwrappable

from gpjax.conditioning import (
    ExactPosterior,
    LatentPosterior,
    Posterior,
    _build_fourier_features_fn,
)
from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import (
    AbstractHeteroscedasticLikelihood,
    AbstractLikelihood,
    Gaussian,
    NonGaussian,
)
from gpjax.linalg.utils import add_jitter
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.parameters import Real
from gpjax.summary import _SummaryMixin
from gpjax.typing import (
    Array,
    FunctionalSample,
    KeyArray,
)

K = tp.TypeVar("K", bound=AbstractKernel)
M = tp.TypeVar("M", bound=AbstractMeanFunction)
L = tp.TypeVar("L", bound=AbstractLikelihood)
NGL = tp.TypeVar("NGL", bound=NonGaussian)
GL = tp.TypeVar("GL", bound=Gaussian)
HL = tp.TypeVar("HL", bound=AbstractHeteroscedasticLikelihood)


#######################
# GP Priors
#######################
class Prior(_SummaryMixin, eqx.Module, tp.Generic[M, K]):
    r"""A Gaussian process prior object.

    A Gaussian process prior parameterised by a mean function $m(\cdot)$ and a kernel
    function $k(\cdot, \cdot)$ is given by
    $p(f(\cdot)) = \mathcal{GP}(m(\cdot), k(\cdot, \cdot))$.

    To invoke a `Prior` distribution, a kernel and mean function must be specified.

    Example:
        >>> import gpjax as gpx
        >>> kernel = gpx.kernels.RBF()
        >>> meanf = gpx.mean_functions.Zero()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel = kernel)

    .. seealso::

        :doc:`/examples/intro_to_gps` derives the prior from first principles, and
        :doc:`/examples/regression` puts one to work end to end.
    """

    kernel: K
    mean_function: M
    jitter: float = eqx.field(static=True, default=1e-6)

    def __init__(
        self,
        kernel: K,
        mean_function: M,
        jitter: float = 1e-6,
    ):
        r"""Construct a Gaussian process prior.

        Args:
            kernel: kernel object inheriting from AbstractKernel.
            mean_function: mean function object inheriting from AbstractMeanFunction.
            jitter: the model's single numerical-stabilisation knob. Applied
                exactly once, inside conditioning.
        """
        self.kernel = kernel
        self.mean_function = mean_function
        self.jitter = jitter

    if tp.TYPE_CHECKING:

        @tp.overload
        def __mul__(self, other: GL) -> "ConjugateModel[M, K, GL]": ...

        @tp.overload
        def __mul__(self, other: NGL) -> "NonConjugateModel[M, K, NGL]": ...

        @tp.overload
        def __mul__(self, other: L) -> "JointModel[M, K, L]": ...

    def __mul__(self, other):
        r"""Combine the prior with a likelihood to form a joint model.

        The product of a prior and likelihood is the joint distribution over
        latent function and observations,

        .. math::

            p(f(\cdot), y) = p(y \mid f(\cdot))\,p(f(\cdot)),

        where $p(y | f(\cdot))$ is the likelihood and $p(f(\cdot))$ is the
        prior. Conditioning the returned model on data yields the posterior.

        Example:
            >>> import gpjax as gpx
            >>> meanf = gpx.mean_functions.Zero()
            >>> kernel = gpx.kernels.RBF()
            >>> prior = gpx.gps.Prior(mean_function=meanf, kernel = kernel)
            >>> likelihood = gpx.likelihoods.Gaussian()
            >>> model = prior * likelihood

        Args:
            other (AbstractLikelihood): The likelihood of the observations.

        Returns:
            JointModel: The joint model for the given prior and likelihood.
                The concrete type reflects conjugacy.
        """
        return construct_model(prior=self, likelihood=other)

    if tp.TYPE_CHECKING:

        @tp.overload
        def __rmul__(self, other: GL) -> "ConjugateModel[M, K, GL]": ...

        @tp.overload
        def __rmul__(self, other: NGL) -> "NonConjugateModel[M, K, NGL]": ...

        @tp.overload
        def __rmul__(self, other: L) -> "JointModel[M, K, L]": ...

    def __rmul__(self, other):
        r"""Order-invariant product: ``likelihood * prior``."""
        return self.__mul__(other)

    def __call__(
        self,
        test_inputs: Num[Array, "N D"],
        *,
        covariance: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Evaluate the prior process at the given points.

        Args:
            test_inputs: Input locations where the GP should be evaluated.
            covariance: Whether to return the dense joint covariance at the
                test inputs or only the marginal (diagonal) variances.

        Returns:
            GaussianDistribution: A multivariate normal random variable
                representation of the Gaussian process.
        """
        return self.predict(test_inputs, covariance=covariance)

    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        *,
        covariance: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Compute the prior predictive distribution at the test inputs.

        Example:
            >>> import gpjax as gpx
            >>> import jax.numpy as jnp
            >>> kernel = gpx.kernels.RBF()
            >>> mean_function = gpx.mean_functions.Zero()
            >>> prior = gpx.gps.Prior(mean_function=mean_function, kernel=kernel)
            >>> prior.predict(jnp.linspace(0, 1, 100)[:, None])

        Args:
            test_inputs (Float[Array, "N D"]): The inputs at which to evaluate
                the prior distribution.
            covariance: Whether to return the dense joint covariance at the
                test inputs or only the marginal (diagonal) variances.

        Returns:
            GaussianDistribution: A multivariate normal random variable
                representation of the Gaussian process.
        """
        mean_at_test = self.mean_function(test_inputs)
        if covariance == "dense":
            gram_dense = add_jitter(
                self.kernel.gram(test_inputs).as_matrix(), self.jitter
            )
            cov = lx.MatrixLinearOperator(gram_dense)
        else:
            gram_diag = lx.diagonal(self.kernel.diagonal(test_inputs))
            var = gram_diag + self.jitter
            cov = lx.DiagonalLinearOperator(jnp.atleast_1d(var.squeeze()))

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean_at_test.squeeze()), scale=cov
        )

    def sample_approx(
        self,
        num_samples: int,
        key: KeyArray,
        num_features: int | None = 100,
    ) -> FunctionalSample:
        r"""Approximate samples from the Gaussian process prior.

        Build an approximate sample from the Gaussian process prior via the
        finite feature approximation
        $\hat{f}(x) = \sum_{i=1}^m\phi_i(x)\theta_i$ where $\phi_i$ are $m$
        features sampled from the Fourier feature decomposition of the model's
        kernel and $\theta_i$ are samples from a unit Gaussian.

        The same sample draw is evaluated for all queries, at constant cost
        per query.

        Example:
            >>> import gpjax as gpx
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> key = jr.key(123)
            >>>
            >>> meanf = gpx.mean_functions.Zero()
            >>> kernel = gpx.kernels.RBF(n_dims=1)
            >>> prior = gpx.gps.Prior(mean_function=meanf, kernel = kernel)
            >>>
            >>> sample_fn = prior.sample_approx(10, key)
            >>> sample_fn(jnp.linspace(0, 1, 100).reshape(-1, 1))

        Args:
            num_samples (int): The desired number of samples.
            key (KeyArray): The random seed used for the sample(s).
            num_features (int): The number of features used when approximating
                the kernel.

        Returns:
            FunctionalSample: A function representing an approximate sample
                from the Gaussian process prior.
        """
        if (not isinstance(num_samples, int)) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        freq_key, weight_key = jr.split(key)
        fourier_feature_fn = _build_fourier_features_fn(self, num_features, freq_key)

        feature_weights = jr.normal(weight_key, [num_samples, 2 * num_features])

        def sample_fn(test_inputs: Float[Array, "N D"]) -> Float[Array, "N B"]:
            feature_evals = fourier_feature_fn(test_inputs)
            evaluated_sample = jnp.inner(feature_evals, feature_weights)
            return self.mean_function(test_inputs) + evaluated_sample

        return sample_fn


#######################
# Joint models
#######################
class JointModel(_SummaryMixin, eqx.Module, tp.Generic[M, K, L]):
    r"""The joint distribution $p(f, y) = p(y \mid f)\,p(f)$.

    Pairs a :class:`Prior` with a likelihood. This is the *trainable* object:
    ``gpx.fit`` optimises its hyperparameters. Conditioning it on data —
    ``model.condition(D)`` or ``model | D`` — produces the posterior process.

    The base class carries no inference of its own; concrete subclasses
    (:class:`ConjugateModel`, :class:`NonConjugateModel`,
    :class:`HeteroscedasticModel`) define what conditioning means for their
    likelihood. A bare ``JointModel`` is a lightweight pairing used where
    inference is delegated elsewhere (e.g. variational families over a
    latent noise process).
    """

    prior: Prior
    likelihood: tp.Any

    def __init__(self, prior: Prior[M, K], likelihood: L):
        r"""Construct a joint model.

        Args:
            prior (Prior): The prior process.
            likelihood (AbstractLikelihood): The observation likelihood.
        """
        self.prior = prior
        self.likelihood = likelihood

    def condition(self, train_data: Dataset) -> Posterior:
        r"""Condition the joint model on data, returning the posterior process.

        Args:
            train_data: The observations to condition on.

        Returns:
            Posterior: The conditioned process $p(f \mid \mathcal{D})$.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define direct conditioning; "
            "use a variational family for inference."
        )

    def __or__(self, train_data: Dataset) -> Posterior:
        r"""Sugar for conditioning: ``model | D`` reads as $p(f \mid \mathcal{D})$."""
        return self.condition(train_data)

    def _prepare(self, train_data: Dataset) -> "JointModel":
        r"""Hook for data-dependent initialisation; returns a ready-to-fit model."""
        del train_data
        return self

    def __call__(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: Dataset,
        *,
        covariance: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Sugar: condition on ``train_data`` and query at ``test_inputs``.

        Equivalent to ``self.condition(train_data)(test_inputs)``.
        """
        return self.predict(test_inputs, train_data, covariance=covariance)

    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: Dataset,
        *,
        covariance: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Sugar: condition on ``train_data`` and query at ``test_inputs``.

        Defined as exactly ``self.condition(train_data)(test_inputs)``. When
        making repeated predictions, condition once and reuse the returned
        posterior — the factorisation is cached there.

        Args:
            test_inputs: A Jax array of test inputs.
            train_data: A `gpx.Dataset` to condition on.
            covariance: Whether to return the dense joint covariance at the
                test inputs or only the marginal (diagonal) variances.

        Returns:
            GaussianDistribution: The predictive distribution.
        """
        return self.condition(train_data)(test_inputs, covariance=covariance)


class ConjugateModel(JointModel[M, K, GL]):
    r"""A joint model with Gaussian likelihood: conditioning is exact.

    For a Gaussian process prior $p(\mathbf{f})$ and a Gaussian likelihood
    $p(y | \mathbf{f}) = \mathcal{N}(y\mid \mathbf{f}, \sigma^2))$, the latent
    function can be analytically integrated out. Conditioning returns the
    closed-form posterior

    .. math::

        \begin{aligned}
        p(\mathbf{f}^{\star}\mid \mathbf{y}) & =\mathcal{N}(\mathbf{f}^{\star};
        \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}),\\
        \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y},  \\
        \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
        \end{aligned}

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp
        >>>
        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> D = gpx.Dataset(X=xtrain, y=jnp.sin(xtrain))
        >>>
        >>> prior = gpx.gps.Prior(
        ...     mean_function = gpx.mean_functions.Zero(),
        ...     kernel = gpx.kernels.RBF()
        ... )
        >>> model = prior * gpx.likelihoods.Gaussian()
        >>> posterior = model.condition(D)
        >>> predictive = posterior(xtrain)
        >>> evidence = posterior.log_marginal_likelihood
    """

    def condition(self, train_data: Dataset) -> ExactPosterior:
        r"""Condition on data exactly.

        Returns:
            ExactPosterior: The closed-form posterior process, with the
                training-covariance factorisation cached. Exposes the
                predictive (via ``__call__``), ``log_marginal_likelihood``,
                ``loo`` and ``sample_approx``.
        """
        return ExactPosterior(self.prior, self.likelihood, train_data)

    def sample_approx(
        self,
        num_samples: int,
        train_data: Dataset,
        key: KeyArray,
        num_features: int | None = 100,
    ) -> FunctionalSample:
        r"""Sugar: ``self.condition(train_data).sample_approx(...)``.

        Draw approximate posterior samples via pathwise conditioning
        (Wilson et al., 2020).
        """
        return self.condition(train_data).sample_approx(num_samples, key, num_features)


class NonConjugateModel(JointModel[M, K, NGL]):
    r"""A joint model with non-Gaussian likelihood.

    Exact conditioning is intractable; the model instead carries a whitened
    latent vector $w_x$ as a trainable parameter, and conditioning produces
    the approximate posterior implied by its current value. Markov chain Monte
    Carlo, variational inference, or MAP optimisation (via
    ``gpx.objectives.log_posterior_density``) refine it.

    The latent is sized by the training data, so it is initialised lazily on
    first contact with data — ``gpx.fit`` does this automatically, or call
    :meth:`init_latent` explicitly.
    """

    latent: tp.Any

    def __init__(
        self,
        prior: Prior[M, K],
        likelihood: NGL,
        latent: Float[Array, "N L"] | AbstractUnwrappable | None = None,
    ):
        r"""Construct a non-conjugate joint model.

        Args:
            prior (Prior): The prior process.
            likelihood (AbstractLikelihood): The observation likelihood.
            latent: Whitened latent function values at the training inputs,
                of shape ``(N, L)`` with one column for each latent function
                of the likelihood (``likelihood.num_latent_functions``).
                ``None`` (the default) defers initialisation to first data
                contact.
        """
        super().__init__(prior=prior, likelihood=likelihood)
        if latent is None or isinstance(latent, AbstractUnwrappable):
            self.latent = latent
        else:
            self.latent = Real(latent)

    def init_latent(
        self, num_datapoints: int, key: KeyArray = jr.key(42)
    ) -> "NonConjugateModel[M, K, NGL]":
        r"""Return a copy of this model with the latent vector initialised.

        Args:
            num_datapoints: The number of training observations the latent
                must cover.
            key: The random seed for the initial values.
        """
        num_latent_functions = self.likelihood.num_latent_functions
        latent = jr.normal(key, shape=(num_datapoints, num_latent_functions))
        return NonConjugateModel(
            prior=self.prior, likelihood=self.likelihood, latent=latent
        )

    def _prepare(self, train_data: Dataset) -> "NonConjugateModel[M, K, NGL]":
        if self.latent is not None:
            return self
        return self.init_latent(train_data.n)

    def condition(self, train_data: Dataset) -> LatentPosterior:
        r"""Return the approximate posterior implied by the current latent.

        A ``None`` latent conditions at the prior mean (zeros in whitened
        space).

        Returns:
            LatentPosterior: The conditioned process. Exposes the predictive
                (via ``__call__``) and ``log_posterior_density``.
        """
        latent = self.latent
        if latent is None:
            latent = jnp.zeros((train_data.n, self.likelihood.num_latent_functions))
        return LatentPosterior(self.prior, self.likelihood, latent, train_data)


class HeteroscedasticModel(JointModel[M, K, HL]):
    r"""A joint model with input-dependent (heteroscedastic) noise.

    The joint holds *two* priors — one over the signal process and one over
    the latent noise process — which is why it is constructed directly rather
    than via ``prior * likelihood``:

    .. code-block:: python

        model = gpx.gps.HeteroscedasticModel(
            prior=signal_prior,
            likelihood=gpx.likelihoods.HeteroscedasticGaussian(),
            noise_prior=noise_prior,
        )

    Inference is delegated to
    :class:`gpjax.variational_families.HeteroscedasticVariationalFamily` and
    the ``heteroscedastic_elbo`` objective; the noise process is exposed as
    the nested joint model :attr:`noise_model`.

    The noise prior is stored *only* inside :attr:`noise_model`, and
    :attr:`noise_prior` reads through to it. Holding it in both places would
    duplicate its parameters across two pytree paths, which an optimiser step
    would then drive out of sync.
    """

    noise_model: tp.Any

    def __init__(
        self,
        prior: Prior[M, K],
        likelihood: HL,
        noise_prior: Prior,
    ):
        r"""Construct a heteroscedastic joint model.

        Args:
            prior (Prior): The prior over the signal process.
            likelihood (AbstractHeteroscedasticLikelihood): The observation
                likelihood.
            noise_prior (Prior): The prior over the latent noise process. It is
                stored as ``self.noise_model.prior``, not as a field of its own.

        Raises:
            ValueError: If ``noise_prior`` is ``None``.
        """
        if noise_prior is None:
            raise ValueError("Heteroscedastic models require a noise_prior.")
        super().__init__(prior=prior, likelihood=likelihood)
        self.noise_model = JointModel(prior=noise_prior, likelihood=likelihood)

    @property
    def noise_prior(self) -> Prior:
        r"""The prior over the latent noise process.

        Read-only view of ``self.noise_model.prior``. The noise prior is not a
        field of this model: a single owner keeps its parameters at one pytree
        path, so training cannot leave a second copy stale.
        """
        return self.noise_model.prior

    def condition(self, train_data: Dataset) -> Posterior:
        r"""Not available: heteroscedastic conditioning has no closed form.

        Args:
            train_data: Unused; present for interface uniformity.

        Raises:
            NotImplementedError: Always. Heteroscedastic models are the one
                documented exclusion from universal conditioning.
        """
        del train_data
        raise NotImplementedError(
            "HeteroscedasticModel has no closed-form conditioned process: the "
            "latent noise process must be inferred jointly with the signal. "
            "Run inference through "
            "gpjax.variational_families.HeteroscedasticVariationalFamily, "
            "fitting it with the gpjax.objectives.heteroscedastic_elbo "
            "objective, then predict with the fitted variational family:\n\n"
            "    from gpjax.objectives import heteroscedastic_elbo\n\n"
            "    q = gpx.variational_families.HeteroscedasticVariationalFamily(\n"
            "        model=model, inducing_inputs=Z\n"
            "    )\n"
            "    q, _ = gpx.fit(\n"
            "        model=q,\n"
            "        objective=lambda q, d: -heteroscedastic_elbo(q, d),\n"
            "        train_data=D,\n"
            "        optim=optax.adam(1e-2),\n"
            "        key=key,\n"
            "    )\n"
            "    predictive = q(xtest)"
        )


#######################
# Utils
#######################


@tp.overload
def construct_model(prior: Prior, likelihood: GL) -> ConjugateModel: ...


@tp.overload
def construct_model(prior: Prior, likelihood: NGL) -> NonConjugateModel: ...


def construct_model(prior: Prior, likelihood: AbstractLikelihood) -> "JointModel":
    r"""Construct the joint model for a prior/likelihood pair.

    Selects the concrete :class:`JointModel` subclass from the likelihood's
    conjugacy. This is what ``prior * likelihood`` calls.

    Args:
        prior (Prior): The prior process.
        likelihood (AbstractLikelihood): The observation likelihood.

    Returns:
        JointModel: A ``ConjugateModel`` for Gaussian likelihoods, a
            ``NonConjugateModel`` otherwise.

    Raises:
        ValueError: For heteroscedastic likelihoods, which carry a second
            prior and must be constructed directly via
            ``HeteroscedasticModel(prior, likelihood, noise_prior=...)``.
    """
    # Multi-output validation
    from gpjax.kernels.multioutput.base import MultiOutputKernel
    from gpjax.likelihoods import MultiOutputGaussian

    is_mo_kernel = isinstance(prior.kernel, MultiOutputKernel)
    is_mo_likelihood = isinstance(likelihood, MultiOutputGaussian)

    if is_mo_likelihood and not is_mo_kernel:
        raise ValueError(
            "MultiOutputGaussian likelihood requires a multi-output kernel "
            "(e.g., ICMKernel)."
        )
    if is_mo_kernel and not is_mo_likelihood:
        raise ValueError(
            "Multi-output kernels require a MultiOutputGaussian likelihood."
        )

    if isinstance(likelihood, AbstractHeteroscedasticLikelihood):
        raise ValueError(
            "Heteroscedastic likelihoods carry a second (noise) prior, which "
            "the two-operand product cannot express. Construct the model "
            "directly: HeteroscedasticModel(prior, likelihood, "
            "noise_prior=...)."
        )

    if isinstance(likelihood, Gaussian):
        return ConjugateModel(prior=prior, likelihood=likelihood)

    return NonConjugateModel(prior=prior, likelihood=likelihood)


__all__ = [
    "ConjugateModel",
    "HeteroscedasticModel",
    "JointModel",
    "NonConjugateModel",
    "Prior",
    "construct_model",
]
