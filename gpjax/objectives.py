from typing import TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
import typing_extensions as tpe

from gpjax.dataset import Dataset
from gpjax.gps import (
    ConjugateModel,
    NonConjugateModel,
)
from gpjax.likelihoods import (
    AbstractHeteroscedasticLikelihood,
)
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)
from gpjax.variational_families import (
    AbstractVariationalFamily,
    DualVariationalGaussian,
    HeteroscedasticVariationalFamily,
)

VF = TypeVar("VF", bound=AbstractVariationalFamily)
HVF = TypeVar("HVF", bound=HeteroscedasticVariationalFamily)
DVF = TypeVar("DVF", bound=DualVariationalGaussian)


Objective = tpe.Callable[[eqx.Module, Dataset], ScalarFloat]
LogPriorFn = tpe.Callable[[eqx.Module], ScalarFloat]


def conjugate_mll(model: ConjugateModel, data: Dataset) -> ScalarFloat:
    r"""Evaluate the marginal log-likelihood of the Gaussian process.

    Compute the marginal log-likelihood function of the Gaussian process.
    The returned function can then be used for gradient based optimisation
    of the model's parameters or for model comparison. The implementation
    given here enables exact estimation of the Gaussian process' latent
    function values.

    For a training dataset $\{x_n, y_n\}_{n=1}^N$, set of test inputs
    $\mathbf{x}^{\star}$ the corresponding latent function evaluations are given
    by $\mathbf{f}=f(\mathbf{x})$ and $\mathbf{f}^{\star}f(\mathbf{x}^{\star})$,
    the marginal log-likelihood is given by:

    .. math::

        \begin{aligned}
            \log p(\mathbf{y}) & = \int p(\mathbf{y}\mid\mathbf{f})
            p(\mathbf{f}, \mathbf{f}^{\star})\mathrm{d}\mathbf{f}^{\star}\\
            & = 0.5\left(-\mathbf{y}^{\top}\left(k(\mathbf{x}, \mathbf{x}')
            + \sigma^2\mathbf{I}_N\right)^{-1}\mathbf{y} \right.\\
            & \quad\left. -\log\lvert k(\mathbf{x}, \mathbf{x}')
            + \sigma^2\mathbf{I}_N\rvert - n\log 2\pi \right).
        \end{aligned}

    Example:
        >>> import gpjax as gpx

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian()
        >>> prior = gpx.gps.Prior(mean_function = meanf, kernel=kernel)
        >>> model = prior * likelihood

        >>> gpx.objectives.conjugate_mll(model, D)

        Our goal is to maximise the marginal log-likelihood. Therefore, when optimising
        the model's parameters with respect to the parameters, we use the negative
        marginal log-likelihood. This can be realised through

        >>> nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)

    Args:
        model (ConjugateModel): The joint model for which we want to compute
            the marginal log-likelihood.
        data: The training dataset used to compute the
            marginal log-likelihood.

    Returns:
        ScalarFloat: The marginal log-likelihood of the Gaussian process.
    """
    return model.condition(data).log_marginal_likelihood


def conjugate_loocv(model: ConjugateModel, data: Dataset) -> ScalarFloat:
    r"""Evaluate the leave-one-out log predictive probability of the Gaussian process following
    section 5.4.2 of Rasmussen et al. 2006 - Gaussian Processes for Machine Learning. This metric
    calculates the average performance of all models that can be obtained by training on all but one
    data point, and then predicting the left out data point.

    For multi-output likelihoods this performs **leave-one-scalar-out** on the
    flattened NP system (per-element predictive), the natural generalisation of
    the scalar R&W LOOCV to multiple outputs.  Per-datapoint LOOCV has no
    closed form in the multi-output case.

    The returned metric can then be used for gradient based optimisation
    of the model's parameters or for model comparison. The implementation
    given here enables exact estimation of the Gaussian process' latent
    function values.

    For a given :class:`~gpjax.gps.ConjugateModel`, the following code snippet
    shows how the leave-one-out log predictive probability can be evaluated.

    Example:
        >>> import gpjax as gpx
        ...
        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)
        ...
        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian()
        >>> prior = gpx.gps.Prior(mean_function = meanf, kernel=kernel)
        >>> model = prior * likelihood
        ...
        >>> gpx.objectives.conjugate_loocv(model, D)

        Our goal is to maximise the leave-one-out log predictive probability. Therefore, when
        optimising the model's parameters with respect to the parameters, we use the negative
        leave-one-out log predictive probability. This can be realised through

        >>> nloocv = lambda p, d: -gpx.objectives.conjugate_loocv(p, d)

    Args:
        model (ConjugateModel): The joint model for which we want to compute
            the leave-one-out predictive probability.
        data: The training dataset used to compute the
            leave-one-out predictive probability.

    Returns:
        ScalarFloat: The leave-one-out log predictive probability.
    """
    return jnp.sum(model.condition(data).loo())


def log_posterior_density(model: NonConjugateModel, data: Dataset) -> ScalarFloat:
    r"""The log-posterior density of a non-conjugate Gaussian process. This is
    sometimes referred to as the marginal log-likelihood.

    Evaluate the log-posterior density of a Gaussian process.

    Compute the marginal log-likelihood, or log-posterior density of the Gaussian
    process. The returned function can then be used for gradient based optimisation
    of the model's parameters or for model comparison. The implementation given
    here is general and will work for any likelihood support by GPJax.

    Conditioning a :class:`~gpjax.gps.ConjugateModel` yields an
    :class:`~gpjax.conditioning.ExactPosterior`, whose
    :attr:`~gpjax.conditioning.ExactPosterior.log_marginal_likelihood` is available
    in closed form. Conditioning a :class:`~gpjax.gps.NonConjugateModel` instead
    yields a :class:`~gpjax.conditioning.LatentPosterior`, which has no exact
    marginal log-likelihood: it represents the posterior as a function of the
    model's hyperparameters and the latent function, and exposes the unnormalised
    :attr:`~gpjax.conditioning.LatentPosterior.log_posterior_density` in its place.
    Markov chain Monte Carlo, variational inference, or Laplace approximations can
    then be used to sample from, or optimise an approximation to, the posterior
    distribution.

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Bernoulli()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> model = (prior * likelihood).init_latent(D.n)

        >>> gpx.objectives.log_posterior_density(model, D)

    Args:
        model (NonConjugateModel): The joint model for which we want to
            compute the log-posterior density.
        data: The training dataset used to compute the
            log-posterior density.

    Returns:
        ScalarFloat: The log-posterior density of the Gaussian process.
    """
    if model.latent is None:
        raise ValueError(
            "NonConjugateModel.latent is uninitialised: fit the model or call "
            "model.init_latent(data.n) first."
        )
    return model.condition(data).log_posterior_density


non_conjugate_mll = log_posterior_density


def elbo(
    variational_family: VF,
    data: Dataset,
    *,
    key: KeyArray | None = None,
) -> ScalarFloat:
    r"""Compute the evidence lower bound of a variational approximation.

    Compute the evidence lower bound under this model. In short, this requires
    evaluating the expectation of the model's log-likelihood under the variational
    approximation. To this, we sum the KL divergence from the variational posterior
    to the prior. When batching occurs, the result is scaled by the batch size
    relative to the full dataset size.

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Bernoulli()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> z = jnp.linspace(0, 1, 10).reshape(-1, 1)
        >>> q = gpx.variational_families.VariationalGaussian(
        ...     model=posterior, inducing_inputs=z
        ... )

        >>> gpx.objectives.elbo(q, D)

    A likelihood whose integrator draws samples, such as ``Categorical`` with
    its default ``MonteCarloIntegrator``, needs a random key. Split it from
    every other key in the run and close over it, so that the samples are the
    same at each optimisation step:

    .. code-block:: python

        key, fit_key, elbo_key = jr.split(key, 3)
        objective = lambda q, d: -gpx.objectives.elbo(q, d, key=elbo_key)

    Args:
        variational_family: The variational
            approximation for whose parameters we should maximise the ELBO with
            respect to.
        data: The training data for which we should maximise the
            ELBO with respect to.
        key: The random key for a likelihood whose integrator draws samples.
            Deterministic integrators ignore it. See ADR-0002.

    Returns:
        ScalarFloat: The evidence lower bound of the variational approximation.
    """
    # KL[q(f(.)) || p(f(.))]
    kl = variational_family.prior_kl()

    # int[log(p(y|f(.))) q(f(.))] df(.)
    var_exp = variational_expectation(variational_family, data, key=key)

    # For batch size b, we compute  n/b * sum_i[ int log(p(y|f(xi))) q(f(xi)) df(xi)] - KL[q(f(.)) || p(f(.))]
    return jnp.sum(var_exp) * data.full_size / data.n - kl


def variational_expectation(
    variational_family: VF,
    data: Dataset,
    *,
    key: KeyArray | None = None,
) -> Float[Array, " N"]:
    r"""Compute the variational expectation.

    Compute the expectation of our model's log-likelihood under our variational
    distribution. Batching can be done here to speed up computation.

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Bernoulli()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> z = jnp.linspace(0, 1, 10).reshape(-1, 1)
        >>> q = gpx.variational_families.VariationalGaussian(
        ...     model=posterior, inducing_inputs=z
        ... )

        >>> gpx.objectives.variational_expectation(q, D)

    Args:
        variational_family: The variational family that we
            are using to approximate the posterior.
        data: The batch for which the expectation should be computed for.
        key: The random key for a likelihood whose integrator draws samples.
            The key is used once, by the integrator.

    Returns:
        Array: The expectation of the model's log-likelihood under our
        variational distribution.
    """
    # Unpack training batch
    x, y = data.X, data.y

    # Variational distribution q(f(.)) = N(f(.); mu(.), Sigma(., .))
    q = variational_family

    # Marginal moments mu(x) and diag(Sigma(x, x)) at the training inputs,
    # through the conditioned posterior's diagonal path. `train_data` is passed
    # for interface uniformity; the sparse families carry q(u) internally and
    # ignore it.
    qx = q.condition(data)(x, covariance="diagonal")

    # The moments are flattened in blocked order (index l N + n); one column
    # for each latent function of the likelihood.
    likelihood = q.model.likelihood
    num_latent = likelihood.num_latent_functions
    mean = qx.mean.reshape(num_latent, -1).T
    variance = qx.variance.reshape(num_latent, -1).T

    # approx int[log(p(y|f(x))) q(f(x))] df(x)
    expectation = likelihood.expected_log_likelihood(y, mean, variance, key=key)
    return expectation


def dual_elbo(variational_family: DVF, data: Dataset) -> ScalarFloat:
    r"""Compute the evidence lower bound of a dual (t-SVGP) approximation.

    The *same functional* as :func:`elbo`, but evaluated as a function of the stored
    dual sites and the kernel hyperparameters, never of $(m, S)$:

    .. math::

        \mathcal{L}_{\text{dual}}(\lambda_1, \Lambda_2;\theta)
            = \frac{N}{B}\sum_{i\in\mathcal{B}}
              \mathbb{E}_{q(f_i)}\left[\log p(y_i\mid f_i)\right]
              - \operatorname{KL}\left[q(u)\mid\mid p_{\theta}(u)\right],
        \qquad
        S = \left(\mathbf{K}_{zz}(\theta)^{-1} + \Lambda_2\right)^{-1}.

    Following Adam, Chang, Khan and Solin (2021),
    `arXiv:2111.03412 <https://arxiv.org/abs/2111.03412>`_.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> import gpjax as gpx

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Bernoulli()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> z = jnp.linspace(0, 1, 10).reshape(-1, 1)
        >>> q = gpx.variational_families.DualVariationalGaussian(
        ...     model=posterior, inducing_inputs=z
        ... )

        >>> gpx.objectives.dual_elbo(q, D).shape
        ()

    Args:
        variational_family: The dual variational approximation whose sites and
            hyperparameters the bound is evaluated at.
        data: The training data, or a mini-batch of it.

    Returns:
        ScalarFloat: The evidence lower bound of the dual variational approximation.

    Notes:
        Its **value** equals :func:`elbo` at the implied moments for any sites and
        any $\theta$; its **hyperparameter gradient** differs, because $q$ moves
        with $\theta$ through $\mathbf{K}_{zz}$ while the sites stay frozen. The
        extra term,
        $\langle\nabla_{\eta}\mathcal{L},\ \partial\eta_0(\theta)/\partial\theta\rangle$,
        vanishes at a converged E-step and is the source of the tighter M-step
        behaviour Adam et al. report. Do **not** wrap $\mathbf{K}_{zz}$ in
        ``lax.stop_gradient``, and do not cache the implied moments on the family:
        that implicit dependence is the entire point, and removing it is a silent
        bug -- identical values, wrong gradients.

        The marginals are computed in one batched
        :meth:`~gpjax.variational_families.DualVariationalGaussian.marginals` call
        directly from the working matrix $\mathbf{R}$, rather than through the
        moment conversion that :func:`elbo` performs when it conditions the
        family. Both are $\mathcal{O}(M^3 + NM^2)$, but this path skips forming
        and factorising $\mathbf{S}$. :func:`elbo` called directly on a
        ``DualVariationalGaussian`` is still correct and returns the same value and
        the same gradients; ``dual_elbo`` is the fast path, not a different bound.

        Train a ``DualVariationalGaussian`` with :func:`~gpjax.fit.fit_natgrads`.
        Plain :func:`~gpjax.fit.fit` computes the right gradients, but it takes
        unconstrained steps on ``dual_matrix``, and nothing then keeps
        $\Lambda_2$ positive semi-definite. A step that leaves the PSD cone makes
        the Cholesky factorisation of $\mathbf{R}$ fail and the objective return
        NaN (`#751 <https://github.com/QuantClimate/GPJax/issues/751>`_). Only
        the natural-gradient update keeps the dual parameters valid.
    """
    # KL[q(u) || p(u)], evaluated through R = Kzz + Kzz Lambda_2 Kzz.
    kl = variational_family.prior_kl()

    # Batched marginals of q(f(x)); O(M^3 + N M^2).
    mean, variance = variational_family.marginals(data.X)

    likelihood = variational_family.model.likelihood
    expectation = likelihood.expected_log_likelihood(
        data.y, mean[:, None], variance[:, None]
    )

    # For batch size b, n/b * sum_i E_q[log p(y_i | f(x_i))] - KL[q(u) || p(u)].
    return jnp.sum(expectation) * data.full_size / data.n - kl


def collapsed_elbo(variational_family: VF, data: Dataset) -> ScalarFloat:
    r"""Compute a single step of the collapsed evidence lower bound.

    Compute the evidence lower bound under this model. In short, this requires
    evaluating the expectation of the model's log-likelihood under the variational
    approximation. To this, we sum the KL divergence from the variational posterior
    to the prior. This collapsed bound is evaluated on the full dataset supplied in
    ``data`` and does not apply minibatch scaling.

    The bound is the ``elbo_bound`` view of the conditioned
    :class:`~gpjax.conditioning.CollapsedPosterior` — this objective is not a
    second derivation.

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> z = jnp.linspace(0, 1, 10).reshape(-1, 1)
        >>> q = gpx.variational_families.CollapsedVariationalGaussian(
        ...     model=posterior, inducing_inputs=z
        ... )

        >>> gpx.objectives.collapsed_elbo(q, D)

    Args:
        variational_family: The variational
            approximation for whose parameters we should maximise the ELBO with
            respect to.
        data: The training data for which we should maximise the
            ELBO with respect to.

    Returns:
        ScalarFloat: The evidence lower bound of the variational approximation.
    """
    return variational_family.condition(data).elbo_bound


def heteroscedastic_elbo_conjugate(
    variational_family: HVF, data: Dataset
) -> ScalarFloat:
    r"""Tight bound from Lazaro-Gredilla & Titsias (2011) for heteroscedastic Gaussian likelihoods."""
    likelihood = variational_family.model.likelihood
    mean_f, var_f, mean_g, var_g = variational_family.predict(data.X)

    expected_ll, _ = likelihood.expected_log_likelihood(
        data.y,
        mean_f,
        var_f,
        mean_g=mean_g,
        variance_g=var_g,
        return_parts=True,
    )

    scale = data.full_size / data.n
    return scale * jnp.sum(expected_ll) - variational_family.prior_kl()


def heteroscedastic_elbo_chained(variational_family: HVF, data: Dataset) -> ScalarFloat:
    r"""Generic chained bound for heteroscedastic likelihoods."""
    likelihood: AbstractHeteroscedasticLikelihood = variational_family.model.likelihood
    mean_f, var_f, mean_g, var_g = variational_family.predict(data.X)
    noise_stats = likelihood.noise_statistics(mean_g, var_g)

    expected_ll = likelihood.expected_log_likelihood(
        data.y,
        mean_f,
        var_f,
        mean_g=mean_g,
        variance_g=var_g,
        noise_stats=noise_stats,
    )

    scale = data.full_size / data.n
    return scale * jnp.sum(expected_ll) - variational_family.prior_kl()


def heteroscedastic_elbo(variational_family: HVF, data: Dataset) -> ScalarFloat:
    r"""Compute the evidence lower bound of a heteroscedastic approximation.

    Dispatches on the likelihood: those that admit the tight Lazaro-Gredilla &
    Titsias (2011) bound use :func:`heteroscedastic_elbo_conjugate`, and every
    other heteroscedastic likelihood uses the generic chained bound,
    :func:`heteroscedastic_elbo_chained`.

    Args:
        variational_family: The heteroscedastic variational approximation whose
            parameters the bound is evaluated at.
        data: The training data, or a mini-batch of it.

    Returns:
        ScalarFloat: The evidence lower bound of the variational approximation.
    """
    likelihood = variational_family.model.likelihood
    if likelihood.supports_tight_bound():
        return heteroscedastic_elbo_conjugate(variational_family, data)
    return heteroscedastic_elbo_chained(variational_family, data)


__all__ = [
    "LogPriorFn",
    "Objective",
    "collapsed_elbo",
    "conjugate_loocv",
    "conjugate_mll",
    "elbo",
    "heteroscedastic_elbo",
    "heteroscedastic_elbo_chained",
    "heteroscedastic_elbo_conjugate",
    "log_posterior_density",
    "non_conjugate_mll",
    "variational_expectation",
]
