"""StateSpacePrior and StateSpaceConjugateModel classes.

See plans/2026-04-21-state-space-gps-design.md.
"""

from __future__ import annotations

from typing import Literal

import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import (
    Bool,
    Num,
)
import lineax as lx

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.gps import ConjugateModel, Prior
from gpjax.likelihoods import Gaussian, MultiOutputGaussian
from gpjax.linalg.utils import add_jitter
from gpjax.parameters import val
from gpjax.state_space.conditioning import StateSpacePosterior
from gpjax.typing import Array


class StateSpacePrior(Prior):
    """Prior for a state-space (Markovian) GP.

    Predictions match ``gpjax.gps.Prior`` for both covariance modes.
    ``covariance="diagonal"`` (the default) uses the SDE's stationary state
    covariance directly rather than routing through the kernel, since the
    prior is stationary in time; ``covariance="dense"`` returns the kernel's
    own dense gram over the test inputs, since the state-space SDE is an
    *exact* representation of the kernel with no training data to
    marginalise out. Either way, this predictive is Liskov-substitutable for
    the dense ``gpjax.gps.Prior`` predictive.

    Example:
        >>> import gpjax as gpx
        >>> from gpjax.state_space import StateSpacePrior
        >>> prior = StateSpacePrior(
        ...     mean_function=gpx.mean_functions.Zero(),
        ...     kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        ... )
        >>> isinstance(prior.kernel, gpx.kernels.Matern32)
        True
    """

    def __call__(self, test_inputs, *, covariance="diagonal"):
        return self.predict(test_inputs, covariance=covariance)

    def predict(self, test_inputs, *, covariance="diagonal"):
        mean_at_test = self.mean_function(test_inputs)
        loc = jnp.atleast_1d(mean_at_test.squeeze())

        if covariance == "dense":
            gram_dense = add_jitter(
                self.kernel.gram(test_inputs).as_matrix(), self.jitter
            )
            scale = lx.MatrixLinearOperator(gram_dense)
            return GaussianDistribution(loc=loc, scale=scale)

        from gpjax.state_space.kernels import to_sde

        sde = to_sde(self.kernel)
        H = sde.observation_matrix
        L_inf = sde.stationary_state_cov_sqrt
        P_inf = L_inf @ L_inf.T
        marginal_variance = (H @ P_inf @ H.T).squeeze() + self.jitter

        n_test = test_inputs.shape[0]
        scale = lx.DiagonalLinearOperator(jnp.full(n_test, marginal_variance))
        return GaussianDistribution(loc=loc, scale=scale)

    def __mul__(self, other):
        _require_scalar_gaussian_likelihood(other)
        return StateSpaceConjugateModel(prior=self, likelihood=other)


class StateSpaceConjugateModel(ConjugateModel):
    """Joint model for a state-space (Markovian) GP: conditioning is Kalman.

    Conditioning returns a
    :class:`~gpjax.state_space.conditioning.StateSpacePosterior`, whose queries
    run the square-root Kalman recursions in :math:`O(N d^3)` time. The
    inherited :class:`~gpjax.gps.ConjugateModel` conditioning is deliberately
    overridden: its dense :math:`O(N^3)` Cholesky is exactly the cost this
    model exists to avoid.

    v1 prediction surface, all of it sugar over ``condition``:
      - ``condition`` / ``__or__`` : the conditioned process
      - ``predict`` / ``__call__`` : ``condition(D)(t)``, the smoothed predictive
      - ``predict_filter``         : ``condition(D).filtered(t)``, the causal
        (filter-only) predictive

    **Predictive contract:** ``predict``/``__call__`` (the smoothed
    predictive) supports both ``covariance="diagonal"`` (marginal variances,
    the default) and ``covariance="dense"`` (the full joint covariance across
    test points, via the RTS smoother's cross-covariance recursion), so it is
    Liskov-substitutable for the dense ``gpjax.gps.ConjugateModel``
    predictive. ``predict_filter`` (the causal predictive) has no dense joint
    form: each test point conditions on a different information set, so
    ``covariance="dense"`` there still raises ``NotImplementedError`` — see
    :class:`~gpjax.state_space.conditioning.StateSpacePosterior`.

    Example:
        >>> import gpjax as gpx
        >>> from gpjax.state_space import StateSpacePrior
        >>> prior = StateSpacePrior(
        ...     mean_function=gpx.mean_functions.Zero(),
        ...     kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        ... )
        >>> likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
        >>> posterior = prior * likelihood
        >>> type(posterior).__name__
        'StateSpaceConjugateModel'
    """

    def condition(
        self,
        train_data: Dataset,
        *,
        observation_mask: tp.Optional[Bool[Array, " N"]] = None,
    ) -> StateSpacePosterior:
        r"""Condition on data through the Kalman recursions.

        Args:
            train_data: The observations to condition on.
            observation_mask: Optional boolean mask over the training points;
                ``False`` entries are not conditioned on. ``None`` conditions
                on every point.

        Returns:
            StateSpacePosterior: The conditioned process. Exposes the smoothed
                predictive (via ``__call__``), the causal predictive (via
                ``filtered``), and ``log_marginal_likelihood``.
        """
        return StateSpacePosterior(self, train_data, observation_mask=observation_mask)

    def __call__(
        self,
        test_inputs: Num[Array, "M 1"],
        train_data: Dataset,
        *,
        covariance: Literal["dense", "diagonal"] = "diagonal",
        observation_mask: tp.Optional[Bool[Array, " N"]] = None,
    ) -> GaussianDistribution:
        r"""Sugar: condition on ``train_data`` and query at ``test_inputs``."""
        return self.predict(
            test_inputs,
            train_data,
            covariance=covariance,
            observation_mask=observation_mask,
        )

    def predict(
        self,
        test_inputs: Num[Array, "M 1"],
        train_data: Dataset,
        *,
        covariance: Literal["dense", "diagonal"] = "diagonal",
        observation_mask: tp.Optional[Bool[Array, " N"]] = None,
    ) -> GaussianDistribution:
        r"""Sugar for the smoothed predictive: ``condition(D)(t)``.

        When making repeated predictions, condition once and reuse the
        returned posterior.

        Args:
            test_inputs: Test timestamps of shape ``(M, 1)``.
            train_data: The observations to condition on.
            covariance: ``"diagonal"`` for marginal variances or ``"dense"``
                for the full joint covariance.
            observation_mask: Optional boolean mask over the training points.

        Returns:
            GaussianDistribution: The smoothed predictive.
        """
        return self.condition(train_data, observation_mask=observation_mask)(
            test_inputs, covariance=covariance
        )

    def predict_filter(
        self,
        test_inputs: Num[Array, "M 1"],
        train_data: Dataset,
        *,
        covariance: Literal["dense", "diagonal"] = "diagonal",
        observation_mask: tp.Optional[Bool[Array, " N"]] = None,
    ) -> GaussianDistribution:
        r"""Sugar for the causal predictive: ``condition(D).filtered(t)``.

        Each test point conditions only on training observations at timestamps
        less than or equal to its own, rather than on the whole training set.

        Args:
            test_inputs: Test timestamps of shape ``(M, 1)``.
            train_data: The observations to condition on.
            covariance: Must be ``"diagonal"``; the v1 state-space predictive
                has no dense joint form.
            observation_mask: Optional boolean mask over the training points.

        Returns:
            GaussianDistribution: The filtered predictive.
        """
        return self.condition(train_data, observation_mask=observation_mask).filtered(
            test_inputs, covariance=covariance
        )

    def sample_approx(self, num_samples, train_data, key, num_features=100):
        r"""Not available for state-space models.

        The inherited pathwise sampler is built on the dense conditioned
        process, which state-space models never form. Raising is deliberate:
        silently falling back would reintroduce the :math:`O(N^3)` cost this
        model exists to avoid.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "sample_approx is not implemented for state-space models; the "
            "pathwise sampler requires the dense conditioned process that the "
            "state-space path deliberately never forms. Use predict or "
            "predict_filter for marginals."
        )


def _require_scalar_gaussian_likelihood(likelihood) -> None:
    """Raise if ``likelihood`` is not a single-output, scalar-stddev Gaussian.

    State-space v1 supports only ``gpjax.likelihoods.Gaussian`` with a scalar
    ``obs_stddev`` and ``num_outputs == 1``.
    """
    if isinstance(likelihood, MultiOutputGaussian):
        raise TypeError(
            "State-space inference requires a single-output Gaussian likelihood; "
            "MultiOutputGaussian is not supported in v1."
        )
    if not isinstance(likelihood, Gaussian):
        raise TypeError(
            f"State-space inference requires a Gaussian (conjugate) likelihood; "
            f"got {type(likelihood).__name__}."
        )
    obs_stddev_value = val(likelihood.obs_stddev)
    if jnp.asarray(obs_stddev_value).ndim != 0:
        raise ValueError(
            f"State-space Gaussian likelihood requires a scalar obs_stddev; "
            f"got shape {jnp.asarray(obs_stddev_value).shape}."
        )
