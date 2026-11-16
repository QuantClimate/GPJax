r"""The conditioned process for state-space (Markovian) Gaussian processes.

:class:`~gpjax.conditioning.Posterior` is the one home of "a GP conditioned on
data" in GPJax, and this module supplies the state-space member of that family.
Where :class:`~gpjax.conditioning.ExactPosterior` factorises a dense
:math:`N \times N` covariance, a state-space model never forms one: every
conditioned quantity — the smoothed predictive, the filtered predictive, and
the evidence — is read off a square-root Kalman pass that costs
:math:`O(N d^3)` in time and :math:`O(N d^2)` in memory.

Because the Kalman recursions run on a *merged* train-plus-test grid, there is
no factorisation shared between queries to cache; conditioning is therefore a
cheap, eager bundling of the model with its training set, and each query runs
its own linear-time pass. The value of the object is interface uniformity, not
amortisation: ``model | D`` and ``model.condition(D)`` now reach the Kalman
path rather than silently falling back on the dense algebra the state-space
model exists to avoid.

See plans/2026-04-21-state-space-gps-design.md for the full design.
"""

from __future__ import annotations

from typing import Literal

import beartype.typing as tp
from jaxtyping import (
    Bool,
    Num,
)

from gpjax.conditioning import Posterior
from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.typing import (
    Array,
    ScalarFloat,
)


def _dense_not_implemented(query: str) -> NotImplementedError:
    """Build the shared ``covariance="dense"`` rejection for a named query.

    Args:
        query: The name of the predictive query being rejected, used to open
            the message (for example ``"prediction"`` or ``"predict_filter"``).

    Returns:
        NotImplementedError: The error to raise.
    """
    return NotImplementedError(
        f"State-space posterior {query} returns diagonal (marginal) covariance "
        "only; a dense joint predictive is not implemented in v1 and is tracked "
        "as a follow-up. The marginal variances returned are exact, so for "
        "diagonal-only use pass covariance='diagonal'."
    )


class StateSpacePosterior(Posterior):
    r"""A state-space GP conditioned on data, :math:`p(f \mid \mathcal{D})`.

    Produced by :meth:`gpjax.state_space.gps.StateSpaceConjugateModel.condition`
    (equivalently ``model | train_data``). Calling it runs the square-root
    Kalman filter and RTS smoother over the merged train-plus-test time grid,
    so the predictive is the *smoothed* latent posterior. :meth:`filtered`
    gives the causal (filter-only) alternative, and
    :attr:`log_marginal_likelihood` gives the evidence from the same filter.

    Predictive contract: :meth:`__call__` (the smoothed predictive) supports
    both ``covariance="diagonal"`` (marginal variances, the default) and
    ``covariance="dense"`` (the full joint covariance across test points, via
    the RTS smoother's cross-covariance recursion — see
    :func:`gpjax.state_space.prediction._dense_smoothed_test_covariance`), so
    it is Liskov-substitutable for a dense
    :class:`gpjax.conditioning.ExactPosterior` predictive. :meth:`filtered`
    (the causal predictive) has no dense joint form in v1 — each test point
    conditions on a *different* information set (data up to its own
    timestamp), so a "joint" filtered covariance is not the same kind of
    object as the dense conjugate predictive it would otherwise be compared
    against — and ``covariance="dense"`` there still raises
    :class:`NotImplementedError`.

    Time ordering: the predictive queries merge and sort the train and test
    grids internally, so they are order-insensitive.
    :attr:`log_marginal_likelihood` is not — it inherits the time-sorted-input
    assumption of :func:`gpjax.state_space.objectives.state_space_mll`. Fitting
    through the ``gpjax.state_space.fit*`` wrappers sorts the data for you.

    Attributes:
        model: The :class:`StateSpaceConjugateModel` that was conditioned.
        train_data: The observations conditioned on.
        observation_mask: Optional boolean mask over the training points;
            ``False`` entries are not conditioned on. ``None`` conditions on
            every point.

    Example:
        >>> import jax.numpy as jnp
        >>> import gpjax as gpx
        >>> from gpjax.state_space import StateSpacePrior
        >>> X = jnp.linspace(0.0, 5.0, 20).reshape(-1, 1)
        >>> D = gpx.Dataset(X=X, y=jnp.sin(X))
        >>> model = StateSpacePrior(
        ...     mean_function=gpx.mean_functions.Zero(),
        ...     kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        ... ) * gpx.likelihoods.Gaussian(obs_stddev=0.1)
        >>> posterior = model | D
        >>> type(posterior).__name__
        'StateSpacePosterior'
        >>> predictive = posterior(jnp.array([[1.0], [2.0]]))
        >>> predictive.mean.shape
        (2,)
    """

    model: tp.Any
    train_data: Dataset
    observation_mask: tp.Optional[Bool[Array, " N"]]

    def __init__(
        self,
        model: tp.Any,
        train_data: Dataset,
        *,
        observation_mask: tp.Optional[Bool[Array, " N"]] = None,
    ):
        r"""Bundle a state-space model with the data it is conditioned on.

        Args:
            model: The :class:`StateSpaceConjugateModel` being conditioned.
            train_data: The observations to condition on.
            observation_mask: Optional boolean mask over the training points.
        """
        self.model = model
        self.train_data = train_data
        self.observation_mask = observation_mask

    @property
    def prior(self) -> tp.Any:
        """The conditioned model's prior process."""
        return self.model.prior

    @property
    def likelihood(self) -> tp.Any:
        """The conditioned model's observation likelihood."""
        return self.model.likelihood

    def __call__(
        self,
        test_inputs: Num[Array, "M 1"],
        *,
        covariance: Literal["dense", "diagonal"] = "diagonal",
    ) -> GaussianDistribution:
        r"""Evaluate the smoothed conditioned process at the test inputs.

        Runs the square-root Kalman filter followed by the RTS smoother on the
        merged train-plus-test grid, so each test point conditions on the whole
        training set. Test inputs are returned in caller order regardless of
        their time ordering.

        Args:
            test_inputs: Test timestamps of shape ``(M, 1)``.
            covariance: ``"diagonal"`` returns the marginal variances only;
                ``"dense"`` returns the full ``M x M`` joint covariance, built
                from the RTS smoother's cross-covariance recursion (Särkkä &
                Solin 2019 §12.2) rather than a dense ``N x N`` gram over the
                training set.

        Returns:
            GaussianDistribution: The smoothed predictive, carrying an
                ``lx.DiagonalLinearOperator`` scale when ``covariance`` is
                ``"diagonal"`` or an ``lx.MatrixLinearOperator`` when it is
                ``"dense"``.
        """
        from gpjax.state_space.prediction import predict_smoothed

        return predict_smoothed(
            self.model,
            self.train_data,
            test_inputs,
            observation_mask=self.observation_mask,
            covariance=covariance,
        )

    def predict(
        self,
        test_inputs: Num[Array, "M 1"],
        train_data: tp.Optional[Dataset] = None,
        *,
        covariance: Literal["dense", "diagonal"] = "diagonal",
    ) -> GaussianDistribution:
        r"""Sugar for calling the posterior: ``predict(t) == self(t)``.

        Retained for signature compatibility with the pre-v1.0 API;
        ``train_data`` is accepted and ignored — this process is already
        conditioned on its training set.

        Args:
            test_inputs: Test timestamps of shape ``(M, 1)``.
            train_data: Ignored.
            covariance: ``"diagonal"`` for marginal variances or ``"dense"``
                for the full joint covariance.

        Returns:
            GaussianDistribution: The smoothed predictive.
        """
        del train_data
        return self(test_inputs, covariance=covariance)

    def filtered(
        self,
        test_inputs: Num[Array, "M 1"],
        *,
        covariance: Literal["dense", "diagonal"] = "diagonal",
    ) -> GaussianDistribution:
        r"""Evaluate the causal (filtered) conditioned process.

        Each test point conditions only on training observations at timestamps
        less than or equal to its own, with train sorting before test on ties.
        Marginals come from the forward filter trajectory rather than the
        smoother, which makes this the right query for online or forecasting
        use where conditioning on the future would leak information.

        Args:
            test_inputs: Test timestamps of shape ``(M, 1)``.
            covariance: Must be ``"diagonal"``; the v1 state-space predictive
                has no dense joint form.

        Returns:
            GaussianDistribution: The filtered predictive, carrying an
                ``lx.DiagonalLinearOperator`` scale.

        Raises:
            NotImplementedError: If ``covariance="dense"``.
        """
        if covariance != "diagonal":
            raise _dense_not_implemented("predict_filter")
        from gpjax.state_space.prediction import predict_filtered

        return predict_filtered(
            self.model,
            self.train_data,
            test_inputs,
            observation_mask=self.observation_mask,
        )

    @property
    def log_marginal_likelihood(self) -> ScalarFloat:
        r"""The evidence :math:`\log p(y)`, from the square-root Kalman filter.

        Accumulated as :math:`\sum_i \log p(y_i \mid y_{<i})` over the forward
        pass, so it costs :math:`O(N d^3)` rather than the :math:`O(N^3)` of a
        dense Cholesky. Computed on demand rather than cached at ``condition``
        time: the filter runs on the training grid alone, so it shares no work
        with the predictive queries, which run on merged train-plus-test grids.

        Assumes the training data is sorted in time — see the class docstring.

        Returns:
            ScalarFloat: The marginal log-likelihood of the conditioned data.
        """
        from gpjax.state_space.objectives import state_space_mll

        return state_space_mll(
            self.model, self.train_data, observation_mask=self.observation_mask
        )


__all__ = ["StateSpacePosterior"]
