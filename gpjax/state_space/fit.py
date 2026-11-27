"""Fit wrappers (fit_scipy, fit_lbfgs, fit) for state-space GPs.

These thin wrappers validate inputs, optionally sort time-series data, and
delegate to the existing ``gpjax.fit.fit_*`` optimisers with a
``state_space_mll``-derived negative-log-likelihood objective.
"""

from __future__ import annotations

import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.state_space._validation import (
    sort_state_space_data,
    validate_state_space_data,
)
from gpjax.state_space.objectives import state_space_mll


def _prepare_data(train_data: Dataset, observation_mask=None):
    """Validate and sort training data; return sorted (Dataset, mask)."""
    validate_state_space_data(train_data.X, train_data.y, observation_mask)
    X_sorted, y_sorted, mask_sorted = sort_state_space_data(
        train_data.X,
        train_data.y,
        observation_mask,
    )
    return Dataset(X=X_sorted, y=y_sorted), mask_sorted


def _make_objective(observation_mask):
    """Closure over the (possibly None) observation mask returning a negative-MLL objective."""

    def objective(model, train_data):
        return -state_space_mll(model, train_data, observation_mask=observation_mask)

    return objective


def fit_scipy(
    *,
    model,
    train_data: Dataset,
    observation_mask=None,
    max_iters: int = 500,
    verbose: bool = True,
    safe: bool = True,
):
    """Fit a state-space posterior with SciPy's L-BFGS-B.

    Thin wrapper around ``gpx.fit_scipy``. Validates data, sorts if necessary,
    and uses ``state_space_mll`` as the objective.

    Warns:
        OptimizeWarning: If SciPy reports that L-BFGS-B did not converge, for
            example because it reached ``max_iters``.

    Example:
        >>> import jax.numpy as jnp
        >>> import gpjax as gpx
        >>> from gpjax.state_space import StateSpacePrior, fit_scipy
        >>> X = jnp.linspace(0.0, 5.0, 20).reshape(-1, 1)
        >>> y = jnp.sin(X)
        >>> prior = StateSpacePrior(
        ...     mean_function=gpx.mean_functions.Zero(),
        ...     kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        ... )
        >>> likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
        >>> posterior = prior * likelihood
        >>> fitted, history = fit_scipy(
        ...     model=posterior,
        ...     train_data=gpx.Dataset(X=X, y=y),
        ...     max_iters=2,
        ...     verbose=False,
        ... )
        >>> bool(jnp.all(jnp.isfinite(history)))
        True
    """
    sorted_data, sorted_mask = _prepare_data(train_data, observation_mask)
    objective = _make_objective(sorted_mask)
    return gpx.fit_scipy(
        model=model,
        objective=objective,
        train_data=sorted_data,
        max_iters=max_iters,
        verbose=verbose,
        safe=safe,
    )


def fit_lbfgs(
    *,
    model,
    train_data: Dataset,
    observation_mask=None,
    max_iters: int = 500,
    safe: bool = True,
):
    """Fit a state-space posterior with Optax's L-BFGS (``while_loop`` driver).

    Thin wrapper around ``gpx.fit_lbfgs``.

    Example:
        >>> import jax.numpy as jnp
        >>> import gpjax as gpx
        >>> from gpjax.state_space import StateSpacePrior, fit_lbfgs
        >>> X = jnp.linspace(0.0, 5.0, 20).reshape(-1, 1)
        >>> y = jnp.sin(X)
        >>> prior = StateSpacePrior(
        ...     mean_function=gpx.mean_functions.Zero(),
        ...     kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        ... )
        >>> likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
        >>> posterior = prior * likelihood
        >>> fitted, history = fit_lbfgs(
        ...     model=posterior,
        ...     train_data=gpx.Dataset(X=X, y=y),
        ...     max_iters=2,
        ... )
        >>> fitted is not None
        True
    """
    sorted_data, sorted_mask = _prepare_data(train_data, observation_mask)
    objective = _make_objective(sorted_mask)
    return gpx.fit_lbfgs(
        model=model,
        objective=objective,
        train_data=sorted_data,
        max_iters=max_iters,
        safe=safe,
    )


def fit(
    *,
    model,
    train_data: Dataset,
    optim,
    observation_mask=None,
    key=None,
    num_iters: int = 100,
    batch_size: int = -1,
    log_rate: int = 10,
    verbose: bool = True,
    unroll: int = 1,
    safe: bool = True,
):
    """Fit a state-space posterior with Optax (gradient-descent style).

    Thin wrapper around ``gpx.fit``. Rejects ``batch_size != -1`` because
    state-space MLL is intrinsically full-batch (the temporal scan cannot be
    minibatched without breaking the Markov chain).

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import optax as ox
        >>> import gpjax as gpx
        >>> from gpjax.state_space import StateSpacePrior, fit
        >>> X = jnp.linspace(0.0, 5.0, 20).reshape(-1, 1)
        >>> y = jnp.sin(X)
        >>> prior = StateSpacePrior(
        ...     mean_function=gpx.mean_functions.Zero(),
        ...     kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        ... )
        >>> likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
        >>> posterior = prior * likelihood
        >>> fitted, history = fit(
        ...     model=posterior,
        ...     train_data=gpx.Dataset(X=X, y=y),
        ...     optim=ox.adam(1e-2),
        ...     num_iters=2,
        ...     key=jr.key(0),
        ...     verbose=False,
        ... )
        >>> bool(jnp.all(jnp.isfinite(history)))
        True
    """
    if batch_size != -1:
        raise ValueError(
            "state_space.fit requires full-batch optimisation (batch_size=-1); "
            f"got batch_size={batch_size}. State-space MLL is computed via a "
            "temporal scan that cannot be minibatched."
        )
    sorted_data, sorted_mask = _prepare_data(train_data, observation_mask)
    objective = _make_objective(sorted_mask)
    fit_kwargs = dict(
        model=model,
        objective=objective,
        train_data=sorted_data,
        optim=optim,
        num_iters=num_iters,
        batch_size=batch_size,
        log_rate=log_rate,
        verbose=verbose,
        unroll=unroll,
        safe=safe,
    )
    if key is not None:
        fit_kwargs["key"] = key
    return gpx.fit(**fit_kwargs)
