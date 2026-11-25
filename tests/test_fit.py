# Copyright 2023 The thomaspinder Contributors. All Rights Reserved.
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

import sys

import equinox as eqx
import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.fit import (
    _check_batch_size,
    _check_log_rate,
    _check_model,
    _check_natgrad_lr,
    _check_num_iters,
    _check_optim,
    _check_train_data,
    _check_verbose,
    fit,
    fit_lbfgs,
    fit_natgrads,
    fit_scipy,
    get_batch,
)
from gpjax.gps import (
    ConjugateModel,
    Prior,
)
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import (
    AbstractMeanFunction,
    Constant,
)
from gpjax.objectives import (
    conjugate_mll,
    dual_elbo,
    elbo,
)
from gpjax.parameters import (
    PositiveReal,
    val,
)
from gpjax.typing import Array
from gpjax.variational_families import (
    CollapsedVariationalGaussian,
    DualVariationalGaussian,
    VariationalGaussian,
)
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import (
    Float,
    Num,
)
import numpy as np
import optax as ox
import paramax
import pytest
import scipy

from tests._reference.conjugate_svgp import conjugate_optimum


class LinearModel(eqx.Module):
    weight: PositiveReal
    bias: float = eqx.field(static=True, default=1.0)

    def __init__(self, weight: float = 1.0, bias: float = 1.0):
        self.weight = PositiveReal(weight)
        self.bias = bias

    def __call__(self, x):
        return val(self.weight) * x + self.bias


# The mll and the elbo are both maximised, whereas fit functions minimise,
# so optimisation is always driven by the negated quantity.
def _negative_conjugate_mll(model, data):
    return -conjugate_mll(model, data)


def _negative_elbo(model, data):
    return -elbo(model, data)


def test_fit_simple() -> None:
    # Create dataset:
    X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    y = 2.0 * X + 1.0 + 10 * jr.normal(jr.key(0), X.shape).reshape(-1, 1)
    D = Dataset(X, y)

    model = LinearModel(weight=1.0, bias=1.0)

    # Define loss function:
    def mse(model, data):
        pred = model(data.X)
        return jnp.mean((pred - data.y) ** 2)

    # Train!
    trained_model, hist = fit(
        model=model,
        objective=mse,
        train_data=D,
        optim=ox.sgd(0.001),
        num_iters=100,
        key=jr.key(123),
    )

    # Ensure we return a history of the correct length
    assert len(hist) == 100

    # Ensure we return a model of the same class
    assert isinstance(trained_model, LinearModel)

    # Test reduction in loss:
    assert mse(trained_model, D) < mse(model, D)

    # Test stop_gradient on bias (static field, not trained):
    assert trained_model.bias == 1.0


def test_fit_scipy_simple():
    # Create dataset:
    X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    y = 2.0 * X + 1.0 + 10 * jr.normal(jr.key(0), X.shape).reshape(-1, 1)
    D = Dataset(X, y)

    model = LinearModel(weight=1.0, bias=1.0)

    # Define loss function:
    def mse(model, data):
        pred = model(data.X)
        return jnp.mean((pred - data.y) ** 2)

    # Train with bfgs!
    trained_model, hist = fit_scipy(
        model=model,
        objective=mse,
        train_data=D,
        max_iters=10,
    )

    # Ensure we return a history of the correct length
    assert len(hist) > 2

    # Ensure we return a model of the same class
    assert isinstance(trained_model, LinearModel)

    # Test reduction in loss:
    assert mse(trained_model, D) < mse(model, D)

    # Test stop_gradient on bias (static field, not trained):
    assert trained_model.bias == 1.0


def test_fit_lbfgs_simple():
    # Create dataset:
    X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    y = 2.0 * X + 1.0 + 10 * jr.normal(jr.key(0), X.shape).reshape(-1, 1)
    D = Dataset(X, y)

    model = LinearModel(weight=1.0, bias=1.0)

    # Define loss function:
    def mse(model, data):
        pred = model(data.X)
        return jnp.mean((pred - data.y) ** 2)

    # Train with bfgs!
    trained_model, _final_loss = fit_lbfgs(
        model=model,
        objective=mse,
        train_data=D,
        max_iters=10,
    )

    # Ensure we return a model of the same class
    assert isinstance(trained_model, LinearModel)

    # Test reduction in loss:
    assert mse(trained_model, D) < mse(model, D)

    # Test stop_gradient on bias (static field, not trained):
    assert trained_model.bias == 1.0


@pytest.mark.parametrize("n_data", [20])
@pytest.mark.parametrize("verbose", [True, False])
def test_fit_gp_regression(n_data: int, verbose: bool) -> None:
    # Create dataset:
    key = jr.key(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, 1)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define GP model:
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = Gaussian()
    posterior = prior * likelihood

    # Train!
    trained_model, history = fit(
        model=posterior,
        objective=_negative_conjugate_mll,
        train_data=D,
        optim=ox.adam(0.1),
        num_iters=15,
        verbose=verbose,
        key=jr.key(123),
    )

    # Ensure the trained model is a Gaussian process posterior
    assert isinstance(trained_model, ConjugateModel)

    # Ensure we return a history of the correct length
    assert len(history) == 15

    # Ensure we improve the marginal log-likelihood
    assert conjugate_mll(trained_model, D) > conjugate_mll(posterior, D)


@pytest.mark.parametrize("n_data", [20])
def test_fit_lbfgs_gp_regression(n_data: int) -> None:
    # Create dataset:
    key = jr.key(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, 1)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define GP model:
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = Gaussian()
    posterior = prior * likelihood

    # Train with BFGS!
    trained_model_bfgs, _final_loss = fit_lbfgs(
        model=posterior,
        objective=_negative_conjugate_mll,
        train_data=D,
        max_iters=40,
    )

    # Ensure the trained model is a Gaussian process posterior
    assert isinstance(trained_model_bfgs, ConjugateModel)

    # Ensure we improve the marginal log-likelihood
    assert conjugate_mll(trained_model_bfgs, D) > conjugate_mll(posterior, D)


@pytest.mark.parametrize(
    "run_fit",
    [
        lambda model, objective, data: fit(
            model=model,
            objective=objective,
            train_data=data,
            optim=ox.adam(0.1),
            num_iters=20,
            key=jr.key(123),
        )[0],
        lambda model, objective, data: fit_scipy(
            model=model, objective=objective, train_data=data, max_iters=20
        )[0],
        lambda model, objective, data: fit_lbfgs(
            model=model, objective=objective, train_data=data, max_iters=20
        )[0],
    ],
    ids=["fit", "fit_scipy", "fit_lbfgs"],
)
def test_fitters_hold_frozen_parameter_constant(run_fit) -> None:
    """Every fitter honours paramax.non_trainable.

    Each optimiser reaches the parameters differently -- an Optax update rule, an
    Optax while_loop, and a raveled vector handed to SciPy -- so the guarantee is
    worth asserting for each. Note that it is a property of the *optimiser* as
    much as of the wrapper: ox.adamw would still shrink the frozen parameter,
    because decoupled weight decay applies regardless of the gradient.
    """
    # Create dataset:
    key = jr.key(123)
    x = jnp.sort(jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(20, 1)), axis=0)
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define GP model, freezing the observation noise:
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = eqx.tree_at(
        lambda lik: lik.obs_stddev, Gaussian(), replace_fn=paramax.non_trainable
    )
    posterior = prior * likelihood

    # Train!
    trained_model = run_fit(posterior, _negative_conjugate_mll, D)

    # Ensure the frozen noise is held exactly
    assert val(trained_model.likelihood.obs_stddev) == val(
        posterior.likelihood.obs_stddev
    )

    # Ensure the remaining hyperparameters are still optimised
    assert not jnp.allclose(
        val(trained_model.prior.kernel.lengthscale),
        val(posterior.prior.kernel.lengthscale),
    )

    # Ensure we improve the marginal log-likelihood
    assert conjugate_mll(trained_model, D) > conjugate_mll(posterior, D)


def test_fit_scipy_error_raises() -> None:
    # Create dataset:
    D = Dataset(
        X=jnp.array([[0.0]], dtype=jnp.float64), y=jnp.array([[0.0]], dtype=jnp.float64)
    )

    # build crazy mean function so that opt fails
    class CrazyMean(AbstractMeanFunction):
        def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N O"]:
            return jnp.heaviside(x, 100.0)

    # Define GP model with crazy mean function:
    prior = Prior(kernel=RBF(), mean_function=CrazyMean())
    likelihood = Gaussian()
    posterior = prior * likelihood

    with pytest.raises(scipy.optimize.OptimizeWarning):
        fit_scipy(
            model=posterior,
            objective=conjugate_mll,
            train_data=D,
            max_iters=10,
        )

    # also check fails if no given enough steps
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = Gaussian()
    posterior = prior * likelihood

    with pytest.raises(scipy.optimize.OptimizeWarning):
        fit_scipy(
            model=posterior,
            objective=conjugate_mll,
            train_data=D,
            max_iters=1,
        )


@pytest.mark.parametrize("num_iters", [1, 5])
@pytest.mark.parametrize("batch_size", [1, 20, 50])
@pytest.mark.parametrize("n_data", [50])
@pytest.mark.parametrize("verbose", [True, False])
def test_fit_batch(num_iters: int, batch_size: int, n_data: int, verbose: bool) -> None:
    # Create dataset:
    key = jr.key(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, 1)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define GP model:
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = Gaussian()
    posterior = prior * likelihood

    # Define variational family:
    z = jnp.linspace(-2.0, 2.0, 10).reshape(-1, 1)
    q = VariationalGaussian(model=posterior, inducing_inputs=z)

    # Train!
    trained_model, history = fit(
        model=q,
        objective=_negative_elbo,
        train_data=D,
        optim=ox.adam(0.1),
        num_iters=num_iters,
        batch_size=batch_size,
        verbose=verbose,
        key=jr.key(123),
    )

    # Ensure the trained model is a Gaussian process posterior
    assert isinstance(trained_model, VariationalGaussian)

    # Ensure we return a history of the correct length
    assert len(history) == num_iters

    # Ensure we improve the elbo
    assert elbo(trained_model, D) > elbo(q, D)


def _svgp_setup(n_data: int, n_inducing: int = 5, jitter: float = 1e-8):
    """Build a conjugate SVGP and its training data for the fit_natgrads tests."""
    key = jr.key(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, 1)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    prior = Prior(kernel=RBF(), mean_function=Constant(), jitter=jitter)
    likelihood = Gaussian()
    posterior = prior * likelihood

    z = jnp.linspace(-2.0, 2.0, n_inducing).reshape(-1, 1)
    q = VariationalGaussian(model=posterior, inducing_inputs=z)
    return q, D


def _conjugate_optimal_q(q, data):
    r"""Closed-form optimal $(m^\star, S^\star)$ for an unwhitened conjugate SVGP.

    Delegates to the single shared transcription in ``tests/_reference`` so that this
    file and ``tests/test_natural_gradients.py`` cannot drift apart.
    """
    optimal_mean, optimal_covariance, _ = conjugate_optimum(q, data)
    return optimal_mean, optimal_covariance


def test_fit_natgrads_simple() -> None:
    q, D = _svgp_setup(n_data=20)

    trained_model, history = fit_natgrads(
        model=q,
        objective=_negative_elbo,
        train_data=D,
        optim=ox.adam(0.05),
        natgrad_lr=0.5,
        num_iters=20,
        verbose=False,
        key=jr.key(123),
    )

    assert isinstance(trained_model, VariationalGaussian)
    assert history.shape == (20,)
    assert history[-1] < history[0]


@pytest.mark.parametrize("n_data", [10, 20])
@pytest.mark.parametrize("verbose", [True, False])
def test_fit_natgrads_gp_regression(n_data: int, verbose: bool) -> None:
    q, D = _svgp_setup(n_data=n_data)

    initial_lengthscale = val(q.model.prior.kernel.lengthscale)
    initial_obs_stddev = val(q.model.likelihood.obs_stddev)

    trained_model, history = fit_natgrads(
        model=q,
        objective=_negative_elbo,
        train_data=D,
        optim=ox.adam(0.1),
        natgrad_lr=0.5,
        num_iters=15,
        verbose=verbose,
        key=jr.key(123),
    )

    assert isinstance(trained_model, VariationalGaussian)
    assert len(history) == 15
    assert bool(jnp.all(jnp.isfinite(history)))
    assert history[-1] < history[0]

    assert not jnp.allclose(
        val(trained_model.model.prior.kernel.lengthscale), initial_lengthscale
    )
    assert not jnp.allclose(
        val(trained_model.model.likelihood.obs_stddev), initial_obs_stddev
    )


@pytest.mark.parametrize("num_iters", [1, 5])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_data", [20])
@pytest.mark.parametrize("verbose", [True, False])
def test_fit_natgrads_batch(
    num_iters: int, batch_size: int, n_data: int, verbose: bool
) -> None:
    q, D = _svgp_setup(n_data=n_data)

    trained_model, history = fit_natgrads(
        model=q,
        objective=_negative_elbo,
        train_data=D,
        optim=ox.adam(0.1),
        natgrad_lr=0.1,
        num_iters=num_iters,
        batch_size=batch_size,
        verbose=verbose,
        key=jr.key(123),
    )

    assert isinstance(trained_model, VariationalGaussian)
    assert history.shape == (num_iters,)
    assert bool(jnp.all(jnp.isfinite(history)))
    assert bool(jnp.all(jnp.isfinite(val(trained_model.variational_mean))))
    assert bool(jnp.all(jnp.isfinite(val(trained_model.variational_root_covariance))))


def test_fit_natgrads_conjugate_single_step_is_exact() -> None:
    """One full-batch iteration at ``natgrad_lr=1`` lands on the exact optimum."""
    q, D = _svgp_setup(n_data=20)

    trained_model, _ = fit_natgrads(
        model=q,
        objective=_negative_elbo,
        train_data=D,
        optim=ox.sgd(0.0),
        natgrad_lr=1.0,
        num_iters=1,
        batch_size=-1,
        verbose=False,
    )

    trained_mean = val(trained_model.variational_mean)
    trained_root = val(trained_model.variational_root_covariance)
    optimal_mean, optimal_covariance = _conjugate_optimal_q(q, D)

    np.testing.assert_allclose(
        np.float64(trained_mean), np.float64(optimal_mean), atol=1e-10
    )
    np.testing.assert_allclose(
        np.float64(trained_root @ trained_root.T),
        np.float64(optimal_covariance),
        atol=1e-10,
    )


def test_fit_natgrads_history_matches_fit_convention() -> None:
    """``history[0]`` is the loss at the *initial* parameters, as in ``fit``."""
    q, D = _svgp_setup(n_data=20)

    _, history = fit_natgrads(
        model=q,
        objective=_negative_elbo,
        train_data=D,
        optim=ox.sgd(0.0),
        natgrad_lr=1e-12,
        num_iters=3,
        verbose=False,
    )

    np.testing.assert_allclose(
        np.float64(history[0]),
        np.float64(_negative_elbo(q, D)),
        rtol=1e-12,
    )


def test_fit_natgrads_accepts_optax_schedule() -> None:
    q, D = _svgp_setup(n_data=20)
    schedule = ox.exponential_decay(
        1e-4, transition_steps=5, decay_rate=10.0, end_value=1e-1
    )

    _, history = fit_natgrads(
        model=q,
        objective=_negative_elbo,
        train_data=D,
        optim=ox.adam(0.05),
        natgrad_lr=schedule,
        num_iters=10,
        verbose=False,
    )

    assert history.shape == (10,)
    assert bool(jnp.all(jnp.isfinite(history)))


def test_fit_natgrads_rejects_unsupported_family() -> None:
    q, D = _svgp_setup(n_data=20)
    collapsed = CollapsedVariationalGaussian(
        model=q.model, inducing_inputs=val(q.inducing_inputs)
    )

    with pytest.raises(NotImplementedError, match="CollapsedVariationalGaussian"):
        fit_natgrads(
            model=collapsed,
            objective=gpx.objectives.collapsed_elbo,
            train_data=D,
            optim=ox.adam(0.1),
            num_iters=2,
            verbose=False,
        )


def test_fit_natgrads_holds_frozen_hyperparameter_constant() -> None:
    q, D = _svgp_setup(n_data=20)
    frozen = eqx.tree_at(
        lambda tree: tree.model.likelihood.obs_stddev,
        q,
        replace_fn=paramax.non_trainable,
    )

    trained_model, history = fit_natgrads(
        model=frozen,
        objective=_negative_elbo,
        train_data=D,
        optim=ox.adam(0.1),
        natgrad_lr=0.5,
        num_iters=15,
        verbose=False,
        key=jr.key(123),
    )

    # Ensure the frozen noise is held exactly
    assert val(trained_model.model.likelihood.obs_stddev) == val(
        frozen.model.likelihood.obs_stddev
    )

    # Ensure the remaining hyperparameters are still optimised by the Optax half
    assert not jnp.allclose(
        val(trained_model.model.prior.kernel.lengthscale),
        val(frozen.model.prior.kernel.lengthscale),
    )

    # Ensure the variational coordinates are still optimised by the natgrad half
    assert not jnp.allclose(
        val(trained_model.variational_mean), val(frozen.variational_mean)
    )

    # Ensure we reduce the loss
    assert history[-1] < history[0]


def test_fit_natgrads_rejects_frozen_coordinates(monkeypatch) -> None:
    """A frozen coordinate is rejected by the validator, not from inside the scan.

    ``vscan`` is replaced by a sentinel: if the guard fired only from the traced step
    body, the sentinel would be raised first and a dangling progress bar left behind.
    """
    q, D = _svgp_setup(n_data=20)
    frozen = eqx.tree_at(
        lambda tree: tree.variational_mean,
        q,
        paramax.non_trainable(q.variational_mean),
    )

    def unreachable(*args, **kwargs):
        raise AssertionError("the scan was reached before the frozen-coordinate guard")

    monkeypatch.setattr(sys.modules["gpjax.fit"], "vscan", unreachable)

    with pytest.raises(ValueError, match="variational_mean"):
        fit_natgrads(
            model=frozen,
            objective=_negative_elbo,
            train_data=D,
            optim=ox.adam(0.1),
            num_iters=5,
            verbose=True,
        )


@pytest.mark.parametrize("natgrad_lr", [1, jnp.asarray(0.5)])
def test_fit_natgrads_accepts_non_float_step_sizes(natgrad_lr) -> None:
    """The entry point honours everything ``_check_natgrad_lr`` blesses.

    ``_check_natgrad_lr`` accepts an ``int`` and a 0-d array, so the beartype-checked
    signature must too; testing the validator alone would not catch a mismatch.
    """
    q, D = _svgp_setup(n_data=20)

    _, history = fit_natgrads(
        model=q,
        objective=_negative_elbo,
        train_data=D,
        optim=ox.sgd(0.0),
        natgrad_lr=natgrad_lr,
        num_iters=3,
        verbose=False,
    )

    assert history.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(history)))


def test_fit_natgrads_forwards_log_rate(capsys) -> None:
    """``log_rate`` reaches ``vscan`` rather than being silently ignored.

    Asserting on the *number* of tqdm postfix updates rather than on an exact cadence
    keeps the test independent of ``vscan``'s remainder handling; all that matters is
    that a smaller ``log_rate`` logs strictly more often.
    """
    q, D = _svgp_setup(n_data=20)

    def count_updates(log_rate: int) -> int:
        fit_natgrads(
            model=q,
            objective=_negative_elbo,
            train_data=D,
            optim=ox.adam(0.1),
            natgrad_lr=0.1,
            num_iters=30,
            log_rate=log_rate,
            verbose=True,
        )
        captured = capsys.readouterr()
        return (captured.err + captured.out).count("Value")

    assert count_updates(1) > count_updates(10)


@pytest.mark.filterwarnings("ignore:X is not of type float64")
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_fit_natgrads_preserves_float32() -> None:
    """A float32 model trains without a ``lax.scan`` carry-dtype mismatch."""
    q, D = _svgp_setup(n_data=20)
    cast = lambda leaf: jnp.asarray(leaf, dtype=jnp.float32)
    q = jtu.tree_map(cast, q)
    D = Dataset(X=cast(D.X), y=cast(D.y))

    trained_model, history = fit_natgrads(
        model=q,
        objective=_negative_elbo,
        train_data=D,
        optim=ox.adam(0.05),
        natgrad_lr=0.1,
        num_iters=5,
        verbose=False,
    )

    assert bool(jnp.all(jnp.isfinite(history)))
    assert all(leaf.dtype == jnp.float32 for leaf in jtu.tree_leaves(trained_model))


@pytest.mark.parametrize("n_data", [50])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 2, 50])
def test_get_batch(n_data: int, n_dim: int, batch_size: int):
    key = jr.key(123)

    # Create dataset:
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, n_dim)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Sample out a batch:
    B = get_batch(D, batch_size, key)

    # Check batch is correct size and shape dimensions:
    assert B.n == batch_size
    assert B.X.shape[1:] == x.shape[1:]
    assert B.y.shape[1:] == y.shape[1:]

    # Ensure no caching of batches:
    key, subkey = jr.split(key)
    New = get_batch(D, batch_size, subkey)
    assert New.n == batch_size
    assert New.X.shape[1:] == x.shape[1:]
    assert New.y.shape[1:] == y.shape[1:]
    assert jnp.sum(New.X == B.X) <= n_dim * batch_size / n_data
    assert jnp.sum(New.y == B.y) <= n_dim * batch_size / n_data


@pytest.fixture
def valid_model() -> eqx.Module:
    """Return a valid model for testing."""
    return LinearModel(weight=1.0, bias=1.0)


@pytest.fixture
def valid_dataset() -> Dataset:
    """Return a valid dataset for testing."""
    X = jnp.array([[1.0], [2.0], [3.0]])
    y = jnp.array([[1.0], [2.0], [3.0]])
    return Dataset(X=X, y=y)


def test_check_model_valid(valid_model: eqx.Module) -> None:
    """Test that a valid model passes validation."""
    _check_model(valid_model)


def test_check_model_invalid() -> None:
    """Test that an invalid model raises a TypeError."""
    model = "not a model"
    with pytest.raises(
        TypeError, match=r"Expected model to be a subclass of eqx\.Module"
    ):
        _check_model(model)


def test_check_train_data_valid(valid_dataset: Dataset) -> None:
    """Test that valid training data passes validation."""
    _check_train_data(valid_dataset)


def test_check_train_data_invalid() -> None:
    """Test that invalid training data raises a TypeError."""
    train_data = "not a dataset"
    with pytest.raises(
        TypeError, match=r"Expected train_data to be of type gpjax\.Dataset"
    ):
        _check_train_data(train_data)


def test_check_optim_valid() -> None:
    """Test that a valid optimiser passes validation."""
    optim = ox.sgd(0.1)
    _check_optim(optim)


def test_check_optim_invalid() -> None:
    """Test that an invalid optimiser raises a TypeError."""
    optim = "not an optimiser"
    with pytest.raises(
        TypeError, match=r"Expected optim to be of type optax\.GradientTransformation"
    ):
        _check_optim(optim)


@pytest.mark.parametrize("num_iters", [1, 10, 100])
def test_check_num_iters_valid(num_iters: int) -> None:
    """Test that valid number of iterations passes validation."""
    _check_num_iters(num_iters)


def test_check_num_iters_invalid_type() -> None:
    """Test that an invalid num_iters type raises a TypeError."""
    num_iters = "not an int"
    with pytest.raises(TypeError, match="Expected num_iters to be of type int"):
        _check_num_iters(num_iters)


@pytest.mark.parametrize("num_iters", [0, -5])
def test_check_num_iters_invalid_value(num_iters: int) -> None:
    """Test that an invalid num_iters value raises a ValueError."""
    with pytest.raises(ValueError, match="Expected num_iters to be positive"):
        _check_num_iters(num_iters)


@pytest.mark.parametrize("log_rate", [1, 10, 100])
def test_check_log_rate_valid(log_rate: int) -> None:
    """Test that a valid log rate passes validation."""
    _check_log_rate(log_rate)


def test_check_log_rate_invalid_type() -> None:
    """Test that an invalid log_rate type raises a TypeError."""
    log_rate = "not an int"
    with pytest.raises(TypeError, match="Expected log_rate to be of type int"):
        _check_log_rate(log_rate)


@pytest.mark.parametrize("log_rate", [0, -5])
def test_check_log_rate_invalid_value(log_rate: int) -> None:
    """Test that an invalid log_rate value raises a ValueError."""
    with pytest.raises(ValueError, match="Expected log_rate to be positive"):
        _check_log_rate(log_rate)


@pytest.mark.parametrize("verbose", [True, False])
def test_check_verbose_valid(verbose: bool) -> None:
    """Test that valid verbose values pass validation."""
    _check_verbose(verbose)


def test_check_verbose_invalid() -> None:
    """Test that an invalid verbose value raises a TypeError."""
    verbose = "not a bool"
    with pytest.raises(TypeError, match="Expected verbose to be of type bool"):
        _check_verbose(verbose)


@pytest.mark.parametrize("batch_size", [1, 10, 100, -1])
def test_check_batch_size_valid(batch_size: int) -> None:
    """Test that valid batch sizes pass validation."""
    _check_batch_size(batch_size)


def test_check_batch_size_invalid_type() -> None:
    """Test that an invalid batch_size type raises a TypeError."""
    batch_size = "not an int"
    with pytest.raises(TypeError, match="Expected batch_size to be of type int"):
        _check_batch_size(batch_size)


@pytest.mark.parametrize("batch_size", [0, -2, -5])
def test_check_batch_size_invalid_value(batch_size: int) -> None:
    """Test that invalid batch_size values raise a ValueError."""
    with pytest.raises(ValueError, match="Expected batch_size to be positive or -1"):
        _check_batch_size(batch_size)


@pytest.mark.parametrize("natgrad_lr", [0.1, 1.0, 1, jnp.asarray(0.5)])
def test_check_natgrad_lr_valid(natgrad_lr) -> None:
    """Test that valid natural-gradient step sizes pass validation.

    Everything blessed here must also satisfy ``fit_natgrads``' beartype-checked
    signature -- see ``test_fit_natgrads_accepts_non_float_step_sizes``.
    """
    _check_natgrad_lr(natgrad_lr)


def test_check_natgrad_lr_valid_schedule() -> None:
    """Test that an optax schedule passes validation."""
    _check_natgrad_lr(ox.exponential_decay(1e-3, transition_steps=5, decay_rate=2.0))


@pytest.mark.parametrize("natgrad_lr", ["0.1", True, False, jnp.ones(3)])
def test_check_natgrad_lr_invalid_type(natgrad_lr) -> None:
    """Test that an invalid natgrad_lr type raises a TypeError.

    ``bool`` is an ``int`` subclass, so it would otherwise slip through and be read
    silently as $\\gamma=1$; a non-scalar array would break the step's shape contract.
    """
    with pytest.raises(TypeError, match="Expected natgrad_lr to be of type float"):
        _check_natgrad_lr(natgrad_lr)


@pytest.mark.parametrize("natgrad_lr", [0.0, -0.1])
def test_check_natgrad_lr_invalid_value(natgrad_lr: float) -> None:
    """Test that non-positive natgrad_lr values raise a ValueError."""
    with pytest.raises(ValueError, match="Expected natgrad_lr to be positive"):
        _check_natgrad_lr(natgrad_lr)


def test_fit_freeze_kernel_variance() -> None:
    """Test that fit can freeze kernel variance parameter using paramax.non_trainable."""
    key = jr.key(42)
    X = jr.uniform(key, (20, 1), minval=-3.0, maxval=3.0)
    y = jnp.sin(X) + 0.1 * jr.normal(jr.key(43), (20, 1))
    D = Dataset(X, y)

    # Create GP with RBF kernel
    meanf = gpx.mean_functions.Zero()
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian()
    posterior = prior * likelihood

    # Record initial variance value
    initial_variance = posterior.prior.kernel.variance.unwrap()

    # Freeze variance using paramax.non_trainable + eqx.tree_at
    frozen_posterior = eqx.tree_at(
        lambda m: m.prior.kernel.variance,
        posterior,
        replace_fn=paramax.non_trainable,
    )

    trained_posterior, _ = fit(
        model=frozen_posterior,
        objective=_negative_conjugate_mll,
        train_data=D,
        optim=ox.sgd(0.01),
        num_iters=10,
        verbose=False,
    )

    # Assert variance has not changed
    assert jnp.allclose(val(trained_posterior.prior.kernel.variance), initial_variance)

    # Assert lengthscale has changed, and in the direction that improves the mll
    assert not jnp.allclose(val(trained_posterior.prior.kernel.lengthscale), 1.0)
    assert conjugate_mll(trained_posterior, D) > conjugate_mll(frozen_posterior, D)


def test_fit_zero_mean_function_is_frozen_by_default() -> None:
    """Zero needs no manual freezing: it must stay at zero through `fit`.

    Regression test for #330/#712.
    """
    key = jr.key(42)
    X = jr.uniform(key, (20, 1), minval=-3.0, maxval=3.0)
    y = jnp.full_like(X, 25.0)  # data with a large non-zero mean
    D = Dataset(X, y)

    posterior = (
        gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=gpx.kernels.RBF(lengthscale=1.0, variance=1.0),
        )
        * gpx.likelihoods.Gaussian()
    )

    trained_posterior, _ = fit(
        model=posterior,
        objective=_negative_conjugate_mll,
        train_data=D,
        optim=ox.adam(0.1),
        num_iters=50,
        verbose=False,
    )

    assert jnp.allclose(val(trained_posterior.prior.mean_function.constant), 0.0)


def test_fit_constant_mean_function_with_parameter() -> None:
    """Test that Constant mean function works with trainable Parameter."""
    key = jr.key(42)
    X = jr.uniform(key, (20, 1), minval=-3.0, maxval=3.0)
    y = 5.0 * jnp.ones_like(X) + 0.1 * jr.normal(jr.key(43), X.shape)  # Mean of 5.0
    D = Dataset(X, y)

    # Create GP with Constant mean function using Parameter
    from gpjax.parameters import Real

    meanf = gpx.mean_functions.Constant(constant=Real(1.0))  # Start with mean 1.0
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
    posterior = prior * likelihood

    # Record initial mean function constant
    initial_constant = meanf.constant.unwrap()

    # Train (should train the mean function Parameter)
    trained_posterior, _ = fit(
        model=posterior,
        objective=_negative_conjugate_mll,
        train_data=D,
        optim=ox.adam(0.01),
        num_iters=20,
        verbose=False,
    )

    # Assert mean function constant has changed (parameter is trainable)
    final_constant = trained_posterior.prior.mean_function.constant.unwrap()
    assert not jnp.allclose(final_constant, initial_constant)
    # Just verify the parameter changed (direction depends on optimization dynamics)
    assert jnp.isfinite(final_constant)  # Not NaN/Inf


def test_fit_constant_mean_function_frozen_with_non_trainable() -> None:
    """Test that Constant mean function raw value can be frozen using paramax.non_trainable.

    In the equinox backend, plain JAX arrays are trainable by default.
    To freeze the constant, use paramax.non_trainable via eqx.tree_at.
    """
    key = jr.key(42)
    X = jr.uniform(key, (20, 1), minval=-3.0, maxval=3.0)
    y = 5.0 * jnp.ones_like(X) + 0.1 * jr.normal(jr.key(43), X.shape)  # Mean of 5.0
    D = Dataset(X, y)

    # Create GP with Constant mean function using raw value
    meanf = gpx.mean_functions.Constant(constant=1.0)  # Fixed mean 1.0
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=0.1)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
    posterior = prior * likelihood

    # Record initial mean function constant
    initial_constant = meanf.constant

    # Freeze the constant using paramax.non_trainable
    frozen_posterior = eqx.tree_at(
        lambda m: m.prior.mean_function.constant,
        posterior,
        replace_fn=paramax.non_trainable,
    )

    # Train (constant should NOT change because it is frozen)
    trained_posterior, _ = fit(
        model=frozen_posterior,
        objective=_negative_conjugate_mll,
        train_data=D,
        optim=ox.sgd(0.1),
        num_iters=50,
        verbose=False,
    )

    # Assert mean function constant has NOT changed (frozen with non_trainable)
    assert jnp.allclose(
        val(trained_posterior.prior.mean_function.constant), initial_constant
    )


def test_fit_freeze_by_non_trainable() -> None:
    """Test freezing specific parameter types using paramax.non_trainable."""
    key = jr.key(42)
    X = jr.uniform(key, (20, 1), minval=-3.0, maxval=3.0)
    y = jnp.sin(X) + 0.1 * jr.normal(jr.key(43), (20, 1))
    D = Dataset(X, y)

    # Create GP with RBF kernel
    meanf = gpx.mean_functions.Zero()
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian()
    posterior = prior * likelihood

    # Record initial values
    initial_variance = posterior.prior.kernel.variance.unwrap()
    initial_lengthscale = posterior.prior.kernel.lengthscale.unwrap()
    initial_obs_stddev = posterior.likelihood.obs_stddev.unwrap()

    # Freeze variance and obs_stddev, train only lengthscale
    frozen_posterior = eqx.tree_at(
        lambda m: m.prior.kernel.variance,
        posterior,
        replace_fn=paramax.non_trainable,
    )
    frozen_posterior = eqx.tree_at(
        lambda m: m.likelihood.obs_stddev,
        frozen_posterior,
        replace_fn=paramax.non_trainable,
    )

    trained_posterior, _ = fit(
        model=frozen_posterior,
        objective=_negative_conjugate_mll,
        train_data=D,
        optim=ox.sgd(0.01),
        num_iters=10,
        verbose=False,
    )

    # Assert that frozen parameters have not changed
    assert jnp.allclose(val(trained_posterior.prior.kernel.variance), initial_variance)
    assert jnp.allclose(
        val(trained_posterior.likelihood.obs_stddev), initial_obs_stddev
    )

    # The trainable lengthscale must have moved so as to improve the mll
    assert conjugate_mll(trained_posterior, D) > conjugate_mll(frozen_posterior, D)

    # Assert lengthscale has changed
    assert not jnp.allclose(
        val(trained_posterior.prior.kernel.lengthscale), initial_lengthscale
    )


# ---------------------------------------------------------------------------
# The dual (t-SVGP) branch of fit_natgrads
# ---------------------------------------------------------------------------
def _dual_svgp_setup(n_data: int, n_inducing: int = 5, jitter: float = 1e-8):
    """The ``_svgp_setup`` model, re-expressed in the dual parameterisation."""
    q, D = _svgp_setup(n_data=n_data, n_inducing=n_inducing, jitter=jitter)
    dual = DualVariationalGaussian(
        model=q.model,
        inducing_inputs=val(q.inducing_inputs),
    )
    return dual, D


def _negative_dual_elbo(model, data):
    return -dual_elbo(model, data)


def test_fit_natgrads_dual_end_to_end() -> None:
    q, D = _dual_svgp_setup(n_data=20)

    trained_model, history = fit_natgrads(
        model=q,
        objective=_negative_dual_elbo,
        train_data=D,
        optim=ox.adam(0.05),
        natgrad_lr=0.5,
        num_iters=20,
        verbose=False,
        key=jr.key(123),
    )

    assert isinstance(trained_model, DualVariationalGaussian)
    assert history.shape == (20,)
    assert bool(jnp.all(jnp.isfinite(history)))
    assert history[-1] < history[0]


def test_fit_natgrads_dual_rejects_rate_above_one() -> None:
    """rho > 1 overshoots the site target and can break Lambda_2 >= 0.

    The Salimbeni families have no such restriction, so the same value must be
    accepted there -- otherwise the check is testing the wrong thing.
    """
    dual, D = _dual_svgp_setup(n_data=10)
    moment, _ = _svgp_setup(n_data=10)

    with pytest.raises(ValueError, match=r"natgrad_lr to lie in \(0, 1\]"):
        fit_natgrads(
            model=dual,
            objective=_negative_dual_elbo,
            train_data=D,
            optim=ox.adam(0.05),
            natgrad_lr=1.5,
            num_iters=1,
            verbose=False,
        )

    _check_natgrad_lr(1.5, moment)


def test_fit_natgrads_dual_rejects_schedule_above_one() -> None:
    """A schedule is fully determined up front, so it is checked up front.

    Before this guard an out-of-range schedule sailed past ``_check_natgrad_lr`` and
    the run returned an all-NaN history instead of raising -- the one failure mode the
    scalar check exists to prevent.
    """
    dual, D = _dual_svgp_setup(n_data=10)
    moment, _ = _svgp_setup(n_data=10)

    with pytest.raises(ValueError, match=r"natgrad_lr to lie in \(0, 1\]"):
        fit_natgrads(
            model=dual,
            objective=_negative_dual_elbo,
            train_data=D,
            optim=ox.adam(0.05),
            natgrad_lr=ox.constant_schedule(5.0),
            num_iters=3,
            verbose=False,
        )

    # A schedule that only exceeds 1 outside the horizon is fine, and the Salimbeni
    # families are unrestricted above either way.
    _check_natgrad_lr(ox.linear_schedule(0.5, 5.0, 100), dual, 3)
    _check_natgrad_lr(ox.constant_schedule(5.0), moment, 3)
    # Without ``num_iters`` a schedule stays unexamined, as documented.
    _check_natgrad_lr(ox.constant_schedule(5.0), dual)


def test_check_natgrad_schedule_rejects_non_positive_rates() -> None:
    """The lower bound binds for every family, not just the dual one.

    ``rho <= 0`` is not a small step: zero wastes the iteration and a negative rate
    extrapolates *away* from the target, which can leave the PSD cone in either
    parameterisation. The scalar path has always rejected it; before this guard a
    schedule that decayed through zero inside the horizon sailed past unexamined.
    """
    dual, _ = _dual_svgp_setup(n_data=10)
    moment, _ = _svgp_setup(n_data=10)

    # A ramp that hits exactly zero at the end of the horizon, and one that crosses.
    touches_zero = ox.linear_schedule(0.5, 0.0, 9)
    crosses_zero = ox.linear_schedule(0.5, -0.5, 5)

    for model in (dual, moment):
        for schedule in (touches_zero, crosses_zero):
            with pytest.raises(ValueError, match=r"natgrad_lr to be positive"):
                _check_natgrad_lr(schedule, model, 10)

    # A constant negative schedule is rejected for both families too.
    for model in (dual, moment):
        with pytest.raises(ValueError, match=r"natgrad_lr to be positive"):
            _check_natgrad_lr(ox.constant_schedule(-0.1), model, 3)

    # The valid ramp the notebooks use must still pass, for both families.
    valid_ramp = ox.exponential_decay(
        init_value=1e-4, transition_steps=100, decay_rate=1000.0, end_value=1e-1
    )
    for model in (dual, moment):
        _check_natgrad_lr(valid_ramp, model, 200)

    # A decay that only reaches zero past the horizon is still fine.
    _check_natgrad_lr(touches_zero, dual, 5)
    # And without ``num_iters`` nothing is examined, as documented.
    _check_natgrad_lr(crosses_zero, dual)


def test_fit_natgrads_rejects_a_schedule_that_decays_through_zero() -> None:
    """The guard fires at the entry point, not only in the helper."""
    moment, D = _svgp_setup(n_data=10)

    with pytest.raises(ValueError, match=r"natgrad_lr to be positive"):
        fit_natgrads(
            model=moment,
            objective=_negative_elbo,
            train_data=D,
            optim=ox.adam(0.05),
            natgrad_lr=ox.linear_schedule(0.5, -0.5, 5),
            num_iters=10,
            verbose=False,
        )


def test_fit_on_dual_family_still_works() -> None:
    """Plain gradient descent in the dual coordinates remains a valid optimiser."""
    q, D = _dual_svgp_setup(n_data=20)

    trained_model, history = fit(
        model=q,
        objective=_negative_dual_elbo,
        train_data=D,
        optim=ox.adam(1e-2),
        num_iters=20,
        verbose=False,
        key=jr.key(123),
    )

    assert isinstance(trained_model, DualVariationalGaussian)
    assert history.shape == (20,)
    assert bool(jnp.all(jnp.isfinite(history)))
    assert history[-1] < history[0]
