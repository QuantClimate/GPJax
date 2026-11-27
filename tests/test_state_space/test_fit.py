"""Tests for state_space.fit wrappers."""

import subprocess
import sys
import textwrap

import gpjax as gpx
from gpjax.parameters import val
from gpjax.state_space.fit import (
    fit,
    fit_lbfgs,
    fit_scipy,
)
from gpjax.state_space.gps import StateSpacePrior
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import paramax
import pytest


def _build_matern52_synthetic(
    n=200, lengthscale_true=0.7, variance_true=1.5, obs_stddev_true=0.3, seed=42
):
    """Generate sorted samples from a Matern52 GP for hyperparameter recovery."""
    key = jr.key(seed)
    key_x, key_eps_kern, key_eps_obs = jr.split(key, 3)
    X = jnp.sort(jr.uniform(key_x, shape=(n,), minval=0.0, maxval=20.0))
    kernel = gpx.kernels.Matern52(lengthscale=lengthscale_true, variance=variance_true)
    K = kernel.gram(X.reshape(-1, 1)).as_matrix()
    L = jnp.linalg.cholesky(K + 1e-9 * jnp.eye(n))
    f = L @ jr.normal(key_eps_kern, shape=(n,))
    y = f + obs_stddev_true * jr.normal(key_eps_obs, shape=(n,))
    return X, y


def test_fit_scipy_recovers_matern52_hyperparameters_at_N_2000():
    """End-to-end: fit_scipy on Matern52 should recover lengthscale within ~50% of truth.

    Phase 11 hyperparameter-recovery integration gate at N=2000. Exercises the
    gradient-safe ``_psd_sqrt`` path: at this density, ``Q = P_inf - A P_inf A.T``
    routinely picks up round-off-induced near-zero eigenvalues, and a naive
    ``maximum(eigvals, 0) -> sqrt`` would produce NaN reverse-mode gradients.
    """
    n = 2000
    lengthscale_true = 0.7
    variance_true = 1.5
    obs_stddev_true = 0.3
    X, y = _build_matern52_synthetic(
        n=n,
        lengthscale_true=lengthscale_true,
        variance_true=variance_true,
        obs_stddev_true=obs_stddev_true,
        seed=0,
    )
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))

    # Initialise away from truth.
    init_lengthscale = 1.5
    init_variance = 1.0
    init_obs_stddev = 0.5
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern52(
            lengthscale=init_lengthscale, variance=init_variance
        ),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=init_obs_stddev)
    posterior = prior * likelihood

    fitted_posterior, history = fit_scipy(
        model=posterior,
        train_data=train_data,
        max_iters=200,
        verbose=False,
    )

    # Loose recovery checks (within ~50% of truth).
    assert 0.4 < float(val(fitted_posterior.prior.kernel.lengthscale)) < 1.4, (
        f"lengthscale recovery off: got {float(val(fitted_posterior.prior.kernel.lengthscale))}, true {lengthscale_true}"
    )
    assert 0.5 < float(val(fitted_posterior.prior.kernel.variance)) < 4.0
    assert 0.15 < float(val(fitted_posterior.likelihood.obs_stddev)) < 0.55

    # Final negative-MLL strictly less than initial.
    assert history[-1] < history[0]


# The test stops L-BFGS-B after 10 iterations to stay fast, so SciPy can report
# that it did not converge. Convergence is not what this test checks.
@pytest.mark.filterwarnings("ignore::scipy.optimize.OptimizeWarning")
def test_fit_scipy_returns_state_space_posterior_type():
    n = 50
    X, y = _build_matern52_synthetic(n=n, seed=1)
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.3)
    posterior = prior * likelihood
    from gpjax.state_space.gps import StateSpaceConjugateModel

    fitted_posterior, _ = fit_scipy(
        model=posterior,
        train_data=train_data,
        max_iters=10,
        verbose=False,
    )
    assert isinstance(fitted_posterior, StateSpaceConjugateModel)


# The test stops L-BFGS-B after 10 iterations to stay fast, so SciPy can report
# that it did not converge. Convergence is not what this test checks.
@pytest.mark.filterwarnings("ignore::scipy.optimize.OptimizeWarning")
def test_fit_scipy_warns_on_unsorted_input_and_sorts_internally():
    """Passing unsorted X should emit a UserWarning and still complete successfully."""
    n = 50
    X_sorted, y_sorted = _build_matern52_synthetic(n=n, seed=2)
    perm = np.random.RandomState(0).permutation(n)
    X_unsorted = X_sorted[perm]
    y_unsorted = y_sorted[perm]
    train_data = gpx.Dataset(X=X_unsorted.reshape(-1, 1), y=y_unsorted.reshape(-1, 1))
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.3)
    posterior = prior * likelihood
    with pytest.warns(UserWarning, match=r"unsorted|sort"):
        fit_scipy(model=posterior, train_data=train_data, max_iters=10, verbose=False)


def test_fit_scipy_points_the_convergence_warning_at_the_caller():
    """The warning names the caller, not the GPJax wrapper that calls SciPy.

    The test suite wraps every GPJax function with jaxtyping and beartype (see
    tests/conftest.py), and the warning would then name a jaxtyping file. So the
    fit runs in a new Python process, without those wrappers, where the caller
    is the code passed to ``python -c``.
    """
    code = textwrap.dedent(
        """
        import warnings

        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from scipy.optimize import OptimizeWarning

        import gpjax as gpx
        from gpjax.state_space import StateSpacePrior, fit_scipy

        X = jnp.linspace(0.0, 5.0, 20).reshape(-1, 1)
        prior = StateSpacePrior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
        )
        posterior = prior * gpx.likelihoods.Gaussian(obs_stddev=0.3)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            fit_scipy(
                model=posterior,
                train_data=gpx.Dataset(X=X, y=jnp.sin(X)),
                max_iters=1,
                verbose=False,
            )
        print(next(w.filename for w in record if w.category is OptimizeWarning))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "<string>"


def test_fit_lbfgs_does_not_expose_verbose():
    """fit_lbfgs must not advertise a verbose flag the optimiser ignores."""
    import inspect

    from gpjax.state_space import fit_lbfgs

    assert "verbose" not in inspect.signature(fit_lbfgs).parameters


def test_fit_lbfgs_returns_state_space_posterior_type():
    n = 50
    X, y = _build_matern52_synthetic(n=n, seed=3)
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.3)
    posterior = prior * likelihood
    from gpjax.state_space.gps import StateSpaceConjugateModel

    fitted_posterior, _ = fit_lbfgs(
        model=posterior,
        train_data=train_data,
        max_iters=10,
    )
    assert isinstance(fitted_posterior, StateSpaceConjugateModel)


def test_fit_optax_runs_full_batch():
    """state_space.fit with default batch_size=-1 should run successfully."""
    n = 50
    X, y = _build_matern52_synthetic(n=n, seed=4)
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.3)
    posterior = prior * likelihood
    from gpjax.state_space.gps import StateSpaceConjugateModel

    fitted_posterior, history = fit(
        model=posterior,
        train_data=train_data,
        optim=ox.adam(0.01),
        num_iters=10,
        verbose=False,
    )
    assert isinstance(fitted_posterior, StateSpaceConjugateModel)
    assert history.shape == (10,)


def test_fit_rejects_minibatch_with_value_error():
    n = 50
    X, y = _build_matern52_synthetic(n=n, seed=5)
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.3)
    posterior = prior * likelihood
    with pytest.raises(ValueError, match=r"batch_size|full[- ]batch"):
        fit(
            model=posterior,
            train_data=train_data,
            optim=ox.adam(0.01),
            num_iters=5,
            batch_size=10,
            verbose=False,
        )


def test_paramax_non_trainable_freezes_lengthscale():
    """A paramax.non_trainable-wrapped parameter must remain unchanged after fit_scipy.

    Uses Matern12 (rather than Matern52) because the latter currently produces
    NaN gradients in state_space_mll for the data sizes used here — an upstream
    SDE-discretisation issue independent of the freeze plumbing under test.
    """
    import equinox as eqx

    n = 200
    X, y = _build_matern52_synthetic(n=n, seed=6)
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))

    init_lengthscale_value = 0.5
    init_variance = 1.0
    init_obs_stddev = 0.3

    kernel = gpx.kernels.Matern12(
        lengthscale=init_lengthscale_value,
        variance=init_variance,
    )
    # Freeze the lengthscale by wrapping its parameter in paramax.non_trainable.
    frozen_kernel = eqx.tree_at(
        lambda k: k.lengthscale,
        kernel,
        replace_fn=paramax.non_trainable,
    )
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=frozen_kernel,
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=init_obs_stddev)
    posterior = prior * likelihood

    fitted_posterior, _ = fit_scipy(
        model=posterior,
        train_data=train_data,
        max_iters=50,
        verbose=False,
    )
    fitted_lengthscale_value = float(val(fitted_posterior.prior.kernel.lengthscale))
    np.testing.assert_allclose(
        fitted_lengthscale_value,
        init_lengthscale_value,
        atol=1e-12,
    )

    # Sanity: variance and/or obs_stddev DID move (otherwise the freeze test is vacuous).
    fitted_variance = float(val(fitted_posterior.prior.kernel.variance))
    fitted_obs_stddev = float(val(fitted_posterior.likelihood.obs_stddev))
    assert (
        abs(fitted_variance - init_variance) > 1e-3
        or abs(fitted_obs_stddev - init_obs_stddev) > 1e-3
    )
