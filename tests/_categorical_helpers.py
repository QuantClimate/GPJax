"""Shared fixtures for the ``Categorical`` (multi-class) likelihood.

``tests/test_conditioning.py``, ``tests/test_variational_families.py``,
``tests/test_objectives.py`` and ``tests/test_fit.py`` all need the same
three-class data, prior and whitened family. They are written once here.
"""

import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr

NUM_CLASSES = 3


def three_class_data(dtype=jnp.float64) -> gpx.Dataset:
    r"""Four inputs on $[-1, 1]$ with the class indices 0, 1, 2, 1."""
    inputs = jnp.linspace(-1.0, 1.0, 4, dtype=dtype).reshape(-1, 1)
    labels = jnp.array([[0.0], [1.0], [2.0], [1.0]], dtype=dtype)
    return gpx.Dataset(X=inputs, y=labels)


def zero_mean_rbf_prior() -> gpx.gps.Prior:
    r"""A Gaussian process prior with a zero mean function and an RBF kernel."""
    return gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(), kernel=gpx.kernels.RBF()
    )


def whitened_family(num_inducing: int = 3, seed: int = 0, dtype=jnp.float64):
    r"""A ``WhitenedVariationalGaussian`` with random moments for three classes.

    Returns:
        The family, its inducing inputs, its variational mean of shape
        ``(M, 3)`` and its root of shape ``(3, M, M)``. Each root is lower
        triangular with a diagonal of at least 0.5.
    """
    model = zero_mean_rbf_prior() * gpx.likelihoods.Categorical(num_classes=NUM_CLASSES)
    inducing_inputs = jnp.linspace(-1.0, 1.0, num_inducing, dtype=dtype).reshape(-1, 1)
    mean_key, root_key = jr.split(jr.key(seed))
    mean = jr.normal(mean_key, (num_inducing, NUM_CLASSES), dtype=dtype)
    root = jnp.tril(
        jr.normal(root_key, (NUM_CLASSES, num_inducing, num_inducing), dtype=dtype)
    )
    diagonal_index = jnp.arange(num_inducing)
    root = root.at[:, diagonal_index, diagonal_index].set(
        jnp.abs(jnp.diagonal(root, axis1=1, axis2=2)) + 0.5
    )
    family = gpx.variational_families.WhitenedVariationalGaussian(
        model=model,
        inducing_inputs=inducing_inputs,
        variational_mean=mean,
        variational_root_covariance=root,
    )
    return family, inducing_inputs, mean, root
