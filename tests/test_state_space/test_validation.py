"""Tests for state-space validation helpers."""

import warnings

import gpjax as gpx
from gpjax.state_space._validation import (
    _validate_temporal_kernel,
    sort_state_space_data,
    validate_state_space_data,
)
import jax.numpy as jnp
import numpy as np
import pytest


def test_validate_temporal_kernel_accepts_scalar_kernel():
    kernel = gpx.kernels.Matern12(lengthscale=1.0, variance=1.0)
    _validate_temporal_kernel(kernel)


def test_validate_temporal_kernel_rejects_vector_lengthscale():
    kernel = gpx.kernels.Matern12(lengthscale=jnp.ones(3), n_dims=3)
    with pytest.raises(ValueError, match=r"scalar lengthscale|1-D"):
        _validate_temporal_kernel(kernel)


def test_validate_temporal_kernel_rejects_active_dims_list():
    kernel = gpx.kernels.Matern12(active_dims=[0, 1], lengthscale=1.0, n_dims=2)
    with pytest.raises(ValueError, match=r"active_dims|1-D"):
        _validate_temporal_kernel(kernel)


def test_kernel_construction_rejects_tuple_active_dims_upstream():
    """active_dims is constrained to list|slice at construction, so non-list
    multi-axis selectors (tuple/ndarray) can never reach _validate_temporal_kernel.
    This pins the upstream contract that makes the list-only check sufficient.

    Construction rejects a tuple either via ``_check_active_dims`` (a manual
    ``TypeError`` reading "list or slice") or, when the beartype import hook is
    active (as in this test suite), via a jaxtyping/beartype type-check whose
    message names the ``list[int], slice`` annotation. Either way the tuple
    cannot reach ``_validate_temporal_kernel``.
    """
    with pytest.raises(
        Exception, match=r"list or slice|list\[int\], slice|list\[int\] \| slice"
    ):
        gpx.kernels.Matern32(active_dims=(0, 1))


def test_validate_temporal_kernel_rejects_n_dims_greater_than_one():
    kernel = gpx.kernels.Matern12(lengthscale=jnp.ones(2), n_dims=2)
    with pytest.raises(ValueError, match=r"n_dims|1-D"):
        _validate_temporal_kernel(kernel)


def test_validate_state_space_data_accepts_valid_inputs():
    X = jnp.array([[0.0], [1.0], [2.5]])
    y = jnp.array([[0.1], [0.2], [0.3]])
    mask = jnp.array([True, True, False])
    validate_state_space_data(X, y, mask)


def test_validate_state_space_data_rejects_1d_X():
    X = jnp.zeros((3,))
    y = jnp.zeros((3, 1))
    with pytest.raises(ValueError, match=r"2-D|shape"):
        validate_state_space_data(X, y)


def test_validate_state_space_data_rejects_3d_X():
    X = jnp.zeros((3, 1, 1))
    y = jnp.zeros((3, 1))
    with pytest.raises(ValueError, match=r"2-D|shape"):
        validate_state_space_data(X, y)


def test_validate_state_space_data_rejects_multi_d_X():
    X = jnp.zeros((3, 2))
    y = jnp.zeros((3, 1))
    with pytest.raises(ValueError, match=r"1-D|single|column"):
        validate_state_space_data(X, y)


def test_validate_state_space_data_rejects_mismatched_lengths():
    X = jnp.zeros((3, 1))
    y = jnp.zeros((4, 1))
    with pytest.raises(ValueError, match=r"length|N"):
        validate_state_space_data(X, y)


def test_validate_state_space_data_rejects_non_finite_X():
    X = jnp.array([[0.0], [jnp.nan], [2.0]])
    y = jnp.array([[0.0], [0.0], [0.0]])
    with pytest.raises(ValueError, match=r"finite|nan|NaN"):
        validate_state_space_data(X, y)


def test_validate_state_space_data_rejects_mask_length_mismatch():
    X = jnp.array([[0.0], [1.0], [2.0]])
    y = jnp.array([[0.0], [0.0], [0.0]])
    mask = jnp.array([True, True])  # wrong length
    with pytest.raises(ValueError, match=r"mask|length"):
        validate_state_space_data(X, y, mask)


def test_sort_state_space_data_returns_sorted_inputs_unchanged_when_already_sorted():
    X = jnp.array([[0.0], [1.0], [2.0]])
    y = jnp.array([[0.1], [0.2], [0.3]])
    mask = jnp.array([True, True, False])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # any warning becomes an error
        X_sorted, y_sorted, mask_sorted = sort_state_space_data(X, y, mask)
    np.testing.assert_array_equal(np.asarray(X_sorted), np.asarray(X))
    np.testing.assert_array_equal(np.asarray(y_sorted), np.asarray(y))
    np.testing.assert_array_equal(np.asarray(mask_sorted), np.asarray(mask))


def test_sort_state_space_data_warns_and_sorts_unsorted():
    X = jnp.array([[2.0], [0.0], [1.0]])
    y = jnp.array([[0.3], [0.1], [0.2]])
    mask = jnp.array([False, True, True])
    with pytest.warns(UserWarning, match="unsorted|sort"):
        X_sorted, y_sorted, mask_sorted = sort_state_space_data(X, y, mask)
    np.testing.assert_allclose(
        np.asarray(X_sorted).squeeze(), np.array([0.0, 1.0, 2.0])
    )
    np.testing.assert_allclose(
        np.asarray(y_sorted).squeeze(), np.array([0.1, 0.2, 0.3])
    )
    np.testing.assert_array_equal(
        np.asarray(mask_sorted), np.array([True, True, False])
    )
