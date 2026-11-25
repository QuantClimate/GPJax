"""Validation helpers for state-space kernels and datasets."""

from __future__ import annotations

import warnings

import jax.numpy as jnp

from gpjax.parameters import val


def _validate_temporal_kernel(kernel) -> None:
    """Raise ``ValueError`` unless ``kernel`` is configured for a 1-D temporal input.

    State-space GPs require kernels with scalar lengthscale, no active_dims
    selection (or trivial ``slice(None)``), and ``n_dims`` either ``None`` or 1.
    """
    lengthscale = val(kernel.lengthscale)
    if jnp.asarray(lengthscale).ndim != 0:
        raise ValueError(
            "State-space kernels require a scalar lengthscale (1-D temporal input). "
            f"Got lengthscale of shape {jnp.asarray(lengthscale).shape}."
        )

    n_dims = getattr(kernel, "n_dims", None)
    if n_dims is not None and n_dims != 1:
        raise ValueError(
            "State-space kernels require n_dims == 1 (1-D temporal input). "
            f"Got n_dims={n_dims}."
        )

    # GPJax constrains active_dims to list | slice at construction
    # (kernels.base._check_active_dims rejects tuples/ndarrays with TypeError;
    # the jaxtyping/beartype annotation list[int] | slice | None rejects them too
    # when the import hook is active), and a slice selects a single contiguous
    # axis given 1-D temporal data, so the only multi-axis selector that can
    # reach here is a length>1 list.
    active_dims = getattr(kernel, "active_dims", None)
    if isinstance(active_dims, list) and len(active_dims) > 1:
        raise ValueError(
            "State-space kernels require single-axis active_dims (1-D temporal input). "
            f"Got active_dims={active_dims}."
        )


def validate_state_space_data(X, y, observation_mask=None) -> None:
    """Validate ``(X, y, observation_mask)`` for state-space inference.

    Raises:
        ValueError: If ``X`` is not 2-D, has more than one column, has a
            different length from ``y``, contains non-finite entries, or if
            ``observation_mask`` has a length mismatch with ``X``.

    See plans/2026-04-21-state-space-gps-design.md §Stage 4 (Task 7.1).
    """
    X_array = jnp.asarray(X)
    y_array = jnp.asarray(y)

    if X_array.ndim != 2:
        raise ValueError(
            f"X must be a 2-D array of shape (N, 1); got shape {X_array.shape}."
        )
    if X_array.shape[1] != 1:
        raise ValueError(
            "State-space inference requires 1-D temporal input — X must have a "
            f"single column; got shape {X_array.shape}."
        )
    if y_array.shape[0] != X_array.shape[0]:
        raise ValueError(
            "X and y must share length N; got "
            f"X.shape={X_array.shape}, y.shape={y_array.shape}."
        )
    if not bool(jnp.all(jnp.isfinite(X_array))):
        raise ValueError("X contains non-finite (NaN or Inf) entries.")
    if not bool(jnp.all(jnp.isfinite(y_array))):
        raise ValueError("y contains non-finite (NaN or Inf) entries.")
    if X_array.dtype != jnp.float64 or y_array.dtype != jnp.float64:
        warnings.warn(
            f"Running state-space inference with X dtype={X_array.dtype}, "
            f"y dtype={y_array.dtype}. Numerical conditioning can degrade for "
            "long sequences; enable jax_enable_x64.",
            UserWarning,
            stacklevel=2,
        )
    if observation_mask is not None:
        mask_array = jnp.asarray(observation_mask)
        if mask_array.shape[0] != X_array.shape[0]:
            raise ValueError(
                f"observation_mask length {mask_array.shape[0]} does not match "
                f"data length {X_array.shape[0]}."
            )


def sort_state_space_data(X, y, observation_mask=None):
    """Jointly sort ``(X, y, observation_mask)`` by ascending time if unsorted.

    State-space inference assumes the inputs are sorted in time. If they are
    not, a :class:`UserWarning` is emitted and the triple is jointly re-ordered
    by the argsort of ``X``. If the inputs are already sorted, they are
    returned unchanged.

    Args:
        X (Float[Array, "num_train 1"]):
        y (Float[Array, "num_train ..."]):
        observation_mask (Bool[Array, "num_train"] | None):

    Returns:
        tuple: ``(X_sorted, y_sorted, mask_sorted)`` — the (possibly reordered)
            triple. ``mask_sorted`` is ``None`` iff the input mask was ``None``.

    See plans/2026-04-21-state-space-gps-design.md §Stage 4 (Task 7.2).
    """
    X_array = jnp.asarray(X)
    y_array = jnp.asarray(y)
    mask_array = jnp.asarray(observation_mask) if observation_mask is not None else None
    times = X_array.squeeze(-1)
    if not bool(jnp.all(jnp.diff(times) >= 0)):
        warnings.warn(
            "X is unsorted; jointly sorting (X, y, mask) by ascending time. "
            "State-space inference assumes time-sorted inputs.",
            UserWarning,
            stacklevel=2,
        )
        order = jnp.argsort(times)
        X_array = X_array[order]
        y_array = y_array[order]
        if mask_array is not None:
            mask_array = mask_array[order]
    return X_array, y_array, mask_array
