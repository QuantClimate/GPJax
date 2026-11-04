"""Numerically stable scaled modified Bessel functions for state-space GPs.

Leaf module imported by both `sde.py` and `kernels.py`.

See plans/2026-04-21-state-space-gps-design.md §Bessel stability.

Three regimes are dispatched via nested `jax.lax.cond` so that only the taken
branch contributes under `jit` and `grad`:

- `c > truncation_order`        : forward recurrence seeded by `i0e`/`i1e`
- `1e-8 <= c <= truncation_order` : Miller's downward recurrence rescaled by
  `i0e(c)`
- `c < 1e-8`                    : power-series ``Ĩ_k(c) ≈ (c/2)^k / k! · e^{-c}``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp


def _large_c_scaled_ive(c, truncation_order: int):
    """Forward recurrence for Ĩ_k(c) = e^{-c} I_k(c), seeded by i0e/i1e.

    The recurrence

        Ĩ_{k+1}(c) = Ĩ_{k-1}(c) - (2k/c) Ĩ_k(c)

    inherits stability from the unscaled identity since the e^{-c} factor
    cancels. For ``c > truncation_order`` the values ``I_k(c)`` are comparable
    in magnitude across ``k``, so upward (forward) recurrence is numerically
    stable here; downward (Miller) recurrence is reserved for
    ``c <= truncation_order``, where ``I_k(c)`` decays in ``k`` and forward
    recurrence would amplify round-off.
    """
    c = jnp.asarray(c)
    bessel_at_order_0 = jsp.i0e(c)
    bessel_at_order_1 = jsp.i1e(c)

    if truncation_order == 0:
        return jnp.stack([bessel_at_order_0])
    if truncation_order == 1:
        return jnp.stack([bessel_at_order_0, bessel_at_order_1])

    def recurrence_step(carry, current_order):
        bessel_at_lower, bessel_at_current = carry
        bessel_at_upper = (
            bessel_at_lower - (2.0 * current_order / c) * bessel_at_current
        )
        return (bessel_at_current, bessel_at_upper), bessel_at_upper

    middle_orders = jnp.arange(1, truncation_order)
    _, higher_order_values = jax.lax.scan(
        recurrence_step,
        (bessel_at_order_0, bessel_at_order_1),
        middle_orders,
    )
    return jnp.concatenate(
        [jnp.stack([bessel_at_order_0, bessel_at_order_1]), higher_order_values]
    )


def _miller_downward_scaled_ive(c, truncation_order: int):
    """Miller's downward recurrence for Ĩ_k(c), rescaled by i0e(c).

    The recurrence runs on the ratios r_k = I_k(c) / I_{k-1}(c). It starts at
    `k_max = 2 * truncation_order + 20` with r_{k_max + 1} = 0 and steps down with
        r_k = c / (2k + c r_{k+1}),
    which is Miller's recurrence I_{k-1} = (2k / c) I_k + I_{k+1} divided through by
    I_k. Then Ĩ_k(c) = i0e(c) · r_1 · … · r_k. Each ratio is in (0, 1), so nothing
    overflows: the unnormalised Miller values grow like (2 k_max / c)^{k_max} and
    overflow for small c and large truncation orders, which gives zeros or NaNs
    after the rescale. `k_max` is static (compile-time); `c` stays traced.
    """
    c = jnp.asarray(c)
    k_max = 2 * truncation_order + 20

    def recurrence_step(ratio_above, order):
        ratio = c / (2.0 * order + c * ratio_above)
        return ratio, ratio

    orders = jnp.arange(k_max, 0, -1, dtype=c.dtype)
    _, ratios = jax.lax.scan(recurrence_step, jnp.zeros((), c.dtype), orders)
    # `ratios` holds r_{k_max}, ..., r_1; keep r_1, ..., r_truncation_order.
    ratios = ratios[::-1][:truncation_order]
    return jsp.i0e(c) * jnp.concatenate([jnp.ones(1, c.dtype), jnp.cumprod(ratios)])


def _tiny_c_series_scaled_ive(c, truncation_order: int):
    """Power-series Ĩ_k(c) ≈ (c/2)^k / k! · e^{-c}. Exact for c → 0."""
    c = jnp.asarray(c)
    bessel_orders = jnp.arange(truncation_order + 1)
    log_factorial = jsp.gammaln(bessel_orders + 1.0)
    # Use log-space to avoid underflow at k_max.
    return jnp.exp(bessel_orders * jnp.log(c / 2.0) - log_factorial - c)


def _stable_scaled_ive(c, truncation_order: int):
    """Compute Ĩ_k(c) = e^{-c} I_k(c) for k = 0, …, truncation_order.

    Three regimes, selected via nested `jax.lax.cond` so only the taken branch
    contributes under jit and grad:

    - ``c > truncation_order``         : forward recurrence seeded by i0e/i1e
    - ``1e-8 <= c <= truncation_order`` : Miller downward recurrence + i0e rescale
    - ``c < 1e-8``                     : power-series Ĩ_k(c) ≈ (c/2)^k/k! · e^{-c}

    `truncation_order` is static (Python int), so both branch closures compile
    to fixed-shape computations.
    """
    c = jnp.asarray(c)

    def large_branch(c_):
        return _large_c_scaled_ive(c_, truncation_order)

    def small_branch(c_):
        return jax.lax.cond(
            c_ <= 1e-8,
            lambda cc: _tiny_c_series_scaled_ive(cc, truncation_order),
            lambda cc: _miller_downward_scaled_ive(cc, truncation_order),
            c_,
        )

    return jax.lax.cond(c > float(truncation_order), large_branch, small_branch, c)
