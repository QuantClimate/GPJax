"""Shared fixtures and helpers for ASV benchmarks.

The contract every benchmark in this directory follows:

1. Float64 is enabled globally here (mirroring tests/conftest.py).
2. Every timed call must end with realise(result) so the GPU/dispatcher
   actually completes the work before the timer stops. JAX is async;
   without this we time dispatch, not work.
3. No conftest.py exists in this directory — beartype and hypothesis
   must NOT activate during benchmark runs.
"""

from __future__ import annotations

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)


KEY = jr.key(0)

# Shared sizes for cross-suite comparison plots. Suites that opt into the
# same axis (e.g. ConjugateMll vs Elbo vs CollapsedElbo) parametrise over
# this so points line up on the dashboard.
ALIGNED_NS = (1000, 3000)

# Default inducing-point count used wherever M is not explicitly varied
# (e.g. ConjugateMllSuite alignment, HeteroscedasticElboSuite, the
# variational-parametrisation suite, the compile suite).
M_INDUCING = 50

# Inducing-point counts for VFE / SVGP suites that vary M alongside N.
M_INDUCING_GRID = (32, 128)

# SVGP mini-batch sizes. Both must be <= min(ALIGNED_NS).
SVGP_BATCH_SIZES = (128, 512)


def realise(result):
    """Block until every JAX array leaf in ``result`` has finished computing.

    Pass anything: a JAX array, a tuple, a dict, an Equinox Module — anything
    a JAX PyTree can hold. Non-array leaves are ignored.
    """
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        result,
    )
    return result


def make_inputs(n: int, d: int = 1, key=KEY) -> jnp.ndarray:
    """Standard input grid: uniform on [0, 1]^d, shape (n, d)."""
    return jr.uniform(key, (n, d), dtype=jnp.float64)


def make_outputs(n: int, q: int = 1, key=KEY) -> jnp.ndarray:
    """Standard output array, shape (n, q)."""
    return jr.normal(key, (n, q), dtype=jnp.float64)


def make_shared_dataset(n: int) -> gpx.Dataset:
    """Single-source-of-truth Gaussian dataset for cross-suite comparisons.

    Every suite that participates in the sparse-vs-full comparison must
    obtain its data here. Identical bytes at identical n; deterministic
    across runs.
    """
    X = make_inputs(n)
    y = make_outputs(n, key=jr.fold_in(KEY, 1))
    return gpx.Dataset(X=X, y=y)
