# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: gpjax (3.14.0)
#     language: python
#     name: python3
# ---

# %% [markdown] marimo={"config": {"hide_code": true}, "md_prefix": "r"}
# # Safe optimization
#
# We sample a series of design points $x^1, \ldots, x^m$ in pursuit of a minimum but without $f(x^{(i)})$ exceeding a critical safety threshold $y_{max}$.
#
# The SafeOpt algorithm uses GPs as surrogate models for prediction. At each iteration we fit a GP to the noisy samples. After the $i$th sample, SafeOpt calculates the upper and lower confidence bounds using the posterior distribution.
#
# Since the GP predicts a distribution over $f(x)$ over any design point, we can provide a probabilistic guarantee of safety up to an arbitrary factor.
#
# This enables one to define a predicted safe region $\mathcal{S}$ which consists of design points that provide a probability of safety greater than the required level $p_{\text{safe}}$. SafeOpt aims to choose a sample point that balances the desire to localize a reachable minimizer of $f$ and to expand the safe region.

# %%
from itertools import accumulate
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxtyping import Array, Bool, Float

from utils import (
    use_mpl_style,
)

jax.config.update('jax_enable_x64', True)

# set the default style for plotting
use_mpl_style()

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

key = jr.PRNGKey(0)


# %% [markdown] marimo={"config": {"hide_code": true}, "md_prefix": "r"}
# ## Dataset
#
# We use the Forrester function on $[0, 1]$:
#
# $$f(x) = (6x - 2)^2 \sin(12x - 4).$$
#
# Its landscape (with $y_{max} = 0$) has two safe basins where $f < 0$:
#
# - *Reachable* basin on $x \in (0.0715,\ 0.3333)$, containing the left local minimum at $x \approx 0.143$;
# - *Global* basin on $x \in (0.5951,\ 0.8569)$, containing the global minimum at $x \approx 0.757$ ($f \approx -6.02$),
#
# separated by an unsafe barrier on $(0.3333,\ 0.5951)$ where $f > 0$. SafeOpt seeded in the reachable basin can never cross the barrier, so it can only find the local minimum.

# %%
def forrester(x):
    return (6.0 * x - 2.0) ** 2 * jnp.sin(12.0 * x - 4.0)


# %%
x_grid = jnp.linspace(0.0, 1.0, 200).reshape(-1, 1)
y_true = forrester(x_grid)

# Observations clustered near the basin minimum. This keeps the safe
# region narrow enough that its boundary can still grow, so the expander
# test confirms genuine expanders just outside S on both edges.
obs_idx = jnp.array([22, 26, 30, 34])
x_obs = x_grid[obs_idx]
noise_key = jr.fold_in(key, 0)
sigma_n = 0.1
eps = sigma_n * jr.normal(noise_key, shape=(x_obs.shape))
y_obs = forrester(x_obs) + eps

dataset = gpx.Dataset(X=x_obs, y=y_obs)

# %%
plt.figure(figsize=(10, 4))
plt.plot(x_grid, y_true, label="true f (Forrester)")
plt.scatter(x_obs, y_obs, color="black", zorder=5, label="safe obs")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend();


# %% [markdown] marimo={"config": {"hide_code": true}, "md_prefix": "r"}
# ## Surrogate model
#
# GP hyperparameters are **fixed** where we use a constant mean function and a Matern kernel.

# %%
class ConfidenceInterval(NamedTuple):
    lower: Float[Array, ""]
    upper: Float[Array, ""]


# %%
type Mask = Bool[Array, ""]

type Safe = Mask
type Minimizers = Mask
type Expanders = Mask

# %%
lengthscale = 0.1
signal_var = 4.0
prior_mean = 0.0

kernel = gpx.kernels.Matern52(
    lengthscale=jnp.array(lengthscale),
    variance=jnp.array(signal_var),
)
mean = gpx.mean_functions.Constant(jnp.array(prior_mean))
prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)

likelihood = gpx.likelihoods.Gaussian(obs_stddev=jnp.array(sigma_n))

posterior = prior * likelihood

latent_dist = posterior.predict(x_grid, train_data=dataset)

# Using the latent distribution for the bounds (noiseless)
mu = latent_dist.mean
v = latent_dist.variance

# %% [markdown] marimo={"config": {"hide_code": true}, "md_prefix": "r"}
# ## Safe region
#
# The SafeOpt algorithm uses the GP to predict a distribution over design points $x$. At each iteration, a GP is fitted to the noisy samples. After the $i$th sample, SafeOpt calculates the upper and lower confidence bounds using the posterior distribution.
#
# Since the GP predicts a distribution $f(x)$ over any design point, we can provide a probabilistic guarantee of safety up to an arbitrary factor.
#
# This enables one to define a predicted safe region $\mathcal{S}$ which consists of design points that provide a probability of safety greater than the required level $p_{\text{safe}}$. SafeOpt aims to choose a sample point that balances the desire to localize a reachable minimizer of $f$ and to expand the safe region.

# %%
beta = 3.0
y_max = 0.0
u = mu + jnp.sqrt(beta * v)
l = mu - jnp.sqrt(beta * v)

ci = ConfidenceInterval(lower=l, upper=u)


# %%
def safety_probability(mu, v, y_max, beta):
    # Assumes zero mean and unit variance
    p_safe = jax.scipy.stats.norm.cdf((y_max - mu) / jnp.sqrt(v))
    p_thresh = jax.scipy.stats.norm.cdf(jnp.sqrt(beta))
    return p_safe, p_thresh


# %%
xg = x_grid.ravel()
# Predicted safe set S = {x : u(x) <= y_max}
S = u <= y_max
p_safe, p_thresh = safety_probability(mu, v, y_max, beta)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 4), height_ratios=(2, 1))

ax[0].plot(xg, y_true.ravel(), "k--", lw=1.2, label="true f")
ax[0].plot(xg, mu, color="C0", label=r"GP mean $\hat\mu$")
ax[0].fill_between(xg, l, u, alpha=0.2, color="C0",
                 label=r"$[\ell, u] = \hat\mu \pm \sqrt{\beta \hat v}$")
ax[0].axhline(y_max, color="red", ls=":", label=r"$y_{max}$")
ax[0].scatter(x_obs.ravel(), y_obs.ravel(), color="black", zorder=5, label="obs")
ax[0].plot(xg[S], jnp.full(int(S.sum()), float(y_true.min())), "|",
         color="green", ms=10, label="safe set S")
ax[0].set_xlabel("x")
ax[0].set_ylabel("f(x)")
ax[0].legend(fontsize=8, ncol=2)

ax[1].plot(xg, p_safe, color="green")
ax[1].axhline(p_thresh, color="black", lw=0.75);


# %% [markdown] marimo={"config": {"hide_code": true}, "md_prefix": "r"}
# ## Potential minimizers
#
# The set of potential minimizers $\mathcal{M}$ consists of the safe points whose
# *lower* confidence bound lies below the best (lowest) *upper* bound over the safe
# set. Visually, $\mathcal{M}$ is where the bottom edge of the confidence band drops below the dashed line at $\min_{x' \in S} u(x')$.

# %%
def potential_minimizers(ci: ConfidenceInterval, safe: Safe) -> Minimizers:
    lowest_ub = jnp.min(jnp.where(safe, ci.upper, jnp.inf))
    M: Minimizers = safe & (ci.lower <= lowest_ub)
    return M


# %% jupyter={"source_hidden": true} marimo={"config": {"hide_code": true}} tags=["remove-input"]
def plot_potential_minimizers(
    xg, 
    mu, 
    u, 
    l, 
    safe, 
    M,
    E,
    y_max,
    x_obs=None, 
    y_obs=None, 
    y_true=None,
    ax=None,
    color=None,
):
    color = cols[0] if color is None else color

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    xg = jnp.ravel(xg)
    lowest_ub = float(jnp.min(jnp.where(safe, u, jnp.inf)))   # best safe UB
    y_lo = float(jnp.min(l)) - 0.5
    y_hi = float(jnp.max(u)) + 0.5

    # safe region and minimizer region as vertical bands (where= handles gaps)
    ax.fill_between(xg, y_lo, y_hi, where=safe, step="mid",
                    color=cols[2], alpha=0.10, label="safe region $S$")
    ax.fill_between(xg, y_lo, y_hi, where=M, step="mid",
                    color=cols[1], alpha=0.30, label="potential minimizers $M$")
    ax.fill_between(xg, y_lo, y_hi, where=E, step="mid",
                    color=cols[4], alpha=0.30, label="potential expanders $E$")

    if y_true is not None:
        ax.plot(xg, jnp.ravel(y_true), "k--", lw=1.0, label="true $f$")

    ax.plot(xg, mu, color=color, label=r"GP mean $\hat\mu$")
    ax.fill_between(xg, l, u, color=color, alpha=0.15, label=r"$[\ell, u]$")

    ax.axhline(y_max, color="red", ls=":", label=r"$y_{max}$")
    ax.axhline(lowest_ub, color=cols[3], ls="--", lw=1.0,
               label=r"best safe UB $\min_{S} u$")

    if x_obs is not None:
        ax.scatter(jnp.ravel(x_obs), jnp.ravel(y_obs), color="black",
                   zorder=6, label="obs")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_ylim(y_lo, y_hi)
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    return ax


# %% [markdown] marimo={"config": {"hide_code": true}, "md_prefix": "r"}
# ## Potential expanders
#
# Potential expanders $\mathcal{E}$ are the design points in the safe region, that if added to the surrogate, optimistically assuming the lower bound, produce a posterior distribution with a larger safe set.

# %%
def potential_expanders(
    minimizers: Minimizers,
    safe: Safe,
    ci: ConfidenceInterval,
    dataset: gpx.Dataset,
    x_grid: Array,
    y_max: float,
    bounds: callable
) -> Expanders:
    width = ci.upper - ci.lower
    w_max = jnp.max(jnp.where(minimizers, width, -jnp.inf))

    # Only safe non-minimizers wider than every minimizer can argmax
    candidates = safe & ~minimizers & (width > w_max)

    def expands(i: int):
        optimistic = dataset + gpx.Dataset(
            X=x_grid[i][jnp.newaxis],
            y=ci.lower[i].reshape(1, 1)
        )
        ci_new = bounds(optimistic, x_grid[~safe])
        return bool(jnp.any(ci_new.upper <= y_max))

    expanders = jnp.array([bool(c) and expands(i) for i, c in enumerate(candidates)])
    return expanders


# %% [markdown] marimo={"config": {"hide_code": true}, "md_prefix": "r"}
# ## Safe optimization algorithm
#
# The SafeOpt algorithm is broken down into three primary functions:
#
# * `init`. Set the initial state.
# * `propose`. Propose a new design point.
# * `update`. Transition to a new state.

# %%
class State(NamedTuple):
    dataset: gpx.Dataset
    ci: ConfidenceInterval
    safe: Safe
    minimizers: Minimizers
    expanders: Expanders


# %%
class SafeOptimization(NamedTuple):
    init: callable
    propose: callable
    update: callable


# %%
def safe_opt(bounds_fn, x_design, y_max):
    """SafeOpt over a fixed grid of design points `x_design`."""

    def transition(dataset: gpx.Dataset) -> State:
        """Computes new state for `dataset`."""
        ci = bounds_fn(dataset, x_design)
        S: Safe = ci.upper <= y_max
        M: Minimizers = potential_minimizers(ci, S)
        E: Expanders = potential_expanders(M, S, ci, dataset, x_design, y_max, bounds_fn)
        return State(dataset=dataset, ci=ci, safe=S, minimizers=M, expanders=E)

    def init_fn(dataset: gpx.Dataset) -> State:
        """Returns the initial safe set S, potential minimizers M, and expanders E."""
        return transition(dataset)

    def proposal_fn(state: State) -> Array:
        """The proposed design point is selected among sets M and E with the 
        greatest uncertainty where uncertainty corresponds to the width: 
        
        $w_i(x) = u(x) - l(x)$        
        """
        ME = state.minimizers | state.expanders
        width = state.ci.upper - state.ci.lower
        idx = jnp.argmax(jnp.where(ME, width, -jnp.inf))
        return x_design[idx][jnp.newaxis]

    def update_fn(state: State, x_next: Array, y_next: Array) -> State:
        """Appends new observations to `dataset` and recomputes ci, S, M, E."""
        return transition(state.dataset + gpx.Dataset(X=x_next, y=y_next))

    return SafeOptimization(
        init=init_fn,
        propose=proposal_fn,
        update=update_fn
    )


# %%
def make_confidence_region(prior, obs_stddev, beta):
    def confidence_region(dataset, X):
        """Using the posterior distribution conditioned on `dataset`, compute the
        confidence_region at `X`.
        """
        likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
        dist = (prior * likelihood).predict(X, train_data=dataset)
        w = jnp.sqrt(beta * dist.variance)
        return ConfidenceInterval(lower=dist.loc - w, upper=dist.loc + w)

    return confidence_region


# %%
confidence_region = make_confidence_region(prior, sigma_n, beta)

opt = safe_opt(confidence_region, x_grid, y_max)

def step(state, i):
    x_next = opt.propose(state)
    y_next = forrester(x_next) + sigma_n * jr.normal(jr.fold_in(key, i), x_next.shape)
    return opt.update(state, x_next, y_next)

n_iter = 5
states = list(accumulate(range(1, n_iter + 1), step, initial=opt.init(dataset)))

# %%
for state in states:
    plot_potential_minimizers(
        x_grid,
        (state.ci.lower + state.ci.upper) / 2,
        state.ci.upper,
        state.ci.lower,
        state.safe,
        state.minimizers,
        state.expanders,
        y_max,
        x_obs=state.dataset.X,
        y_obs=state.dataset.y,
        y_true=y_true,
        color=cols[0],
    )
    plt.show()
