# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Likelihood Guide
#
# Download this notebook: {nb-download}`likelihoods_guide.ipynb`
#
# In this notebook, we will walk users through the process of creating a new likelihood
# in GPJax.
#
# ## Background
#
# In this section we'll provide a short introduction to likelihoods and why they are
# important. For users who are already familiar with likelihoods, feel free to skip to
# the next section, and for users who would like more information than is provided
# here, please see our [introduction to Gaussian processes notebook](intro_to_gps.py).
#
# ### What is a likelihood?
#
# We adopt the notation of our
# [introduction to Gaussian processes notebook](intro_to_gps.py) where we have a
# Gaussian process (GP) $f(\cdot)\sim\mathcal{GP}(m(\cdot), k(\cdot, \cdot))$ and a
# dataset $\mathbf{y} = \{y_n\}_{n=1}^N$ observed at corresponding inputs
# $\mathbf{x} = \{x_n\}_{n=1}^N$. The evaluation of $f$ at $\mathbf{x}$ is denoted by
# $\mathbf{f} = \{f(x_n)\}_{n=1}^N$. The _likelihood function_ of the GP is then given
# by
#
# $$
# p(\mathbf{y}\mid \mathbf{f}) = \prod_{n=1}^N p(y_n\mid f(x_n))\,.
# $$ (eq-likelihoods-factorisation)
#
# Conceptually, this conditional distribution describes the probability of the observed
# data, conditional on the latent function values.
#
# ### Why is the likelihood important?
#
# Choosing the correct likelihood function when building a GP, or any Bayesian model
# for that matter, is crucial. The likelihood function encodes our assumptions about
# the data and the noise that we expect to observe. For example, if we are modelling
# air pollution, then we would not expect to observe negative values of pollution. In
# this case, we would choose a likelihood function that is only defined for positive
# values. Similarly, if our data is the proportion of people who voted for a particular
# political party, then we would expect to observe values between 0 and 1. In this
# case, we would choose a likelihood function that is only defined for values between
# 0 and 1.
#
# ## Likelihoods in GPJax
#
# In GPJax, all likelihoods are a subclass of the
# [`AbstractLikelihood`](#gpjax.likelihoods.AbstractLikelihood) class — the
# [likelihoods reference](../reference/likelihoods.md) lists every implementation
# shipped with the library. This base
# abstract class contains the three core methods that all likelihoods must implement:
# [`predict`](#gpjax.likelihoods.AbstractLikelihood.predict),
# [`link_function`](#gpjax.likelihoods.AbstractLikelihood.link_function), and
# [`expected_log_likelihood`](#gpjax.likelihoods.AbstractLikelihood.expected_log_likelihood).
# We will discuss each of
# these methods in the forthcoming sections, but first, we will show how to instantiate
# a likelihood object. To do this, we'll need a dataset.

# %% tags=["remove-cell"]
import logging

# The build host may not have the ledger fonts (Public Sans, Spectral); hide
# matplotlib's font fallback messages.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# %% mystnb={"figure": {"caption": "Fifty noisy observations of a sinusoid, shown alongside the latent function that generated them.", "name": "fig-likelihoods-guide-data"}}
import jax

# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import gpjax as gpx

config.update("jax_enable_x64", True)


# set the default style for plotting
gpx.plotting.use_style()
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

key = jr.key(42)

n = 50
x = jnp.sort(jr.uniform(key=key, shape=(n, 1), minval=-3.0, maxval=3.0), axis=0)
xtest = jnp.linspace(-3, 3, 100)[:, None]
f = lambda x: jnp.sin(x)
y = f(x) + 0.1 * jr.normal(key, shape=x.shape)
D = gpx.Dataset(x, y)

fig, ax = plt.subplots()
ax.plot(x, y, "o", label="Observations")
ax.plot(x, f(x), label="Latent function")
ax.legend()
plt.show()

# %% [markdown]
# In this example, our observations have support $[-3, 3]$ and are generated from a
# sinusoidal function with Gaussian noise. As such, our response values $\mathbf{y}$
# range between $-1$ and $1$, subject to Gaussian noise. Due to this, a
# [Gaussian likelihood](#gpjax.likelihoods.Gaussian) is appropriate for this dataset as
# it allows for negative values.
#
# As we see in {eq}`eq-likelihoods-factorisation`, the likelihood function factorises over the
# $n$ observations. As such, we must provide this information to GPJax when
# instantiating a likelihood object. We do this by specifying the `num_datapoints`
# argument.

# %%
gpx.likelihoods.Gaussian()

# %% [markdown]
# ### Likelihood parameters
#
# Some likelihoods, such as the Gaussian likelihood, contain parameters that we seek
# to infer. In the case of the Gaussian likelihood, we have a single parameter
# $\sigma^2$ that determines the observation noise. In GPJax, we can specify the value
# of $\sigma$ when instantiating the likelihood object. If we do not specify a
# value, then the likelihood will be initialised with a default value. In the case of
# the Gaussian likelihood, the default value is $1.0$. If we instead wanted to
# initialise the likelihood standard deviation with a value of $0.5$, then we would do
# this as follows:

# %%
gpx.likelihoods.Gaussian(obs_stddev=0.5)

# %% [markdown]
#
# ### Prediction
#
# The `predict` method of a likelihood object transforms the latent distribution of
# the Gaussian process. In the case of a Gaussian likelihood, this simply applies the
# observational noise value to the diagonal values of the covariance matrix. For other
# likelihoods, this may be a more complex transformation. For example, the
# [Bernoulli likelihood](#gpjax.likelihoods.Bernoulli) transforms the latent
# distribution of the Gaussian process into a distribution over binary values.
#
# We visualise this in {numref}`fig-likelihoods-guide-gaussian-samples` for the Gaussian
# likelihood function. In blue we can see
# samples of $\mathbf{f}^{\star}$, whilst in red we see samples of
# $\mathbf{y}^{\star}$.

# %% mystnb={"figure": {"caption": "Three draws of the latent function together with the corresponding samples from the Gaussian predictive distribution.", "name": "fig-likelihoods-guide-gaussian-samples"}}
kernel = gpx.kernels.Matern32()
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(kernel=kernel, mean_function=meanf)

likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)

posterior = prior * likelihood

latent_dist = posterior.predict(xtest, D)

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(9, 2))
key, subkey = jr.split(key)

for ax in axes.ravel():
    subkey, _ = jr.split(subkey)
    ax.plot(
        latent_dist.sample(sample_shape=(1,), key=subkey).T,
        lw=1,
        color=cols[0],
        label="Latent samples",
    )
    ax.plot(
        likelihood.predict(latent_dist).sample(sample_shape=(1,), key=subkey).T,
        "o",
        markersize=5,
        alpha=0.3,
        color=cols[1],
        label="Predictive samples",
    )

# %% [markdown]
# Similarly, for a Bernoulli likelihood function, the samples of $y$ would be binary.

# %% mystnb={"figure": {"caption": "The same latent draws passed through a Bernoulli likelihood, whose predictive samples are constrained to be binary.", "name": "fig-likelihoods-guide-bernoulli-samples"}}
likelihood = gpx.likelihoods.Bernoulli()


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(9, 2))
key, subkey = jr.split(key)

for ax in axes.ravel():
    subkey, _ = jr.split(subkey)
    ax.plot(
        latent_dist.sample(sample_shape=(1,), key=subkey).T,
        lw=1,
        color=cols[0],
        label="Latent samples",
    )
    ax.plot(
        likelihood.predict(latent_dist).sample(sample_shape=(1,), key=subkey).T,
        "o",
        markersize=3,
        alpha=0.5,
        color=cols[1],
        label="Predictive samples",
    )

# %% [markdown]
# ### Link functions
#
# In {numref}`fig-likelihoods-guide-bernoulli-samples`, we can see the latent samples
# being constrained to be either 0 or
# 1 when a Bernoulli likelihood is specified. This is achieved by the
# `inverse link_function` $\eta(\cdot)$ of the likelihood. The link function is a
# deterministic function that maps the latent distribution of the Gaussian process to
# the support of the likelihood function. For example, the link function of the
# Bernoulli likelihood that is used in GPJax is the inverse probit function
#
# $$
# \eta(x) = 0.5\left(1 + \Phi\left(\frac{x}{\sqrt{2}}\right) * (1-2)\right)\,,
# $$ (eq-likelihoods-inverse-probit)
#
# where $\Phi$ is the cumulative distribution function of the standard normal
# distribution.
#
# A table of commonly used link functions and their corresponding likelihood can be
# found [here](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function).
#
# ### Expected log likelihood
#
# The final method that is associated with a likelihood function in GPJax is the
# expected log-likelihood. This term is evaluated in the
# [stochastic variational Gaussian process](uncollapsed_vi.py) in the ELBO term. For a
# variational approximation $q(f)= \mathcal{N}(f\mid m, S)$, the ELBO can be written as
#
# $$
# \begin{aligned}
#     \mathcal{L}(q) & = \mathbb{E}_{f\sim q(f)}\left[ p(\mathbf{y}\mid f)\right] \\
#                    & \quad - \mathrm{KL}\left(q(f)\mid\mid p(f)\right)\,.
# \end{aligned}
# $$ (eq-likelihoods-elbo)
#
# As both $q(f)$ and $p(f)$ are Gaussian distributions, the Kullback-Leibler term can
# be analytically computed. However, the expectation term is not always so easy to
# compute. Fortunately, the bound in {eq}`eq-likelihoods-elbo` can be decomposed as a
# sum of the datapoints
#
# $$
# \begin{aligned}
#     \mathcal{L}(q) & = \sum_{n=1}^N \mathbb{E}_{f\sim q(f)}\left[ p(y_n\mid f)\right] \\
#                    & \quad - \mathrm{KL}\left(q(f)\mid\mid p(f)\right)\,.
# \end{aligned}
# $$ (eq-likelihoods-elbo-decomposed)
#
# This simplifies computation of the expectation as it is now a series of $N$
# 1-dimensional integrals. As such, GPJax by default uses quadrature to compute these
# integrals. However, for some likelihoods, such as the Gaussian likelihood, the
# expectation can be computed analytically. In these cases, we can supply an object
# that inherits from [`AbstractIntegrator`](#gpjax.integrators.AbstractIntegrator) to
# the likelihood upon instantiation. To see
# this, let us consider a Gaussian likelihood where we'll first define a variational
# approximation to the posterior.

# %%
z = jnp.linspace(-3.0, 3.0, 10).reshape(-1, 1)
q = gpx.variational_families.VariationalGaussian(model=posterior, inducing_inputs=z)


def q_moments(x):
    qx = q(x)
    return qx.mean, qx.variance


mean, variance = jax.vmap(q_moments)(x[:, None])

# %% [markdown]
# Now that we have the variational mean and variational (co)variance, we can compute
# the expected log-likelihood using the `expected_log_likelihood` method of the
# likelihood object.

# %%
jnp.sum(likelihood.expected_log_likelihood(y=y, mean=mean, variance=variance))

# %% [markdown]
# However, had we wanted to do this using quadrature, then we would have done the
# following:

# %%
lquad = gpx.likelihoods.Gaussian(
    obs_stddev=jnp.array([0.1]),
    integrator=gpx.integrators.GHQuadratureIntegrator(num_points=20),
)

# %% [markdown]
# However, this is not recommended for the Gaussian likelihood given that the
# expectation can be computed analytically.

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
