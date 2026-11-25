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
# # Classification
#
# Download this notebook: {nb-download}`classification.ipynb`
#
# In this notebook we demonstrate how to perform inference for Gaussian process models
# with non-Gaussian likelihoods via maximum a posteriori (MAP). We focus on a classification task here.
# The conjugate case, where a Gaussian likelihood leaves the posterior available in
# closed form, is covered in the [regression notebook](regression.py); everything that
# follows is a consequence of losing that closed form.

# %% tags=["remove-cell"]
import logging

# The build host may not have the ledger fonts (Public Sans, Spectral); hide
# matplotlib's font fallback messages.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# %%
import equinox as eqx
from gpjax.linalg import add_jitter, cholesky_factor
from gpjax.parameters import val
import jax

# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
import lineax as lx
import matplotlib.pyplot as plt
import numpyro.distributions as npd
import optax as ox

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


identity_matrix = jnp.eye

# set the default style for plotting
gpx.plotting.use_style()

key = jr.key(42)
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Dataset
#
# With the necessary modules imported, we simulate a dataset
# $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y}) = \{(x_i, y_i)\}_{i=1}^{100}$ with inputs
# $\boldsymbol{x}$ sampled uniformly on $(-1., 1)$ and corresponding binary outputs
#
# $$
# \begin{aligned}
# \boldsymbol{y} & = 0.5 * \text{sign}(\cos(3 \boldsymbol{x} + \boldsymbol{\epsilon})) + 0.5, \\
# \boldsymbol{\epsilon} & \sim \mathcal{N} \left(\textbf{0}, \textbf{I} * (0.05)^{2} \right).
# \end{aligned}
# $$ (eq-classification-data-generating)
#
# We store our data $\mathcal{D}$ as a GPJax [`Dataset`](#gpjax.dataset.Dataset) and
# create test inputs for later.

# %%
key, subkey = jr.split(key)
x = jr.uniform(key, shape=(100, 1), minval=-1.0, maxval=1.0)
y = 0.5 * jnp.sign(jnp.cos(3 * x + jr.normal(subkey, shape=x.shape) * 0.05)) + 0.5

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-1.0, 1.0, 500).reshape(-1, 1)

fig, ax = plt.subplots()
ax.scatter(x, y)

# %% [markdown]
# ## MAP inference
#
# We begin by defining a Gaussian process prior with a radial basis function
# ([`RBF`](#gpjax.kernels.RBF)) kernel, chosen for the purpose of exposition. Since our
# observations are binary, we choose a
# [`Bernoulli`](#gpjax.likelihoods.Bernoulli) likelihood with a probit link function. Our
# [likelihood guide](likelihoods_guide.py#link-functions) shows what that link function
# does to the latent distribution.

# %%
kernel = gpx.kernels.RBF()
meanf = gpx.mean_functions.Constant()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
likelihood = gpx.likelihoods.Bernoulli()

# %% [markdown]
# We construct the posterior through the product of our prior and likelihood.

# %%
posterior = prior * likelihood
print(type(posterior))

# %% [markdown]
# Whilst the latent function is Gaussian, the posterior distribution is non-Gaussian
# since our generative model first samples the latent GP and propagates these samples
# through the likelihood function's inverse link function. This step prevents us from
# being able to analytically integrate the latent function's values out of our
# posterior, and we must instead adopt alternative inference techniques. We begin with
# maximum a posteriori (MAP) estimation, a fast inference procedure to obtain point
# estimates for the latent function and the kernel's hyperparameters by maximising the
# marginal log-likelihood. Sampling the latent function instead of optimising it is the
# other route out of this problem, and we take it in the
# [count data regression notebook](poisson.py).

# %% [markdown]
# We can obtain a MAP estimate by optimising the
# [log-posterior density](#gpjax.objectives.log_posterior_density) with
# [Optax](inv:optax#index)'s optimisers.

# %%
optimiser = ox.adam(learning_rate=0.01)
opt_posterior, history = gpx.fit(
    model=posterior,
    # we use the negative lpd as we are minimising
    objective=lambda p, d: -gpx.objectives.log_posterior_density(p, d),
    train_data=D,
    optim=ox.adamw(learning_rate=0.01),
    num_iters=1000,
    key=key,
)

# %% [markdown]
# From which we can [make predictions](#gpjax.gps.JointModel.predict) at
# novel inputs, as illustrated in {numref}`fig-classification-map-predictive`.

# %% mystnb={"figure": {"caption": "The MAP predictive mean and its one-sigma band over the binary observations.", "name": "fig-classification-map-predictive"}}
map_latent_dist = opt_posterior.predict(xtest, train_data=D)
predictive_dist = opt_posterior.likelihood(map_latent_dist)

predictive_mean = predictive_dist.mean
predictive_std = jnp.sqrt(predictive_dist.variance)

fig, ax = plt.subplots()
ax.scatter(x, y, label="Observations", color=cols[0])
ax.plot(xtest, predictive_mean, label="Predictive mean", color=cols[1])
ax.fill_between(
    xtest.squeeze(),
    predictive_mean - predictive_std,
    predictive_mean + predictive_std,
    alpha=0.2,
    color=cols[1],
    label="One sigma",
)
ax.plot(
    xtest,
    predictive_mean - predictive_std,
    color=cols[1],
    linestyle="--",
    linewidth=1,
)
ax.plot(
    xtest,
    predictive_mean + predictive_std,
    color=cols[1],
    linestyle="--",
    linewidth=1,
)

ax.legend()
plt.show()
# %% [markdown]
# Here we projected the map estimates $\hat{\boldsymbol{f}}$ for the function values
# $\boldsymbol{f}$ at the data points $\boldsymbol{x}$ to get predictions over the
# whole domain,
#
# $$
# \begin{aligned}
# p(f(\cdot)| \mathcal{D}) & \approx q_{map}(f(\cdot)) \\
# & := \int p(f(\cdot)| \boldsymbol{f}) \delta(\boldsymbol{f} - \hat{\boldsymbol{f}}) d \boldsymbol{f} \\
# & = \mathcal{N}(\mathbf{K}_{\boldsymbol{(\cdot)x}}  \mathbf{K}_{\boldsymbol{xx}}^{-1} \hat{\boldsymbol{f}},  \mathbf{K}_{\boldsymbol{(\cdot, \cdot)}} - \mathbf{K}_{\boldsymbol{(\cdot)\boldsymbol{x}}} \mathbf{K}_{\boldsymbol{xx}}^{-1} \mathbf{K}_{\boldsymbol{\boldsymbol{x}(\cdot)}}).
# \end{aligned}
# $$ (eq-classification-map-predictive)

# %% [markdown]
# However, as a point estimate, MAP estimation is severely limited for uncertainty
# quantification, providing only a single piece of information about the posterior.

# %% [markdown]
# ## Laplace approximation
# The Laplace approximation improves uncertainty quantification by incorporating
# curvature induced by the marginal log-likelihood's Hessian to construct an
# approximate Gaussian distribution centered on the MAP estimate. Writing
# $\tilde{p}(\boldsymbol{f}|\mathcal{D}) = p(\boldsymbol{y}|\boldsymbol{f}) p(\boldsymbol{f})$
# as the unormalised posterior for function values $\boldsymbol{f}$ at the datapoints
# $\boldsymbol{x}$, we can expand the log of this about the posterior mode
# $\hat{\boldsymbol{f}}$ via a Taylor expansion. This gives:
#
# $$
# \begin{aligned}
# \log\tilde{p}(\boldsymbol{f}|\mathcal{D}) & = \log\tilde{p}(\hat{\boldsymbol{f}}|\mathcal{D}) + \left[\nabla \log\tilde{p}({\boldsymbol{f}}|\mathcal{D})|_{\hat{\boldsymbol{f}}}\right]^{T} (\boldsymbol{f}-\hat{\boldsymbol{f}}) \\
# & \quad + \frac{1}{2} (\boldsymbol{f}-\hat{\boldsymbol{f}})^{T} \left[\nabla^2 \tilde{p}(\boldsymbol{y}|\boldsymbol{f})|_{\hat{\boldsymbol{f}}} \right] (\boldsymbol{f}-\hat{\boldsymbol{f}}) \\
# & \quad + \mathcal{O}(\lVert \boldsymbol{f} - \hat{\boldsymbol{f}} \rVert^3).
# \end{aligned}
# $$ (eq-classification-laplace-taylor)
#
# Since $\nabla \log\tilde{p}({\boldsymbol{f}}|\mathcal{D})$ is zero at the mode,
# this suggests the following approximation
#
# $$
# \begin{aligned}
# \tilde{p}(\boldsymbol{f}|\mathcal{D}) & \approx \log\tilde{p}(\hat{\boldsymbol{f}}|\mathcal{D}) \exp\left\{ \frac{1}{2} (\boldsymbol{f}-\hat{\boldsymbol{f}})^{T} \left[-\nabla^2 \tilde{p}(\boldsymbol{y}|\boldsymbol{f})|_{\hat{\boldsymbol{f}}} \right] (\boldsymbol{f}-\hat{\boldsymbol{f}}) \right\},
# \end{aligned}
# $$ (eq-classification-laplace-approximation)
#
# that we identify as a Gaussian distribution,
# $p(\boldsymbol{f}| \mathcal{D}) \approx q(\boldsymbol{f}) := \mathcal{N}(\hat{\boldsymbol{f}}, [-\nabla^2 \tilde{p}(\boldsymbol{y}|\boldsymbol{f})|_{\hat{\boldsymbol{f}}} ]^{-1} )$.
# Since the negative Hessian is positive definite, we can use the Cholesky
# decomposition to obtain the covariance matrix of the Laplace approximation at the
# datapoints below.

# %%
gram, cross_covariance = (kernel.gram, kernel.cross_covariance)
jitter = 1e-6

# Compute (latent) function value map estimates at training points:
Kxx = opt_posterior.prior.kernel.gram(x).as_matrix()
Kxx = add_jitter(Kxx, jitter)
Lx = jnp.linalg.cholesky(Kxx)
f_hat = Lx @ val(opt_posterior.latent)

# Negative Hessian,  H = -∇²p_tilde(y|f):
params, static = eqx.partition(opt_posterior, eqx.is_array)


def loss(params, D):
    model = eqx.combine(params, static)
    return -gpx.objectives.log_posterior_density(model, D)


jacobian = jax.jacfwd(jax.jacrev(loss))(params, D)
H = jacobian.latent.value.latent.value[:, 0, :, 0]
L = jnp.linalg.cholesky(H + identity_matrix(D.n) * jitter)

# H⁻¹ = H⁻¹ I = (LLᵀ)⁻¹ I = L⁻ᵀL⁻¹ I
L_inv = jsp.linalg.solve_triangular(L, identity_matrix(D.n), lower=True)
H_inv = jsp.linalg.solve_triangular(L.T, L_inv, lower=False)
LH = jnp.linalg.cholesky(H_inv)
laplace_approximation = npd.MultivariateNormal(f_hat.squeeze(), scale_tril=LH)


# %% [markdown]
# For novel inputs, we must project the above approximating distribution through the
# Gaussian conditional distribution $p(f(\cdot)| \boldsymbol{f})$,
#
# $$
# \begin{aligned}
# p(f(\cdot)| \mathcal{D}) & \approx q_{Laplace}(f(\cdot)) \\
# & := \int p(f(\cdot)| \boldsymbol{f}) q(\boldsymbol{f}) d \boldsymbol{f} \\
# & = \mathcal{N}(\mathbf{K}_{\boldsymbol{(\cdot)x}}  \mathbf{K}_{\boldsymbol{xx}}^{-1} \hat{\boldsymbol{f}},  \mathbf{K}_{\boldsymbol{(\cdot, \cdot)}} - \mathbf{K}_{\boldsymbol{(\cdot)\boldsymbol{x}}} \mathbf{K}_{\boldsymbol{xx}}^{-1} (\mathbf{K}_{\boldsymbol{xx}} - [-\nabla^2 \tilde{p}(\boldsymbol{y}|\boldsymbol{f})|_{\hat{\boldsymbol{f}}} ]^{-1}) \mathbf{K}_{\boldsymbol{xx}}^{-1} \mathbf{K}_{\boldsymbol{\boldsymbol{x}(\cdot)}}).
# \end{aligned}
# $$ (eq-classification-laplace-predictive)
#
# This is the same approximate distribution $q_{map}(f(\cdot))$, but we have perturbed
# the covariance by a curvature term of
# $\mathbf{K}_{\boldsymbol{(\cdot)\boldsymbol{x}}} \mathbf{K}_{\boldsymbol{xx}}^{-1} [-\nabla^2 \tilde{p}(\boldsymbol{y}|\boldsymbol{f})|_{\hat{\boldsymbol{f}}} ]^{-1} \mathbf{K}_{\boldsymbol{xx}}^{-1} \mathbf{K}_{\boldsymbol{\boldsymbol{x}(\cdot)}}$.
# We take the latent distribution computed in the previous section and add this term
# to the covariance to construct $q_{Laplace}(f(\cdot))$.


# %%
def construct_laplace(test_inputs: Float[Array, "N D"]) -> npd.MultivariateNormal:
    map_latent_dist = opt_posterior.predict(xtest, train_data=D)

    Kxt = opt_posterior.prior.kernel.cross_covariance(x, test_inputs)
    Kxx = opt_posterior.prior.kernel.gram(x).as_matrix()
    Kxx = add_jitter(Kxx, jitter)

    # Kxx⁻¹ Kxt
    Kxx_inv_Kxt = jnp.linalg.solve(Kxx, Kxt)

    # Ktx Kxx⁻¹[ H⁻¹ ] Kxx⁻¹ Kxt
    laplace_cov_term = jnp.matmul(jnp.matmul(Kxx_inv_Kxt.T, H_inv), Kxx_inv_Kxt)

    mean = map_latent_dist.mean
    covariance = map_latent_dist.covariance_matrix + laplace_cov_term
    L = jnp.linalg.cholesky(covariance)
    return npd.MultivariateNormal(jnp.atleast_1d(mean.squeeze()), scale_tril=L)


# %% [markdown]
# From this we can construct the predictive distribution at the test points.
# %% mystnb={"figure": {"caption": "The Laplace predictive mean and its one-sigma band, whose covariance carries the extra curvature term added to the MAP approximation.", "name": "fig-classification-laplace-predictive"}}
laplace_latent_dist = construct_laplace(xtest)
predictive_dist = opt_posterior.likelihood(laplace_latent_dist)

predictive_mean = predictive_dist.mean
predictive_std = jnp.sqrt(predictive_dist.variance)

fig, ax = plt.subplots()
ax.scatter(x, y, label="Observations", color=cols[0])
ax.plot(xtest, predictive_mean, label="Predictive mean", color=cols[1])
ax.fill_between(
    xtest.squeeze(),
    predictive_mean - predictive_std,
    predictive_mean + predictive_std,
    alpha=0.2,
    color=cols[1],
    label="One sigma",
)
ax.plot(
    xtest,
    predictive_mean - predictive_std,
    color=cols[1],
    linestyle="--",
    linewidth=1,
)
ax.plot(
    xtest,
    predictive_mean + predictive_std,
    color=cols[1],
    linestyle="--",
    linewidth=1,
)
ax.legend()
plt.show()

# %% [markdown]
# ## System configuration

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -a "Thomas Pinder & Daniel Dodd"
