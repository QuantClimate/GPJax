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
# # Working with Gridded Data
#
# Download this notebook: {nb-download}`xarray_workflow.ipynb`
#
# Climate and environmental data rarely arrive as a tidy matrix. They arrive as
# labelled [xarray](https://docs.xarray.dev/) objects: a temperature field over
# latitude and longitude, a covariate such as elevation, and gaps where a sensor
# failed or a cloud covered the scene. A GP in GPJax, on the other hand, consumes a
# [`Dataset`](#gpjax.dataset.Dataset) of flat inputs $\mathbf{X} \in \mathbb{R}^{N
# \times D}$ and outputs $\mathbf{y} \in \mathbb{R}^{N \times 1}$.
#
# The [`gpjax.xarray`](../reference/xarray.md) module converts between the two at the
# edges of a workflow. In this notebook we
#
# 1. flatten a gappy, labelled field into a `Dataset` with
#    [`from_xarray`](#gpjax.xarray.from_xarray),
# 2. fit a GP exactly as we would on any other `Dataset`,
# 3. predict the mean and variance on a finer grid with
#    [`GridSpec.predict`](#gpjax.xarray.GridSpec.predict), and
# 4. draw joint posterior samples on that grid with
#    [`GridSpec.inputs_for`](#gpjax.xarray.GridSpec.inputs_for) and
#    [`GridSpec.to_xarray`](#gpjax.xarray.GridSpec.to_xarray).
#
# For the same workflow on real data, see
# [Infilling Global Surface Temperature](infilling_surface_temperature.py).
#
# The module needs the optional extra: `pip install "gpjax[xarray]"`.

# %% tags=["remove-cell"]
import logging

# The build host may not have the ledger fonts (Public Sans, Spectral); hide
# matplotlib's font fallback messages.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# %%
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

config.update("jax_enable_x64", True)

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.xarray import (
        Standardise,
        from_xarray,
    )

key = jr.key(42)
gpx.plotting.use_style()

# %% [markdown]
# ## A synthetic temperature field
#
# We simulate near-surface temperature on a regional latitude-longitude grid. It
# cools towards the pole and with elevation (a lapse rate of roughly 6.5 K per
# kilometre), with a smooth large-scale anomaly on top. Elevation is a separate
# variable over the same grid. The data are synthetic, so the notebook needs no
# download and every number in it can be checked against the truth.


# %%
def elevation_at(lat, lon):
    """A single mountain range, in metres."""
    return 2500.0 * np.exp(-(((lon - 12.0) / 4.0) ** 2) - ((lat - 47.0) / 3.0) ** 2)


def temperature_at(lat, lon, elevation):
    """Temperature in kelvin: latitude gradient, lapse rate and an anomaly."""
    anomaly = 1.5 * np.sin(lon / 3.0) * np.cos(lat / 4.0)
    return 290.0 - 0.6 * (lat - 40.0) - 0.0065 * elevation + anomaly


def regional_field(lats, lons) -> xr.Dataset:
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    elevation = elevation_at(lat_grid, lon_grid)
    return xr.Dataset(
        {
            "t2m": (
                ("lat", "lon"),
                temperature_at(lat_grid, lon_grid, elevation),
                {"units": "K", "long_name": "2 m air temperature"},
            ),
            "elevation": (("lat", "lon"), elevation, {"units": "m"}),
        },
        coords={
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
    )


coarse = regional_field(np.linspace(40.0, 54.0, 12), np.linspace(2.0, 22.0, 16))

# %% [markdown]
# Real observations have holes. We knock out a block of cells, as a cloud would,
# and add a little measurement noise to the rest.

# %%
key, noise_key = jr.split(key)
noise = 0.2 * np.asarray(jr.normal(noise_key, coarse["t2m"].shape))
observed = coarse.copy(deep=True)
observed["t2m"] = observed["t2m"] + noise
observed["t2m"].attrs = coarse["t2m"].attrs
observed["t2m"][4:7, 9:13] = np.nan

observed["t2m"].plot(cmap="coolwarm")
plt.title("Observed temperature (gaps in white)")
plt.show()

# %% [markdown]
# ## From labelled data to a `Dataset`
#
# `from_xarray` takes the target variable and the inputs we want the GP to depend
# on. Inputs can be coordinates (`lat`, `lon`) or other data variables
# (`elevation`), and the columns of $\mathbf{X}$ follow the order we list them in.
# Cells where the target or any input is NaN are dropped by default, and the
# returned `GridSpec` records which ones.
#
# Temperature varies over degrees of latitude and longitude, but over hundreds of
# metres of elevation. The [`Standardise`](#gpjax.xarray.Standardise) transform
# scales each input to zero mean and unit standard deviation over the training
# cells, so one starting lengthscale suits every input. The spec keeps the
# training mean and standard deviation, and applies them to every grid that we
# predict on.

# %%
inputs = ["lat", "lon", "elevation"]
data, spec = from_xarray(
    observed, target="t2m", inputs=inputs, transforms=[Standardise()]
)

print(data)
print(spec)

# %% [markdown]
# `data` is an ordinary `Dataset`, so nothing downstream knows it came from xarray.
# The `GridSpec` stays with us, outside the model, until we want labelled output
# again.
#
# ## Fitting the model
#
# We give the RBF kernel one lengthscale per input, in units of that input's
# standard deviation. A constant mean absorbs the ~285 K offset.

# %%
prior = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Constant(jnp.array([285.0])),
    kernel=gpx.kernels.RBF(lengthscale=jnp.ones(3), variance=25.0),
)
model = prior * gpx.likelihoods.Gaussian(obs_stddev=jnp.array(0.5))

model, history = gpx.fit_scipy(
    model=model,
    objective=lambda candidate, train_data: -gpx.objectives.conjugate_mll(
        candidate, train_data
    ),
    train_data=data,
    verbose=False,
)

# %% [markdown]
# ## Predicting on a finer grid
#
# `spec.predict` gives the predictive mean and variance on any grid that holds the
# same input variables, encoded exactly as in training. Here we predict on a grid
# four times finer in each direction, including the cells that were missing from
# the observations. The function we pass maps a block of inputs to a
# distribution. We use the likelihood's predictive distribution, so the variance
# includes observation noise, and a diagonal covariance, because we need only the
# variance of each cell.
#
# `spec.predict` sends the cells to this function in chunks of `chunk_size`, so
# the memory use stays the same for a global grid. If the grid holds Dask
# arrays, the result is lazy, and each Dask block is predicted only when it is
# computed or written with `to_netcdf`.

# %%
fine = regional_field(np.linspace(40.0, 54.0, 45), np.linspace(2.0, 22.0, 61))
posterior = model.condition(data)

prediction = spec.predict(
    lambda x: model.likelihood(posterior(x, covariance="diagonal")),
    fine[["elevation"]],
    chunk_size=1024,
)
prediction

# %% [markdown]
# The result is a labelled `xr.Dataset` on the fine grid, with the target's
# attributes carried over. The variance is in $\mathrm{K}^2$. Everything xarray
# offers, from plotting to `to_netcdf`, works on it directly.

# %%
fig, (mean_ax, std_ax, error_ax) = plt.subplots(1, 3, figsize=(15, 4))
prediction["t2m_mean"].plot(ax=mean_ax, cmap="coolwarm")
mean_ax.set_title("Predictive mean")
np.sqrt(prediction["t2m_variance"]).plot(ax=std_ax, cmap="viridis")
std_ax.set_title("Predictive standard deviation")
(prediction["t2m_mean"] - fine["t2m"]).plot(ax=error_ax, cmap="RdBu_r", center=0.0)
error_ax.set_title("Error against the true field")
for ax in (mean_ax, std_ax, error_ax):
    ax.add_patch(
        plt.Rectangle(
            (observed.lon[9], observed.lat[4]),
            float(observed.lon[12] - observed.lon[9]),
            float(observed.lat[6] - observed.lat[4]),
            fill=False,
            linestyle="--",
        )
    )
plt.show()

# %% [markdown]
# The standard deviation grows inside the dashed box where observations were
# missing, and the error stays small across the mountain range because elevation is
# an input.
#
# ## Joint samples and regional averages
#
# The mean and variance describe each cell on its own. Many questions are about
# several cells together, such as the average temperature over the Alpine box
# $\mathcal{R}$. Its variance depends on the covariance between the cells,
#
# $$
# \operatorname{Var}\Big[\tfrac{1}{|\mathcal{R}|} \sum_{i \in \mathcal{R}} f_i\Big]
# = \tfrac{1}{|\mathcal{R}|^2} \sum_{i, j \in \mathcal{R}} \operatorname{Cov}[f_i, f_j],
# $$ (eq-xarray-regional-variance)
#
# which the per-cell variances alone cannot give. For this we need the full
# covariance, so we build all the prediction inputs at once with
# `spec.inputs_for`. Passing `num_samples` to `to_xarray` then draws from the joint
# predictive distribution, and returns the draws with a leading `sample`
# dimension. Averaging each draw over the region gives samples of the regional
# mean.

# %%
test_inputs, test_spec = spec.inputs_for(fine[["elevation"]])
latent = posterior(test_inputs)  # the field itself, without observation noise
key, sample_key = jr.split(key)
samples = test_spec.to_xarray(latent, num_samples=500, key=sample_key)

alps = dict(lat=slice(45.0, 49.0), lon=slice(8.0, 16.0))
regional_mean = samples["t2m"].sel(**alps).mean(["lat", "lon"])
true_regional_mean = float(fine["t2m"].sel(**alps).mean())

# The same latent distribution, but treating the cells as independent.
latent_variance = test_spec.to_xarray(latent)["t2m_variance"].sel(**alps)
joint_std = float(regional_mean.std("sample"))
naive_std = float(np.sqrt(latent_variance.sum()) / latent_variance.size)

print(f"Regional mean: {float(regional_mean.mean()):.2f} K (truth {true_regional_mean:.2f} K)")
print(f"Standard deviation from joint samples:        {joint_std:.3f} K")
print(f"Standard deviation if cells were independent: {naive_std:.3f} K")

# %% [markdown]
# Treating the cells as independent understates the uncertainty in the regional
# average by roughly an order of magnitude, because neighbouring cells tend to be
# wrong in the same direction. Against the joint standard deviation the true
# regional mean is a plausible outcome; against the independent one it would look
# like a many-sigma surprise. Joint
# samples keep that correlation, which is why `to_xarray` refuses to draw samples
# from a distribution that only holds marginal variances (as returned by
# `posterior(test_inputs, covariance="diagonal")`).
#
# ## Global grids and seasonal cycles
#
# Latitude and longitude in degrees are not good inputs for a global field. A
# degree of longitude is about 111 km at the equator but only 38 km at 70°N, and
# longitude 359° is next to 0°. The
# [`UnitSphere`](#gpjax.xarray.UnitSphere) transform replaces `lat` and `lon` with
# the three coordinates of a point on the unit sphere. A stationary kernel on these
# coordinates uses the chord distance through the Earth, which has no seam and no
# distortion at the poles. In the same way, [`Cyclic`](#gpjax.xarray.Cyclic)
# encodes a periodic input as a point on a circle. Datetime inputs are days since
# their first timestamp, so a period of 365.25 gives the seasonal cycle:
#
# ```python
# data, spec = from_xarray(
#     ds,
#     target="t2m",
#     inputs=["lat", "lon", "time", "elevation"],
#     transforms=[UnitSphere(), Cyclic("time", 365.25), Standardise(["elevation"])],
# )
# spec.columns
# # ('sphere_x', 'sphere_y', 'sphere_z', 'time_sin', 'time_cos', 'elevation')
# ```
#
# The transforms run in order, and `spec.columns` names the columns of
# $\mathbf{X}$ that they produce, one for each kernel lengthscale.
# [Infilling Global Surface Temperature](infilling_surface_temperature.py) uses
# `UnitSphere` on a real global field.
#
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
