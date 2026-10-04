# Copyright 2026 The thomaspinder Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the xarray bridge, exercised only through `gpjax.xarray`."""

import subprocess
import sys

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.xarray import (
    Cyclic,
    Standardise,
    UnitSphere,
    from_xarray,
)
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import xarray as xr


@pytest.fixture
def field() -> xr.Dataset:
    """A (time=2, lat=2, lon=3) temperature field with a (lat, lon) covariate."""
    times = np.array(["2020-01-01", "2020-01-03"], dtype="datetime64[ns]")
    lats = np.array([10.0, 20.0])
    lons = np.array([0.0, 1.0, 2.0])
    temperature = np.arange(12, dtype=float).reshape(2, 2, 3)
    elevation = np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]])
    return xr.Dataset(
        {
            "t2m": (("time", "lat", "lon"), temperature, {"units": "K"}),
            "elevation": (("lat", "lon"), elevation),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )


def test_columns_follow_input_order(field):
    data, _ = from_xarray(field, target="t2m", inputs=["lon", "lat"])

    assert isinstance(data, Dataset)
    assert data.X.shape == (12, 2)
    assert data.y.shape == (12, 1)
    # C-order over (time, lat, lon): the first three cells walk lon at lat=10.
    np.testing.assert_array_equal(data.X[:3], [[0.0, 10.0], [1.0, 10.0], [2.0, 10.0]])
    np.testing.assert_array_equal(data.y[:, 0], np.arange(12.0))


def test_covariate_repeats_across_missing_dims(field):
    data, _ = from_xarray(field, target="t2m", inputs=["elevation"])

    # elevation(lat, lon) is the same at both times: rows 0-5 and 6-11 match.
    expected = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0] * 2
    np.testing.assert_array_equal(data.X[:, 0], expected)


def test_input_with_dim_absent_from_target_is_rejected(field):
    field["depth"] = (("lat", "lon", "level"), np.zeros((2, 3, 4)))

    with pytest.raises(ValueError, match=r"depth.*level"):
        from_xarray(field, target="t2m", inputs=["lat", "depth"])


def test_nan_cells_are_dropped_and_counted(field):
    field["t2m"][0, 0, 0] = np.nan  # a gap in the target
    field["elevation"][1, 2] = np.nan  # a gap in a covariate, at both times

    data, spec = from_xarray(field, target="t2m", inputs=["lat", "elevation"])

    assert spec.n_dropped == 3
    assert data.n == 9
    assert not np.isnan(np.asarray(data.X)).any()
    assert not np.isnan(np.asarray(data.y)).any()


def test_dropna_false_raises_on_nan(field):
    field["t2m"][0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="1 cell"):
        from_xarray(field, target="t2m", inputs=["lat"], dropna=False)


def test_all_nan_target_raises(field):
    field["t2m"][:] = np.nan

    with pytest.raises(ValueError, match="no cells"):
        from_xarray(field, target="t2m", inputs=["lat"])


def _diagonal_gaussian(mean, variance) -> GaussianDistribution:
    return GaussianDistribution(
        loc=jnp.asarray(mean), scale=lx.DiagonalLinearOperator(jnp.asarray(variance))
    )


def test_round_trip_restores_the_field_and_its_gaps(field):
    field["t2m"][0, 0, 0] = np.nan
    data, spec = from_xarray(field, target="t2m", inputs=["lat", "lon"])

    variance = np.full(data.n, 4.0)
    out = spec.to_xarray(_diagonal_gaussian(data.y[:, 0], variance))

    expected_mean = field["t2m"].rename("t2m_mean")
    xr.testing.assert_identical(out["t2m_mean"], expected_mean)
    assert np.isnan(out["t2m_variance"][0, 0, 0])
    assert float(out["t2m_variance"][1, 1, 2]) == 4.0
    assert out["t2m_variance"].attrs["units"] == "(K)^2"


def test_read_out_rejects_a_distribution_from_another_grid(field):
    _, spec = from_xarray(field, target="t2m", inputs=["lat"])

    with pytest.raises(ValueError, match=r"5 points.*expects 12"):
        spec.to_xarray(_diagonal_gaussian(np.zeros(5), np.ones(5)))


@pytest.fixture
def two_cell_field() -> xr.Dataset:
    return xr.Dataset(
        {"t2m": (("site",), [1.0, 2.0], {"units": "K"})},
        coords={"site": [0.0, 1.0]},
    )


def test_samples_are_joint_draws_with_a_sample_dim(two_cell_field):
    _, spec = from_xarray(two_cell_field, target="t2m", inputs=["site"])
    covariance = np.array([[1.0, 0.9], [0.9, 1.0]])
    dist = GaussianDistribution(
        loc=jnp.array([3.0, -1.0]),
        scale=lx.MatrixLinearOperator(
            jnp.asarray(covariance), lx.positive_semidefinite_tag
        ),
    )

    out = spec.to_xarray(dist, num_samples=20_000, key=jr.key(0))

    draws = out["t2m"]
    assert draws.dims == ("sample", "site")
    assert draws.attrs["units"] == "K"
    assert draws.attrs["description"] == "joint posterior predictive draws"
    np.testing.assert_allclose(draws.mean("sample"), [3.0, -1.0], atol=0.03)
    np.testing.assert_allclose(np.cov(draws.values.T), covariance, atol=0.03)


def test_samples_from_diagonal_covariance_are_refused(two_cell_field):
    _, spec = from_xarray(two_cell_field, target="t2m", inputs=["site"])
    dist = _diagonal_gaussian(np.zeros(2), np.ones(2))

    with pytest.raises(ValueError, match='covariance="dense"'):
        spec.to_xarray(dist, num_samples=10, key=jr.key(0))


def test_samples_require_a_key(two_cell_field):
    _, spec = from_xarray(two_cell_field, target="t2m", inputs=["site"])
    dist = _diagonal_gaussian(np.zeros(2), np.ones(2))

    with pytest.raises(ValueError, match="key"):
        spec.to_xarray(dist, num_samples=10)


def test_datetimes_become_days_since_the_first_timestamp(field):
    data, _ = from_xarray(field, target="t2m", inputs=["time"])

    # 2020-01-01 -> 0 days, 2020-01-03 -> 2 days; six cells at each time.
    np.testing.assert_array_equal(data.X[:, 0], [0.0] * 6 + [2.0] * 6)


def test_non_numeric_input_is_rejected(field):
    field["station"] = (("lat",), np.array(["north", "south"]))

    with pytest.raises(TypeError, match="station"):
        from_xarray(field, target="t2m", inputs=["station"])


def test_inputs_for_the_training_grid_reproduces_training_inputs(field):
    inputs = ["time", "lat", "elevation"]
    data, spec = from_xarray(field, target="t2m", inputs=inputs)

    test_inputs, _ = spec.inputs_for(field)

    np.testing.assert_array_equal(test_inputs, data.X)


def test_inputs_for_reuses_the_training_time_origin(field):
    _, spec = from_xarray(field, target="t2m", inputs=["time"])
    later = xr.Dataset(
        coords={"time": np.array(["2020-01-03", "2020-01-05"], dtype="datetime64[ns]")}
    )

    test_inputs, _ = spec.inputs_for(later)

    np.testing.assert_array_equal(test_inputs[:, 0], [2.0, 4.0])


def test_inputs_for_masks_nan_inputs_and_reads_out_nan_there(field):
    _, spec = from_xarray(field, target="t2m", inputs=["lat", "lon", "elevation"])
    grid = field.drop_vars("t2m")
    grid["elevation"][0, 1] = np.nan

    test_inputs, test_spec = spec.inputs_for(grid)
    out = test_spec.to_xarray(
        _diagonal_gaussian(np.ones(test_inputs.shape[0]), np.ones(test_inputs.shape[0]))
    )

    assert test_inputs.shape == (5, 3)
    assert out["t2m_mean"].dims == ("lat", "lon")
    assert out["t2m_mean"].attrs["units"] == "K"
    assert np.isnan(out["t2m_mean"][0, 1])
    assert int(out["t2m_mean"].count()) == 5


def test_inputs_for_rejects_a_time_input_that_is_now_numeric(field):
    _, spec = from_xarray(field, target="t2m", inputs=["time"])
    numeric_time = xr.Dataset(coords={"time": [0.0, 1.0]})

    with pytest.raises(TypeError, match="time"):
        spec.inputs_for(numeric_time)


def test_inputs_for_rejects_a_numeric_input_that_is_now_datetime(field):
    _, spec = from_xarray(field, target="t2m", inputs=["lat"])
    dated = xr.Dataset(coords={"lat": np.array(["2020-01-01"], dtype="datetime64[ns]")})

    with pytest.raises(TypeError, match="lat"):
        spec.inputs_for(dated)


def test_every_missing_name_is_reported(field):
    with pytest.raises(ValueError, match=r"'t3m'.*'altitude'"):
        from_xarray(field, target="t3m", inputs=["lat", "altitude"])


def test_inputs_for_reports_a_missing_input(field):
    _, spec = from_xarray(field, target="t2m", inputs=["lat", "elevation"])

    with pytest.raises(ValueError, match="elevation"):
        spec.inputs_for(field.drop_vars("elevation"))


@pytest.mark.parametrize(
    "inputs, message",
    [
        (["lat", "lat"], "duplicate"),
        (["lat", "t2m"], "target"),
        ([], "at least one"),
    ],
)
def test_ambiguous_inputs_are_rejected(field, inputs, message):
    with pytest.raises(ValueError, match=message):
        from_xarray(field, target="t2m", inputs=inputs)


def test_a_named_dataarray_is_accepted(field):
    data, _ = from_xarray(field["t2m"], target="t2m", inputs=["lat", "lon"])

    assert data.X.shape == (12, 2)


def test_an_unnamed_dataarray_is_rejected(field):
    with pytest.raises(ValueError, match="name"):
        from_xarray(field["t2m"].rename(None), target="t2m", inputs=["lat"])


def test_the_dataset_is_a_plain_pytree_under_jit(field):
    data, _ = from_xarray(field, target="t2m", inputs=["lat", "lon"])

    total = jax.jit(lambda dataset: dataset.y.sum())(data)

    assert float(total) == 66.0


def test_importing_gpjax_does_not_import_xarray():
    probe = "import sys, gpjax; assert 'xarray' not in sys.modules"

    subprocess.run([sys.executable, "-c", probe], check=True)


def test_spec_repr_summarises_the_grid(field):
    field["t2m"][0, 0, 0] = np.nan
    _, spec = from_xarray(field, target="t2m", inputs=["lat", "lon"])

    assert repr(spec) == (
        "GridSpec(target='t2m', inputs=('lat', 'lon'), "
        "grid={'time': 2, 'lat': 2, 'lon': 3}, kept=11, dropped=1)"
    )


def test_fit_on_a_coarse_grid_and_predict_on_a_finer_one():
    import gpjax as gpx

    def smooth_field(lat, lon):
        return np.sin(lat / 20.0) + np.cos(lon / 25.0)

    def gridded(lats, lons) -> xr.Dataset:
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        return xr.Dataset(
            {"t2m": (("lat", "lon"), smooth_field(lat_grid, lon_grid))},
            coords={"lat": lats, "lon": lons},
        )

    coarse = gridded(np.linspace(-60.0, 60.0, 7), np.linspace(0.0, 120.0, 7))
    fine = gridded(np.linspace(-50.0, 50.0, 11), np.linspace(10.0, 110.0, 13))
    data, spec = from_xarray(coarse, target="t2m", inputs=["lat", "lon"])

    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.RBF(lengthscale=jnp.array([20.0, 20.0])),
    )
    model = prior * gpx.likelihoods.Gaussian(obs_stddev=jnp.array(0.01))
    model, _ = gpx.fit_scipy(
        model=model,
        objective=lambda model, data: -gpx.objectives.conjugate_mll(model, data),
        train_data=data,
        max_iters=50,
        verbose=False,
    )

    posterior = model.condition(data)
    test_inputs, test_spec = spec.inputs_for(fine.drop_vars("t2m"))
    out = test_spec.to_xarray(posterior(test_inputs))

    assert out["t2m_mean"].sizes == {"lat": 11, "lon": 13}
    np.testing.assert_allclose(out["t2m_mean"], fine["t2m"], atol=0.1)

    chunked = spec.predict(
        lambda x: posterior(x, covariance="diagonal"),
        fine.drop_vars("t2m"),
        chunk_size=50,
    )
    xr.testing.assert_allclose(chunked, out)


def test_samples_from_a_tagged_diagonal_covariance_are_refused(two_cell_field):
    _, spec = from_xarray(two_cell_field, target="t2m", inputs=["site"])
    tagged = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(jnp.ones(2)), lx.diagonal_tag
    )
    dist = GaussianDistribution(loc=jnp.zeros(2), scale=tagged)

    with pytest.raises(ValueError, match='covariance="dense"'):
        spec.to_xarray(dist, num_samples=10, key=jr.key(0))


def test_inputs_for_keeps_the_training_dim_order(field):
    _, spec = from_xarray(field, target="t2m", inputs=["lon", "lat"])

    _, test_spec = spec.inputs_for(field)
    out = test_spec.to_xarray(_diagonal_gaussian(np.zeros(6), np.ones(6)))

    assert out["t2m_mean"].dims == ("lat", "lon")


def test_inputs_for_keeps_non_dimension_coords(field):
    field = field.assign_coords(region=("lat", ["south", "north"]))
    _, spec = from_xarray(field, target="t2m", inputs=["lat", "lon"])

    _, test_spec = spec.inputs_for(field)
    out = test_spec.to_xarray(_diagonal_gaussian(np.zeros(6), np.ones(6)))

    assert list(out["region"].values) == ["south", "north"]


def test_an_all_nat_datetime_input_raises_clearly(field):
    field["time"] = np.array(["NaT", "NaT"], dtype="datetime64[ns]")

    with pytest.raises(ValueError, match="no cells"):
        from_xarray(field, target="t2m", inputs=["time"])


@pytest.mark.parametrize(
    "values",
    [
        np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        np.array(["warm", "cold"]),
    ],
)
def test_a_non_numeric_target_is_rejected(values):
    dataset = xr.Dataset({"label": (("site",), values)}, coords={"site": [0.0, 1.0]})

    with pytest.raises(TypeError, match="label"):
        from_xarray(dataset, target="label", inputs=["site"])


def test_a_bare_string_for_inputs_is_rejected(field):
    with pytest.raises(TypeError, match="list"):
        from_xarray(field, target="t2m", inputs="lat")


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.bool_])
def test_inputs_of_any_numeric_dtype_become_floats(dtype):
    dataset = xr.Dataset(
        {"t2m": (("site",), [1.0, 2.0])},
        coords={"site": np.array([0, 1], dtype=dtype)},
    )

    data, _ = from_xarray(dataset, target="t2m", inputs=["site"])

    np.testing.assert_array_equal(data.X[:, 0], [0.0, 1.0])
    assert data.X.dtype == jnp.float64


# --- Input transforms -------------------------------------------------------


def test_standardise_gives_columns_with_zero_mean_and_unit_std(field):
    data, spec = from_xarray(
        field, target="t2m", inputs=["lat", "elevation"], transforms=[Standardise()]
    )

    np.testing.assert_allclose(data.X.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(data.X.std(axis=0), 1.0)
    (standardise,) = spec.transforms
    assert standardise.loc == {"lat": 15.0, "elevation": 350.0}


def test_standardise_scales_a_new_grid_with_the_training_statistics(field):
    data, spec = from_xarray(
        field, target="t2m", inputs=["elevation"], transforms=[Standardise()]
    )
    (standardise,) = spec.transforms
    grid = xr.Dataset(
        {"elevation": ("site", [350.0, 350.0 + standardise.scale["elevation"]])}
    )

    test_inputs, _ = spec.inputs_for(grid)

    np.testing.assert_allclose(test_inputs[:, 0], [0.0, 1.0])
    # elevation(lat, lon) alone spans 6 cells: the first time step of training.
    training_inputs, _ = spec.inputs_for(field.drop_vars("t2m"))
    np.testing.assert_allclose(training_inputs, data.X[:6])


def test_standardise_changes_only_the_named_columns(field):
    data, _ = from_xarray(
        field,
        target="t2m",
        inputs=["lat", "elevation"],
        transforms=[Standardise(["elevation"])],
    )

    np.testing.assert_array_equal(np.unique(data.X[:, 0]), [10.0, 20.0])
    np.testing.assert_allclose(data.X[:, 1].std(), 1.0)


def test_standardise_rejects_a_constant_input(field):
    field["flat"] = (("lat", "lon"), np.ones((2, 3)))

    with pytest.raises(ValueError, match=r"cannot standardise \['flat'\]"):
        from_xarray(field, target="t2m", inputs=["flat"], transforms=[Standardise()])


def test_standardise_rejects_a_bare_string():
    with pytest.raises(TypeError, match=r"\['lat'\]"):
        Standardise("lat")


def test_unit_sphere_replaces_lat_and_lon_with_unit_vectors():
    lons = np.array([0.0, 90.0, 360.0])
    field = xr.Dataset(
        {"t2m": (("lat", "lon"), np.zeros((2, 3)))},
        coords={"lat": [0.0, 90.0], "lon": lons, "level": 1.0},
    )
    field["height"] = (("lat", "lon"), np.arange(6.0).reshape(2, 3))

    data, spec = from_xarray(
        field, target="t2m", inputs=["height", "lat", "lon"], transforms=[UnitSphere()]
    )

    assert spec.columns == ("height", "sphere_x", "sphere_y", "sphere_z")
    xyz = np.asarray(data.X[:, 1:])
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 1.0)
    # On the equator: lon 0 -> +x, lon 90 -> +y, and lon 360 is lon 0 again.
    np.testing.assert_allclose(xyz[:3], [[1, 0, 0], [0, 1, 0], [1, 0, 0]], atol=1e-12)
    # At the pole every longitude is the same point.
    np.testing.assert_allclose(xyz[3:], [[0, 0, 1]] * 3, atol=1e-12)


def test_unit_sphere_rejects_latitude_outside_the_degree_range(field):
    field = field.assign_coords(lat=[10.0, 100.0])

    with pytest.raises(ValueError, match=r"outside \[-90, 90\]"):
        from_xarray(
            field, target="t2m", inputs=["lat", "lon"], transforms=[UnitSphere()]
        )


def test_cyclic_maps_values_one_period_apart_to_the_same_point(field):
    data, spec = from_xarray(
        field, target="t2m", inputs=["lon"], transforms=[Cyclic("lon", period=2.0)]
    )

    assert spec.columns == ("lon_sin", "lon_cos")
    # lon is 0, 1, 2 along each row: 0 and 2 are one period apart.
    np.testing.assert_allclose(data.X[0], data.X[2], atol=1e-12)
    np.testing.assert_allclose(data.X[1], [0.0, -1.0], atol=1e-12)


def test_cyclic_on_a_datetime_input_uses_days_since_the_origin(field):
    data, _ = from_xarray(
        field, target="t2m", inputs=["time"], transforms=[Cyclic("time", period=8.0)]
    )

    # The second timestamp is 2 days, a quarter period, after the first.
    np.testing.assert_allclose(data.X[0], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(data.X[-1], [1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("period", [0.0, -1.0])
def test_cyclic_rejects_a_non_positive_period(period):
    with pytest.raises(ValueError, match="period must be positive"):
        Cyclic("time", period=period)


def test_a_transform_reports_a_missing_column(field):
    with pytest.raises(ValueError, match=r"UnitSphere needs columns \['lon'\]"):
        from_xarray(field, target="t2m", inputs=["lat"], transforms=[UnitSphere()])


def test_a_transform_may_not_overwrite_another_column(field):
    field["lon_sin"] = (("lat", "lon"), np.zeros((2, 3)))

    with pytest.raises(ValueError, match=r"\['lon_sin'\].*already taken"):
        from_xarray(
            field,
            target="t2m",
            inputs=["lon", "lon_sin"],
            transforms=[Cyclic("lon", period=360.0)],
        )


def test_transforms_run_in_order_and_the_spec_names_the_columns(field):
    data, spec = from_xarray(
        field,
        target="t2m",
        inputs=["lat", "lon", "elevation"],
        transforms=[UnitSphere(), Standardise(["elevation"])],
    )

    assert data.X.shape == (12, 4)
    assert spec.columns == ("sphere_x", "sphere_y", "sphere_z", "elevation")
    assert "columns=('sphere_x', 'sphere_y', 'sphere_z', 'elevation')" in repr(spec)


# --- Chunked prediction ------------------------------------------------------


def _toy_predict_fn(inputs):
    """A deterministic stand-in for a posterior: one mean and variance per row."""
    return _diagonal_gaussian(inputs.sum(axis=1), inputs[:, 0] ** 2 + 1.0)


@pytest.fixture
def gappy_grid(field) -> xr.Dataset:
    """The field's inputs, with one NaN covariate cell and lat attrs."""
    grid = field.drop_vars("t2m")
    grid["elevation"][1, 2] = np.nan
    grid["lat"].attrs = {"units": "degrees_north"}
    return grid


@pytest.mark.parametrize("chunk_size", [1, 5, 12, 100])
def test_predict_matches_inputs_for_then_to_xarray(field, gappy_grid, chunk_size):
    _, spec = from_xarray(
        field,
        target="t2m",
        inputs=["time", "lat", "elevation"],
        transforms=[Standardise()],
    )
    test_inputs, test_spec = spec.inputs_for(gappy_grid)
    expected = test_spec.to_xarray(_toy_predict_fn(test_inputs))

    out = spec.predict(_toy_predict_fn, gappy_grid, chunk_size=chunk_size)

    xr.testing.assert_allclose(out, expected)
    assert np.isnan(out["t2m_mean"][:, 1, 2]).all()
    assert out["t2m_mean"].attrs == {"units": "K"}
    assert out["t2m_variance"].attrs == {"units": "(K)^2"}


def test_predict_compiles_once_for_every_chunk(field):
    _, spec = from_xarray(field, target="t2m", inputs=["lat", "lon"])
    traces = []

    def predict_fn(inputs):
        traces.append(inputs.shape)
        return _toy_predict_fn(inputs)

    spec.predict(predict_fn, field.drop_vars("t2m"), chunk_size=5)

    assert traces == [(5, 2)]


def test_predict_keeps_a_dask_grid_lazy(field, gappy_grid):
    pytest.importorskip("dask")
    _, spec = from_xarray(field, target="t2m", inputs=["time", "lat", "elevation"])
    calls = []

    def predict_fn(inputs):
        calls.append(inputs.shape)
        return _toy_predict_fn(inputs)

    lazy = spec.predict(predict_fn, gappy_grid.chunk({"lat": 1}), chunk_size=4)

    assert lazy["t2m_mean"].chunks is not None
    assert calls == []
    eager = spec.predict(_toy_predict_fn, gappy_grid, chunk_size=4)
    xr.testing.assert_allclose(lazy.compute(), eager)


@pytest.mark.parametrize("chunk_size", [0, -3])
def test_predict_rejects_a_non_positive_chunk_size(field, chunk_size):
    _, spec = from_xarray(field, target="t2m", inputs=["lat"])

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        spec.predict(_toy_predict_fn, field, chunk_size=chunk_size)


def test_predict_rejects_a_predict_fn_of_the_wrong_shape(field):
    _, spec = from_xarray(field, target="t2m", inputs=["lat"])

    def one_value(inputs):
        return _diagonal_gaussian(inputs.sum(keepdims=True)[0], jnp.ones(1))

    with pytest.raises(ValueError, match="one value per input"):
        spec.predict(one_value, field, chunk_size=4)
