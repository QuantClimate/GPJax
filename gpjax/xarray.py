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
r"""Bridge between labelled xarray data and GPJax.

A GP in GPJax takes a :class:`~gpjax.dataset.Dataset` of flat ``(N, D)`` inputs
and ``(N, 1)`` outputs. Gridded data arrives as labelled xarray objects instead.
This module converts between the two at the edges of a workflow and nowhere
else:

.. code-block:: python

    data, spec = from_xarray(ds, target="t2m", inputs=["lat", "lon", "elevation"])
    posterior = model.condition(data)
    test_inputs, test_spec = spec.inputs_for(grid)
    out = test_spec.to_xarray(posterior(test_inputs))

``data`` is an ordinary :class:`~gpjax.dataset.Dataset`. Everything needed to
rebuild the grid lives on the :class:`GridSpec`, which never enters ``fit``,
``condition`` or a traced computation.

Input transforms (:class:`Standardise`, :class:`UnitSphere`, :class:`Cyclic`)
turn the named inputs into the columns of ``X``. The spec records them, fitted on
the training data, and applies the same ones to every new grid. For large grids,
:meth:`GridSpec.predict` gives the predictive mean and variance in chunks of
bounded size, and keeps a Dask-backed grid lazy.

Requires the optional ``xarray`` extra: ``pip install "gpjax[xarray]"``.
"""

import abc
from dataclasses import (
    dataclass,
    field,
    replace,
)

import beartype.typing as tp
import jax
import jax.numpy as jnp
from jaxtyping import Float
import lineax as lx
import numpy as np

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.typing import (
    Array,
    KeyArray,
)

try:
    import xarray as xr
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "gpjax.xarray requires xarray; install it with pip install 'gpjax[xarray]'"
    ) from error

Columns = dict[str, np.ndarray]


class InputTransform(abc.ABC):
    r"""A map from named input columns to the named columns of ``X``.

    A transform receives the columns in order, keyed by name, and returns a new
    ordered mapping. :func:`from_xarray` calls :meth:`fit` on the training
    columns, and the :class:`GridSpec` then applies the fitted transform to the
    training data and to every new grid, so both see the same encoding.
    Datetime inputs are already float days since their time origin.
    """

    def fit(self, columns: Columns) -> "InputTransform":
        r"""Return this transform fitted to the training ``columns``.

        The default returns the transform unchanged; override it when the
        transform learns a state from the training data.
        """
        return self

    @abc.abstractmethod
    def __call__(self, columns: Columns) -> Columns:
        r"""Transform ``columns`` and return the new ordered columns."""


@dataclass(frozen=True)
class Standardise(InputTransform):
    r"""Centre and scale inputs to zero mean and unit standard deviation.

    The mean and standard deviation come from the training cells. A new grid
    is scaled with the training values, so one lengthscale means the same
    distance in training and in prediction.

    Attributes:
        names: The columns to standardise. ``None`` standardises every column
            present when the transform runs.
        loc: The fitted training mean of each column; ``None`` before fitting.
        scale: The fitted training standard deviation of each column; ``None``
            before fitting.
    """

    names: tp.Optional[tp.Sequence[str]] = None
    loc: tp.Optional[dict[str, float]] = field(default=None, compare=False)
    scale: tp.Optional[dict[str, float]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.names, str):
            raise TypeError(
                f"names must be a list of names, e.g. [{self.names!r}], not a string"
            )
        if self.names is not None:
            object.__setattr__(self, "names", tuple(self.names))

    def fit(self, columns: Columns) -> "Standardise":
        r"""Record the mean and standard deviation of each named column.

        Raises:
            ValueError: If a name is not a column, or a column is constant.
        """
        names = self.names if self.names is not None else tuple(columns)
        _require(columns, names, "Standardise")
        loc = {name: float(np.mean(columns[name])) for name in names}
        scale = {name: float(np.std(columns[name])) for name in names}
        constant = [name for name in names if not scale[name] > 0.0]
        if constant:
            raise ValueError(
                f"cannot standardise {constant}: the column is constant over the "
                "training cells"
            )
        return replace(self, names=names, loc=loc, scale=scale)

    def __call__(self, columns: Columns) -> Columns:
        r"""Apply the training mean and standard deviation."""
        if self.loc is None or self.scale is None:
            raise RuntimeError("Standardise must be fitted before it is applied")
        _require(columns, tuple(self.loc), "Standardise")
        return {
            name: (values - self.loc[name]) / self.scale[name]
            if name in self.loc
            else values
            for name, values in columns.items()
        }


@dataclass(frozen=True)
class UnitSphere(InputTransform):
    r"""Map latitude and longitude in degrees to a point on the unit sphere.

    The two columns are replaced, at the position of ``lat``, by
    ``{name}_x = cos(lat) cos(lon)``, ``{name}_y = cos(lat) sin(lon)`` and
    ``{name}_z = sin(lat)``. A stationary kernel on these columns uses the
    chord distance through the sphere, which is valid on the whole globe: it
    has no seam at the antimeridian, no singularity at the poles, and it does
    not stretch distances at high latitudes. A lengthscale is then in Earth
    radii (one radius is about 6371 km).

    Attributes:
        lat: Name of the latitude column, in degrees north.
        lon: Name of the longitude column, in degrees east.
        name: Prefix of the three new columns.
    """

    lat: str = "lat"
    lon: str = "lon"
    name: str = "sphere"

    def __call__(self, columns: Columns) -> Columns:
        r"""Replace ``lat`` and ``lon`` with the three unit-sphere columns.

        Raises:
            ValueError: If a name is not a column, a new column name is already
                taken, or a latitude is outside [-90, 90] degrees.
        """
        _require(columns, (self.lat, self.lon), "UnitSphere")
        lat_degrees = columns[self.lat]
        if np.any(np.abs(lat_degrees[np.isfinite(lat_degrees)]) > 90.0):
            raise ValueError(
                f"{self.lat!r} has values outside [-90, 90]; UnitSphere expects "
                "latitude in degrees north"
            )
        lat = np.deg2rad(lat_degrees)
        lon = np.deg2rad(columns[self.lon])
        sphere = {
            f"{self.name}_x": np.cos(lat) * np.cos(lon),
            f"{self.name}_y": np.cos(lat) * np.sin(lon),
            f"{self.name}_z": np.sin(lat),
        }
        return _substitute(columns, (self.lat, self.lon), sphere)


@dataclass(frozen=True)
class Cyclic(InputTransform):
    r"""Encode a periodic input as a point on a circle.

    The column is replaced by ``{name}_sin`` and ``{name}_cos`` of
    ``2 pi value / period``, so values one period apart get the same encoding.
    Datetime inputs are days since their time origin: ``Cyclic("time", 365.25)``
    encodes the seasonal cycle, and ``Cyclic("lon", 360.0)`` removes the seam
    in longitude.

    Attributes:
        name: The column to encode.
        period: The period, in the units of the column.
    """

    name: str
    period: float

    def __post_init__(self) -> None:
        if not self.period > 0.0:
            raise ValueError(f"period must be positive, not {self.period}")

    def __call__(self, columns: Columns) -> Columns:
        r"""Replace the column with its sine and cosine.

        Raises:
            ValueError: If the name is not a column, or a new column name is
                already taken.
        """
        _require(columns, (self.name,), "Cyclic")
        phase = 2.0 * np.pi * columns[self.name] / self.period
        circle = {f"{self.name}_sin": np.sin(phase), f"{self.name}_cos": np.cos(phase)}
        return _substitute(columns, (self.name,), circle)


@dataclass(frozen=True, repr=False)
class GridSpec:
    r"""The labelled grid behind a flattened :class:`~gpjax.dataset.Dataset`.

    Row ``i`` of the flattened data is the ``i``-th kept cell of the grid in C
    order over ``dims``. That correspondence, and the encoding of the inputs as
    the columns of ``X``, is all this object records, and all
    :meth:`inputs_for`, :meth:`predict` and :meth:`to_xarray` need.

    Attributes:
        target: Name of the modelled variable.
        target_attrs: The target's attributes (units, long_name, ...), carried
            onto predictions.
        inputs: Input names, in the order they are read from the data.
        dims: The grid's dims, in the target's order.
        coords: The grid's coordinates, used to rebuild labelled output.
        time_origins: For each datetime input, the timestamp encoded as day 0.
        mask: Boolean array over the full grid; ``True`` marks a cell that has a
            row in the flattened data.
        n_dropped: Number of grid cells dropped for containing NaN.
        transforms: The input transforms, fitted on the training cells, that
            turn ``inputs`` into the columns of ``X``.
    """

    target: str
    target_attrs: dict[str, tp.Any]
    inputs: tuple[str, ...]
    dims: tuple[str, ...]
    coords: tp.Mapping[tp.Hashable, xr.DataArray]
    time_origins: dict[str, np.datetime64]
    mask: np.ndarray
    n_dropped: int
    transforms: tuple[InputTransform, ...] = ()

    @property
    def n_kept(self) -> int:
        r"""Number of grid cells with a row in the flattened data."""
        return int(self.mask.sum())

    @property
    def columns(self) -> tuple[str, ...]:
        r"""Names of the columns of ``X``, after the input transforms."""
        empty = {name: np.empty(0) for name in self.inputs}
        return tuple(_apply(self.transforms, empty))

    def __repr__(self) -> str:
        r"""Summarise the grid without printing its coordinates or mask."""
        grid = dict(zip(self.dims, self.mask.shape, strict=True))
        columns = "" if self.columns == self.inputs else f", columns={self.columns}"
        return (
            f"GridSpec(target={self.target!r}, inputs={self.inputs}{columns}, "
            f"grid={grid}, kept={self.n_kept}, dropped={self.n_dropped})"
        )

    def inputs_for(
        self, obj: tp.Union[xr.Dataset, xr.DataArray]
    ) -> tuple[Float[Array, "M D"], "GridSpec"]:
        r"""Build prediction inputs on a new grid, encoded as in training.

        The new grid is the broadcast of this spec's inputs as found in ``obj``;
        no target is needed. Its dims follow the training grid's order, with any
        new dims after them. Datetime inputs reuse the training time origins, so
        a date maps to the same number here as it did in training, and the
        fitted input transforms are applied as they were in training. Cells
        where an input is NaN get no row, and come back as NaN from
        :meth:`to_xarray`.

        This builds every row at once. For a large grid, :meth:`predict` gives
        the mean and variance in chunks instead.

        Args:
            obj: Labelled data holding every input named by this spec.

        Returns:
            The ``(M, D)`` prediction inputs and the ``GridSpec`` of the new grid,
            which carries this spec's target name, attributes and transforms.

        Raises:
            ValueError: If an input is missing from ``obj``.
            TypeError: If an input is non-numeric, or is a datetime now but was
                not in training (or the reverse).
        """
        dataset, variables, dims = self._grid_of(obj)
        grid = xr.broadcast(*variables)[0].transpose(*dims)
        input_matrix = _input_matrix(variables, grid, dims, self.time_origins)
        keep = np.isfinite(input_matrix).all(axis=1)
        test_spec = replace(
            self,
            target_attrs=dict(self.target_attrs),
            dims=dims,
            coords=_grid_coords(dataset, dims),
            mask=keep.reshape(grid.shape),
            n_dropped=int(keep.size - keep.sum()),
        )
        return jnp.asarray(self._encode(input_matrix[keep])), test_spec

    def predict(
        self,
        predict_fn: tp.Callable[[Float[Array, "M D"]], tp.Any],
        obj: tp.Union[xr.Dataset, xr.DataArray],
        *,
        chunk_size: int = 4096,
    ) -> xr.Dataset:
        r"""Predict the mean and variance on a new grid, in chunks.

        The grid is built from ``obj`` as in :meth:`inputs_for`, but the cells
        go to ``predict_fn`` at most ``chunk_size`` at a time, so the memory use
        does not grow with the size of the grid. The last chunk is padded to
        ``chunk_size``, so ``predict_fn`` is compiled once, with ``jax.jit``.

        If ``obj`` holds Dask arrays, the result is lazy: each Dask block is
        predicted when it is computed, for example by ``.compute()`` or
        ``to_netcdf``. Chunk ``obj`` along the grid's dims to control the size
        of a block.

        Args:
            predict_fn: Maps ``(M, D)`` inputs to a distribution with ``mean``
                and ``variance`` of shape ``(M,)``, for example
                ``lambda x: posterior(x, covariance="diagonal")``. Use the
                diagonal covariance: a dense one costs ``chunk_size**2`` memory
                and gives the same marginals.
            obj: Labelled data holding every input named by this spec.
            chunk_size: The number of cells given to ``predict_fn`` at once.

        Returns:
            An ``xr.Dataset`` holding ``{target}_mean`` and ``{target}_variance``
            over the grid. Cells where an input is NaN are NaN.

        Raises:
            ValueError: If ``chunk_size`` is not positive, an input is missing
                from ``obj``, or ``predict_fn`` returns the wrong shape.
            TypeError: If an input is non-numeric, or is a datetime now but was
                not in training (or the reverse).
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, not {chunk_size}")
        dataset, variables, dims = self._grid_of(obj)
        grid_inputs = [var.transpose(*dims) for var in xr.broadcast(*variables)]
        moments = jax.jit(lambda inputs: _moments(predict_fn(inputs)))

        def predict_block(*blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            columns = [
                _encode(name, block.ravel(), self.time_origins)
                for name, block in zip(self.inputs, blocks, strict=True)
            ]
            input_matrix = np.stack(columns, axis=1)
            keep = np.isfinite(input_matrix).all(axis=1)
            mean = np.full(keep.size, np.nan)
            variance = np.full(keep.size, np.nan)
            if keep.any():
                inputs = self._encode(input_matrix[keep])
                mean[keep], variance[keep] = _in_chunks(moments, inputs, chunk_size)
            shape = blocks[0].shape
            return mean.reshape(shape), variance.reshape(shape)

        mean, variance = xr.apply_ufunc(
            predict_block,
            *grid_inputs,
            output_core_dims=[[], []],
            dask="parallelized",
            output_dtypes=[float, float],
        )
        prediction = self._moments_dataset(mean, variance)
        return prediction.assign_coords(_grid_coords(dataset, dims))

    def to_xarray(
        self,
        dist: GaussianDistribution,
        *,
        num_samples: tp.Optional[int] = None,
        key: tp.Optional[KeyArray] = None,
    ) -> xr.Dataset:
        r"""Map a predictive distribution back onto the labelled grid.

        Args:
            dist: A distribution over exactly the cells this spec kept, in
                flattened order -- e.g. ``posterior(test_inputs)`` for inputs
                built by :meth:`inputs_for`.
            num_samples: If given, return this many joint draws from ``dist``
                instead of its mean and variance.
            key: PRNG key for the draws; required with ``num_samples``.

        Returns:
            By default, an ``xr.Dataset`` holding ``{target}_mean`` and
            ``{target}_variance`` over the grid. With ``num_samples``, one
            variable ``{target}`` over ``("sample", *dims)``. Dropped cells are
            NaN either way.

        Raises:
            ValueError: If ``dist`` does not match the number of kept cells, if
                ``num_samples`` is given without ``key``, or if samples are
                requested from a distribution holding only marginal variances.
        """
        n_points = dist.mean.shape[0]
        if n_points != self.n_kept:
            raise ValueError(
                f"distribution has {n_points} points but this spec expects "
                f"{self.n_kept}; was it built from a different grid?"
            )
        if num_samples is not None:
            return self._samples(dist, num_samples, key)
        return self._moments_dataset(
            self._scatter(dist.mean, {}), self._scatter(dist.variance, {})
        )

    def _grid_of(
        self, obj: tp.Union[xr.Dataset, xr.DataArray]
    ) -> tuple[xr.Dataset, list[xr.DataArray], tuple[str, ...]]:
        r"""The inputs of this spec in ``obj``, and the dims of their grid.

        The dims follow the training grid's order, with any new dims after them.
        """
        dataset = _as_dataset(obj)
        variables = _resolve(dataset, self.inputs)
        input_dims = dict.fromkeys(dim for var in variables for dim in var.dims)
        dims = tuple(
            [dim for dim in self.dims if dim in input_dims]
            + [dim for dim in input_dims if dim not in self.dims]
        )
        return dataset, variables, dims

    def _encode(self, input_matrix: np.ndarray) -> np.ndarray:
        r"""Apply the fitted transforms to rows of raw inputs."""
        columns = dict(zip(self.inputs, input_matrix.T, strict=True))
        return np.stack(list(_apply(self.transforms, columns).values()), axis=1)

    def _moments_dataset(
        self, mean: xr.DataArray, variance: xr.DataArray
    ) -> xr.Dataset:
        r"""Name ``mean`` and ``variance`` after the target, with its attributes."""
        variance_attrs = dict(self.target_attrs)
        if "units" in variance_attrs:
            variance_attrs["units"] = f"({variance_attrs['units']})^2"
        mean, variance = mean.copy(deep=False), variance.copy(deep=False)
        mean.attrs, variance.attrs = dict(self.target_attrs), variance_attrs
        return xr.Dataset(
            {f"{self.target}_mean": mean, f"{self.target}_variance": variance}
        )

    def _samples(
        self,
        dist: GaussianDistribution,
        num_samples: int,
        key: tp.Optional[KeyArray],
    ) -> xr.Dataset:
        r"""Joint draws from ``dist``, scattered onto the grid per sample."""
        if key is None:
            raise ValueError("num_samples requires a PRNG key, e.g. key=jr.key(0)")
        if lx.is_diagonal(dist.scale):
            # Independent draws would ignore the spatial correlation, so any
            # aggregate over cells (a region, a portfolio) would be overconfident.
            raise ValueError(
                "this distribution holds only marginal variances, so it cannot "
                'give joint samples; predict with covariance="dense" to keep '
                "the correlation between cells"
            )
        attrs = {
            **self.target_attrs,
            "description": "joint posterior predictive draws",
        }
        draws = self._scatter(dist.sample(key, (num_samples,)), attrs, ("sample",))
        return xr.Dataset({self.target: draws})

    def _scatter(
        self,
        values: Float[Array, "... N"],
        attrs: dict[str, tp.Any],
        leading_dims: tuple[str, ...] = (),
    ) -> xr.DataArray:
        r"""Place per-kept-cell ``values`` into a NaN-filled labelled grid.

        Any leading axes of ``values`` (e.g. samples) are kept as
        ``leading_dims`` in front of the grid's dims.
        """
        values = np.asarray(values)
        leading_shape = values.shape[:-1]
        grid = np.full((*leading_shape, *self.mask.shape), np.nan)
        grid[..., self.mask] = values
        return xr.DataArray(
            grid, coords=self.coords, dims=(*leading_dims, *self.dims), attrs=attrs
        )


def from_xarray(
    obj: tp.Union[xr.Dataset, xr.DataArray],
    target: str,
    inputs: tp.Sequence[str],
    *,
    transforms: tp.Sequence[InputTransform] = (),
    dropna: bool = True,
) -> tuple[Dataset, GridSpec]:
    r"""Flatten labelled xarray data into a :class:`~gpjax.dataset.Dataset`.

    Every input is broadcast onto the target's grid, so an input on fewer dims
    (e.g. ``elevation(lat, lon)`` for a ``(time, lat, lon)`` target) repeats
    along the rest. Datetime inputs become float days since their earliest
    timestamp. The ``transforms`` then run in order, fitted on the kept cells,
    to give the columns of ``X``; :attr:`GridSpec.columns` names them.

    Args:
        obj: The labelled data. A ``DataArray`` must be named, and that name is
            the target.
        target: Name of the data variable to model. Its dims define the grid.
        inputs: Coordinates and/or data variables to use as inputs, in the order
            of the columns of ``X`` before any transforms.
        transforms: Input transforms, such as :class:`Standardise`,
            :class:`UnitSphere` or :class:`Cyclic`, applied in order.
        dropna: Drop grid cells where the target or any input is NaN. When
            ``False``, such cells raise instead.

    Returns:
        The flattened ``Dataset`` and the ``GridSpec`` needed to map
        predictions back onto the grid.

    Raises:
        ValueError: If a name is missing, ``inputs`` is empty, repeats a name or
            includes the target, an input has a dim the target lacks, a
            ``DataArray`` is unnamed, no cells survive NaN handling (or any
            NaN is present when ``dropna=False``), or a transform rejects its
            columns.
        TypeError: If ``inputs`` is a single string, or the target or an input
            is not numeric (inputs may also be ``datetime64``).
    """
    if isinstance(inputs, str):
        raise TypeError(
            f"inputs must be a list of names, e.g. [{inputs!r}], not a string"
        )
    if not inputs:
        raise ValueError("inputs must name at least one coordinate or variable")
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"inputs {list(inputs)} contain duplicate names")
    if target in inputs:
        raise ValueError(f"the target {target!r} cannot also be an input")
    dataset = _as_dataset(obj)
    target_values, *variables = _resolve(dataset, [target, *inputs])
    dims = tuple(target_values.dims)
    time_origins = {
        variable.name: _earliest(variable.values)
        for variable in variables
        if np.issubdtype(variable.dtype, np.datetime64)
    }
    input_matrix = _input_matrix(variables, target_values, dims, time_origins)
    outputs = _as_float(target_values.values.reshape(-1, 1), f"target {target!r}")

    keep = np.isfinite(input_matrix).all(axis=1) & np.isfinite(outputs[:, 0])
    n_dropped = int(keep.size - keep.sum())
    if n_dropped and not dropna:
        raise ValueError(
            f"{n_dropped} cell(s) contain NaN in the target or an input; pass "
            "dropna=True to drop them"
        )
    if not keep.any():
        raise ValueError("no cells are left once NaN cells are dropped")

    columns = dict(zip(inputs, input_matrix[keep].T, strict=True))
    fitted = []
    for transform in transforms:
        fitted.append(transform.fit(columns))
        columns = fitted[-1](columns)

    data = Dataset(
        X=jnp.asarray(np.stack(list(columns.values()), axis=1)),
        y=jnp.asarray(outputs[keep]),
    )
    spec = GridSpec(
        target=target,
        target_attrs=dict(target_values.attrs),
        inputs=tuple(inputs),
        dims=dims,
        coords=target_values.coords,
        time_origins=time_origins,
        mask=keep.reshape(target_values.shape),
        n_dropped=n_dropped,
        transforms=tuple(fitted),
    )
    return data, spec


def _apply(transforms: tp.Sequence[InputTransform], columns: Columns) -> Columns:
    r"""Run fitted ``transforms`` over ``columns`` in order."""
    for transform in transforms:
        columns = transform(columns)
    return columns


def _require(columns: Columns, names: tp.Sequence[str], transform: str) -> None:
    r"""Raise if a name that ``transform`` reads is not a column."""
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(
            f"{transform} needs columns {missing}, but the columns are {list(columns)}"
        )


def _substitute(columns: Columns, replaced: tuple[str, ...], new: Columns) -> Columns:
    r"""Swap the ``replaced`` columns for ``new``, at the first one's position."""
    taken = [name for name in new if name in columns and name not in replaced]
    if taken:
        raise ValueError(f"cannot add columns {taken}: the names are already taken")
    out = {}
    for name, values in columns.items():
        if name == replaced[0]:
            out.update(new)
        elif name not in replaced:
            out[name] = values
    return out


def _moments(dist: tp.Any) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    r"""The mean and marginal variance of a predictive distribution."""
    return dist.mean, dist.variance


def _in_chunks(
    moments: tp.Callable[[Float[Array, "M D"]], tuple[Array, Array]],
    inputs: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Evaluate ``moments`` over ``inputs``, ``chunk_size`` rows at a time.

    The last chunk is padded with copies of its first row, so every call has
    the same shape and a jitted ``moments`` compiles once.
    """
    n_rows = inputs.shape[0]
    means, variances = [], []
    for start in range(0, n_rows, chunk_size):
        chunk = inputs[start : start + chunk_size]
        n_valid = chunk.shape[0]
        if n_valid < chunk_size:
            padding = np.repeat(chunk[:1], chunk_size - n_valid, axis=0)
            chunk = np.concatenate([chunk, padding])
        mean, variance = moments(jnp.asarray(chunk))
        mean = np.asarray(mean).reshape(-1)
        variance = np.asarray(variance).reshape(-1)
        if mean.shape != (chunk_size,) or variance.shape != (chunk_size,):
            raise ValueError(
                f"predict_fn returned a mean of shape {mean.shape} and a variance "
                f"of shape {variance.shape} for {chunk_size} inputs; it must "
                "return one value per input"
            )
        means.append(mean[:n_valid])
        variances.append(variance[:n_valid])
    return np.concatenate(means), np.concatenate(variances)


def _as_dataset(obj: tp.Union[xr.Dataset, xr.DataArray]) -> xr.Dataset:
    r"""Promote a named ``DataArray`` to a one-variable ``Dataset``."""
    if isinstance(obj, xr.Dataset):
        return obj
    if obj.name is None:
        raise ValueError(
            "a DataArray must have a name to be used as the target; set one with "
            ".rename('name') or pass an xr.Dataset"
        )
    return obj.to_dataset()


def _resolve(dataset: xr.Dataset, names: tp.Sequence[str]) -> list[xr.DataArray]:
    r"""Look up coordinates or data variables by name, reporting every miss."""
    missing = [name for name in names if name not in dataset.variables]
    if missing:
        raise ValueError(
            f"{missing} not found; available coordinates and variables are "
            f"{sorted(map(str, dataset.variables))}"
        )
    return [dataset[name] for name in names]


def _grid_coords(
    dataset: xr.Dataset, dims: tuple[str, ...]
) -> dict[tp.Hashable, xr.DataArray]:
    r"""Every coordinate of ``dataset`` that lives on the grid's dims."""
    return {
        name: coord
        for name, coord in dataset.coords.items()
        if set(coord.dims) <= set(dims)
    }


def _earliest(timestamps: np.ndarray) -> np.datetime64:
    r"""The earliest non-NaT timestamp, or NaT if there is none."""
    valid = timestamps[~np.isnat(timestamps)]
    return valid.min() if valid.size else np.datetime64("NaT")


def _input_matrix(
    variables: list[xr.DataArray],
    grid: xr.DataArray,
    dims: tuple[str, ...],
    time_origins: dict[str, np.datetime64],
) -> np.ndarray:
    r"""Stack ``variables`` into an ``(N, D)`` float matrix over ``grid``."""
    columns = [
        _encode(variable.name, _column(variable, grid, dims), time_origins)
        for variable in variables
    ]
    return np.stack(columns, axis=1)


def _encode(
    name: tp.Hashable, column: np.ndarray, time_origins: dict[str, np.datetime64]
) -> np.ndarray:
    r"""Encode one flat input column as floats.

    A datetime input must have a time origin, and an input with a time origin
    must still be a datetime: otherwise the same number would mean different
    things in training and prediction.
    """
    is_datetime = np.issubdtype(column.dtype, np.datetime64)
    if is_datetime != (name in time_origins):
        then, now = ("a", "not a") if name in time_origins else ("not a", "a")
        raise TypeError(
            f"input {name!r} was {then} datetime when the spec was built but "
            f"is {now} datetime now; encode it the same way in both"
        )
    if is_datetime:
        days = (column - time_origins[name]) / np.timedelta64(1, "D")
        return np.where(np.isnat(column), np.nan, days)
    return _as_float(column, f"input {name!r}")


def _column(
    variable: xr.DataArray, grid: xr.DataArray, dims: tuple[str, ...]
) -> np.ndarray:
    r"""Broadcast ``variable`` onto ``grid`` and flatten it in C order over ``dims``.

    A variable on fewer dims than the grid repeats along the missing ones. A
    variable on a dim the grid lacks has no cell to land in, so it is rejected.
    """
    extra_dims = [dim for dim in variable.dims if dim not in dims]
    if extra_dims:
        raise ValueError(
            f"input {variable.name!r} has dims {extra_dims} that the grid "
            f"{dims} does not have"
        )
    return xr.broadcast(grid, variable)[1].transpose(*dims).values.ravel()


def _as_float(values: np.ndarray, description: str) -> np.ndarray:
    r"""Cast numeric or boolean ``values`` to float; reject anything else."""
    if np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
        return values.astype(float)
    raise TypeError(
        f"{description} has dtype {values.dtype}, which is not numeric; convert "
        "it before calling from_xarray"
    )


__all__ = [
    "Cyclic",
    "GridSpec",
    "InputTransform",
    "Standardise",
    "UnitSphere",
    "from_xarray",
]
