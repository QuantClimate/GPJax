"""Pull the public reference datasets used by the notebooks into checked-in files.

This script is a *reproducibility record*, not part of the test suite or the
notebook execution path. It is run once (manually) to regenerate the files in
this directory; the notebooks then read those local copies so that
``uv run poe docs`` never touches the network.

Usage
-----
    uv run --extra docs python docs/examples/data/_pull_reference_datasets.py

The ``--extra docs`` is only needed for the UCI Auto MPG pull, which uses
``ucimlrepo``. The NCEP reanalysis pull reads netCDF4, so it also needs
``--with h5netcdf --with h5py``. The other pulls need nothing beyond ``pandas``
and ``requests``.

Data sources
------------
- Mauna Loa CO2 (``mauna_loa_co2.csv``): monthly mean atmospheric CO2 measured
  at the Mauna Loa Observatory, Hawaii. Source:
  https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv
  Provider: NOAA Global Monitoring Laboratory (Lan, Tans & Thoning). Work of the
  US Government, so public domain / not subject to copyright; NOAA request
  citation of the dataset. Vendored verbatim, including the ``#`` comment header
  that carries NOAA's own citation and contact details.
- Gulf of Mexico velocities (``gulfdata_train.csv``, ``gulfdata_test.csv``):
  drifter observations (train) and a gridded ocean-current field (test) over the
  Gulf of Mexico. Source:
  https://raw.githubusercontent.com/JaxGaussianProcesses/static/main/data/gulfdata_{train,test}.csv
  These live in the JaxGaussianProcesses organisation's own ``static`` repository
  and are redistributed here under that repository's terms. Vendored verbatim.
- UCI Auto MPG (``auto_mpg.csv``): fuel consumption for 398 cars. Source:
  https://archive.ics.uci.edu/dataset/9/auto-mpg (fetched via ``ucimlrepo``).
  Licence: CC BY 4.0. The features and the target are concatenated into a single
  frame so the notebook can split them back out without ``ucimlrepo``.
- NCEP-NCAR Reanalysis 1 temperature anomaly (``ncep_air_anomaly_2024.nc``):
  the 2024 annual-mean anomaly of near-surface (0.995 sigma) air temperature,
  relative to the 1991-2020 mean of each grid cell, on the native 2.5-degree
  grid. A reanalysis has a value in every cell, so the field is complete.
  Source: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html (monthly
  means, ``air.mon.mean.nc``). Provider: NOAA Physical Sciences Laboratory
  (Kalnay et al., 1996, https://doi.org/10.1175/1520-0477(1996)077<0437:TNYRP>2.0.CO;2).
  Work of the US Government, so public domain; PSL ask for the acknowledgement
  "NCEP-NCAR Reanalysis 1 data provided by the NOAA PSL, Boulder, Colorado, USA,
  from their website at https://psl.noaa.gov". Written as netCDF3, which xarray
  reads with SciPy, so the notebook needs no netCDF4 library.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

import pandas as pd
import requests

HERE = Path(__file__).parent


def _download(url: str, name: str) -> None:
    """Save one URL verbatim, with basic retries."""
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            path = HERE / name
            path.write_bytes(resp.content)
            print(f"  wrote {name}: {len(resp.content)} bytes")
            return
        except requests.RequestException as err:
            last_err = err
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _save(df: pd.DataFrame, name: str) -> None:
    path = HERE / name
    df.to_csv(path, index=False)
    print(f"  wrote {name}: {len(df)} rows, {list(df.columns)}")


# --------------------------------------------------------------------------- #
# intro_to_kernels, state_space_gps — Mauna Loa monthly mean CO2 record.       #
# --------------------------------------------------------------------------- #
def pull_mauna_loa_co2() -> None:
    print("intro_to_kernels / state_space_gps: Mauna Loa monthly mean CO2")
    _download(
        "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
        "mauna_loa_co2.csv",
    )


# --------------------------------------------------------------------------- #
# oceanmodelling — Gulf of Mexico drifters (train) and current field (test).   #
# --------------------------------------------------------------------------- #
def pull_gulf_velocities() -> None:
    print("oceanmodelling: Gulf of Mexico drifter and ocean-current velocities")
    base = "https://raw.githubusercontent.com/JaxGaussianProcesses/static/main/data/"
    for name in ("gulfdata_train.csv", "gulfdata_test.csv"):
        _download(base + name, name)


# --------------------------------------------------------------------------- #
# oak — UCI Auto MPG, features and target concatenated into one frame.        #
# --------------------------------------------------------------------------- #
def pull_auto_mpg() -> None:
    print("oak: UCI Auto MPG (id=9)")
    from ucimlrepo import fetch_ucirepo

    auto_mpg = fetch_ucirepo(id=9)
    features = auto_mpg.data.features
    targets = auto_mpg.data.targets
    # Column order is load-bearing: the notebook reconstructs ``features`` by
    # selecting every column except the target, and reports them in this order.
    _save(pd.concat([features, targets], axis=1), "auto_mpg.csv")


# --------------------------------------------------------------------------- #
# xarray_workflow — NCEP-NCAR Reanalysis 1 temperature anomaly for 2024.       #
# --------------------------------------------------------------------------- #
NCEP_URL = (
    "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/"
    "Monthlies/surface/air.mon.mean.nc"
)


def _download_resumable(url: str, path: Path, attempts: int = 8) -> None:
    """Download ``url`` to ``path``, resuming when the server cuts it short."""
    for _ in range(attempts):
        start = path.stat().st_size if path.exists() else 0
        headers = {"User-Agent": "gpjax-docs", "Range": f"bytes={start}-"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code == 416:  # nothing left to fetch
                    return
                resp.raise_for_status()
                total = start + int(resp.headers["Content-Length"])
                mode = "ab" if resp.status_code == 206 else "wb"
                with path.open(mode) as file:
                    for chunk in resp.iter_content(1 << 20):
                        file.write(chunk)
            if path.stat().st_size >= total:
                return
        except requests.RequestException as err:
            print(f"  retrying after: {err}")
        time.sleep(2)
    raise RuntimeError(f"Failed to fetch {url} in {attempts} attempts")


def pull_ncep_anomaly(nc_path: Path | None = None) -> None:
    """Save the 2024 NCEP anomaly; pass ``nc_path`` to reuse a download."""
    print("xarray_workflow: NCEP-NCAR Reanalysis 1 temperature anomaly, 2024")
    import xarray as xr

    with tempfile.TemporaryDirectory() as tmp:
        if nc_path is None:
            nc_path = Path(tmp) / "air.mon.mean.nc"
            _download_resumable(NCEP_URL, nc_path)
        with xr.open_dataset(nc_path, engine="h5netcdf") as monthly:
            annual = monthly["air"].resample(time="YS").mean().load()
    climatology = annual.sel(time=slice("1991", "2020")).mean("time")
    anomaly = annual.sel(time="2024").squeeze("time", drop=True) - climatology
    # Longitude from 0..357.5 to -180..177.5, so maps are centred on Greenwich.
    anomaly = anomaly.assign_coords(lon=(anomaly["lon"] + 180.0) % 360.0 - 180.0)
    anomaly = anomaly.sortby(["lat", "lon"]).astype("float32")
    anomaly.attrs = {
        "long_name": "Near-surface air temperature anomaly",
        "units": "K",
        "cell_methods": "time: mean (2024, anomaly relative to 1991-2020)",
    }
    anomaly["lat"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    anomaly["lon"].attrs = {"standard_name": "longitude", "units": "degrees_east"}
    dataset = anomaly.to_dataset(name="tas_anomaly")
    dataset.attrs = {
        "title": "2024 annual-mean near-surface air temperature anomaly",
        "source": "NCEP-NCAR Reanalysis 1, monthly air.sig995 (air.mon.mean.nc)",
        "references": "Kalnay et al. (1996), doi:10.1175/1520-0477(1996)077<0437:TNYRP>2.0.CO;2",
        "acknowledgement": (
            "NCEP-NCAR Reanalysis 1 data provided by the NOAA PSL, Boulder, "
            "Colorado, USA, from their website at https://psl.noaa.gov"
        ),
        "history": "Annual means of the monthly means; 2024 minus the 1991-2020 mean.",
        "Conventions": "CF-1.8",
    }
    path = HERE / "ncep_air_anomaly_2024.nc"
    dataset.to_netcdf(path, engine="scipy", format="NETCDF3_64BIT")
    print(f"  wrote {path.name}: {dict(dataset.sizes)}")


if __name__ == "__main__":
    pull_mauna_loa_co2()
    pull_gulf_velocities()
    pull_auto_mpg()
    pull_ncep_anomaly()
    print("\nDone.")
