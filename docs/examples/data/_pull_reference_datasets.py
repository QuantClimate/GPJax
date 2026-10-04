"""Pull the public reference datasets used by the notebooks into checked-in files.

This script is a *reproducibility record*, not part of the test suite or the
notebook execution path. It is run once (manually) to regenerate the files in
this directory; the notebooks then read those local copies so that
``uv run poe docs`` never touches the network.

Usage
-----
    uv run --extra docs python docs/examples/data/_pull_reference_datasets.py

The ``--extra docs`` is only needed for the UCI Auto MPG pull, which uses
``ucimlrepo``; the other three pulls need nothing beyond ``pandas`` and
``requests``.

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
- Colorado precipitation normals (``colorado_precipitation_normals.nc``): the
  1991-2020 annual precipitation normal at each Colorado COOP and first-order
  station, with the station location and elevation. Source: NOAA NCEI U.S.
  Climate Normals, one CSV per station at
  https://www.ncei.noaa.gov/data/normals-annualseasonal/1991-2020/access/ ,
  with Colorado stations found from
  https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt .
  Work of the US Government, so public domain. Cite Palecki et al. (2021),
  U.S. Climate Normals 1991-2020, NOAA NCEI.
- European winter temperature anomalies (``ncep_europe_winter_2019.nc``): daily
  near-surface (sigma 0.995) air temperature anomalies over Europe and the
  north-east Atlantic, 1 January to 1 March 2019, on a 7.5 x 10 degree subset of
  the 2.5 degree grid. The anomaly is the daily value minus the 1991-2020 daily
  long-term mean. Source: NCEP-NCAR Reanalysis 1 (Kalnay et al., 1996),
  https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/Dailies/surface/air.sig995.2019.nc
  and .../ncep.reanalysis.derived/surface/air.sig995.day.ltm.1991-2020.nc .
  Public domain; "NCEP-NCAR Reanalysis 1 data provided by the NOAA PSL,
  Boulder, Colorado, USA". The source files are netCDF4, so this pull needs
  ``uv run --extra docs --with h5netcdf --with h5py python ...``.
"""

from __future__ import annotations

from pathlib import Path
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
# nonstationary_terrain — Colorado annual precipitation normals, 1991-2020.     #
# --------------------------------------------------------------------------- #
NORMALS_URL = "https://www.ncei.noaa.gov/data/normals-annualseasonal/1991-2020/access/"
STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"


def _get(url: str) -> requests.Response:
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 404:
                return resp
            resp.raise_for_status()
            return resp
        except requests.RequestException as err:
            last_err = err
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def pull_colorado_precipitation() -> None:
    print("nonstationary_terrain: Colorado precipitation normals 1991-2020")
    import io
    import re

    import xarray as xr

    stations = _get(STATIONS_URL).text.splitlines()
    colorado = sorted(
        line[:11]
        for line in stations
        if line[38:40] == "CO" and line[:3] in ("USC", "USW")
    )
    listing = _get(NORMALS_URL).text
    available = set(re.findall(r'href="(US[CW]\d+)\.csv"', listing))
    rows = []
    for station in (s for s in colorado if s in available):
        frame = pd.read_csv(io.StringIO(_get(f"{NORMALS_URL}{station}.csv").text))
        if "ANN-PRCP-NORMAL" not in frame:
            continue
        precipitation = pd.to_numeric(frame["ANN-PRCP-NORMAL"], errors="coerce")
        if not precipitation.iloc[0] > 0:
            continue
        rows.append(
            {
                "station": station,
                "name": str(frame["NAME"].iloc[0]).strip(),
                "lat": float(frame["LATITUDE"].iloc[0]),
                "lon": float(frame["LONGITUDE"].iloc[0]),
                "elevation": float(frame["ELEVATION"].iloc[0]),
                # The normals are in inches.
                "precipitation": 25.4 * float(precipitation.iloc[0]),
            }
        )
    table = pd.DataFrame(rows)
    ds = xr.Dataset(
        {
            "precipitation": (
                "station",
                table["precipitation"].to_numpy(),
                {
                    "long_name": "Annual precipitation normal, 1991-2020",
                    "units": "mm",
                },
            ),
            "elevation": (
                "station",
                table["elevation"].to_numpy(),
                {"long_name": "Station elevation", "units": "m"},
            ),
        },
        coords={
            "station": ("station", table["station"].to_numpy()),
            "name": ("station", table["name"].to_numpy()),
            "lat": ("station", table["lat"].to_numpy(), {"units": "degrees_north"}),
            "lon": ("station", table["lon"].to_numpy(), {"units": "degrees_east"}),
        },
        attrs={
            "title": "Colorado annual precipitation normals, 1991-2020",
            "source": "NOAA NCEI U.S. Climate Normals 1991-2020 (annual/seasonal)",
            "references": "Palecki et al. (2021), U.S. Climate Normals 1991-2020, "
            "NOAA National Centers for Environmental Information.",
            "license": "Public domain (work of the US Government).",
            "Conventions": "CF-1.8",
        },
    )
    path = HERE / "colorado_precipitation_normals.nc"
    ds.to_netcdf(path, engine="scipy", format="NETCDF3_64BIT")
    print(f"  wrote {path.name}: {ds.sizes['station']} stations")


# --------------------------------------------------------------------------- #
# spacetime_temperature — daily NCEP-NCAR R1 temperature anomalies, Europe.   #
# --------------------------------------------------------------------------- #
PSL_THREDDS = "https://psl.noaa.gov/thredds/fileServer/Datasets/"


def _download_resumable(url: str, path: Path) -> None:
    """Download a large file. The PSL server drops long transfers, so resume."""
    for _ in range(20):
        done = path.stat().st_size if path.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code == 416:
                    return
                resp.raise_for_status()
                if done and resp.status_code != 206:
                    done = 0
                total = done + int(resp.headers.get("Content-Length", 0))
                with path.open("ab" if done else "wb") as handle:
                    for chunk in resp.iter_content(1 << 20):
                        handle.write(chunk)
            if path.stat().st_size >= total:
                return
        except requests.RequestException:
            time.sleep(3)
    raise RuntimeError(f"Failed to fetch {url}")


def pull_europe_winter_temperature() -> None:
    print("spacetime_temperature: NCEP-NCAR R1 daily temperature anomalies")
    import tempfile

    import numpy as np
    import xarray as xr

    with tempfile.TemporaryDirectory() as tmp:
        daily_path = Path(tmp) / "air.sig995.2019.nc"
        normal_path = Path(tmp) / "air.sig995.day.ltm.1991-2020.nc"
        _download_resumable(
            PSL_THREDDS + "ncep.reanalysis/Dailies/surface/air.sig995.2019.nc",
            daily_path,
        )
        _download_resumable(
            PSL_THREDDS
            + "ncep.reanalysis.derived/surface/air.sig995.day.ltm.1991-2020.nc",
            normal_path,
        )
        daily = xr.open_dataset(daily_path, engine="h5netcdf")["air"]
        normal = xr.open_dataset(normal_path, engine="h5netcdf", decode_times=False)[
            "air"
        ]
        daily = daily.sel(time=slice("2019-01-01", "2019-03-01"))
        day_of_year = np.minimum(daily.time.dt.dayofyear.to_numpy(), 365) - 1
        anomaly = (
            daily - normal.isel(time=xr.DataArray(day_of_year, dims="time")).values
        )
        # Longitudes 340E-40E, as -20 to 40 degrees east.
        anomaly = anomaly.assign_coords(lon=((anomaly.lon + 180) % 360) - 180)
        anomaly = anomaly.sortby("lon").sel(lat=slice(70, 40), lon=slice(-20, 40))
        anomaly = anomaly.isel(lat=slice(None, None, 3), lon=slice(None, None, 4))
        anomaly = anomaly.load()

    ds = xr.Dataset(
        {
            "tas_anomaly": (
                ("time", "lat", "lon"),
                anomaly.to_numpy().astype("float32"),
                {
                    "long_name": "Daily near-surface air temperature anomaly "
                    "(sigma 0.995) against the 1991-2020 daily mean",
                    "units": "K",
                },
            )
        },
        coords={
            "time": anomaly.time.to_numpy(),
            "lat": ("lat", anomaly.lat.to_numpy(), {"units": "degrees_north"}),
            "lon": ("lon", anomaly.lon.to_numpy(), {"units": "degrees_east"}),
        },
        attrs={
            "title": "Daily temperature anomalies over Europe, winter 2019",
            "source": "NCEP-NCAR Reanalysis 1, NOAA PSL",
            "references": "Kalnay et al. (1996), The NCEP/NCAR 40-year "
            "reanalysis project, Bull. Amer. Meteor. Soc. 77, 437-471.",
            "acknowledgment": "NCEP-NCAR Reanalysis 1 data provided by the NOAA "
            "PSL, Boulder, Colorado, USA, from their website at "
            "https://psl.noaa.gov",
            "license": "Public domain.",
            "Conventions": "CF-1.8",
        },
    )
    path = HERE / "ncep_europe_winter_2019.nc"
    ds.to_netcdf(path, engine="scipy", format="NETCDF3_64BIT")
    print(f"  wrote {path.name}: {dict(ds.sizes)}")


if __name__ == "__main__":
    pull_mauna_loa_co2()
    pull_gulf_velocities()
    pull_auto_mpg()
    pull_colorado_precipitation()
    pull_europe_winter_temperature()
    print("\nDone.")
