"""Hourly direct and diffuse irradiance for a point from the NREL NSRDB typical meteorological year (TMY), read straight from the public AWS copy by HTTP range requests.

NSRDB TMY: 8,760 hourly values of DNI and DHI (W/m2) at 4 km resolution, assembled from the most typical month of a 20-plus-year satellite record. The file is 850 GB; only the nearest site's column is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ept import CACHE

URL = "https://nrel-pds-nsrdb.s3.amazonaws.com/GOES/tmy/v3.0.6/nsrdb_tmy-2018.h5"


def fetch_tmy(lon, lat, tag):
    """Nearest-site hourly TMY (cached): DataFrame indexed by the file's time stamps with dni, dhi, ghi, solar_zenith_angle, temperature; plus the site metadata."""
    p = CACHE / "solar" / f"tmy_{tag}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    import fsspec
    import h5py
    fs = fsspec.filesystem("http", block_size=8 * 2 ** 20)
    f = h5py.File(fs.open(URL, "rb", cache_type="readahead"), "r")
    co = f["coordinates"][:]
    i = int(np.argmin((co[:, 0] - lat) ** 2 + ((co[:, 1] - lon) * np.cos(np.radians(lat))) ** 2))
    ti = pd.to_datetime([t.decode() if isinstance(t, bytes) else str(t) for t in f["time_index"][:]])
    cols = {k: f[k][:, i].astype("float64") for k in ("dni", "dhi", "ghi", "solar_zenith_angle", "air_temperature")}
    df = pd.DataFrame(cols, index=ti)
    meta = {k: (v.decode() if isinstance(v, bytes) else float(v) if isinstance(v, (np.floating, float)) else v) for k, v in zip(f["meta"].dtype.names, f["meta"][i])}
    out = dict(df=df, site_lat=float(co[i, 0]), site_lon=float(co[i, 1]), meta={k: str(v) for k, v in meta.items()}, source=URL)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, p)
    return out
