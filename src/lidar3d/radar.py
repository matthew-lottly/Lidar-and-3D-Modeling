"""Sentinel-1 radiometrically terrain-corrected backscatter (Gamma0, VV and VH) from the public Indigo Ag / Sentinel-1 RTC archive on AWS.

Data: 10 m COGs per MGRS 100 km square and acquisition, 2017 to April 2021, UTM-gridded. We read only the study-area window.
Gamma0 is linear power backscatter; decibels are 10*log10(gamma0).
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.windows import from_bounds

from .aoi import AOI
from .ept import CACHE

BASE = "https://sentinel-s1-rtc-indigo.s3.us-west-2.amazonaws.com"
PREFIX = "tiles/RTC/1/IW"


@dataclass(frozen=True)
class Scene:
    prefix: str
    date: str
    sat: str
    direction: str            # ASC / DSC


def _env():
    proxy = os.environ.get("HTTPS_PROXY", "").replace("http://", "").replace("https://", "")
    o = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2", CURL_CA_BUNDLE="/root/.ccr/ca-bundle.crt")
    if proxy:
        o["GDAL_HTTP_PROXY"] = proxy
    return rasterio.Env(**o)


def list_prefixes(prefix: str) -> list[str]:
    out, tok = [], None
    while True:
        p = {"list-type": 2, "prefix": prefix, "delimiter": "/"}
        if tok:
            p["continuation-token"] = tok
        t = requests.get(BASE + "/", params=p, timeout=60).text
        out += [x for x in re.findall(r"<Prefix>([^<]*)</Prefix>", t) if x != prefix]
        m = re.search(r"<NextContinuationToken>([^<]*)", t)
        if not m:
            return out
        tok = m.group(1)


def mgrs_square(lon: float, lat: float) -> tuple[str, str, str]:
    import mgrs
    m = mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)
    mm = re.match(r"(\d+)([A-Z])([A-Z]{2})", m)
    return mm.group(1), mm.group(2), mm.group(3)


def scenes(lon: float, lat: float) -> list[Scene]:
    z, band, sq = mgrs_square(lon, lat)
    out = []
    for ydir in list_prefixes(f"{PREFIX}/{z}/{band}/{sq}/"):
        for p in list_prefixes(ydir):
            m = re.search(r"/(S1[AB])_(\d{8})_\w+_(ASC|DSC)/$", p)
            if m:
                d = m.group(2)
                out.append(Scene(p, f"{d[:4]}-{d[4:6]}-{d[6:]}", m.group(1), m.group(3)))
    return sorted(out, key=lambda s: s.date)


def read_window(url: str, aoi: AOI):
    """Window of a COG covering the AOI; returns (array, transform). Reprojection is not needed: tiles are already UTM 10 m."""
    with _env(), rasterio.open(url) as src:
        assert src.crs.to_string() in ("EPSG:32610", "EPSG:32610"), src.crs
        x0, y0, x1, y1 = aoi.bounds
        w = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
        a = src.read(1, window=w, masked=True)
        return a, src.window_transform(w), src.nodata


def fetch_stack(sc: list[Scene], aoi: AOI, pols=("VV", "VH"), workers=6, log=print):
    """Reads every scene's AOI window for each polarisation; caches as one npz per (pol). Returns dict pol -> (dates, array[n, rows, cols])."""
    out = {}
    for pol in pols:
        f = CACHE / "s1" / f"stack_{pol}.npz"
        if f.exists():
            d = np.load(f, allow_pickle=True)
            out[pol] = (list(d["dates"]), list(d["dirs"]), d["arr"])
            continue
        def one(s):
            try:
                fn = "local_incident_angle" if pol == "LIA" else f"Gamma0_{pol}"
                a, _, _ = read_window(f"/vsicurl/{BASE}/{s.prefix}{fn}.tif", aoi)
                return s, np.ma.filled(a.astype("float32"), np.nan)
            except Exception as e:
                return s, None
        res = []
        with ThreadPoolExecutor(workers) as ex:
            for i, (s, a) in enumerate(ex.map(one, sc), 1):
                if a is not None:
                    res.append((s, a))
                if i % 50 == 0:
                    log(f"  {pol}: {i}/{len(sc)}")
        shp = res[0][1].shape
        keep = [(s, a) for s, a in res if a.shape == shp]
        arr = np.stack([a for _, a in keep])
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, dates=np.array([s.date for s, _ in keep]), dirs=np.array([f"{s.sat}_{s.direction}" for s, _ in keep]), arr=arr)
        out[pol] = ([s.date for s, _ in keep], [f"{s.sat}_{s.direction}" for s, _ in keep], arr)
    return out


# ------------------------------------------------------------------ analysis
import datetime as _dt


@dataclass
class RadarSet:
    dates: np.ndarray          # datetime64[D]
    vv_db: np.ndarray          # (n, rows, cols)
    vh_db: np.ndarray
    lia: np.ndarray            # local incidence angle, degrees
    direction: str
    sats: list
    transform: object
    extent: tuple              # (xmin, xmax, ymin, ymax) of the radar window in UTM


def to_db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 10 * np.log10(a)
    d[~np.isfinite(d)] = np.nan
    return d


def load_set(aoi: AOI, sc: list[Scene], min_scenes=25) -> RadarSet:
    """Load the cached stacks and keep the largest group of scenes that share viewing geometry (same direction and local incidence angle within 1 degree),
    so that differences between dates reflect the surface and not the look angle."""
    st = fetch_stack(sc, aoi, pols=("VV", "VH", "LIA"), log=lambda s: None)
    keys = {p: [f"{d}|{r}" for d, r in zip(st[p][0], st[p][1])] for p in st}
    common = [k for k in keys["VV"] if k in set(keys["VH"]) and k in set(keys["LIA"])]
    ix = lambda p: np.array([keys[p].index(k) for k in common])
    dates = [k.split("|")[0] for k in common]; dirs = [k.split("|")[1] for k in common]
    vv, vh, lia = st["VV"][2][ix("VV")], st["VH"][2][ix("VH")], st["LIA"][2][ix("LIA")].astype("float32") / 100.0
    mlia = np.nanmean(lia.reshape(len(dates), -1), axis=1)
    key = [f"{d.split('_')[1]}_{round(m)}" if np.isfinite(m) else "x_nan" for d, m in zip(dirs, mlia)]
    from collections import Counter
    best, n = Counter(k for k in key if k != 'x_nan').most_common(1)[0]
    sel = np.array([k == best for k in key])
    with _env():
        a, tr, _ = read_window(f"/vsicurl/{BASE}/{sc[0].prefix}Gamma0_VV.tif", aoi)
    h, w = a.shape
    ext = (tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f)
    d = np.array(dates)[sel].astype("datetime64[D]")
    o = np.argsort(d)
    return RadarSet(d[o], to_db(vv[sel][o]), to_db(vh[sel][o]), lia[sel][o], best.split("_")[0], [dirs[i] for i in np.where(sel)[0][o]], tr, ext)


def season_median(rs: RadarSet, arr: np.ndarray, months: tuple[int, ...]):
    m = np.array([int(str(d)[5:7]) in months for d in rs.dates])
    return np.nanmedian(arr[m], axis=0), int(m.sum())


def block_mean(arr: np.ndarray, aoi_x0: float, aoi_y1: float, res: float, rs: RadarSet) -> np.ndarray:
    """Mean of a fine raster (origin at the AOI north-west corner, `res` m cells) inside every radar pixel; NaN where the radar pixel is not fully inside the raster."""
    xr0, xr1, yr0, yr1 = rs.extent
    n_r, n_c = rs.vv_db.shape[1:]
    px = (xr1 - xr0) / n_c
    py = (yr1 - yr0) / n_r
    out = np.full((n_r, n_c), np.nan)
    k = int(round(px / res))
    for i in range(n_r):
        top = yr1 - i * py
        r0 = int(round((aoi_y1 - top) / res)); r1 = r0 + int(round(py / res))
        for j in range(n_c):
            left = xr0 + j * px
            c0 = int(round((left - aoi_x0) / res)); c1 = c0 + k
            if r0 >= 0 and c0 >= 0 and r1 <= arr.shape[0] and c1 <= arr.shape[1]:
                out[i, j] = np.nanmean(arr[r0:r1, c0:c1])
    return out
