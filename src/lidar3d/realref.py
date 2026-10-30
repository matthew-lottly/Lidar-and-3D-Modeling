"""Independent reference data for accuracy assessment, from the Microsoft Planetary Computer STAC catalogue (public, no key): the USGS-processed 3DEP lidar products (DTM, HAG, DSM at 2 m), 3DEP seamless DEMs,
ESA WorldCover, Esri land cover, USGS LCMAP, Chesapeake 1 m land cover, JRC global surface water, Deltares flood maps, and FIA forest plots.

Every function reads a window of a reference raster resampled onto the analysis grid, in the analysis CRS, so the comparison is cell to cell. Nothing here feeds the analyses; it is read after they are done.
"""
from __future__ import annotations

import os

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{}"
_tok = {}


def search(collection, bbox=None, point=None, datetime=None, limit=100, query=None):
    body = {"collections": [collection], "limit": limit}
    if bbox:
        body["bbox"] = list(bbox)
    if point:
        body["intersects"] = {"type": "Point", "coordinates": list(point)}
    if datetime:
        body["datetime"] = datetime
    if query:
        body["query"] = query
    r = requests.post(STAC, json=body, timeout=90)
    r.raise_for_status()
    return r.json()["features"]


def token(collection):
    if collection not in _tok:
        _tok[collection] = requests.get(SAS.format(collection), timeout=60).json()["token"]
    return _tok[collection]


def _env():
    os.environ.setdefault("CURL_CA_BUNDLE", os.environ.get("AWS_CA_BUNDLE", "/root/.ccr/ca-bundle.crt"))
    return rasterio.Env(GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")


def read_grid(items, collection, asset, x0, y1, res, shape, crs, resampling=Resampling.nearest, band=1, nodata=None):
    """Mosaic of the `asset` of every item, resampled onto the grid with top-left corner (x0, y1), cell size `res` and `shape` = (rows, cols) in `crs`. Returns an array with NaN where no item has data
    (row 0 = north, like a normal raster)."""
    tf = from_origin(x0, y1, res, res)
    out = np.full(shape, np.nan, "float32")
    tok = token(collection)
    with _env():
        for it in items:
            href = it["assets"][asset]["href"] + "?" + tok
            try:
                with rasterio.open(href) as src, WarpedVRT(src, crs=crs, transform=tf, width=shape[1], height=shape[0], resampling=resampling, nodata=src.nodata if nodata is None else nodata) as v:
                    a = v.read(band, masked=True).astype("float32")
            except Exception:
                continue
            m = ~np.ma.getmaskarray(a) & np.isnan(out)
            out[m] = np.ma.getdata(a)[m]
    return out


def lidar_product(kind, project, bbox, x0, y1, res, shape, crs):
    """USGS-processed 3DEP lidar product ('dtm', 'hag' or 'dsm', 2 m) of one named project, on the analysis grid. Returns None if the project has no product over the box.

    The STAC footprints of these items are unreliable (a bbox search can return the wrong tiles), so the tiles are found from the files themselves: the item ids end in a tile row-column pair, the tiles sit on a regular 8192 m lattice
    in the project's UTM zone, and the lattice origin is read from the one tile the search does return."""
    from pyproj import Transformer
    col = f"3dep-lidar-{kind}"
    seed = [i for i in search(col, bbox=bbox, limit=100) if i["id"].startswith(project + "-")]
    if not seed:
        return None
    tok = token(col)
    with _env(), rasterio.open(seed[0]["assets"]["data"]["href"] + "?" + tok) as s0:
        tcrs, step = s0.crs, 8192.0
        ti, tj = (int(v) for v in seed[0]["id"].rsplit("-", 2)[-2:])
        left0, top0 = s0.bounds.left, s0.bounds.top
    x1, y0 = x0 + shape[1] * res, y1 - shape[0] * res
    tr = Transformer.from_crs(crs, tcrs, always_xy=True)
    xs, ys = tr.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    ids = []
    for i in range(ti - 6, ti + 7):
        for j in range(tj - 6, tj + 7):
            l, t = left0 + (i - ti) * step, top0 + (j - tj) * step
            if l <= max(xs) and l + step >= min(xs) and t - step <= max(ys) and t >= min(ys):
                ids.append(f"{project}-{kind}-2m-{i}-{j}")
    r = requests.post(STAC, json={"collections": [col], "ids": ids, "limit": 100}, timeout=90)
    r.raise_for_status()
    items = r.json()["features"]
    return read_grid(items, col, "data", x0, y1, res, shape, crs, Resampling.bilinear) if items else None


def seamless(bbox, x0, y1, res, shape, crs, collection="3dep-seamless"):
    """3DEP seamless 1/3 arc-second DEM (about 10 m, built from the best available lidar and other sources) on the analysis grid."""
    items = [i for i in search(collection, bbox=bbox) if "gsd" in i["properties"] and abs(i["properties"]["gsd"] - 10) < 1.5]
    return read_grid(items, collection, "data", x0, y1, res, shape, crs, Resampling.bilinear) if items else None


# ------------------------------------------------------------------ land cover, water, trees
WORLDCOVER = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up", 60: "bare / sparse vegetation", 70: "snow and ice", 80: "permanent water", 90: "herbaceous wetland", 95: "mangroves", 100: "moss and lichen"}
CHESAPEAKE13 = {1: "water", 2: "emergent wetlands", 3: "tree canopy", 4: "shrubland", 5: "low vegetation", 6: "barren", 7: "structures", 8: "other impervious", 9: "roads", 10: "tree canopy over structures", 11: "tree canopy over other impervious", 12: "tree canopy over roads", 15: "aberration"}
IOLULC = {1: "water", 2: "trees", 4: "flooded vegetation", 5: "crops", 7: "built area", 8: "bare ground", 9: "snow/ice", 10: "clouds", 11: "rangeland"}


def worldcover(bbox, x0, y1, res, shape, crs, year=2021):
    items = [i for i in search("esa-worldcover", bbox=bbox) if i["properties"]["start_datetime"].startswith(str(year))]
    return read_grid(items, "esa-worldcover", "map", x0, y1, res, shape, crs, Resampling.nearest) if items else None


def esri_lulc(bbox, x0, y1, res, shape, crs, year=2022):
    items = [i for i in search("io-lulc-annual-v02", bbox=bbox, datetime=f"{year}-01-01/{year}-12-31")]
    return read_grid(items, "io-lulc-annual-v02", "data", x0, y1, res, shape, crs, Resampling.nearest) if items else None


def chesapeake(bbox, x0, y1, res, shape, crs):
    items = search("chesapeake-lc-13", bbox=bbox)
    return read_grid(items, "chesapeake-lc-13", "data", x0, y1, res, shape, crs, Resampling.nearest) if items else None


def jrc_water(bbox, x0, y1, res, shape, crs, layer="occurrence"):
    """JRC Global Surface Water layer (occurrence = percent of observations 1984 to 2021 with water; 30 m)."""
    items = search("jrc-gsw", bbox=bbox)
    return read_grid(items, "jrc-gsw", layer, x0, y1, res, shape, crs, Resampling.nearest) if items else None


def reachable(href, collection):
    """True when the host of an asset answers a small range request (the session's network policy may block the blob hosts)."""
    try:
        r = requests.get(href + "?" + token(collection), headers={"Range": "bytes=0-100"}, timeout=30)
        return r.status_code in (200, 206)
    except Exception:
        return False
