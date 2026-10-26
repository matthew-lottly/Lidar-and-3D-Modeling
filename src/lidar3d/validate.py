"""Independent checks of the lidar land cover: Sentinel-2 NDVI inside each class, and agreement with ESA WorldCover."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from . import imagery
from .landcover import CLASSES, compare_worldcover
from .radar import _env

WC_URL = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N45W123_Map.tif"


def worldcover(aoi) -> np.ndarray:
    x0, y0, x1, y1 = aoi.bounds
    n = int(round(aoi.size_m / 10))
    dst = np.zeros((n, n), np.uint8)
    with _env(), rasterio.open(WC_URL) as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=from_origin(x0, y1, 10, 10), dst_crs=aoi.crs, resampling=Resampling.nearest)
    return dst


def sentinel2(aoi, lon, lat):
    best = imagery.best_scene(lon, lat, aoi)
    scene = best[1]
    red, _ = imagery.read(scene, "B04", aoi, pad=0)
    nir, _ = imagery.read(scene, "B08", aoi, pad=0)
    tci, _ = imagery.read(scene, "TCI", aoi, pad=0)
    r = np.ma.filled(red[0].astype(float), np.nan) / 1e4
    n = np.ma.filled(nir[0].astype(float), np.nan) / 1e4
    return scene, (n - r) / (n + r), np.moveaxis(np.ma.filled(tci, 0).astype(np.uint8), 0, -1)


def run(P, LC, lon, lat) -> dict:
    scene, ndvi, tci = sentinel2(P.aoi, lon, lat)
    n = ndvi.shape[0]
    blk = LC.cls.shape[0] // n
    ndvi_by = {}
    for k in CLASSES:
        share = (LC.cls == k).astype(float)[: n * blk, : n * blk].reshape(n, blk, n, blk).mean((1, 3))
        sel = share > 0.8
        ndvi_by[k] = float(np.nanmean(ndvi[sel])) if sel.sum() >= 3 else None
    wc = worldcover(P.aoi)
    return dict(s2_scene=scene.prefix.split("/")[-1], s2_date=scene.date, ndvi=ndvi_by, wc=compare_worldcover(LC, P.aoi, wc, None), tci=tci, ndvi_map=ndvi)
