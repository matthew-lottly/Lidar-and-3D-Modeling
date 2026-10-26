"""Load the study-area point cloud (cached) and derive rasters: DTM, DSM, CHM, intensity, density, terrain derivatives."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .aoi import AOI
from .ept import CACHE, Cloud, EPT

PROJECT = "WA_CentralWildfire_2_D22"
NPZ = CACHE / "aoi_cloud_full.npz"


def load_cloud(aoi: AOI, log=print) -> Cloud:
    if NPZ.exists():
        d = np.load(NPZ)
        return Cloud(**{k: d[k] for k in ("x", "y", "z", "cls", "intensity", "ret", "nret", "gps")}, crs=str(d["crs"]))
    c = EPT(PROJECT).fetch(aoi.bbox_ll, max_depth=14, log=log)
    x0, y0, x1, y1 = aoi.bounds
    m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & ~np.isin(c.cls, [7, 18])        # drop flagged noise
    c = Cloud(c.x[m], c.y[m], c.z[m], c.cls[m], c.intensity[m], c.ret[m], c.nret[m], c.gps, c.crs)
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, x=c.x, y=c.y, z=c.z, cls=c.cls, intensity=c.intensity, ret=c.ret, nret=c.nret, gps=c.gps, crs=c.crs)
    return c


class Rasters:
    """Everything on one grid: origin at the AOI's north-west corner, `res` metres per cell, row 0 = north."""

    def __init__(self, aoi: AOI, c: Cloud, res: float = 1.0):
        self.aoi, self.c, self.res = aoi, c, res
        x0, y0, x1, y1 = aoi.bounds
        self.x0, self.y1 = x0, y1
        self.shape = (int(round((y1 - y0) / res)), int(round((x1 - x0) / res)))
        self.col = np.clip(((c.x - x0) / res).astype(int), 0, self.shape[1] - 1)
        self.row = np.clip(((y1 - c.y) / res).astype(int), 0, self.shape[0] - 1)

    @property
    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.x0, self.y1, self.res, self.res)

    def _stat(self, mask, fn, init):
        out = np.full(self.shape, init)
        fn.at(out, (self.row[mask], self.col[mask]), self.c.z[mask])
        return out

    def count(self, mask=None) -> np.ndarray:
        m = np.ones(len(self.c), bool) if mask is None else mask
        return np.bincount(self.row[m] * self.shape[1] + self.col[m], minlength=self.shape[0] * self.shape[1]).reshape(self.shape)

    def dtm(self) -> np.ndarray:
        """Bare earth: lowest ground-class return per cell, holes filled by linear interpolation then nearest."""
        g = self.c.cls == 2
        a = self._stat(g, np.minimum, np.inf)
        a[~np.isfinite(a)] = np.nan
        return fill_nan(a)

    def dsm(self) -> np.ndarray:
        """Surface: highest return per cell."""
        a = self._stat(np.ones(len(self.c), bool), np.maximum, -np.inf)
        a[~np.isfinite(a)] = np.nan
        return fill_nan(a)

    def intensity(self) -> np.ndarray:
        f = self.c.ret == 1
        n = np.bincount(self.row[f] * self.shape[1] + self.col[f], minlength=self.shape[0] * self.shape[1]).astype(float)
        s = np.bincount(self.row[f] * self.shape[1] + self.col[f], weights=self.c.intensity[f].astype(float), minlength=self.shape[0] * self.shape[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            a = (s / n).reshape(self.shape)
        return fill_nan(a)

    def write_tif(self, path: Path, arr: np.ndarray, nodata=-9999.0):
        import rasterio
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32", crs=self.aoi.crs, transform=self.transform,
                           nodata=nodata, compress="deflate", predictor=3, tiled=True) as dst:
            dst.write(np.where(np.isnan(arr), nodata, arr).astype("float32"), 1)


def fill_nan(a: np.ndarray) -> np.ndarray:
    m = np.isnan(a)
    if not m.any():
        return a
    idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
    out = a[tuple(idx)]
    # smooth only the filled cells so measured cells stay exact
    sm = ndimage.gaussian_filter(out, 1.5)
    return np.where(m, sm, out)


def hillshade(z: np.ndarray, res: float, azimuth=315.0, altitude=40.0, zf=1.0) -> np.ndarray:
    gy, gx = np.gradient(z * zf, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(360 - azimuth + 90), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def multi_hillshade(z, res, zf=1.0):
    return np.mean([hillshade(z, res, az, 38, zf) for az in (225, 270, 315, 360)], axis=0)


def slope_deg(z, res):
    gy, gx = np.gradient(z, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def tpi(z, radius_cells=20):
    """Topographic position index: cell minus mean of a surrounding window (positive = ridge, negative = valley)."""
    k = 2 * radius_cells + 1
    return z - ndimage.uniform_filter(z, size=k, mode="nearest")


def tree_tops(chm: np.ndarray, res: float, min_h=3.0, window_m=3.0):
    """Local maxima of a lightly smoothed canopy height model above `min_h`: row, col, height."""
    sm = ndimage.gaussian_filter(chm, 1.0)
    mx = ndimage.maximum_filter(sm, size=int(round(window_m / res)) | 1)
    r, c = np.where((sm == mx) & (sm >= min_h))
    return r, c, sm[r, c]
