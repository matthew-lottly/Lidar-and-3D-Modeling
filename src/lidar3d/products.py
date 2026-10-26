"""Compute and cache every derived raster and statistic from the point cloud. Nothing here is typed in: numbers come from the data."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .aoi import AOI, RAINIER_SUMMIT
from .cloud import Rasters, load_cloud, multi_hillshade, slope_deg, tpi, tree_tops
from .ept import CACHE, ROOT

PROC = ROOT / "data" / "processed"


@dataclass
class Products:
    aoi: AOI
    R: Rasters
    dtm: np.ndarray
    dsm: np.ndarray
    chm: np.ndarray
    inten: np.ndarray
    density: np.ndarray
    ground_density: np.ndarray
    hs: np.ndarray
    slope: np.ndarray
    tpi_s: np.ndarray
    tops: tuple
    stats: dict = field(default_factory=dict)

    @property
    def extent(self):
        x0, y0, x1, y1 = self.aoi.bounds
        return (x0, x1, y0, y1)


def collection_dates(gps: np.ndarray):
    """LAS 'adjusted standard GPS time' is GPS seconds minus 1e9; convert to calendar dates (UTC, leap seconds ignored: error < 20 s)."""
    g = gps[gps > 0]
    d = [(dt.datetime(1980, 1, 6) + dt.timedelta(seconds=float(v) + 1e9)).date() for v in np.percentile(g, [0.5, 99.5])]
    return d[0], d[1]


def build(aoi: AOI | None = None, log=print) -> Products:
    aoi = aoi or AOI.default()
    c = load_cloud(aoi, log)
    R = Rasters(aoi, c, 1.0)
    dtm, dsm = R.dtm(), R.dsm()
    chm = np.clip(dsm - dtm, 0, None)
    inten = R.intensity()
    dens = R.count() / R.res ** 2
    gdens = R.count(c.cls == 2) / R.res ** 2
    dtm_s = ndimage.gaussian_filter(dtm, 1.5)
    slope = slope_deg(dtm_s, 1.0)
    tp = tpi(dtm_s, 25)
    tops = tree_tops(chm, 1.0, 3.0, 3.0)
    P = Products(aoi, R, dtm, dsm, chm, inten, dens, gdens, multi_hillshade(dtm_s, 1.0, 1.5), slope, tp, tops)
    d0, d1 = collection_dates(c.gps)
    area_km2 = (aoi.size_m / 1000) ** 2
    th = tops[2]
    cls_n = {int(k): int(v) for k, v in zip(*np.unique(c.cls, return_counts=True))}
    sl_bins = [0, 10, 20, 30, 40, 50, 90]
    sl_pct = {f"{a}-{b}": float(((slope >= a) & (slope < b)).mean() * 100) for a, b in zip(sl_bins[:-1], sl_bins[1:])}
    ri, ci = np.unravel_index(np.nanargmax(dtm), dtm.shape)
    from pyproj import Transformer
    t = Transformer.from_crs(aoi.crs, "EPSG:4326", always_xy=True)
    px, py = aoi.bounds[0] + (ci + .5), aoi.bounds[3] - (ri + .5)
    lon, lat = t.transform(px, py)
    sx, sy = Transformer.from_crs("EPSG:4326", aoi.crs, always_xy=True).transform(*RAINIER_SUMMIT)
    P.stats = dict(
        points=int(len(c)), area_km2=area_km2, pts_per_m2=len(c) / (area_km2 * 1e6), ground_share=float((c.cls == 2).mean()), classes=cls_n,
        collected=[str(d0), str(d1)], z_min=float(c.z.min()), z_max=float(c.z.max()), dtm_min=float(dtm.min()), dtm_max=float(dtm.max()), relief_m=float(dtm.max() - dtm.min()),
        ground_cell_cover=float((R.count(c.cls == 2) > 0).mean()), empty_cells=float((R.count() == 0).mean()),
        canopy_cover_3m=float((chm > 3).mean()), canopy_cover_8m=float((chm > 8).mean()), chm_max=float(chm.max()),
        trees=int(len(th)), tree_h_mean=float(th.mean()) if len(th) else 0.0, tree_h_p90=float(np.percentile(th, 90)) if len(th) else 0.0, tree_h_max=float(th.max()) if len(th) else 0.0,
        tree_density_per_ha=float(len(th) / (area_km2 * 100)), slope_mean=float(slope.mean()), slope_pct=sl_pct, steep_over_40=float((slope >= 40).mean() * 100),
        summit_lonlat=[float(lon), float(lat)], summit_en=[float(px), float(py)], summit_z=float(dtm[ri, ci]),
        km_to_rainier=float(np.hypot(sx - aoi.cx, sy - aoi.cy) / 1000),
    )
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "stats.json").write_text(json.dumps(P.stats, indent=1))
    for name, arr in (("dtm_1m", dtm), ("dsm_1m", dsm), ("chm_1m", chm)):
        R.write_tif(PROC / f"{name}.tif", arr)
    return P
