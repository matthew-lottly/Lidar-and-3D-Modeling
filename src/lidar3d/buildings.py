"""Building models from a 3DEP point cloud and Overture footprints.

For every footprint: the ground level from ground-class returns in a ring around it, the roof level from the highest return per 0.5 m cell inside it,
and the dominant roof plane by RANSAC. The result is an LOD1 block model (flat roof) plus attributes, not an LOD2 roof reconstruction.
"""
from __future__ import annotations

import numpy as np
import rasterio.features
from shapely.ops import unary_union

from .cloud import Rasters

TOP_PCT = 70          # LOD1 roof level = this percentile of the roof surface inside the footprint (ignores small rooftop structures, keeps the main roof)


def fit_plane(pts, tol=0.25, iters=300, rng=None):
    """RANSAC plane z = a x + b y + c on (n,3) points. Returns (coeffs, inlier_mask) for the plane with most inliers, refined by least squares."""
    rng = rng or np.random.default_rng(0)
    n = len(pts)
    if n < 6:
        return None, np.zeros(n, bool)
    sub = pts if n <= 4000 else pts[rng.choice(n, 4000, replace=False)]
    best, best_k = None, 0
    for _ in range(iters):
        s = sub[rng.choice(len(sub), 3, replace=False)]
        A = np.c_[s[:, :2], np.ones(3)]
        if abs(np.linalg.det(A)) < 1e-6:
            continue
        co = np.linalg.solve(A, s[:, 2])
        k = int((np.abs(sub[:, 2] - (sub[:, :2] @ co[:2] + co[2])) < tol).sum())
        if k > best_k:
            best, best_k = co, k
    if best is None:
        return None, np.zeros(n, bool)
    for _ in range(2):
        m = np.abs(pts[:, 2] - (pts[:, :2] @ best[:2] + best[2])) < tol
        if m.sum() < 3:
            break
        best = np.linalg.lstsq(np.c_[pts[m, :2], np.ones(m.sum())], pts[m, 2], rcond=None)[0]
    return best, np.abs(pts[:, 2] - (pts[:, :2] @ best[:2] + best[2])) < tol


def analyse(aoi, cloud, footprints, res=0.5, min_area=15.0):
    """Returns (GeoDataFrame with one row per modelled building, dict of rasters used)."""
    R = Rasters(aoi, cloud, res=res)
    tf, shape = R.transform, R.shape
    dsm = R._stat(np.ones(len(cloud), bool), np.maximum, -np.inf)
    dsm[~np.isfinite(dsm)] = np.nan
    gnd = R._stat(cloud.cls == 2, np.minimum, np.inf)
    gnd[~np.isfinite(gnd)] = np.nan
    xs = R.x0 + (np.arange(shape[1]) + 0.5) * res
    ys = R.y1 - (np.arange(shape[0]) + 0.5) * res
    fp = footprints[footprints.geometry.area >= min_area].copy().reset_index(drop=True)
    allfp = unary_union(list(footprints.geometry.buffer(1.0)))
    rast = lambda g: rasterio.features.rasterize([(g, 1)], out_shape=shape, transform=tf, fill=0, dtype="uint8").astype(bool) if not g.is_empty else np.zeros(shape, bool)
    rows = []
    for _, r in fp.iterrows():
        g = r.geometry
        inner = g.buffer(-0.75)
        m = rast(inner if not inner.is_empty else g)
        z = dsm[m]
        z = z[np.isfinite(z)]
        # ground: ring around the footprint, outside every footprint
        gz = np.array([])
        for ring_w in (4.0, 12.0):
            ring = g.buffer(ring_w).difference(allfp)
            gz = gnd[rast(ring)]
            gz = gz[np.isfinite(gz)]
            if len(gz) >= 20:
                break
        row = dict(r.drop("geometry"))
        row.update(area_m2=float(g.area), lidar_cells=int(len(z)), cells_expected=int(round(inner.area / res ** 2)) if not inner.is_empty else 0, ground_cells=int(len(gz)))
        if len(z) < 10 or len(gz) < 5:
            row.update(status="no_lidar", geometry=g)
            rows.append(row)
            continue
        gmed, glow = float(np.median(gz)), float(np.percentile(gz, 10))
        top = float(np.percentile(z, TOP_PCT))
        zmax = float(z.max())
        yy, xx = np.nonzero(m & np.isfinite(dsm))
        P = np.c_[xs[xx], ys[yy], dsm[yy, xx]]
        co, inl = fit_plane(P)
        tilt = float(np.degrees(np.arctan(np.hypot(*co[:2])))) if co is not None else np.nan
        share = float(inl.mean()) if co is not None else 0.0
        row.update(status="ok", ground_z=gmed, base_z=glow, roof_z=top, roof_max_z=zmax, height_m=top - gmed, height_max_m=zmax - gmed, roof_spread_m=float(np.percentile(z, 90) - np.percentile(z, 10)),
                   plane_share=share, plane_tilt_deg=tilt, roof_type=("flat" if (share > 0.5 and tilt < 5) else "pitched" if share > 0.5 else "complex"), volume_m3=float(g.area * max(top - glow, 0)))
        row["geometry"] = g
        rows.append(row)
    import geopandas as gpd
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=footprints.crs)
    return out, dict(dsm=dsm, gnd=gnd, res=res)
