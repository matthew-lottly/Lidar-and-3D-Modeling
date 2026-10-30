"""Topographic differencing between lidar epochs: registration on stable ground (ICP), DEMs of difference with a level of detection, M3C2 along the bluff face, volumes with uncertainty, and crest retreat.

Coordinates are local metres (UTM minus the area centre); heights are metres above the vertical datum of each survey as delivered (no datum conversion is applied, so a vertical datum difference between
surveys shows up as a uniform offset, which the registration step measures on stable ground and removes).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import cKDTree

from .aoi import AOI
from .ept import CACHE, EPT


@dataclass
class Epoch:
    tag: str
    project: str
    date: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    cls: np.ndarray
    nret: np.ndarray

    @property
    def xyz(self):
        return np.c_[self.x, self.y, self.z]


def make_aoi(site) -> AOI:
    cx, cy = Transformer.from_crs("EPSG:4326", site["crs"], always_xy=True).transform(*site["lonlat"])
    return AOI(round(cx), round(cy), site["half"], site["crs"])


def load_epoch(site, tag, project, log=print) -> Epoch:
    """All returns of one project inside the area (low and high noise classes dropped), cached as float32 local metres. The date comes from the GPS times when present, otherwise 'unknown'."""
    npz = CACHE / "topodiff" / f"{site['slug']}_{tag}.npz"
    aoi = make_aoi(site)
    if not npz.exists():
        c = EPT(project).fetch(aoi.bbox_ll, max_depth=20, log=log)
        x0, y0, x1, y1 = aoi.bounds
        m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & ~np.isin(c.cls, (7, 18))
        g = c.gps[c.gps > 0] if len(c.gps) else c.gps
        if len(g):
            v = float(np.median(g))
            v += 1e9 if v < 1e9 else 0.0
            date = str((dt.datetime(1980, 1, 6) + dt.timedelta(seconds=v)).date())
        else:
            date = "unknown"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, x=(c.x[m] - aoi.cx).astype("float32"), y=(c.y[m] - aoi.cy).astype("float32"), z=c.z[m].astype("float32"), cls=c.cls[m], nret=c.nret[m], date=date)
    d = np.load(npz)
    return Epoch(tag, project, str(d["date"]), d["x"], d["y"], d["z"], d["cls"], d["nret"])


def _cell_index(x, y, half, res):
    n = int(round(2 * half / res))
    return np.floor((y + half) / res).astype(int).clip(0, n - 1), np.floor((x + half) / res).astype(int).clip(0, n - 1), n


def surface_points(e: Epoch, half, res=1.0, tol=0.30, veg_margin=1.5, size=5, ground_like=(1, 2, 9, 10)):
    """Points on the bare surface chosen by geometry, because class codes differ between surveys: per `res` cell the lowest point, then points within `tol` plus the slope allowance of it, and cells whose lowest point
    stands more than `veg_margin` plus the slope allowance above the lowest in the surrounding `size` x `size` cells are dropped as canopy or roofs. Returns a boolean mask over the epoch."""
    ok = np.isin(e.cls, ground_like) | (e.cls == 6) & False
    iy, ix, n = _cell_index(e.x, e.y, half, res)
    key = iy * n + ix
    lo = np.full(n * n, np.inf)
    np.minimum.at(lo, key[ok], e.z[ok])
    cm = np.where(np.isfinite(lo), lo, np.nan).reshape(n, n)
    filled = np.where(np.isnan(cm), np.nanmedian(cm), cm)
    gy, gx = np.gradient(ndimage.gaussian_filter(filled, 1.0), res)
    allow = tol + res * np.hypot(gx, gy)
    local_min = ndimage.minimum_filter(np.where(np.isnan(cm), np.inf, cm), size=size)
    cell_ok = np.isfinite(cm) & (cm <= local_min + veg_margin + allow)
    return ok & cell_ok.ravel()[key] & (e.z <= lo[key] + allow.ravel()[key])


def dem(x, y, z, half, res=1.0, min_pts=1):
    """Mean height per cell of the points (row 0 = south); NaN where a cell has fewer than `min_pts` points. Also returns the point count and the standard deviation per cell."""
    iy, ix, n = _cell_index(x, y, half, res)
    key = iy * n + ix
    c = np.bincount(key, minlength=n * n).astype("float64")
    s = np.bincount(key, z.astype("float64"), minlength=n * n)
    s2 = np.bincount(key, z.astype("float64") ** 2, minlength=n * n)
    m = np.where(c >= min_pts, s / np.maximum(c, 1), np.nan)
    sd = np.sqrt(np.maximum(s2 / np.maximum(c, 1) - (s / np.maximum(c, 1)) ** 2, 0))
    return m.reshape(n, n), c.reshape(n, n), sd.reshape(n, n)


def slope_deg(d, res):
    gy, gx = np.gradient(np.where(np.isfinite(d), d, np.nanmedian(d)), res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def stable_zone(d_ref, half, res, min_dist=150.0, max_slope=8.0):
    """Cells far from the coast (`min_dist` m from any empty cell, i.e. from the sea or a void) and gently sloping: ground that is not expected to have changed. Chosen from geometry only, never from the differences."""
    land = np.isfinite(d_ref)
    dist = ndimage.distance_transform_edt(land, sampling=res)
    return land & (dist >= min_dist) & (slope_deg(d_ref, res) <= max_slope)


def normals_pca(P, k=12):
    t = cKDTree(P)
    _, nb = t.query(P, k=k)
    N = P[nb] - P[nb].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", N, N)
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]
    n[n[:, 2] < 0] *= -1
    return n


def icp_point_to_plane(src, dst, iters=40, max_dist=1.5, trim=0.8, tol=1e-5, rng=None):
    """Rigid point-to-plane ICP of `src` onto `dst` (both (n,3) on stable ground), small-angle linearisation, residuals trimmed to the best `trim` share and capped at `max_dist` m each iteration.
    Returns (R (3x3), t (3), history of rms residual). The transform maps src to dst: x' = R x + t."""
    nd = normals_pca(dst)
    tree = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    hist = []
    for _ in range(iters):
        S = src @ R.T + t
        d, j = tree.query(S, distance_upper_bound=max_dist)
        ok = np.isfinite(d)
        if ok.sum() < 100:
            break
        n = nd[j[ok]]
        r = ((dst[j[ok]] - S[ok]) * n).sum(1)
        keep = np.abs(r) <= np.quantile(np.abs(r), trim)
        Sk, nk, rk = S[ok][keep], n[keep], r[keep]
        A = np.c_[np.cross(Sk, nk), nk]
        x = np.linalg.lstsq(A, rk, rcond=None)[0]
        a, b, g = x[:3]
        dR = np.array([[1, -g, b], [g, 1, -a], [-b, a, 1]])
        u, _, vt = np.linalg.svd(dR)
        dR = u @ vt
        R, t = dR @ R, dR @ t + x[3:]
        hist.append(float(np.sqrt(np.mean(rk ** 2))))
        if np.abs(x).max() < tol:
            break
    return R, t, hist


def apply(R, t, e: Epoch, centre=(0.0, 0.0, 0.0)):
    """Apply a rigid transform to an epoch's points (about local origin) and return a new Epoch."""
    P = e.xyz.astype("float64") @ R.T + t
    return Epoch(e.tag, e.project, e.date, P[:, 0].astype("float32"), P[:, 1].astype("float32"), P[:, 2].astype("float32"), e.cls, e.nret)


def rotation_angles_deg(R):
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(R[2, 0]), np.arctan2(R[1, 0], R[0, 0])])
