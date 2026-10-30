"""Accuracy assessment against independent real data. Every function takes a finished analysis product and an independent reference (a second lidar survey, imagery, a third-party map) and returns plain numbers.
References that could not be reached are reported as such by the caller; nothing here invents a reference. Independent here means: a different survey, sensor or producer than the one the analysis used."""
from __future__ import annotations

import datetime as dt

import numpy as np
from scipy import ndimage

from . import forestry as FO
from .aoi import AOI
from .ept import CACHE, EPT


def nmad(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else float("nan")


def error_summary(d):
    """Median (bias), normalised median absolute deviation (robust spread), rms, 95th percentile of |error| and count for a vector of differences."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(n=0)
    return dict(n=int(len(d)), bias_m=float(np.median(d)), nmad_m=nmad(d), rmse_m=float(np.sqrt(np.mean(d ** 2))), mae_m=float(np.mean(np.abs(d))), p95_abs_m=float(np.percentile(np.abs(d), 95)))


def load_project_points(slug, project, aoi: AOI, log=lambda s: None):
    """All returns of another 3DEP project inside the analysis square (noise classes dropped), cached; local metres relative to the same centre as the analysis. Returns dict x, y, z, cls, nret, date or None
    if the project has no points there."""
    npz = CACHE / "validation" / f"{slug}_{project}.npz"
    if not npz.exists():
        try:
            c = EPT(project).fetch(aoi.bbox_ll, max_depth=20, log=log)
        except Exception:
            return None
        x0, y0, x1, y1 = aoi.bounds
        m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & ~np.isin(c.cls, (7, 18))
        if not m.any():
            return None
        g = c.gps[c.gps > 0] if len(c.gps) else c.gps
        date = "unknown"
        if len(g):
            v = float(np.median(g))
            v += 1e9 if v < 1e9 else 0.0
            date = str((dt.datetime(1980, 1, 6) + dt.timedelta(seconds=v)).date())
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, x=(c.x[m] - aoi.cx).astype("float32"), y=(c.y[m] - aoi.cy).astype("float32"), z=c.z[m].astype("float32"), cls=c.cls[m], nret=c.nret[m], date=date)
    d = np.load(npz)
    return dict(x=d["x"], y=d["y"], z=d["z"], cls=d["cls"], nret=d["nret"], date=str(d["date"]))


def checkpoint_errors(dem, half, res, ref, classes=(2,), mask=None, max_slope_deg=None):
    """Heights of an independent survey's ground returns (`classes`) minus the analysis DEM at the same places (a DEM error where nothing has changed). `mask` (boolean grid like `dem`) restricts to cells of
    interest, `max_slope_deg` drops steep cells where horizontal offsets turn into vertical error. Returns the vector of errors."""
    m = np.isin(ref["cls"], classes)
    x, y, z = ref["x"][m], ref["y"][m], ref["z"][m]
    pred = FO.sample(dem, half, res, x, y)
    if mask is not None:
        ok = FO.sample(mask.astype("float32"), half, res, x, y) > 0.99
        pred = np.where(ok, pred, np.nan)
    if max_slope_deg is not None:
        gy, gx = np.gradient(np.where(np.isfinite(dem), dem, np.nanmedian(dem)), res)
        sl = np.degrees(np.arctan(np.hypot(gx, gy)))
        pred = np.where(FO.sample(sl, half, res, x, y) <= max_slope_deg, pred, np.nan)
    return z - pred


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def binary_agreement(pred, ref, valid=None):
    """Confusion-matrix numbers of a boolean prediction against a boolean reference over the valid cells: overall accuracy, kappa, precision, recall (of the positive class), IoU."""
    p, r = np.asarray(pred, bool), np.asarray(ref, bool)
    if valid is not None:
        p, r = p[valid], r[valid]
    tp, fp, fn, tn = [float(v) for v in ((p & r).sum(), (p & ~r).sum(), (~p & r).sum(), (~p & ~r).sum())]
    n = tp + fp + fn + tn
    po = (tp + tn) / n if n else float("nan")
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2 if n else float("nan")
    return dict(n=int(n), accuracy=po, kappa=(po - pe) / (1 - pe) if n and pe < 1 else float("nan"), precision=tp / (tp + fp) if tp + fp else float("nan"), recall=tp / (tp + fn) if tp + fn else float("nan"),
                iou=tp / (tp + fp + fn) if tp + fp + fn else float("nan"), reference_positive_share=float(r.mean()), predicted_positive_share=float(p.mean()))


def match_points(a, b, radius):
    """One-to-one nearest matching of 2D points `a` to `b` within `radius`. Returns (index in b or -1 for each a, distances)."""
    from scipy.spatial import cKDTree
    out = -np.ones(len(a), int)
    dist = np.full(len(a), np.nan)
    if not len(a) or not len(b):
        return out, dist
    d, j = cKDTree(b).query(a, distance_upper_bound=radius)
    order = np.argsort(d)
    used = set()
    for i in order:
        if np.isfinite(d[i]) and int(j[i]) not in used:
            out[i], dist[i] = int(j[i]), d[i]
            used.add(int(j[i]))
    return out, dist


def smooth_mask(m, px):
    return ndimage.binary_opening(ndimage.binary_closing(m, iterations=px), iterations=px)
