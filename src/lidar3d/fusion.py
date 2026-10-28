"""Fusing lidar, aerial imagery and radar evidence for utility supports (poles, towers) and lines.

Nothing here uses mapped features (OpenStreetMap or otherwise): the evidence layers and the model are built from the three sensors alone, so a later comparison with OSM is a real check
and not a fit. Candidate supports come from lidar (vertical columns, deliberately many and loose); imagery and radar then add evidence at each candidate, and a two-component Gaussian mixture
separates 'structure' from 'everything else' without labels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import sato

from .utility import match_points

LIDAR = ["top_hag", "height", "linearity", "radius", "log_points", "log_ring", "wire_near"]
IMAGERY = ["ndvi_base", "bright_base", "ridge_near"]
RADAR = ["radar_vh_z", "radar_vv_z"]


# ------------------------------------------------------------------ imagery
def ndvi(rgbn: np.ndarray) -> np.ndarray:
    r, n = rgbn[0].astype(float), rgbn[3].astype(float)
    return (n - r) / np.maximum(n + r, 1.0)


def ridge_map(rgbn: np.ndarray, sigmas=(1.0, 1.5, 2.0)) -> np.ndarray:
    """Thin-line response of the image (Sato ridge filter on the grey image, bright and dark lines), scaled to 0..1 by the 99th percentile. Roads, fences, roof edges and power lines all respond."""
    g = rgbn[:3].astype(float).mean(0)
    r = sato(g, sigmas=sigmas, black_ridges=False) + sato(g, sigmas=sigmas, black_ridges=True)
    return np.clip(r / np.percentile(r, 99), 0, 1)


class Raster:
    """A north-up grid in local metres (x east, y north, relative to the study-area centre) that can be sampled at points."""

    def __init__(self, arr: np.ndarray, x0: float, y1: float, res: float):
        self.arr, self.x0, self.y1, self.res = arr, x0, y1, res

    def at(self, x, y, radius=0.0, stat="mean"):
        """Value at points, or the mean / max inside a disk of `radius` metres."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        out = np.full(len(x), np.nan)
        rr = max(int(np.ceil(radius / self.res)), 0)
        for i, (a, b) in enumerate(zip(x, y)):
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            c, r = int((a - self.x0) / self.res), int((self.y1 - b) / self.res)
            if not (0 <= r < self.arr.shape[0] and 0 <= c < self.arr.shape[1]):
                continue
            w = self.arr[max(r - rr, 0):r + rr + 1, max(c - rr, 0):c + rr + 1]
            out[i] = np.nanmax(w) if stat == "max" else np.nanmean(w)
        return out


# ------------------------------------------------------------------ candidate features
def wire_context(cands_xy, tops, wire_xyz, hag_w, radius=8.0):
    """For each candidate: number of wire points within `radius` in plan whose height above ground is at least 60% of the candidate's top (wires hang from the top of a support)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(wire_xyz[:, :2])
    out = np.zeros(len(cands_xy))
    for i, (p, t) in enumerate(zip(cands_xy, tops)):
        idx = tree.query_ball_point(p, radius)
        if idx:
            out[i] = (hag_w[idx] >= 0.6 * t).sum()
    return out


def build_features(cand, ndvi_r: Raster, bright_r: Raster, ridge_r: Raster, wire_xyz, wire_hag, radar_vh: Raster | None = None, radar_vv: Raster | None = None):
    """Adds imagery, radar and wire-context columns to a candidate DataFrame (columns x, y, top_hag, points, ring_ratio, ...)."""
    f = cand.copy()
    f["log_points"] = np.log1p(f["points"])
    f["log_ring"] = np.log1p(f["ring_ratio"])
    f["wire_near"] = np.log1p(wire_context(f[["x", "y"]].to_numpy(), f["top_hag"].to_numpy(), wire_xyz, wire_hag))
    f["ndvi_base"] = ndvi_r.at(f.x, f.y, 2.5)
    f["bright_base"] = bright_r.at(f.x, f.y, 2.5)
    f["ridge_near"] = ridge_r.at(f.x, f.y, 3.0, "max")
    for name, r in (("radar_vh_z", radar_vh), ("radar_vv_z", radar_vv)):
        f[name] = r.at(f.x, f.y) if r is not None else np.nan
    return f


def zscore_raster(arr: np.ndarray, size=7) -> np.ndarray:
    """Local z-score of a coarse raster against its own neighbourhood: how unusual a pixel is for its surroundings."""
    m = ndimage.uniform_filter(arr, size, mode="nearest")
    s = np.sqrt(np.maximum(ndimage.uniform_filter(arr ** 2, size, mode="nearest") - m ** 2, 1e-9))
    return (arr - m) / s


# ------------------------------------------------------------------ unsupervised fusion
def fit_mixture(feats, columns, anchor="wire_near", seed=0, n_init=10):
    """Two-component Gaussian mixture on standardised `columns`. The component whose mean of `anchor` (a physical cue: wires hang from supports) is higher is called 'structure'.
    Returns (posterior probability of 'structure' per row, dict of component means in original units)."""
    from sklearn.mixture import GaussianMixture
    X = feats[columns].to_numpy(float)
    keep = np.isfinite(X).all(1)
    mu, sd = X[keep].mean(0), X[keep].std(0) + 1e-9
    gm = GaussianMixture(2, covariance_type="diag", n_init=n_init, random_state=seed, reg_covar=1e-3).fit((X[keep] - mu) / sd)
    ai = columns.index(anchor)
    k = int(np.argmax(gm.means_[:, ai]))
    p = np.full(len(feats), np.nan)
    p[keep] = gm.predict_proba((X[keep] - mu) / sd)[:, k]
    means = {c: (float(gm.means_[k, j] * sd[j] + mu[j]), float(gm.means_[1 - k, j] * sd[j] + mu[j])) for j, c in enumerate(columns)}
    return p, means


def non_max_suppress(feats, score, radius=3.0):
    """Keep the highest-scoring candidate within `radius` metres of each other."""
    order = np.argsort(-np.nan_to_num(score, nan=-1))
    kept = []
    xy = feats[["x", "y"]].to_numpy()
    for i in order:
        if all(np.hypot(*(xy[i] - xy[j])) > radius for j in kept):
            kept.append(i)
    return np.array(sorted(kept))


def evaluate_supports(det_xy, ref_xy, radius=3.0):
    pairs, ud, ur = match_points(np.asarray(det_xy).reshape(-1, 2), np.asarray(ref_xy).reshape(-1, 2), radius)
    n_d, n_r = len(det_xy), len(ref_xy)
    return dict(detected=int(n_d), reference=int(n_r), matched=len(pairs), precision=len(pairs) / max(n_d, 1), recall=len(pairs) / max(n_r, 1),
                median_offset_m=float(np.median([p[2] for p in pairs])) if pairs else None)


def fit_mixture_k(feats, columns, k=6, anchor="wire_near", anchor_min=1.0, compact="spread_", compact_max=1.6, seed=0, n_init=10):
    """Gaussian mixture with `k` components on standardised `columns`; a component counts as 'structure' when its mean `anchor` is at least `anchor_min` (wires present near the top, in log1p units)
    and its mean `compact` is at most `compact_max` (a narrow footprint). The criterion is set from the physics of a support, not from any map. Returns (posterior of structure, list of structure component means)."""
    from sklearn.mixture import GaussianMixture
    X = feats[columns].to_numpy(float)
    keep = np.isfinite(X).all(1)
    mu, sd = X[keep].mean(0), X[keep].std(0) + 1e-9
    gm = GaussianMixture(k, covariance_type="diag", n_init=n_init, random_state=seed, reg_covar=1e-3).fit((X[keep] - mu) / sd)
    means = gm.means_ * sd + mu
    ai, ci = columns.index(anchor), columns.index(compact)
    good = [j for j in range(k) if means[j, ai] >= anchor_min and means[j, ci] <= compact_max]
    p = np.full(len(feats), np.nan)
    p[keep] = gm.predict_proba((X[keep] - mu) / sd)[:, good].sum(1) if good else 0.0
    return p, [dict(zip(columns, map(float, means[j]))) for j in good]


# a priori signs: +1 when a larger value is evidence for a support, -1 when a smaller value is. Set from the physics of a support before any comparison with a map.
SIGNS_LIDAR = {"obj_dist": -1, "log_cond": +1, "extent": +1, "spread": -1}
SIGNS_IMAGERY = {"ndvi_base": -1, "ridge_near": +1}
SIGNS_RADAR = {"radar_vv_z": +1, "radar_vh_z": +1}


def composite_score(feats, signs: dict):
    """Mean of the signed z-scores of the named features (missing values count as 0, i.e. no evidence either way). Features are z-scored over the candidates themselves, so the score is relative
    to the other candidates at the site. Returns an array; higher = more support-like."""
    cols = []
    for c, sg in signs.items():
        v = feats[c].to_numpy(float)
        ok = np.isfinite(v)
        z = np.zeros(len(v))
        if ok.sum() > 2 and v[ok].std() > 0:
            z[ok] = (v[ok] - v[ok].mean()) / v[ok].std()
        cols.append(sg * z)
    return np.mean(cols, axis=0)
