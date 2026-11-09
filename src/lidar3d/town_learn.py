"""Learned layers on top of the geometric utility detectors at town scale: why mapped lines go unfitted, a learned support (pole/tower) detector scored on spatially held-out tiles, pole-top equipment, and a learned
score for candidate spans that lets lines continue through gaps. Labels come from Overture's power features, which are incomplete, so every precision and recall here is a lower bound on the truth.
Everything is evaluated out of fold: a model never scores a tile it was trained on."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

HEIGHT_BANDS = (5.0, 8.0, 11.0, 15.0, 20.0, 60.0)


class Grid:
    """Sampling helpers for a raster in local metres (x east, y north) with row 0 at the SOUTH (`south_up=True`, the ground grid) or at the NORTH (NAIP)."""

    def __init__(self, a, half, res=1.0, south_up=True):
        self.a, self.half, self.res, self.south_up = a, half, res, south_up

    def at(self, x, y, fill=np.nan):
        ix = np.floor((np.asarray(x) + self.half) / self.res).astype(int)
        iy = np.floor(((np.asarray(y) + self.half) if self.south_up else (self.half - np.asarray(y))) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.a.shape[1]) & (iy >= 0) & (iy < self.a.shape[0])
        out = np.full(np.shape(x), fill, float)
        out[ok] = self.a[iy[ok], ix[ok]]
        return out

    def window_mean(self, x, y, r):
        """Mean of the raster within `r` m of each (x, y) (square window)."""
        k = int(round(r / self.res))
        out = np.full(len(x), np.nan)
        ix = np.floor((np.asarray(x) + self.half) / self.res).astype(int)
        iy = np.floor(((np.asarray(y) + self.half) if self.south_up else (self.half - np.asarray(y))) / self.res).astype(int)
        for n in range(len(x)):
            a, b = iy[n], ix[n]
            w = self.a[max(a - k, 0):a + k + 1, max(b - k, 0):b + k + 1]
            if w.size:
                out[n] = np.nanmean(w)
        return out


def candidates(P, hag, cell=3.0, half=1500.0, min_pts=8, hmin=5.0):
    """Candidate support locations: centres of `cell` m cells holding at least `min_pts` elevated non-mass returns (height above ground `hmin` or more) within a 3x3 block of cells."""
    m = hag >= hmin
    n = int(2 * half / cell)
    ix = np.clip(((P[m, 0] + half) / cell).astype(int), 0, n - 1)
    iy = np.clip(((P[m, 1] + half) / cell).astype(int), 0, n - 1)
    c = np.bincount(iy * n + ix, minlength=n * n).reshape(n, n).astype(float)
    from scipy import ndimage
    s = ndimage.uniform_filter(c, 3, mode="constant") * 9
    jj, ii = np.nonzero(s >= min_pts)
    return np.c_[(ii + 0.5) * cell - half, (jj + 0.5) * cell - half]


def column_filter(C, P, hag, tree, r=1.5, min_pts=5, min_extent=3.0, max_spread=0.7):
    """Keep candidates whose points within `r` m form a column: at least `min_pts` returns, `min_extent` m of height range and a plan spread of at most `max_spread` m. Removes most tree crowns and roofs before any learning."""
    ball = tree.query_ball_point(C, r)
    keep = np.zeros(len(C), bool)
    for k, ids in enumerate(ball):
        if len(ids) < min_pts:
            continue
        h = hag[ids]
        if np.ptp(h) < min_extent:
            continue
        p = P[ids]
        if np.sqrt(p[:, 0].var() + p[:, 1].var()) <= max_spread:
            keep[k] = True
    return keep


def support_features(C, P, hag, tree, W, Whag, wtree, ndvi, r=2.5):
    """One row of features per candidate location C (n, 2): the point mass per height band within `r` m, spread and verticality of the column, isolation, nearby wire points, and tree cover."""
    nC = len(C)
    F = []
    names = [f"n_h{int(a)}_{int(b)}" for a, b in zip(HEIGHT_BANDS[:-1], HEIGHT_BANDS[1:])] + ["n_total", "hmax", "hmed", "extent", "spread", "vertical", "linearity", "ring_ratio", "wire_8m", "wire_25m", "wire_hag_minus_top", "ndvi_3m", "ndvi_12m", "n_r1", "spread_r1", "occupied_1m_bins", "top_is_local_max"]
    ball = tree.query_ball_point(C, r)
    ring = np.array(tree.query_ball_point(C, 6.0, return_length=True)) - np.array(tree.query_ball_point(C, 3.5, return_length=True))
    w8 = np.array(wtree.query_ball_point(C, 8.0, return_length=True)) if len(W) else np.zeros(nC)
    w25 = np.array(wtree.query_ball_point(C, 25.0, return_length=True)) if len(W) else np.zeros(nC)
    n3 = ndvi.window_mean(C[:, 0], C[:, 1], 3.0)
    n12 = ndvi.window_mean(C[:, 0], C[:, 1], 12.0)
    for k in range(nC):
        ids = ball[k]
        if len(ids) < 3:
            F.append([0] * (len(HEIGHT_BANDS) - 1) + [0, 0, 0, 0, 0, 0, 0, 0, w8[k], w25[k], np.nan, n3[k], n12[k], 0, 0, 0, 0])
            continue
        h = hag[ids]
        p = P[ids]
        cnt = np.histogram(h, HEIGHT_BANDS)[0]
        d = p - p.mean(0)
        w, v = np.linalg.eigh(d.T @ d / len(p) + 1e-9 * np.eye(3))
        top = np.percentile(h, 99)
        wh = np.nan
        if w8[k] > 0 or w25[k] > 0:
            near = wtree.query_ball_point(C[k], 15.0)
            wh = float(np.median(Whag[near]) - top) if near else np.nan
        ids1 = [i for i in ids if (P[i, 0] - C[k, 0]) ** 2 + (P[i, 1] - C[k, 1]) ** 2 <= 1.0]
        bins = len(np.unique(np.floor(h).astype(int))) / max(np.ptp(h), 1.0)
        wide = tree.query_ball_point(C[k], 6.0)
        locmax = float(top >= np.percentile(hag[wide], 99) - 0.5)
        sp1 = float(np.sqrt(P[ids1, 0].var() + P[ids1, 1].var())) if len(ids1) > 2 else np.nan
        F.append(list(cnt) + [len(ids), top, np.median(h), np.percentile(h, 95) - np.percentile(h, 5), float(np.sqrt(d[:, 0].var() + d[:, 1].var())), float(abs(v[2, 2])), float((w[2] - w[1]) / max(w[2], 1e-9)),
                              ring[k] / max(len(ids), 1), w8[k], w25[k], wh, n3[k], n12[k], len(ids1), sp1, bins, locmax])
    return np.array(F, float), names


def nms(xy, score, radius):
    """Keep the highest-scoring point within every `radius` m (greedy non-maximum suppression). Returns indices."""
    order = np.argsort(-score)
    t = cKDTree(xy)
    gone = np.zeros(len(xy), bool)
    keep = []
    for i in order:
        if gone[i]:
            continue
        keep.append(i)
        for j in t.query_ball_point(xy[i], radius):
            gone[j] = True
    return np.array(keep, int)


def match(det, ref, radius):
    """One-to-one greedy matching of detections to references within `radius` m. Returns (matched detection indices, matched reference indices, distances)."""
    if not len(det) or not len(ref):
        return np.array([], int), np.array([], int), np.array([])
    t = cKDTree(ref)
    d, j = t.query(det)
    order = np.argsort(d)
    used, mi, mj, md = set(), [], [], []
    for i in order:
        if d[i] > radius:
            break
        if j[i] in used:
            continue
        used.add(j[i]); mi.append(i); mj.append(j[i]); md.append(d[i])
    return np.array(mi, int), np.array(mj, int), np.array(md)


def pr_curve(xy, score, ref, radius, thresholds):
    """Precision and recall of NMS-thinned detections above each threshold against the reference points."""
    rows = []
    for t in thresholds:
        m = score >= t
        if not m.any():
            rows.append((t, 0, np.nan, 0.0)); continue
        k = nms(xy[m], score[m], 6.0)
        mi, mj, _ = match(xy[m][k], ref, radius)
        rows.append((t, int(len(k)), len(mi) / len(k), len(mi) / max(len(ref), 1)))
    return rows
