"""Support-anchored span fitting: hypothesise a span between two supports and test it against every elevated lidar return, instead of growing wires from fragments.

A wire between two supports is straight in plan (a line in the along-span / sideways plane, which may slant because the cross-arms differ at the two ends) and a parabola in height.
Looking for that pair of shapes in a corridor around the support pair finds sparse or broken wires that a local-shape detector drops, and every accepted span is complete by construction:
the part that was not seen is filled from the fitted curve, which is anchored by the observed parts and by the end heights.
Only lidar returns are used. Mapped features are never read here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def candidate_points(xyz, hag, iso_tree=None, min_hag=5.0, max_hag=60.0, iso_radius=3.0, iso_max=80):
    """Elevated returns that are not part of a dense mass (roofs, canopy): fewer than `iso_max` neighbours within `iso_radius`."""
    m = (hag >= min_hag) & (hag <= max_hag)
    P = xyz[m]
    n = cKDTree(P).query_ball_point(P, iso_radius, return_length=True)
    keep = n <= iso_max
    return P[keep], np.flatnonzero(m)[keep]


def _sideways_lines(s, t, L, grid, tol):
    """Best straight (sideways = t0 + (t1 - t0) s / L) lines by number of points within `tol`. Returns [(t0, t1, count)], best first, non-redundant."""
    T0, T1 = np.meshgrid(grid, grid, indexing="ij")
    T0, T1 = T0.ravel(), T1.ravel()
    cnt = np.zeros(len(T0), int)
    for i in range(0, len(T0), 64):
        pred = T0[i:i + 64, None] + (T1[i:i + 64] - T0[i:i + 64])[:, None] * (s[None] / L)
        cnt[i:i + 64] = (np.abs(t[None] - pred) <= tol).sum(1)
    return T0, T1, cnt


def _fit_parabola(s, z, tol=0.35, iters=250, rng=None):
    """Robust parabola z = c0 + c1 s + c2 s^2 by RANSAC then least squares on the inliers. Returns (coefficients, inlier mask) or (None, None)."""
    if len(s) < 8:
        return None, None
    rng = rng or np.random.default_rng(0)
    best, bm = 0, None
    for _ in range(iters):
        i = rng.choice(len(s), 3, replace=False)
        try:
            c = np.polyfit(s[i], z[i], 2)
        except Exception:
            continue
        m = np.abs(np.polyval(c, s) - z) <= tol
        if m.sum() > best:
            best, bm = m.sum(), m
    if bm is None or best < 8:
        return None, None
    c = np.polyfit(s[bm], z[bm], 2)
    m = np.abs(np.polyval(c, s) - z) <= tol
    c = np.polyfit(s[m], z[m], 2)
    m = np.abs(np.polyval(c, s) - z) <= tol
    return c, m


def fit_pair(P, a, b, hag_fn, corridor=6.0, tol_side=0.6, tol_z=0.35, bin_m=2.0, min_cov=0.5, min_n=25, max_wires=9, end_hag=(4.0, 60.0), end_margin=4.0, rng=None):
    """Wires between supports a and b (xy). P = elevated candidate returns (n, 3). Returns a list of dicts: s, side0, side1, c (z coefficients), coverage, n, observed mask per bin."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.hypot(*(b - a)))
    u = (b - a) / L
    nrm = np.array([-u[1], u[0]])
    d = P[:, :2] - a
    s = d @ u
    t = d @ nrm
    m = (s > end_margin) & (s < L - end_margin) & (np.abs(t) <= corridor)
    if m.sum() < min_n:
        return []
    Q = P[m]; s, t = s[m], t[m]
    out = []
    grid = np.arange(-corridor + 1.0, corridor - 1.0 + 1e-6, 0.5)
    nb = max(int(L // bin_m), 1)
    for _ in range(max_wires):
        if len(s) < min_n:
            break
        T0, T1, cnt = _sideways_lines(s, t, L, grid, tol_side)
        k = int(np.argmax(cnt))
        if cnt[k] < min_n:
            break
        pred = T0[k] + (T1[k] - T0[k]) * s / L
        inl = np.abs(t - pred) <= tol_side
        zz = Q[inl, 2]; ss = s[inl]
        for _ in range(4):                                  # several conductors can share a plan line (stacked circuits)
            c, mm = _fit_parabola(ss, zz, tol_z, rng=rng)
            if c is None:
                break
            ze = np.polyval(c, [0.0, L])
            zm = np.polyval(c, L / 2)
            sag = (ze.mean() - zm)
            ok = c[2] > 0 and 0.15 <= sag <= 0.06 * L + 1.0 and np.all(ze > 0)
            if ok:
                bins = np.unique(np.clip((ss[mm] // bin_m).astype(int), 0, nb - 1))
                cov = len(bins) / nb
                if cov >= min_cov and mm.sum() >= min_n:
                    obs = np.zeros(nb, bool); obs[bins] = True
                    out.append(dict(a=a, b=b, L=L, side0=float(T0[k]), side1=float(T1[k]), c=c, coverage=float(cov), n=int(mm.sum()), observed=obs, pts=np.c_[ss[mm], zz[mm]], sag=float(sag), z_ends=ze, z_mid=float(zm)))
            ss = np.delete(ss, np.flatnonzero(mm)); zz = np.delete(zz, np.flatnonzero(mm))
            if not ok:
                break
        keep = np.ones(len(s), bool)
        keep[np.flatnonzero(inl)] = False                   # these plan-line points are spent either way, so the next plan line is a different one
        Q, s, t = Q[keep], s[keep], t[keep]
    return out


def wire_xyz(w, step=1.0):
    """Points along an accepted span: (n, 3) and a flag per point (1 = this stretch had returns, 2 = filled by the curve)."""
    L = w["L"]
    ss = np.arange(0.0, L + 1e-9, step)
    u = (w["b"] - w["a"]) / L
    nrm = np.array([-u[1], u[0]])
    side = w["side0"] + (w["side1"] - w["side0"]) * ss / L
    xy = w["a"][None] + ss[:, None] * u[None] + side[:, None] * nrm[None]
    z = np.polyval(w["c"], ss)
    nb = len(w["observed"])
    bi = np.clip((ss // 2.0).astype(int), 0, nb - 1)
    return np.c_[xy, z], np.where(w["observed"][bi], 1, 2)


def pairs(sup_xy, min_len=20.0, max_len=450.0, max_neighbours=6):
    """Support pairs to test: each support with its nearest others within range."""
    t = cKDTree(sup_xy)
    out = set()
    for i, p in enumerate(sup_xy):
        d, j = t.query(p, k=min(max_neighbours + 1, len(sup_xy)))
        for dd, jj in zip(np.atleast_1d(d), np.atleast_1d(j)):
            if jj != i and min_len <= dd <= max_len:
                out.add((min(i, jj), max(i, jj)))
    return sorted(out)


def anchor_candidates(sag_supports, objects_xy, chain_ends, radius=8.0):
    """Places a wire can be anchored, from lidar only: peaks of conductor height profiles, compact tall objects, and the ends of detected wire chains. Anchors within `radius` metres are merged."""
    pts = np.vstack([np.asarray(a, float).reshape(-1, 2) for a in (sag_supports, objects_xy, chain_ends) if len(a)])
    used = np.zeros(len(pts), bool)
    out = []
    for i in range(len(pts)):
        if used[i]:
            continue
        d = np.hypot(*(pts - pts[i]).T)
        g = (d <= radius) & ~used
        used |= g
        out.append(pts[g].mean(0))
    return np.array(out)
