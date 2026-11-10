"""One consistent model of a utility corridor or yard from the lidar: a hole-free ground surface, every detected wire (fitted spans first, then the observed chains the spans did not explain, with gaps filled by a sag curve),
and every support (span supports, detected poles, towers fitted to their own returns). Built so that nothing the detectors found is silently left out, and measured: `coverage` reports how much of what was detected is in the model."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from . import conductors as C
from . import infra3d, spans, utility


# ------------------------------------------------------------------------------------------------ ground
def ground_surface(x, y, z, cls, extent, res=0.5, near_m=3.0):
    """Hole-free ground model on a `res` m grid over extent = (x0, y0, x1, y1): the median of class-2 returns per cell, holes filled by linear interpolation between the returns (nearest outside their hull), then lightly smoothed
    only where it was filled. Returns dict(z, filled (bool), dist_to_return (m), x0, y0, res). `filled` is True for cells with no return of their own; `dist_to_return` says how far an interpolated cell is from real ground."""
    x0, y0, x1, y1 = extent
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    m = cls == 2
    ix = np.clip(((x[m] - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(((y[m] - y0) / res).astype(int), 0, ny - 1)
    key = iy * nx + ix
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], z[m][order]
    uk, start = np.unique(ks, return_index=True)
    med = np.array([np.median(a) for a in np.split(zs, start[1:])])
    g = np.full(nx * ny, np.nan)
    g[uk] = med
    g = g.reshape(ny, nx)
    have = np.isfinite(g)
    dist = ndimage.distance_transform_edt(~have) * res
    jj, ii = np.nonzero(have)
    pts = np.c_[ii, jj].astype(float)
    holes = np.nonzero(~have)
    q = np.c_[holes[1], holes[0]].astype(float)
    out = g.copy()
    if len(q):
        lin = griddata(pts, g[have], q, method="linear")
        bad = ~np.isfinite(lin)
        if bad.any():
            lin[bad] = griddata(pts, g[have], q[bad], method="nearest")
        out[holes] = lin
        sm = ndimage.gaussian_filter(out, 1.5)
        out = np.where(have, out, sm)
    return dict(z=out, filled=~have, dist_to_return=dist, x0=x0, y0=y0, res=res)


def ground_fn(G):
    from scipy.interpolate import RegularGridInterpolator
    ny, nx = G["z"].shape
    f = RegularGridInterpolator((np.arange(ny) * G["res"] + G["y0"] + G["res"] / 2, np.arange(nx) * G["res"] + G["x0"] + G["res"] / 2), G["z"], bounds_error=False, fill_value=None)

    def at(px, py):
        return f(np.c_[np.atleast_1d(py), np.atleast_1d(px)])
    return at


# ------------------------------------------------------------------------------------------------ wires
def densify(P, step=0.25):
    """Points every `step` m along the polyline P (n, 3), so that distances to it are distances to the line and not to its vertices."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.c_[np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1]), np.interp(t, s, P[:, 2])]


def chain_wires(xyz, hag, wlab, ground_at, min_pts=8, step=2.0, relink=0.9, min_len=1.5):
    """Every detected wire chain as its own polyline at the position it was observed (median of the returns in `step` m bins along the chain, flag 1), plus the stretches `conductors.build_conductors` fills between collinear chains with a sag
    curve (flag 2, absolute height from the ground model). Observed chains are never merged sideways, so parallel wires stay parallel. Returns (list of dict(xyz, flag), conductors)."""
    ids = [c for c in np.unique(wlab[wlab >= 0]) if (wlab == c).sum() >= min_pts]
    chains = [xyz[wlab == c] for c in ids]
    hags = [hag[wlab == c] for c in ids]
    out = []
    # the detector joins points that are 2 m apart, which welds parallel wires into one chain; re-link at `relink` m in 3D so each wire keeps its own polyline
    wi = np.flatnonzero(wlab >= 0)
    n2, lab2 = utility.components(xyz[wi], relink)
    fine = []
    for c in range(n2):
        q = xyz[wi[lab2 == c]]
        if len(q) >= 4 and np.hypot(*np.ptp(q[:, :2], axis=0)) >= min_len:
            fine.append(q)
    for ch in fine:
        d = ch[:, :2] - ch[:, :2].mean(0)
        u = np.linalg.svd(d, full_matrices=False)[2][0]
        s_ = d @ u
        st = min(step, max((s_.max() - s_.min()) / 3.0, 0.4))
        bins = np.floor((s_ - s_.min()) / st).astype(int)
        pts = [np.r_[ch[bins == b, :2].mean(0), np.median(ch[bins == b, 2])] for b in np.unique(bins) if (bins == b).sum() >= 2]
        if len(pts) < 2:                                   # a short fragment: draw it as the line through its extreme returns
            lo, hi = ch[s_.argmin()], ch[s_.argmax()]
            pts = [lo, hi]
        out.append(dict(xyz=np.array(pts), flag=np.ones(len(pts), int)))
    cond = C.build_conductors(chains, hags)
    for c in cond:
        xy, zf, fl = c["xy"], c["z_fill"], np.asarray(c["flag"])
        run = []
        for k in range(len(xy)):
            if fl[k] == 2 and np.isfinite(zf[k]):
                run.append(k)
            if (fl[k] != 2 or not np.isfinite(zf[k]) or k == len(xy) - 1) and run:
                if len(run) >= 2:
                    idx = np.arange(max(run[0] - 1, 0), min(run[-1] + 2, len(xy)))
                    ok = np.isfinite(zf[idx])
                    idx = idx[ok]
                    out.append(dict(xyz=np.c_[xy[idx], zf[idx] + ground_at(xy[idx, 0], xy[idx, 1])], flag=np.where(fl[idx] == 2, 2, 1)))
                run = []
    return out, cond


def merge_wires(span_wires, chain_list, tol=0.8, min_len=1.5):
    """Chain samples that a fitted span already explains (within `tol` m in 3D) are dropped; what remains, split into runs of at least `min_len` m, is added as chain wires.
    span_wires: list of (n, 3) arrays. Returns (list of dict(xyz, flag, source) for the additions, share of chain length kept)."""
    if span_wires:
        S = np.vstack([densify(w, 0.5) for w in span_wires])
        tree = cKDTree(S)
    adds, kept, total = [], 0.0, 0.0
    for ch in chain_list:
        P = densify(ch["xyz"], 0.5)
        f = np.interp(np.linspace(0, 1, len(P)), np.linspace(0, 1, len(ch["flag"])), ch["flag"]).round().astype(int)
        if span_wires:
            near = tree.query(P)[0] <= tol
        else:
            near = np.zeros(len(P), bool)
        seg = np.hypot(*np.diff(P[:, :2], axis=0).T)
        total += seg.sum()
        run = []
        for k in range(len(P)):
            if not near[k]:
                run.append(k)
            if near[k] or k == len(P) - 1:
                if len(run) >= 2:
                    L = np.hypot(*np.diff(P[run, :2], axis=0).T).sum()
                    if L >= min_len:
                        adds.append(dict(xyz=P[run], flag=f[run], source="chain"))
                        kept += L
                run = []
    return adds, (kept / total if total else 0.0)


# ------------------------------------------------------------------------------------------------ supports and towers
def tower_from_returns(pts, base_xy, ground_z, bins=2.0):
    """Lattice tower fitted to its own returns: width along and across the principal axis per height bin (5th to 95th percentile of the returns), four legs through the bin corners and zig-zag braces. `pts` (n, 3) absolute.
    Returns (verts, faces) parts to merge, and the measured dict(height, base_width, top_width)."""
    h = pts[:, 2] - ground_z
    xy = pts[:, :2] - np.asarray(base_xy)
    w, v = np.linalg.eigh(np.cov(xy.T) + 1e-9 * np.eye(2))
    u = v[:, 1]
    a, b = xy @ u, xy @ np.array([-u[1], u[0]])
    top = float(np.percentile(h, 99.5))
    edges = np.arange(0.0, top + bins, bins)
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (h >= lo) & (h < hi)
        if s.sum() >= 6:
            prof.append((0.5 * (lo + hi), max(np.ptp(np.percentile(a[s], [5, 95])), 0.8), max(np.ptp(np.percentile(b[s], [5, 95])), 0.8), float(a[s].mean()), float(b[s].mean())))
    if len(prof) < 3:
        return [], dict(height=top, base_width=None, top_width=None)
    P = np.array(prof)
    # a tower is symmetric about a vertical axis and tapers: one centre, and width fitted linearly against height (not allowed to grow upward)
    ca, cb = float(np.median(P[:, 3])), float(np.median(P[:, 4]))
    wt = np.maximum(P[:, 1], P[:, 2])
    k, c0 = np.polyfit(P[:, 0], wt, 1)
    k = min(k, 0.0)
    nrm = np.array([-u[1], u[0]])
    base = np.asarray(base_xy)
    ctr = base + ca * u + cb * nrm
    H = float(P[-1, 0] + bins / 2)
    hs = np.linspace(0.0, H, max(int(H / bins) + 1, 4))
    ws = np.maximum(c0 + k * hs, 0.9)
    legs = [[], [], [], []]
    for hh, w_ in zip(hs, ws):
        for q, (sa, sb) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            legs[q].append(np.r_[ctr + sa * w_ / 2 * u + sb * w_ / 2 * nrm, ground_z + hh])
    parts = [infra3d.tube(np.array(lg), 0.09, 4) for lg in legs]
    for k_ in range(len(hs) - 1):
        for k1, k2 in ((0, 1), (1, 2), (2, 3), (3, 0)):
            lo, hi = (k1, k2) if k_ % 2 == 0 else (k2, k1)
            parts.append(infra3d.tube(np.array([legs[lo][k_], legs[hi][k_ + 1]]), 0.035, 4))
    wa = ws
    wb = ws
    return parts, dict(height=top, base_width=float(ws[0]), top_width=float(ws[-1]))


def pole_part(x, y, gz, height, radius=0.22):
    return infra3d.tube(np.array([[x, y, gz], [x, y, gz + height]]), radius, 8)


# ------------------------------------------------------------------------------------------------ measures
def coverage(wire_points, model_wires, supports_xy=None, tol=1.0):
    """Share of detected wire points within `tol` m (3D) of a model wire, and (given supports) how many model-wire ends have a support within 3 m horizontally."""
    M = np.vstack([densify(w["xyz"] if isinstance(w, dict) else w, 0.25) for w in model_wires]) if model_wires else np.zeros((0, 3))
    out = dict(model_wire_samples=int(len(M)))
    if len(M) and len(wire_points):
        d = cKDTree(M).query(wire_points)[0]
        out.update({f"share_wire_points_within_{tol:g}m": float((d <= tol).mean()), "share_within_0.5m": float((d <= 0.5).mean()), "share_within_2m": float((d <= 2.0).mean())})
    if supports_xy is not None and len(supports_xy) and model_wires:
        ends = np.array([e[:2] for w in model_wires for e in ((w["xyz"] if isinstance(w, dict) else w)[0], (w["xyz"] if isinstance(w, dict) else w)[-1])])
        dd = cKDTree(supports_xy).query(ends)[0]
        out["wire_ends"] = int(len(ends)); out["ends_with_support_within_3m"] = float((dd <= 3.0).mean()); out["ends_free_in_air"] = int((dd > 3.0).sum())
    return out


def bridge_ends(wires, supports_xy, sag_k, max_gap=90.0, min_gap=3.0, max_angle=22.0, free_m=3.0):
    """Connect free wire ends (no support within `free_m`) to the free end of another wire that continues in the same direction: the gap must be `min_gap` to `max_gap` m and both wires must point at each other within `max_angle` degrees.
    The bridge is a parabola between the two end heights with sag coefficient `sag_k` (sag = k * L^2, fitted from the observed spans). wires: list of (n, 3) arrays. Returns a list of dict(xyz, flag=2, source='bridge')."""
    st = cKDTree(supports_xy) if len(supports_xy) else None
    ends = []
    for wi, w in enumerate(wires):
        if len(w) < 2 or np.hypot(*(w[-1, :2] - w[0, :2])) < 6.0:
            continue
        for k, (e, nb) in enumerate(((w[0], w[min(len(w) - 1, 6)]), (w[-1], w[max(0, len(w) - 7)]))):
            if st is not None and st.query(e[:2])[0] <= free_m:
                continue
            v = e[:2] - nb[:2]
            n = np.hypot(*v)
            if n < 1e-6:
                continue
            ends.append((wi, k, e, v / n))
    out, used = [], set()
    for i, (wa, ka, ea, va) in enumerate(ends):
        best = None
        for j, (wb, kb, eb, vb) in enumerate(ends):
            if j <= i or wb == wa or (i in used) or (j in used):
                continue
            g = eb[:2] - ea[:2]
            L = np.hypot(*g)
            if not (min_gap <= L <= max_gap):
                continue
            g = g / L
            a1 = np.degrees(np.arccos(np.clip(va @ g, -1, 1)))
            a2 = np.degrees(np.arccos(np.clip(vb @ -g, -1, 1)))
            if a1 <= max_angle and a2 <= max_angle and (best is None or L < best[0]):
                best = (L, j)
        if best is not None:
            L, j = best
            eb = ends[j][2]
            used.update((i, j))
            s = np.linspace(0, L, max(int(L / 1.0), 3))
            t = s / L
            xy = ea[:2] + (eb[:2] - ea[:2]) * t[:, None]
            z = ea[2] + (eb[2] - ea[2]) * t - sag_k * s * (L - s)
            out.append(dict(xyz=np.c_[xy, z], flag=np.full(len(s), 2), source="bridge"))
    return out


def roughness(w):
    """RMS of the second difference of height per metre along a polyline (a straight or sagging wire is smooth; jitter raises it)."""
    P = densify(w, 1.0)
    if len(P) < 4:
        return np.nan
    return float(np.sqrt(np.mean(np.diff(P[:, 2], 2) ** 2)))


def clean_wires(wires, min_len=3.0, dup_tol=0.45, dup_share=0.7, smooth=5):
    """Tidy a set of wire polylines ((n, 3) arrays): (1) drop pieces shorter than `min_len`; (2) drop a piece when at least `dup_share` of it lies within `dup_tol` m (3D) of a longer kept piece, i.e. the same wire drawn twice
    by a span fit and an observed piece; (3) smooth heights with a running mean of `smooth` samples (1 m spacing), keeping the end points. Returns (cleaned list, dict of counts)."""
    ws = [np.asarray(w, float) for w in wires if len(w) >= 2]
    lens = [float(np.hypot(*np.diff(w[:, :2], axis=0).T).sum()) for w in ws]
    order = np.argsort(lens)[::-1]
    kept, dropped_short, dropped_dup = [], 0, 0
    trees = []
    for i in order:
        w = ws[i]
        if lens[i] < min_len:
            dropped_short += 1
            continue
        D = densify(w, 0.5)
        dup = False
        for P_, t in trees:
            if len(P_) and (t.query(D)[0] <= dup_tol).mean() >= dup_share:
                dup = True
                break
        if dup:
            dropped_dup += 1
            continue
        kept.append(w)
        trees.append((D, cKDTree(D)))
    out = []
    for w in kept:
        P = densify(w, 1.0)
        if len(P) > smooth + 2:
            k = np.ones(smooth) / smooth
            z = np.convolve(np.pad(P[:, 2], (smooth // 2, smooth // 2), mode="edge"), k, mode="valid")
            z[0], z[-1] = P[0, 2], P[-1, 2]
            P = np.c_[P[:, :2], z]
        out.append(P)
    return out, dict(before=len(ws), after=len(out), dropped_short=dropped_short, dropped_duplicate=dropped_dup, length_before_m=float(sum(lens)), length_after_m=float(sum(np.hypot(*np.diff(p[:, :2], axis=0).T).sum() for p in out)))
