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


def connect_wires(wires, anchors, ground_at, sag_k, snap_sup=4.0, join_end=3.0, tap=1.0, reach=70.0, cone=28.0, near=15.0, arm=2.2, min_keep=25.0):
    """Make every wire end terminate on something: a support, another wire's end, or a point on another wire.
    For each end, in order: (1) a support within `snap_sup` m (plan) -> the end moves onto the support's cross-arm (pole position plus the end's offset across the line, at most `arm` m, height capped at the support top);
    (2) another wire's end within `join_end` m (3D) -> the two ends are welded at their midpoint; (3) a point of another wire within `tap` m -> the end is moved onto it (a tap);
    (4) a support or wire point ahead of the end (within `cone` degrees of the wire's direction, up to `reach` m) -> a sagging span (sag = sag_k * s * (L - s)) is added to it;
    (5) a support or wire point in any direction within `near` m -> a straight drop to it. An end still free after that gets an inferred pole if its wire is at least `min_keep` m long, otherwise the wire is dropped.
    anchors: (m, 3) support tops. Returns (wires, inferred poles as (k, 3) tops, counts)."""
    ws = [np.asarray(w, float).copy() for w in wires if len(w) >= 2]
    A = np.asarray(anchors, float).reshape(-1, 3)
    at = cKDTree(A[:, :2]) if len(A) else None
    counts = dict(ends=0, support=0, weld=0, tap=0, span_ahead=0, drop_near=0, inferred_pole=0, dropped_wires=0)
    span_len = 0.0

    def dens_all(skip):
        P, I = [], []
        for j, w in enumerate(ws):
            if j == skip or w is None:
                continue
            D = densify(w, 0.5)
            P.append(D); I.append(np.full(len(D), j))
        if not P:
            return None, None, None
        P = np.vstack(P); I = np.concatenate(I)
        return P, I, cKDTree(P)

    def span(a, b):
        L = float(np.hypot(*(b[:2] - a[:2])))
        s = np.linspace(0, L, max(int(L), 2))
        t = s / max(L, 1e-9)
        xy = a[:2] + (b[:2] - a[:2]) * t[:, None]
        z = a[2] + (b[2] - a[2]) * t - (sag_k * s * (L - s) if L > 8 else 0.0)
        return np.c_[xy, z]

    free = []
    for i in range(len(ws)):
        for side in (0, 1):
            w = ws[i]
            if w is None:
                break
            counts["ends"] += 1
            e = w[0] if side == 0 else w[-1]
            nb = w[min(len(w) - 1, 5)] if side == 0 else w[max(0, len(w) - 6)]
            v = e[:2] - nb[:2]
            v = v / max(np.hypot(*v), 1e-9)
            # (1) support
            if at is not None:
                d, k = at.query(e[:2])
                if d <= snap_sup:
                    perp = np.array([-v[1], v[0]])
                    off = float(np.clip((e[:2] - A[k, :2]) @ perp, -arm, arm))
                    p = np.r_[A[k, :2] + perp * off, min(e[2], A[k, 2] + 0.3)]
                    seg = span(e, p) if np.hypot(*(p[:2] - e[:2])) > 0.6 else p[None]
                    ws[i] = np.vstack([seg[::-1], w[1:]]) if side == 0 else np.vstack([w[:-1], seg])
                    counts["support"] += 1
                    continue
            P, I, T = dens_all(i)
            if T is not None:
                # (2) weld to another end
                best = None
                for j, o in enumerate(ws):
                    if j == i or o is None:
                        continue
                    for oe_side, oe in ((0, o[0]), (1, o[-1])):
                        dd = float(np.linalg.norm(oe - e))
                        if dd <= join_end and (best is None or dd < best[0]):
                            best = (dd, j, oe_side)
                if best is not None:
                    _, j, os_ = best
                    m = (e + (ws[j][0] if os_ == 0 else ws[j][-1])) / 2
                    if side == 0:
                        ws[i][0] = m
                    else:
                        ws[i][-1] = m
                    if os_ == 0:
                        ws[j][0] = m
                    else:
                        ws[j][-1] = m
                    counts["weld"] += 1
                    continue
                # (3) tap onto another wire
                d, k = T.query(e)
                if d <= tap:
                    if side == 0:
                        ws[i][0] = P[k]
                    else:
                        ws[i][-1] = P[k]
                    counts["tap"] += 1
                    continue
            # (4) something ahead
            cand = []
            if at is not None:
                for k in at.query_ball_point(e[:2], reach):
                    g = A[k, :2] - e[:2]; L = np.hypot(*g)
                    if L > 0.5 and np.degrees(np.arccos(np.clip(v @ (g / L), -1, 1))) <= cone:
                        cand.append((L, np.r_[A[k, :2], min(A[k, 2] + 0.3, max(e[2], A[k, 2] - 3.0))]))
            if T is not None:
                for k in T.query_ball_point(e, reach)[::3]:
                    g = P[k, :2] - e[:2]; L = np.hypot(*g)
                    if L > 0.5 and np.degrees(np.arccos(np.clip(v @ (g / L), -1, 1))) <= cone and abs(P[k, 2] - e[2]) <= 0.25 * L + 2.0:
                        cand.append((L * 1.15, P[k]))
            if cand:
                L, p = min(cand, key=lambda c: c[0])
                seg = span(e, p)
                ws[i] = np.vstack([seg[::-1], ws[i][1:]]) if side == 0 else np.vstack([ws[i][:-1], seg])
                span_len += float(np.hypot(*(p[:2] - e[:2])))
                counts["span_ahead"] += 1
                continue
            # (5) anything near
            cand = []
            if at is not None:
                d, k = at.query(e[:2])
                if d <= near:
                    cand.append((d, np.r_[A[k, :2], min(e[2], A[k, 2] + 0.3)]))
            if T is not None:
                d, k = T.query(e)
                if d <= near:
                    cand.append((d, P[k]))
            if cand:
                _, p = min(cand, key=lambda c: c[0])
                seg = span(e, p)
                ws[i] = np.vstack([seg[::-1], ws[i][1:]]) if side == 0 else np.vstack([ws[i][:-1], seg])
                counts["drop_near"] += 1
                continue
            free.append((i, side))
    poles = []
    for i, side in free:
        w = ws[i]
        if w is None:
            continue
        L = float(np.hypot(*np.diff(w[:, :2], axis=0).T).sum())
        if L < min_keep:
            ws[i] = None
            counts["dropped_wires"] += 1
            continue
        e = w[0] if side == 0 else w[-1]
        poles.append([e[0], e[1], e[2] + 0.3])
        counts["inferred_pole"] += 1
    out = [w for w in ws if w is not None]
    counts["wires_out"] = len(out); counts["added_span_m"] = span_len
    return out, np.array(poles, float).reshape(-1, 3), counts


def free_ends(wires, anchors, tol_sup=2.6, tol_wire=0.6, points=False):
    """Ends that touch neither a support top (plan distance) nor any other wire (3D). A check, independent of how the ends were joined."""
    A = np.asarray(anchors, float).reshape(-1, 3)
    at = cKDTree(A[:, :2]) if len(A) else None
    D = [densify(np.asarray(w, float), 0.25) for w in wires]
    n = 0
    pts = []
    for i, w in enumerate(wires):
        others = [D[j] for j in range(len(D)) if j != i]
        T = cKDTree(np.vstack(others)) if others else None
        for e in (w[0], w[-1]):
            ok = (at is not None and at.query(e[:2])[0] <= tol_sup) or (T is not None and T.query(e)[0] <= tol_wire)
            n += 0 if ok else 1
            if not ok:
                pts.append(e)
    return (n, np.array(pts).reshape(-1, 3)) if points else n


def wire_rows(x, y, top_z, radius, eps=7.0, dz=1.2, max_r=2.0, min_n=5, lin=0.97, min_len=20.0):
    """Wire returns the wire detector missed still form ridges in the canopy height model, and the tree segmentation puts a row of narrow 'crowns' along them, all at the conductor's height.
    Find those rows: narrow tops (radius <= `max_r`) linked when within `eps` m in plan and `dz` m in height, groups of at least `min_n` whose plan spread is a line (first PCA share >= `lin`) at least `min_len` m long.
    Returns (boolean mask of tops in a row, list of (n, 3) polylines ordered along each row)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    cand = np.flatnonzero(radius <= max_r)
    P = np.c_[x[cand], y[cand]]
    tr = cKDTree(P)
    pairs = np.array(sorted(tr.query_pairs(eps)), int).reshape(-1, 2)
    if len(pairs):
        pairs = pairs[np.abs(top_z[cand][pairs[:, 0]] - top_z[cand][pairs[:, 1]]) <= dz]
    n = len(cand)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, lab = connected_components(g, directed=False)
    mask = np.zeros(len(x), bool)
    rows = []
    for l in np.unique(lab):
        idx = np.flatnonzero(lab == l)
        if len(idx) < min_n:
            continue
        Q = P[idx] - P[idx].mean(0)
        ev, evec = np.linalg.eigh(np.cov(Q.T))
        if ev[-1] / max(ev.sum(), 1e-9) < lin:
            continue
        t = Q @ evec[:, -1]
        if t.max() - t.min() < min_len:
            continue
        o = np.argsort(t)
        sel = cand[idx][o]
        mask[sel] = True
        # one smooth line per row: the PCA axis in plan, with the median offset and height in 6 m steps along it (the tops of parallel conductors interleave, so joining them in order zig-zags)
        ax = evec[:, -1]; nrm = np.array([-ax[1], ax[0]])
        off = Q @ nrm
        zz = top_z[cand][idx]
        edges = np.arange(t.min(), t.max() + 6.0, 6.0)
        tc, oc, zc = [], [], []
        for a_, b_ in zip(edges[:-1], edges[1:]):
            m_ = (t >= a_) & (t <= b_)
            if m_.any():
                tc.append(np.median(t[m_])); oc.append(np.median(off[m_])); zc.append(np.median(zz[m_]))
        tc = np.r_[t.min(), tc, t.max()]; oc = np.r_[oc[0], oc, oc[-1]]; zc = np.r_[zc[0], zc, zc[-1]]
        oc = np.convolve(np.pad(oc, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        c0 = P[idx].mean(0)
        xy = c0 + tc[:, None] * ax + oc[:, None] * nrm
        rows.append(np.c_[xy, zc])
    return mask, rows
