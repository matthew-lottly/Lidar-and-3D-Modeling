"""Conductor model: join broken wire chains into continuous conductors, locate supports at height peaks and fill gaps with a fitted sag curve.

A wire hung between supports sags, so its height above ground is highest at the supports and lowest mid-span (a catenary; a parabola is a good approximation for the spans here).
Lidar sees only parts of it (occluded by trees, thin, or missed by the detector), so the observed wire is a set of fragments. Fragments of one conductor are collinear in plan and
close in height, so they can be grouped, the height profile along the group read for peaks (supports), and each span between peaks fitted with a parabola that bridges the gaps.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def chain_geometry(xyz: np.ndarray):
    """Principal axis of a wire chain in plan: centroid, unit direction (angle folded to 0..pi), and the along-axis range."""
    c = xyz[:, :2].mean(0)
    u, s, vt = np.linalg.svd(xyz[:, :2] - c, full_matrices=False)
    d = vt[0]
    if d[1] < 0 or (abs(d[1]) < 1e-9 and d[0] < 0):
        d = -d
    a = (xyz[:, :2] - c) @ d
    return dict(centroid=c, direction=d, s_min=float(a.min()), s_max=float(a.max()), n=len(xyz))


def group_chains(chains: list, max_angle_deg=4.0, max_lateral=1.5, max_gap=60.0, max_dz=6.0):
    """Union-find over chains: two fragments belong to the same conductor when their axes differ by at most `max_angle_deg`, one lies within `max_lateral` metres of the other's axis line, the gap
    between their ends along the axis is at most `max_gap`, and their mean heights differ by at most `max_dz`. `chains` is a list of (N, 3) arrays (x, y, hag). Returns a group label per chain."""
    g = [chain_geometry(c) for c in chains]
    n = len(chains)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cosmax = np.cos(np.radians(max_angle_deg))
    for i in range(n):
        for j in range(i + 1, n):
            di, dj = g[i]["direction"], g[j]["direction"]
            if abs(di @ dj) < cosmax:
                continue
            # lateral distance of each chain's centroid from the other's axis line
            d_ij = abs(_cross2(di, g[j]["centroid"] - g[i]["centroid"]))
            d_ji = abs(_cross2(dj, g[i]["centroid"] - g[j]["centroid"]))
            if max(d_ij, d_ji) > max_lateral:
                continue
            # gap along the shared axis
            ai = (chains[i][:, :2] - g[i]["centroid"]) @ di
            aj = (chains[j][:, :2] - g[i]["centroid"]) @ di
            gap = max(aj.min() - ai.max(), ai.min() - aj.max(), 0.0)
            if gap > max_gap:
                continue
            if abs(chains[i][:, 2].mean() - chains[j][:, 2].mean()) > max_dz:
                continue
            parent[find(i)] = find(j)
    roots = {}
    return np.array([roots.setdefault(find(i), len(roots)) for i in range(n)])


def profile(points: np.ndarray, bin_m=2.0):
    """Reduce a conductor's points (x, y, z) to a height profile along its fitted axis: returns (axis point, direction, s of each bin centre, median z per bin or NaN, lateral offset per bin)."""
    c = points[:, :2].mean(0)
    u, s, vt = np.linalg.svd(points[:, :2] - c, full_matrices=False)
    d = vt[0]
    a = (points[:, :2] - c) @ d
    lat = (points[:, :2] - c) @ np.array([-d[1], d[0]])
    edges = np.arange(a.min(), a.max() + bin_m, bin_m)
    idx = np.digitize(a, edges) - 1
    z = np.full(len(edges) - 1, np.nan)
    t = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            z[b] = np.median(points[m, 2])
            t[b] = np.median(lat[m])
    return c, d, 0.5 * (edges[:-1] + edges[1:]), z, t


def find_supports(s, z, min_prominence=0.6, min_spacing=25.0, bin_m=2.0, smooth_bins=3):
    """Supports at the peaks of the height profile. Peaks are searched on the profile with its gaps linearly bridged and lightly smoothed, need a prominence of `min_prominence` metres
    and are at least `min_spacing` apart. The ends of the conductor are included as supports when they are higher than their neighbours (a wire ends at a support). Returns indices into s."""
    ok = np.isfinite(z)
    if ok.sum() < 5:
        return np.array([], int)
    zz = np.interp(s, s[ok], z[ok])
    zz = ndimage.uniform_filter1d(zz, smooth_bins, mode="nearest")
    pk, _ = signal.find_peaks(zz, prominence=min_prominence, distance=max(int(min_spacing / bin_m), 1))
    ends = []
    k = max(int(10.0 / bin_m), 2)
    if len(zz) > 2 * k:
        if zz[0] >= zz[1:k + 1].max() and ok[:k].any():
            ends.append(0)
        if zz[-1] >= zz[-k - 1:-1].max() and ok[-k:].any():
            ends.append(len(zz) - 1)
    return np.unique(np.r_[pk, ends]).astype(int)


def fit_spans(s, z, supports, max_rms=0.8, min_fill=0.35, max_span=450.0):
    """Between consecutive supports fit z = a + b s + c s^2 to the observed bins and fill the gaps. A span is filled only when at least `min_fill` of its bins are observed, it is no longer than
    `max_span` and the fit's rms is at most `max_rms` metres. Returns (z_filled, flag per bin: 0 gap left, 1 observed, 2 interpolated, spans list of dict)."""
    out = np.full(len(s), np.nan)
    flag = np.zeros(len(s), int)
    spans = []
    ok = np.isfinite(z)
    bounds = list(supports)
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = np.arange(a, b + 1)
        obs = seg[ok[seg]]
        length = float(s[b] - s[a])
        info = dict(s0=float(s[a]), s1=float(s[b]), length=length, observed_share=float(len(obs) / len(seg)), rms=None, filled=False)
        if len(obs) >= 4 and length <= max_span:
            co = np.polyfit(s[obs], z[obs], 2)
            res = z[obs] - np.polyval(co, s[obs])
            info["rms"] = float(np.sqrt(np.mean(res ** 2)))
            info["sag"] = float(0.5 * (np.polyval(co, s[a]) + np.polyval(co, s[b])) - np.polyval(co, 0.5 * (s[a] + s[b])))
            if info["observed_share"] >= min_fill and info["rms"] <= max_rms and co[0] > -1e-9:         # a hanging wire is concave up (c >= 0)
                out[seg] = np.polyval(co, s[seg])
                info["filled"] = True
        spans.append(info)
        if info["filled"]:
            flag[seg] = np.where(ok[seg], 1, 2)
    flag[(flag == 0) & ok] = 1
    out[ok & ~np.isfinite(out)] = z[ok & ~np.isfinite(out)]
    return out, flag, spans


def build_conductors(chains: list, hag_chains: list, **kw):
    """Run the whole model. `chains` = list of (N, 3) arrays of (x, y, z); `hag_chains` = list of (N,) heights above ground. Returns a list of conductors, each a dict with the axis, supports (x, y),
    the filled height profile with flags, the spans, and counts of observed and interpolated bins."""
    xyh = [np.c_[c[:, :2], h] for c, h in zip(chains, hag_chains)]
    lab = group_chains(xyh, **{k: v for k, v in kw.items() if k in ("max_angle_deg", "max_lateral", "max_gap", "max_dz")})
    out = []
    for g in np.unique(lab):
        members = np.flatnonzero(lab == g)
        pts = np.vstack([np.c_[chains[i][:, :2], hag_chains[i]] for i in members])
        if len(pts) < 8:
            continue
        c, d, s, z, t = profile(pts)
        sup = find_supports(s, z)
        if len(sup) < 2:
            sup = np.array([0, len(s) - 1])
        zf, flag, spans = fit_spans(s, z, sup)
        lat = np.nanmedian(t)
        nrm = np.array([-d[1], d[0]])
        xy = c[None, :] + s[:, None] * d[None, :] + (np.where(np.isfinite(t), t, lat))[:, None] * nrm[None, :]
        out.append(dict(members=members.tolist(), xy=xy, s=s, z_obs=z, z_fill=zf, flag=flag, supports=sup, support_xy=xy[sup], spans=spans,
                        observed_bins=int((flag == 1).sum()), interpolated_bins=int((flag == 2).sum()), length=float(s[-1] - s[0])))
    return out


def merge_supports(support_lists: list, radius=6.0):
    """Supports from several conductors that fall within `radius` metres are one physical support (a pole carries all its wires). Returns an array (n, 3): x, y, number of conductors that peak there."""
    pts = [p for sl in support_lists for p in np.asarray(sl)]
    if not pts:
        return np.zeros((0, 3))
    P = np.array(pts)
    used = np.zeros(len(P), bool)
    out = []
    for i in range(len(P)):
        if used[i]:
            continue
        d = np.hypot(*(P - P[i]).T)
        grp = (d <= radius) & ~used
        used |= grp
        out.append((*P[grp].mean(0), int(grp.sum())))
    return np.array(out)


def lateral_conductors_near(conductors: list, line_xy: np.ndarray, max_dist=6.0, min_overlap=20.0):
    """Number of distinct conductors whose observed points lie within `max_dist` of a mapped line and run along it for at least `min_overlap` metres: a check on the line's cable count. `line_xy` is (n, 2)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(line_xy)
    n = 0
    for c in conductors:
        obs = c["xy"][c["flag"] == 1]
        if len(obs) == 0:
            continue
        d, _ = tree.query(obs)
        if (d <= max_dist).sum() * 2.0 >= min_overlap:
            n += 1
    return n
