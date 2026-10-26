"""The Space Needle as a surface of revolution fitted to its lidar returns, replacing the extruded footprint.

The tower is close to axisymmetric (saucer radius 20.3 to 21.7 m by compass sector; leg radius 12.6 to 15.4 m), so the model is the lidar envelope (95th percentile radius per 2 m of height) revolved about the axis.
Where the steel structure returns almost no points (116 to 150 m), the radius is interpolated and flagged.
"""
from __future__ import annotations

import numpy as np
import trimesh


def extract(cloud, footprint_geom, ground_z, reach=45.0):
    """Returns points (x, y relative to the tower axis, z above ground, original index) within `reach` m of the footprint centre, and the axis (x, y) in the cloud's CRS."""
    c = footprint_geom.centroid
    d = np.hypot(cloud.x - c.x, cloud.y - c.y)
    m = (d < reach) & (cloud.z > ground_z + 2)
    x, y, z = cloud.x[m] - c.x, cloud.y[m] - c.y, cloud.z[m] - ground_z
    hi = z > 90                                              # the upper tower defines the axis (the base is surrounded by other structures)
    ax, ay = x[hi].mean(), y[hi].mean()
    return np.c_[x - ax, y - ay, z], (c.x + ax, c.y + ay)


def profile(pts, step=2.0, nmin=20, rmax=30.0, z_base=18.0, pct=95.0, shaft_top=125.0, shaft_nmin=5, shaft_rmax=18.0):
    """Envelope radius by height. The narrow shaft below `shaft_top` returns few points per slice, so it accepts `shaft_nmin` points within `shaft_rmax` of the axis; above it (saucer, top house) `nmin` points within `rmax`. Slices with too few points are interpolated (returned flag), except above the last populated slice, where the sparse returns of the spire set a constant thin radius.
    Below z_base the surrounding pavilions contaminate the returns, so the radius is held at its z_base value. Returns dict(z, r, interpolated) with z in metres above ground, from 0 to the highest return."""
    from scipy.ndimage import median_filter
    r = np.hypot(pts[:, 0], pts[:, 1])
    z = pts[:, 2]
    top = float(z.max())
    z0s = np.arange(z_base, top, step)
    zc = np.minimum(z0s + step / 2, top)
    rad = np.full(len(zc), np.nan)
    for i, z0 in enumerate(z0s):
        shaft = z0 < shaft_top
        s = (z >= z0) & (z < z0 + step) & (r < (shaft_rmax if shaft else rmax))
        if s.sum() >= (shaft_nmin if shaft else nmin):
            rad[i] = np.percentile(r[s], pct)
    ok = np.isfinite(rad)
    last = np.flatnonzero(ok).max()
    sm = rad.copy()
    sm[ok] = median_filter(rad[ok], size=3, mode="nearest")      # drops single-slice spikes from neighbouring structures
    interp = ~ok
    out = np.interp(zc, zc[ok], sm[ok])
    tip_pts = r[(z > zc[last] + step / 2) & (r < 5)]
    tip_r = float(np.median(tip_pts)) if len(tip_pts) else 0.5
    out[last + 1:] = tip_r
    zz = np.r_[0.0, zc, top]
    rr = np.r_[out[0], out, tip_r]
    return dict(z=zz, r=rr, interpolated=np.r_[False, interp, False], top=top)


def lathe(prof, axis_xy, ground_z, segments=72):
    """Closed mesh of revolution in the cloud's CRS (absolute coordinates). Normals point outward."""
    z, r = prof["z"], prof["r"]
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    V = [np.c_[axis_xy[0] + np.outer(r[k], np.cos(th)).ravel(), axis_xy[1] + np.outer(r[k], np.sin(th)).ravel(), np.full(segments, ground_z + z[k])] for k in range(len(z))]
    V = np.vstack(V + [np.array([[axis_xy[0], axis_xy[1], ground_z + z[0]], [axis_xy[0], axis_xy[1], ground_z + z[-1]]])])
    n = len(z)
    F = []
    for k in range(n - 1):
        for j in range(segments):
            a, b = k * segments + j, k * segments + (j + 1) % segments
            c, d = (k + 1) * segments + j, (k + 1) * segments + (j + 1) % segments
            F += [[a, b, d], [a, d, c]]
    bot, top = n * segments, n * segments + 1
    for j in range(segments):
        F.append([bot, j, (j + 1) % segments][::-1][::-1])
        F.append([top, (n - 1) * segments + (j + 1) % segments, (n - 1) * segments + j])
    m = trimesh.Trimesh(V, np.array(F), process=False)
    trimesh.repair.fix_normals(m)
    return m


def fit_quality(pts, prof, tol=1.5):
    """Points above 18 m within 30 m of the axis: share inside the envelope + tol, and over slices with real returns the median gap between the envelope and the 99th-percentile point radius."""
    r = np.hypot(pts[:, 0], pts[:, 1])
    z = pts[:, 2]
    rm = np.interp(z, prof["z"], prof["r"])
    ok = (z >= 18) & (r < 30)
    gaps = []
    for k in range(1, len(prof["z"]) - 1):
        if prof["interpolated"][k]:
            continue
        s = ok & (z >= prof["z"][k] - 1) & (z < prof["z"][k] + 1)
        if s.sum() >= 20:
            gaps.append(float(np.percentile(r[s], 99) - prof["r"][k]))
    inter = prof["interpolated"][1:-1]
    return dict(points=int(ok.sum()), share_inside=float((r[ok] <= rm[ok] + tol).mean()), median_gap_to_p99_m=float(np.median(gaps)), interpolated_height_m=float(inter.sum() * 2.0), top_m=prof["top"])


def symmetry(pts, z_lo, z_hi, sectors=8, pct=95.0, rmax=30.0):
    """95th-percentile radius per compass sector for returns between z_lo and z_hi: how far from axisymmetric the tower is at that height. Returns (min, max) over sectors with at least 5 returns."""
    r = np.hypot(pts[:, 0], pts[:, 1])
    z = pts[:, 2]
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360
    m = (z >= z_lo) & (z < z_hi) & (r < rmax)
    vals = [np.percentile(r[m & (az >= a) & (az < a + 360 / sectors)], pct) for a in np.arange(0, 360, 360 / sectors) if (m & (az >= a) & (az < a + 360 / sectors)).sum() >= 5]
    return float(min(vals)), float(max(vals))
