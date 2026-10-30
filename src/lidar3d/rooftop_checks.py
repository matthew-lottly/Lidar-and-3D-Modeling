"""Accuracy and sensitivity tests for the rooftop pipeline: planted roofs, split-half repeatability, footprints against Overture, sun and shadow geometry."""
from __future__ import annotations

import numpy as np

from . import rooftop as RT
from . import solar as SO


def planted_roofs(n, density, noise, el, az, dni, dhi, seed=0):
    """`n` random gable roofs (slope 12 to 40 degrees, any orientation, 7 to 12 m wide, 10 to 18 m long, eave 4 m) and flat roofs, sampled at `density` returns per m2 with `noise` m of height error, run through the
    RANSAC pipeline. Returns per-roof errors of the largest recovered plane against the true plane it overlaps most: slope (deg), aspect (deg), annual open-sky beam+diffuse irradiance (kWh/m2), and the plane count."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        flat = i % 5 == 4
        W, L = rng.uniform(7, 12), rng.uniform(10, 18)
        slope = 0.0 if flat else rng.uniform(12, 40)
        theta = rng.uniform(0, 2 * np.pi)                     # direction of the long axis
        npts = int(density * W * L)
        u, v = rng.uniform(-L / 2, L / 2, npts), rng.uniform(-W / 2, W / 2, npts)    # u along the ridge, v across
        t = np.tan(np.radians(slope))
        z = 4.0 + (W / 2 - np.abs(v)) * t if not flat else np.full(npts, 4.0)
        c, s = np.cos(theta), np.sin(theta)
        x, y = u * c - v * s, u * s + v * c
        P = np.c_[x, y, z + rng.normal(0, noise, npts)]
        planes = RT.merge_coplanar(RT.segment_roof(P, seed=i))
        if not planes:
            out.append((np.nan, np.nan, np.nan, 0, slope))
            continue
        big = max(planes, key=lambda p: len(p.pts))
        # true plane the recovered one sits on: compare with the two true planes (or the flat one)
        if flat:
            truth = [(0.0, 0.0)]
        else:
            # the plane for v > 0 faces the +v direction (down-slope away from the ridge): its facing direction is (-sin theta, cos theta)
            f1 = np.array([-s, c]); f2 = -f1
            truth = [(slope, np.degrees(np.arctan2(f[0], f[1])) % 360) for f in (f1, f2)]
        best = None
        for ts, ta in truth:
            nt = np.array([[np.sin(np.radians(ta)) * np.sin(np.radians(ts)), np.cos(np.radians(ta)) * np.sin(np.radians(ts)), np.cos(np.radians(ts))]])
            nr = SO.plane_normals([big.a], [big.b])
            ang = np.degrees(np.arccos(np.clip((nt * nr).sum(), -1, 1)))
            if best is None or ang < best[0]:
                best = (ang, ts, ta, nt, nr)
        ang, ts, ta, nt, nr = best
        dsl = abs(big.slope_deg - ts)
        da = abs((big.aspect_deg - ta + 180) % 360 - 180) if ts > 1.0 and not np.isnan(big.aspect_deg) else np.nan
        zero = np.zeros((1, 72), np.float32)
        e_est = sum(SO.annual_irradiance(zero, nr, el, az, dni, dhi))
        e_true = sum(SO.annual_irradiance(zero, nt, el, az, dni, dhi))
        out.append((dsl, da, float(e_est[0] - e_true[0]), len(planes), slope))
    return np.array(out, float)


def split_half(B, C_pts_by_building, seed=0):
    """Repeatability on real roofs: split each building's returns at random into two halves, fit planes to each, and compare the largest plane of half A with the closest-oriented plane of half B (tilt of the normals, and slope difference of the two largest)."""
    rng = np.random.default_rng(seed)
    rows = []
    for bid, P in C_pts_by_building.items():
        if len(P) < 150:
            continue
        m = rng.random(len(P)) < 0.5
        pl, allb = [], []
        for sel in (m, ~m):
            ps = RT.merge_coplanar(RT.segment_roof(P[sel], seed=bid))
            if not ps:
                break
            pl.append(max(ps, key=lambda p: len(p.pts)))
            allb.append(ps)
        if len(pl) < 2:
            continue
        pl = [pl[0], pl[1], allb[1]]
        n1 = SO.plane_normals([pl[0].a], [pl[0].b])
        ang = 90.0
        for q in pl[2]:                         # the largest plane of half A against the closest plane of half B (a gable's two slopes are equally large, so "the largest" alone is ambiguous)
            n2 = SO.plane_normals([q.a], [q.b])
            ang = min(ang, float(np.degrees(np.arccos(np.clip((n1 * n2).sum(), -1, 1)))))
        rows.append((abs(pl[0].slope_deg - pl[1].slope_deg), ang, len(P)))
    return np.array(rows)


def footprint_check(polys, overture, min_area=30.0):
    """Lidar-derived footprints against Overture's: for each lidar footprint the best IoU with an Overture polygon, and the share of Overture buildings (30 m2 or more) that a lidar footprint covers by at least half."""
    from shapely.strtree import STRtree
    ov = [g for g in overture if g.area >= min_area]
    tree = STRtree(ov)
    ious = []
    for p in polys:
        best = 0.0
        for j in tree.query(p):
            q = ov[j]
            inter = p.intersection(q).area
            if inter > 0:
                best = max(best, inter / (p.area + q.area - inter))
        ious.append(best)
    from shapely.ops import unary_union
    lu = unary_union(polys)
    cov = [lu.intersection(q).area / q.area for q in ov]
    return np.array(ious), np.array(cov)
