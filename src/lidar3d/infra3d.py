"""Turning fitted spans into supports and 3D geometry (poles, lattice towers, cross-arms, insulators, wire tubes) and placing them in a lidar and imagery scene.

Support heights, positions and the height and width of every cross-arm come from where the fitted wires attach. Pole radius, tower taper and insulator length are assumed typical proportions, not measurements.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import trimesh

from . import spans as SP


def supports_from_spans(A, res, ground, min_wires=2, tier_gap=1.3):
    """One support per anchor with at least `min_wires` fitted wire ends attached. `ground(x, y)` gives ground height.
    Returns a DataFrame (x, y, ground_z, top_z, height, n_wires, width, kind, azimuth of the line) and a list of cross-arm tiers per support."""
    att = defaultdict(list)
    for i, j, w in res:
        L = w["L"]; u = (w["b"] - w["a"]) / L; nrm = np.array([-u[1], u[0]])
        att[i].append((w["a"] + w["side0"] * nrm, w["z_ends"][0], u, w["side0"]))
        att[j].append((w["b"] + w["side1"] * nrm, w["z_ends"][1], -u, w["side1"]))
    rows, tiers = [], []
    for k, v in att.items():
        if len(v) < min_wires:
            continue
        xy = np.array([a[0] for a in v]); z = np.array([a[1] for a in v])
        u = np.array([a[2] for a in v]); u = u * np.sign(u @ u[0] + 1e-9)[:, None]
        ax = u.mean(0); ax /= np.hypot(*ax); nrm = np.array([-ax[1], ax[0]])
        lat = (xy - xy.mean(0)) @ nrm
        c = xy.mean(0) + nrm * (lat.min() + lat.max()) / 2
        o = np.argsort(-z); tier = []; cur = [o[0]]
        for q in o[1:]:
            if z[cur[-1]] - z[q] <= tier_gap:
                cur.append(q)
            else:
                tier.append(cur); cur = [q]
        tier.append(cur)
        arms = [dict(z=float(z[t].mean()), lat0=float(((xy[t] - c) @ nrm).min()), lat1=float(((xy[t] - c) @ nrm).max()), n=len(t)) for t in tier]
        g = float(ground(*c))
        top = float(z.max()) + 1.5
        width = float(lat.max() - lat.min())
        rows.append(dict(anchor=int(k), x=float(c[0]), y=float(c[1]), ground_z=g, top_z=top, height=top - g, n_wires=len(v), width=width, kind="tower" if (top - g >= 24 or width >= 6) else "pole", axis_x=float(ax[0]), axis_y=float(ax[1])))
        tiers.append(arms)
    return pd.DataFrame(rows), tiers


def through_wire_counts(res, sup, tol=2.5, margin=8.0):
    """For every support: the number of fitted wires that pass straight over it (within `tol` m sideways, away from their own ends, with returns observed right there) without attaching.
    A real termination or cross-arm has none; an anchor that wires run unbroken across is not a support."""
    out = []
    for _, s in sup.iterrows():
        n = 0
        for i, j, w in res:
            if int(s.anchor) in (i, j):
                continue
            L = w["L"]; u = (w["b"] - w["a"]) / L; nr = np.array([-u[1], u[0]])
            d = np.array([s.x, s.y]) - w["a"]
            t = d @ u
            if margin < t < L - margin and abs(d @ nr - (w["side0"] + (w["side1"] - w["side0"]) * t / L)) <= tol and w["observed"][int(np.clip(t // 2, 0, len(w["observed"]) - 1))]:
                n += 1
        out.append(n)
    return np.array(out)


def tube(points, radius, sides=6):
    """Prism around a polyline: vertices (n*sides, 3) and triangle faces."""
    p = np.asarray(points, float)
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    ref = np.where(np.abs(t[:, 2:3]) > 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    n1 = np.cross(t, ref); n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-9)
    n2 = np.cross(t, n1)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[None, :, None] * n1[:, None] + np.sin(ang)[None, :, None] * n2[:, None]
    v = (p[:, None] + radius * ring).reshape(-1, 3)
    f = []
    for i in range(len(p) - 1):
        for s in range(sides):
            a, b = i * sides + s, i * sides + (s + 1) % sides
            c, d = a + sides, b + sides
            f += [(a, c, b), (b, c, d)]
    return v, np.array(f)


def _merge(parts, color):
    v, f, off = [], [], 0
    for pv, pf in parts:
        v.append(pv); f.append(pf + off); off += len(pv)
    m = trimesh.Trimesh(np.vstack(v), np.vstack(f), process=False)
    m.visual.face_colors = np.tile(np.array(color + (255,), np.uint8), (len(m.faces), 1))
    return m


def box(c, size, axes):
    """Oriented box centred at c; `axes` = 3x3 rows (unit vectors along the box's own x, y, z)."""
    m = trimesh.creation.box(extents=size)
    T = np.eye(4); T[:3, :3] = np.asarray(axes).T; T[:3, 3] = c
    m.apply_transform(T)
    return m.vertices.copy(), m.faces.copy()


def support_meshes(sup, tiers):
    """Meshes per material: poles (wood brown), lattice towers (galvanised grey), cross-arms, insulators. Proportions are assumed typical values."""
    poles, steel, arms_p, ins = [], [], [], []
    for (_, s), arms in zip(sup.iterrows(), tiers):
        ax = np.array([s.axis_x, s.axis_y, 0.0]); nr = np.array([-ax[1], ax[0], 0.0]); up = np.array([0, 0, 1.0])
        base = np.array([s.x, s.y, s.ground_z]); H = s.top_z - s.ground_z
        if s.tier != "support":                       # a wire ends here but no unbroken wire passes: drawn as a thin post so no wire ends in mid-air
            poles.append(tube(np.c_[np.linspace(base, base + [0, 0, H], 4)], 0.14, 6))
        elif s.kind == "pole":
            poles.append(tube(np.c_[np.linspace(base, base + [0, 0, H], 6)], 0.25, 8))
        else:
            w0, w1 = 0.1 * H + 0.8, 0.5
            lv = np.linspace(0, 1, 7)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts = np.array([base + [0, 0, H * t] + (sx * nr + sy * ax) * (w0 + (w1 - w0) * t) / 2 for t in lv])
                    steel.append(tube(pts, 0.09, 4))
            for k in range(6):                                  # braces: zig-zag between legs on the two faces
                t0, t1 = lv[k], lv[k + 1]
                for sy in (-1, 1):
                    a = base + [0, 0, H * t0] + (-nr + sy * ax) * (w0 + (w1 - w0) * t0) / 2
                    b = base + [0, 0, H * t1] + (nr * (1 if k % 2 == 0 else -1) * -1 + sy * ax) * (w0 + (w1 - w0) * t1) / 2
                    steel.append(tube(np.linspace(a, b, 2), 0.04, 4))
        for a in arms:
            mid = base.copy(); mid[2] = a["z"] + (1.0 if s.kind == "tower" else 0.0)
            lat0, lat1 = a["lat0"] - 0.6, a["lat1"] + 0.6
            c = mid + nr * (lat0 + lat1) / 2
            arms_p.append(box(c, (0.25, abs(lat1 - lat0) + 0.0, 0.25), np.array([ax, nr, up])))
            for lat in np.linspace(a["lat0"], a["lat1"], max(a["n"], 1)) if a["n"] > 1 else [(a["lat0"] + a["lat1"]) / 2]:
                top = mid + nr * lat; top[2] = a["z"] + (1.0 if s.kind == "tower" else 0.0)
                bot = top.copy(); bot[2] = a["z"] + 0.15
                ins.append(tube(np.array([top, bot]), 0.07, 6))
    out = {}
    for nm, parts, col in (("poles", poles, (110, 80, 55)), ("steel", steel, (150, 155, 160)), ("arms", arms_p, (100, 75, 50)), ("insulators", ins, (186, 140, 100))):
        if parts:
            out[nm] = _merge(parts, col)
    return out


def wire_mesh(res, step=3.0, radius=0.05, color=(60, 60, 65)):
    parts = []
    for _, _, w in res:
        p, _ = SP.wire_xyz(w, step)
        if len(p) >= 2:
            parts.append(tube(p, radius, 4))
    return _merge(parts, color)


def sample_mesh(m, density):
    """Points on the mesh surface with its face colour, about `density` per m^2."""
    n = int(max(m.area * density, 10))
    pts, fi = trimesh.sample.sample_surface(m, n)
    return pts, m.visual.face_colors[fi][:, :3]


def sample_wires(res, step=0.06):
    P, F = [], []
    for _, _, w in res:
        p, f = SP.wire_xyz(w, step)
        P.append(p); F.append(f)
    return np.vstack(P), np.concatenate(F)
