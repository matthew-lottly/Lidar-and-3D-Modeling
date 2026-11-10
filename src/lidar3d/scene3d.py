"""Mesh builders for the classified and realistic scenes: tree crowns, buildings with roof planes, substation equipment shaped from the measured envelope (transformer, tall apparatus, cabinet), yard fence and ground. All return
PyVista polydata with per-vertex 'rgb' so one call draws thousands of objects. Sizes come from the lidar; shapes are templates and are labelled as such wherever they are shown."""
from __future__ import annotations

import numpy as np
import pyvista as pv


def _poly(verts, faces, rgb):
    f = np.c_[np.full(len(faces), 3), faces].ravel()
    p = pv.PolyData(np.asarray(verts, float), f)
    p["rgb"] = np.asarray(rgb, np.uint8)
    return p


def merge(polys):
    polys = [p for p in polys if p is not None and p.n_points]
    if not polys:
        return None
    return pv.merge(polys, merge_points=False) if len(polys) > 1 else polys[0]


# ------------------------------------------------------------------------------------------------ primitives
def _unit_sphere(nt=10, np_=7):
    s = pv.Sphere(radius=1.0, theta_resolution=nt, phi_resolution=np_).triangulate()
    return np.asarray(s.points), s.faces.reshape(-1, 4)[:, 1:]


def _unit_cone(n=10):
    c = pv.Cone(center=(0, 0, 0), direction=(0, 0, 1), height=1.0, radius=1.0, resolution=n).triangulate()
    return np.asarray(c.points), c.faces.reshape(-1, 4)[:, 1:]


def _cyl(c, r, h, n=8, axis="z"):
    cy = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=r, height=h, resolution=n, capping=True).triangulate()
    v = np.asarray(cy.points).copy()
    if axis == "x":
        v = v[:, [2, 1, 0]]
    elif axis == "y":
        v = v[:, [0, 2, 1]]
    return v + c, cy.faces.reshape(-1, 4)[:, 1:]


def _box(c, size):
    b = pv.Box(bounds=(-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2, -size[2] / 2, size[2] / 2)).triangulate()
    return np.asarray(b.points) + c, b.faces.reshape(-1, 4)[:, 1:]


def _rot(v, c, yaw):
    ca, sa = np.cos(yaw), np.sin(yaw)
    d = v - c
    return np.c_[c[0] + ca * d[:, 0] - sa * d[:, 1], c[1] + sa * d[:, 0] + ca * d[:, 1], v[:, 2]]


# ------------------------------------------------------------------------------------------------ trees
def trees_mesh(T, ground_at, colors, trunk_color=(70, 52, 38)):
    """Each tree: a short trunk up to the crown base and a crown: an ellipsoid, or three stacked cones when the crown is long and narrow (conifer-shaped). The crown radius is the measured crown radius, widened a little because the
    watershed crown is cut at 40% of the tree's height, and never narrower than a fixed share of the crown length. T: dict of x, y, height, radius, base arrays; colors (n, 3) uint8 per tree."""
    sv, sf = _unit_sphere()
    cv, cf = _unit_cone()
    V, F, C = [], [], []
    off = 0
    gz = ground_at(T["x"], T["y"])
    for i in range(len(T["x"])):
        h = float(T["height"][i])
        b = float(min(T["base"][i], 0.6 * h))
        L = max(h - b, 1.5)
        r = max(float(T["radius"][i]) * 1.25, 0.6)
        conifer = L / max(2 * r, 0.5) > 1.6
        z0 = gz[i]
        cx, cy = float(T["x"][i]), float(T["y"][i])
        col = colors[i]
        if conifer:
            r = max(r, 0.2 * L)
            for k in range(3):
                seg_h = L * 0.45
                zc = z0 + b + L * 0.27 * k + seg_h / 2
                v = cv * np.array([r * (1.0 - 0.28 * k), r * (1.0 - 0.28 * k), seg_h]) + np.array([cx, cy, zc])
                f = cf
                tone = np.clip(col.astype(float) * (0.92 + 0.06 * k), 0, 255).astype(np.uint8)
                V.append(v); F.append(f + off); C.append(np.tile(tone, (len(v), 1))); off += len(v)
        else:
            r = max(r, 0.3 * L)
            v = sv * np.array([r, r, L / 2]) + np.array([cx, cy, z0 + b + L / 2])
            V.append(v); F.append(sf + off); C.append(np.tile(col, (len(v), 1))); off += len(v)
        if b > 1.2:
            tv, tf_ = _cyl(np.array([cx, cy, z0 + b / 2]), 0.07 + 0.008 * h, b, 5)
            V.append(tv); F.append(tf_ + off); C.append(np.tile(np.array(trunk_color, np.uint8), (len(tv), 1))); off += len(tv)
    return _poly(np.vstack(V), np.vstack(F), np.vstack(C))


# ------------------------------------------------------------------------------------------------ buildings
def buildings_mesh(blds, offset, roof_color, wall_color=(206, 200, 190)):
    """All buildings merged (rooftop.building_mesh per building, then shifted by `offset` (x, y) into the scene's local frame). roof_color(building, plane_index) -> RGB."""
    from . import rooftop as RT
    V, F, C = [], [], []
    off = 0
    for b in blds:
        v, f, c = RT.building_mesh(b, plane_color=lambda j, b=b: roof_color(b, j), wall_color=wall_color)
        if not len(v):
            continue
        v = v + np.array([offset[0], offset[1], 0.0])
        V.append(v); F.append(f + off); C.append(c); off += len(v)
    return _poly(np.vstack(V), np.vstack(F), np.vstack(C)) if V else None


# ------------------------------------------------------------------------------------------------ substation equipment
GREY_GREEN, PORCELAIN, STEEL, DARK = (150, 160, 154), (196, 168, 140), (150, 156, 162), (52, 56, 60)


def equipment_templates(blocks):
    """A template per equipment block, scaled to its measured box (size along yaw, across, height; `centre` z is the box centre in the scene frame):
    large and low -> a transformer (tank, two radiator banks with fins, conservator, three bushings); tall and narrow -> a porcelain column apparatus (stacked insulators and a head); everything else -> a cabinet or control house
    (box with a slightly overhanging roof and a door). Returns (polydata, list of dict(kind, size, centre)). The envelope is measured; the parts are assumed."""
    V, F, C = [], [], []
    off = 0
    kinds = []

    def add(v, f, col, c0, yaw):
        nonlocal off
        v = _rot(v, c0, yaw)
        V.append(v); F.append(f + off); C.append(np.tile(np.array(col, np.uint8), (len(v), 1))); off += len(v)

    for b in blocks:
        L, W, H = float(b["size"][0]), float(b["size"][1]), float(b["size"][2])
        cx, cy, cz = [float(v) for v in b["centre"]]
        z0 = cz - H / 2
        c0 = np.array([cx, cy, cz])
        yaw = float(b["yaw"])
        if L < W:
            L, W, yaw = W, L, yaw + np.pi / 2
        if L >= 6.0 and W >= 3.0 and 2.5 <= H <= 8.0:
            kinds.append("transformer")
            tank = (L * 0.62, W * 0.72, H * 0.62)
            add(*_box(np.array([cx, cy, z0 + 0.35 + tank[2] / 2]), tank), GREY_GREEN, c0, yaw)
            for s in (-1, 1):
                rb = (L * 0.5, 0.35, H * 0.55)
                add(*_box(np.array([cx, cy + s * (tank[1] / 2 + 0.45), z0 + 0.5 + rb[2] / 2]), rb), GREY_GREEN, c0, yaw)
                for k in np.linspace(-0.45, 0.45, 8):
                    add(*_box(np.array([cx + k * rb[0], cy + s * (tank[1] / 2 + 0.7), z0 + 0.5 + rb[2] / 2]), (0.06, 0.45, rb[2] * 0.95)), (130, 140, 135), c0, yaw)
            add(*_cyl(np.array([cx - L * 0.12, cy, z0 + 0.35 + tank[2] + 0.45]), 0.42, tank[0] * 0.55, 10, "x"), GREY_GREEN, c0, yaw)
            for k in (-1, 0, 1):
                add(*_cyl(np.array([cx + k * tank[0] * 0.3, cy - 0.25, z0 + 0.35 + tank[2] + 0.65]), 0.17, 1.3, 7), PORCELAIN, c0, yaw)
            add(*_box(np.array([cx, cy, z0 + 0.17]), (L * 0.9, W * 0.95, 0.35)), (110, 112, 114), c0, yaw)
        elif max(L, W) < 4.5 and H >= 3.0:
            kinds.append("column apparatus")
            nseg = max(int(H / 0.8), 3)
            add(*_cyl(np.array([cx, cy, z0 + 0.15]), max(W, 0.7) * 0.5, 0.3, 8), STEEL, c0, yaw)
            for k in range(nseg):
                add(*_cyl(np.array([cx, cy, z0 + 0.3 + (k + 0.5) * (H - 0.9) / nseg]), 0.16 + 0.03 * (k % 2), (H - 0.9) / nseg * 0.92, 8), PORCELAIN, c0, yaw)
            add(*_box(np.array([cx, cy, z0 + H - 0.3]), (min(L, 1.6), min(W, 0.7), 0.6)), STEEL, c0, yaw)
        else:
            kinds.append("cabinet or small building")
            add(*_box(np.array([cx, cy, z0 + H / 2]), (L, W, H)), (188, 190, 186), c0, yaw)
            add(*_box(np.array([cx, cy, z0 + H + 0.08]), (L + 0.4, W + 0.4, 0.16)), (128, 130, 134), c0, yaw)
            add(*_box(np.array([cx + L / 2 + 0.02, cy, z0 + min(H, 2.2) / 2]), (0.06, min(W * 0.35, 1.0), min(H, 2.2))), DARK, c0, yaw)
    return (_poly(np.vstack(V), np.vstack(F), np.vstack(C)) if V else None), kinds


def fence_mesh(ring_xy, ground_at, height=2.4, post_gap=3.0):
    """Chain-link style fence along a polygon ring: posts every `post_gap` m, and top, middle and bottom rails."""
    from . import infra3d
    ring = np.asarray(ring_xy, float)
    parts = []
    seg = np.hypot(*np.diff(ring, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1], post_gap)
    px = np.interp(t, s, ring[:, 0]); py = np.interp(t, s, ring[:, 1])
    pz = ground_at(px, py)
    for x, y, z in zip(px, py, pz):
        parts.append(infra3d.tube(np.array([[x, y, z], [x, y, z + height]]), 0.04, 4))
    for hh in (0.25, height * 0.5, height):
        for k in range(len(px) - 1):
            parts.append(infra3d.tube(np.array([[px[k], py[k], pz[k] + hh], [px[k + 1], py[k + 1], pz[k + 1] + hh]]), 0.015, 3))
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v); F.append(f + off); off += len(v)
    return _poly(np.vstack(V), np.vstack(F), np.tile(np.array([120, 124, 128], np.uint8), (off, 1)))
