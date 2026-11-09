"""Close-up 3D models of utility objects from the lidar: substation yard posts, bus wires and equipment blocks, one transformer, and poles with their fitted wires. Everything is a model fitted to returns (a cylinder per post, a tube per wire,
an oriented box per equipment block); sizes are measured from the returns, shapes are assumed (box, cylinder). Meant to be looked at and measured, not as as-built drawings."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from . import infra3d, utility


def oriented_box(p):
    """Oriented box of points p (n, 3): yaw from the principal axis of the plan view, extents along (yaw, perpendicular, z). Returns dict(centre, size, yaw, z0, z1, n)."""
    xy = p[:, :2] - p[:, :2].mean(0)
    w, v = np.linalg.eigh(np.cov(xy.T) + 1e-9 * np.eye(2))
    u = v[:, 1]
    yaw = float(np.arctan2(u[1], u[0]))
    a = xy @ u
    b = xy @ np.array([-u[1], u[0]])
    lo = np.percentile(p[:, 2], 1); hi = np.percentile(p[:, 2], 99)
    cxy = p[:, :2].mean(0) + u * (a.min() + a.max()) / 2 + np.array([-u[1], u[0]]) * (b.min() + b.max()) / 2
    return dict(centre=np.array([cxy[0], cxy[1], (lo + hi) / 2]), size=np.array([a.max() - a.min(), b.max() - b.min(), hi - lo]), yaw=yaw, z0=float(lo), z1=float(hi), n=int(len(p)))


def box_part(b):
    """(vertices, faces) of the oriented box, in the form `infra3d._merge` takes."""
    m = trimesh.creation.box(extents=b["size"])
    R = trimesh.transformations.rotation_matrix(b["yaw"], [0, 0, 1])
    R[:3, 3] = b["centre"]
    m.apply_transform(R)
    return np.asarray(m.vertices), np.asarray(m.faces)


def equipment_blocks(xyz, hag, exclude, centre, half=14.0, vox=0.5, min_pts=25, min_dim=0.7, max_h=10.0):
    """Compact, bulky blocks of returns around `centre` (x, y), after removing `exclude` (posts and wires): occupied 0.5 m voxels joined in 3D, kept when at least `min_pts` returns and every box dimension is at least `min_dim` m."""
    m = (np.abs(xyz[:, 0] - centre[0]) <= half) & (np.abs(xyz[:, 1] - centre[1]) <= half) & (hag >= 0.8) & (hag <= max_h) & ~exclude
    P, H = xyz[m], hag[m]
    if len(P) < min_pts:
        return []
    o = np.array([centre[0] - half, centre[1] - half, 0.0])
    ij = np.floor((np.c_[P[:, :2], H] - np.array([o[0], o[1], 0.0])) / vox).astype(int)
    shp = ij.max(0) + 1
    occ = np.zeros(shp, np.uint8)
    np.add.at(occ, tuple(ij.T), 1)
    dense = ndimage.binary_closing(occ >= 1, iterations=1) & (ndimage.uniform_filter(occ.astype(float), 3) * 27 >= 6)
    lab, n = ndimage.label(dense, structure=np.ones((3, 3, 3)))
    out = []
    for c in range(1, n + 1):
        sel = lab[tuple(ij.T)] == c
        if sel.sum() < min_pts:
            continue
        b = oriented_box(np.c_[P[sel, :2], H[sel]])
        if b["size"].min() >= min_dim:
            b["points"] = P[sel]
            b["hag_points"] = H[sel]
            out.append(b)
    return sorted(out, key=lambda b: -b["n"])


def wire_polylines(xyz, wlab, step=2.0):
    """Each wire chain as a smooth polyline sorted along its principal axis (median height in `step` m bins)."""
    out = []
    for c in np.unique(wlab[wlab >= 0]):
        p = xyz[wlab == c]
        d = p[:, :2] - p[:, :2].mean(0)
        u = np.linalg.svd(d, full_matrices=False)[2][0]
        s = d @ u
        bins = np.floor((s - s.min()) / step).astype(int)
        pts = [np.r_[p[bins == b, :2].mean(0), np.median(p[bins == b, 2])] for b in np.unique(bins) if (bins == b).sum() >= 2]
        if len(pts) >= 3:
            out.append(np.array(pts))
    return out


def tube_mesh(polys, radius, color):
    parts = [infra3d.tube(p, radius, 4) for p in polys if len(p) >= 2]
    return infra3d._merge(parts, color) if parts else None
