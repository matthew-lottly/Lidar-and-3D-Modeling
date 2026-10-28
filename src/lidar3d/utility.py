"""Utility structures from a lidar point cloud: height above ground, local shape (linear / planar / volumetric), wires, poles and towers.

Point-wise shape comes from the eigenvalues of the covariance of each point's nearest neighbours (Demantke et al. 2011; Weinmann et al. 2015): a wire is one-dimensional (linearity near 1),
a roof or wall is two-dimensional, a tree crown is three-dimensional. No class codes are used, because the 3DEP projects here carry no wire or tower classes.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def height_above_ground(x, y, z, cls, extent, res=1.0, ground_class=2):
    """Height of every point above the lowest ground-class return in its cell, with empty cells filled from the nearest filled one. `extent` = (xmin, ymin, xmax, ymax) of the grid.
    Returns (hag, ground_grid). Terrain under buildings or canopy is interpolated, so heights under them are approximate."""
    x0, y0, x1, y1 = extent
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    ix = ((x - x0) / res).astype(int).clip(0, nx - 1)
    iy = ((y - y0) / res).astype(int).clip(0, ny - 1)
    g = np.full((ny, nx), np.inf)
    m = cls == ground_class
    np.minimum.at(g, (iy[m], ix[m]), z[m])
    g[~np.isfinite(g)] = np.nan
    idx = ndimage.distance_transform_edt(np.isnan(g), return_distances=False, return_indices=True)
    g = g[tuple(idx)]
    return z - g[iy, ix], g


def shape_features(xyz: np.ndarray, k=12, rmax=2.0):
    """Per point: linearity, planarity, scattering from the eigenvalues of the k-neighbour covariance, the unit direction of the first eigenvector, and the distance to the k-th neighbour.
    A point whose k-th neighbour is farther than `rmax` gets NaN features (too sparse to describe a local shape). Returns a dict of arrays."""
    tree = cKDTree(xyz)
    d, idx = tree.query(xyz, k=k)
    n = len(xyz)
    lin = np.full(n, np.nan); pla = np.full(n, np.nan); sca = np.full(n, np.nan); vec = np.full((n, 3), np.nan)
    ok = d[:, -1] <= rmax
    for a in range(0, n, 200000):
        b = min(a + 200000, n)
        sel = np.flatnonzero(ok[a:b]) + a
        if not len(sel):
            continue
        nb = xyz[idx[sel]]
        nb = nb - nb.mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb) / k
        w, v = np.linalg.eigh(cov)                       # ascending
        l1, l2, l3 = w[:, 2].clip(1e-12), w[:, 1].clip(0), w[:, 0].clip(0)
        lin[sel] = (l1 - l2) / l1
        pla[sel] = (l2 - l3) / l1
        sca[sel] = l3 / l1
        vec[sel] = v[:, :, 2]
    return dict(linearity=lin, planarity=pla, scattering=sca, direction=vec, kdist=d[:, -1])


def neighbour_count(xyz: np.ndarray, query: np.ndarray, radius: float) -> np.ndarray:
    """Number of points of `xyz` within `radius` (3D) of each point of `query`, itself included. A wire has open air around it, so few; a roof, wall or tree crown has many."""
    tree = cKDTree(xyz)
    return tree.query_ball_point(query, radius, return_length=True, workers=-1)


def components(xyz: np.ndarray, link: float):
    """Connected components of the graph that joins points closer than `link`. Returns (n_components, labels)."""
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(link, output_type="ndarray")
    n = len(xyz)
    g = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)


def detect_wires(xyz, hag, min_hag=4.0, max_hag=60.0, lin_min=0.9, max_dz=0.6, link=2.0, min_length=15.0, k=12, rmax=2.0, iso_radius=3.0, iso_max=80):
    """Wire points: elevated points whose neighbourhood is a line that is near-horizontal (|direction z| < max_dz), joined into chains of at least `min_length` metres. A catenary sags,
    so the direction test is loose. Points with more than `iso_max` neighbours within `iso_radius` m (roofs, canopy) are dropped. Returns (mask over the input points, component label per input point or -1, per-component dict of length and mean height above ground)."""
    el = np.flatnonzero((hag >= min_hag) & (hag <= max_hag))
    f = shape_features(xyz[el], k=k, rmax=rmax)
    cand = (f["linearity"] >= lin_min) & (np.abs(f["direction"][:, 2]) < max_dz)
    if iso_max is not None and cand.any():
        cnt = neighbour_count(xyz[el], xyz[el[cand]], iso_radius)
        keep = np.zeros(len(el), bool)
        keep[np.flatnonzero(cand)[cnt <= iso_max]] = True
        cand = keep
    ci = el[cand]
    mask = np.zeros(len(xyz), bool)
    label = np.full(len(xyz), -1)
    info = {}
    if len(ci) < 5:
        return mask, label, info
    n, lab = components(xyz[ci], link)
    for c in range(n):
        pts = xyz[ci[lab == c]]
        ext = np.ptp(pts[:, :2], axis=0)
        length = float(np.hypot(*ext))
        if len(pts) >= 8 and length >= min_length:
            mask[ci[lab == c]] = True
            label[ci[lab == c]] = c
            info[c] = dict(points=int(len(pts)), length_m=length, mean_hag=float(hag[ci[lab == c]].mean()))
    return mask, label, info


def detect_poles(xyz, hag, exclude=None, min_hag=2.5, max_hag=60.0, lin_min=0.7, min_dz=0.85, link=1.5, min_height=4.0, max_radius=1.0, k=12, rmax=1.5, ring=(1.5, 3.5), ring_ratio=2.0, ring_all=False):
    """Vertical columns: elevated points on a vertical line (|direction z| > min_dz), joined in plan and height, kept when the chain rises at least `min_height` and stays within `max_radius`
    of its axis. `exclude` is a bool mask of points to ignore (for example wire points). Columns with more than `ring_ratio` times (all columns keep their ring_ratio when `ring_all` is set and ring_ratio is None) their own point count of elevated points in a plan ring around them (inside a tree crown or beside a wall) are dropped. Returns a list of dicts: x, y, top_hag, height, points."""
    el = np.flatnonzero((hag >= min_hag) & (hag <= max_hag) & (~exclude if exclude is not None else True))
    f = shape_features(xyz[el], k=k, rmax=rmax)
    cand = (f["linearity"] >= lin_min) & (np.abs(f["direction"][:, 2]) > min_dz)
    ci = el[cand]
    out = []
    if len(ci) < 5:
        return out
    n, lab = components(xyz[ci], link)
    for c in range(n):
        s = ci[lab == c]
        p = xyz[s]
        rise = np.ptp(p[:, 2])
        if len(s) < 8 or rise < min_height:
            continue
        cxy = np.median(p[:, :2], axis=0)
        if np.hypot(*(p[:, :2] - cxy).T).max() > max_radius:
            continue
        out.append(dict(x=float(cxy[0]), y=float(cxy[1]), top_hag=float(hag[s].max()), height=float(rise), points=int(len(s)), radius=float(np.hypot(*(p[:, :2] - cxy).T).max()),
                        linearity=float(np.nanmean(f["linearity"][np.flatnonzero(cand)[lab == c]]))))
    if (ring_ratio is not None or ring_all) and out:
        # a pole stands in the open: few elevated points in a ring around its axis compared with the column itself; a trunk inside a crown does not
        elp = np.flatnonzero((hag >= min_hag) & (hag <= max_hag))
        tree = cKDTree(xyz[elp, :2])
        kept = []
        for o in out:
            a = len(tree.query_ball_point([o["x"], o["y"]], ring[1])) - len(tree.query_ball_point([o["x"], o["y"]], ring[0]))
            o["ring_ratio"] = a / max(o["points"], 1)
            if ring_ratio is None or o["ring_ratio"] <= ring_ratio:
                kept.append(o)
        out = kept
    return out


def match_points(det: np.ndarray, ref: np.ndarray, radius: float):
    """One-to-one nearest matching of detections to reference points within `radius` (greedy by distance). Returns (matched pairs [(i_det, i_ref, distance)], unmatched detection indices, unmatched reference indices)."""
    if len(det) == 0 or len(ref) == 0:
        return [], list(range(len(det))), list(range(len(ref)))
    d = np.hypot(det[:, None, 0] - ref[None, :, 0], det[:, None, 1] - ref[None, :, 1])
    pairs = []
    used_d, used_r = set(), set()
    for flat in np.argsort(d, axis=None):
        i, j = divmod(int(flat), d.shape[1])
        if d[i, j] > radius:
            break
        if i in used_d or j in used_r:
            continue
        used_d.add(i); used_r.add(j)
        pairs.append((i, j, float(d[i, j])))
    return pairs, [i for i in range(len(det)) if i not in used_d], [j for j in range(len(ref)) if j not in used_r]


def line_coverage(lines_xy: list, wire_xy: np.ndarray, radius: float, step=2.0) -> float:
    """Length-weighted share of reference polylines (each an (n, 2) array, already clipped to the study window) that have a detected wire point within `radius` in plan."""
    tree = cKDTree(wire_xy)
    covered = total = 0.0
    for ln in lines_xy:
        seg = np.hypot(*np.diff(ln, axis=0).T)
        s = np.arange(0, seg.sum(), step)
        if not len(s):
            continue
        cum = np.r_[0, np.cumsum(seg)]
        pts = np.c_[np.interp(s, cum, ln[:, 0]), np.interp(s, cum, ln[:, 1])]
        d, _ = tree.query(pts)
        covered += (d <= radius).sum() * step
        total += len(s) * step
    return float(covered / total) if total else float("nan")


def yard_structures(xyz, hag, inside: np.ndarray, wire_mask: np.ndarray, min_hag=1.5, link=1.0, min_points=12):
    """Elevated structures in a yard: connected groups of points more than `min_hag` above ground that are not wires, with plan position, height and size. `inside` is a bool mask of points in the yard."""
    sel = np.flatnonzero(inside & (hag >= min_hag) & ~wire_mask)
    if len(sel) < min_points:
        return []
    n, lab = components(xyz[sel], link)
    out = []
    for c in range(n):
        s = sel[lab == c]
        if len(s) < min_points:
            continue
        p = xyz[s]
        out.append(dict(x=float(np.median(p[:, 0])), y=float(np.median(p[:, 1])), top_hag=float(hag[s].max()), extent_m=float(np.hypot(*np.ptp(p[:, :2], axis=0))), points=int(len(s))))
    return out


def compact_objects(xyz, hag, extent, grid=2.0, radius=2.5, min_hag=3.0, max_hag=60.0, min_points=15, ring=(3.5, 6.0)):
    """Candidate tall, compact objects without assuming a line shape. On a `grid` m lattice, every location with at least `min_points` elevated points (hag between min_hag and max_hag)
    within `radius` m in plan is described by those points: count, plan spread (root of the xy covariance trace), vertical extent (5th to 95th percentile), top height, share of points in the upper
    40% of the height, vertical share of the first eigenvector, and isolation (elevated points in a plan ring around it relative to inside). A pole is tall, narrow and isolated; a tree crown is wide and
    surrounded. `extent` = (xmin, ymin, xmax, ymax). Returns a list of dicts (x, y at the point centroid)."""
    el = (hag >= min_hag) & (hag <= max_hag)
    P, H = xyz[el], hag[el]
    tree = cKDTree(P[:, :2])
    x0, y0, x1, y1 = extent
    gx, gy = np.meshgrid(np.arange(x0 + grid / 2, x1, grid), np.arange(y0 + grid / 2, y1, grid))
    cen = np.c_[gx.ravel(), gy.ravel()]
    idx = tree.query_ball_point(cen, radius)
    n_ring = np.array([len(a) for a in tree.query_ball_point(cen, ring[1])]) - np.array([len(a) for a in tree.query_ball_point(cen, ring[0])])
    out = []
    for k, ids in enumerate(idx):
        if len(ids) < min_points:
            continue
        p, h = P[ids], H[ids]
        c = p.mean(0)
        d = p - c
        cov = d.T @ d / len(p)
        w, v = np.linalg.eigh(cov)
        top = float(np.percentile(h, 99))
        ext = float(np.percentile(h, 95) - np.percentile(h, 5))
        out.append(dict(x=float(c[0]), y=float(c[1]), n=int(len(p)), spread=float(np.sqrt(cov[0, 0] + cov[1, 1])), extent=ext, top_hag=top, upper_share=float((h >= 0.6 * top).mean()),
                        vertical=float(abs(v[2, 2])), linearity=float((w[2] - w[1]) / max(w[2], 1e-9)), ring_ratio=float(n_ring[k] / max(len(p), 1))))
    return out
