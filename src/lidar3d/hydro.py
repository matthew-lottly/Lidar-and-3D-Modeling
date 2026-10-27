"""Lidar hydrology: own ground filter, hydro-flattening, depression filling, flat resolution, D8 flow direction and accumulation, channel extraction, flood scenarios.

All rasters are on the project grid: row 0 = north, `res` metres per cell. Pure numpy, scipy and scikit-image.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import reconstruction

# D8 neighbours: (drow, dcol) in the order used throughout (E, SE, S, SW, W, NW, N, NE)
D8 = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


# ------------------------------------------------------------------ ground classification
def lowest_grid(row, col, z, shape):
    """Lowest return per cell (NaN where empty)."""
    out = np.full(shape, np.inf)
    np.minimum.at(out, (row, col), z)
    out[~np.isfinite(out)] = np.nan
    return out


def pmf_ground(zmin: np.ndarray, res=1.0, max_window_m=33.0, dh0=0.3, slope_cap_deg=60.0, slope_sigma=8.0):
    """Progressive morphological filter (Zhang et al. 2003) on a lowest-return grid, with a local slope allowance. A cell is non-ground when it stands more than a window-dependent height above the opened surface:
    threshold = dh0 + tan(local slope) * (window - 1) * res / 2, windows 3, 5, 9, 17, 33 cells. Local slope comes from the lowest-return grid smoothed with a Gaussian (sigma cells), so steep ground gets a larger
    allowance than flat ground and a stand of shrubs on a flat bench is rejected. Returns a boolean ground mask (True = ground); empty cells are False."""
    z = np.where(np.isnan(zmin), np.nanmax(zmin), zmin)
    gy, gx = np.gradient(ndimage.gaussian_filter(ndimage.minimum_filter(z, 3), slope_sigma), res)
    tans = np.tan(np.minimum(np.arctan(np.hypot(gx, gy)), np.radians(slope_cap_deg)))
    ground = ~np.isnan(zmin)
    surf = z.copy()
    w = 3
    while w * res <= max_window_m + 1e-9:
        opened = ndimage.grey_opening(surf, size=(w, w))
        ground &= (surf - opened) <= dh0 + tans * (w - 1) * res / 2.0
        surf = opened
        w = 2 * w - 1
    return ground & ~np.isnan(zmin)


def agreement(ours: np.ndarray, zmin: np.ndarray, dtm: np.ndarray, canopy: np.ndarray, tol=0.15, off=0.5):
    """Compare our ground mask with the published terrain model (built from the published ground class). A cell's lowest return is 'at terrain' when within `tol` of the published DTM and 'above terrain'
    when more than `off` above it. Returns omission (at-terrain cells we call non-ground), commission (above-terrain cells we call ground), and the same inside canopy cells (`canopy` mask)."""
    dz = zmin - dtm
    at, above = np.abs(dz) <= tol, dz > off
    res = dict(ground_share=float(ours.mean()), omission=float((at & ~ours).sum() / max(at.sum(), 1)), commission=float((above & ours).sum() / max(above.sum(), 1)), n_at=int(at.sum()), n_above=int(above.sum()))
    ca, cb = at & canopy, above & canopy
    res.update(canopy_omission=float((ca & ~ours).sum() / max(ca.sum(), 1)), canopy_commission=float((cb & ours).sum() / max(cb.sum(), 1)), n_above_canopy=int(cb.sum()))
    return res


# ------------------------------------------------------------------ hydro-flattening
def water_mask(water_count: np.ndarray, min_area=200, close_iter=3):
    """Water body cells from the cells that contain water-class returns: close gaps, fill holes, drop small blobs. Returns (mask, labels)."""
    m = ndimage.binary_closing(water_count > 0, iterations=close_iter)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    sz = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.isin(lab, np.flatnonzero(sz >= min_area) + 1)
    lab2, n2 = ndimage.label(keep)
    return keep, lab2


def hydro_flatten(dtm: np.ndarray, mask: np.ndarray, labels: np.ndarray, water_z: np.ndarray, water_lab: np.ndarray):
    """Set every water body to one elevation: the median of its own water-class returns. `water_z`/`water_lab` are the z and component label of each water return. Returns (flattened DTM, {label: level}).
    Land cells that sit lower than the water level next to a water body are left alone; they are real banks below the surface only if the lidar says so."""
    out = dtm.copy()
    levels = {}
    for k in range(1, labels.max() + 1):
        zk = water_z[water_lab == k]
        if len(zk) < 20:
            continue
        levels[k] = float(np.median(zk))
        out[labels == k] = levels[k]
    return out, levels


# ------------------------------------------------------------------ depression filling and flats
def fill_depressions(z: np.ndarray):
    """Fill all closed depressions (pits and basins) so that every cell drains to the grid edge: morphological reconstruction by erosion from a seed that is the surface itself on the border and +inf elsewhere."""
    seed = np.full(z.shape, z.max() + 1.0)
    seed[0, :], seed[-1, :], seed[:, 0], seed[:, -1] = z[0, :], z[-1, :], z[:, 0], z[:, -1]
    return reconstruction(seed, z, method="erosion")


def _neighbour_view(z, dr, dc, fill):
    """z shifted so that out[i, j] = z[i + dr, j + dc]; cells whose neighbour is off the grid get `fill`."""
    h, w = z.shape
    out = np.full(z.shape, fill, float)
    r0, r1 = max(0, -dr), h - max(0, dr)
    c0, c1 = max(0, -dc), w - max(0, dc)
    out[r0:r1, c0:c1] = z[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def resolve_flats(z: np.ndarray, eps=1e-4):
    """Give flat areas a tiny gradient toward their outlets so D8 can route across them.
    Unresolved cells have no strictly lower neighbour. An outlet is a flat cell that does have a strictly lower neighbour (or lies on the grid edge, where water leaves).
    Every unresolved cell is raised by eps times its path length (in cell steps, through other cells of the same flat) to the nearest outlet. eps = 0.1 mm per step is far below the lidar's precision."""
    lower = np.zeros(z.shape, bool)
    equal = np.zeros(z.shape, bool)
    for dr, dc in D8:
        nb = _neighbour_view(z, dr, dc, np.inf)
        lower |= nb < z
        equal |= nb == z
    edge = np.zeros(z.shape, bool); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    unresolved = ~lower & ~edge
    outlet = equal & lower | (edge & equal)                              # plateau cells already draining somewhere
    dist = np.zeros(z.shape)
    front = outlet.copy()
    todo = unresolved & ~outlet
    k = 0
    st = np.ones((3, 3), bool)
    while todo.any():
        k += 1
        grow = ndimage.binary_dilation(front, structure=st) & todo
        # only cells of the same flat (same elevation) as an adjacent frontier cell
        same_z = np.zeros(z.shape, bool)
        for dr, dc in D8:
            same_z |= (_neighbour_view(front.astype(float), dr, dc, 0.0) > 0) & (_neighbour_view(z, dr, dc, np.nan) == z)
        grow &= same_z
        if not grow.any():
            break
        dist[grow] = k
        todo &= ~grow
        front = grow
    return z + eps * dist


# ------------------------------------------------------------------ D8
def d8_direction(z: np.ndarray, res=1.0):
    """Steepest-descent neighbour index (0..7 into D8), -1 where no neighbour is lower (only grid-edge cells and true sinks). Diagonal drops are divided by sqrt(2)."""
    h, w = z.shape
    best = np.full(z.shape, -1, np.int8)
    best_s = np.zeros(z.shape)
    pad = np.pad(z, 1, constant_values=-np.inf)                              # outside the grid counts as lower, so edge cells drain off the grid
    for k, (dr, dc) in enumerate(D8):
        nb = pad[1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w]
        s = (z - nb) / (res * (np.sqrt(2) if dr and dc else 1.0))
        s = np.where(np.isinf(s), 1e6, s)
        better = s > best_s
        best[better] = k
        best_s[better] = s[better]
    return best


def flow_accumulation(z: np.ndarray, direction: np.ndarray, weights=None):
    """Number of cells (or sum of `weights`) draining through each cell, itself included. Cells are processed from highest to lowest, so each is complete before it passes its total downstream."""
    h, w = z.shape
    acc = np.ones(z.shape) if weights is None else weights.astype(float).copy()
    order = np.argsort(-z, axis=None, kind="stable")
    d = direction.ravel()
    dr = np.array([a for a, b in D8]); dc = np.array([b for a, b in D8])
    r, c = np.divmod(order, w)
    k = d[order]
    nr, nc = r + np.where(k >= 0, dr[k], 0), c + np.where(k >= 0, dc[k], 0)
    ok = (k >= 0) & (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    tgt = np.where(ok, nr * w + nc, -1)
    flat = acc.ravel()
    for i, t in zip(order.tolist(), tgt.tolist()):
        if t >= 0:
            flat[t] += flat[i]
    return flat.reshape(z.shape)


def channels(acc: np.ndarray, threshold_cells: float):
    """Channel cells: contributing area at or above the threshold."""
    return acc >= threshold_cells


def channel_length(mask: np.ndarray, res=1.0, direction=None):
    """Approximate channel length: cells times res, with diagonal steps counted sqrt(2) when `direction` is given."""
    if direction is None:
        return float(mask.sum() * res)
    diag = np.isin(direction, (1, 3, 5, 7))
    return float(((mask & ~diag).sum() + (mask & diag).sum() * np.sqrt(2)) * res)


# ------------------------------------------------------------------ height above nearest drainage
def hand(z: np.ndarray, direction: np.ndarray, channel: np.ndarray):
    """Height Above the Nearest Drainage: for every cell, its elevation minus the elevation of the channel cell its D8 path first reaches. Cells whose path never meets a channel get NaN.
    Processed from lowest to highest cell, so each cell's downstream neighbour is already done. A standard terrain proxy for flood-prone ground: low HAND = close to the water in height."""
    h, w = z.shape
    dr = np.array([a for a, b in D8]); dc = np.array([b for a, b in D8])
    order = np.argsort(z, axis=None, kind="stable")
    r, c = np.divmod(order, w)
    k = direction.ravel()[order]
    nr, nc = r + np.where(k >= 0, dr[k], 0), c + np.where(k >= 0, dc[k], 0)
    ok = (k >= 0) & (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    tgt = np.where(ok, nr * w + nc, -1).tolist()
    zf = z.ravel()
    base = np.full(h * w, np.nan)                         # elevation of the channel cell each cell drains to
    ch = channel.ravel()
    for i, t in zip(order.tolist(), tgt):
        if ch[i]:
            base[i] = zf[i]
        elif t >= 0 and not np.isnan(base[t]):
            base[i] = base[t]
    return (zf - base).reshape(z.shape)


# ------------------------------------------------------------------ comparison with mapped streams and channel character
def near(mask: np.ndarray, lines_mask: np.ndarray, dist_m: float, res=1.0) -> np.ndarray:
    """Cells of `mask` within `dist_m` of any cell of `lines_mask`."""
    if not lines_mask.any():
        return np.zeros(mask.shape, bool)
    d = ndimage.distance_transform_edt(~lines_mask) * res
    return mask & (d <= dist_m)


def incision(dtm: np.ndarray, size=15) -> np.ndarray:
    """Depth of each cell below the local mean of the terrain in a size x size window (positive in a hollow)."""
    return ndimage.uniform_filter(dtm, size=size, mode="nearest") - dtm


def catchment_of(direction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cells whose D8 path reaches any cell of `target`. Iterates upstream from the target: a cell joins when its downstream neighbour is already in."""
    h, w = direction.shape
    inside = target.copy()
    dr = np.array([a for a, b in D8]); dc = np.array([b for a, b in D8])
    rr, cc = np.mgrid[0:h, 0:w]
    ok = direction >= 0
    k = np.where(ok, direction, 0)
    tr, tc = np.clip(rr + dr[k], 0, h - 1), np.clip(cc + dc[k], 0, w - 1)
    while True:
        add = ok & ~inside & inside[tr, tc]
        if not add.any():
            return inside
        inside |= add
