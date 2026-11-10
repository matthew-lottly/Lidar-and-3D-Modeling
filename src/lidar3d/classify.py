"""Classes the 3DEP point cloud does not carry (this project has only unclassified, ground and noise): individual tree crowns from a canopy height model, relief-displacement-corrected image colours for returns above ground, and the
ground cover under the objects. Buildings come from `rooftop` (planar returns, footprints, roof planes). Each step reports a measure against an independent reference where one exists."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def canopy_height_model(x, y, hag, extent, res=0.5, hmin=1.5, hmax=60.0):
    """Highest return per `res` m cell (height above ground), NaN where nothing is above `hmin`. extent = (x0, y0, x1, y1); row 0 = south."""
    x0, y0, x1, y1 = extent
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    m = (hag >= hmin) & (hag <= hmax)
    ix = np.clip(((x[m] - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(((y[m] - y0) / res).astype(int), 0, ny - 1)
    chm = np.full(nx * ny, np.nan)
    np.fmax.at(chm, iy * nx + ix, hag[m])
    return chm.reshape(ny, nx)


def segment_trees(chm, extent, res=0.5, min_h=3.0, sigma=1.0, min_dist_m=1.8):
    """Crowns by marker-controlled watershed on the smoothed canopy height model: markers are local maxima at least `min_dist_m` apart and `min_h` tall; each crown is its watershed basin down to 40% of its own height.
    Returns (label grid, DataFrame-like dict of arrays: x, y, height, radius, area)."""
    filled = np.where(np.isfinite(chm), chm, 0.0)
    sm = ndimage.gaussian_filter(filled, sigma / res * 0.5)
    pk = peak_local_max(sm, min_distance=max(int(min_dist_m / res), 1), threshold_abs=min_h, exclude_border=False)
    markers = np.zeros(sm.shape, int)
    markers[pk[:, 0], pk[:, 1]] = np.arange(1, len(pk) + 1)
    lab = watershed(-sm, markers, mask=sm >= max(min_h * 0.5, 1.5))
    # crown = cells above 40% of the tree's own top height
    top = ndimage.maximum(sm, lab, np.arange(1, len(pk) + 1))
    keep = sm >= 0.4 * np.r_[0, top][lab]
    lab = np.where(keep, lab, 0)
    x0, y0 = extent[0], extent[1]
    ids = np.arange(1, len(pk) + 1)
    area = ndimage.sum(lab > 0, lab, ids) * res * res
    th = np.array([sm[p[0], p[1]] for p in pk])
    xs = x0 + (pk[:, 1] + 0.5) * res
    ys = y0 + (pk[:, 0] + 0.5) * res
    ok = (area >= 2.0)
    return lab, dict(id=ids[ok], x=xs[ok], y=ys[ok], height=np.maximum(th[ok], np.array([np.nanmax(chm[lab == i]) if (lab == i).any() else 0 for i in ids[ok]])), radius=np.sqrt(area[ok] / np.pi), area=area[ok])


def crown_bases(tree_lab, ids, x, y, hag, extent, res=0.5):
    """Height of the lowest foliage of each crown: the 12th percentile of the returns at least 1 m above ground inside the crown."""
    x0, y0, x1, y1 = extent
    ny, nx = tree_lab.shape
    ix = np.clip(((x - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(((y - y0) / res).astype(int), 0, ny - 1)
    lab = tree_lab[iy, ix]
    m = (lab > 0) & (hag >= 1.0)
    order = np.argsort(lab[m], kind="stable")
    ls, hs = lab[m][order], hag[m][order]
    uk, st = np.unique(ls, return_index=True)
    out = {int(k): float(np.percentile(h, 12)) for k, h in zip(uk, np.split(hs, st[1:]))}
    return np.array([out.get(int(i), 1.0) for i in ids])


def ndvi_at(ndvi, px, py, shift_x=0.0, shift_y=0.0):
    """Bilinear NDVI at pixel coordinates; ndvi is a raster, px/py are float pixel coordinates (col, row)."""
    return ndimage.map_coordinates(ndvi, [py + shift_y, px + shift_x], order=1, mode="nearest")


def fit_relief_displacement(ndvi_img, tf, cx, cy, x, y, h, n=60000, seed=0):
    """The aerial image is not a true orthophoto: a feature h m above ground is displaced away from the camera's nadir point by h * (k * (p - p0)), so a crown top is photographed beside its base. Fit (k, bx, by) in
    shift_east = h * (k * x + bx), shift_north = h * (k * y + by) (x, y local metres, shifts in metres) by maximising the mean NDVI under returns at least 8 m above ground, which should sit on green crowns. Returns the fit and the
    mean NDVI before (no shift) and after."""
    rng = np.random.default_rng(seed)
    m = np.flatnonzero(h >= 8.0)
    m = m[rng.choice(len(m), min(n, len(m)), replace=False)]
    px0 = (x[m] + cx - tf.c) / tf.a
    py0 = (y[m] + cy - tf.f) / tf.e
    hh = h[m]

    def shifted(p, sel=slice(None)):
        k, bx, by = p
        de = hh[sel] * (k * x[m][sel] + bx)
        dn = hh[sel] * (k * y[m][sel] + by)
        return ndvi_at(ndvi_img, px0[sel] + de / tf.a, py0[sel] + dn / tf.e)

    base = float(np.mean(shifted((0, 0, 0))))
    best = None
    for start in ((0, 0, 0), (1e-4, -0.2, 0.0), (-1e-4, 0.2, 0.0), (0, 0, 0.2), (0, 0, -0.2)):
        r = minimize(lambda p: -np.mean(shifted(p)), start, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-5, maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    k, bx, by = best.x
    return dict(k=float(k), bx=float(bx), by=float(by), mean_ndvi_before=base, mean_ndvi_after=float(-best.fun),
                shift_at_20m_height_m=float(np.hypot(20 * bx, 20 * by)), n=int(len(m)))


def displaced_rgb(img_rgb, tf, cx, cy, x, y, h, fit):
    """Image colour for returns at height h above ground using the fitted relief displacement (nearest pixel)."""
    de = h * (fit["k"] * x + fit["bx"])
    dn = h * (fit["k"] * y + fit["by"])
    c = ((x + de + cx - tf.c) / tf.a).astype(int).clip(0, img_rgb.shape[2] - 1)
    r = ((y + dn + cy - tf.f) / tf.e).astype(int).clip(0, img_rgb.shape[1] - 1)
    return np.stack([img_rgb[b][r, c] for b in range(3)], -1)


def fit_displacement_from_roofs(gray, tf, cx, cy, outlines, seed=0):
    """Relief displacement from building roofs, which are well posed (unlike canopy colour): the outline of a roof, drawn at its height h above ground, should sit on the strong edge the roof makes in the image once it is shifted by
    h * (k * p + b). `outlines` is a list of (xy (n, 2) local metres, h). Fits (k, bx, by) by maximising the mean gradient magnitude of the image under the shifted outlines. Returns the fit with the mean edge strength before and after."""
    gy_, gx_ = np.gradient(ndimage.gaussian_filter(gray.astype(float), 1.0))
    edge = np.hypot(gx_, gy_)
    pts = np.vstack([np.c_[o[0], np.full(len(o[0]), o[1])] for o in outlines])
    px0 = (pts[:, 0] + cx - tf.c) / tf.a
    py0 = (pts[:, 1] + cy - tf.f) / tf.e
    h = pts[:, 2]

    def score(p):
        k, bx, by = p
        de = h * (k * pts[:, 0] + bx)
        dn = h * (k * pts[:, 1] + by)
        return float(np.mean(ndimage.map_coordinates(edge, [py0 + dn / tf.e, px0 + de / tf.a], order=1, mode="nearest")))

    base = score((0, 0, 0))
    best = None
    for start in ((0, 0, 0), (1e-4, -0.2, 0.0), (-1e-4, 0.2, 0.0), (0, 0, 0.2), (0, 0, -0.2), (0, 0.3, 0.3), (0, -0.3, -0.3)):
        r = minimize(lambda p: -score(p), start, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    k, bx, by = best.x
    return dict(k=float(k), bx=float(bx), by=float(by), edge_before=base, edge_after=float(-best.fun), n=int(len(pts)), buildings=int(len(outlines)),
                shift_at_10m_height_m=float(np.hypot(10 * bx, 10 * by)))
