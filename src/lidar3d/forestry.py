"""Forest vertical structure from airborne lidar: ground and vegetation separation, terrain, surface and canopy height models, individual tree crowns, canopy gaps and fragmentation.

Pipeline (per area): stream the 3DEP returns for the area -> bare-earth terrain from ground returns -> height above ground for every return -> canopy surface -> canopy height model (CHM = surface - terrain)
-> tree tops by a variable-window local maximum filter -> crowns by marker-controlled watershed -> counts, heights, crown sizes, density, gaps and patch metrics.
All coordinates are local metres (UTM minus the area centre); heights are metres above the local ground.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from scipy import ndimage

from .aoi import AOI
from .ept import CACHE, EPT

SITES = {
    "houston": dict(slug="nw_houston", name="Northwest Houston (Cypress), Texas", lonlat=(-95.69, 30.00), half=750.0, project="TX_Coastal_B1_2018", crs="EPSG:32615", max_height=45.0),
    "tahoe": dict(slug="tahoe_nf", name="Mixed-conifer forest north of Truckee, California (beside Tahoe National Forest)", lonlat=(-120.32, 39.40), half=500.0, project="CA_SierraNevada_5_2022", crs="EPSG:32610", max_height=70.0),
}


def make_aoi(site) -> AOI:
    cx, cy = Transformer.from_crs("EPSG:4326", site["crs"], always_xy=True).transform(*site["lonlat"])
    return AOI(round(cx), round(cy), site["half"], site["crs"])


@dataclass
class Points:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    cls: np.ndarray
    ret: np.ndarray
    nret: np.ndarray
    date: str
    cx: float
    cy: float


def load_points(site, log=print) -> Points:
    """Returns inside the area, noise classes (7 low, 18 high) dropped, cached as float32 local metres."""
    npz = CACHE / "forestry" / f"{site['slug']}.npz"
    aoi = make_aoi(site)
    if not npz.exists():
        c = EPT(site["project"]).fetch(aoi.bbox_ll, max_depth=20, log=log)
        x0, y0, x1, y1 = aoi.bounds
        m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & (c.cls != 7) & (c.cls != 18)
        g = c.gps[c.gps > 0]
        v = float(np.median(g))
        v += 1e9 if v < 1e9 else 0.0
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, x=(c.x[m] - aoi.cx).astype("float32"), y=(c.y[m] - aoi.cy).astype("float32"), z=c.z[m].astype("float32"), cls=c.cls[m], ret=c.ret[m], nret=c.nret[m],
                 date=str((dt.datetime(1980, 1, 6) + dt.timedelta(seconds=v)).date()))
    d = np.load(npz)
    return Points(d["x"], d["y"], d["z"], d["cls"], d["ret"], d["nret"], str(d["date"]), float(aoi.cx), float(aoi.cy))


def _cells(x, y, half, res):
    n = int(round(2 * half / res))
    return np.floor((y + half) / res).astype(int).clip(0, n - 1), np.floor((x + half) / res).astype(int).clip(0, n - 1), n


def terrain(x, y, z, half, res=1.0, max_fill=80.0):
    """Digital terrain model from ground returns: mean height per cell, empty cells filled by normalised Gaussian convolution at growing scales (a smooth fill; under dense canopy it is an interpolation, not a measurement).
    Returns (dtm, ground-return count per cell)."""
    iy, ix, n = _cells(x, y, half, res)
    key = iy * n + ix
    s = np.bincount(key, z.astype("float64"), minlength=n * n)
    c = np.bincount(key, minlength=n * n).astype("float64")
    known = c > 0
    dem = np.where(known, s / np.maximum(c, 1), np.nan).reshape(n, n)
    cnt = c.reshape(n, n)
    kn = known.reshape(n, n)
    out = dem.copy()
    v, w = np.where(kn, dem, 0.0), kn.astype(float)
    todo = ~kn
    for sg in (1.5, 3.0, 6.0, 12.0, 25.0, 50.0):
        if not todo.any():
            break
        num, den = ndimage.gaussian_filter(v, sg / res), ndimage.gaussian_filter(w, sg / res)
        ok = todo & (den > 0.05)
        out[ok] = num[ok] / den[ok]
        v, w = np.where(np.isfinite(out), out, 0.0), np.isfinite(out).astype(float)
        todo = ~np.isfinite(out)
    out[ndimage.distance_transform_edt(~kn, sampling=res) > max_fill] = np.nan
    return out, cnt


def sample(grid, half, res, x, y):
    """Bilinear value of a grid (row 0 = south) at points; NaN where a surrounding cell is NaN."""
    fx, fy = (x + half) / res - 0.5, (y + half) / res - 0.5
    v = ndimage.map_coordinates(np.nan_to_num(grid, nan=0.0), [fy, fx], order=1, mode="nearest")
    ok = ndimage.map_coordinates(np.isfinite(grid).astype("float32"), [fy, fx], order=1, mode="nearest")
    return np.where(ok > 0.999, v, np.nan)


def height_above_ground(P: Points, dtm, half, res=1.0):
    return P.z - sample(dtm, half, res, P.x, P.y)


def classify(P: Points, hag):
    """Return categories used by the analysis: ground (class 2), vegetation, building (class 6), other. Vegetation is classes 3 to 5 where the project has them; where it does not (unclassified class 1 only),
    vegetation is any non-ground return at least 0.5 m above the terrain that is not a building."""
    has_veg = np.isin(P.cls, (3, 4, 5)).any()
    ground = P.cls == 2
    bld = P.cls == 6
    veg = np.isin(P.cls, (3, 4, 5)) if has_veg else ((~ground) & (~bld) & (hag >= 0.5))
    if has_veg:
        veg = veg | ((P.cls == 1) & (hag >= 2.0))        # unclassified returns well above the ground in a project that does classify vegetation are almost always canopy
    return ground, veg, bld, has_veg


def canopy_height_model(x, y, hag, half, res=0.5, fill_px=1, smooth_px=0.0):
    """Canopy height model: highest return height above ground per `res` cell, single-pixel pits (a pulse that passed between crown tops) filled by a 3x3 maximum when the cell has no return,
    optional Gaussian smoothing. Cells with no return and no filled neighbour are 0 (open ground). Returns the CHM (row 0 = south)."""
    iy, ix, n = _cells(x, y, half, res)
    key = iy * n + ix
    chm = np.full(n * n, -np.inf)
    np.maximum.at(chm, key, hag)
    chm = chm.reshape(n, n)
    empty = ~np.isfinite(chm)
    if fill_px:
        fm = ndimage.maximum_filter(np.where(empty, -np.inf, chm), size=2 * fill_px + 1)
        chm = np.where(empty & np.isfinite(fm), fm, chm)
    chm = np.where(np.isfinite(chm), chm, 0.0)
    chm = np.clip(chm, 0.0, None)
    if smooth_px > 0:
        chm = ndimage.gaussian_filter(chm, smooth_px)
    return chm


# ------------------------------------------------------------------ individual tree crowns
def _disk(r_px):
    k = int(np.ceil(r_px))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (yy ** 2 + xx ** 2) <= r_px ** 2 + 1e-9


def tree_tops(chm_s, res, min_h=3.0, a=1.0, b=0.08, r_min=1.5, r_max=5.0):
    """Tree tops by a variable-window local-maximum filter: a pixel is a top when it is the highest within a disk of radius r = a + b * h metres (h = its own smoothed height, so tall trees need a
    wider empty neighbourhood than short ones), at least `min_h` high. Ties on flat tops are broken with a fixed pseudo-random offset so each flat top yields one tree. Returns a boolean grid."""
    rng = np.random.default_rng(1)
    h = chm_s + rng.uniform(0, 1e-4, chm_s.shape)
    rr = np.clip(a + b * h, r_min, r_max)
    levels = np.arange(r_min, r_max + 0.25, 0.25)
    top = np.zeros(h.shape, bool)
    for lo, hi in zip(levels[:-1], levels[1:]):
        sel = (rr >= lo) & (rr < hi) & (h >= min_h)
        if not sel.any():
            continue
        mx = ndimage.maximum_filter(h, footprint=_disk(lo / res), mode="nearest")
        top |= sel & (h >= mx)
    sel = (rr >= levels[-1]) & (h >= min_h)
    if sel.any():
        mx = ndimage.maximum_filter(h, footprint=_disk(levels[-1] / res), mode="nearest")
        top |= sel & (h >= mx)
    return top


def segment_crowns(chm, res, smooth_m=1.0, min_h=3.0, crown_frac=0.4, min_area_m2=3.0, **kw):
    """Individual tree crowns: smooth the CHM, find tops, grow crowns with a marker-controlled watershed on the inverted CHM, then cut each crown at `crown_frac` of its tree's height
    (so a crown does not run out over low vegetation or lawn) and drop crowns smaller than `min_area_m2`.
    Returns (labels grid, trees table as a dict of arrays: row, col, height, crown_area_m2, crown_diameter_m)."""
    from skimage.segmentation import watershed
    cs = ndimage.gaussian_filter(chm, smooth_m / res) if smooth_m > 0 else chm
    tops = tree_tops(cs, res, min_h=min_h, **kw)
    mk, n = ndimage.label(tops, structure=np.ones((3, 3)))
    ws = watershed(-cs, mk, mask=cs >= max(2.0, min_h * 0.6))
    # height of each tree = highest CHM value in its watershed region
    idx = np.arange(1, n + 1)
    hmax = ndimage.maximum(chm, ws, idx)
    keep_px = chm >= crown_frac * np.concatenate([[0.0], hmax])[ws]
    lab = np.where(keep_px, ws, 0)
    area = ndimage.sum(np.ones_like(chm), lab, idx) * res * res
    good = area >= min_area_m2
    remap = np.zeros(n + 1, int)
    remap[1:][good] = np.arange(1, good.sum() + 1)
    lab = remap[lab]
    keep_idx = idx[good]
    com = np.array(ndimage.maximum_position(cs, ws, keep_idx)) if len(keep_idx) else np.zeros((0, 2))
    ar = area[good]
    trees = dict(row=com[:, 0] if len(com) else np.zeros(0), col=com[:, 1] if len(com) else np.zeros(0), height=hmax[good], crown_area_m2=ar, crown_diameter_m=2 * np.sqrt(ar / np.pi))
    return lab, trees


# ------------------------------------------------------------------ canopy cover, gaps and fragmentation
def canopy_patches(chm, res, min_h=2.0):
    """Canopy mask (CHM at or above `min_h`) and 8-connected canopy patches. Returns (mask, labels, areas in m2)."""
    m = chm >= min_h
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    areas = ndimage.sum(m, lab, np.arange(1, n + 1)) * res * res
    return m, lab, areas


def fragmentation_metrics(chm, res, min_h=2.0):
    """Landscape metrics of the canopy mask: cover, patch count and mean/median/largest patch area, largest patch index (share of the area in the largest patch), edge density (m of canopy edge per ha)
    and effective mesh size (the area-weighted mean patch area, sum(A_i^2) / total area: the area of the patch a random point would be in)."""
    m, lab, areas = canopy_patches(chm, res, min_h)
    total = chm.size * res * res
    edge = (np.abs(np.diff(m.astype(int), axis=0)).sum() + np.abs(np.diff(m.astype(int), axis=1)).sum()) * res
    return dict(area_ha=total / 1e4, canopy_cover=float(m.mean()), patches=int(len(areas)), mean_patch_m2=float(areas.mean()) if len(areas) else 0.0, median_patch_m2=float(np.median(areas)) if len(areas) else 0.0,
                largest_patch_m2=float(areas.max()) if len(areas) else 0.0, largest_patch_index=float(areas.max() / total) if len(areas) else 0.0, edge_density_m_per_ha=float(edge / (total / 1e4)),
                effective_mesh_ha=float((areas ** 2).sum() / total / 1e4))


def canopy_gaps(chm, res, min_h=2.0, min_area=10.0):
    """Openings in the canopy: 4-connected regions of CHM below `min_h` of at least `min_area` m2. Regions that touch the edge of the area are kept but flagged, because their true size is cut off.
    Returns (labels, table dict with area_m2, touches_edge, max_inside_distance_m)."""
    op = chm < min_h
    lab, n = ndimage.label(op)
    idx = np.arange(1, n + 1)
    area = ndimage.sum(op, lab, idx) * res * res
    edge_labels = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    dist = ndimage.distance_transform_edt(op, sampling=res)
    mx = ndimage.maximum(dist, lab, idx)
    keep = area >= min_area
    remap = np.zeros(n + 1, int)
    remap[1:][keep] = np.arange(1, keep.sum() + 1)
    return remap[lab], dict(area_m2=area[keep], touches_edge=np.array([int(i) in edge_labels for i in idx[keep]]), max_inside_distance_m=mx[keep])


def density_grid(rows, cols, shape, res, cell_m=50.0):
    """Trees per hectare in `cell_m` cells (row 0 = south) from tree row/col positions."""
    k = int(round(cell_m / res))
    ny, nx = shape[0] // k, shape[1] // k
    h = np.zeros((ny, nx))
    ok = (rows < ny * k) & (cols < nx * k)
    np.add.at(h, ((rows[ok] // k).astype(int), (cols[ok] // k).astype(int)), 1)
    return h / (cell_m * cell_m / 1e4)


# ------------------------------------------------------------------ synthetic stands for the accuracy test
def synthetic_stand(density_ha, size=200.0, mean_h=18.0, sd_h=5.0, pulse_density=8.0, seed=0, min_spacing=3.0):
    """Trees of known position, height and crown size, scanned by a simulated lidar: half-ellipsoid crowns (radius 0.12 h + 0.8 m, depth 0.55 h) with 65% of returns on the outer surface and the rest inside
    (the pulses that penetrate), plus ground returns thinned under crowns. Returns (points (n,3), truth dict). Positions are metres from the centre of a `size` m square."""
    rng = np.random.default_rng(seed)
    n = int(density_ha * size * size / 1e4)
    pos = []
    tries = 0
    while len(pos) < n and tries < n * 60:
        p = rng.uniform(-size / 2, size / 2, 2)
        tries += 1
        if not pos or np.min(np.hypot(*(np.array(pos) - p).T)) >= min_spacing:
            pos.append(p)
    pos = np.array(pos)
    h = np.clip(rng.normal(mean_h, sd_h, len(pos)), 4.0, 45.0)
    rad = (0.12 * h + 0.8) * rng.uniform(0.9, 1.1, len(pos))
    depth = 0.55 * h
    area = size * size
    nv = int(pulse_density * area * 0.9)
    pts = []
    tree_id = rng.integers(0, len(pos), nv)                       # draw returns in proportion to crown surface roughly: uniform over trees weighted by crown area
    w = rad ** 2
    tree_id = rng.choice(len(pos), nv, p=w / w.sum())
    u = rng.uniform(0, 1, nv)
    surf = u < 0.65
    th = rng.uniform(0, 2 * np.pi, nv)
    ph = np.arccos(np.where(surf, rng.uniform(0, 1, nv), rng.uniform(0, 1, nv) ** (1 / 3) * rng.uniform(0, 1, nv)))       # upper hemisphere angle
    rr = np.where(surf, 1.0, rng.uniform(0, 1, nv) ** (1 / 3))
    px = pos[tree_id, 0] + rad[tree_id] * rr * np.sin(ph) * np.cos(th)
    py = pos[tree_id, 1] + rad[tree_id] * rr * np.sin(ph) * np.sin(th)
    pz = h[tree_id] - depth[tree_id] + depth[tree_id] * rr * np.cos(ph)
    pts.append(np.c_[px, py, pz])
    ng = int(pulse_density * area * 0.35)
    g = rng.uniform(-size / 2, size / 2, (ng, 2))
    d = np.hypot(g[:, None, 0] - pos[None, :, 0], g[:, None, 1] - pos[None, :, 1]) if ng * len(pos) < 4e7 else None
    keep = np.ones(ng, bool)
    if d is not None:
        under = (d < rad[None]).any(1)
        keep = ~under | (rng.uniform(0, 1, ng) < 0.12)
    pts.append(np.c_[g[keep], np.zeros(keep.sum())])
    P = np.vstack(pts)
    P = P[(np.abs(P[:, 0]) < size / 2) & (np.abs(P[:, 1]) < size / 2)]
    return P, dict(x=pos[:, 0], y=pos[:, 1], height=h, crown_radius=rad)


def match_trees(det_xy, det_h, tru_xy, tru_h, tru_r, min_tol=1.5, frac=0.6):
    """One-to-one greedy matching of detected to true trees by distance, within max(`min_tol`, `frac` x the true crown radius). Returns (index of matched detection for each true tree or -1)."""
    from scipy.spatial import cKDTree
    out = -np.ones(len(tru_xy), int)
    if len(det_xy) == 0:
        return out
    t = cKDTree(det_xy)
    pairs = []
    for i, (p, r) in enumerate(zip(tru_xy, tru_r)):
        for j in t.query_ball_point(p, max(min_tol, frac * r)):
            pairs.append((np.hypot(*(det_xy[j] - p)), i, j))
    used = set()
    for dd, i, j in sorted(pairs):
        if out[i] < 0 and j not in used:
            out[i] = j
            used.add(j)
    return out


# ------------------------------------------------------------------ outputs
def write_geotiff(path, arr, aoi: AOI, res, dtype="float32", nodata=None):
    """Write a grid (row 0 = south) as a north-up GeoTIFF in the area's UTM zone."""
    import rasterio
    from rasterio.transform import from_origin
    a = np.flipud(arr).astype(dtype)
    tf = from_origin(aoi.cx - aoi.half, aoi.cy + aoi.half, res, res)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype=dtype, crs=aoi.crs, transform=tf, nodata=nodata, compress="deflate", predictor=2 if "float" in dtype else 1) as d:
        d.write(a, 1)
    return path


def rgba_png_b64(rgba):
    import base64
    import io

    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(np.flipud(rgba).astype(np.uint8), "RGBA").save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()
