"""Bitemporal change detection between 3DEP lidar epochs: surface selection, registration on stable ground, and M3C2 (Lague et al. 2013) via py4dgeo.

Epoch classes are not comparable between projects (the 2014 project labels the bluff face and beach 17/18, the 2021 project labels the bluff face ground), so surface points are chosen geometrically.
Coordinates are local metres (UTM minus the area centre); distance = epoch 2 minus epoch 1 along the surface normal (negative = surface moved landward / lowered).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from scipy import ndimage

from .aoi import AOI
from .ept import CACHE, EPT

SHORE_ANGLE_DEG = -13.7       # tilt of the bluff crest from local north (fitted to the 14-16 m contour), used to cut the shore into alongshore bands

SITE = dict(slug="capecod_wellfleet", lonlat=(-69.97151507, 41.91278736), half=300.0, crs="EPSG:32619",
            epochs=(("2014", "MA_NE_CMGP_Sandy_Z19_A1_2015"), ("2021", "MA_CentralEastern_2_2021")))


@dataclass
class Epoch:
    tag: str
    project: str
    date: str
    x: np.ndarray        # local metres east
    y: np.ndarray        # local metres north
    z: np.ndarray        # elevation, m
    cls: np.ndarray
    ret: np.ndarray
    nret: np.ndarray

    def __len__(self):
        return len(self.x)

    @property
    def xyz(self):
        return np.c_[self.x, self.y, self.z]


def make_aoi(site=SITE) -> AOI:
    cx, cy = Transformer.from_crs("EPSG:4326", site["crs"], always_xy=True).transform(*site["lonlat"])
    return AOI(round(cx), round(cy), site["half"], site["crs"])


def gps_date(gps: np.ndarray) -> str:
    """Median collection date from a sample of GPS times. LAS 'adjusted standard' time is GPS seconds minus 1e9; values already above 1e9 are taken as plain GPS seconds."""
    g = gps[gps > 0]
    v = float(np.median(g))
    v += 1e9 if v < 1e9 else 0.0
    return str((dt.datetime(1980, 1, 6) + dt.timedelta(seconds=v)).date())


def load_epoch(tag: str, project: str, aoi: AOI, site=SITE, log=print) -> Epoch:
    """Points of one 3DEP project inside the AOI, cached under data/cache/change. Noise class 7 (low noise) is dropped; class 18 is kept because it holds real surface in some projects."""
    npz = CACHE / "change" / f"{site['slug']}_{tag}.npz"
    if not npz.exists():
        c = EPT(project).fetch(aoi.bbox_ll, max_depth=20, log=log)
        x0, y0, x1, y1 = aoi.bounds
        m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & (c.cls != 7)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, x=c.x[m] - aoi.cx, y=c.y[m] - aoi.cy, z=c.z[m], cls=c.cls[m], ret=c.ret[m], nret=c.nret[m], date=gps_date(c.gps))
    d = np.load(npz)
    return Epoch(tag, project, str(d["date"]), d["x"], d["y"], d["z"], d["cls"], d["ret"], d["nret"])


def surface_points(e: Epoch, res=1.0, tol=0.25, veg_margin=1.0, size=5) -> np.ndarray:
    """Boolean mask of the points that lie on the bare surface, without using class codes.
    Per `res` metre cell take the lowest point; keep points within a slope-dependent tolerance of it (`tol` + res * tan(slope), because a sloping surface spans that much height inside one cell),
    and drop cells whose lowest point stands more than `veg_margin` plus the slope allowance above the lowest point in the surrounding `size` x `size` cells (canopy with no ground return)."""
    n = int(np.ceil(2 * 300 / res)) + 2
    ix = np.floor((e.x + 300) / res).astype(int).clip(0, n - 1)
    iy = np.floor((e.y + 300) / res).astype(int).clip(0, n - 1)
    key = iy * n + ix
    order = np.lexsort((e.z, key))
    ks, zs = key[order], e.z[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    cell_min = np.full(n * n, np.nan)
    cell_min[ks[first]] = zs[first]
    cm = cell_min.reshape(n, n)
    filled = np.where(np.isnan(cm), np.nanmedian(cm), cm)
    gy, gx = np.gradient(ndimage.gaussian_filter(filled, 1.0), res)
    allow = tol + res * np.tan(np.arctan(np.hypot(gx, gy)).clip(0, np.radians(80)))
    local_min = ndimage.minimum_filter(np.where(np.isnan(cm), np.inf, cm), size=size)
    cell_ok = np.isfinite(cm) & (cm <= local_min + veg_margin + allow)
    ok_pt = cell_ok.ravel()[key] & (e.z <= cell_min[key] + allow.ravel()[key])
    return ok_pt


def thin(xyz: np.ndarray, voxel: float, seed=0) -> np.ndarray:
    """Index of one point per `voxel` m cube (the first in a random order), for an even sample in 3D."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(xyz))
    k = np.floor(xyz[perm] / voxel).astype(np.int64)
    _, first = np.unique(k, axis=0, return_index=True)
    return np.sort(perm[first])


def stable_mask(xyz: np.ndarray, stable_x_max: float) -> np.ndarray:
    """Points of the stable inland area: x (local east) below `stable_x_max`. The area is chosen by looking at the data (the DEM difference there is flat); it is a choice, not a measurement."""
    return xyz[:, 0] < stable_x_max


def m3c2(p1: np.ndarray, p2: np.ndarray, core: np.ndarray, normal_radii=(2.0, 4.0, 6.0), cyl_radius=1.5, max_distance=15.0, registration_error=0.0):
    """Multiscale Model-to-Model Cloud Comparison with py4dgeo. Normals come from epoch 1 at the scale with the best planarity, distances are epoch 2 minus epoch 1 along the normal,
    and the level of detection (95 %) is 1.96 * sqrt(spread1^2/n1 + spread2^2/n2) + registration_error. Returns (distance, lod, spread1, spread2, n1, n2, normals)."""
    import py4dgeo
    e1, e2 = py4dgeo.Epoch(p1), py4dgeo.Epoch(p2)
    alg = py4dgeo.M3C2(epochs=(e1, e2), corepoints=core, normal_radii=list(normal_radii), cyl_radius=cyl_radius, max_distance=max_distance, registration_error=registration_error)
    d, u = alg.run()
    return d, u["lodetection"], u["spread1"], u["spread2"], u["num_samples1"], u["num_samples2"], alg.directions()


def significant(d: np.ndarray, lod: np.ndarray) -> np.ndarray:
    """Change larger than the level of detection (and finite)."""
    return np.isfinite(d) & np.isfinite(lod) & (np.abs(d) > lod)


def volume_budget(core: np.ndarray, d: np.ndarray, lod: np.ndarray, spacing: float) -> dict:
    """Net surface displacement integrated over the core-point area each point represents (`spacing` m grid): loss = significant negative distances, gain = significant positive.
    This is a surface-normal displacement times plan-view cell area, an approximation of volume that ignores the extra area of steep faces (it understates their volume change by a factor of cos(slope))."""
    sig = significant(d, lod)
    a = spacing ** 2
    return dict(loss_m3=float(-(d[sig & (d < 0)]).sum() * a), gain_m3=float(d[sig & (d > 0)].sum() * a), n=int(len(d)), n_valid=int(np.isfinite(d).sum()), n_significant=int(sig.sum()))


def regular_cores(p1: np.ndarray, spacing: float, min_pts=3) -> np.ndarray:
    """Core points on a regular plan grid: cell centre x, y and the median epoch-1 surface height in the cell; cells with fewer than `min_pts` points are skipped. Each core point stands for spacing^2 m^2 of plan area."""
    ix = np.floor((p1[:, 0] + 300) / spacing).astype(int)
    iy = np.floor((p1[:, 1] + 300) / spacing).astype(int)
    n = int(np.ceil(600 / spacing)) + 1
    key = iy * n + ix
    order = np.argsort(key)
    ks = key[order]
    cuts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= min_pts:
            k = ks[a]
            out.append((((k % n) + 0.5) * spacing - 300, ((k // n) + 0.5) * spacing - 300, np.median(p1[order[a:b], 2])))
    return np.array(out)


def robust_sd(v: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def volume_from_normals(d, nrm, sig, plan_area):
    """Volume = distance along the normal x surface area, with surface area = plan area / |n_z| (|n_z| floored at 0.3). Returns (loss, gain) in m^3, counting significant points only."""
    nz = np.abs(nrm[:, 2]).clip(0.3, 1.0)
    v = d * plan_area / nz
    return float(-v[sig & (d < 0)].sum()), float(v[sig & (d > 0)].sum())


def grid_dem(p: np.ndarray, res=2.0, min_pts=3) -> np.ndarray:
    """Median surface height per `res` m cell over the 600 m area, rows = south to north (row 0 is y = -300), NaN where a cell has fewer than `min_pts` points."""
    n = int(round(600 / res))
    ix = np.floor((p[:, 0] + 300) / res).astype(int).clip(0, n - 1)
    iy = np.floor((p[:, 1] + 300) / res).astype(int).clip(0, n - 1)
    key = iy * n + ix
    o = np.argsort(key)
    ks, zs = key[o], p[o, 2]
    cuts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    out = np.full(n * n, np.nan)
    for i, j in zip(cuts[:-1], cuts[1:]):
        if j - i >= min_pts:
            out[ks[i]] = np.median(zs[i:j])
    return out.reshape(n, n)


def zone_stats(core, d, sig, nrm, ok_cells, zones, spacing, registration_error=0.0):
    """Per-zone significant volume change. `zones` maps a name to a boolean mask over the core points. uncertainty_m3 = registration error x surface area of the zone: the volume a uniform offset of that size would create."""
    res = {}
    for k, m in zones.items():
        loss, gain = volume_from_normals(d, nrm, sig & m, spacing ** 2)
        surf = float((spacing ** 2 / np.abs(nrm[m, 2]).clip(0.3, 1.0)).sum())
        res[k] = dict(cells=int(m.sum()), surface_area_m2=surf, loss_m3=loss, gain_m3=gain, net_m3=gain - loss, uncertainty_m3=registration_error * surf)
    return res


def dod_check(p1: np.ndarray, p2: np.ndarray, core: np.ndarray, d: np.ndarray, sig: np.ndarray, nrm: np.ndarray, ok_cells: np.ndarray, zones: dict, spacing=2.0):
    """Independent volume check on the same cells: DEM of difference (vertical difference x plan area, both epochs binned on the core grid) against the M3C2 volume. `ok_cells` selects the cells both use.
    Returns {zone: dict(cells, dod_net_m3, m3c2_net_m3, difference_pct)}."""
    D0, D1 = grid_dem(p1, spacing), grid_dem(p2, spacing)
    n = D0.shape[0]
    ci = np.floor((core[:, 0] + 300) / spacing).astype(int).clip(0, n - 1)
    cj = np.floor((core[:, 1] + 300) / spacing).astype(int).clip(0, n - 1)
    dz = (D1 - D0)[cj, ci]
    out = {}
    for k, m in zones.items():
        use = m & ok_cells & np.isfinite(dz)
        dod = float(dz[use].sum() * spacing ** 2)
        _, _ = None, None
        loss, gain = volume_from_normals(d, nrm, sig & use, spacing ** 2)
        m3 = gain - loss
        out[k] = dict(cells=int(use.sum()), dod_net_m3=dod, m3c2_net_m3=m3, difference_pct=float(100 * (m3 - dod) / abs(dod)) if abs(dod) > 1 else None)
    return out


def export_change(out, core, d, sig, smooth, aoi):
    """Writes the core points with their change (CSV, UTM coordinates), a 2 m GeoTIFF of significant change (NaN where nothing significant or not interpreted) and a PLY coloured by change."""
    import rasterio
    from matplotlib import colormaps
    from rasterio.transform import from_origin
    from pathlib import Path
    from .exports import point_ply
    out = Path(out)
    keep = np.isfinite(d)
    xyz = core[keep]
    utm = np.c_[xyz[:, 0] + aoi.cx, xyz[:, 1] + aoi.cy, xyz[:, 2]]
    np.savetxt(out / "m3c2_core_points.csv", np.c_[utm, d[keep], sig[keep] & smooth[keep]], delimiter=",", fmt=["%.2f", "%.2f", "%.2f", "%.3f", "%d"], header="easting_m,northing_m,z_m,m3c2_distance_m,significant_and_smooth", comments="")
    n = 300
    img = np.full((n, n), np.nan, "float32")
    ci = np.floor((core[:, 0] + 300) / 2).astype(int).clip(0, n - 1)
    cj = np.floor((core[:, 1] + 300) / 2).astype(int).clip(0, n - 1)
    m = sig & smooth & keep
    img[cj[m], ci[m]] = d[m]
    tf = from_origin(aoi.cx - 300, aoi.cy + 300, 2.0, 2.0)
    with rasterio.open(out / "m3c2_significant_change_2m.tif", "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs=aoi.crs, transform=tf, nodata=np.nan, compress="deflate") as dst:
        dst.write(img[::-1], 1)
    col = (colormaps["RdBu"](np.clip((np.where(m, d, 0) + 3) / 6, 0, 1))[:, :3] * 255).astype(np.uint8)
    col[~m] = (200, 198, 190)
    point_ply(core, col, out / "m3c2_core_points.ply")
