"""Buildings and roof planes from airborne lidar: footprints from class-6 returns, sequential RANSAC for the planar roof segments of each building, and a simple LOD2-style mesh.

No mapped footprints are used to build anything (Overture footprints are only a check). RANSAC here is a numpy implementation (Open3D was not installable in this environment).
All coordinates are local metres (UTM minus the area centre); the roof plane is z = a x + b y + c.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import rasterio.features
from pyproj import Transformer
from rasterio import Affine
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, shape

from . import forestry as FO
from .aoi import AOI
from .ept import CACHE, EPT

SITES = {
    "westu": dict(slug="houston_westu", name="West University Place (houses), Houston, Texas", lonlat=(-95.4335, 29.7175), half=200.0, context=300.0, project="TX_Coastal_B1_2018", crs="EPSG:32615"),
    "downtown": dict(slug="houston_downtown", name="Downtown Houston (high-rise core), Texas", lonlat=(-95.365, 29.757), half=200.0, context=300.0, project="TX_Coastal_B1_2018", crs="EPSG:32615"),
}


def make_aoi(site, half=None) -> AOI:
    cx, cy = Transformer.from_crs("EPSG:4326", site["crs"], always_xy=True).transform(*site["lonlat"])
    return AOI(round(cx), round(cy), half or site["half"], site["crs"])


@dataclass
class Cloud:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    cls: np.ndarray
    nret: np.ndarray
    date: str
    half: float          # half-width of the loaded square (analysis area + shading context)
    cx: float
    cy: float


def load_cloud(site, log=print) -> Cloud:
    """Returns in the analysis square plus a shading context ring (class 7 and 18 noise dropped), cached as float32 local metres."""
    half = site["half"] + site["context"]
    npz = CACHE / "rooftop" / f"{site['slug']}_{int(half)}_v2.npz"
    aoi = make_aoi(site, half)
    if not npz.exists():
        c = EPT(site["project"]).fetch(aoi.bbox_ll, max_depth=20, log=log)
        x0, y0, x1, y1 = aoi.bounds
        m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & ~np.isin(c.cls, (7, 18))
        g = c.gps[c.gps > 0]
        v = float(np.median(g))
        v += 1e9 if v < 1e9 else 0.0
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, x=(c.x[m] - aoi.cx).astype("float32"), y=(c.y[m] - aoi.cy).astype("float32"), z=c.z[m].astype("float32"), cls=c.cls[m], nret=c.nret[m], date=str((dt.datetime(1980, 1, 6) + dt.timedelta(seconds=v)).date()))
    d = np.load(npz)
    return Cloud(d["x"], d["y"], d["z"], d["cls"], d["nret"], str(d["date"]), half, float(aoi.cx), float(aoi.cy))


def surfaces(C: Cloud, res=1.0, trees=True):
    """Terrain (class 2), height above ground of every return, and the surface model (highest return per cell, absolute elevation; with trees = vegetation classes included).
    Returns (dtm, hag, dsm); grids row 0 = south, origin at (-half, -half)."""
    g = C.cls == 2
    dtm, _ = FO.terrain(C.x[g], C.y[g], C.z[g], C.half, res)
    hag = C.z - FO.sample(dtm, C.half, res, C.x, C.y)
    keep = np.ones(len(C.x), bool) if trees else ~np.isin(C.cls, (3, 4, 5))
    iy, ix, n = FO._cells(C.x[keep], C.y[keep], C.half, res)
    dsm = np.full(n * n, -np.inf)
    np.maximum.at(dsm, iy * n + ix, C.z[keep])
    dsm = dsm.reshape(n, n)
    empty = ~np.isfinite(dsm)
    dsm = np.where(empty, dtm, dsm)
    return dtm, hag, np.maximum(dsm, np.where(np.isfinite(dtm), dtm, -np.inf))


# ------------------------------------------------------------------ structure points without trusting the class codes
def planar_points(C: Cloud, hag, min_h=2.5, k=10, rms=0.10, max_tilt=70.0, radius=2.0, max_hag=250.0):
    """Returns that sit on a locally planar, non-vertical surface at least `min_h` above ground: for each non-ground return, a plane is fitted to its `k` nearest returns within `radius` m; it passes when
    the rms distance of those neighbours to the plane is below `rms` m and the plane's tilt from horizontal is under `max_tilt` degrees. Roofs pass; tree crowns (rough), walls (vertical) and wires mostly fail.
    This replaces the class-6 label, which in this project also covers tree returns over houses. Returns a boolean mask over all returns."""
    cand = np.flatnonzero((C.cls != 2) & (hag >= min_h) & (hag <= max_hag))
    X = np.c_[C.x[cand], C.y[cand], C.z[cand]].astype("float64")
    t = cKDTree(X)
    ok = np.zeros(len(cand), bool)
    for i in range(0, len(cand), 400_000):
        d, nb = t.query(X[i:i + 400_000], k=k, distance_upper_bound=radius)
        valid = np.isfinite(d).all(1)
        nbv = np.where(np.isfinite(d), nb, 0)
        N = X[nbv] - X[nbv].mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", N, N) / k
        w, v = np.linalg.eigh(cov)
        tilt = np.degrees(np.arccos(np.clip(np.abs(v[:, 2, 0]), 0, 1)))
        ok[i:i + 400_000] = valid & (np.sqrt(np.maximum(w[:, 0], 0)) < rms) & (tilt < max_tilt)
    m = np.zeros(len(C.x), bool)
    m[cand[ok]] = True
    return m


# ------------------------------------------------------------------ footprints
def footprints(C: Cloud, hag, res=1.0, min_area=30.0, min_h=2.5, close_px=1, simplify=0.6, use=None, min_pts_cell=2, open_px=1):
    """Building footprints from the returns selected by `use` (default class 6) at least `min_h` above ground: rasterise (`res` cells with at least `min_pts_cell` returns), open to drop speckle, close small gaps, fill holes, label 8-connected components, drop those under `min_area` m2.
    Returns (label grid, list of shapely polygons in local metres, one per label, label ids start at 1)."""
    m6 = (use if use is not None else (C.cls == 6)) & (hag >= min_h)
    iy, ix, n = FO._cells(C.x[m6], C.y[m6], C.half, res)
    cnt = np.zeros((n, n), np.int32)
    np.add.at(cnt, (iy, ix), 1)
    mask = cnt >= min_pts_cell
    if open_px:
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)), iterations=open_px)
    if close_px:
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=close_px)
    mask = ndimage.binary_fill_holes(mask)
    lab, k = ndimage.label(mask, structure=np.ones((3, 3)))
    area = ndimage.sum(mask, lab, np.arange(1, k + 1)) * res * res
    keep = np.flatnonzero(area >= min_area) + 1
    remap = np.zeros(k + 1, int)
    remap[keep] = np.arange(1, len(keep) + 1)
    lab = remap[lab]
    tf = Affine(res, 0, -C.half, 0, res, -C.half)       # row 0 = south, so y increases with row
    polys = {}
    for geom, v in rasterio.features.shapes(lab.astype("int32"), mask=lab > 0, transform=tf, connectivity=8):
        if v > 0:
            p = shape(geom)
            polys[int(v)] = polys[int(v)].union(p) if int(v) in polys else p
    out = []
    for i in range(1, len(keep) + 1):
        p = polys[i]
        p = p.simplify(simplify) if simplify else p
        out.append(p if p.geom_type == "Polygon" else max(p.geoms, key=lambda q: q.area))
    return lab, out


# ------------------------------------------------------------------ roof planes by RANSAC
@dataclass
class Plane:
    a: float
    b: float
    c: float
    n_points: int
    rms: float
    pts: np.ndarray = field(repr=False, default=None)

    @property
    def slope_deg(self):
        return float(np.degrees(np.arctan(np.hypot(self.a, self.b))))

    @property
    def aspect_deg(self):
        """Compass direction the plane faces (down-slope direction), degrees clockwise from north; flat planes return nan."""
        if np.hypot(self.a, self.b) < 1e-9:
            return float("nan")
        return float(np.degrees(np.arctan2(-self.a, -self.b)) % 360)

    def z(self, x, y):
        return self.a * x + self.b * y + self.c


def _ransac_plane(P, rng, tol, iters=400):
    n = len(P)
    idx = rng.integers(0, n, (iters, 3))
    A, B, Cc = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    nrm = np.cross(B - A, Cc - A)
    ok = np.abs(nrm[:, 2]) > 1e-6
    nrm = nrm[ok]
    A = A[ok]
    if not len(nrm):
        return None
    a, b = -nrm[:, 0] / nrm[:, 2], -nrm[:, 1] / nrm[:, 2]
    c = A[:, 2] - a * A[:, 0] - b * A[:, 1]
    sl = np.hypot(a, b)
    good = sl < np.tan(np.radians(80))
    a, b, c = a[good], b[good], c[good]
    if not len(a):
        return None
    cnt = np.empty(len(a), int)
    for i in range(0, len(a), 100):
        r = P[:, 2][None] - (a[i:i + 100, None] * P[:, 0][None] + b[i:i + 100, None] * P[:, 1][None] + c[i:i + 100, None])
        cnt[i:i + 100] = (np.abs(r) < tol * np.sqrt(1 + a[i:i + 100, None] ** 2 + b[i:i + 100, None] ** 2)).sum(1)
    j = int(np.argmax(cnt))
    return a[j], b[j], c[j]


def segment_roof(P, tol=0.12, min_points=40, max_planes=8, link=2.5, seed=0, min_area_m2=10.0):
    """Sequential RANSAC on one building's roof returns P (n,3): find the best plane (inliers within `tol` m of it), keep its largest spatially connected part (returns within `link` m of each other),
    refit by least squares, remove it, repeat. Returns a list of Plane (with the inlier points)."""
    rng = np.random.default_rng(seed)
    rest = P.copy()
    planes = []
    while len(rest) >= min_points and len(planes) < max_planes:
        m = _ransac_plane(rest, rng, tol)
        if m is None:
            break
        a, b, c = m
        for _ in range(2):
            r = rest[:, 2] - (a * rest[:, 0] + b * rest[:, 1] + c)
            inl = np.abs(r) < tol * np.sqrt(1 + a * a + b * b)
            if inl.sum() < 3:
                break
            sol = np.linalg.lstsq(np.c_[rest[inl, :2], np.ones(inl.sum())], rest[inl, 2], rcond=None)[0]
            a, b, c = sol
        r = rest[:, 2] - (a * rest[:, 0] + b * rest[:, 1] + c)
        inl = np.abs(r) < tol * np.sqrt(1 + a * a + b * b)
        idx = np.flatnonzero(inl)
        if len(idx) < min_points:
            break
        t = cKDTree(rest[idx, :2])
        pairs = t.query_pairs(link, output_type="ndarray")
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
        _, cl = connected_components(g, directed=False)
        big = np.argmax(np.bincount(cl))
        sel = idx[cl == big]
        if len(sel) < min_points:
            rest = np.delete(rest, idx[:1], axis=0)           # nothing coherent around this seed: drop a point so the loop moves on
            continue
        sol = np.linalg.lstsq(np.c_[rest[sel, :2], np.ones(len(sel))], rest[sel, 2], rcond=None)[0]
        res = rest[sel, 2] - (rest[sel, :2] @ sol[:2] + sol[2])
        hull_area = _hull_area(rest[sel, :2])
        if hull_area >= min_area_m2:
            planes.append(Plane(float(sol[0]), float(sol[1]), float(sol[2]), int(len(sel)), float(np.sqrt(np.mean(res ** 2))), rest[sel].copy()))
        rest = np.delete(rest, sel, axis=0)
    return planes


def _hull_area(xy):
    from scipy.spatial import ConvexHull
    try:
        return float(ConvexHull(xy).volume)
    except Exception:
        return 0.0


def merge_coplanar(planes, max_angle=4.0, max_offset=0.20):
    """Merge planes whose normals differ by less than `max_angle` degrees and whose surfaces agree within `max_offset` m at the points of the other plane (one flat roof split by an obstruction, or a
    RANSAC split of one slope). Returns the merged list, refit on the union of points."""
    planes = list(planes)
    changed = True
    while changed and len(planes) > 1:
        changed = False
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                p, q = planes[i], planes[j]
                n1 = np.array([-p.a, -p.b, 1.0]); n1 /= np.linalg.norm(n1)
                n2 = np.array([-q.a, -q.b, 1.0]); n2 /= np.linalg.norm(n2)
                if np.degrees(np.arccos(np.clip(n1 @ n2, -1, 1))) > max_angle:
                    continue
                d1 = np.abs(q.pts[:, 2] - p.z(q.pts[:, 0], q.pts[:, 1])).mean()
                d2 = np.abs(p.pts[:, 2] - q.z(p.pts[:, 0], p.pts[:, 1])).mean()
                if max(d1, d2) > max_offset:
                    continue
                U = np.vstack([p.pts, q.pts])
                sol = np.linalg.lstsq(np.c_[U[:, :2], np.ones(len(U))], U[:, 2], rcond=None)[0]
                res = U[:, 2] - (U[:, :2] @ sol[:2] + sol[2])
                planes[i] = Plane(float(sol[0]), float(sol[1]), float(sol[2]), int(len(U)), float(np.sqrt(np.mean(res ** 2))), U)
                del planes[j]
                changed = True
                break
            if changed:
                break
    return planes


@dataclass
class Building:
    bid: int
    footprint: Polygon
    ground_z: float
    planes: list
    cells: np.ndarray = field(repr=False, default=None)        # (n, 2) centres of 1 m roof cells (local m)
    cell_plane: np.ndarray = field(repr=False, default=None)   # plane index per cell, -1 = no plane
    height: float = 0.0
    n_returns: int = 0


def build_buildings(C: Cloud, hag, dtm, lab, polys, use=None, res_cell=1.0, assign_radius=2.0, tol=0.12, min_points=40, log=None):
    """For every footprint: the class-6 returns inside it (plus one cell of margin), RANSAC planes, merged, and each 1 m cell inside the footprint assigned to the nearest plane's returns within
    `assign_radius` m. Height = highest plane point above the footprint's ground."""
    m6 = (use if use is not None else (C.cls == 6)) & (hag >= 1.5)
    x, y, z = C.x[m6], C.y[m6], C.z[m6]
    iy, ix, n = FO._cells(x, y, C.half, 1.0)
    glab = ndimage.grey_dilation(lab, size=(3, 3))
    pl = glab[iy, ix]
    order = np.argsort(pl, kind="stable")
    pl_s = pl[order]
    starts = np.searchsorted(pl_s, np.arange(1, len(polys) + 2))
    pts_all = np.c_[x, y, z][order]
    out = []
    for i, poly in enumerate(polys, 1):
        P = pts_all[starts[i - 1]:starts[i]]
        if len(P) < min_points:
            continue
        planes = merge_coplanar(segment_roof(P, tol=tol, min_points=min_points, seed=i))
        if not planes:
            continue
        gz = float(np.nanmedian(FO.sample(dtm, C.half, 1.0, *np.array(poly.exterior.coords).T)))
        minx, miny, maxx, maxy = poly.bounds
        gx = np.arange(np.floor(minx) + 0.5, maxx, res_cell)
        gy = np.arange(np.floor(miny) + 0.5, maxy, res_cell)
        XX, YY = np.meshgrid(gx, gy)
        from shapely import contains_xy
        inside = contains_xy(poly, XX, YY)
        cells = np.c_[XX[inside], YY[inside]]
        allp = np.vstack([p.pts for p in planes])
        pid = np.concatenate([np.full(len(p.pts), j) for j, p in enumerate(planes)])
        d, k = cKDTree(allp[:, :2]).query(cells)
        cp = np.where(d <= assign_radius, pid[k], -1)
        top = max(float(p.pts[:, 2].max()) for p in planes)
        out.append(Building(i, poly, gz, planes, cells, cp, top - gz, int(len(P))))
    return out


def plane_stats(b: Building):
    """Per plane: true area (m2, from the assigned 1 m cells divided by cos(slope)), slope, aspect, mean height above ground."""
    rows = []
    for j, p in enumerate(b.planes):
        k = int((b.cell_plane == j).sum())
        if k == 0:
            continue
        rows.append(dict(plane=j, cells=k, area_m2=k / np.cos(np.radians(p.slope_deg)), slope_deg=p.slope_deg, aspect_deg=p.aspect_deg, rms_m=p.rms, n_points=p.n_points, mean_h=float(np.mean(p.pts[:, 2]) - b.ground_z)))
    return rows


# ------------------------------------------------------------------ mesh
def building_mesh(b: Building, plane_color=None, wall_color=(205, 198, 188)):
    """LOD2-style mesh of one building: each roof plane is its assigned cells' outline lifted onto the plane (triangulated), walls run from the ground to the roof along the footprint outline.
    Returns (vertices, faces, vertex_colors); plane_color(j) gives the RGB of plane j. Parts do not share vertices, so a vertex colour is a part colour."""
    import trimesh
    from rasterio import features
    verts, faces, cols = [], [], []
    cells = b.cells
    if cells is None or not len(cells):
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3), np.uint8)
    x0, y0 = np.floor(cells.min(0)) - 1
    nx, ny = int(np.ceil(cells[:, 0].max() - x0)) + 2, int(np.ceil(cells[:, 1].max() - y0)) + 2
    ci = np.floor(cells - [x0, y0]).astype(int)

    def add(P, F, col):
        base = sum(len(v) for v in verts)
        verts.append(P)
        faces.append(F + base)
        cols.append(np.tile(np.array(col, np.uint8), (len(P), 1)))
    for j, p in enumerate(b.planes):
        sel = b.cell_plane == j
        if sel.sum() < 2:
            continue
        g = np.zeros((ny, nx), np.int32)
        g[ci[sel, 1], ci[sel, 0]] = 1
        g = ndimage.binary_closing(g, structure=np.ones((3, 3))).astype(np.int32)
        for geom, v in features.shapes(g, mask=g > 0, transform=Affine(1, 0, x0, 0, 1, y0), connectivity=4):
            poly = shape(geom).simplify(0.45)
            polys = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
            for q in polys:
                if q.area < 1.5:
                    continue
                try:
                    pv, pf = trimesh.creation.triangulate_polygon(q)
                except Exception:
                    continue
                add(np.c_[pv, p.z(pv[:, 0], pv[:, 1])], pf, plane_color(j) if plane_color else (160, 160, 160))
    # walls: footprint outline from the ground to the roof surface (nearest roof plane evaluated at the vertex)
    allp = np.vstack([p.pts for p in b.planes])
    tree = cKDTree(allp[:, :2])
    ring = np.array(b.footprint.exterior.coords)
    ring = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    _, k = tree.query(ring)
    pid = np.concatenate([np.full(len(p.pts), j) for j, p in enumerate(b.planes)])
    top = np.array([b.planes[pid[kk]].z(*ring[i]) for i, kk in enumerate(k)])
    top = np.clip(top, b.ground_z + 1.0, None)
    n = len(ring)
    P = np.vstack([np.c_[ring, np.full(n, b.ground_z)], np.c_[ring, top]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [(i, j, n + j), (i, n + j, n + i)]
    add(P, np.array(F), wall_color)
    return np.vstack(verts), np.vstack(faces), np.vstack(cols)
