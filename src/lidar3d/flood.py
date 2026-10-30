"""Micro-topography, drainage and flood-stage mapping from lidar: bare-earth terrain (ground and road-surface returns), hydrological conditioning with pysheds, closed depressions and culvert candidates at road
embankments, and 'bathtub' inundation at incremental water levels with road vulnerability. Heights are metres in the survey's vertical datum (NAVD88 for 3DEP); local coordinates are UTM minus the centre."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from scipy import ndimage

from . import forestry as FO
from .aoi import AOI
from .ept import CACHE, EPT

SITE = dict(slug="norfolk_ghent", name="Ghent and Hague, Norfolk, Virginia", lonlat=(-76.3010, 36.8680), half=700.0, project="USGS_LPC_VA_Norfolk_2013_LAS_2015", crs="EPSG:32618")


def make_aoi(site=SITE) -> AOI:
    cx, cy = Transformer.from_crs("EPSG:4326", site["crs"], always_xy=True).transform(*site["lonlat"])
    return AOI(round(cx), round(cy), site["half"], site["crs"])


@dataclass
class Cloud:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    cls: np.ndarray
    date: str
    cx: float
    cy: float
    half: float


def load_cloud(site=SITE, log=print) -> Cloud:
    npz = CACHE / "flood" / f"{site['slug']}_{int(site['half'])}.npz"
    aoi = make_aoi(site)
    if not npz.exists():
        c = EPT(site["project"]).fetch(aoi.bbox_ll, max_depth=20, log=log)
        x0, y0, x1, y1 = aoi.bounds
        m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & ~np.isin(c.cls, (7, 18))
        g = c.gps[c.gps > 0] if len(c.gps) else c.gps
        if len(g):
            v = float(np.median(g))
            v += 1e9 if v < 1e9 else 0.0
            date = str((dt.datetime(1980, 1, 6) + dt.timedelta(seconds=v)).date())
        else:
            date = "unknown"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, x=(c.x[m] - aoi.cx).astype("float32"), y=(c.y[m] - aoi.cy).astype("float32"), z=c.z[m].astype("float32"), cls=c.cls[m], date=date)
    d = np.load(npz)
    return Cloud(d["x"], d["y"], d["z"], d["cls"], str(d["date"]), float(aoi.cx), float(aoi.cy), site["half"])


def bare_earth(C: Cloud, res=1.0, max_embankment=1.5, max_cut=1.0):
    """Terrain from ground returns (class 2) plus road-surface returns (class 11) that lie within `max_embankment` m above or `max_cut` m below a first terrain built from the ground returns alone: road returns that
    high above the ground-only terrain are bridge decks, ramps and overpasses (which water passes under), not embankments, and are left out. The result keeps embankments up to `max_embankment` m, which is what holds
    back water. Gaps are filled by smooth normalised convolution. Returns (dtm, count of returns used per cell, share of road returns kept)."""
    g = C.cls == 2
    t0, _ = FO.terrain(C.x[g], C.y[g], C.z[g], C.half, res, max_fill=80.0)
    r = C.cls == 11
    above = C.z[r] - FO.sample(t0, C.half, res, C.x[r], C.y[r])
    keep = np.isfinite(above) & (above <= max_embankment) & (above >= -max_cut)
    m = np.zeros(len(C.x), bool)
    m[np.flatnonzero(r)[keep]] = True
    m |= g
    dtm, cnt = FO.terrain(C.x[m], C.y[m], C.z[m], C.half, res, max_fill=60.0)
    return dtm, cnt, float(keep.mean()) if len(keep) else 1.0


# ------------------------------------------------------------------ voids that are water
def water_mask(C: Cloud, res=1.0, min_area=300.0):
    """Open water as cells with no returns of any class (water absorbs the pulse), closed over small gaps; components of at least `min_area` m2. Buildings have roof returns, so they are not voids."""
    iy, ix, n = FO._cells(C.x, C.y, C.half, res)
    cnt = np.zeros((n, n), np.int32)
    np.add.at(cnt, (iy, ix), 1)
    v = ndimage.binary_closing(cnt == 0, structure=np.ones((3, 3)), iterations=2)
    lab, k = ndimage.label(v)
    if not k:
        return np.zeros((n, n), bool)
    area = ndimage.sum(v, lab, np.arange(1, k + 1)) * res * res
    return np.isin(lab, 1 + np.flatnonzero(area >= min_area))


# ------------------------------------------------------------------ hydrological conditioning (pysheds)
def _grid(dem, half, res, crs):
    import pyproj
    from affine import Affine
    from pysheds.grid import Grid
    from pysheds.sview import Raster, ViewFinder
    if not hasattr(np, "in1d"):
        np.in1d = np.isin                                         # pysheds 0.5 still calls the function numpy 2 removed
    vf = ViewFinder(affine=Affine(res, 0, -half, 0, -res, half), shape=dem.shape, crs=pyproj.Proj(crs), nodata=np.nan)
    return Grid(viewfinder=vf), Raster(np.flipud(dem).astype("float64"), viewfinder=vf)


def condition(dem, half, res, crs):
    """Fill pits and depressions, resolve flats, D8 flow directions and flow accumulation (cells draining through each cell) with pysheds. All grids are returned with row 0 = south, like `dem`.
    Returns dict(filled, depth (filled - dem, the depth of each closed depression), flowdir, acc)."""
    grid, r = _grid(dem, half, res, crs)
    pit = grid.fill_pits(r)
    filled = grid.fill_depressions(pit)
    flat = grid.resolve_flats(filled)
    fd = grid.flowdir(flat, routing="d8")
    acc = grid.accumulation(fd, routing="d8")
    f = lambda a: np.flipud(np.asarray(a))
    return dict(filled=f(filled), depth=np.clip(f(filled) - dem, 0, None), flowdir=f(fd), acc=f(acc))


def streams(acc, min_cells):
    return acc >= min_cells


def depressions(dem, filled, res, min_depth=0.15, min_area=60.0):
    """Closed depressions: connected cells where the filled surface lies at least `min_depth` above the terrain, of at least `min_area` m2. Returns (labels, table dict of arrays: area_m2, max_depth_m, volume_m3, spill_level_m,
    min_level_m, centroid row/col, spill row/col)."""
    depth = filled - dem
    lab, k = ndimage.label(depth >= min_depth)
    idx = np.arange(1, k + 1)
    area = ndimage.sum(np.ones_like(dem), lab, idx) * res * res
    keep = area >= min_area
    remap = np.zeros(k + 1, int)
    remap[1:][keep] = np.arange(1, keep.sum() + 1)
    lab = remap[lab]
    idx = np.arange(1, keep.sum() + 1)
    d = np.where(lab > 0, depth, 0)
    vol = ndimage.sum(d, lab, idx) * res * res
    mx = ndimage.maximum(d, lab, idx)
    spill = ndimage.maximum(np.where(lab > 0, filled, -np.inf), lab, idx)
    mn = ndimage.minimum(np.where(lab > 0, dem, np.inf), lab, idx)
    cen = np.array(ndimage.center_of_mass(lab > 0, lab, idx)) if len(idx) else np.zeros((0, 2))
    # the pour point: the cell of the filled lake surface (cells of the filled DEM at exactly the spill level, connected to the depression) that touches filled terrain lower than the spill level
    nb_min = ndimage.minimum_filter(filled, size=3)
    sp = []
    for i, s_ in zip(idx, spill):
        r0 = np.argwhere(lab == i)
        lo_r, lo_c = np.maximum(r0.min(0) - 40, 0)
        hi_r, hi_c = r0.max(0) + 41
        sl = (slice(lo_r, hi_r), slice(lo_c, hi_c))
        at_level = np.abs(filled[sl] - s_) < 1e-4
        lake = ndimage.binary_propagation(lab[sl] == i, mask=at_level | (lab[sl] == i), structure=np.ones((3, 3)))
        out_ = lake & (nb_min[sl] < s_ - 0.01)
        rr, cc = np.nonzero(out_)
        if len(rr):
            j = int(np.argmin(nb_min[sl][rr, cc]))
            sp.append((rr[j] + lo_r, cc[j] + lo_c))
        else:
            sp.append((-1, -1))
    return lab, dict(area_m2=area[keep], max_depth_m=mx, volume_m3=vol, spill_level_m=spill, min_level_m=mn, centroid=cen, spill=np.array(sp))


def breach_paths(dem, lab, tab, res, road_mask, max_len_m=60.0, min_drop=0.05, min_barrier=0.10):
    """Culvert candidates: depressions whose spill point lies on or beside the road surface (road mask) are held back by the embankment. For each, the least-cost path (cost = height above the depression's lowest
    cell, so the path stays low) from the depression's lowest cell to the nearest outside cell that is lower than the spill level by at least `min_drop` is found within `max_len_m`; the excavation needed
    to open it is the area between the path's profile and the lowest level it must reach. A candidate must also be a real barrier: the spill level must stand at least `min_barrier` m above the terrain where the
    path ends beyond the road (otherwise the depression is simply nearly full, not held back). Returns a list of dicts and the DEM with every path cut to a monotone non-rising profile (a 'culverted' DEM)."""
    from skimage.graph import route_through_array
    out = []
    cut = dem.copy()
    near_road = ndimage.binary_dilation(road_mask, iterations=int(round(4.0 / res)))
    n = len(tab["area_m2"])
    for i in range(n):
        sr, sc = tab["spill"][i]
        if sr < 0 or not near_road[sr, sc]:
            continue
        r0, c0 = max(sr - int(max_len_m / res), 0), max(sc - int(max_len_m / res), 0)
        r1, c1 = min(sr + int(max_len_m / res), dem.shape[0]), min(sc + int(max_len_m / res), dem.shape[1])
        sub = dem[r0:r1, c0:c1]
        inside = lab[r0:r1, c0:c1] == i + 1
        if not inside.any():
            continue
        sr_, sc_ = np.unravel_index(np.argmin(np.where(inside, sub, np.inf)), sub.shape)       # lowest cell of the depression inside the search window
        start_level = float(sub[sr_, sc_])
        low = (sub <= tab["spill_level_m"][i] - min_drop) & ~inside & ~road_mask[r0:r1, c0:c1]      # a target beyond the road, not a point on the embankment itself
        if not low.any():
            continue
        # nearest such cell to the spill point, going out of the depression on the far side
        rr, cc = np.nonzero(low)
        d2 = (rr + r0 - sr) ** 2 + (cc + c0 - sc) ** 2
        k = int(np.argmin(d2))
        tgt = (rr[k] + r0, cc[k] + c0)
        if np.sqrt(d2[k]) * res > max_len_m:
            continue
        cost = np.clip(sub - start_level, 0, None) + 0.05
        path, _ = route_through_array(cost, (sr_, sc_), (tgt[0] - r0, tgt[1] - c0), fully_connected=True, geometric=True)
        pr = np.array([(p[0] + r0, p[1] + c0) for p in path])
        prof = dem[pr[:, 0], pr[:, 1]]
        target_level = min(prof.min(), start_level)
        barrier = float(tab["spill_level_m"][i] - prof[-1])
        if barrier < min_barrier:
            continue
        # cut a monotone non-rising profile from the depression bottom to the end of the path (never raising the terrain)
        mono = np.minimum.accumulate(prof)
        mono = np.minimum(prof, np.maximum(mono, prof[-1]))        # never cut below the level of the path's last cell, so the cut cannot make a new pit at its end
        excav = float(np.clip(prof - mono, 0, None).sum() * res * res * 1.0)           # one cell wide
        cut[pr[:, 0], pr[:, 1]] = np.minimum(cut[pr[:, 0], pr[:, 1]], mono)
        out.append(dict(depression=i + 1, spill_row=int(sr), spill_col=int(sc), area_m2=float(tab["area_m2"][i]), max_depth_m=float(tab["max_depth_m"][i]), volume_m3=float(tab["volume_m3"][i]), spill_level_m=float(tab["spill_level_m"][i]),
                        path_length_m=float(len(pr) * res), barrier_m=barrier, excavation_m3=excav, hold_height_m=float(tab["spill_level_m"][i] - target_level)))
    return out, cut


# ------------------------------------------------------------------ bathtub flooding
def bathtub(dem, level, source, connected=True):
    """Flooded cells at water level `level`: terrain below it that is connected (4-neighbour) to the source water (`source` boolean grid), or, with connected=False, all terrain below it (low spots that would
    hold rain). Returns a boolean grid. A static model: no flow, no rainfall, no drains or pumps."""
    low = (dem < level) | source
    if not connected:
        return dem < level
    lab, k = ndimage.label(low)
    keep = np.unique(lab[source & (lab > 0)])
    return np.isin(lab, keep[keep > 0]) & (dem < level)


def first_flood_level(dem, source, levels, connected=True):
    """For each cell the first of `levels` at which it floods (NaN if never)."""
    out = np.full(dem.shape, np.nan)
    for lv in levels:
        f = bathtub(dem, lv, source, connected)
        out[f & np.isnan(out)] = lv
    return out


# ------------------------------------------------------------------ roads
ROAD_HALF_WIDTH = dict(motorway=8.0, trunk=7.0, primary=6.0, secondary=5.5, tertiary=5.0, residential=4.0, unclassified=4.0, service=3.0)
DRIVABLE = tuple(ROAD_HALF_WIDTH)


def road_raster(lines, half, res):
    """Rasterise drivable road lines (list of (class, coords)) as a mask, each buffered by half its class width."""
    import rasterio.features
    from rasterio import Affine
    from shapely.geometry import LineString
    shapes = [(LineString(c).buffer(ROAD_HALF_WIDTH[k]), 1) for k, c in lines if k in ROAD_HALF_WIDTH]
    tf = Affine(res, 0, -half, 0, res, -half)
    return rasterio.features.rasterize(shapes, out_shape=(int(round(2 * half / res)),) * 2, transform=tf, fill=0, dtype="uint8").astype(bool) if shapes else np.zeros((int(round(2 * half / res)),) * 2, bool)


def sample_line(grid, coords, half, res, step=2.0, across=0.0):
    """Values of a grid (row 0 = south) at points every `step` m along a polyline; returns (xy points, values). With `across` > 0 each value is the median of three samples, on the centreline and `across` m to
    either side, so a gutter or crown on one side does not decide the depth for the whole width of the road."""
    from shapely.geometry import LineString
    ln = LineString(coords)
    n = max(int(ln.length // step), 1)
    d = np.linspace(0, ln.length, n + 1)
    pts = np.array([ln.interpolate(v).coords[0] for v in d])
    v0 = FO.sample(grid, half, res, pts[:, 0], pts[:, 1])
    if across <= 0:
        return pts, v0
    ahead = np.array([ln.interpolate(min(v + 1.0, ln.length)).coords[0] for v in d]) - np.array([ln.interpolate(max(v - 1.0, 0.0)).coords[0] for v in d])
    nrm = np.c_[-ahead[:, 1], ahead[:, 0]] / np.maximum(np.hypot(ahead[:, 0], ahead[:, 1]), 1e-9)[:, None]
    vl = FO.sample(grid, half, res, pts[:, 0] + across * nrm[:, 0], pts[:, 1] + across * nrm[:, 1])
    vr = FO.sample(grid, half, res, pts[:, 0] - across * nrm[:, 0], pts[:, 1] - across * nrm[:, 1])
    return pts, np.nanmedian(np.c_[v0, vl, vr], axis=1)


def network_access(segments, flooded, min_length=0.0):
    """Share of the drivable network length that stays connected to the largest connected piece when flooded segments (boolean per segment) are removed. `segments` = list of (node_a, node_b, length)."""
    import networkx as nx
    G = nx.Graph()
    for (a, b, L), f in zip(segments, flooded):
        if not f:
            G.add_edge(a, b, length=L)
    tot = sum(L for _, _, L in segments)
    if not G.number_of_edges():
        return 0.0, 0
    comps = sorted(nx.connected_components(G), key=lambda c: -sum(d["length"] for _, _, d in G.subgraph(c).edges(data=True)))
    big = sum(d["length"] for _, _, d in G.subgraph(comps[0]).edges(data=True))
    return big / tot, len(comps)


# ------------------------------------------------------------------ what each road segment sits on
def road_structure(coords, dtm, top, cnt, half, res, tagged_bridge, tagged_tunnel, step=3.0, deck_min=3.0):
    """Classify a road segment from the lidar and the map tags, so overpasses and the roads beneath them are not confused (the highest return above a road is a deck for both).
    Samples every `step` m along the centreline: `observed_share` = share of samples with ground or road-surface returns within one cell (the road surface itself was seen); `deck_share` (overhead share) = share where the highest
    return stands more than `deck_min` m above the terrain: a deck, but also tree canopy or a roof, so by itself it does not mean an overpass. Categories:
      tunnel: tagged as tunnel and covered; not assessed (the lidar cannot see inside).
      at_grade_open_cut: tagged as tunnel but the road surface is seen from above (a trench approach); assessed like at-grade roads, and the lowest, most flood-prone kind.
      bridge: tagged as bridge or elevated, with a deck seen or no road surface seen at ground level; the road surface is the deck, so water at ground level does not reach it.
      at_grade: the road surface was observed at ground level.
      at_grade_unobserved: untagged, surface not seen (an underpass or heavy overhead cover); treated as at ground level on the interpolated terrain and flagged, because this is the road that floods under an overpass.
    A bridge tag with no deck and an observed surface is treated as at_grade (tag_conflict). Returns a dict with category, shares, deck height (median of top minus terrain over deck samples), tag_conflict and the samples."""
    from shapely.geometry import LineString
    ln = LineString(coords)
    n = max(int(ln.length // step), 1)
    pts = np.array([ln.interpolate(i * ln.length / n).coords[0] for i in range(n + 1)])
    iy, ix, _ = FO._cells(pts[:, 0], pts[:, 1], half, res)
    obs = ndimage.maximum_filter((cnt > 0).astype("uint8"), size=3)[iy, ix] > 0
    z = dtm[iy, ix]
    zt = ndimage.maximum_filter(np.nan_to_num(top, nan=-9999.0), size=3)[iy, ix]
    deck = (zt - z) > deck_min
    obs_s, deck_s = float(obs.mean()), float(deck.mean())
    conflict = False
    if tagged_tunnel:
        cat = "at_grade_open_cut" if obs_s >= 0.6 and deck_s < 0.3 else "tunnel"       # a tunnel tag on a stretch whose road surface the lidar sees is an open-cut approach, which floods like any road in a trench
    elif tagged_bridge:
        if deck_s < 0.2 and obs_s >= 0.6:
            cat, conflict = "at_grade", True
        else:
            cat = "bridge"
    else:
        cat = "at_grade" if obs_s >= 0.5 else "at_grade_unobserved"
    return dict(category=cat, observed_share=obs_s, deck_share=deck_s, deck_height_m=float(np.median((zt - z)[deck])) if deck.any() else 0.0, tag_conflict=conflict, pts=pts, terrain=z, top=np.where(zt > -9000, zt, np.nan))
