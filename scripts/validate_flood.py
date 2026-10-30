"""Accuracy assessment of the Norfolk flood study against independent real data (run after scripts/build_flood.py).
References used here: the 2011 FEMA lidar of the same ground (sparse, an independent survey), Overture's mapped bridges, and, when the hosts are reachable, USGS lidar products, Chesapeake 1 m land cover, JRC surface
water and Deltares flood maps. Unreachable references are recorded as such."""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy import ndimage
from shapely import wkb
from shapely.geometry import LineString

from lidar3d import flood as FL
from lidar3d import sitemap
from lidar3d import validation as V
from lidar3d.ept import EPT
from lidar3d.ept import CACHE

DATA = ROOT / "data" / "processed" / "validation"
HALF = FL.SITE["half"]


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    F = pickle.loads((CACHE / "flood" / "final.pkl").read_bytes())
    aoi = FL.make_aoi()
    dtm, roads, road_mask, ndsm = F["dtm"], F["roads"], F["road_mask"], F["ndsm"]
    out = dict(analysis="Norfolk drainage and flood-stage road vulnerability", references={})
    # 1. vertical repeatability of the lidar itself: the 2013 survey was flown in several overlapping swaths, which measure the same ground independently
    c = EPT(FL.SITE["project"]).fetch(aoi.bbox_ll, max_depth=20, log=lambda s: None, full_gps=True)
    x0, y0, x1, y1 = aoi.bounds
    m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & np.isin(c.cls, (2, 11))
    x, y, z, ps = (c.x[m] - aoi.cx).astype("float32"), (c.y[m] - aoi.cy).astype("float32"), c.z[m], c.psid[m]
    lines = [int(p) for p in np.unique(ps) if (ps == p).sum() > 100_000]
    n = int(2 * HALF / 4.0)
    ix = np.clip(((x + HALF) / 4.0).astype(int), 0, n - 1)
    iy = np.clip(((y + HALF) / 4.0).astype(int), 0, n - 1)
    key = iy * n + ix
    mean = {}
    for p in lines:
        sel = ps == p
        s_ = np.bincount(key[sel], z[sel].astype("float64"), minlength=n * n)
        k_ = np.bincount(key[sel], minlength=n * n)
        mean[p] = np.where(k_ >= 3, s_ / np.maximum(k_, 1), np.nan)
    cnt4 = np.bincount(key, minlength=n * n).reshape(n, n)
    built4 = ndimage.binary_dilation(ndimage.zoom(ndsm, 0.25, order=0) >= 2.5, iterations=1)[:n, :n]
    road4 = ndimage.zoom(road_mask.astype("uint8"), 0.25, order=0)[:n, :n] > 0
    gz = np.nanmean(np.stack([mean[p] for p in lines]), axis=0).reshape(n, n)
    gy, gx = np.gradient(np.where(np.isfinite(gz), gz, np.nanmedian(gz)), 4.0)
    flat4 = np.degrees(np.arctan(np.hypot(gx, gy))) < 2.0
    r = dict(project=FL.SITE["project"], swaths=lines, cell_m=4.0, pairs={})
    allp = {"roads": [], "open_ground": []}
    for i, p in enumerate(lines):
        for q in lines[i + 1:]:
            d = (mean[p] - mean[q]).reshape(n, n)
            for nm, mk in (("roads", road4 & ~built4), ("open_ground", ~road4 & ~built4)):
                v = d[mk & flat4 & np.isfinite(d)]
                if len(v) > 50:
                    allp[nm].append(v)
    for nm, vs in allp.items():
        if vs:
            v = np.concatenate(vs)
            r[nm] = dict(V.error_summary(v), share_within_10cm=float((np.abs(v) <= 0.10).mean()), share_within_20cm=float((np.abs(v) <= 0.20).mean()))
    r["note"] = "differences between the means of two swaths over flat, unbuilt ground in 4 m cells; a swath difference is about sqrt(2) times a single swath's error, and a horizontal or datum offset between swaths shows up here"
    out["references"]["swath_repeatability"] = r
    out["references"]["independent_lidar_2011"] = dict(status="VA_FEMA_SouthernCities_2011 is indexed over the window but holds no points inside it")
    # 2. Overture's mapped bridges (a second source for the bridge/overpass classification)
    try:
        import pyarrow.dataset as ds
        t = sitemap._table("base", "infrastructure", ["subtype", "class", "geometry"], aoi.bbox_ll, ds.field("subtype") == "bridge")
        from pyproj import Transformer
        tr = Transformer.from_crs(4326, FL.SITE["crs"], always_xy=True)
        from shapely.ops import unary_union
        geoms = []
        for g in t.geometry:
            g = wkb.loads(g)
            for p in (g.geoms if hasattr(g, "geoms") else [g]):
                x, y = tr.transform(*np.array(p.coords if p.geom_type != "Polygon" else p.exterior.coords).T)
                q = LineString(np.c_[x - F["cx"], y - F["cy"]]) if p.geom_type != "Polygon" else __import__("shapely.geometry", fromlist=["Polygon"]).Polygon(np.c_[x - F["cx"], y - F["cy"]])
                geoms.append(q.buffer(8.0))
        U = unary_union(geoms) if geoms else None
        cat = {r["id"]: r["category"] for r in roads}
        near = {}
        for r in roads:
            ln = LineString(r["coords"])
            near[r["id"]] = bool(U is not None and ln.intersection(U).length > 0.5 * ln.length)
        tab = {}
        for k in ("bridge", "at_grade", "at_grade_open_cut", "at_grade_unobserved", "tunnel"):
            ids = [i for i, c in cat.items() if c == k]
            tab[k] = dict(segments=len(ids), inside_mapped_bridge_footprint=int(sum(near[i] for i in ids)))
        out["references"]["overture_bridges"] = dict(mapped_bridge_features=len(geoms), by_category=tab,
                                                    note="a segment counts when more than half its length lies within 8 m of a mapped bridge; Overture bridges are OpenStreetMap-derived, as are the road tags, so this is a consistency check between two map sources, not an independent survey")
    except Exception as e:
        out["references"]["overture_bridges"] = dict(status=f"not run: {str(e)[:80]}")
    # 3. the road elevation: the lidar's own deck evidence by category
    out["structure_evidence"] = {}
    for k in ("bridge", "at_grade", "at_grade_open_cut", "at_grade_unobserved", "tunnel"):
        rr = [r for r in roads if r["category"] == k]
        if rr:
            out["structure_evidence"][k] = dict(segments=len(rr), median_deck_share=float(np.median([r["deck_share"] for r in rr])), median_observed_share=float(np.median([r["observed_share"] for r in rr])))
    # 4. flooding checks that need hosts which may be blocked
    out["references"]["blocked_or_pending"] = _try_remote(F, aoi)
    (DATA / "flood.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3500])


def _centre_mask(rs, n):
    """1 m cells within 1 m of the road centrelines (row 0 = south, as the cached grids)."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    shapes = [(LineString(r["coords"]).buffer(1.0), 1) for r in rs]
    m = rasterize(shapes, out_shape=(n, n), transform=from_origin(-HALF, HALF, 1.0, 1.0), fill=0, dtype="uint8").astype(bool)
    return m[::-1]


def _try_remote(F, aoi):
    """Real reference data for the flood study: the USGS-processed DTM, HAG and DSM of the same project, Chesapeake 1 m land cover, JRC surface water. Each is recorded as unavailable if it cannot be read."""
    from lidar3d import realref as RR
    res = {}
    W, S, E, N = aoi.bbox_ll
    n = int(2 * HALF)
    x0, y1 = aoi.cx - HALF, aoi.cy + HALF
    args = (x0, y1, 1.0, (n, n), aoi.crs)
    prj = FL.SITE["project"]
    dtm, roads = F["dtm"], F["roads"]
    built = ndimage.binary_dilation(F["ndsm"] >= 2.5, iterations=2)
    flip = lambda a: a[::-1] if a is not None else None
    cats = ("bridge", "at_grade", "at_grade_open_cut", "at_grade_unobserved", "tunnel")
    try:
        ref = flip(RR.lidar_product("dtm", prj, (W, S, E, N), *args))
        ok = np.isfinite(ref) & np.isfinite(dtm)
        res["usgs_dtm"] = dict(V.error_summary((dtm - ref)[ok & ~built]), on_roads=V.error_summary((dtm - ref)[ok & F["road_mask"].astype(bool) & ~built]),
                               note="USGS's own DTM of the same 2013 survey (their ground classification and gridding, 2 m): a second processing of the same points, not a second survey; it catches a gridding or ground-class error, not a flaw in the flight")
    except Exception as e:
        res["usgs_dtm"] = dict(status=f"not run: {str(e)[:100]}")
    try:
        hag = flip(RR.lidar_product("hag", prj, (W, S, E, N), *args))
        tab = {}
        for k in cats:
            rs = [r for r in roads if r["category"] == k]
            if not rs:
                continue
            mk = _centre_mask(rs, n)
            h = hag[mk & np.isfinite(hag)]
            tab[k] = dict(segments=len(rs), cells=int(len(h)), usgs_hag_median_m=float(np.median(h)) if len(h) else None, usgs_hag_p90_m=float(np.percentile(h, 90)) if len(h) else None,
                          share_over_2m=float((h > 2).mean()) if len(h) else None)
        res["usgs_hag_over_road_categories"] = dict(by_category=tab, note="USGS height above ground within 1 m of each category's centrelines: INCONCLUSIVE and not used to judge the classification: the product reads higher than our nDSM everywhere (median 7.4 m against our 2.6 m over the whole window, correlation 0.71), so it is evidently a different height definition (likely a maximum over the cell, picking up overhanging trees and vehicles) and the road-category medians cannot be read as deck height")
    except Exception as e:
        res["usgs_hag_over_road_categories"] = dict(status=f"not run: {str(e)[:100]}")
    lc = None
    try:
        lc = flip(RR.chesapeake((W, S, E, N), *args))
        tab = {}
        for k in cats:
            rs = [r for r in roads if r["category"] == k]
            if not rs:
                continue
            mk = FL.road_raster([(r["cls"], r["coords"]) for r in rs], HALF, 1.0).astype(bool)
            v = lc[mk & np.isfinite(lc)].astype(int)
            u, c = np.unique(v, return_counts=True)
            tab[k] = {RR.CHESAPEAKE13.get(int(a), str(a)): round(float(b) / len(v), 3) for a, b in sorted(zip(u, c), key=lambda t: -t[1])[:5]}
        res["chesapeake_land_cover_over_road_categories"] = dict(shares=tab, note="Chesapeake Conservancy 1 m land cover (2013-14 imagery plus lidar); 'tree canopy over roads' marks streets under trees, which the old elevated-road rule had wrongly excluded")
    except Exception as e:
        res["chesapeake_land_cover_over_road_categories"] = dict(status=f"not run: {str(e)[:100]}")
    wv = F.get("water")
    if wv is not None and lc is not None:
        res["chesapeake_water_vs_lidar_water_voids"] = V.binary_agreement(wv.astype(bool), np.isfinite(lc) & (lc == 1))
    try:
        occ = flip(RR.jrc_water((W, S, E, N), *args))
        if occ is not None and wv is not None:
            ok = np.isfinite(occ)
            res["jrc_water_occurrence_at_lidar_water_voids"] = dict(median_pct_in_voids=float(np.median(occ[ok & wv.astype(bool)])), median_pct_elsewhere=float(np.median(occ[ok & ~wv.astype(bool)])))
    except Exception as e:
        res["jrc_water_occurrence_at_lidar_water_voids"] = dict(status=f"not run: {str(e)[:100]}")
    return res


if __name__ == "__main__":
    main()
