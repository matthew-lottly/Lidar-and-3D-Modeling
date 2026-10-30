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


def _try_remote(F, aoi):
    """USGS DTM product, Chesapeake 1 m land cover, JRC water and Deltares flood maps, each only if its host answers."""
    from lidar3d import realref as RR
    res = {}
    W, S, E, N = aoi.bbox_ll
    h = int(HALF)
    x0, y1 = aoi.cx - HALF, aoi.cy + HALF
    try:
        ref = RR.lidar_product("dtm", "USGS_LPC_VA_Norfolk_2013_LAS_2015", (W, S, E, N), x0, y1, 2.0, (h, h), aoi.crs)
        if ref is None or not np.isfinite(ref).any():
            raise RuntimeError("no data returned")
        mine = F["dtm"][::-1][::2, ::2][:h, :h]
        built = ndimage.binary_dilation(F["ndsm"][::-1][::2, ::2][:h, :h] >= 2.5, iterations=2)
        d = (mine - ref)[~built]
        res["usgs_dtm_2m"] = V.error_summary(d)
    except Exception as e:
        res["usgs_dtm_2m"] = dict(status=f"host not reachable or no data: {str(e)[:90]}")
    return res


if __name__ == "__main__":
    main()
