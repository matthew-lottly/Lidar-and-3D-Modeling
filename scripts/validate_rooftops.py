"""Accuracy assessment of the rooftop study against real third-party data (run after scripts/build_rooftops.py all):
- footprints against Overture buildings (OpenStreetMap, Microsoft and Google derived) by area overlap, and by size class;
- lidar building heights against the heights Overture carries where a mapper recorded one;
- the hourly typical-year sunlight against the physical upper bound (clear-sky irradiance computed with pvlib's Ineichen model) as a sanity check on the weather input.
Usage: python scripts/validate_rooftops.py"""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import pvlib
from shapely.affinity import translate
from shapely.ops import unary_union
from shapely.strtree import STRtree

from lidar3d import city
from lidar3d import rooftop as RT
from lidar3d import validation as V
from lidar3d.ept import CACHE

DATA = ROOT / "data" / "processed" / "validation"


def one(key):
    site = RT.SITES[key]
    F = pickle.loads((CACHE / "rooftop" / f"{key}_final.pkl").read_bytes())
    aoi = RT.make_aoi(site)
    ov = city.overture_buildings(aoi)
    C = RT.load_cloud(site, log=lambda s: None)
    ov["geom_local"] = [translate(g, -C.cx, -C.cy) for g in ov.geometry]
    A = site["half"]
    from shapely.geometry import box
    win = box(-A, -A, A, A)
    ov = ov[[g.intersects(win) for g in ov.geom_local]].reset_index(drop=True)
    lid = {int(b.bid): b for b in F["B"]}
    polys = [b.footprint for b in lid.values()]
    ids = list(lid)
    ovp = [g.intersection(win) for g in ov.geom_local]
    tree = STRtree(ovp)
    rows = []
    for i, p in zip(ids, polys):
        best, bj = 0.0, -1
        for j in tree.query(p):
            q = ovp[j]
            inter = p.intersection(q).area
            if inter > 0:
                u = inter / (p.area + q.area - inter)
                if u > best:
                    best, bj = u, j
        rows.append((i, best, bj, p.area))
    R = pd.DataFrame(rows, columns=["building", "iou", "ov_index", "area"])
    out = dict(site=site["name"], overture_buildings_in_window=int(len(ov)), lidar_buildings=int(len(R)))
    lu, ou = unary_union(polys), unary_union([q for q in ovp if q.area >= 30])
    inter = lu.intersection(ou).area
    out["footprints"] = dict(lidar_area_inside_overture=float(inter / lu.area), overture_area_inside_lidar=float(inter / ou.area), median_iou=float(R.iou.median()), share_iou_ge_0p5=float((R.iou >= 0.5).mean()), share_no_overlap=float((R.iou == 0).mean()))
    for lo, hi in ((30, 150), (150, 400), (400, 1e9)):
        m = (R.area >= lo) & (R.area < hi)
        if m.any():
            out["footprints"][f"area_{lo}_{hi if hi < 1e8 else 'plus'}_m2"] = dict(n=int(m.sum()), median_iou=float(R.iou[m].median()))
    # Overture buildings that no lidar footprint covers, by size
    cov = [lu.intersection(q).area / q.area for q in ovp if q.area >= 30]
    out["footprints"]["overture_buildings_30m2_plus"] = int(len(cov))
    out["footprints"]["share_covered_half_or_more"] = float(np.mean(np.array(cov) >= 0.5))
    # heights
    h = ov.height.to_numpy() if "height" in ov else np.full(len(ov), np.nan)
    ok = (R.iou >= 0.3) & (R.ov_index >= 0)
    hh = []
    for _, r in R[ok].iterrows():
        oh = h[int(r.ov_index)]
        if np.isfinite(oh):
            hh.append((lid[int(r.building)].height, oh))
    hh = np.array(hh)
    out["heights"] = dict(overture_with_height=int(np.isfinite(h).sum()), matched_pairs=int(len(hh)))
    if len(hh) >= 3:
        d = hh[:, 0] - hh[:, 1]
        out["heights"].update(V.error_summary(d))
        out["heights"]["pearson_r"] = float(np.corrcoef(hh[:, 0], hh[:, 1])[0, 1]) if len(hh) > 3 else None
        out["heights"]["note"] = "Overture heights are mapper-entered or model-derived, not surveyed; lidar height is the highest roof point above the ground"
    # weather sanity: TMY hourly GHI against clear-sky GHI
    df = F["df"]
    lon, lat = site["lonlat"]
    loc = pvlib.location.Location(lat, lon)
    cs = loc.get_clearsky(df.index, model="ineichen")
    over = (df.ghi.to_numpy() > 1.05 * cs.ghi.to_numpy() + 20)
    out["weather"] = dict(hours=int(len(df)), hours_ghi_above_clearsky_plus_5pct=int(over.sum()), annual_ghi_kwh_m2=float(df.ghi.sum() / 1000), annual_clearsky_ghi_kwh_m2=float(cs.ghi.sum() / 1000), ratio=float(df.ghi.sum() / cs.ghi.sum()),
                          note="the typical year must not exceed what a clear sky allows; a ratio of 0.7 to 0.85 is typical of humid climates")
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    res = {"analysis": "Rooftop solar potential"}
    for k in ("westu", "downtown"):
        res[k] = one(k)
    (DATA / "rooftops.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:3500])


if __name__ == "__main__":
    main()
