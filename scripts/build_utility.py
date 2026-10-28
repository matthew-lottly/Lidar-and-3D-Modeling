"""Poles, wires and substation structures from 3DEP lidar at White River Substation (Pierce County, WA), checked against Overture Maps power features. Usage: python scripts/build_utility.py"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from pyarrow import fs
from pyproj import Transformer
from shapely import wkb
from shapely.geometry import box
from shapely.ops import transform as shp_transform

from lidar3d import utility
from lidar3d.ept import CACHE, EPT
from lidar3d.names import RELEASE

PROJECT = "WA_PierceCounty_1_2020"
LONLAT = (-122.21035, 47.23898)
CRS = "EPSG:32610"
HALF = 400.0
OUT = ROOT / "data" / "processed" / "utility"


def load_points():
    npz = CACHE / "utility" / "white_river_v2.npz"
    tr = Transformer.from_crs(4326, CRS, always_xy=True)
    cx, cy = tr.transform(*LONLAT)
    if not npz.exists():
        tb = Transformer.from_crs(CRS, 4326, always_xy=True)
        W, S = tb.transform(cx - HALF - 30, cy - HALF - 30)
        E, N = tb.transform(cx + HALF + 30, cy + HALF + 30)
        c = EPT(PROJECT).fetch((W, S, E, N), max_depth=20, log=lambda s: None)
        m = (c.x >= cx - HALF) & (c.x < cx + HALF) & (c.y >= cy - HALF) & (c.y < cy + HALF) & (c.cls != 7)
        g = c.gps[c.gps > 0]
        v = float(np.median(g))
        v += 1e9 if v < 1e9 else 0.0
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, x=c.x[m] - cx, y=c.y[m] - cy, z=c.z[m], cls=c.cls[m], inten=c.intensity[m], ret=c.ret[m], nret=c.nret[m], date=str((dt.datetime(1980, 1, 6) + dt.timedelta(seconds=v)).date()))
    d = np.load(npz)
    load_points.extra = dict(inten=d["inten"], ret=d["ret"], nret=d["nret"])
    return d["x"], d["y"], d["z"], d["cls"], str(d["date"]), cx, cy, tr


def overture(cx, cy, tr):
    tb = Transformer.from_crs(CRS, 4326, always_xy=True)
    W, S = tb.transform(cx - HALF - 20, cy - HALF - 20)
    E, N = tb.transform(cx + HALF + 20, cy + HALF + 20)
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=os.environ.get("HTTPS_PROXY"))
    flt = (ds.field("bbox", "xmin") < E) & (ds.field("bbox", "xmax") > W) & (ds.field("bbox", "ymin") < N) & (ds.field("bbox", "ymax") > S) & (ds.field("subtype") == "power")
    t = ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/theme=base/type=infrastructure", filesystem=s3, format="parquet").to_table(filter=flt, columns=["class", "names", "geometry"]).to_pandas()
    win = box(-HALF, -HALF, HALF, HALF)
    feats = {}
    for _, r in t.iterrows():
        g = shp_transform(lambda x, y, z=None: tuple(np.array(tr.transform(np.asarray(x), np.asarray(y))) - np.array([[cx], [cy]])), wkb.loads(r.geometry))
        g = g.intersection(win)
        if not g.is_empty:
            feats.setdefault(r["class"], []).append((g, r["names"]))
    return feats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, z, cls, date, cx, cy, tr = load_points()
    xyz = np.c_[x, y, z]
    hag, _ = utility.height_above_ground(x, y, z, cls, (-HALF, -HALF, HALF, HALF), 1.0)
    feats = overture(cx, cy, tr)
    wmask, wlab, winfo = utility.detect_wires(xyz, hag)
    poles = utility.detect_poles(xyz, hag, exclude=wmask)
    # reference features inside the window
    lines = [np.asarray(g.coords) for cl in ("power_line", "minor_line") for g, _ in feats.get(cl, []) if g.geom_type == "LineString"]
    lines += [np.asarray(p.coords) for cl in ("power_line", "minor_line") for g, _ in feats.get(cl, []) if g.geom_type == "MultiLineString" for p in g.geoms]
    pts = lambda cl: np.array([(g.x, g.y) for g, _ in feats.get(cl, []) if g.geom_type == "Point"]).reshape(-1, 2)
    towers, ovpoles = pts("power_tower"), pts("power_pole")
    equip = np.vstack([pts(c) for c in ("switch", "transformer", "portal")])
    sub = [g for g, _ in feats.get("substation", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    wxy = xyz[wmask, :2]
    from scipy.spatial import cKDTree
    d_w, _ = cKDTree(np.vstack([np.c_[np.linspace(a[0], b[0], max(int(np.hypot(*(b - a)) / 1.0), 2)), np.linspace(a[1], b[1], max(int(np.hypot(*(b - a)) / 1.0), 2))] for ln in lines for a, b in zip(ln[:-1], ln[1:])])).query(wxy)
    P = np.array([(p["x"], p["y"]) for p in poles]).reshape(-1, 2)
    ref = np.vstack([towers, ovpoles])
    pr, ud, ur = utility.match_points(P, ref, 3.0)
    # substation yard
    inside = np.zeros(len(xyz), bool)
    if sub:
        from shapely import contains_xy
        inside = contains_xy(sub[0].buffer(0), x, y)
    structs = utility.yard_structures(xyz, hag, inside, wmask)
    S = np.array([(s["x"], s["y"]) for s in structs]).reshape(-1, 2)
    eq_in = np.array([sub and sub[0].contains(__import__("shapely").geometry.Point(*e)) for e in equip]) if sub and len(equip) else np.zeros(len(equip), bool)
    ep, _, _ = utility.match_points(equip[eq_in], S, 3.0) if eq_in.any() else ([], [], [])
    null = {}
    if sub and len(S) and eq_in.any():
        from shapely import contains_xy as cxy_
        rng = np.random.default_rng(0)
        minx, miny, maxx, maxy = sub[0].bounds
        R = rng.uniform([minx, miny], [maxx, maxy], (40000, 2))
        R = R[cxy_(sub[0], R[:, 0], R[:, 1])][:5000]
        tS = cKDTree(S)
        dm, dr = tS.query(equip[eq_in])[0], tS.query(R)[0]
        null = dict(mapped_within_1p5m=float((dm < 1.5).mean()), random_within_1p5m=float((dr < 1.5).mean()), mapped_within_3m=float((dm < 3).mean()), random_within_3m=float((dr < 3).mean()),
                    median_distance_mapped_m=float(np.median(dm)), median_distance_random_m=float(np.median(dr)))
    stats = dict(
        project=PROJECT, collected=date, center_lonlat=LONLAT, window_m=2 * HALF, points=int(len(xyz)), pts_per_m2=float(len(xyz) / (2 * HALF) ** 2), classes={int(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))},
        overture={k: len(v) for k, v in feats.items()},
        wires=dict(chains=len(winfo), points=int(wmask.sum()), total_chain_length_m=float(sum(v["length_m"] for v in winfo.values())), median_hag_m=float(np.median([v["mean_hag"] for v in winfo.values()])) if winfo else None,
                   share_points_within_5m_of_mapped_line=float((d_w < 5).mean()), mapped_line_length_in_window_m=float(sum(np.hypot(*np.diff(l, axis=0).T).sum() for l in lines)),
                   mapped_line_share_with_wire_within_6m=utility.line_coverage(lines, wxy, 6.0)),
        poles=dict(detected=len(poles), overture_towers=len(towers), overture_poles=len(ovpoles), matched_3m=len(pr), detected_unmatched=len(ud), overture_found=len(pr), overture_missed=len(ur),
                   recall=len(pr) / max(len(ref), 1), precision=len(pr) / max(len(poles), 1), median_match_distance_m=float(np.median([p[2] for p in pr])) if pr else None,
                   median_height_m=float(np.median([p["height"] for p in poles])) if poles else None),
        yard=dict(area_m2=float(sub[0].area) if sub else None, equipment_mapped=int(eq_in.sum()), structures_found=len(structs), mapped_equipment_with_structure_within_3m=len(ep),
                  median_structure_top_hag_m=float(np.median([s["top_hag"] for s in structs])) if structs else None, versus_random_points=null),
    )
    np.savez_compressed(OUT / "detections.npz", wmask=wmask, wlab=wlab, hag=hag.astype("float32"), P=P, S=S, ref=ref, equip=equip)
    rows = []
    for c, v in winfo.items():
        p = xyz[wlab == c]
        rows.append(dict(chain=c, points=v["points"], length_m=round(v["length_m"], 1), mean_hag_m=round(v["mean_hag"], 1), easting=round(float(np.median(p[:, 0])) + cx, 1), northing=round(float(np.median(p[:, 1])) + cy, 1)))
    pd.DataFrame(rows).to_csv(OUT / "wire_chains.csv", index=False)
    pd.DataFrame([dict(easting=round(p["x"] + cx, 2), northing=round(p["y"] + cy, 2), height_m=round(p["height"], 1), top_hag_m=round(p["top_hag"], 1), points=p["points"], matched_overture=int(i in {a for a, _, _ in pr})) for i, p in enumerate(poles)]).to_csv(OUT / "detected_poles.csv", index=False)
    pd.DataFrame([dict(easting=round(s["x"] + cx, 2), northing=round(s["y"] + cy, 2), top_hag_m=round(s["top_hag"], 1), extent_m=round(s["extent_m"], 1), points=s["points"]) for s in structs]).to_csv(OUT / "yard_structures.csv", index=False)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    from lidar3d import figures_utility
    A = dict(np.load(OUT / "detections.npz"))
    pt = lambda c: np.array([(g.x, g.y) for g, _ in feats.get(c, []) if g.geom_type == "Point"]).reshape(-1, 2)
    poly = [g for g, _ in feats.get("substation", []) if g.geom_type == "Polygon"]
    figures_utility.fig_overview(stats, A, (x, y, z, cls), hag, figures_utility._lines(feats), pt("power_tower"), pt("power_pole"), poly[0] if poly else None)
    figures_utility.fig_profile(stats, A, (x, y, z, cls), hag, poly[0] if poly else None)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
