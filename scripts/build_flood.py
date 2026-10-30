"""Drainage, culvert-obstruction candidates and flood-stage road vulnerability for Ghent and Hague, Norfolk, Virginia, from 2013 3DEP lidar.
Usage: python scripts/build_flood.py   (downloads the lidar and Overture roads the first time, about 10 minutes)

Stages are water-surface heights in the lidar's vertical datum (NAVD88). The model is a static bathtub (no rainfall, flow, drains or pumps): water reaches any terrain that is below the stage and connected to the
source water (the tidal creek, found as the large void of no returns) or, for ponding, any terrain below the stage."""
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
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from shapely import wkb
from shapely.geometry import LineString

from lidar3d import flood as FL
from lidar3d import forestry as FO
from lidar3d import sitemap
from lidar3d.ept import CACHE

SITE = FL.SITE
HALF = SITE["half"]
RES = 1.0
STAGES = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
DEPTH_DIFFICULT, DEPTH_IMPASSABLE = 0.15, 0.30        # scenario thresholds (m of water on the road): difficult for cars, impassable
OUT = ROOT / "outputs" / "flood"
DATA = ROOT / "data" / "processed" / "flood"


def cached(name, fn):
    p = CACHE / "flood" / name
    if p.exists():
        return pickle.loads(p.read_bytes())
    v = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(v))
    return v


def write_tif(path, arr, aoi, res=RES, dtype="float32", nodata=np.nan):
    a = np.flipud(arr).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype=dtype, crs=aoi.crs, transform=from_origin(aoi.cx - HALF, aoi.cy + HALF, res, res), nodata=nodata, compress="deflate") as d:
        d.write(a, 1)


def overture_roads(aoi, cx, cy):
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, SITE["crs"], always_xy=True)
    t = sitemap._table("transportation", "segment", ["id", "class", "subtype", "geometry", "names", "road_flags", "level_rules"], aoi.bbox_ll)
    out = []
    for _, r in t.iterrows():
        g = wkb.loads(r.geometry)
        for p in (g.geoms if g.geom_type == "MultiLineString" else [g]):
            if p.geom_type != "LineString":
                continue
            x, y = tr.transform(*np.array(p.coords).T)
            q = LineString(np.c_[x - cx, y - cy])
            q = q.intersection(__import__("shapely.geometry", fromlist=["box"]).box(-HALF, -HALF, HALF, HALF))
            for s in (q.geoms if hasattr(q, "geoms") else [q]):
                if s.geom_type == "LineString" and s.length > 5:
                    lst = lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []
                    flags = [v for f in lst(r.road_flags) for v in lst(f.get("values"))]
                    levels = [f.get("value") for f in lst(r.level_rules)]
                    out.append(dict(id=r["id"], cls=r["class"], name=(r.names or {}).get("primary") if isinstance(r.names, dict) else None, coords=np.array(s.coords),
                                    tunnel=("is_tunnel" in flags) or any(v is not None and v < 0 for v in levels), bridge=("is_bridge" in flags) or any(v is not None and v > 0 for v in levels)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    C = FL.load_cloud(SITE, log=lambda s: None)
    aoi = FL.make_aoi(SITE)
    dtm, cnt, road_kept = FL.bare_earth(C, RES)
    water = FL.water_mask(C, RES)
    n = dtm.shape[0]
    iy, ix, _ = FO._cells(C.x, C.y, HALF, RES)
    top = np.full(n * n, -np.inf)
    np.maximum.at(top, iy * n + ix, C.z)
    top = top.reshape(n, n)
    ndsm = np.where(np.isfinite(top), top - dtm, 0.0)
    st = dict(site=SITE["name"], lonlat=SITE["lonlat"], project=SITE["project"], collected=C.date, points=int(len(C.x)), density_per_m2=float(len(C.x) / (2 * HALF) ** 2), classes={int(k): int(v) for k, v in zip(*np.unique(C.cls, return_counts=True))},
              ground_and_road_density_per_m2=float(np.isin(C.cls, (2, 11)).sum() / (2 * HALF) ** 2), res_m=RES, vertical_datum="NAVD88 (3DEP)", road_returns_kept_as_embankment_share=road_kept, empty_cells_share=float((cnt == 0).mean()), water_void_m2=float(water.sum() * RES ** 2))
    # ---- DTM accuracy: hold out 10% of the ground and road returns
    m = (C.cls == 2) | ((C.cls == 11) & np.isfinite(FO.sample(dtm, HALF, RES, C.x, C.y)) & (np.abs(C.z - FO.sample(dtm, HALF, RES, C.x, C.y)) <= 1.0))
    x, y, z, cl = C.x[m], C.y[m], C.z[m], C.cls[m]
    h = np.random.default_rng(0).random(len(x)) < 0.10
    d2, _ = FO.terrain(x[~h], y[~h], z[~h], HALF, RES, max_fill=60.0)
    pr = FO.sample(d2, HALF, RES, x[h], y[h])
    ok = np.isfinite(pr)
    r = (pr - z[h])[ok]
    c_h = cl[h][ok]
    st["dtm_holdout"] = dict(n=int(ok.sum()), rmse_m=float(np.sqrt(np.mean(r ** 2))), mae_m=float(np.mean(np.abs(r))), p95_abs_m=float(np.percentile(np.abs(r), 95)), bias_m=float(np.median(r)),
                             road_surface_rmse_m=float(np.sqrt(np.mean(r[c_h == 11] ** 2))), ground_rmse_m=float(np.sqrt(np.mean(r[c_h == 2] ** 2))))
    # ---- roads
    roads = cached("roads.pkl", lambda: overture_roads(aoi, C.cx, C.cy))
    drivable = [r for r in roads if r["cls"] in FL.DRIVABLE]
    road_mask = FL.road_raster([(r["cls"], r["coords"]) for r in drivable], HALF, RES)
    st["roads"] = dict(segments_total=len(roads), drivable_segments=len(drivable), drivable_km=float(sum(LineString(r["coords"]).length for r in drivable) / 1000), by_class_km={k: float(sum(LineString(r["coords"]).length for r in drivable if r["cls"] == k) / 1000) for k in FL.DRIVABLE})
    # what each segment sits on: bridge deck, at-grade road, a road beneath a structure, or a tunnel
    for r in drivable:
        r.update(FL.road_structure(r["coords"], dtm, top, cnt, HALF, RES, r["bridge"], r["tunnel"]))
        r["elevated"] = r["category"] in ("bridge", "tunnel")           # only these are exempt from flooding
    cat_km = {}
    for k in ("at_grade", "at_grade_open_cut", "at_grade_unobserved", "bridge", "tunnel"):
        cat_km[k] = dict(segments=int(sum(r["category"] == k for r in drivable)), km=float(sum(LineString(r["coords"]).length for r in drivable if r["category"] == k) / 1000))
    bt = [r for r in drivable if r["bridge"]]
    st["roads"]["categories"] = cat_km
    st["roads"]["structure_check"] = dict(
        tagged_bridge=len(bt), tagged_bridge_with_deck_seen=int(sum(r["deck_share"] >= 0.3 for r in bt)), tag_conflicts_treated_at_grade=int(sum(r["tag_conflict"] for r in drivable)),
        untagged_segments_with_anything_overhead_incl_tree_canopy=int(sum((not r["bridge"] and not r["tunnel"]) and r["deck_share"] >= 0.3 for r in drivable)),
        untagged_underpass_candidates_surface_not_seen=int(sum(r["category"] == "at_grade_unobserved" for r in drivable)),
        median_deck_height_of_bridges_m=float(np.median([r["deck_height_m"] for r in bt if r["deck_share"] >= 0.3])) if any(r["deck_share"] >= 0.3 for r in bt) else None,
        old_rule_excluded_km_mostly_tree_covered_streets=float(sum(LineString(r["coords"]).length for r in drivable if (not r["bridge"] and not r["tunnel"]) and r["deck_share"] >= 0.5) / 1000),
        note="old rule: any segment with a return more than 3 m above its centreline (a deck, but also tree canopy or a roof) was treated as elevated and never flooded, which removed tree-lined streets and the road under an overpass from the flood count")
    # ---- hydrology
    H = FL.condition(dtm, HALF, RES, SITE["crs"])
    min_cells = 3000
    stream_raw = H["acc"] >= min_cells
    lab, tab = FL.depressions(dtm, H["filled"], RES, 0.15, 60.0)
    cands, cut = cached("breach.pkl", lambda: FL.breach_paths(dtm, lab, tab, RES, road_mask))
    Hc = FL.condition(cut, HALF, RES, SITE["crs"])
    st["hydrology"] = dict(acc_stream_threshold_cells=min_cells, stream_length_km_raw=float(stream_raw.sum() * RES / 1000), max_acc_cells=float(H["acc"].max()),
                           depressions=int(len(tab["area_m2"])), depression_volume_m3=float(tab["volume_m3"].sum()), depression_area_ha=float(tab["area_m2"].sum() / 1e4), max_depression_depth_m=float(tab["max_depth_m"].max()),
                           cells_in_depression_share=float((H["depth"] >= 0.15).mean()), culvert_candidates=len(cands), candidate_storage_m3=float(sum(c["volume_m3"] for c in cands)), candidate_storage_after_culverts_m3=float(sum(np.clip(Hc["filled"] - cut, 0, None)[lab == c["depression"]].sum() for c in cands) * RES * RES), candidate_excavation_m3=float(sum(c["excavation_m3"] for c in cands)),
                          )
    # candidates annotated with the nearest named road and whether the nearest road is elevated
    rt = [(LineString(r["coords"]), r) for r in drivable]
    for c in cands:
        px, py = c["spill_col"] * RES - HALF + RES / 2, c["spill_row"] * RES - HALF + RES / 2
        from shapely.geometry import Point
        pt = Point(px, py)
        best = min(rt, key=lambda t: t[0].distance(pt))
        c.update(x=px + C.cx, y=py + C.cy, local_x=px, local_y=py, road=best[1]["name"], road_class=best[1]["cls"], road_distance_m=float(best[0].distance(pt)), road_elevated=bool(best[1]["elevated"]))
    pd.DataFrame([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()} for c in cands]).sort_values("volume_m3", ascending=False).to_csv(DATA / "culvert_candidates.csv", index=False)
    # ---- flood stages (raw terrain, and with the candidate culverts opened)
    src = water
    fl = {}
    for lv in STAGES:
        res_ = {}
        for nm, dm in (("raw", dtm), ("culverts_opened", cut)):
            conn = FL.bathtub(dm, lv, src, True)
            low = FL.bathtub(dm, lv, src, False)
            built = (ndsm >= 2.5) & ~road_mask
            res_[nm] = dict(connected_ha=float(conn.sum() * RES ** 2 / 1e4), connected_outside_buildings_ha=float((conn & ~built).sum() * RES ** 2 / 1e4), below_stage_ha=float(low.sum() * RES ** 2 / 1e4),
                            below_stage_outside_buildings_ha=float((low & ~built).sum() * RES ** 2 / 1e4), connected_on_roads_ha=float((conn & road_mask).sum() * RES ** 2 / 1e4), low_on_roads_ha=float((low & road_mask).sum() * RES ** 2 / 1e4))
        # vertical-error band on the connected flooding of the raw terrain
        for dz_ in (0.1, 0.2):
            res_[f"raw_stage_minus_{dz_}"] = float(FL.bathtub(dtm, lv - dz_, src, True).sum() * RES ** 2 / 1e4)
            res_[f"raw_stage_plus_{dz_}"] = float(FL.bathtub(dtm, lv + dz_, src, True).sum() * RES ** 2 / 1e4)
        fl[f"{lv:g}"] = res_
    st["stages"] = fl
    first_conn = FL.first_flood_level(dtm, src, STAGES, True)
    first_low = FL.first_flood_level(dtm, src, STAGES, False)
    # ---- road vulnerability: per segment, depth along the centreline at each stage (ponding scenario: any terrain below the stage; and connected)
    seg_rows, nodes = [], {}
    nid = lambda p: nodes.setdefault((round(p[0], 0), round(p[1], 0)), len(nodes))
    for r in drivable:
        pts, zz = FL.sample_line(dtm, r["coords"], HALF, RES, 2.0, across=0.5 * FL.ROAD_HALF_WIDTH[r["cls"]])
        if r["category"] == "bridge":                      # the road surface is the deck: use the highest returns along the line
            _, zz = FL.sample_line(top, r["coords"], HALF, RES, 2.0)
            zz = np.where(np.isfinite(zz), zz, np.nanmedian(zz))
        _, f_conn = FL.sample_line(np.where(np.isnan(first_conn), 99.0, first_conn), r["coords"], HALF, RES, 2.0)
        length = float(LineString(r["coords"]).length)
        row = dict(id=r["id"], road=r["name"], cls=r["cls"], length_m=length, elevated=r["elevated"], category=r["category"], observed_share=r["observed_share"], deck_share=r["deck_share"], min_elev_m=float(np.nanmin(zz)), p10_elev_m=float(np.nanpercentile(zz, 10)))
        for lv in STAGES:
            d_ = np.clip(lv - zz, 0, None) if r["category"] != "tunnel" else np.zeros_like(zz)
            row[f"max_depth_{lv:g}"] = float(np.nanmax(d_))
            row[f"len_deeper_{DEPTH_IMPASSABLE}_{lv:g}"] = float((d_ >= DEPTH_IMPASSABLE).mean() * length)
            conn_ok = np.nan_to_num(f_conn, nan=99.0) <= lv
            row[f"connected_depth_ge_{DEPTH_IMPASSABLE}_{lv:g}"] = bool(((d_ >= DEPTH_IMPASSABLE) & conn_ok).any())
        a, b = nid(r["coords"][0]), nid(r["coords"][-1])
        row["node_a"], row["node_b"] = a, b
        seg_rows.append(row)
    S = pd.DataFrame(seg_rows)
    segs = list(zip(S.node_a, S.node_b, S.length_m))
    net = {}
    for lv in STAGES:
        tot = float(S.length_m.sum())
        for scen, col in (("any_low_ground", f"max_depth_{lv:g}"), ("connected_to_tidal_water", None)):
            if scen == "any_low_ground":
                imp = (S[f"max_depth_{lv:g}"] >= DEPTH_IMPASSABLE).to_numpy()
                diff = ((S[f"max_depth_{lv:g}"] >= DEPTH_DIFFICULT) & ~imp).to_numpy()
            else:
                imp = S[f"connected_depth_ge_{DEPTH_IMPASSABLE}_{lv:g}"].to_numpy()
                diff = np.zeros(len(S), bool)
            share, comps = FL.network_access(segs, imp)
            net.setdefault(f"{lv:g}", {})[scen] = dict(segments_impassable=int(imp.sum()), km_impassable=float(S.length_m[imp].sum() / 1000), km_difficult_only=float(S.length_m[diff].sum() / 1000), share_of_network_km_impassable=float(S.length_m[imp].sum() / tot),
                                                       largest_connected_share_of_km=float(share), pieces=int(comps))
    st["road_vulnerability"] = net
    by_cls = {}
    for lv in (2.0, 3.0, 4.0):
        imp = S[f"max_depth_{lv:g}"] >= DEPTH_IMPASSABLE
        by_cls[f"{lv:g}"] = {k: dict(km=float(g.length_m.sum() / 1000), km_impassable=float(g.length_m[imp[g.index]].sum() / 1000)) for k, g in S.groupby("cls")}
    st["road_vulnerability_by_class_any_low_ground"] = by_cls
    S["first_impassable_stage"] = [next((lv for lv in STAGES if row[f"max_depth_{lv:g}"] >= DEPTH_IMPASSABLE), np.nan) for row in seg_rows]
    S.round(3).to_csv(DATA / "road_segments.csv", index=False)
    # ---- sensitivity: resolution and conditioning
    sens = {}
    d2m = FO.terrain(C.x[m], C.y[m], C.z[m], HALF, 2.0, max_fill=60.0)[0]
    w2 = ndimage.zoom(water.astype(float), 0.5, order=0) > 0.5
    for lv in (2.5, 3.5):
        sens[f"stage_{lv:g}"] = dict(res_1m_ha=fl[f"{lv:g}"]["raw"]["below_stage_ha"], res_2m_ha=float((d2m < lv).sum() * 4 / 1e4), connected_1m_ha=fl[f"{lv:g}"]["raw"]["connected_ha"], connected_2m_ha=float(FL.bathtub(d2m, lv, w2, True).sum() * 4 / 1e4))
    st["sensitivity_resolution"] = sens
    # ---- planted barrier test: a synthetic valley with an embankment road and a culvert
    st["synthetic_culvert_test"] = cached("synthetic_culvert.pkl", lambda: synthetic_culvert_test(st["dtm_holdout"]["rmse_m"]))
    # ---- outputs
    write_tif(OUT / "dtm_m.tif", dtm, aoi)
    write_tif(OUT / "filled_dem_m.tif", H["filled"], aoi)
    write_tif(OUT / "depression_depth_m.tif", H["depth"], aoi)
    write_tif(OUT / "flow_accumulation_cells.tif", np.where(np.isfinite(H["acc"]), H["acc"], 0), aoi)
    write_tif(OUT / "first_flood_level_connected_m.tif", first_conn, aoi)
    write_tif(OUT / "first_flood_level_any_low_ground_m.tif", first_low, aoi)
    for lv in (2.0, 3.0, 4.0):
        conn = FL.bathtub(dtm, lv, src, True)
        write_tif(OUT / f"flood_depth_stage_{lv:g}m_connected.tif", np.where(conn, lv - dtm, np.nan), aoi)
    pickle.dump(dict(dtm=dtm.astype("float32"), filled=H["filled"].astype("float32"), depth=H["depth"].astype("float32"), acc=H["acc"].astype("float32"), water=water, ndsm=ndsm.astype("float32"), road_mask=road_mask, roads=drivable,
                     first_conn=first_conn.astype("float32"), first_low=first_low.astype("float32"), S=S, cands=cands, lab=lab, tab=tab, cut=cut.astype("float32"), acc_cut=Hc["acc"].astype("float32"), cx=C.cx, cy=C.cy), open(CACHE / "flood" / "final.pkl", "wb"))
    (DATA / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    print(json.dumps(st, indent=1, default=float)[:6000])


def synthetic_culvert_test(noise=0.065, trials=24):
    """A V-shaped valley draining north, crossed by a road embankment (10 m wide) of a given height, with a culvert at the valley floor in half the trials. Noise of the size measured for the real terrain is added.
    The depression search and breach search must find the held-back storage behind the blocked embankments and nothing behind the open ones. Returns detection and false-alarm rates by embankment height."""
    n = 240
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    rng = np.random.default_rng(3)
    out = {}
    for hgt in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        det = {"blocked": 0, "culvert": 0}
        stor = {"blocked": [], "culvert": []}
        for kind in ("blocked", "culvert"):
            for k in range(trials):
                valley = 0.04 * np.abs(xx - 120) + 0.015 * (n - yy) + 6.0 * np.exp(-yy / 12.0)
                y0 = 100 + (k * 7) % 40
                rd = (yy > y0) & (yy < y0 + 10)
                dem = np.where(rd, valley + hgt, valley)
                if kind == "culvert":
                    dem = np.where(rd & (np.abs(xx - 120) < 2), valley, dem)
                dem = dem + rng.normal(0, noise, dem.shape)
                H = FL.condition(dem, n / 2, 1.0, "EPSG:32618")
                lab, tab = FL.depressions(dem, H["filled"], 1.0, 0.15, 50.0)
                c_, _ = FL.breach_paths(dem, lab, tab, 1.0, rd) if len(tab["area_m2"]) else ([], dem)
                det[kind] += int(len(c_) > 0)
                stor[kind].append(float(tab["volume_m3"].sum()) if len(tab["area_m2"]) else 0.0)
        out[f"{hgt:g}"] = dict(embankment_height_m=hgt, blocked_detected=det["blocked"] / trials, open_culvert_flagged=det["culvert"] / trials, blocked_mean_storage_m3=float(np.mean(stor["blocked"])), culvert_mean_storage_m3=float(np.mean(stor["culvert"])))
    return dict(noise_m=noise, trials_per_cell=trials, by_embankment_height=out)


if __name__ == "__main__":
    main()
