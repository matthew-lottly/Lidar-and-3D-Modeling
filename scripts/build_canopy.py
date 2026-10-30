"""Canopy structure pipeline: 3DEP lidar -> terrain, surface and canopy height models -> individual trees -> density, gaps and fragmentation, with accuracy tests.
Usage: python scripts/build_canopy.py [houston|tahoe|all]    (first run downloads the lidar; about 10 minutes per site)

Per site, writes GeoTIFFs (dtm, dsm, chm, crowns, tree_density_per_ha, canopy_gaps), trees.csv and an interactive map to outputs/canopy/<site>/, stats to data/processed/canopy/<site>.json and the plates."""
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

from lidar3d import connectivity as CN
from lidar3d import forestry as F
from lidar3d import naip
from lidar3d.ept import CACHE

RES = 1.0                      # model resolution, m
MAP_GAP, MAP_ALPHA = 3.0, 5.0   # gap-crossing distance and reach scale used for the connectivity maps (m): crowns nearly touching; 10 m and wider gaps join almost everything at both sites
NAIP_YEAR = dict(houston=2018, tahoe=2022)


def cached(name, fn):
    p = CACHE / "forestry" / name
    if p.exists():
        return pickle.loads(p.read_bytes())
    v = fn()
    p.write_bytes(pickle.dumps(v))
    return v


def run_pipeline(P, half, thin=1.0, res=RES, seed=0, max_h=70.0):
    """Points -> (dtm, dsm, chm, labels, trees, parts). `thin` keeps that random share of the returns (used by the density test)."""
    if thin < 1.0:
        keep = np.random.default_rng(seed).random(len(P.x)) < thin
        P = F.Points(P.x[keep], P.y[keep], P.z[keep], P.cls[keep], P.ret[keep], P.nret[keep], P.date, P.cx, P.cy)
    g = P.cls == 2
    dtm, gcnt = F.terrain(P.x[g], P.y[g], P.z[g], half, 1.0)
    hag = F.height_above_ground(P, dtm, half, 1.0)
    gr, veg, bld, has_veg = F.classify(P, hag)
    m = np.isfinite(hag) & (~gr) & (~bld) & (veg | ~has_veg)
    m &= (hag >= 0.0) & (hag <= max_h)          # returns above the tallest plausible tree (towers, cranes, birds) are not canopy
    chm = F.canopy_height_model(P.x[m], P.y[m], hag[m], half, res)
    dsm_h = chm
    if res != 1.0:
        dtm_r = np.kron(dtm, np.ones((int(round(1 / res)),) * 2)) if res < 1 else dtm
    else:
        dtm_r = dtm
    lab, trees = F.segment_crowns(chm, res)
    return dict(dtm=dtm, dtm_r=dtm_r, gcnt=gcnt, chm=chm, dsm=dtm_r + dsm_h if dtm_r.shape == chm.shape else None, lab=lab, trees=trees, hag=hag, masks=(gr, veg, bld, has_veg), P=P)


def main(key):
    site = F.SITES[key]
    half = site["half"]
    OUT = ROOT / "outputs" / "canopy" / key
    DATA = ROOT / "data" / "processed" / "canopy"
    DATA.mkdir(parents=True, exist_ok=True)
    P = F.load_points(site)
    aoi = F.make_aoi(site)
    R = run_pipeline(P, half, max_h=site["max_height"])
    chm, lab, trees, dtm = R["chm"], R["lab"], R["trees"], R["dtm"]
    gr, veg, bld, has_veg = R["masks"]
    st = dict(site=site["name"], key=key, project=site["project"], collected=P.date, area_ha=(2 * half) ** 2 / 1e4, lonlat=site["lonlat"], crs=site["crs"], points=int(len(P.x)), density_per_m2=float(len(P.x) / (2 * half) ** 2),
              classes={int(c): int(n) for c, n in zip(*np.unique(P.cls, return_counts=True))}, vegetation_classes_present=bool(has_veg), ground_returns=int(gr.sum()), ground_density_per_m2=float(gr.sum() / (2 * half) ** 2),
              multi_return_share=float(np.mean(P.nret > 1)), res_m=RES)
    # ---- structure
    cover = chm >= 2
    st["structure"] = dict(canopy_cover_ge2m=float(cover.mean()), cover_ge5m=float((chm >= 5).mean()), mean_chm_m=float(chm.mean()), mean_canopy_height_m=float(chm[cover].mean()), p95_m=float(np.percentile(chm[cover], 95)), max_m=float(chm.max()),
                           rumple_note="surface roughness not computed", terrain_relief_m=float(np.nanmax(dtm) - np.nanmin(dtm)))
    st["trees"] = dict(n=int(len(trees["height"])), per_ha=float(len(trees["height"]) / st["area_ha"]), mean_height_m=float(trees["height"].mean()), p95_height_m=float(np.percentile(trees["height"], 95)),
                       median_crown_diameter_m=float(np.median(trees["crown_diameter_m"])), crown_cover_of_canopy=float((lab > 0).sum() / max(cover.sum(), 1)),
                       height_classes={"3-10": int(((trees["height"] >= 3) & (trees["height"] < 10)).sum()), "10-20": int(((trees["height"] >= 10) & (trees["height"] < 20)).sum()), "20-30": int(((trees["height"] >= 20) & (trees["height"] < 30)).sum()), "30+": int((trees["height"] >= 30).sum())})
    st["fragmentation"] = F.fragmentation_metrics(chm, RES)
    gl, gaps = F.canopy_gaps(chm, RES)
    a = gaps["area_m2"]
    bins = [(10, 100), (100, 1000), (1000, 1e4), (1e4, 1e9)]
    st["gaps"] = dict(n=int(len(a)), open_area_share=float((chm < 2).mean()), by_size={f"{lo:g}-{hi:g} m2": dict(n=int(((a >= lo) & (a < hi)).sum()), area_share_of_openings=float(a[(a >= lo) & (a < hi)].sum() / max(a.sum(), 1))) for lo, hi in bins},
                      touching_edge=int(gaps["touches_edge"].sum()), max_distance_to_canopy_m=float(ndimage.distance_transform_edt(chm < 2, sampling=RES).max()))
    # ---- accuracy tests
    # (1) terrain: hold out 10% of ground returns
    gx, gy, gz = P.x[gr], P.y[gr], P.z[gr]
    h = np.random.default_rng(0).random(len(gx)) < 0.10
    d2, _ = F.terrain(gx[~h], gy[~h], gz[~h], half, 1.0)
    pr = F.sample(d2, half, 1.0, gx[h], gy[h])
    res_ = pr - gz[h]
    okm = np.isfinite(res_)
    cvals = F.sample(chm.astype("float32"), half, RES, gx[h], gy[h])
    under = cvals >= 5

    def rm(sel):
        r = res_[okm & sel]
        return dict(n=int(len(r)), rmse_m=float(np.sqrt(np.mean(r ** 2))), bias_m=float(np.median(r)), p95_abs_m=float(np.percentile(np.abs(r), 95)))
    st["terrain_holdout"] = dict(all=rm(np.ones_like(okm)), open_ground=rm(~under), under_canopy_ge5m=rm(under))
    # (2) thinning: keep 50% and 25% of the returns, compare trees
    thin = {}
    base_xy = np.c_[trees["col"] + 0.5 - half, trees["row"] + 0.5 - half]
    for f in (0.5, 0.25):
        Rt = cached(f"{key}_thin{f}.pkl", lambda f=f: (lambda r: dict(trees=r["trees"], chm=r["chm"].astype("float32")))(run_pipeline(P, half, thin=f, seed=1, max_h=site["max_height"])))
        tt = Rt["trees"]
        xy = np.c_[tt["col"] + 0.5 - half, tt["row"] + 0.5 - half]
        m = F.match_trees(base_xy, trees["height"], xy, tt["height"], np.full(len(xy), 3.0), min_tol=2.0, frac=0.0)   # base trees as 'detections' of the thinned set: matched within 2 m
        mm = F.match_trees(xy, tt["height"], base_xy, trees["height"], np.full(len(base_xy), 3.0), min_tol=2.0, frac=0.0)
        ok = mm >= 0
        thin[str(f)] = dict(trees=int(len(xy)), tree_count_change=float(len(xy) / len(trees["height"]) - 1), share_of_full_trees_found=float(ok.mean()), height_bias_m=float(np.mean(tt["height"][mm[ok]] - trees["height"][ok])),
                            height_rmse_m=float(np.sqrt(np.mean((tt["height"][mm[ok]] - trees["height"][ok]) ** 2))), cover_change=float((Rt["chm"] >= 2).mean() - cover.mean()))
        del m
    st["thinning"] = thin
    # (3) CHM resolution
    rs = {}
    chm_by_res = {}
    for r_ in (0.5, 2.0):
        Rr = cached(f"{key}_res{r_}.pkl", lambda r_=r_: (lambda r: dict(n=len(r["trees"]["height"]), cover=float((r["chm"] >= 2).mean()), mean_h=float(r["trees"]["height"].mean()), chm=r["chm"].astype("float32")))(run_pipeline(P, half, res=r_, max_h=site["max_height"])))
        chm_by_res[r_] = Rr.pop("chm")
        rs[str(r_)] = Rr
    rs["1.0"] = dict(n=int(len(trees["height"])), cover=float(cover.mean()), mean_h=float(trees["height"].mean()))
    st["resolution_sensitivity"] = rs
    # (4) synthetic stands with known trees
    syn = {}
    ph = float(np.clip(st["density_per_m2"], 4, 30))
    for kind, sp in (("regular", 5.5), ("clumped", 3.0)):
        for dens in (100, 200, 400):
            Ps, T = F.synthetic_stand(dens, pulse_density=ph, seed=2, min_spacing=sp, mean_h=float(trees["height"].mean()), sd_h=float(trees["height"].std()))
            chs = F.canopy_height_model(Ps[:, 0], Ps[:, 1], Ps[:, 2], 100.0, RES)
            lb, tr = F.segment_crowns(chs, RES)
            xy = np.c_[tr["col"] + 0.5 - 100, tr["row"] + 0.5 - 100]
            m = F.match_trees(xy, tr["height"], np.c_[T["x"], T["y"]], T["height"], T["crown_radius"])
            ok = m >= 0
            hh = T["height"]
            q = np.percentile(hh, [33, 66])
            syn[f"{kind}_{dens}"] = dict(true=int(len(hh)), detected=int(len(xy)), recall=float(ok.mean()), precision=float(ok.sum() / max(len(xy), 1)),
                                         recall_by_height_tercile=[float(ok[s].mean()) for s in (hh < q[0], (hh >= q[0]) & (hh < q[1]), hh >= q[1])], height_rmse_m=float(np.sqrt(np.mean((tr["height"][m[ok]] - hh[ok]) ** 2))) if ok.any() else None,
                                         actual_trees_per_ha=float(len(hh) / 4.0), pulse_density=ph)
    st["synthetic_stands"] = syn
    # (5) canopy cover against NAIP vegetation index
    img, tf, ndate = naip.fetch(aoi, NAIP_YEAR[key], res=1.0, slug=site["slug"])
    r_, n_ = img[0].astype("float32"), img[3].astype("float32")
    nd = np.flipud((n_ - r_) / np.maximum(n_ + r_, 1.0))                      # row 0 south, like the CHM
    from skimage.filters import threshold_otsu
    thr = float(threshold_otsu(nd))
    best = None
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            c2 = np.roll(np.roll(cover, dy, 0), dx, 1)
            agree = float(((nd > thr) == c2).mean())
            if best is None or agree > best[0]:
                best = (agree, dy, dx)
    c2 = np.roll(np.roll(cover, best[1], 0), best[2], 1)
    nv = nd > thr
    open_ = ~c2
    st["naip_agreement"] = dict(naip_date=ndate, ndvi_threshold_otsu=thr, best_shift_px=[best[1], best[2]], lidar_canopy_share=float(c2.mean()), naip_vegetation_share=float(nv.mean()),
                                lidar_canopy_that_naip_calls_vegetation=float((nv & c2).sum() / max(c2.sum(), 1)), lidar_open_that_naip_calls_vegetation=float((nv & open_).sum() / max(open_.sum(), 1)),
                                note="NAIP vegetation includes lawn, grass and shrubs, so only the first ratio is a check on the lidar canopy; the second is mostly lawn and low vegetation")
    # (6) connectivity of the canopy: graph of patches (20 m2 or larger) and gaps between them
    land_m2 = (2 * half) ** 2
    g1 = CN.patch_graph(chm, RES, 2.0, 20.0)
    st["connectivity"] = CN.summary(g1, land_m2)
    st["connectivity"]["map_gap_m"], st["connectivity"]["map_alpha_m"] = MAP_GAP, MAP_ALPHA
    cl_map, ca_map = CN.clusters(g1, MAP_GAP)
    rmap = CN.pc_metrics(g1, MAP_ALPHA, land_m2)
    stones = CN.stepping_stones(g1, MAP_GAP)
    st["connectivity"]["stepping_stones"] = int(stones.sum())
    st["connectivity"]["link_gaps"] = dict(median_m=float(np.median(g1["edges"][:, 2])) if len(g1["edges"]) else None, links_le_3m=int((g1["edges"][:, 2] <= 3).sum()), links_le_10m=int((g1["edges"][:, 2] <= 10).sum()))
    sens = {}

    def conn_of(ch, rs_, min_h=2.0, min_area=20.0):
        gg = CN.patch_graph(ch, rs_, min_h, min_area)
        lab_, ca_ = CN.clusters(gg, MAP_GAP)
        r_ = CN.pc_metrics(gg, MAP_ALPHA, land_m2)
        return dict(nodes=int(gg["n"]), largest_cluster_share=float(ca_.max() / gg["area"].sum()), clusters=int(len(ca_)), eca_share_of_canopy=float(r_["eca_m2"] / gg["area"].sum()))
    sens["baseline_1m_h2_a20"] = conn_of(chm, RES)
    sens["cell_0.5m"] = conn_of(chm_by_res[0.5], 0.5)
    sens["cell_2m"] = conn_of(chm_by_res[2.0], 2.0)
    sens["canopy_5m_or_higher"] = conn_of(chm, RES, min_h=5.0)
    sens["min_patch_10m2"] = conn_of(chm, RES, min_area=10.0)
    sens["min_patch_50m2"] = conn_of(chm, RES, min_area=50.0)
    for f in ("0.5", "0.25"):
        sens["half_returns" if f == "0.5" else "quarter_returns"] = conn_of(cached(f"{key}_thin{f}.pkl", None)["chm"], RES)
    st["connectivity"]["sensitivity"] = sens
    # ---- outputs
    F.write_geotiff(OUT / "dtm.tif", dtm, aoi, 1.0, nodata=-9999.0)
    F.write_geotiff(OUT / "chm.tif", chm, aoi, RES, nodata=-9999.0)
    F.write_geotiff(OUT / "dsm.tif", dtm + chm, aoi, RES, nodata=-9999.0)
    F.write_geotiff(OUT / "crowns.tif", lab, aoi, RES, dtype="int32", nodata=0)
    dens = F.density_grid(trees["row"], trees["col"], chm.shape, RES, 50.0)
    F.write_geotiff(OUT / "tree_density_per_ha.tif", dens, aoi, 50.0)
    cl_img = np.where(g1["labels"] > 0, np.concatenate([[0], cl_map + 1])[g1["labels"]], 0)
    F.write_geotiff(OUT / "canopy_clusters_3m_gap.tif", cl_img, aoi, RES, dtype="int32", nodata=0)
    F.write_geotiff(OUT / "canopy_reachable_area_ha.tif", np.concatenate([[0.0], rmap["reach_m2"] / 1e4])[g1["labels"]], aoi, RES, nodata=0.0)
    F.write_geotiff(OUT / "canopy_stepping_stones.tif", np.concatenate([[False], stones])[g1["labels"]].astype("uint8"), aoi, RES, dtype="uint8", nodata=0)
    F.write_geotiff(OUT / "canopy_gaps.tif", (gl > 0).astype("uint8"), aoi, RES, dtype="uint8", nodata=0)
    from pyproj import Transformer
    ex, ey = trees["col"] + 0.5 - half + aoi.cx, trees["row"] + 0.5 - half + aoi.cy
    lon, lat = Transformer.from_crs(aoi.crs, "EPSG:4326", always_xy=True).transform(ex, ey)
    import pandas as pd
    pd.DataFrame(dict(tree_id=np.arange(1, len(ex) + 1), utm_e=np.round(ex, 2), utm_n=np.round(ey, 2), lon=np.round(lon, 7), lat=np.round(lat, 7), height_m=np.round(trees["height"], 2),
                      crown_area_m2=np.round(trees["crown_area_m2"], 1), crown_diameter_m=np.round(trees["crown_diameter_m"], 2))).to_csv(OUT / "trees.csv", index=False)
    cen_e, cen_n = g1["centroid"][:, 1] + 0.5 - half + aoi.cx, g1["centroid"][:, 0] + 0.5 - half + aoi.cy
    pd.DataFrame(dict(patch_id=np.arange(1, g1["n"] + 1), utm_e=np.round(cen_e, 1), utm_n=np.round(cen_n, 1), area_m2=np.round(g1["area"], 1), cluster_3m_gap=cl_map + 1, reachable_area_ha_alpha5=np.round(rmap["reach_m2"] / 1e4, 3),
                      stepping_stone=stones.astype(int), touches_window_edge=g1["touches_edge"].astype(int))).to_csv(OUT / "canopy_patches.csv", index=False)
    pd.DataFrame(dict(patch_a=g1["edges"][:, 0].astype(int) + 1, patch_b=g1["edges"][:, 1].astype(int) + 1, gap_m=np.round(g1["edges"][:, 2], 2))).query("gap_m <= 50").to_csv(OUT / "canopy_links.csv", index=False)
    st["density_grid"] = dict(cell_m=50.0, mean_trees_per_ha=float(dens.mean()), p90_trees_per_ha=float(np.percentile(dens, 90)), max_trees_per_ha=float(dens.max()))
    (DATA / f"{key}.json").write_text(json.dumps(st, indent=1, default=float))
    pickle.dump(dict(graph=g1, cl_map=cl_map, ca_map=ca_map, reach_map=rmap["reach_m2"], stones=stones, dtm=dtm.astype("float32"), chm=chm.astype("float32"), lab=lab, trees=trees, gl=gl, dens=dens, nd=nd.astype("float32"), gcnt=R["gcnt"].astype("float32")), open(CACHE / "forestry" / f"{key}_final.pkl", "wb"))
    print(json.dumps(st, indent=1, default=float))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k in (["houston", "tahoe"] if which == "all" else [which]):
        main(k)
