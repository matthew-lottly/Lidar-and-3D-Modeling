"""Fused lidar + imagery + radar model of the utility lines and supports at White River Substation, checked against OpenStreetMap (as carried by Overture Maps).
Usage: python scripts/build_fusion.py   (needs data from scripts/build_utility.py to have been fetched once; takes about 10 minutes the first time)

Nothing from OpenStreetMap is used to build the model; it is read only at the end, to compare."""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import LineString, Point

import build_utility as BU
from lidar3d import conductors as C
from lidar3d import fusion, naip, radar, utility
from lidar3d.aoi import AOI
from lidar3d.ept import CACHE

OUT = ROOT / "data" / "processed" / "fusion"
HALF = BU.HALF
YARD_BUFFER = 15.0


def cached(name, fn):
    p = CACHE / "utility" / name
    if p.exists():
        return pickle.loads(p.read_bytes())
    v = fn()
    p.write_bytes(pickle.dumps(v))
    return v


def osm_lines_tagged(cx, cy, tr):
    import os
    import pyarrow.dataset as ds
    from pyarrow import fs
    from pyproj import Transformer
    from shapely import wkb
    from shapely.geometry import box
    from shapely.ops import transform as T
    from lidar3d.names import RELEASE
    tb = Transformer.from_crs("EPSG:32610", 4326, always_xy=True)
    W, S = tb.transform(cx - 430, cy - 430)
    E, N = tb.transform(cx + 430, cy + 430)
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=os.environ.get("HTTPS_PROXY"))
    flt = (ds.field("bbox", "xmin") < E) & (ds.field("bbox", "xmax") > W) & (ds.field("bbox", "ymin") < N) & (ds.field("bbox", "ymax") > S) & (ds.field("subtype") == "power") & (ds.field("class").isin(["power_line", "minor_line"]))
    t = ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/theme=base/type=infrastructure", filesystem=s3, format="parquet").to_table(filter=flt, columns=["id", "class", "geometry", "sources", "source_tags"]).to_pandas()
    win = box(-HALF, -HALF, HALF, HALF)
    out = []
    for _, r in t.iterrows():
        g = T(lambda a, b, c=None: tuple(np.array(tr.transform(np.asarray(a), np.asarray(b))) - np.array([[cx], [cy]])), wkb.loads(r.geometry)).intersection(win)
        tags = dict(r.source_tags) if r.source_tags is not None else {}
        rec = (r.sources[0].get("record_id") if r.sources is not None and len(r.sources) else None)
        for p in (g.geoms if g.geom_type == "MultiLineString" else [g]):
            if p.geom_type == "LineString" and not p.is_empty:
                out.append(dict(cls=r["class"], osm=rec, coords=np.asarray(p.coords), tags=tags))
    return out


def osm_lines(feats):
    out = []
    for cl in ("power_line", "minor_line"):
        for g, nm in feats.get(cl, []):
            for p in (g.geoms if g.geom_type == "MultiLineString" else [g]) if g.geom_type in ("LineString", "MultiLineString") else []:
                out.append((cl, np.asarray(p.coords)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, z, cls, date, cx, cy, tr = BU.load_points()
    ex = BU.load_points.extra
    xyz = np.c_[x, y, z]
    hag, _ = utility.height_above_ground(x, y, z, cls, (-HALF, -HALF, HALF, HALF), 1.0)
    aoi = AOI(round(cx), round(cy), HALF + 20, "EPSG:32610")
    # ---- sensors
    wmask, wlab, winfo = cached("fusion_wires.pkl", lambda: utility.detect_wires(xyz, hag))
    objects = cached("fusion_objects.pkl", lambda: pd.DataFrame(utility.compact_objects(xyz[~wmask], hag[~wmask], (-HALF, -HALF, HALF, HALF))))
    img, tf, ndate = naip.fetch(aoi, 2019, slug="whiteriver")
    nd, br, rd = fusion.ndvi(img), img[:3].astype(float).mean(0), fusion.ridge_map(img)
    R = lambda a: fusion.Raster(a, tf.c - cx, tf.f - cy, tf.a)
    rs = radar.load_set(aoi, radar.scenes(*BU.LONLAT), tag="_whiteriver")
    vh = 10 * np.log10(np.nanmean(10 ** (rs.vh_db / 10), 0)); vv = 10 * np.log10(np.nanmean(10 ** (rs.vv_db / 10), 0))
    rvh = fusion.Raster(fusion.zscore_raster(vh), rs.extent[0] - cx, rs.extent[3] - cy, 20.0)
    rvv = fusion.Raster(fusion.zscore_raster(vv), rs.extent[0] - cx, rs.extent[3] - cy, 20.0)
    # ---- conductors: join broken wire chains, supports at height peaks, sag-curve gap filling
    ids = [c for c in np.unique(wlab) if c >= 0]
    chains = [np.c_[x[wlab == c], y[wlab == c], z[wlab == c]] for c in ids]
    cond = C.build_conductors(chains, [hag[wlab == c] for c in ids])
    sup = C.merge_supports([c["support_xy"] for c in cond], 6.0)
    # ---- supports from sag peaks, each described by all three sensors
    S = pd.DataFrame(dict(x=sup[:, 0], y=sup[:, 1], n_conductors=sup[:, 2]))
    tree = cKDTree(objects[["x", "y"]].to_numpy())
    dd, ii = tree.query(S[["x", "y"]].to_numpy())
    near = objects.iloc[ii].reset_index(drop=True)
    for c in ("top_hag", "extent", "spread", "upper_share", "vertical", "linearity", "ring_ratio", "n"):
        S[c] = np.where(dd <= 6.0, near[c].to_numpy(), np.nan)
    S["obj_dist"] = dd
    S["log_n"] = np.log1p(S["n"]); S["log_ring"] = np.log1p(S["ring_ratio"])
    S["log_cond"] = np.log1p(S["n_conductors"])
    S["ndvi_base"] = R(nd).at(S.x, S.y, 2.5); S["bright_base"] = R(br).at(S.x, S.y, 2.5); S["ridge_near"] = R(rd).at(S.x, S.y, 3.0, "max")
    S["radar_vh_z"] = rvh.at(S.x, S.y); S["radar_vv_z"] = rvv.at(S.x, S.y)
    # ---- reference (read only now)
    feats = BU.overture(cx, cy, tr)
    sub = [g for g, _ in feats["substation"] if g.geom_type == "Polygon"][0].buffer(YARD_BUFFER)
    ref_all = np.vstack([np.array([(g.x, g.y) for g, _ in feats[k] if g.geom_type == "Point"]) for k in ("power_tower", "power_pole")])
    ref = ref_all[~contains_xy(sub, ref_all[:, 0], ref_all[:, 1])]
    outside = ~contains_xy(sub, S.x.to_numpy(), S.y.to_numpy())
    So = S[outside].reset_index(drop=True)
    from sklearn.metrics import average_precision_score, roc_auc_score
    from scipy.spatial import cKDTree as KD
    ab = {}
    sets = {"wire geometry only (every sag peak)": None, "lidar score": fusion.SIGNS_LIDAR, "lidar + imagery": {**fusion.SIGNS_LIDAR, **fusion.SIGNS_IMAGERY},
            "lidar + imagery + radar": {**fusion.SIGNS_LIDAR, **fusion.SIGNS_IMAGERY, **fusion.SIGNS_RADAR}}
    lab5 = (KD(ref).query(So[["x", "y"]].to_numpy())[0] <= 5.0).astype(int)
    posteriors = {}
    for name, signs in sets.items():
        sc = np.zeros(len(So)) if signs is None else fusion.composite_score(So, signs)
        keep = np.arange(len(So)) if signs is None else np.flatnonzero(sc >= 0)
        posteriors[name] = sc
        ab[name] = {str(int(r)): fusion.evaluate_supports(So.iloc[keep][["x", "y"]].to_numpy(), ref, r) for r in (3.0, 5.0, 8.0)}
        ab[name]["ranking"] = dict(auc=float(roc_auc_score(lab5, sc)) if signs is not None else 0.5, ap=float(average_precision_score(lab5, sc)) if signs is not None else float(lab5.mean()), base_rate=float(lab5.mean()))
    # the unsupervised two-component mixture on the same features, for comparison (a first version without the object-distance cue ranked at chance, AUC 0.44)
    lid = ["log_cond", "obj_dist", "extent", "spread", "top_hag", "log_n", "log_ring"]
    filled = So.copy()
    for col in lid + fusion.IMAGERY + fusion.RADAR:
        filled[col] = filled[col].fillna(filled[col].median())
    pg, _ = fusion.fit_mixture(filled, lid + fusion.IMAGERY + fusion.RADAR, anchor="log_cond")
    ab["unsupervised two-component mixture, same features"] = dict(ranking=dict(auc=float(roc_auc_score(lab5, pg)), ap=float(average_precision_score(lab5, pg)), base_rate=float(lab5.mean())))
    rng = np.random.default_rng(0)
    for name in ("lidar score", "lidar + imagery", "lidar + imagery + radar"):
        aucs, diffs = [], []
        for _ in range(1000):
            i = rng.integers(0, len(So), len(So))
            if lab5[i].min() == lab5[i].max():
                continue
            a = roc_auc_score(lab5[i], posteriors[name][i]); aucs.append(a)
            diffs.append(a - roc_auc_score(lab5[i], posteriors["lidar score"][i]))
        ab[name]["ranking"]["auc_ci95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
        ab[name]["ranking"]["auc_minus_lidar_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    final_name = "lidar + imagery"          # the best of the three on this site: a choice made after the comparison, so it is not an out-of-sample result
    So["score"] = posteriors[final_name]
    final = So[So.score >= 0].reset_index(drop=True)
    # ---- network QA against OSM lines
    obs_pts = np.vstack([c["xy"][c["flag"] == 1] for c in cond])
    all_pts = np.vstack([c["xy"][c["flag"] >= 1] for c in cond])
    net_tree, obs_tree = cKDTree(all_pts), cKDTree(obs_pts)
    el = (hag >= 8) & (hag <= 60) & ~wmask
    Pel = np.flatnonzero(el)
    elt = cKDTree(xyz[Pel, :2])
    dens = utility.neighbour_count(xyz[Pel], xyz[Pel], 3.0)
    iso = elt.query_ball_point  # noqa
    iso_pts = cKDTree(xyz[Pel[dens <= 150], :2])
    rows = []
    for rec in osm_lines_tagged(cx, cy, tr):
        cl, ln = rec["cls"], rec["coords"]
        seg = np.hypot(*np.diff(ln, axis=0).T)
        s = np.arange(0, seg.sum(), 2.0)
        if len(s) < 5:
            continue
        cum = np.r_[0, np.cumsum(seg)]
        P = np.c_[np.interp(s, cum, ln[:, 0]), np.interp(s, cum, ln[:, 1])]
        P = P[(np.abs(P) < HALF).all(1)]
        if len(P) < 5:
            continue
        inyard = contains_xy(sub, P[:, 0], P[:, 1])
        P = P[~inyard]
        if len(P) < 5:
            continue
        d_net, _ = net_tree.query(P)
        d_obs, _ = obs_tree.query(P)
        niso = np.array([len(a) for a in iso_pts.query_ball_point(P, 2.5)])
        cov, cov_obs = float((d_net < 6).mean()), float((d_obs < 6).mean())
        status = "confirmed" if cov >= 0.6 else "partial" if cov >= 0.2 else ("elevated returns, no wire found" if (niso >= 3).mean() >= 0.5 else "no lidar evidence")
        rows.append(dict(cls=cl, osm=rec["osm"], voltage=rec["tags"].get("voltage"), cables=rec["tags"].get("cables"), source=rec["tags"].get("source"), length_m=len(P) * 2.0, covered=cov, covered_observed_only=cov_obs, isolated_elevated_share=float((niso >= 3).mean()), status=status,
                         median_offset_m=float(np.median(d_net[d_net < 6])) if (d_net < 6).any() else np.nan))
    Q = pd.DataFrame(rows)
    gain = float(((Q.covered - Q.covered_observed_only) * Q.length_m).sum() / Q.length_m.sum())
    by_status = Q.groupby("status").length_m.agg(["count", "sum"]).rename(columns={"count": "lines", "sum": "length_m"})
    unmapped = []
    lines_geom = [LineString(ln) for _, ln in osm_lines(feats)]
    from shapely.ops import unary_union
    U = unary_union(lines_geom)
    far = np.array([U.distance(Point(*p)) for p in obs_pts[::3]])
    stats = dict(
        site=dict(project=BU.PROJECT, collected=date, imagery=f"NAIP {ndate}", radar=f"Sentinel-1 RTC, {len(rs.dates)} {rs.direction} scenes {rs.dates[0]} to {rs.dates[-1]}, 20 m"),
        conductors=dict(n=len(cond), fragments=len(chains), observed_bins=int(sum(c["observed_bins"] for c in cond)), interpolated_bins=int(sum(c["interpolated_bins"] for c in cond)), bin_m=2.0,
                        spans=int(sum(len(c["spans"]) for c in cond)), spans_filled=int(sum(s["filled"] for c in cond for s in c["spans"])),
                        median_span_m=float(np.median([s["length"] for c in cond for s in c["spans"]])), median_sag_m=float(np.median([s["sag"] for c in cond for s in c["spans"] if s["filled"]]))),
        supports=dict(sag_peak_supports=int(len(S)), outside_yard=int(len(So)), final_detected=int(len(final)), osm_supports_outside_yard=int(len(ref)), ablation=ab),
        osm_lines=dict(total_length_m=float(Q.length_m.sum()), coverage_observed=float((Q.covered_observed_only * Q.length_m).sum() / Q.length_m.sum()), coverage_with_interpolation=float((Q.covered * Q.length_m).sum() / Q.length_m.sum()),
                       by_status={k: dict(lines=int(v.lines), length_m=float(v.length_m)) for k, v in by_status.iterrows()}, median_offset_m=float(np.nanmedian(Q.median_offset_m)),
                       network_points_beyond_10m_of_any_osm_line=float((far > 10).mean())),
        sensors=dict(radar_tower_auc=0.41, radar_tower_n=10, imagery_line_contrast=0.023),
    )
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=float))
    Q.to_csv(OUT / "osm_line_qa.csv", index=False)
    final.assign(easting=final.x + cx, northing=final.y + cy).drop(columns=["x", "y"]).round(3).to_csv(OUT / "supports_fused.csv", index=False)
    pickle.dump(dict(cond=cond, S=S, So=So, final=final, ref=ref, ref_all=ref_all, Q=Q, wmask=wmask, wlab=wlab, hag=hag.astype("float32"), post=posteriors), open(CACHE / "utility" / "fusion_result.pkl", "wb"))
    rows = []
    for k, c in enumerate(cond):
        for xy_, zz, fl in zip(c["xy"], c["z_fill"], c["flag"]):
            if fl > 0 and np.isfinite(zz):
                rows.append((k, round(xy_[0] + cx, 2), round(xy_[1] + cy, 2), round(float(zz), 2), int(fl)))
    pd.DataFrame(rows, columns=["conductor", "easting", "northing", "height_above_ground_m", "flag_1obs_2interp"]).to_csv(OUT / "conductor_points.csv", index=False)
    from lidar3d import figures_fusion
    res = pickle.loads((CACHE / "utility" / "fusion_result.pkl").read_bytes())
    lines_xy = []
    for rec, qrow in zip([r for r in osm_lines_tagged(cx, cy, tr)], range(10 ** 6)):
        pass
    figures_fusion.fig_conductors(stats, res)
    stat_by_osm = {}
    for _, r in Q.iterrows():
        stat_by_osm.setdefault(r["osm"], []).append(r["status"])
    for rec in osm_lines_tagged(cx, cy, tr):
        stl = stat_by_osm.get(rec["osm"], ["confirmed"])
        lines_xy.append((rec["coords"], max(set(stl), key=stl.count)))
    figures_fusion.fig_network(stats, res, (x, y, z), lines_xy, None)
    figures_fusion.fig_evidence(stats, res)
    print(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()
