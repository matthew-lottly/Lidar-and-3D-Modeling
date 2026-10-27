"""Seattle Center buildings: lidar heights, LOD1 models, plates and exports. Usage: python scripts/build_seattle.py"""
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import trimesh

from lidar3d import buildings, city, exports, figures_city, naip, needle
from lidar3d.cloud import Rasters, hillshade, multi_hillshade
from lidar3d.ept import CACHE


def main():
    out = ROOT / "data" / "processed" / "seattle"
    out.mkdir(parents=True, exist_ok=True)
    aoi = city.make_aoi()
    c = city.load_cloud(aoi)
    foot = city.overture_buildings(aoi)
    from shapely.geometry import box
    bx = box(*aoi.bounds)
    foot["clipped"] = ~foot.geometry.within(bx)                       # cut by the edge of the study area: measured on the part inside
    foot["geometry"] = foot.geometry.intersection(bx)
    foot = foot[foot.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    g = c.gps[c.gps > 0]
    d0, d1 = [(dt.datetime(1980, 1, 6) + dt.timedelta(seconds=float(v) + 1e9)).date() for v in np.percentile(g, [0.5, 99.5])]
    dd = str(d0) if d0 == d1 else f"{d0} to {d1}"
    B, ras = buildings.analyse(aoi, c, foot)
    nrow = B[B.name == "Space Needle"].iloc[0]
    npts, naxis = needle.extract(c, nrow.geometry, nrow.ground_z)
    nprof = needle.profile(npts)
    nmesh = needle.lathe(nprof, naxis, nrow.ground_z)
    assert nmesh.is_watertight and nmesh.volume > 0
    solids = {"Space Needle": nmesh}
    R = Rasters(aoi, c, res=0.5)
    dtm, dsm = R.dtm(), R.dsm()
    ok = B[B.status == "ok"]
    # outputs: attribute table, CityJSON, GLB
    keep = [k for k in B.columns if k not in ("sources",)]
    B[keep].astype({k: str for k in B.columns if B[k].dtype == object and k not in ("geometry",)}).to_file(out / "buildings_lod1.gpkg", driver="GPKG")
    n = exports.cityjson_lod1(B, out / "seattle_center_lod1.city.json", solids=solids, source=f"LOD1 from 3DEP WA_KingCo_1_2021 lidar ({dd}) and Overture building footprints.")
    bad = exports.cityjson_open_edges(json.loads((out / "seattle_center_lod1.city.json").read_text()))
    print("cityjson buildings", n, "open shells", bad)
    cz = float(np.median(dtm))
    centre = (aoi.cx, aoi.cy, cz)
    exports.buildings_glb(B, dtm, R.x0, R.y1, 0.5, centre, out / "seattle_center_lod1.glb", solids=solids)
    # plates
    P = dict(x0=R.x0, y1=R.y1, w=aoi.size_m, h=aoi.size_m, npts=len(c), ppm2=len(c) / aoi.size_m ** 2, cx=aoi.cx, cy=aoi.cy, cz=cz,
             hs=multi_hillshade(dsm, 0.5, 1.0), solids=solids)
    _, top = figures_city.fig_buildings_map(P, B, dd)
    R1 = Rasters(aoi, c, res=1.0)
    dtm1 = R1.dtm()
    hag = c.z - dtm1[R1.row, R1.col]
    rng = np.random.default_rng(3)
    idx = rng.choice(len(c), 2_400_000, replace=False)
    pts = np.c_[c.x[idx] - aoi.cx, c.y[idx] - aoi.cy, c.z[idx] - cz]
    zs = dtm[::4, ::4]
    h, w = zs.shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    v = np.c_[R.x0 + (jj.ravel() * 4 + 0.5) * 0.5 - aoi.cx, R.y1 - (ii.ravel() * 4 + 0.5) * 0.5 - aoi.cy, zs.ravel() - cz]
    ix = np.arange(h * w).reshape(h, w)
    a_, b_, d_, e_ = ix[:-1, :-1].ravel(), ix[:-1, 1:].ravel(), ix[1:, :-1].ravel(), ix[1:, 1:].ravel()
    terrain = trimesh.Trimesh(v, np.r_[np.c_[a_, d_, b_], np.c_[b_, d_, e_]], process=False)
    figures_city.fig_buildings_3d(P, B, pts, hag[idx], terrain, dd)
    n_clipped = int(ok.clipped.sum())
    res = figures_city.fig_buildings_check(B, dd, len(foot))
    # imagery: NAIP corrected with the lidar surface, draped on the models
    year = 2019
    rgbn, tf, ndates = naip.fetch(aoi, year)
    R6 = Rasters(aoi, c, res=tf.a)
    dsm6, gnd6 = R6.dsm(), R6.dtm()
    rows = naip.measure_shifts(B, rgbn, tf, aoi)
    model = naip.fit_displacement(rows[rows[:, 5] >= 0.3])
    import rasterio.features
    raw6 = R6._stat(np.ones(len(c), bool), np.maximum, -np.inf)
    raw6[~np.isfinite(raw6)] = np.nan
    fmask = rasterio.features.rasterize([(g, 1) for g in B[B.status == "ok"].geometry], out_shape=raw6.shape, transform=tf, fill=0, dtype="uint8").astype(bool)
    zroof = naip.repaired_surface_abs(raw6, fmask, dsm6)
    curvy = [i for i, r in B.iterrows() if r.status == "ok" and r.roof_spread_m > 6 and r.area_m2 >= 150 and r["name"] != "Space Needle" and r.roof_z > r.base_z + 0.5]
    hh6 = naip.per_cell_roofs(naip.rigid_roof_heights(naip.building_height_field(raw6, fmask, dsm6, gnd6), B, tf), B, curvy, zroof, gnd6, tf)
    fixed, occ = naip.true_ortho(rgbn, tf, dsm6, gnd6, model, aoi.cx, aoi.cy, hh=hh6)
    evidence = naip.alignment_check(B, rgbn, fixed, tf)
    print("displacement model", model, "\nevidence", evidence, "occluded share", float(occ.mean()))
    exports.ortho_tif(fixed, tf, aoi.crs, out / f"naip_{year}_lidar_corrected.tif", ndates)
    surf, walls = exports.textured_parts(B, dtm, R.x0, R.y1, 0.5, aoi.size_m, fixed, centre, solids=solids, roof_rows=set(curvy), roof_surface=(zroof, tf))
    surf_old, walls_old = exports.textured_parts(B, dtm, R.x0, R.y1, 0.5, aoi.size_m, fixed, centre)
    surf_flat, walls_flat = exports.textured_parts(B, dtm, R.x0, R.y1, 0.5, aoi.size_m, fixed, centre, solids=solids)
    trimesh.Scene({"surface": surf, "walls": walls}).export(out / "seattle_center_textured.glb")
    P["d0"] = dd
    mrow = B[B["name"] == "MoPOP: Museum of Pop Culture"].iloc[0]
    mm = rasterio.features.rasterize([(mrow.geometry, 1)], out_shape=raw6.shape, transform=tf, fill=0, dtype="uint8").astype(bool)
    hole_share = float(np.isnan(raw6[mm]).mean())
    parts_fn = lambda excl: exports.textured_parts(B.drop(index=list(excl)), dtm, R.x0, R.y1, 0.5, aoi.size_m, fixed, centre, solids=solids, roof_rows=set(curvy) - set(excl), roof_surface=(zroof, tf))
    figures_city.fig_naip(P, B, rgbn, fixed, tf, model, evidence, surf, walls, year, ndates, parts_fn=parts_fn)
    figures_city.fig_curved_roof(P, B, rgbn, fixed, tf, (surf_flat, walls_flat), (surf, walls), hole_share, len(curvy))
    nq = needle.fit_quality(npts, nprof)
    sym_s, sym_l = needle.symmetry(npts, 154, 160), needle.symmetry(npts, 30, 60)
    nloc = (naxis[0] - aoi.cx, naxis[1] - aoi.cy)
    figures_city.fig_needle(P, npts, nprof, nq, sym_s, sym_l, (surf_old, walls_old), (surf, walls), nloc, float(nrow.geometry.area * (nrow.roof_z - nrow.base_z)), float(nmesh.volume), dd)
    print("needle", nq, sym_s, sym_l)
    # point-cloud sample + viewer
    sel = rng.choice(len(c), 600_000, replace=False)
    from matplotlib import colormaps
    col = (colormaps["viridis"](np.clip(hag[sel] / 45, 0, 1))[:, :3] * 255).astype(np.uint8)
    col[hag[sel] < 0.5] = (205, 200, 190)
    exports.html_viewer(np.c_[c.x[sel], c.y[sel], c.z[sel]], col, ROOT / "outputs" / "seattle_point_cloud_viewer.html", len(c), dd, dd,
                        "height above ground (viridis, 0 to 45 m); ground and low objects in grey", title="Lidar point cloud, Seattle Center")
    stats = dict(points=len(c), area_km2=aoi.size_m ** 2 / 1e6, pts_per_m2=len(c) / aoi.size_m ** 2, collected=dd, footprints=len(foot), modelled=int(len(ok)), clipped_by_edge=int(ok.clipped.sum()), cityjson_buildings=n,
                 open_shells=bad, needle=dict(quality=nq, symmetry_saucer=sym_s, symmetry_legs=sym_l, volume_m3=float(nmesh.volume)), naip=dict(curved_roofs=len(curvy), year=year, dates=ndates, model=model, alignment=evidence, occluded_share=float(occ.mean())), height_check=res, roof_types=ok.roof_type.value_counts().to_dict(), total_footprint_m2=float(ok.area_m2.sum()), top10=top[["name", "height_m", "height_max_m"]].to_dict("records"))
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k != "top10"}, indent=1, default=str))


if __name__ == "__main__":
    main()
