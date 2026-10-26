"""Seattle Center buildings: lidar heights, LOD1 models, plates and exports. Usage: python scripts/build_seattle.py"""
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import trimesh

from lidar3d import buildings, city, exports, figures_city
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
    R = Rasters(aoi, c, res=0.5)
    dtm, dsm = R.dtm(), R.dsm()
    ok = B[B.status == "ok"]
    # outputs: attribute table, CityJSON, GLB
    keep = [k for k in B.columns if k not in ("sources",)]
    B[keep].astype({k: str for k in B.columns if B[k].dtype == object and k not in ("geometry",)}).to_file(out / "buildings_lod1.gpkg", driver="GPKG")
    n = exports.cityjson_lod1(B, out / "seattle_center_lod1.city.json", source=f"LOD1 from 3DEP WA_KingCo_1_2021 lidar ({dd}) and Overture building footprints.")
    bad = exports.cityjson_open_edges(json.loads((out / "seattle_center_lod1.city.json").read_text()))
    print("cityjson buildings", n, "open shells", bad)
    cz = float(np.median(dtm))
    centre = (aoi.cx, aoi.cy, cz)
    exports.buildings_glb(B, dtm, R.x0, R.y1, 0.5, centre, out / "seattle_center_lod1.glb")
    # plates
    P = dict(x0=R.x0, y1=R.y1, w=aoi.size_m, h=aoi.size_m, npts=len(c), ppm2=len(c) / aoi.size_m ** 2, cx=aoi.cx, cy=aoi.cy, cz=cz,
             hs=multi_hillshade(dsm, 0.5, 1.0))
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
    # point-cloud sample + viewer
    sel = rng.choice(len(c), 600_000, replace=False)
    from matplotlib import colormaps
    col = (colormaps["viridis"](np.clip(hag[sel] / 45, 0, 1))[:, :3] * 255).astype(np.uint8)
    col[hag[sel] < 0.5] = (205, 200, 190)
    exports.html_viewer(np.c_[c.x[sel], c.y[sel], c.z[sel]], col, ROOT / "outputs" / "seattle_point_cloud_viewer.html", len(c), dd, dd,
                        "height above ground (viridis, 0 to 45 m); ground and low objects in grey", title="Lidar point cloud, Seattle Center")
    stats = dict(points=len(c), area_km2=aoi.size_m ** 2 / 1e6, pts_per_m2=len(c) / aoi.size_m ** 2, collected=dd, footprints=len(foot), modelled=int(len(ok)), clipped_by_edge=int(ok.clipped.sum()), cityjson_buildings=n,
                 open_shells=bad, height_check=res, roof_types=ok.roof_type.value_counts().to_dict(), total_footprint_m2=float(ok.area_m2.sum()), top10=top[["name", "height_m", "height_max_m"]].to_dict("records"))
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k != "top10"}, indent=1, default=str))


if __name__ == "__main__":
    main()
