"""Rebuild every product, figure and export. Usage: python scripts/build_all.py [stage ...]
Stages: figures exports (default: both). Raw data is fetched on first run and cached in data/cache."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from matplotlib import cm
from pyproj import Transformer

from lidar3d import exports, figures, names, radar, validate
from lidar3d.aoi import AOI
from lidar3d.landcover import classify
from lidar3d.products import build

LON, LAT = -121.6615, 46.7065


def point_colours(P, LC, idx):
    c = P.R.c
    cl, hag = c.cls[idx], LC.hag[idx]
    col = np.full((len(idx), 3), (200, 190, 170), np.uint8)
    col[cl == 2] = (170, 150, 125)
    veg = hag > 1.0
    col[veg] = (cm.YlGn(0.35 + 0.65 * np.clip(hag / 25, 0, 1))[:, :3] * 255).astype(np.uint8)[veg]
    col[cl == 9] = (43, 108, 176)
    col[cl == 21] = (226, 232, 240)
    return col


def main(stages):
    stages = stages or ["figures", "exports"]
    t0 = time.time()
    P = build()
    LC = classify(P.R, P.dtm, P.chm)
    ck = validate.run(P, LC, LON, LAT)
    print(f"products + land cover + checks {time.time() - t0:.0f}s")
    out = ROOT / "data" / "processed"
    lc_stats = dict(areas_m2={k: v for k, v in LC.areas(1.0).items()}, s2_scene=ck["s2_scene"], s2_date=str(ck["s2_date"]), ndvi_by_class=ck["ndvi"])
    if "figures" in stages:
        nm = names.features(P.aoi.bbox_ll)
        tr = Transformer.from_crs("EPSG:4326", P.aoi.crs, always_xy=True)
        for n in nm:
            n["x"], n["y"] = tr.transform(n["x"], n["y"])
        figures.fig_overview(P, nm, None)
        figures.fig_terrain(P)
        figures.fig_pointcloud(P, LC)
        figures.fig_canopy(P)
        figures.fig_landcover(P, LC, ck)
        rs = radar.load_set(P.aoi, radar.scenes(LON, LAT))
        figures.fig_radar(P, rs)
        _, rstats = figures.fig_radar_series(P, LC, rs)
        lc_stats["radar_lidar_correlations"] = rstats
        lc_stats["radar_scenes"] = len(rs.dates)
        figures.fig_model3d(P, LC, ck)
    if "exports" in stages:
        exports.landcover_tif(P, LC)
        z, tex = figures._drape(P, LC, ck)
        exports.terrain_mesh(z, P.aoi.bounds[0], P.aoi.bounds[3], 4, tex[0], path_glb=out / "terrain_s2_draped.glb")
        idx, pts = figures.cloud_arrays(P, n=500_000, seed=11)
        col = point_colours(P, LC, idx)
        loc = pts - np.array([P.aoi.cx, P.aoi.cy, 0.0])
        exports.point_ply(loc, col, out / "point_cloud_500k.ply")
        d0, d1 = P.stats["collected"][0], P.stats["collected"][1]
        exports.html_viewer(pts, col, ROOT / "outputs" / "point_cloud_viewer.html", P.stats["points"], d0, d1, "height above ground (green), ground (tan), water (blue), snow (white)")
    (out / "landcover_stats.json").write_text(json.dumps(lc_stats, indent=1, default=float))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
