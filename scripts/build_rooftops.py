"""Rooftop solar potential for two Houston blocks from 3DEP lidar: buildings and roof planes, shading, annual irradiance per roof, with accuracy and sensitivity tests.
Usage: python scripts/build_rooftops.py [westu|downtown|all]     (first run downloads the lidar and one NSRDB column; about 15 minutes per site)

Writes data/processed/rooftops/<site>.json (+ buildings.csv, roof_planes.csv, buildings.geojson), outputs/rooftops/<site>/ (GLB, OBJ, GeoTIFF) and the cache the plotting script reads."""
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
from scipy import ndimage

from lidar3d import block as BL
from lidar3d import city
from lidar3d import rooftop as RT
from lidar3d import rooftop_checks as K
from lidar3d import roofsolar as RS
from lidar3d import solar as SO
from lidar3d.ept import CACHE

OUT = ROOT / "outputs" / "rooftops"
DATA = ROOT / "data" / "processed" / "rooftops"
KWH_RANGE = (600.0, 1500.0)


def cached(name, fn):
    p = CACHE / "rooftop" / name
    if p.exists():
        return pickle.loads(p.read_bytes())
    v = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(v))
    return v


def write_geo(R, T, key):
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.affinity import translate
    import rasterio
    from rasterio.transform import from_origin
    site, C = R["site"], R["C"]
    crs = site["crs"]
    b = {b_.bid: b_ for b_ in R["B"]}
    geoms = [translate(b[i].footprint, C.cx, C.cy) for i in T.building]
    g = gpd.GeoDataFrame(T.round(3), geometry=geoms, crs=crs)
    d_ = OUT / key
    d_.mkdir(parents=True, exist_ok=True)
    g.to_crs(4326).to_file(d_ / "buildings.geojson", driver="GeoJSON")
    # raster of annual total kWh per m2 on roof cells (1 m)
    h = site["half"]
    n = int(2 * h)
    img = np.full((n, n), np.nan, "float32")
    c = R["cells"]
    ix = np.floor(c.x.to_numpy() + h).astype(int); iy = np.floor(h - c.y.to_numpy()).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    img[iy[ok], ix[ok]] = c.total.to_numpy()[ok]
    with rasterio.open(d_ / "roof_irradiance_kwh_m2_yr.tif", "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs=crs, transform=from_origin(C.cx - h, C.cy + h, 1.0, 1.0), nodata=np.nan, compress="deflate") as dst:
        dst.write(img, 1)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(T.x + C.cx, T.y + C.cy)
    return lon, lat


def main(key):
    site = RT.SITES[key]
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    R = RS.analyse(site)
    T = RS.building_table(R)
    C, B, cells = R["C"], R["B"], R["cells"]
    wx = R["weather"]
    df = wx["df"]
    el, az = R["sun"]
    st = dict(site=site["name"], key=key, lonlat=site["lonlat"], crs=site["crs"], lidar_project=site["project"], collected=C.date, analysis_area_ha=(2 * site["half"]) ** 2 / 1e4, context_m=site["context"],
              points=int(len(C.x)), density_per_m2=float(len(C.x) / (2 * C.half) ** 2), class6_share_of_nonground=float((C.cls == 6).sum() / max((C.cls != 2).sum(), 1)),
              weather=dict(source="NREL NSRDB GOES TMY v3.0.6 (nsrdb_tmy-2018)", site_lat=wx["site_lat"], site_lon=wx["site_lon"], annual_ghi_kwh_m2=float(df.ghi.sum() / 1000), annual_dni_kwh_m2=float(df.dni.sum() / 1000), annual_dhi_kwh_m2=float(df.dhi.sum() / 1000)))
    # ---- buildings and roofs
    ps = pd.DataFrame([dict(building=b.bid, **r) for b in B for r in RT.plane_stats(b)])
    st["buildings"] = dict(n=int(len(T)), footprint_median_m2=float(T.footprint_m2.median()), height_median_m=float(T.height_m.median()), height_max_m=float(T.height_m.max()), height_p95_m=float(T.height_m.quantile(0.95)),
                           planes_total=int(len(ps)), planes_per_building_median=float(T.planes.median()), roof_area_ha=float(T.roof_area_m2.sum() / 1e4), footprint_area_ha=float(T.footprint_m2.sum() / 1e4),
                           roof_slope_area_weighted_deg=float(np.average(ps.slope_deg, weights=ps.area_m2)), flat_roof_area_share=float(ps.area_m2[ps.slope_deg < 10].sum() / ps.area_m2.sum()),
                           plane_rms_median_m=float(ps.rms_m.median()))
    asp = ps[ps.slope_deg >= 10]
    bins = np.arange(0, 361, 45)
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ah = np.histogram(((asp.aspect_deg + 22.5) % 360), bins=bins, weights=asp.area_m2)[0]
    st["buildings"]["pitched_roof_area_by_aspect"] = {n: float(v / ah.sum()) for n, v in zip(names, ah)}
    # ---- solar totals and shading
    bm, df_, bo, gwh = RS.annual_totals(R)
    st["solar"] = dict(roof_area_weighted_beam_kwh_m2=bm, diffuse_kwh_m2=df_, beam_open_sky_kwh_m2=bo, beam_shading_loss=1 - bm / bo, total_energy_on_roofs_gwh=gwh, suitable_threshold_kwh_m2=RS.SUITABLE_KWH_M2,
                       suitable_roof_area_share=float(T.suitable_area_m2.sum() / T.roof_area_m2.sum()), pv_estimate_gwh=float(T.pv_kwh_yr_est.sum() / 1e6), pv_assumptions=dict(efficiency=RS.PV_EFFICIENCY, performance_ratio=RS.PV_PERFORMANCE_RATIO),
                       building_total_kwh_yr_quartiles=[float(v) for v in T.total_kwh_yr.quantile([0.25, 0.5, 0.75])], mean_total_kwh_m2_quartiles=[float(v) for v in T.mean_total_kwh_m2.quantile([0.25, 0.5, 0.75])])
    by_aspect = {}
    cp = ps.set_index(["building", "plane"])
    asp_c = np.array([cp.loc[(int(bb), int(pp)), "aspect_deg"] if (int(bb), int(pp)) in cp.index else np.nan for bb, pp in zip(cells.bid, cells.plane)])
    cells["aspect"] = asp_c
    def band(lo, hi):
        a_ = cells.aspect
        return ((a_ >= lo) & (a_ < hi)) if lo >= 0 and hi <= 360 else ((a_ >= 315) | (a_ < 45))
    for nme, lo, hi in (("south", 135, 225), ("east", 45, 135), ("west", 225, 315), ("north", -1, -1)):
        m = band(lo, hi) & (cells.slope >= 10)
        by_aspect[nme] = dict(share_of_pitched_area=float(cells.area[m].sum() / cells.area[cells.slope >= 10].sum()), mean_total_kwh_m2=float((cells.total[m] * cells.area[m]).sum() / cells.area[m].sum()) if m.any() else None)
    m = cells.slope < 10
    by_aspect["flat"] = dict(share_of_roof_area=float(cells.area[m].sum() / cells.area.sum()), mean_total_kwh_m2=float((cells.total[m] * cells.area[m]).sum() / cells.area[m].sum()))
    st["solar"]["by_orientation"] = by_aspect
    # ---- checks
    ck = {}
    # (1) sun and irradiance geometry against pvlib, on a plane set; and horizontal consistency with the TMY global horizontal irradiance
    import pvlib
    chk = []
    for tilt, asp_ in ((0, 180), (20, 180), (30, 90), (40, 270), (90, 180), (25, 0)):
        n_ = np.array([[np.sin(np.radians(asp_)) * np.sin(np.radians(tilt)), np.cos(np.radians(asp_)) * np.sin(np.radians(tilt)), np.cos(np.radians(tilt))]])
        mine = float(SO.unobstructed_beam(n_, el, az, df.dni.values)[0])
        ref = float(pvlib.irradiance.beam_component(tilt, asp_, 90 - el, az, df.dni.values).clip(0).sum() / 1000)
        chk.append(dict(tilt=tilt, aspect=asp_, mine_kwh_m2=mine, pvlib_kwh_m2=ref))
    ck["beam_vs_pvlib"] = chk
    zen = np.radians(90 - el)
    ghi_calc = float((df.dni.values * np.clip(np.cos(zen), 0, None)).sum() / 1000 + df.dhi.sum() / 1000)
    ck["ghi_recomputed_from_dni_dhi_kwh_m2"] = ghi_calc
    ck["ghi_in_file_kwh_m2"] = float(df.ghi.sum() / 1000)
    # (2) shadow geometry against a block of known height
    n_ = 600; hs = 300.0
    dsm = np.zeros((n_, n_)); c_ = np.arange(n_) - hs + 0.5
    X_, Y_ = np.meshgrid(c_, c_)
    dsm[(np.abs(X_) <= 20) & (np.abs(Y_) <= 20)] = 20.0
    sh = []
    for dd in (10, 30, 60, 120):
        Hh = SO.horizon_angles(np.array([[0.0, 20.0 + dd]]), np.array([0.0]), dsm, hs)
        sh.append(dict(distance_m=dd, horizon_deg=float(Hh[0, 36]), expected_deg=float(np.degrees(np.arctan((20 - 0.15) / dd)))))
    ck["shadow_vs_analytic"] = sh
    # (3) planted roofs at this site's density and measured noise
    pts_density = float(sum(b.n_returns for b in R["B"]) / max(sum(b.footprint.area for b in R["B"]), 1.0))
    ck["planted_roofs"] = {}
    for rho_f, sg in ((1.0, st["buildings"]["plane_rms_median_m"]), (0.4, st["buildings"]["plane_rms_median_m"]), (1.0, 0.10)):
        r = cached(f"{key}_planted_{rho_f}_{sg:.3f}.pkl", lambda rho_f=rho_f, sg=sg: K.planted_roofs(60, pts_density * rho_f, sg, el, az, df.dni.values, df.dhi.values, seed=7))
        ck["planted_roofs"][f"density_x{rho_f}_noise_{sg:.3f}"] = dict(density_per_m2=pts_density * rho_f, noise_m=sg, roofs=int(len(r)), slope_err_median_deg=float(np.nanmedian(r[:, 0])), slope_err_p95_deg=float(np.nanpercentile(r[:, 0], 95)),
                                                                     aspect_err_median_deg=float(np.nanmedian(r[:, 1])), aspect_err_p95_deg=float(np.nanpercentile(r[:, 1], 95)), irradiance_err_median_kwh_m2=float(np.nanmedian(np.abs(r[:, 2]))),
                                                                     irradiance_err_p95_kwh_m2=float(np.nanpercentile(np.abs(r[:, 2]), 95)), exactly_right_plane_count_share=float(np.mean(r[np.isfinite(r[:, 0]), 3] == np.where(r[np.isfinite(r[:, 0]), 4] == 0, 1, 2))))
    # (4) split-half repeatability on real roofs
    m6 = R["planar"] & (R["hag"] >= 1.5)
    from lidar3d import forestry as FO2
    iy, ix, nn = FO2._cells(C.x[m6], C.y[m6], C.half, 1.0)
    pb = {}
    glab = ndimage.grey_dilation(R["lab"], size=(3, 3))
    pl = glab[iy, ix]
    X3 = np.c_[C.x[m6], C.y[m6], C.z[m6]]
    for bid in np.unique(pl[pl > 0]):
        pb[int(bid)] = X3[pl == bid]
    sh_ = cached(f"{key}_splithalf.pkl", lambda: K.split_half(B, pb))
    ck["split_half_largest_plane"] = dict(buildings=int(len(sh_)), slope_diff_median_deg=float(np.median(sh_[:, 0])), slope_diff_p95_deg=float(np.percentile(sh_[:, 0], 95)), normal_angle_median_deg=float(np.median(sh_[:, 1])), normal_angle_p95_deg=float(np.percentile(sh_[:, 1], 95)), definition="largest plane of one half vs closest-oriented plane of the other half")
    # (5) footprints against Overture
    aoi = RT.make_aoi(site)
    ov = city.overture_buildings(aoi)
    from shapely.affinity import translate
    ovp = [translate(g_, -C.cx, -C.cy) for g_ in ov.geometry if g_.geom_type == "Polygon"]
    ious, cov = K.footprint_check([b.footprint for b in B], ovp)
    ck["footprints_vs_overture"] = dict(lidar_footprints=int(len(ious)), overture_buildings_30m2_plus=int(len(cov)), lidar_iou_median=float(np.median(ious)), lidar_with_iou_ge_0p5=float((ious >= 0.5).mean()), lidar_with_no_overlap=float((ious == 0).mean()),
                                         overture_covered_ge_half=float((cov >= 0.5).mean()), lidar_area_ha=float(sum(b.footprint.area for b in B) / 1e4), overture_area_ha=float(sum(p.area for p in ovp if p.area >= 30) / 1e4))
    st["checks"] = ck
    # ---- sensitivity of the headline numbers
    sens = {}
    base = dict(total_gwh=gwh, mean_total=float((cells.total * cells.area).sum() / cells.area.sum()), beam=bm, loss=1 - bm / bo)
    sens["baseline"] = base
    for nme, kw in (("no_trees_in_shading_surface", dict(use_trees=False)), ("coarser_2m_shading_surface", dict(dsm_res=2.0))):
        Rv = cached(f"{key}_sens_{nme}.pkl", lambda kw=kw: (lambda r: r["cells"])(RS.analyse(site, weather=wx, reuse=R, **kw)))
        a_ = Rv.area
        sens[nme] = dict(total_gwh=float((Rv.total * a_).sum() / 1e6), mean_total=float((Rv.total * a_).sum() / a_.sum()), beam=float((Rv.beam * a_).sum() / a_.sum()), loss=float(1 - (Rv.beam * a_).sum() / (Rv.beam_open * a_).sum()))
    if key == "westu":
        Rt = cached(f"{key}_sens_thin.pkl", lambda: (lambda r: (r["cells"], len(r["B"])))(RS.analyse(site, weather=wx, thin=0.5)))
        a_ = Rt[0].area
        sens["half_the_returns"] = dict(total_gwh=float((Rt[0].total * a_).sum() / 1e6), mean_total=float((Rt[0].total * a_).sum() / a_.sum()), beam=float((Rt[0].beam * a_).sum() / a_.sum()), loss=float(1 - (Rt[0].beam * a_).sum() / (Rt[0].beam_open * a_).sum()), buildings=int(Rt[1]))
    st["sensitivity"] = sens
    # ---- outputs
    glb, obj, gm, bm = BL.write_mesh(R, OUT / key, KWH_RANGE)
    st["mesh"] = dict(building_faces=int(len(bm.faces)), building_vertices=int(len(bm.vertices)), terrain_faces=int(len(gm.faces)))
    lon, lat = write_geo(R, T, key)
    T = T.assign(lon=np.round(lon, 7), lat=np.round(lat, 7))
    T.round(3).to_csv(DATA / f"{key}_buildings.csv", index=False)
    ps.round(3).to_csv(DATA / f"{key}_roof_planes.csv", index=False)
    (DATA / f"{key}.json").write_text(json.dumps(st, indent=1, default=float))
    keepc = cells[["x", "y", "bid", "plane", "slope", "beam", "diffuse", "beam_open", "total", "area", "horizon_max"]].astype("float32")
    pickle.dump(dict(T=T, ps=ps, cells=keepc, polys={b.bid: np.array(b.footprint.exterior.coords) for b in B}, dsm=R["dsm"].astype("float32"), dtm=R["dtm"].astype("float32"), half=C.half, ahalf=site["half"], sun=(el, az), df=df[["dni", "dhi", "ghi"]],
                     B=B, lab=R["lab"], planar_xy=np.c_[C.x[R["planar"]][::3], C.y[R["planar"]][::3]], ovp=ovp), open(CACHE / "rooftop" / f"{key}_final.pkl", "wb"))
    print(json.dumps(st, indent=1, default=float)[:6000])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k in (["westu", "downtown"] if which == "all" else [which]):
        main(k)
