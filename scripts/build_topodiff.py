"""Topographic differencing of three lidar surveys (2010, 2016, 2023) of the bluffs at Pacifica, California: registration on stable ground (ICP), DEMs of difference with levels of detection, M3C2 on the bluff face,
cut-and-fill volumes with uncertainty, contour retreat, and accuracy tests (held-out registration, planted changes, false-positive rate, closure between intervals).
Usage: python scripts/build_topodiff.py    (downloads three lidar projects the first time, about 10 minutes)"""
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

from lidar3d import change as CH
from lidar3d import topodiff as TD
from lidar3d.ept import CACHE

SITE = dict(slug="pacifica", name="Pacifica coastal bluffs and Mori Point, California", lonlat=(-122.4985, 37.6175), half=450.0, crs="EPSG:32610")
EPOCHS = (("2010", "ARRA-CA_SanFranCoast_2010"), ("2016", "USGS_LPC_CA_WestCoastElNinoUTM10_2016_LAS_2017"), ("2023", "CA_CaliforniaGaps_3_B23"))
HALF = SITE["half"]
RES = 2.0
OUT = ROOT / "outputs" / "topodiff"
DATA = ROOT / "data" / "processed" / "topodiff"


def cached(name, fn):
    p = CACHE / "topodiff" / name
    if p.exists():
        return pickle.loads(p.read_bytes())
    v = fn()
    p.write_bytes(pickle.dumps(v))
    return v


def write_tif(path, arr, aoi, res, nodata=np.nan):
    a = np.flipud(arr).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="float32", crs=aoi.crs, transform=from_origin(aoi.cx - HALF, aoi.cy + HALF, res, res), nodata=nodata, compress="deflate") as d:
        d.write(a, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    aoi = TD.make_aoi(SITE)
    E = {t: TD.load_epoch(SITE, t, p, log=lambda s: None) for t, p in EPOCHS}
    st = dict(site=SITE["name"], lonlat=SITE["lonlat"], crs=SITE["crs"], half_m=HALF, res_m=RES, epochs={t: dict(project=e.project, date=e.date, points=int(len(e.x)), ground_points=int((e.cls == 2).sum()), ground_density_per_m2=float((e.cls == 2).sum() / (2 * HALF) ** 2)) for t, e in E.items()})
    ref = "2023"
    Dref, _ = TD.ground_dem(E[ref], HALF, RES, 1)
    stable = TD.stable_zone(Dref, HALF, RES, 150.0, 8.0)
    dist_sea, sea = TD.sea_distance(Dref, RES)
    st["stable_zone"] = dict(cells=int(stable.sum()), area_ha=float(stable.sum() * RES ** 2 / 1e4), rule="at least 150 m from the sea, slope 8 degrees or less in the 2023 ground DEM")
    n = Dref.shape[0]
    yy = (np.arange(n) + 0.5) * RES - HALF
    south = np.tile((yy < 0)[:, None], (1, n))
    # ---- registration of 2010 and 2016 to 2023 on stable ground, with a held-out half
    reg, regs = {}, {}
    cls2 = {t: E[t].cls == 2 for t in E}
    for t in ("2010", "2016"):
        iy, ix, _ = TD._cell_index(E[t].x, E[t].y, HALF, RES)
        src_all = E[t].xyz[cls2[t] & stable[iy, ix]].astype("float64")
        iy2, ix2, _ = TD._cell_index(E[ref].x, E[ref].y, HALF, RES)
        dst_all = E[ref].xyz[cls2[ref] & stable[iy2, ix2]].astype("float64")
        rng = np.random.default_rng(0)

        def fit(sel_s, sel_d, k=1):
            r = np.random.default_rng(k)
            s_ = src_all[sel_s]; d_ = dst_all[sel_d]
            s_ = s_[r.choice(len(s_), min(len(s_), 120000), replace=False)]; d_ = d_[r.choice(len(d_), min(len(d_), 350000), replace=False)]
            return TD.icp_point_to_plane(s_, d_)
        allm_s, allm_d = np.ones(len(src_all), bool), np.ones(len(dst_all), bool)
        R, tt, hist = cached(f"icp_{t}.pkl", lambda: fit(allm_s, allm_d))
        # held-out: fit on the southern stable half, evaluate on the northern half
        Rs, ts, _ = cached(f"icp_{t}_south.pkl", lambda: fit(src_all[:, 1] < 0, dst_all[:, 1] < 0))
        # bootstrap of the translation (horizontal and vertical) from random halves of the stable points
        boots = cached(f"icp_{t}_boot.pkl", lambda: [fit(np.random.default_rng(100 + b).random(len(src_all)) < 0.5, np.random.default_rng(200 + b).random(len(dst_all)) < 0.5, k=b)[1] for b in range(6)])
        B = np.array(boots)
        e_before = E[t]
        e_full = TD.apply(R, tt, e_before)
        e_south = TD.apply(Rs, ts, e_before)
        Dt_raw, _ = TD.ground_dem(e_before, HALF, RES, 1)
        Dt_full, _ = TD.ground_dem(e_full, HALF, RES, 1)
        Dt_south, _ = TD.ground_dem(e_south, HALF, RES, 1)
        north_test = stable & ~south
        def stats(Dx):
            d = Dref - Dx
            m = north_test & np.isfinite(d)
            return dict(median_m=float(np.median(d[m])), robust_sd_m=TD.robust_sd(d[m]), rms_trim1m=float(np.sqrt(np.mean(d[m & (np.abs(d) < 1)] ** 2))), n=int(m.sum()))
        regs[t] = dict(translation_m=[float(v) for v in tt], rotation_deg=[float(v) for v in TD.rotation_angles_deg(R)], icp_rms_start=float(hist[0]), icp_rms_end=float(hist[-1]), iterations=len(hist),
                       bootstrap_translation_sd_m=[float(v) for v in B.std(0)], south_fit_translation_m=[float(v) for v in ts],
                       held_out_north_half=dict(before=stats(Dt_raw), after_fit_on_south_half=stats(Dt_south), after_fit_on_all=stats(Dt_full)))
        reg[t] = e_full
    reg[ref] = E[ref]
    st["registration"] = regs
    # ---- registered DEMs and differences
    D = {t: TD.ground_dem(reg[t], HALF, RES, 1)[0] for t in reg}
    Draw = {t: TD.ground_dem(E[t], HALF, RES, 1)[0] for t in E}
    slope = TD.slope_deg(D[ref], RES)
    coast = np.isfinite(D[ref]) & (dist_sea <= 60.0)
    inland = np.isfinite(D[ref]) & (dist_sea > 60.0)
    intervals = (("2010", "2016"), ("2016", "2023"), ("2010", "2023"))
    sxy = {t: max(float(np.hypot(*regs[t]["bootstrap_translation_sd_m"][:2])), 0.05) for t in regs}
    sxy[ref] = 0.05
    res_all, dods, lods = {}, {}, {}
    for a, b in intervals:
        dz = D[b] - D[a]
        m = stable & np.isfinite(dz)
        sz_rob = TD.robust_sd(dz[m])
        sz = TD.error_sd(dz[m])                       # at least as large as the spread that gives a 5% false-positive rate on flat stable ground
        sxy_ab = float(np.hypot(sxy[a], sxy[b]))
        lod = TD.level_of_detection(slope, sz, sxy_ab)
        dods[(a, b)], lods[(a, b)] = dz, lod
        fp = float((np.abs(dz[m]) > lod[m]).mean())
        r = dict(sigma_z_stable_m=sz, sigma_z_robust_mad_m=sz_rob, sigma_xy_m=sxy_ab, median_stable_m=float(np.median(dz[m])), stable_cells=int(m.sum()), false_positive_share_on_stable=fp, lod_flat_m=float(1.96 * sz))
        for zone, mask in (("coastal_band_60m", coast), ("inland", inland), ("all_land", np.isfinite(D[ref]))):
            r[zone] = {}
            for lbl, zc in (("lod95", 1.96), ("lod68", 1.0), ("lod99", 2.58), ("none", 0.0)):
                lod_z = TD.level_of_detection(slope, sz, sxy_ab, zc) if zc else np.zeros_like(slope)
                r[zone][lbl] = TD.dod_volumes(dz, lod_z, RES, mask)
        a_c = r["coastal_band_60m"]["lod95"]["significant_area_m2"]
        sd_v, nb = TD.block_bootstrap_volume(dz, stable, RES, max(a_c, 400.0))
        r["coastal_band_60m"]["lod95"]["net_volume_sd_block_bootstrap_m3"] = sd_v
        r["coastal_band_60m"]["lod95"]["bootstrap_blocks"] = nb
        res_all[f"{a}-{b}"] = r
    st["dod"] = res_all
    # closure: V(2010-2023) against V(2010-2016) + V(2016-2023), each at its own level of detection
    cz = {}
    for zone in ("coastal_band_60m", "all_land"):
        v1, v2, v3 = (res_all[k][zone]["lod95"]["net_m3"] for k in ("2010-2016", "2016-2023", "2010-2023"))
        cz[zone] = dict(sum_of_parts_m3=v1 + v2, whole_m3=v3, difference_m3=v3 - (v1 + v2))
    st["closure"] = cz
    # ---- by alongshore segment (coastal band, 2010 to 2023 and each interval)
    rows_land = np.flatnonzero(coast.any(1))
    edges = np.linspace(rows_land.min(), rows_land.max() + 1, 9).astype(int)
    seg = []
    for i in range(8):
        zm = np.zeros_like(coast)
        zm[edges[i]:edges[i + 1]] = coast[edges[i]:edges[i + 1]]
        rec = dict(segment=i + 1, y_from_m=float(edges[i] * RES - HALF), y_to_m=float(edges[i + 1] * RES - HALF), coastal_cells=int(zm.sum()))
        for a, b in intervals:
            v = TD.dod_volumes(dods[(a, b)], lods[(a, b)], RES, zm)
            rec[f"{a}-{b}_net_m3"], rec[f"{a}-{b}_erosion_m3"], rec[f"{a}-{b}_deposition_m3"] = v["net_m3"], v["erosion_m3"], v["deposition_m3"]
        seg.append(rec)
    pd.DataFrame(seg).round(1).to_csv(DATA / "segments.csv", index=False)
    st["segments"] = seg
    # ---- contour retreat on east-west transects
    rows, shore_x = TD.transects_across_coast(D[ref], HALF, RES, 10.0)
    levels = (6.0, 12.0, 20.0)
    CR = TD.contour_retreat({t: D[t] for t in D}, rows, shore_x, RES, levels)
    ret = {}
    for lv in levels:
        for a, b in intervals:
            d = CR[lv][a] - CR[lv][b]           # positive = the contour moved west... inland-position grows when the face retreats
            d = CR[lv][b] - CR[lv][a]
            ok = np.isfinite(d)
            yrs = {"2010-2016": 6.0, "2016-2023": 7.6, "2010-2023": 13.6}[f"{a}-{b}"]
            ret.setdefault(str(lv), {})[f"{a}-{b}"] = dict(transects=int(ok.sum()), median_m=float(np.median(d[ok])) if ok.any() else None, p10_m=float(np.percentile(d[ok], 10)) if ok.any() else None, p90_m=float(np.percentile(d[ok], 90)) if ok.any() else None,
                                                          median_rate_m_per_yr=float(np.median(d[ok]) / yrs) if ok.any() else None, retreat_gt_lod_share=float((d[ok] > 0.5).mean()) if ok.any() else None)
    st["contour_retreat"] = ret
    # ---- M3C2 on the bluff face
    face = coast & (slope >= 20.0)
    ii, jj = np.nonzero(face)
    cores = np.c_[(jj + 0.5) * RES - HALF, (ii + 0.5) * RES - HALF, D[ref][ii, jj]]
    m3 = {}
    for a, b in (("2010", "2016"), ("2016", "2023")):
        def run(a=a, b=b):
            def pts(e):
                m = (e.cls == 2)
                P = e.xyz[m]
                keep = (np.isfinite(P).all(1)) & (CH.significant is not None)
                return P[keep].astype("float64")
            P1, P2 = pts(reg[a]), pts(reg[b])
            # restrict to the coastal strip to keep the search small
            def strip(P):
                iy, ix, _ = TD._cell_index(P[:, 0], P[:, 1], HALF, RES)
                return P[(dist_sea[iy, ix] <= 90.0)]
            P1, P2 = strip(P1), strip(P2)
            d, lod_, s1, s2, n1, n2, nrm = CH.m3c2(P1, P2, cores, registration_error=float(np.hypot(sxy[a], sxy[b])))
            return d, lod_, nrm
        m3[f"{a}-{b}"] = cached(f"m3c2_{a}_{b}.pkl", run)
    m3s = {}
    for k, (d, lod_, nrm) in m3.items():
        a, b = k.split("-")
        sig = CH.significant(d, lod_)
        loss, gain = CH.volume_from_normals(d, nrm, sig, RES ** 2)
        dzc = dods[(a, b)][ii, jj]
        ok = np.isfinite(d) & np.isfinite(dzc)
        # the vertical equivalent of the normal-direction change at each core, for comparison with the DoD
        dz_eq = d * np.abs(nrm[:, 2])
        dod_v = TD.dod_volumes(dods[(a, b)], lods[(a, b)], RES, face)
        m3s[k] = dict(cores=int(len(d)), valid=int(np.isfinite(d).sum()), significant=int(sig.sum()), erosion_m3=loss, deposition_m3=gain, net_m3=gain - loss, dod_net_same_cells_m3=dod_v["net_m3"], dod_erosion_same_cells_m3=dod_v["erosion_m3"],
                      median_normal_change_m=float(np.nanmedian(d[sig])) if sig.any() else None, median_lod_m=float(np.nanmedian(lod_)),
                      corr_with_dod_vertical=float(np.corrcoef(dz_eq[ok], dzc[ok])[0, 1]) if ok.sum() > 10 else None, face_cells=int(face.sum()))
    st["m3c2_face"] = m3s
    # ---- tests with planted change on stable ground, and sensitivity
    dz = dods[("2016", "2023")]
    sz = res_all["2016-2023"]["sigma_z_stable_m"]
    lod_by_slope = {}
    for lo_, hi_ in ((0, 5), (5, 15), (15, 30), (30, 45), (45, 90)):
        mm = np.isfinite(D[ref]) & (slope >= lo_) & (slope < hi_) & coast
        lod_by_slope[f"{lo_}-{hi_}"] = dict(cells=int(mm.sum()), median_lod95_2010_2023_m=float(np.median(lods[("2010", "2023")][mm])) if mm.any() else None, median_lod95_2016_2023_m=float(np.median(lods[("2016", "2023")][mm])) if mm.any() else None)
    st["lod_by_slope_coastal_band"] = lod_by_slope
    lod_flat = lods[("2016", "2023")]
    cand = stable & np.isfinite(dz) & (slope <= 4.0)
    cc = cand
    pi, pj = np.nonzero(cc)
    rng = np.random.default_rng(5)
    planted = {}
    for side_m in (6.0, 12.0):
        k = int(side_m / RES)
        for amp in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            det, vrec, vtrue, n_ = [], [], [], 0
            for _ in range(60):
                q = rng.integers(0, len(pi))
                i0, j0 = pi[q] - k // 2, pj[q] - k // 2
                if i0 < 0 or j0 < 0:
                    continue
                patch = (slice(i0, i0 + k), slice(j0, j0 + k))
                if not cc[patch].all():
                    continue
                sign = rng.choice([-1, 1])
                d2 = dz[patch] + sign * amp
                sg = np.abs(d2) > lod_flat[patch]
                det.append(sg.mean())
                vrec.append(float((d2[sg]).sum() * RES ** 2 * sign))
                vtrue.append(amp * k * k * RES ** 2)
                n_ += 1
            planted[f"{side_m:g}m_{amp}m"] = dict(patches=n_, cells_detected_share=float(np.mean(det)), volume_recovered_share=float(np.sum(vrec) / np.sum(vtrue)), patch_detected_share=float(np.mean(np.array(det) > 0.5)))
    st["planted_change_on_stable_ground"] = dict(note="a step of known size is added to the measured 2016-2023 difference over patches of gentle stable ground; recovery measured with the same level of detection", results=planted)
    sens = {}
    for res_ in (1.0, 2.0, 4.0):
        Dx = {t: TD.ground_dem(reg[t], HALF, res_, 1)[0] for t in ("2010", "2023")}
        sl = TD.slope_deg(Dx["2023"], res_)
        dist_x, _ = TD.sea_distance(Dx["2023"], res_)
        stb = TD.stable_zone(Dx["2023"], HALF, res_, 150.0, 8.0)
        d = Dx["2023"] - Dx["2010"]
        m = stb & np.isfinite(d)
        szx = TD.error_sd(d[m])
        lod = TD.level_of_detection(sl, szx, float(np.hypot(sxy["2010"], sxy["2023"])))
        v = TD.dod_volumes(d, lod, res_, np.isfinite(Dx["2023"]) & (dist_x <= 60))
        both = np.isfinite(Dx["2023"]) & np.isfinite(Dx["2010"]) & (dist_x <= 60)
        sens[f"cell_{res_:g}m"] = dict(sigma_z_m=szx, coastal_cells_with_both_dems_share=float(both.sum() / max((np.isfinite(Dx["2023"]) & (dist_x <= 60)).sum(), 1)), **{k: v[k] for k in ("erosion_m3", "deposition_m3", "net_m3", "significant_area_m2")})
    dz0 = Draw["2023"] - Draw["2010"]
    sens["no_registration_2010_2023"] = dict(**{k: TD.dod_volumes(dz0, TD.level_of_detection(slope, TD.robust_sd(dz0[stable & np.isfinite(dz0)]), 0.05), RES, coast)[k] for k in ("erosion_m3", "deposition_m3", "net_m3")})
    st["sensitivity_2010_2023_coastal_band"] = sens
    # ---- write rasters, tables, cache
    for (a, b), dz_ in dods.items():
        write_tif(OUT / f"dod_{a}_{b}_m.tif", dz_, aoi, RES)
        write_tif(OUT / f"dod_{a}_{b}_significant_m.tif", np.where(np.abs(dz_) > lods[(a, b)], dz_, np.nan), aoi, RES)
        write_tif(OUT / f"lod95_{a}_{b}_m.tif", lods[(a, b)], aoi, RES)
    for t in D:
        write_tif(OUT / f"dem_{t}_m.tif", D[t], aoi, RES)
    for k, (d, lod_, nrm) in m3.items():
        pd.DataFrame(dict(x=cores[:, 0] + aoi.cx, y=cores[:, 1] + aoi.cy, z=cores[:, 2], normal_change_m=np.round(d, 3), lod95_m=np.round(lod_, 3), nx=np.round(nrm[:, 0], 3), ny=np.round(nrm[:, 1], 3), nz=np.round(nrm[:, 2], 3))).to_csv(DATA / f"m3c2_{k}.csv", index=False)
    (DATA / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    pickle.dump(dict(D=D, dods=dods, lods=lods, stable=stable, coast=coast, dist_sea=dist_sea, slope=slope, cores=cores, m3={k: (v[0], v[1], v[2]) for k, v in m3.items()}, rows=rows, shore_x=shore_x, CR=CR, levels=levels, yy=yy, edges=edges,
                     Draw=Draw, face=face), open(CACHE / "topodiff" / "final.pkl", "wb"))
    print(json.dumps(st, indent=1, default=float)[:7000])


if __name__ == "__main__":
    main()
