"""Scroll bars and meander geometry of the Wabash River south of Mount Carmel, Illinois, from one lidar acquisition (USGS 3DEP IL_WabashRiver_2010, flown 2011-09-17), with error and accuracy tests.
Usage: python scripts/build_meander.py   (downloads the lidar and NAIP on the first run, about 15 minutes; later runs use the caches)

The lidar alone gives the ridges, swales, channel and curvature. Imagery from 2011 to 2023 is used only to test the channel position and to measure recent migration; it is not used to build the ridge map."""
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

from lidar3d import meander as M
from lidar3d import naip
from lidar3d.ept import CACHE

OUT = ROOT / "data" / "processed" / "meander"
HALF = M.SITE["half"]
YEARS = (2011, 2012, 2014, 2015, 2017, 2019, 2021, 2023)
PROMINENCE = 0.05          # metres of relief a crest or swale must stand above its surroundings (see the planted-ridge test for what that detects)
BOOT = 12


def cached(name, fn):
    p = CACHE / "meander" / name
    if p.exists():
        return pickle.loads(p.read_bytes())
    v = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(v))
    return v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = M.load_points(log=lambda s: None)
    g = P.cls == 2
    st = dict(site=M.SITE["name"], project=M.SITE["project"], collected=P.date, points=int(len(P.x)), ground_points=int(g.sum()), area_km2=(2 * HALF / 1000) ** 2,
              ground_density_per_m2=float(g.sum() / (2 * HALF) ** 2), flight_lines=[int(v) for v in np.unique(P.psid) if (P.psid == v).sum() > 200_000])
    # ---- DEM and its accuracy
    dem = cached("dem1.pkl", lambda: M.dem_from_points(P.x[g], P.y[g], P.z[g], HALF, 1.0)[0])
    st["dem"] = dict(res_m=1.0, nan_share=float(np.isnan(dem).mean()))
    rng = np.random.default_rng(0)
    h = rng.random(int(g.sum())) < 0.10
    xg, yg, zg = P.x[g], P.y[g], P.z[g]
    d2, _ = M.dem_from_points(xg[~h], yg[~h], zg[~h], HALF, 1.0)
    pr = M.sample_dem(d2, HALF, 1.0, xg[h], yg[h])
    ok = np.isfinite(pr)
    r = (pr - zg[h])[ok]
    gy, gx = np.gradient(ndimage.gaussian_filter(np.nan_to_num(d2), 2.0))
    sl = M.sample_dem(np.degrees(np.arctan(np.hypot(gx, gy))), HALF, 1.0, xg[h], yg[h])[ok]
    st["dem"]["holdout"] = dict(n=int(ok.sum()), rmse_m=float(np.sqrt(np.mean(r ** 2))), mae_m=float(np.mean(np.abs(r))), p95_abs_m=float(np.percentile(np.abs(r), 95)), bias_m=float(np.median(r)),
                                by_slope={f"{lo}-{hi}": dict(n=int(((sl >= lo) & (sl < hi)).sum()), rmse_m=float(np.sqrt(np.mean(r[(sl >= lo) & (sl < hi)] ** 2)))) for lo, hi in ((0, 2), (2, 5), (5, 15), (15, 90))})
    D = M.interswath_differences(P, HALF, res=8.0, min_pts=2, max_slope_deg=2.5)
    st["interswath"] = {f"{a}-{b}": dict(n=int(len(v)), bias_m=float(v.mean()), rms_m=float(np.sqrt(np.mean(v ** 2))), robust_sd_m=float(1.4826 * np.median(np.abs(v - np.median(v))))) for (a, b), v in D.items()}
    # ---- channel, bank line, strip and tracked ridges
    ch, cnt = M.channel_mask(P, HALF, 2.0)
    line, nrm, s = M.bank_line(ch, 2.0, HALF, spacing_m=10.0)
    d, V, C, S, X, Y = M.strip_chains(dem, line, nrm, HALF, prominence=PROMINENCE)
    pk = M.pick_positions(C)
    st["channel"] = dict(area_km2=float(ch.sum() * 4 / 1e6), bank_line_km=float(s[-1] / 1000))
    # ridge statistics by side of the bank line: d > 0 is north of the channel (older floodplain), d < 0 is the point bar south of it
    def spacing(chains, sign):
        by = {}
        for c in chains:
            for si, dd in zip(c["si"], c["d"]):
                if (dd > 60 if sign > 0 else dd < -300):
                    by.setdefault(int(si), []).append(dd)
        sp = [np.diff(np.sort(v)) for v in by.values() if len(v) >= 2]
        sp = np.concatenate(sp) if sp else np.zeros(0)
        return sp[(sp > 8) & (sp < 150)], by
    stats_side = {}
    spacings = {}
    for nm, sign in (("north_relict_set", 1), ("south_point_bar", -1)):
        spc, by = spacing(C, sign)
        spw, _ = spacing(S, sign)
        spacings[nm] = spc
        stats_side[nm] = dict(crest_lines=int(sum(1 for c in C if (np.mean(c["d"]) > 60) == (sign > 0) and abs(np.mean(c["d"])) > (60 if sign > 0 else 300))), crests_per_transect_median=float(np.median([len(v) for v in by.values()])) if by else 0.0,
                              crest_spacing_median_m=float(np.median(spc)), crest_spacing_iqr_m=[float(np.percentile(spc, 25)), float(np.percentile(spc, 75))], crest_spacing_n=int(len(spc)),
                              swale_spacing_median_m=float(np.median(spw)) if len(spw) else None)
    st["ridges"] = dict(prominence_m=PROMINENCE, crest_chains=len(C), swale_chains=len(S), crest_picks=int(len(pk)), median_relief_m=float(np.median(np.concatenate([c["amp"] for c in C]))), **stats_side)
    # ---- detection limit and false picks (planted ridges in the real sampling pattern)
    centre = (1000.0, -1500.0)
    det = {}
    for A in (0.03, 0.05, 0.08, 0.12, 0.2):
        for lam in (30.0, 60.0):
            for dn in (0.0, 0.04):
                det[f"A{A}_L{lam}_corr{dn}"] = cached(f"synth_{A}_{lam}_{dn}.pkl", lambda A=A, lam=lam, dn=dn: M.strip_detection_test(P, HALF, line, nrm, A, lam, 0.03, centre, prominence=PROMINENCE, dem_noise=(dn, 8.0)))
    noise_only = {f"corr{dn}": cached(f"synth_noise_{dn}.pkl", lambda dn=dn: M.strip_detection_test(P, HALF, line, nrm, 0.0, 40.0, 0.03, centre, prominence=PROMINENCE, dem_noise=(dn, 8.0))) for dn in (0.0, 0.04)}
    st["detection_test"] = dict(per_return_noise_m=0.03, results=det, noise_only_picks=noise_only, real_picks=int(len(pk)))
    # ---- repeatability: independent halves, and bootstrap of the ground returns
    rs = np.random.default_rng(3)
    half_sel = rs.random(int(g.sum())) < 0.5
    picks = []
    for sel in (half_sel, ~half_sel):
        dm, _ = M.dem_from_points(xg[sel], yg[sel], zg[sel], HALF, 1.0)
        picks.append(M.pick_positions(M.strip_chains(dm, line, nrm, HALF, prominence=PROMINENCE)[2]))
    st["repeatability"] = dict(split_half=dict(a_in_b=M.pick_agreement(picks[0], picks[1]), b_in_a=M.pick_agreement(picks[1], picks[0]), full_in_half_a=M.pick_agreement(pk, picks[0]), tol_m=4.0, picks=[int(len(p)) for p in picks]))
    boots = cached(f"boot{BOOT}.pkl", lambda: [M.pick_positions(M.strip_chains(M.dem_from_points(xg[i], yg[i], zg[i], HALF, 1.0)[0], line, nrm, HALF, prominence=PROMINENCE)[2]) for i in (np.random.default_rng(100 + b).integers(0, len(xg), len(xg)) for b in range(BOOT))])
    persist = np.array([np.mean([M.pick_agreement(np.array([p]), b, 4.0) for b in boots]) for p in pk[::25]])
    st["repeatability"]["bootstrap"] = dict(runs=BOOT, mean_pick_persistence=float(np.nanmean(persist)), share_picks_in_at_least_80pct_runs=float(np.nanmean(persist >= 0.8)))
    # ---- imagery
    from lidar3d.aoi import AOI
    aoi = M.make_aoi()
    assert isinstance(aoi, AOI)
    imgs = {y: naip.fetch(aoi, y, res=1.0, slug=M.SITE["slug"]) for y in YEARS}
    nd = {y: M.ndvi_image(imgs[y][0]) for y in YEARS}
    dates = {y: imgs[y][2] for y in YEARS}
    chn = np.kron(ch, np.ones((2, 2), bool))[::-1]
    far = ~ndimage.binary_dilation(chn, iterations=450)
    shifts = {}
    for y in YEARS:
        if y == 2011:
            shifts[y] = dict(shift_px=[0.0, 0.0], mad_px=[0.0, 0.0], tiles=0)
            continue
        sh, mad, nt = M.estimate_shift(imgs[2011][0][3].astype("float32"), imgs[y][0][3].astype("float32"), far)
        shifts[y] = dict(shift_px=[float(v) for v in sh], mad_px=[float(v) for v in mad], tiles=int(nt))
        nd[y] = ndimage.shift(nd[y], sh, order=1, mode="nearest")
    st["imagery"] = dict(dates=dates, registration_to_2011=shifts, ndvi_veg_min=0.40, ndvi_bare_max=0.25)
    # NDVI strips on the same transects as the lidar rectification; active-channel edges per year
    strips = {y: M.sample_image(nd[y], HALF, X, Y).astype("float32") for y in YEARS}
    edges = {y: M.channel_edges(strips[y], d) for y in YEARS}
    edge_n = np.array([edges[y][0] for y in YEARS]).T
    edge_s = np.array([edges[y][1] for y in YEARS]).T
    t = np.array([float(dates[y].split(",")[0][:4]) + (int(dates[y].split(",")[0][5:7]) - 1) / 12 for y in YEARS])
    st["migration"] = {}
    for nm, E in (("outer_bank_north_edge", edge_n), ("point_bar_south_edge", edge_s)):
        med, lo, hi, per = M.slope_ci(t, E)
        st["migration"][nm] = dict(rate_m_per_yr=med, ci95=[lo, hi], stations_with_rate=int(np.isfinite(per).sum()), share_stations_rate_abs_below_0p5=float(np.nanmean(np.abs(per) < 0.5)),
                                   median_edge_distance_m={str(y): float(np.nanmedian(E[:, i])) for i, y in enumerate(YEARS)}, edge_found_share={str(y): float(np.isfinite(E[:, i]).mean()) for i, y in enumerate(YEARS)})
    # how much the rate depends on single years (stage and season differ between flights)
    for nm, E in (("outer_bank_north_edge", edge_n), ("point_bar_south_edge", edge_s)):
        loo = {}
        for drop in YEARS:
            k = [i for i, y in enumerate(YEARS) if y != drop]
            m_, lo_, hi_, _ = M.slope_ci(t[k], E[:, k])
            loo[str(drop)] = dict(rate=m_, ci95=[lo_, hi_])
        k = [i for i, y in enumerate(YEARS) if y not in (2017, 2019)]
        m_, lo_, hi_, _ = M.slope_ci(t[k], E[:, k])
        loo["without_2017_and_2019"] = dict(rate=m_, ci95=[lo_, hi_])
        st["migration"][nm]["leave_one_year_out"] = loo
        st["migration"][nm]["between_year_sd_of_median_edge_m"] = float(np.nanstd([np.nanmedian(E[:, i]) for i in range(len(YEARS))]))
    # the lidar water edge is d = 0 by construction; the imagery edge nearest in time (2011-08-25, 23 days before the flight) is not the same thing (a water line against the edge of bare ground) but a fixed offset is expected
    off = edge_n[:, 0][np.isfinite(edge_n[:, 0])]
    st["channel_vs_imagery_2011"] = dict(n=int(len(off)), median_offset_m=float(np.median(off)), iqr_m=[float(np.percentile(off, 25)), float(np.percentile(off, 75))], p05_p95_m=[float(np.percentile(off, 5)), float(np.percentile(off, 95))])
    (OUT / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    pickle.dump(dict(dem_small=M.coarsen(dem, 2).astype("float32"), ch=ch, line=line, nrm=nrm, s=s, d=d, V=V.astype("float32"), C=C, S=S, X=X.astype("float32"), Y=Y.astype("float32"), edge_n=edge_n, edge_s=edge_s, t=t, years=YEARS,
                     strips={y: strips[y] for y in (2011, 2023)}, picks=pk, spacing=spacings), open(CACHE / "meander" / "final.pkl", "wb"))
    from lidar3d import figures_meander as FG
    F = pickle.loads((CACHE / "meander" / "final.pkl").read_bytes())
    FG.fig_site(st, F); FG.fig_ridges(st, F); FG.fig_tests(st); FG.fig_imagery(st, F)
    print(json.dumps(st, indent=1, default=float))


if __name__ == "__main__":
    main()
