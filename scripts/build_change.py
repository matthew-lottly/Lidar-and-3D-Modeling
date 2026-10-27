"""Coastal bluff erosion at Cape Cod from two 3DEP lidar epochs (2014 and 2021) with M3C2. Usage: python scripts/build_change.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from lidar3d import change, figures_change

STABLE_X = -120.0          # local east below which the ground is inland, flat and not eroding (chosen by looking at the DEM difference)
SPACING = 2.0
ROUGH_MAX = 0.10           # m: epoch spread above this = vegetation or debris, not interpreted


def main():
    out = ROOT / "data" / "processed" / "capecod"
    out.mkdir(parents=True, exist_ok=True)
    aoi = change.make_aoi()
    E = [change.load_epoch(t, p, aoi, log=lambda s: None) for t, p in change.SITE["epochs"]]
    masks = [change.surface_points(e) for e in E]
    P = [e.xyz[m] for e, m in zip(E, masks)]
    core = change.regular_cores(P[0], SPACING)
    d, lod0, s1, s2, n1, n2, nrm = change.m3c2(P[0], P[1], core, registration_error=0.0)
    ok = np.isfinite(d) & np.isfinite(s1) & np.isfinite(s2)
    smooth = ok & (s1 < ROUGH_MAX) & (s2 < ROUGH_MAX)
    st = (core[:, 0] < STABLE_X) & smooth
    reg_sd = change.robust_sd(d[st])
    reg_err = float(round(reg_sd, 3))
    d, lod, *_ = change.m3c2(P[0], P[1], core, registration_error=reg_err)
    sig = change.significant(d, lod) & smooth
    area = SPACING ** 2
    loss, gain = change.volume_from_normals(d, nrm, sig, area)
    ang = np.radians(change.SHORE_ANGLE_DEG)
    along = core[:, 0] * np.sin(ang) + core[:, 1] * np.cos(ang)
    band = (core[:, 2] > 6) & (core[:, 2] < 26) & (core[:, 0] > -30) & (core[:, 0] < 130) & smooth
    zones = {"north bluff": band & (along >= 120), "middle bluff": band & (along >= -60) & (along < 120), "south bank": band & (along < -100)}
    zones["beach"] = (core[:, 2] <= 3) & (core[:, 0] > 100) & smooth
    zst = change.zone_stats(core, d, sig, nrm, ok, zones, SPACING, reg_err)
    chk = change.dod_check(P[0], P[1], core, d, sig, nrm, smooth, {k: v for k, v in zones.items() if k != "beach"}, SPACING)
    stats = dict(
        epochs=[dict(tag=e.tag, project=e.project, date=e.date, points=len(e), surface_points=int(m.sum()), pts_per_m2=float(len(e) / 600 ** 2), surface_pts_per_m2=float(m.sum() / 600 ** 2)) for e, m in zip(E, masks)],
        years_between=(int(E[1].date[:4]) - int(E[0].date[:4])) + (int(E[1].date[5:7]) - int(E[0].date[5:7])) / 12,
        core=dict(spacing_m=SPACING, n=int(len(core)), valid=int(ok.sum()), smooth=int(smooth.sum()), rough_excluded_share=float(1 - smooth.sum() / ok.sum())),
        stable=dict(x_max=STABLE_X, n=int(st.sum()), median_m=float(np.median(d[st])), robust_sd_m=reg_sd, std_m=float(d[st].std()), share_over_lod=float(change.significant(d, lod)[st].mean()),
                    share_over_015=float((np.abs(d[st]) > 0.15).mean()), lod_median_m=float(np.nanmedian(lod[st]))),
        registration_error_m=reg_err,
        change=dict(significant=int(sig.sum()), loss_m3=loss, gain_m3=gain, net_m3=gain - loss,
                    max_loss_m=float(-d[sig].min()), max_gain_m=float(d[sig].max())),
    )
    stats["zones"] = zst
    stats["dod_check"] = chk
    stats["along_length_m"] = {"north bluff": 180, "middle bluff": 180, "south bank": 200}
    np.savez(out / "m3c2_arrays.npz", core=core, d=d, lod=lod, s1=s1, s2=s2, n1=n1, n2=n2, nrm=nrm, sig=sig, smooth=smooth, ok=ok)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    # plates and exports
    figures_change.fig_change(stats, np.load(out / "m3c2_arrays.npz"), P[0], P[1], change.SHORE_ANGLE_DEG)
    figures_change.fig_validation(stats, np.load(out / "m3c2_arrays.npz"))
    change.export_change(out, core, d, sig, smooth, aoi)
    print(json.dumps({k: stats[k] for k in ("registration_error_m", "stable", "zones", "dod_check")}, indent=1))


if __name__ == "__main__":
    main()
