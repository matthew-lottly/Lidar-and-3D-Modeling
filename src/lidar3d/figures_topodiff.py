"""Plates 46 to 50: Pacifica bluff topographic differencing (site, DEMs of difference, hotspots with profiles, volumes and retreat, accuracy tests)."""
from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from . import sitemap
from .cloud import hillshade
from .plates import MUTED, Plate

INTERVALS = ("2010-2016", "2016-2023", "2010-2023")
YEARS = {"2010-2016": "about 6 years", "2016-2023": "7.6 years", "2010-2023": "about 13.6 years"}
RED, BLUE, GREY = "#B03A2E", "#1F6FB2", "#7F8C8D"


def _src(st):
    e = st["epochs"]
    return (f"Lidar: 3DEP {e['2010']['project']} (2010, date not in file), {e['2016']['project']} (flown {e['2016']['date']}), {e['2023']['project']} (flown {e['2023']['date']}). Orientation layers: Overture Maps "
            "(OpenStreetMap-derived), imagery NAIP; not used in the differencing.")


def _halo(w=2.0, c="white"):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=w, foreground=c)]


def _scalebar(ax, x0, y0, length, color="k"):
    import matplotlib.patheffects as pe
    halo = "black" if color == "white" else "white"
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.5, solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=4.5, foreground=halo)], zorder=20)
    ax.text(x0 + length / 2, y0 + 8, f"{length:g} m", ha="center", va="bottom", fontsize=7.5, color=color, path_effects=_halo(2.0, halo), zorder=21)


def _vcbar(pl, x, y, h, cmap, norm, label, ticks=None):
    import matplotlib.pyplot as plt
    cb = pl.ax(x, y, 0.1, h)
    v = np.linspace(norm.vmax, norm.vmin, 256)
    cb.imshow(plt.get_cmap(cmap)(norm(v))[:, None, :], aspect="auto", extent=(0, 1, norm.vmin, norm.vmax))
    cb.set_xticks([]); cb.yaxis.tick_right(); cb.yaxis.set_label_position("right"); cb.tick_params(labelsize=7, length=2)
    if ticks is not None:
        cb.set_yticks(ticks)
    cb.set_ylabel(label, fontsize=7.5, color=MUTED, labelpad=4)


def _rows(pl, x, y0, rows, width, fs=8.2):
    for k_, v_ in rows:
        pl.text(x, y0, k_, fontsize=9, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v_, width))
        pl.text(x, y0 - 0.24, w, fontsize=fs, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.27 + 0.16 * (w.count("\n") + 1) + 0.14
    return y0


def fig_site(st, naip_img):
    bbox = (-122.72, 37.45, -122.25, 37.85)
    L = sitemap.regional_layers("pacifica", bbox, "California", pop_min=15000)
    lon, lat = st["lonlat"]
    pl = Plate("Where the Pacifica study area is", "A 900 m square of coastline in San Mateo County, California: Esplanade bluffs to the north, Mori Point headland to the south",
               _src(st), "Red: freeways. Orange: trunk roads. Blue: water. Dashed grey: county lines")
    a = pl.ax(0.6, 1.1, 7.3, 7.2)
    sitemap.draw_regional(a, L, bbox, (lon, lat), (0.0053, 0.0045), label_places=("San Francisco", "Daly City", "Pacifica", "San Bruno", "San Mateo", "Half Moon Bay", "South San Francisco", "Burlingame"))
    a.annotate("study area", xy=(lon, lat), xytext=(lon - 0.14, lat + 0.12), fontsize=9, fontweight="bold", color="#E74C3C", ha="center", arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2), path_effects=_halo())
    pl.label(0.6, 8.35, "Region")
    b = pl.ax(8.4, 3.6, 4.4, 4.4)
    h = st["half_m"]
    b.imshow(np.moveaxis(naip_img[:3], 0, -1), extent=(-h, h, -h, h)); b.set_xticks([]); b.set_yticks([])
    _scalebar(b, -h + 40, -h + 30, 200.0, "white")
    pl.label(8.4, 8.2, "The study area in aerial imagery")
    e = st["epochs"]
    rows = [("Why here", "The bluffs at Pacifica are among the fastest-retreating on the California coast, and three lidar surveys cover them: 2010, April 2016 (just after the 2015-16 El Nino winter) and November 2023 (after the 2022-23 storm winters)."),
            ("The surveys", f"2010: {e['2010']['ground_density_per_m2']:.1f} ground returns per m2; 2016: {e['2016']['ground_density_per_m2']:.1f}; 2023: {e['2023']['ground_density_per_m2']:.1f}. The 2010 file carries no collection date, so intervals are approximate to within the 2010 survey year."),
            ("The window", "Land and cliffs only; the sea (no returns) lies to the west. Stable reference ground is inland, at least 150 m from the sea and under 8 degrees.")]
    _rows(pl, 8.4, 3.35, rows, 76)
    return pl.save("46_pacifica_site_map.png")


def _hs(F, key="2023"):
    d = np.nan_to_num(F["D"][key], nan=np.nanmedian(F["D"][key]))
    return hillshade(d, 2.0, 315, 40)


def fig_dod(st, F):
    pl = Plate("Where the bluffs lost and gained ground", "Elevation change between lidar surveys (DEM of difference, 2 m cells); only changes larger than the local level of detection are coloured",
               _src(st), "Red = surface lowered (cut), blue = surface raised (fill), in metres")
    h = st["half_m"]
    ext = (-h, h, -h, h)
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    W = 3.3
    for k, iv in enumerate(INTERVALS):
        a_, b_ = iv.split("-")
        x0 = 0.6 + k * 3.85
        pl.label(x0, 8.0 + 0.05, f"{a_} to {b_}")
        ax = pl.ax(x0, 3.95, W, 3.95)
        ax.imshow(_hs(F), origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95)
        dz, lod = F["dods"][(a_, b_)], F["lods"][(a_, b_)]
        sig = np.where(np.abs(dz) > lod, dz, np.nan)
        ax.imshow(sig, origin="lower", extent=ext, cmap="RdBu", norm=norm, interpolation="nearest")
        ax.set_xlim(-250, 250); ax.set_ylim(-h, h); ax.set_xticks([]); ax.set_yticks([])
        _scalebar(ax, -230, -h + 30, 100.0)
        r = st["dod"][iv]["coastal_band_60m"]["lod95"]
        pl.text(x0, 3.85, f"coastal band: {r['erosion_m3'] / 1000:.1f}k m3 lowered, {r['deposition_m3'] / 1000:.1f}k raised, net {r['net_m3'] / 1000:+.1f}k", fontsize=7.6, color=MUTED, va="top")
    _vcbar(pl, 12.2, 3.95, 3.95, "RdBu", norm, "change, m", [-4, -2, 0, 2, 4])
    # M3C2 on the face
    cores = F["cores"]
    ax = pl.ax(0.6, 0.95, 3.3, 2.5)
    d, lod, _ = F["m3"]["2016-2023"]
    sc = ax.scatter(cores[:, 0], cores[:, 1], c=np.where(np.abs(d) > lod, d, np.nan), s=1.2, cmap="RdBu", vmin=-1.5, vmax=1.5, lw=0)
    ax.set_facecolor("#EEEEEA"); ax.set_xlim(-250, 250); ax.set_ylim(-h, h); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    del sc
    pl.label(0.6, 3.6, "Bluff face by M3C2, 2016 to 2023 (normal distance, +-1.5 m)")
    m = st["m3c2_face"]
    rows = [("DEM of difference", "Both DEMs are the mean of ground-class returns in 2 m cells, registered first on stable ground. A change counts when it exceeds a 95% level of detection that grows with slope (vertical error of the two DEMs plus slope times the horizontal registration error)."),
            ("M3C2 on the face", f"On the steep face (20 degrees or more) the surface moves along its normal; M3C2 measures that directly. Net volume on the face 2016 to 2023: M3C2 {m['2016-2023']['net_m3'] / 1000:.1f}k m3 against {m['2016-2023']['dod_net_same_cells_m3'] / 1000:.1f}k m3 from the DEM of difference over the same cells; per-cell agreement r = {m['2016-2023']['corr_with_dod_vertical']:.2f}."),
            ("Inland changes", "Changes more than 60 m from the sea are mostly construction, grading and ground-classification differences, not erosion; they are reported separately.")]
    _rows(pl, 4.3, 3.35, rows, 90)
    return pl.save("47_pacifica_dod_maps.png")


def fig_hotspots(st, F):
    pl = Plate("Two places in detail: profiles through the bluff", "East-west cross-sections through the two most eroded 50 m blocks of the 2010 to 2023 difference, in the three surveys",
               _src(st), "Elevations in each survey's own vertical datum after registration on stable ground")
    h = st["half_m"]
    dz = F["dods"][("2010", "2023")]
    lod = F["lods"][("2010", "2023")]
    coast = F["coast"]
    sig = np.abs(dz) > lod
    # blocks of 50 m
    kb = 25
    best = {}
    for nm, sign in (("lowered", -1), ("second", -1)):
        sc = np.zeros((dz.shape[0] // kb, dz.shape[1] // kb))
        for i in range(sc.shape[0]):
            for j in range(sc.shape[1]):
                s_ = (slice(i * kb, (i + 1) * kb), slice(j * kb, (j + 1) * kb))
                m = coast[s_] & sig[s_] & np.isfinite(dz[s_])
                sc[i, j] = (sign * dz[s_][m]).sum() * 4 if m.any() else 0
        if nm == "second":
            i0, j0 = best["lowered"][0] // kb, best["lowered"][1] // kb
            for ii in range(sc.shape[0]):
                for jj in range(sc.shape[1]):
                    if np.hypot(ii - i0, jj - j0) < 5:          # keep the second block at least 250 m from the first
                        sc[ii, jj] = 0
        i, j = np.unravel_index(np.argmax(sc), sc.shape)
        best[nm] = (i * kb + kb // 2, j * kb + kb // 2, sc[i, j])
    cols = {"2010": "#7F8C8D", "2016": "#E67E22", "2023": "#1F6FB2"}
    for c, (nm, ttl) in enumerate((("lowered", "Most ground lost"), ("second", "Second most, 250 m or more away"))):
        i, j, v = best[nm]
        x0 = 0.6 + c * 7.7
        pl.label(x0, 8.35, f"{ttl}: {v:,.0f} m3 in a 50 m block")
        ax = pl.ax(x0, 4.4, 3.5, 3.5)
        ax.imshow(_hs(F), origin="lower", extent=(-h, h, -h, h), cmap="gray", vmin=0.2, vmax=0.95)
        ax.imshow(np.where(sig, dz, np.nan), origin="lower", extent=(-h, h, -h, h), cmap="RdBu", norm=TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4), interpolation="nearest")
        cx, cy = j * 2.0 - h, i * 2.0 - h
        ax.set_xlim(cx - 80, cx + 80); ax.set_ylim(cy - 80, cy + 80); ax.set_xticks([]); ax.set_yticks([])
        ax.plot([cx - 80, cx + 80], [cy, cy], color="k", lw=1.0, ls="--")
        _scalebar(ax, cx - 70, cy - 70, 50.0)
        bx = pl.ax(x0 + 4.3, 4.4, 3.1, 3.5)
        row = i
        xs = np.arange(dz.shape[1]) * 2.0 - h
        for t in ("2010", "2016", "2023"):
            bx.plot(xs, F["D"][t][row], color=cols[t], lw=1.4, label=t)
        bx.set_xlim(cx - 80, cx + 80)
        seg = np.concatenate([F["D"][t][row, max(j - 40, 0):j + 40] for t in F["D"]])
        seg = seg[np.isfinite(seg)]
        bx.set_ylim(seg.min() - 3, seg.max() + 3)
        bx.set_xlabel("east, m (local)", fontsize=8, color=MUTED); bx.set_ylabel("elevation, m", fontsize=8, color=MUTED); bx.tick_params(labelsize=8, length=2); bx.spines[["top", "right"]].set_visible(False)
        bx.legend(fontsize=8, frameon=False, loc="upper left")
        pl.label(x0 + 4.3, 8.0, "Profile along the dashed line")
    rows = [("Reading the profiles", "Where a line drops from one survey to the next, the surface was lowered; the horizontal gap between lines at the same elevation is how far that part of the face retreated. The sea is to the left."),
            ("Caveat", "Ground returns on a steep, vegetated face are sparse, and the 2010 survey has under 0.5 ground returns per m2, so single profiles are noisy; the block totals and the volumes use all cells.")]
    _rows(pl, 0.6, 3.9, rows, 150)
    return pl.save("48_pacifica_hotspots.png")


def fig_volumes(st, F):
    pl = Plate("How much ground moved, and how fast the bluff edge retreated", "Cut and fill volumes along the coast, by interval, with uncertainty; horizontal retreat of elevation contours on the bluff face",
               _src(st), "Coastal band = within 60 m of the sea. Volumes above the 95% level of detection")
    segs = st["segments"]
    x = np.arange(len(segs))
    ax = pl.ax(0.9, 4.5, 6.4, 3.2)
    wd = 0.27
    for k, iv in enumerate(INTERVALS):
        ero = np.array([s[f"{iv}_erosion_m3"] for s in segs]) / 1000
        dep = np.array([s[f"{iv}_deposition_m3"] for s in segs]) / 1000
        ax.bar(x + (k - 1) * wd, -ero, wd, color=("#F5B7B1", "#E74C3C", "#922B21")[k])
        ax.bar(x + (k - 1) * wd, dep, wd, color=("#AED6F1", "#2E86C1", "#1B4F72")[k])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(s['y_from_m'])}\nto {int(s['y_to_m'])}" for s in segs], fontsize=7)
    ax.set_xlabel("stretch of coast, north-south position (m, south to north)", fontsize=8, color=MUTED); ax.set_ylabel("volume, thousand m3 (down = lowered)", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=8, length=2); ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#F5B7B1", label="2010-2016"), Patch(color="#E74C3C", label="2016-2023"), Patch(color="#922B21", label="2010-2023")], fontsize=7.5, frameon=False, ncol=3, loc="lower right", title="red (down): lowered; blue (up): raised, same order", title_fontsize=7.5)
    pl.label(0.9, 8.35, "Volume change by stretch of coast")
    b = pl.ax(8.2, 4.5, 3.2, 3.2)
    lv = "12.0"
    for k, iv in enumerate(INTERVALS):
        r = st["contour_retreat"][lv][iv]
        b.bar(k, r["median_m"], color=("#F5B7B1", "#E74C3C", "#922B21")[k])
        b.plot([k, k], [r["p10_m"], r["p90_m"]], color="k", lw=1.2)
        b.text(k, max(r["p90_m"], r["median_m"]) + 0.25, f"{r['median_m']:+.1f} m\n({r['transects']} lines)", ha="center", fontsize=7.5)
    b.axhline(0, color="k", lw=0.8)
    b.set_xticks(range(3)); b.set_xticklabels(INTERVALS, fontsize=7.5); b.set_ylabel("horizontal retreat of the 12 m contour, m", fontsize=8, color=MUTED); b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False)
    pl.label(8.2, 8.35, "Contour retreat, 12 m contour")
    c = pl.ax(12.0, 4.5, 3.1, 3.2)
    vals, labs = [], ["lod 68%", "lod 95%", "lod 99%", "no lod"]
    for key in ("lod68", "lod95", "lod99", "none"):
        vals.append(st["dod"]["2010-2023"]["coastal_band_60m"][key]["net_m3"] / 1000)
    c.bar(range(4), vals, color="#922B21")
    for i, v in enumerate(vals):
        c.text(i, v - 0.8, f"{v:.1f}", ha="center", va="top", fontsize=7.5, color="white")
    c.set_xticks(range(4)); c.set_xticklabels(labs, fontsize=7.5); c.set_ylabel("net volume 2010-2023, thousand m3", fontsize=8, color=MUTED); c.tick_params(labelsize=8, length=2); c.spines[["top", "right"]].set_visible(False)
    pl.label(12.0, 8.35, "Detection threshold")
    d = st["dod"]
    rows = []
    for iv in INTERVALS:
        r = d[iv]["coastal_band_60m"]["lod95"]
        rows.append(f"{iv}: {r['erosion_m3']:,.0f} m3 lowered, {r['deposition_m3']:,.0f} m3 raised, net {r['net_m3']:+,.0f} m3 ({r['significant_area_m2'] / 1e4:.1f} ha changed of {r['area_m2'] / 1e4:.1f} ha)")
    cl = st["closure"]["coastal_band_60m"]
    inl = d["2010-2023"]["inland"]["lod95"]
    sens = st["sensitivity_2010_2023_coastal_band"]
    _rows(pl, 0.9, 4.05, [("Coastal band (within 60 m of the sea)", ";  ".join(rows) + "."),
                           ("Internal check", f"2010 to 2016 plus 2016 to 2023 gives {cl['sum_of_parts_m3']:+,.0f} m3; 2010 to 2023 directly gives {cl['whole_m3']:+,.0f} m3 (difference {cl['difference_m3']:+,.0f}, {abs(cl['difference_m3'] / cl['whole_m3']) * 100:.0f}%)."),
                           ("Not geomorphic", f"Inland (more than 60 m from the sea) the 2010 to 2023 difference is {inl['net_m3']:+,.0f} m3 (lowered {inl['erosion_m3']:,.0f}, raised {inl['deposition_m3']:,.0f}); that is construction, grading and classification, and is not counted as erosion.")], 100, 8.0)
    _rows(pl, 8.2, 4.05, [("Sensitivity (2010 to 2023, coastal band, net)", f"1 m cells {sens['cell_1m']['net_m3'] / 1000:.1f}k m3 (only {sens['cell_1m']['coastal_cells_with_both_dems_share'] * 100:.0f}% of cells have both surveys), 2 m {sens['cell_2m']['net_m3'] / 1000:.1f}k, 4 m {sens['cell_4m']['net_m3'] / 1000:.1f}k; without the registration step {sens['no_registration_2010_2023']['net_m3'] / 1000:.1f}k m3."),
                          ("Contour retreat", "Only profiles where the contour crosses a steep face are used; the 2010 survey is sparse, so its contour positions carry about +-1 m. Median retreat over 2016 to 2023 is about 0.8 m at the 12 m contour, with 18% of the 67 profiles retreating by 3 m or more.")], 100, 8.0)
    return pl.save("49_pacifica_volumes.png")


def fig_tests(st, F):
    pl = Plate("How far can the change numbers be trusted?", "Registration on stable ground, level of detection, changes of known size planted into the data, agreement between methods and between intervals",
               _src(st), "Stable ground = inland, 150 m or more from the sea, under 8 degrees (0.44 ha)")
    a = pl.ax(0.9, 5.2, 3.4, 2.65)
    for k, t in enumerate(("2010", "2016")):
        r = st["registration"][t]["held_out_north_half"]
        v = [r["before"]["robust_sd_m"], r["after_fit_on_south_half"]["robust_sd_m"], r["after_fit_on_all"]["robust_sd_m"]]
        a.bar(np.arange(3) + k * 0.38 - 0.19, np.array(v) * 100, 0.36, color=("#7F8C8D", "#1F6FB2")[k], label=t)
    a.set_xticks(range(3)); a.set_xticklabels(["before", "fit on the\nsouth half", "fit on all"], fontsize=7.5)
    a.set_ylabel("spread on the held-out north half, cm", fontsize=8, color=MUTED); a.tick_params(labelsize=8, length=2); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=7.5, frameon=False, title="survey", title_fontsize=7.5)
    pl.label(0.9, 8.35, "Registration (ICP) on held-out ground")
    b = pl.ax(5.1, 5.2, 3.4, 2.65)
    ls = st["lod_by_slope_coastal_band"]
    names = list(ls)
    b.bar(np.arange(5) - 0.19, [ls[n]["median_lod95_2016_2023_m"] * 100 for n in names], 0.36, color="#2E86C1", label="2016 to 2023")
    b.bar(np.arange(5) + 0.19, [ls[n]["median_lod95_2010_2023_m"] * 100 for n in names], 0.36, color="#1B4F72", label="2010 to 2023")
    b.set_xticks(range(5)); b.set_xticklabels([f"{n}" for n in names], fontsize=7.5); b.set_xlabel("slope, degrees", fontsize=8, color=MUTED); b.set_ylabel("95% level of detection, cm", fontsize=8, color=MUTED)
    b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False); b.legend(fontsize=7.5, frameon=False, loc="upper left")
    pl.label(5.1, 8.35, "Level of detection grows with slope")
    c = pl.ax(9.4, 5.2, 3.0, 2.65)
    pr = st["planted_change_on_stable_ground"]["results"]
    amps = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    for side, col in (("6", "#E67E22"), ("12", "#1F6FB2")):
        c.plot([a_ * 100 for a_ in amps], [pr[f"{side}m_{a_}m"]["cells_detected_share"] for a_ in amps], "-o", color=col, ms=4, label=f"{side} m patches")
    c.set_xscale("log"); c.set_xlabel("planted change, cm", fontsize=8, color=MUTED); c.set_ylabel("share of planted cells detected", fontsize=8, color=MUTED); c.set_ylim(0, 1.05)
    from matplotlib.ticker import FuncFormatter
    c.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); c.tick_params(labelsize=8, length=2); c.spines[["top", "right"]].set_visible(False); c.legend(fontsize=7.5, frameon=False, loc="center right")
    pl.label(9.4, 8.35, "Planted changes on stable ground")
    d = pl.ax(13.0, 5.2, 2.2, 2.65)
    fp = [st["dod"][iv]["false_positive_share_on_stable"] * 100 for iv in INTERVALS]
    d.bar(range(3), fp, color="#7F8C8D"); d.axhline(5, color="k", lw=0.8, ls="--")
    d.set_xticks(range(3)); d.set_xticklabels(["10-16", "16-23", "10-23"], fontsize=7.5); d.set_ylabel("stable cells flagged, %", fontsize=8, color=MUTED); d.tick_params(labelsize=8, length=2); d.spines[["top", "right"]].set_visible(False)
    pl.label(13.0, 8.35, "False alarms")
    e = pl.ax(0.9, 1.15, 3.4, 2.6)
    m = st["m3c2_face"]
    ks = ("2010-2016", "2016-2023")
    e.bar(np.arange(2) - 0.19, [m[k]["net_m3"] / 1000 for k in ks], 0.36, color="#2E86C1", label="M3C2")
    e.bar(np.arange(2) + 0.19, [m[k]["dod_net_same_cells_m3"] / 1000 for k in ks], 0.36, color="#1B4F72", label="DEM of difference")
    e.axhline(0, color="k", lw=0.8); e.set_xticks(range(2)); e.set_xticklabels(ks, fontsize=8); e.set_ylabel("net volume on the face, thousand m3", fontsize=8, color=MUTED)
    e.tick_params(labelsize=8, length=2); e.spines[["top", "right"]].set_visible(False); e.legend(fontsize=7.5, frameon=False, loc="lower left")
    pl.label(0.9, 4.1, "Two methods on the bluff face")
    f = pl.ax(5.1, 1.15, 3.4, 2.6)
    cz = st["closure"]
    f.bar([0, 1, 2], [cz["coastal_band_60m"]["sum_of_parts_m3"] / 1000, cz["coastal_band_60m"]["whole_m3"] / 1000, st["sensitivity_2010_2023_coastal_band"]["no_registration_2010_2023"]["net_m3"] / 1000], color=["#2E86C1", "#1B4F72", "#7F8C8D"])
    f.set_xticks(range(3)); f.set_xticklabels(["two intervals\nadded", "2010 to 2023\ndirect", "direct, no\nregistration"], fontsize=7.5); f.set_ylabel("net volume, coastal band, thousand m3", fontsize=8, color=MUTED); f.axhline(0, color="k", lw=0.8)
    f.tick_params(labelsize=8, length=2); f.spines[["top", "right"]].set_visible(False)
    pl.label(5.1, 4.1, "Closure and the effect of registration")
    r10, r16 = st["registration"]["2010"], st["registration"]["2016"]
    rows = [("Registration", f"ICP moved 2010 by {np.hypot(*r10['translation_m'][:2]):.2f} m horizontally and {r10['translation_m'][2]:+.2f} m vertically and 2016 by {np.hypot(*r16['translation_m'][:2]):.2f} m and {r16['translation_m'][2]:+.2f} m: the surveys were already close. The shifts are small. Fitting on the southern half of the stable ground and testing on the northern half improves 2016 (spread {st['registration']['2016']['held_out_north_half']['before']['robust_sd_m'] * 100:.1f} to {st['registration']['2016']['held_out_north_half']['after_fit_on_south_half']['robust_sd_m'] * 100:.1f} cm) but not 2010 ({st['registration']['2010']['held_out_north_half']['before']['robust_sd_m'] * 100:.1f} to {st['registration']['2010']['held_out_north_half']['after_fit_on_south_half']['robust_sd_m'] * 100:.1f} cm): the sparse 2010 survey is only weakly constrained, and skipping registration changes the 2010 to 2023 net volume by 17%."),
            ("Level of detection", f"Stable-ground spread: {st['dod']['2016-2023']['sigma_z_stable_m'] * 100:.1f} cm (2016 to 2023) and {st['dod']['2010-2023']['sigma_z_stable_m'] * 100:.1f} cm (2010 to 2023); the 95% level is {st['dod']['2010-2023']['lod_flat_m'] * 100:.0f} cm on flat ground and {ls['45-90']['median_lod95_2010_2023_m'] * 100:.0f} cm on slopes over 45 degrees. It flags 5 to 6% of the stable cells, as designed. The stable ground is small (0.44 ha), so these spreads are themselves uncertain."),
            ("Planted changes", "Steps of 10 cm or more laid on flat stable ground are found in 97% of cells and their volume is recovered within 4%; 5 cm steps are not detected.  The steep face, where the level of detection is about twice as large, is not tested.")]
    _rows(pl, 9.2, 4.2, rows, 62, 7.6)
    return pl.save("50_pacifica_accuracy_tests.png")
