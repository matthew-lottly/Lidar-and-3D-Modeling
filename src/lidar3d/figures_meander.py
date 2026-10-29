"""Plates 28 to 31: the Wabash scroll-bar study (data and DEM accuracy, ridge map, detection and repeatability tests, imagery check)."""
from __future__ import annotations

import textwrap

import numpy as np

from .cloud import hillshade
from .plates import MUTED, Plate

RED, BLUE, NAVY, GREY = "#B03A2E", "#1F6FB2", "#1F4E79", "#8A8A8A"


def _src(st):
    return (f"Lidar: USGS 3DEP {st['project']}, flown {st['collected']}. Imagery (checks only): NAIP 2011 to 2023. Coordinates are local metres from the area centre ({st['site']}).")


def _hs(F):
    return hillshade(np.nan_to_num(F["dem_small"], nan=float(np.nanmedian(F["dem_small"]))), 2.0, 315, 35)


def _chain_xy(F, chains):
    d0, sd = F["d"][0], F["d"][1] - F["d"][0]
    out = []
    for c in chains:
        j = np.clip(np.round((c["d"] - d0) / sd).astype(int), 0, len(F["d"]) - 1)
        out.append((F["X"][c["si"], j], F["Y"][c["si"], j]))
    return out


def fig_site(st, F):
    pl = Plate("Wabash River bend, Illinois: what one lidar flight records", "Bare-earth surface from 3DEP lidar (2011), with the river and a bank line used to lay out transects",
               _src(st), "Hillshade from the 1 m bare-earth DEM, ground returns only")
    a = pl.ax(0.6, 1.1, 7.2, 7.2)
    a.imshow(_hs(F), origin="lower", cmap="gray", extent=(-1750, 1750, -1750, 1750), vmin=0.2, vmax=0.95)
    ch = F["ch"]
    a.imshow(np.ma.masked_where(~ch, ch), origin="lower", cmap="Blues", extent=(-1750, 1750, -1750, 1750), alpha=0.85, vmin=0, vmax=1.4)
    a.plot(F["line"][:, 0], F["line"][:, 1], color="#F39C12", lw=1.2)
    a.set_xlim(-1750, 1750); a.set_ylim(-1750, 1750); a.tick_params(labelsize=7, length=2)
    a.text(60, -1540, "river: no returns (water)", color=NAVY, fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    a.text(-1700, 1650, "orange: bank line used for transects", color="#B9770E", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    pl.label(0.6, 8.35, "Surface with scroll bars on both sides of the river")
    h = st["dem"]["holdout"]
    b = pl.ax(9.2, 5.2, 6.0, 2.6)
    ks = list(h["by_slope"])
    b.bar(range(len(ks)), [h["by_slope"][k]["rmse_m"] * 100 for k in ks], color=NAVY)
    for i, k in enumerate(ks):
        b.text(i, h["by_slope"][k]["rmse_m"] * 100 + 0.8, f"{h['by_slope'][k]['rmse_m'] * 100:.0f}", ha="center", fontsize=8)
    b.set_xticks(range(len(ks))); b.set_xticklabels([f"{k} deg" for k in ks], fontsize=8)
    b.set_ylabel("height error, cm (rms)", fontsize=8, color=MUTED); b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False)
    pl.label(9.2, 8.35, "Surface error: 10% of ground returns held out")
    rows = [("Data", f"{st['points'] / 1e6:.1f} million returns, {st['ground_points'] / 1e6:.1f} million classed as ground ({st['ground_density_per_m2']:.2f} per m2, about 1.3 m apart), {len(st['flight_lines'])} flight lines, {st['area_km2']:.1f} km2."),
            ("Hold-out test", f"The DEM rebuilt without 10% of the ground returns predicts them within {h['rmse_m'] * 100:.1f} cm rms ({h['by_slope']['0-2']['rmse_m'] * 100:.1f} cm on flat ground), 95% within {h['p95_abs_m'] * 100:.0f} cm, bias {h['bias_m'] * 100:+.1f} cm. This measures noise and interpolation, not the absolute datum: no surveyed checkpoints were available."),
            ("Between flight lines", "On flat overlaps (8 m cells), adjacent lines agree with a bias of " + f"{min(v['bias_m'] for v in st['interswath'].values()) * 100:.0f} to {max(v['bias_m'] for v in st['interswath'].values()) * 100:.0f} cm and a robust spread of {min(v['robust_sd_m'] for v in st['interswath'].values()) * 100:.0f} to {max(v['robust_sd_m'] for v in st['interswath'].values()) * 100:.0f} cm. Overlaps are small (a few hundred cells per pair).")]
    y0 = 4.85
    for k, v in rows:
        pl.text(9.2, y0, k, fontsize=9, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v, 80))
        pl.text(9.2, y0 - 0.25, w, fontsize=8.2, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.28 + 0.165 * (w.count("\n") + 1) + 0.2
    return pl.save("28_meander_site_accuracy.png")


def fig_ridges(st, F):
    pl = Plate("Scroll bars and swales traced from the surface", "Crest and swale lines picked along transects normal to the bank, linked along the bank, with the spacing between neighbouring crests",
               _src(st), "Crests: at least 5 cm above their surroundings, 150 m long")
    a = pl.ax(0.6, 1.85, 6.3, 6.3)
    a.imshow(_hs(F), origin="lower", cmap="gray", extent=(-1750, 1750, -1750, 1750), vmin=0.2, vmax=0.95)
    for x, y in _chain_xy(F, F["S"]):
        a.plot(x, y, color="#3498DB", lw=0.7, alpha=0.9)
    for x, y in _chain_xy(F, F["C"]):
        a.plot(x, y, color="#E74C3C", lw=0.9, alpha=0.95)
    a.set_xlim(-1750, 1750); a.set_ylim(-1750, 1750); a.tick_params(labelsize=7, length=2)
    a.text(-1700, 1650, "red: crests   blue: swales", fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    pl.label(0.6, 8.35, "Crest and swale lines")
    r = st["ridges"]
    b = pl.ax(8.6, 4.4, 6.6, 3.4)
    V, d, s = F["V"], F["d"], F["s"]
    b.imshow(V.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4, extent=(0, s[-1], d[0], d[-1]))
    for c in F["C"]:
        b.plot(c["si"] * 10.0, c["d"], color="k", lw=0.7)
    b.axhline(0, color="#F39C12", lw=1.0)
    b.set_xlabel("distance along the bank line, m", fontsize=8, color=MUTED); b.set_ylabel("distance from bank line, m\n(north +, point bar -)", fontsize=8, color=MUTED); b.tick_params(labelsize=7.5, length=2)
    pl.label(8.6, 8.35, "The same surface straightened along the bank")
    h = pl.ax(8.6, 1.55, 6.6, 1.9)
    bins = np.arange(10, 151, 10)
    for nm, col, lab in (("north_relict_set", NAVY, "north of the river"), ("south_point_bar", RED, "point bar south of the river")):
        h.hist(F["spacing"][nm], bins=bins, histtype="step", color=col, lw=1.6, label=f"{lab}: median {r[nm]['crest_spacing_median_m']:.0f} m, {r[nm]['crests_per_transect_median']:.1f} crests per transect")
    h.set_xlabel("distance between neighbouring crests along a transect, m", fontsize=8, color=MUTED); h.set_ylabel("count", fontsize=8, color=MUTED); h.tick_params(labelsize=7.5, length=2); h.spines[["top", "right"]].set_visible(False)
    h.legend(fontsize=7.8, frameon=False)
    pl.label(8.6, 3.55, "Crest spacing")
    t = (f"{r['crest_chains']} crest lines and {r['swale_chains']} swale lines; median crest relief {r['median_relief_m'] * 100:.0f} cm. Both sets run roughly parallel to the present bank. Ridges parallel to a bank are what a migrating bend leaves behind, "
         "but field edges, levees and ditches also make lines; steep lines across the arcs were dropped, parallel ones were not screened by hand.")
    pl.text(0.6, 1.45, "\n".join(textwrap.wrap(t, 108)), fontsize=8.2, color=MUTED, linespacing=1.35, va="top")
    return pl.save("29_meander_scroll_bars.png")


def fig_tests(st):
    pl = Plate("How well are the ridges found, and do they repeat?", "Ridges of known size planted into the real sampling pattern, then the same procedure run on independent halves and resamples of the real returns",
               _src(st), "Planted ridges: cosine arcs; per-return noise 3 cm; correlated surface error 4 cm over 8 m where stated")
    dt = st["detection_test"]["results"]
    amps = [0.03, 0.05, 0.08, 0.12, 0.2]
    a = pl.ax(0.9, 4.4, 6.4, 3.4)
    b = pl.ax(8.6, 4.4, 6.4, 3.4)
    for lam, ls in ((30.0, "-"), (60.0, "--")):
        for dn, col in ((0.0, NAVY), (0.04, RED)):
            rec = [dt[f"A{A}_L{lam}_corr{dn}"]["recall"] for A in amps]
            pre = [dt[f"A{A}_L{lam}_corr{dn}"]["precision"] for A in amps]
            lab = f"wavelength {lam:.0f} m, " + ("white noise only" if dn == 0 else "with correlated error")
            a.plot(np.array(amps) * 100, rec, ls, color=col, marker="o", ms=4, label=lab)
            b.plot(np.array(amps) * 100, pre, ls, color=col, marker="o", ms=4)
    for ax_, yl in ((a, "share of planted crest picks found"), (b, "share of picks that are planted crests")):
        ax_.set_ylim(0, 1.03); ax_.set_xlabel("planted ridge height, cm", fontsize=8, color=MUTED); ax_.set_ylabel(yl, fontsize=8, color=MUTED); ax_.tick_params(labelsize=8, length=2); ax_.spines[["top", "right"]].set_visible(False); ax_.axvline(5, color=GREY, lw=0.7, ls=":")
    a.legend(fontsize=7.3, frameon=False, loc="lower right")
    pl.label(0.9, 8.35, "Recall by planted ridge height")
    pl.label(8.6, 8.35, "Precision by planted ridge height")
    n0 = st["detection_test"]["noise_only_picks"]
    rp = st["repeatability"]
    rows = [("Detection limit", "With 3 cm per-return noise alone, ridges of 5 cm (30 m wavelength) are found at 99%; with 4 cm correlated surface error added (classification and interpolation mistakes are of that kind) recall drops to 73% at 5 cm and 96% at 8 cm. "
                               "Ridges of 3 cm are not reliably separable from error. The 5 cm threshold used for the map is therefore at the edge for gentle ridges and safe from about 8 to 10 cm."),
            ("False picks", f"With no ridges at all the procedure still returns {n0['corr0.0']['picks_detected']:,} picks (white noise) to {n0['corr0.04']['picks_detected']:,} (correlated error), against {st['detection_test']['real_picks']:,} picks on the real surface: "
                            f"up to {n0['corr0.04']['picks_detected'] / st['detection_test']['real_picks'] * 100:.0f}% of the real picks could be noise-like."),
            ("Repeatability", f"Two independent random halves of the ground returns: {rp['split_half']['a_in_b'] * 100:.0f}% and {rp['split_half']['b_in_a'] * 100:.0f}% of one half's picks lie within 4 m of a pick in the other. {rp['bootstrap']['runs']} bootstrap resamples: a pick persists in "
                              f"{rp['bootstrap']['mean_pick_persistence'] * 100:.0f}% of runs on average, and {rp['bootstrap']['share_picks_in_at_least_80pct_runs'] * 100:.0f}% persist in at least 80%. Resampling the same flight shares its systematic errors, so this is a floor on error, not a full account.")]
    y0 = 3.95
    for k, v in rows:
        pl.text(0.9, y0, k, fontsize=9, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v, 185))
        pl.text(0.9, y0 - 0.25, w, fontsize=8.3, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.28 + 0.17 * (w.count("\n") + 1) + 0.2
    return pl.save("30_meander_error_tests.png")


def fig_imagery(st, F):
    pl = Plate("Can aerial imagery measure how fast the bend is moving?", "Active-channel edges (open water and bare bars) from NAIP vegetation index along the same transects, 2011 to 2023",
               _src(st), "Edge = end of the run of bare ground (NDVI below 0.20) that contains the middle of the channel")
    yrs = list(F["years"])
    a = pl.ax(0.9, 4.3, 6.6, 3.2)
    for key, col, lab in (("edge_n", NAVY, "north edge (outer bank)"), ("edge_s", RED, "south edge (point bar)")):
        E = F[key]
        med = np.array([np.nanmedian(E[:, i]) for i in range(len(yrs))])
        q1 = np.array([np.nanpercentile(E[:, i], 25) for i in range(len(yrs))]); q3 = np.array([np.nanpercentile(E[:, i], 75) for i in range(len(yrs))])
        a.errorbar(yrs, med, yerr=[med - q1, q3 - med], fmt="o-", color=col, capsize=3, label=lab, lw=1.2)
    a.set_xlabel("year", fontsize=8, color=MUTED); a.set_ylabel("median edge distance from the 2011 lidar bank line, m", fontsize=8, color=MUTED); a.tick_params(labelsize=8, length=2); a.spines[["top", "right"]].set_visible(False)
    a.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    pl.label(0.9, 8.35, "Edge position by year (bars: middle half of stations)")
    b = pl.ax(8.8, 4.3, 6.4, 3.5)
    m = st["migration"]
    names = ["all years"] + [str(y) for y in yrs] + ["no 2017, 2019"]
    for k, (key, col, off) in enumerate((("outer_bank_north_edge", NAVY, -0.15), ("point_bar_south_edge", RED, 0.15))):
        L = m[key]["leave_one_year_out"]
        vals = [m[key]["rate_m_per_yr"]] + [L[str(y)]["rate"] for y in yrs] + [L["without_2017_and_2019"]["rate"]]
        los = [m[key]["ci95"][0]] + [L[str(y)]["ci95"][0] for y in yrs] + [L["without_2017_and_2019"]["ci95"][0]]
        his = [m[key]["ci95"][1]] + [L[str(y)]["ci95"][1] for y in yrs] + [L["without_2017_and_2019"]["ci95"][1]]
        x = np.arange(len(vals)) + off
        b.errorbar(x, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)], fmt="o", color=col, capsize=2, ms=4)
    b.axhline(0, color="k", lw=0.8)
    b.set_xticks(range(len(names))); b.set_xticklabels([("drop " + n) if n.isdigit() else n for n in names], fontsize=7, rotation=40, ha="right")
    b.set_ylabel("edge rate, m per year (positive = northward)", fontsize=8, color=MUTED); b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False)
    pl.label(8.8, 8.35, "Rate of change, leaving out one year at a time")
    so = m["point_bar_south_edge"]; no = m["outer_bank_north_edge"]
    t = (f"No reliable migration rate comes out of this imagery. Year to year the median edge moves by {so['between_year_sd_of_median_edge_m']:.0f} m (south) and {no['between_year_sd_of_median_edge_m']:.0f} m (north), far more than the few metres per year a bend like this might move, "
         f"because river stage, bar exposure and vegetation differ with the date of each flight. The south-edge rate is {so['rate_m_per_yr']:+.1f} m/yr with all years and {so['leave_one_year_out']['2019']['rate']:+.1f} m/yr without 2019 (95% intervals exclude zero in both, with opposite signs). "
         f"The years 2017 and 2019 stand out (they may be high-water years; stage was not checked against a gauge). The north edge is poorly defined where the bank backs onto bare fields ({no['edge_found_share']['2014'] * 100:.0f}% of stations found in 2014). "
         f"The lidar water edge and the 2011 imagery edge differ by a fixed {st['channel_vs_imagery_2011']['median_offset_m']:.0f} m (middle half {st['channel_vs_imagery_2011']['iqr_m'][0]:.0f} to {st['channel_vs_imagery_2011']['iqr_m'][1]:.0f} m): a water line against the edge of bare bank, not a positioning error. "
         "Imagery registration between years was checked on stable ground away from the river and shows a median shift of at most 0.05 pixel, so mis-registration does not explain the scatter.")
    pl.text(0.9, 3.05, "\n".join(textwrap.wrap(t, 190)), fontsize=8.4, color=MUTED, linespacing=1.4, va="top")
    return pl.save("31_meander_imagery_check.png")
