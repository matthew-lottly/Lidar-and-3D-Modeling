"""Plates 22 to 24: conductor model with gap filling, fused network against OpenStreetMap, and the evidence and QA summary."""
from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.lines import Line2D

from .plates import MUTED, Plate, map_frame, north_arrow, scalebar

SRC = ("Lidar: USGS 3DEP {project}, collected {date}. Imagery: {imagery}. Radar: {radar}. Reference: OpenStreetMap power features as carried by Overture Maps (release 2026-09-23.1, "
       "OSM planet snapshot 2026-09-06), used only to compare with the model, never to build it.")
NAVY, CYAN, MAG, RED, GRN = "#1F4E79", "#17A2B8", "#B22B8A", "#C0392B", "#2F6B3F"


def _src(S):
    s = S["site"]
    return SRC.format(project=s["project"], date=s["collected"], imagery=s["imagery"], radar=s["radar"])


def fig_conductors(S, R):
    cond = R["cond"]
    pl = Plate("Joining broken wires and filling the gaps", "Wire fragments the lidar found are grouped into conductors; each span between supports is fitted with a sag curve that bridges the breaks",
               _src(S), "Heights above ground; supports are the peaks of each conductor's height profile")
    order = sorted(range(len(cond)), key=lambda i: -cond[i]["interpolated_bins"])[:3]
    for k, i in enumerate(order):
        c = cond[i]
        a = pl.ax(1.2, 6.55 - k * 1.8, 10.5, 1.25)
        s = c["s"] - c["s"][0]
        obs = c["flag"] == 1
        a.plot(s[c["flag"] >= 1], c["z_fill"][c["flag"] >= 1], color="#BBBBBB", lw=1.0)
        a.scatter(s[obs], c["z_obs"][obs], s=7, c=NAVY, lw=0, label="observed in the lidar")
        itp = c["flag"] == 2
        a.scatter(s[itp], c["z_fill"][itp], s=9, c=CYAN, lw=0, label="filled by the sag curve")
        a.plot(s[c["supports"]], np.interp(c["supports"], np.arange(len(s)), np.nan_to_num(c["z_fill"], nan=np.nanmean(c["z_fill"]))), "^", mfc="white", mec=MAG, ms=7, mew=1.4, ls="none", label="support at a height peak")
        a.set_ylabel("m above ground", fontsize=7.5, color=MUTED); a.tick_params(labelsize=7.5, length=2); a.spines[["top", "right"]].set_visible(False)
        if k == 0:
            a.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=3)
        if k == 2:
            a.set_xlabel("distance along the conductor, m", fontsize=8, color=MUTED)
        pl.text(0.7, 7.8 - k * 1.8, f"{chr(65 + k)}", fontsize=12, fontweight="bold", va="top")
        pl.text(12.0, 7.8 - k * 1.8, f"{len(c['members'])} fragments, {c['length']:.0f} m long\n{c['observed_bins'] * 2} m observed, {c['interpolated_bins'] * 2} m filled\n{len(c['spans'])} spans, {sum(s_['filled'] for s_ in c['spans'])} fitted",
                fontsize=8.5, color=MUTED, linespacing=1.4, va="top")
    cd = S["conductors"]
    notes = [f"{cd['fragments']} wire fragments became {cd['n']} conductors. {cd['spans']} spans between supports, {cd['spans_filled']} fitted (median span {cd['median_span_m']:.0f} m, median sag {cd['median_sag_m']:.1f} m). "
             f"{cd['interpolated_bins'] * cd['bin_m']:.0f} m of conductor was filled in, against {cd['observed_bins'] * cd['bin_m']:,.0f} m observed.",
             f"Fragments are joined only when they are collinear within 4 degrees and 1.5 m, no more than 60 m apart and within 6 m in height, so parallel conductors in a bundle stay separate. "
             f"A span is filled only when over a third of it was observed and the parabola fits within 0.8 m.",
             f"Check against the map: {100 - S['osm_lines']['network_points_beyond_10m_of_any_osm_line'] * 100:.0f}% of the observed conductor points (every third one) lie within 10 m of a mapped line. Filling adds little coverage "
             f"({S['osm_lines']['coverage_observed'] * 100:.1f}% to {S['osm_lines']['coverage_with_interpolation'] * 100:.1f}% of mapped line length), because most of the missing length is whole spans with no observed wire, which a curve cannot invent."]
    y = 2.15
    for t in notes:
        w = "\n".join(textwrap.wrap(t, 190))
        pl.text(0.9, y, w, fontsize=8.4, color=MUTED, linespacing=1.35, va="top")
        y -= 0.16 * (w.count("\n") + 1) + 0.14
    return pl.save("22_conductor_model.png")


def fig_network(S, R, pts, lines_xy, ref_towers):
    x, y, z = pts
    h = 400
    pl = Plate("Fused line and support model against OpenStreetMap", "Conductors and supports from lidar, imagery and radar (blue, cyan, magenta) over the mapped lines and supports (red, green)",
               _src(S), "Local grid, metres from the substation centre (UTM 10N)")
    a = pl.ax(0.6, 1.2, 7.0, 7.0)
    m = R["hag"] > 3
    a.scatter(x[m][::6], y[m][::6], s=0.1, c="#D5D2C8", lw=0, rasterized=True)
    for ln, st in lines_xy:
        a.plot(ln[:, 0], ln[:, 1], color={"confirmed": "#E67E22", "partial": "#F1C40F", "elevated returns, no wire found": "#C0392B", "no lidar evidence": "#7B241C"}.get(st, RED), lw=1.6, alpha=0.65, zorder=2)
    for c in R["cond"]:
        o = c["flag"] == 1
        a.scatter(c["xy"][o, 0], c["xy"][o, 1], s=1.6, c=NAVY, lw=0, zorder=3)
        f = c["flag"] == 2
        if f.any():
            a.scatter(c["xy"][f, 0], c["xy"][f, 1], s=4, c=CYAN, lw=0, zorder=4)
    ref = R["ref_all"]
    a.plot(ref[:, 0], ref[:, 1], "+", color=GRN, ms=6, mew=1.2, ls="none", zorder=5)
    F = R["final"]
    a.plot(F.x, F.y, "o", mfc="none", mec=MAG, ms=7, mew=1.2, ls="none", zorder=6)
    a.set_xlim(-h, h); a.set_ylim(-h, h); a.set_aspect("equal")
    map_frame(a, (-h, h, -h, h), 100, grid=False); scalebar(a, (-h, h, -h, h), 100); north_arrow(a, (-h, h, -h, h))
    pl.label(0.6, 8.35, "Model over the mapped features")
    leg = [Line2D([], [], color=NAVY, marker="o", ms=3, ls="none", label="conductor, observed"), Line2D([], [], color=CYAN, marker="o", ms=4, ls="none", label="conductor, filled by sag curve"),
           Line2D([], [], marker="o", mfc="none", mec=MAG, ls="none", label="fused support"), Line2D([], [], marker="+", color=GRN, ls="none", label="mapped tower or pole"),
           Line2D([], [], color="#E67E22", lw=2, label="mapped line: confirmed"), Line2D([], [], color="#F1C40F", lw=2, label="partial"), Line2D([], [], color="#C0392B", lw=2, label="elevated returns, no wire found"),
           Line2D([], [], color="#7B241C", lw=2, label="no lidar evidence")]
    lg = a.legend(handles=leg, loc="lower right", fontsize=7, frameon=True, framealpha=0.93); lg.get_frame().set_edgecolor("#CFCFCF")
    O = S["osm_lines"]
    pl.label(8.2, 8.35, "Mapped lines by what the lidar shows (outside the yard)")
    tot = O["total_length_m"]
    y0 = 7.95
    for k, nm in (("confirmed", "Confirmed: network within 6 m of 60% or more of the line"), ("partial", "Partial: 20% to 60%"), ("elevated returns, no wire found", "Under 20% covered, but isolated returns at least 8 m up along half of it"),
                  ("no lidar evidence", "Under 20% covered and no such returns")):
        v = O["by_status"].get(k, dict(lines=0, length_m=0))
        pl.text(8.2, y0, f"{v['length_m'] / 1000:.1f} km  ({v['length_m'] / tot * 100:.0f}%)", fontsize=10, fontweight="bold", va="top")
        pl.text(10.4, y0, nm, fontsize=8.5, color=MUTED, va="top")
        y0 -= 0.42
    A = S["supports"]["ablation"]
    pl.label(8.2, 6.0, "Supports")
    rk = A["lidar + imagery"]["5"]
    txt = (f"{A['lidar + imagery']['5']['detected']} fused supports outside the yard against {S['supports']['osm_supports_outside_yard']} mapped; {rk['matched']} lie within 5 m of a mapped one "
           f"(precision {rk['precision'] * 100:.0f}%, recall {rk['recall'] * 100:.0f}%). Mapped supports are traced from imagery or surveyed to varying accuracy, and OSM is incomplete, so an unmatched support is not necessarily wrong "
           f"and a missed one is not necessarily absent. Median position offset between matched pairs {rk['median_offset_m']:.1f} m.")
    w = "\n".join(textwrap.wrap(txt, 78))
    pl.text(8.2, 5.65, w, fontsize=8.6, color=MUTED, linespacing=1.4, va="top")
    t2 = (f"Lines: the network lies within {S['osm_lines']['median_offset_m']:.1f} m (median) of the mapped line where they agree, and none of the observed conductor points is more than 10 m from a mapped line, so no wire was found that the map lacks at this site. "
          f"All four lines with no lidar evidence are 230 kV, where the detector is weakest (see the next plate): that is a limit of the detector, not evidence of a mapping error.")
    w = "\n".join(textwrap.wrap(t2, 78))
    pl.text(8.2, 3.9, w, fontsize=8.6, color=MUTED, linespacing=1.4, va="top")
    return pl.save("23_fused_vs_osm.png")


def fig_evidence(S, R):
    import pandas as pd
    A = S["supports"]["ablation"]
    pl = Plate("Which sensors helped, and how the map holds up", "Ranking quality of each sensor combination for finding supports, separate tests of radar and imagery, and the map check by voltage",
               _src(S), "AUC 0.5 = chance, 1.0 = perfect")
    a = pl.ax(2.3, 5.0, 5.2, 3.0)
    names = ["lidar score", "lidar + imagery", "lidar + imagery + radar"]
    vals = [A[n]["ranking"]["auc"] for n in names]
    lo = [A[n]["ranking"]["auc"] - A[n]["ranking"]["auc_ci95"][0] for n in names]
    hi = [A[n]["ranking"]["auc_ci95"][1] - A[n]["ranking"]["auc"] for n in names]
    a.barh(range(3), vals, xerr=[lo, hi], color=["#8A8A8A", "#2B6CB0", "#9A3B12"], capsize=4)
    a.set_yticks(range(3)); a.set_yticklabels(["lidar", "lidar + imagery", "lidar + imagery + radar"], fontsize=9); a.invert_yaxis()
    a.axvline(0.5, color="#111111", lw=0.8, ls="--"); a.set_xlim(0.4, 1.0); a.tick_params(labelsize=8, length=2); a.spines[["top", "right"]].set_visible(False)
    a.set_xlabel("AUC (bars: 95% bootstrap interval over candidates)", fontsize=8, color=MUTED)
    pl.label(0.9, 8.35, "Ranking candidate supports")
    d_img = A["lidar + imagery"]["ranking"]["auc_minus_lidar_ci95"]; d_rad = A["lidar + imagery + radar"]["ranking"]["auc_minus_lidar_ci95"]
    t = (f"Wire-sag peaks alone rank at chance (0.50). The lidar object cue brings AUC to {A['lidar score']['ranking']['auc']:.2f}. Adding imagery changes it by {d_img[0]:+.2f} to {d_img[1]:+.2f} and adding radar by {d_rad[0]:+.2f} to {d_rad[1]:+.2f} "
         f"(95% intervals): at this sample size neither is distinguishable from no effect. The score is a mean of signed z-scores with signs set from physics beforehand. An unsupervised two-component mixture on the same features scored {A['unsupervised two-component mixture, same features']['ranking']['auc']:.2f}.")
    w = "\n".join(textwrap.wrap(t, 100))
    pl.text(0.9, 4.35, w, fontsize=8.5, color=MUTED, linespacing=1.4, va="top")
    # sensor tests
    pl.label(7.9, 8.35, "Radar and imagery tested on their own")
    se = S["sensors"]
    rows = [("Radar, towers", f"Sentinel-1 at 20 m shows no tower signal: AUC {se['radar_tower_auc']:.2f} for the {se['radar_tower_n']} mapped towers outside the yard against other pixels. A tower is far smaller than a pixel."),
            ("Radar, corridors", "Cleared corridors have a slightly different backscatter (VV AUC 0.57, VH reversed at 0.41; correlation with lidar canopy about 0.25): too weak to use."),
            ("Imagery, wires", f"A thin-line filter on 0.6 m NAIP gives a mean contrast of {se['imagery_line_contrast']:+.3f} (0 to 1 scale) along mapped lines against shifted copies, and does not separate lines the lidar confirms from those it misses."),
            ("Imagery, supports", "Vegetation index and line response at the base help a little in single-feature tests (AUC 0.58 and 0.68 against the mapped supports) but add nothing once the lidar cue is in.")]
    y0 = 7.95
    for h_, b in rows:
        pl.text(7.9, y0, h_, fontsize=9.2, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(b, 78))
        pl.text(7.9, y0 - 0.26, w, fontsize=8.3, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.26 + 0.17 * (w.count("\n") + 1) + 0.27
    # QA by voltage
    Q = R["Q"]
    g = Q.assign(voltage=Q.voltage.fillna("none")).groupby("voltage").apply(lambda d: pd.Series(dict(lines=len(d), km=d.length_m.sum() / 1000, cov=(d.covered * d.length_m).sum() / d.length_m.sum()))).reset_index()
    g = g.assign(_v=g.voltage.map(lambda v: -1.0 if v == "none" else float(v))).sort_values("_v", ascending=False).reset_index(drop=True)
    b = pl.ax(1.4, 1.2, 6.0, 1.6)
    b.bar([f"{int(float(v)) // 1000} kV" if v != "none" else "no tag" for v in g.voltage], g["cov"] * 100, color="#2B6CB0")
    for i, (km, n) in enumerate(zip(g.km, g.lines)):
        b.text(i, g["cov"].iloc[i] * 100 + 2, f"{km:.1f} km, {int(n)} lines", ha="center", fontsize=8)
    b.set_ylim(0, 100); b.set_ylabel("% of mapped length\nwith a detected wire", fontsize=8, color=MUTED); b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False)
    pl.label(0.9, 3.05, "Map check by voltage tag: where the detector works")
    t2 = (f"Mapped length with a detected wire: {'; '.join(f'{g.km.iloc[i]:.1f} km at {int(g._v.iloc[i]) // 1000} kV is {g[chr(99)+chr(111)+chr(118)].iloc[i] * 100:.0f}% covered' for i in range(len(g)) if g._v.iloc[i] > 0)}. "
          "The 230 kV conductors are the least often found, so an unconfirmed line is as likely a detector limit as a map error. "
          "Only 2 of the 19 line parts carry a source tag (both traced from imagery); the rest give no hint of how they were mapped.")
    w = "\n".join(textwrap.wrap(t2, 88))
    pl.text(7.9, 2.8, w, fontsize=8.4, color=MUTED, linespacing=1.4, va="top")
    return pl.save("24_sensors_and_qa.png")
