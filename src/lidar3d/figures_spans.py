"""Plates 25 to 27: the span-anchored wire model, and the 3D scene with fitted supports and wires."""
from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.lines import Line2D

from .plates import MUTED, Plate

NAVY, CYAN, MAG = "#1F4E79", "#17A2B8", "#B22B8A"


def _src(site):
    return (f"Lidar: USGS 3DEP {site['project']}, collected {site['collected']}. Imagery (3D colour only): NAIP {site['naip']}. Reference: OpenStreetMap power features as carried by Overture Maps, "
            "used only to compare, never to build the model.")


def _crop(im, pad=12):
    m = (im < 250).any(2)
    r, c = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    return im[max(r[0] - pad, 0):r[-1] + pad, max(c[0] - pad, 0):c[-1] + pad]


def fig_model(site, st, F):
    pl = Plate("Wires fitted between supports, not grown from fragments", "Every elevated lidar return is tested against a straight-in-plan, parabolic-in-height wire between each pair of nearby anchors",
               _src(site), "Local grid in metres from the substation centre (UTM 10N)")
    a = pl.ax(0.6, 1.2, 7.2, 7.0)
    P = F["P"]
    a.scatter(P[::5, 0], P[::5, 1], s=0.15, c="#CFCBC0", lw=0, rasterized=True)
    from . import spans
    for _, _, w in F["res"]:
        p, f = spans.wire_xyz(w, 1.0)
        o = f == 1
        a.scatter(p[o, 0], p[o, 1], s=0.6, c=NAVY, lw=0, zorder=3)
        if (~o).any():
            a.scatter(p[~o, 0], p[~o, 1], s=1.4, c=CYAN, lw=0, zorder=4)
    sp_ = F["sup"]
    a.plot(sp_.x[sp_.tier != "support"], sp_.y[sp_.tier != "support"], "x", color="#888888", ms=3.5, mew=0.8, ls="none", zorder=5)
    a.plot(sp_.x[sp_.tier == "support"], sp_.y[sp_.tier == "support"], "^", mfc="none", mec=MAG, ms=6, mew=1.2, ls="none", zorder=6)
    a.set_xlim(-400, 400); a.set_ylim(-400, 400); a.set_aspect("equal"); a.tick_params(labelsize=7, length=2)
    for sp in a.spines.values():
        sp.set_color("#999999")
    leg = [Line2D([], [], marker="o", color=NAVY, ls="none", ms=3, label="wire, returns observed"), Line2D([], [], marker="o", color=CYAN, ls="none", ms=4, label="wire, filled by the fitted curve"),
           Line2D([], [], marker="^", mfc="none", mec=MAG, ls="none", label="support: wires end here, none pass over"), Line2D([], [], marker="x", color="#888888", ls="none", label="wire end with an unbroken wire over it"), Line2D([], [], marker="o", color="#CFCBC0", ls="none", ms=3, label="elevated lidar returns")]
    lg = a.legend(handles=leg, loc="lower right", fontsize=7.2, framealpha=0.93); lg.get_frame().set_edgecolor("#CFCFCF")
    pl.label(0.6, 8.35, "Fitted wires over the elevated returns")
    # hold-out
    h = st["interpolation_holdout"]
    b = pl.ax(9.0, 5.7, 6.0, 2.2)
    gaps = ["20", "40"]
    xs = np.arange(2)
    b.bar(xs - 0.2, [h[g]["rms_linear_m"] * 100 for g in gaps], 0.38, color="#9A9A9A", label="straight line across the gap")
    b.bar(xs + 0.2, [h[g]["rms_parabola_m"] * 100 for g in gaps], 0.38, color=NAVY, label="fitted curve")
    for i, g in enumerate(gaps):
        b.text(i - 0.2, h[g]["rms_linear_m"] * 100 + 0.6, f"{h[g]['rms_linear_m'] * 100:.0f}", ha="center", fontsize=8)
        b.text(i + 0.2, h[g]["rms_parabola_m"] * 100 + 0.6, f"{h[g]['rms_parabola_m'] * 100:.0f}", ha="center", fontsize=8)
    b.set_xticks(xs); b.set_xticklabels([f"{g} m hidden stretch\n({h[g]['n']} tests)" for g in gaps], fontsize=8)
    b.set_ylabel("height error, cm (rms)", fontsize=8, color=MUTED); b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False)
    b.legend(fontsize=7.8, frameon=False, loc="upper left")
    pl.label(9.0, 8.35, "Filling a gap: test by hiding returns")
    t = ("A stretch of a well-observed wire is hidden, the curve is refitted to the rest, and the error is measured in the hidden stretch. The curve is within 4 cm even across 40 m. "
         "The hidden returns were inliers of the original fit, so this is optimistic for noisy wires.")
    pl.text(9.0, 5.1, "\n".join(textwrap.wrap(t, 88)), fontsize=8.4, color=MUTED, linespacing=1.4, va="top")
    rows = [("Wires fitted", f"{st['wires']} wires from {st['pairs']} anchor pairs tested, {st['wire_length_m'] / 1000:.1f} km; {st['observed_share'] * 100:.0f}% of that length has lidar returns, the rest is filled"),
            ("False wires", f"The same search on lidar shifted 120 to 250 m (4 shifts) finds {st['null']['null_to_real_ratio'] * 100:.0f}% as much wire length. An upper bound: shifted returns still land on real wires"),
            ("Against the lidar", f"{st['lidar_recall']['new'] * 100:.0f}% of the wire returns found by the earlier detector (outside the yard) lie within 1 m of a fitted wire"),
            ("Against the map", f"{st['osm']['coverage_new'] * 100:.0f}% of mapped line length lies within 6 m of a fitted wire, against {st['osm']['coverage_old_chain_model'] * 100:.0f}% for the earlier fragment model; "
                                f"{st['osm']['wire_near_map_10m'] * 100:.0f}% of fitted wire lies within 10 m of a mapped line. Of the mapped length with no wire within 6 m, {st['osm']['uncovered_share_within_15m'] * 100:.0f}% has a fitted wire within 15 m (likely a position offset in the map)")]
    y0 = 4.05
    for k, v in rows:
        pl.text(9.0, y0, k, fontsize=9, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v, 92))
        pl.text(9.0, y0 - 0.25, w, fontsize=8.2, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.28 + 0.16 * (w.count("\n") + 1) + 0.2
    return pl.save("25_span_model.png")


def fig_scene(site, st, im):
    pl = Plate("The utility model in its lidar and imagery scene", "Supports, cross-arms, insulators and wires are fitted geometry; ground and surroundings are lidar coloured from NAIP imagery",
               _src(site), "Pole and tower proportions, insulators and wire thickness are assumed, not measured")
    c = _crop(im)
    h, w = c.shape[:2]
    W = 14.8
    H = min(W * h / w, 6.6)
    W = H * w / h
    a = pl.ax(0.6 + (14.8 - W) / 2, 1.1, W, H)
    a.imshow(c); a.axis("off")
    pl.label(0.6, 8.35, "White River Substation, looking north-north-east")
    K = st["support_kinds"]
    t = (f"{st['supports']} supports ({K.get('tower', 0)} lattice towers, {K.get('pole', 0)} poles; type inferred from height and arm width) are where fitted wires end and none passes unbroken over; {st['attachment_only']} more wire ends are drawn as thin posts. "
         f"Of the {st['supports_outside_yard']} supports outside the yard, {st['supports_vs_osm']['5.0']['matched']} lie within 5 m of one of the {st['osm_supports_outside_yard']} mapped towers or poles.")
    pl.text(0.6, 1.0, t, fontsize=8.6, color=MUTED, va="top")
    return pl.save("26_scene_overview.png")


def fig_closeups(site, st, ims, captions):
    pl = Plate("Close views: fitted geometry, and what was observed or filled", "Two stretches, each drawn as it might look (left) and as evidence (right)",
               _src(site), "Pole and tower proportions, insulators and wire thickness are assumed, not measured")
    for k, (im, cap) in enumerate(zip(ims, captions)):
        row, col = divmod(k, 2)
        c = _crop(im)
        h, w = c.shape[:2]
        Wd = 7.3
        Ht = min(Wd * h / w, 2.75)
        Wd = Ht * w / h
        top = 8.25 - row * 3.7
        a = pl.ax(0.6 + col * 7.6 + (7.3 - Wd) / 2, top - Ht - 0.2, Wd, Ht)
        a.imshow(c); a.axis("off")
        pl.label(0.6 + col * 7.6, top, cap[0])
        pl.text(0.6 + col * 7.6, top - Ht - 0.3, "\n".join(textwrap.wrap(cap[1], 90)), fontsize=8.2, color=MUTED, linespacing=1.35, va="top")
    return pl.save("27_scene_closeup.png")
