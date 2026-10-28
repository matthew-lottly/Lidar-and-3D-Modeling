"""Plates 20 and 21: wires, poles and substation structures from lidar compared with mapped Overture power features."""
from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.lines import Line2D

from .plates import MUTED, Plate, map_frame, north_arrow, scalebar

SRC = ("Lidar: USGS 3D Elevation Program (3DEP) {project} on AWS (usgs-lidar-public), collected {date}; no wire or tower classes are present, so everything here is found from the points. "
       "Mapped features: Overture Maps infrastructure (power), release 2026-09-23.1, which derives from OpenStreetMap and other sources.")


def _load(S, A, pts):
    x, y, z, cls = pts
    return x, y, z, cls


def _lines(feats):
    out = []
    for cl in ("power_line", "minor_line"):
        for g, _ in feats.get(cl, []):
            for p in (g.geoms if g.geom_type == "MultiLineString" else [g]):
                out.append(np.asarray(p.coords))
    return out


def fig_overview(S, A, pts, hag, lines, towers, ovpoles, sub_poly):
    x, y, z, cls = pts
    h = S["window_m"] / 2
    pl = Plate("Poles, wires and a substation from lidar", f"What a {S['pts_per_m2']:.0f} points per m² survey shows around White River Substation, Pierce County, Washington, against the mapped power features",
               SRC.format(project=S["project"], date=S["collected"]), "Local grid, metres from the substation centre (UTM 10N)")
    a = pl.ax(0.6, 1.2, 7.0, 7.0)
    m = hag > 2.0
    a.scatter(x[m][::5], y[m][::5], s=0.12, c="#C9C6BC", lw=0, rasterized=True)
    wm = A["wmask"]
    a.scatter(x[wm], y[wm], s=0.9, c="#1F4E79", lw=0, rasterized=True)
    for ln in lines:
        a.plot(ln[:, 0], ln[:, 1], color="#C0392B", lw=0.45, alpha=0.8)
    a.plot(towers[:, 0], towers[:, 1], "^", mfc="white", mec="#111111", ms=5, mew=0.8, ls="none")
    a.plot(ovpoles[:, 0], ovpoles[:, 1], "o", mfc="white", mec="#2F6B3F", ms=3.2, mew=0.8, ls="none")
    P = A["P"]
    a.plot(P[:, 0], P[:, 1], "o", mfc="none", mec="#B22B8A", ms=8, mew=1.2, ls="none")
    if sub_poly is not None:
        xs, ys = sub_poly.exterior.xy
        a.plot(xs, ys, color="#B22B8A", lw=1.2)
    a.set_xlim(-h, h); a.set_ylim(-h, h); a.set_aspect("equal")
    map_frame(a, (-h, h, -h, h), 100, grid=False); scalebar(a, (-h, h, -h, h), 100); north_arrow(a, (-h, h, -h, h))
    pl.label(0.6, 8.35, "Detected wires and columns on mapped lines, towers and poles")
    leg = [Line2D([], [], color="#1F4E79", lw=3, label="wire points found in the lidar"), Line2D([], [], color="#C0392B", lw=1, label="mapped power and minor lines"),
           Line2D([], [], marker="^", mfc="white", mec="#111111", ls="none", label="mapped tower"), Line2D([], [], marker="o", mfc="white", mec="#2F6B3F", ls="none", label="mapped pole"),
           Line2D([], [], marker="o", mfc="none", mec="#B22B8A", ms=8, ls="none", label="vertical column found in the lidar"), Line2D([], [], color="#B22B8A", lw=1.2, label="mapped substation outline")]
    lg = a.legend(handles=leg, loc="lower right", fontsize=7.5, frameon=True, framealpha=0.92)
    lg.get_frame().set_edgecolor("#CFCFCF")
    W, Pl, Y = S["wires"], S["poles"], S["yard"]
    v = Y["versus_random_points"]
    pl.label(8.2, 8.35, "How well the lidar and the map agree")
    blocks = [
        ("Wires", f"{W['chains']} chains, {W['total_chain_length_m'] / 1000:.1f} km, median {W['median_hag_m']:.0f} m above ground. {W['share_points_within_5m_of_mapped_line'] * 100:.0f}% of wire points lie within 5 m of a mapped line, "
                  f"so little is invented. But only {W['mapped_line_share_with_wire_within_6m'] * 100:.0f}% of the {W['mapped_line_length_in_window_m'] / 1000:.1f} km of mapped line has a detected wire within 6 m: the rest is under trees, underground, duplicated in the map, or too sparsely sampled."),
        ("Poles and towers", f"{Pl['detected']} columns found; {Pl['matched_3m']} lie within 3 m of one of the {Pl['overture_towers'] + Pl['overture_poles']} mapped poles and towers (median offset {Pl['median_match_distance_m']:.1f} m). "
                             f"Recall {Pl['recall'] * 100:.0f}%, precision {Pl['precision'] * 100:.0f}%. A pole is a few returns thick at this density, so blind detection is weak; the mapped positions are the better guide, "
                             "and where tested nearly every mapped pole and tower has returns above it."),
        ("Substation yard", f"{Y['structures_found']} elevated structures found inside the yard ({Y['area_m2'] / 1000:.0f} thousand m²), median top {Y['median_structure_top_hag_m']:.0f} m above ground. "
                            f"Of {Y['equipment_mapped']} mapped switches, transformers and portals, {v['mapped_within_3m'] * 100:.0f}% have a lidar structure within 3 m, against {v['random_within_3m'] * 100:.0f}% for random points in the yard "
                            f"({v['mapped_within_1p5m'] * 100:.0f}% against {v['random_within_1p5m'] * 100:.0f}% within 1.5 m)."),
    ]
    y0 = 7.95
    for t, b in blocks:
        pl.text(8.2, y0, t, fontsize=9.5, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(b, 76))
        pl.text(8.2, y0 - 0.3, w, fontsize=8.6, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.3 + 0.2 * (w.count("\n") + 1) + 0.35
    return pl.save("20_utility_overview.png")


def fig_profile(S, A, pts, hag, sub_poly):
    x, y, z, cls = pts
    pl = Plate("Wires and a substation yard, in profile and in plan", "A side view of the lidar through the substation, with the wires the detector found, and the elevated structures in the yard",
               SRC.format(project=S["project"], date=S["collected"]), "Local grid, metres from the substation centre (UTM 10N)")
    wm = A["wmask"]
    # side view along x for a 60 m band of y through the yard
    band = np.abs(y - (-5.0)) < 30
    a = pl.ax(0.8, 5.15, 14.4, 2.85)
    nm = band & ~wm & (hag > 0.3)
    a.scatter(x[nm][::2], z[nm][::2], s=0.12, c="#B8B2A0", lw=0, rasterized=True)
    gr = band & (hag <= 0.3)
    a.scatter(x[gr][::6], z[gr][::6], s=0.1, c="#7A6A52", lw=0, rasterized=True)
    wb = band & wm
    a.scatter(x[wb], z[wb], s=1.4, c="#1F4E79", lw=0, rasterized=True)
    a.set_xlim(-400, 400); a.tick_params(labelsize=8, length=2); a.spines[["top", "right"]].set_visible(False)
    a.set_ylabel("elevation, m", fontsize=8, color=MUTED); a.set_xlabel("east, m from the substation centre", fontsize=8, color=MUTED)
    pl.label(0.8, 8.3, "Side view through the yard (a 60 m wide band): wires in blue")
    # yard plan
    b = pl.ax(0.8, 0.95, 3.8, 3.3)
    inyard = (np.abs(x) < 130) & (np.abs(y) < 110)
    k = inyard & (hag > 1.0) & ~wm
    b.scatter(x[k][::2], y[k][::2], c=np.clip(hag[k][::2], 1, 15), s=0.2, cmap="viridis", lw=0, rasterized=True)
    if sub_poly is not None:
        xs, ys = sub_poly.exterior.xy
        b.plot(xs, ys, color="#B22B8A", lw=1)
    St = A["S"]
    b.plot(St[:, 0], St[:, 1], "x", color="#C0392B", ms=3, mew=0.6, ls="none")
    eq = A["equip"]
    b.plot(eq[:, 0], eq[:, 1], "o", mfc="none", mec="white", ms=3.4, mew=0.5, ls="none")
    b.set_xlim(-130, 130); b.set_ylim(-110, 110); b.set_aspect("equal"); b.tick_params(labelsize=7, length=2)
    b.set_facecolor("#1B1B1B")
    pl.label(0.8, 4.6, "Yard: structures (red x), mapped equipment (white rings)")
    pl.text(5.3, 4.0, "Height above ground in the yard, 1 to 15 m (viridis). The structures are connected groups of points more than 1.5 m above ground that are not wires; "
                      "a group can be one piece of equipment or several that touch, so the count of structures is not a count of equipment.", fontsize=8.3, color=MUTED, va="top", linespacing=1.35, wrap=False) if False else None
    notes = ["Height above ground in the yard runs from 1 m to 15 m (viridis). Structures are connected groups of points more than 1.5 m above ground that are not wires, drawn as red crosses.",
             "A group can be one piece of equipment or several that touch, so the number of structures is not a count of equipment, and the match to the map is a position check, not an identification.",
             "Wires are found where the lidar points form a thin, near-horizontal line with open space around it. Wires hidden under tree crowns, under 4 m above ground, or very thin conductors can be missed.",
             "The side view shows the sag of each span between supports: a catenary curve in the blue points."]
    y0 = 4.4
    for t in notes:
        w = "\n".join(textwrap.wrap(t, 95))
        pl.text(5.3, y0, w, fontsize=8.5, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.2 * (w.count("\n") + 1) + 0.2
    return pl.save("21_utility_profile.png")
