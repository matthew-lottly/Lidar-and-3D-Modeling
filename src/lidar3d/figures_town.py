"""Plates 56 to 65 for the White River town-scale utility model: site map, the full network map, and a map and a figure for each of the four extensions (mapped-line diagnosis, learned supports, equipment, scored and bridged spans)."""
from __future__ import annotations

import json
import pickle
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MPoly

from . import sitemap, spans
from .cloud import hillshade
from .ept import CACHE
from .plates import map_frame, north_arrow, scalebar
from .plates_pro import ACCENT, INK, MUTED, ProPlate

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "town"
HALF = 1500.0
EXT = (-HALF, HALF, -HALF, HALF)
C_CONF, C_ADD, C_BR, C_MAP, C_SUP, C_LEARN, C_POLE = "#1F4E79", "#E67E22", "#C0392B", "#7F8C8D", "#0B3C5D", "#D35400", "#111111"
CAT_COL = {"fitted wire within 6 m": "#1F77B4", "fitted wire 6 to 15 m away (map offset or parallel line)": "#17BECF", "elevated returns on the line, no wire fitted": "#E67E22",
           "no elevated returns, under vegetation": "#2E8B57", "no elevated returns, open ground": "#C0392B"}
CAT_SHORT = {"fitted wire within 6 m": "wire fitted", "fitted wire 6 to 15 m away (map offset or parallel line)": "wire 6-15 m away", "elevated returns on the line, no wire fitted": "returns, no wire fitted",
             "no elevated returns, under vegetation": "no returns, vegetation", "no elevated returns, open ground": "no returns, open ground"}


def J(n):
    return json.loads((PROC / n).read_text())


def load():
    D = {}
    D["model"] = pickle.load(open(CACHE / "town" / "model.pkl", "rb"))
    D["sp"] = pickle.load(open(CACHE / "town" / "spans_extras.pkl", "rb"))
    D["sm"] = pickle.load(open(CACHE / "town" / "support_model.pkl", "rb"))
    D["diag"] = pickle.load(open(CACHE / "town" / "diag.pkl", "rb"))
    up = CACHE / "town" / "unified.pkl"
    D["uni"] = pickle.load(open(up, "rb")) if up.exists() and (PROC / "extras_unified.json").exists() else None
    D["unij"] = J("extras_unified.json") if D["uni"] is not None else None
    D["stats"], D["dg"], D["su"], D["spj"], D["eq"] = J("stats.json"), J("extras_diagnosis.json"), J("extras_support.json"), J("extras_spans.json"), J("extras_equipment.json")
    G = D["model"]["G"]
    D["hs"] = hillshade(np.nan_to_num(G, nan=float(np.nanmedian(G))), 1.0, 315, 40)
    D["G"] = G
    return D


def src(D):
    s = D["stats"]
    return (f"Lidar: USGS 3DEP project {s['project']}, collected {s['collected'][0]}, about 25 returns per m2, read as 36 tiles of 500 m with a 40 m buffer; the survey carries no wire or tower classes. Reference: Overture Maps power features "
            "(OpenStreetMap-derived), used to check results and, for the learned support and span scores, as labels. Imagery: NAIP 2019. Local grid in metres from the substation centre, UTM zone 10N.")


KICK = "White River town model  ·  3 km by 3 km"
MX, MW = 0.6, 6.95            # square map: x and size
SX, SW = 7.95, 7.45           # sidebar next to a square map


def base(ax, D, extent=EXT, shade=0.55, step=500.0):
    ax.imshow(D["hs"], origin="lower", extent=(-HALF, HALF, -HALF, HALF), cmap="gray", vmin=0.15, vmax=1.0, alpha=shade + 0.35, interpolation="bilinear")
    map_frame(ax, extent, step=step, grid=False)


def mapped_lines(D):
    from shapely.geometry import LineString
    lines = []
    LP = D["diag"]
    return LP


def segs_of(res_items):
    return [spans.wire_xyz(w, 2.0)[0][:, :2] for w in res_items]


def line_layers(D):
    sp = D["sp"]
    T, best = sp["T"], sp["best"]
    conf = [spans.wire_xyz(best[(int(r.i), int(r.j))][0], 2.0)[0][:, :2] for r in T[T.baseline].itertuples()]
    add = [spans.wire_xyz(best[(int(r.i), int(r.j))][0], 2.0)[0][:, :2] for r in T[T.accepted & ~T.baseline].itertuples()]
    br = [b["xyz"][:, :2] for b in sp["brs"]]
    return conf, add, br


def lc(ax, segs, color, lw=1.0, ls="-", z=5, alpha=1.0):
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, linestyles=ls, zorder=z, alpha=alpha))


def mapped_polyline_pts(D):
    return D["diag"][["x", "y"]].to_numpy()


def rows(pl, x, y0, items, width, fs=8.2):
    for k_, v_ in items:
        pl.text(x, y0, k_, fontsize=9, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v_, width))
        pl.text(x, y0 - 0.24, w, fontsize=fs, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.27 + 0.16 * (w.count("\n") + 1) + 0.14
    return y0


def legend(ax_or_pl, handles, loc, fs=8):
    return ax_or_pl.legend(handles=handles, loc=loc, fontsize=fs, frameon=True, framealpha=0.9, edgecolor="#CCCCCC")


def pts_legend(color, label, marker="o", ms=6, mfc=None):
    return Line2D([], [], marker=marker, ls="", color=color, mfc=mfc if mfc is not None else color, ms=ms, label=label)


# ------------------------------------------------------------------------------------------------ 56 site
def fig_site(D, naip_img):
    s = D["stats"]
    bbox = (-122.6, 47.0, -121.85, 47.45)
    L = sitemap.regional_layers("whiteriver", bbox, "Washington", pop_min=20000)
    lon, lat = s["centre_lonlat"]
    dlon = HALF / (111000 * np.cos(np.radians(lat))); dlat = HALF / 111000
    pl = ProPlate(56, KICK, "Where the White River town model is", "A 3 km square of the White River valley between Sumner, Auburn and Bonney Lake in Pierce County, Washington, centred on White River Substation", src(D), "Region map: dashed grey county lines, red freeways, blue water")
    pl.panel_title(0.6, 8.14, "A", "Region", w=6.7)
    a = pl.ax(0.6, 1.0, 6.7, 6.95)
    sitemap.draw_regional(a, L, bbox, (lon, lat), (dlon, dlat), label_places=("Seattle", "Tacoma", "Auburn", "Kent", "Puyallup", "Enumclaw", "Federal Way"))
    a.annotate("model area", xy=(lon, lat), xytext=(lon + 0.1, lat - 0.2), fontsize=9, fontweight="bold", color="#E74C3C", ha="center", arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2))
    pl.panel_title(7.65, 8.14, "B", "Aerial photo, 3 km", w=3.7)
    pl.image(np.moveaxis(naip_img[:3], 0, -1), 7.65, 4.75, 3.7, 3.2)
    pl.panel_title(11.7, 8.14, "C", "Bare-earth terrain", w=3.7)
    b = pl.ax(11.7, 4.75, 3.7, 3.2)
    b.imshow(D["G"], origin="lower", extent=EXT, cmap="terrain"); b.set_xticks([]); b.set_yticks([])
    for sp in b.spines.values():
        sp.set_color("#B7BCC3")
    y0 = pl.tiles(7.65, 4.2, 7.75, [(f"{s['area_km2']:.0f} km\u00b2", "modelled, in 36 tiles"), (f"{s['points_read_incl_buffers'] / 1e6:.0f} M", "lidar points read"), (f"{s['overture']['power_poles_and_towers']}", "mapped poles and towers; 38 km of mapped line")])
    pl.notes(7.65, y0 - 0.02, 7.75, [
        ("Why this site", "Of the Pierce County towns scanned, this flight is among the densest and the area holds a transmission corridor and a large substation. Density is about 25 returns per m\u00b2, enough to see wires and large equipment but not to resolve small apparatus."),
        ("What is here", "A flat valley floor with farmland, warehouses and housing on its edges, and about 60 m of relief across the terrain panel. Elevations are NAVD88.")], fs=8.4)
    return pl.save("56_whiteriver_town_site_map.png")


# ------------------------------------------------------------------------------------------------ 57 full network
def fig_full(D):
    s, sp, eq = D["stats"], D["sp"], D["eq"]
    conf, add, br = line_layers(D)
    cov, cn = D["spj"]["mapped_line_covered_within_6m"], D["spj"]["connectivity"]
    pl = ProPlate(57, KICK, "The White River utility network", "Wires fitted between supports, wire pieces the spans missed, bridges across gaps, supports and substation structures", src(D), "Dashed and dotted lines are inferred, not observed")
    pl.panel_title(MX, 8.14, "A", "Full network", w=MW)
    a = pl.ax(MX, 1.0, MW, 6.95)
    base(a, D)
    mp = mapped_polyline_pts(D)
    a.scatter(mp[:, 0], mp[:, 1], s=1.2, color=C_MAP, zorder=3, linewidths=0)
    lc(a, conf, C_CONF, 1.2); lc(a, add, C_ADD, 1.1, "--", 6); lc(a, br, C_BR, 1.4, ":", 7)
    if D["uni"] is not None:
        lc(a, [w["xyz"][:, :2] for w in D["uni"]["adds"]], "#2E8B57", 1.0, "-", 6)
        lc(a, [w["xyz"][:, :2] for w in D["uni"]["br"]], "#7B241C", 1.3, ":", 7)
    sup = D["model"]["sup"]; sup = sup[sup.tier == "support"]
    a.scatter(sup.x, sup.y, s=14, marker="s", color=C_SUP, zorder=9, edgecolors="white", linewidths=0.4)
    LS = sp["LS"]
    a.scatter(LS[:, 0], LS[:, 1], s=16, marker="^", color=C_LEARN, zorder=9, edgecolors="white", linewidths=0.3)
    ref = D["model"]["ref"]
    a.scatter(ref[:, 0], ref[:, 1], s=9, facecolors="none", edgecolors=C_POLE, linewidths=0.5, zorder=8)
    for y in eq["yards"]:
        S = np.array(y["structure_xy"]).reshape(-1, 2)
        if len(S):
            a.scatter(S[:, 0], S[:, 1], s=2.5, color="#8E44AD", zorder=8)
    scalebar(a, EXT, 500.0); north_arrow(a, EXT)
    u = D["unij"]
    pl.panel_title(SX, 8.14, "B", "Key numbers", w=SW)
    y0 = pl.tiles(SX, 7.85, SW, [(f"{u['coverage_after_with_bridges']['share_wire_points_within_1m'] * 100:.0f}%" if u else "-", "of detected wire points are in the model (60% before)"), (f"{cov['plus_bridged'] * 100:.0f}%", "of mapped line has a wire within 6 m (33% first rule)"), (f"{cn['confirmed']['components']} to {cn['plus_bridged']['components']}", "separate network pieces after scoring and bridging")])
    y0 = pl.notes(SX, y0 - 0.05, SW, [
        ("What is drawn", f"{len(conf)} observed wire spans, {len(add)} spans the span model added where returns were weak, {len(br)} bridged line ends, {int(sup.shape[0])} supports from wire attachments, {len(LS)} supports from the learned detector, and the structures of two substation yards." + (f" Wire pieces no fitted span explained are drawn where they were observed ({len(D['uni']['adds'])}, green) and joined by sag-curve bridges where they line up ({len(D['uni']['br'])}, dark red); wire ends with no support or bridge fell from {u['free_ends_before_bridges'] * 100:.0f}% to {u['free_ends_after_bridges'] * 100:.0f}%." if u else "")),
        ("What is still missing", f"Recovery of the mapped network stops near {cov['plus_bridged'] * 100:.0f}%. The largest cause is line that has returns on it but no detected support (41% of mapped length); 15% has no returns at all in open ground, so the map may be wrong there, and 7% is under vegetation (plates 58 and 59)."),
        ("How to read it", "The map is incomplete and sometimes wrong, so agreement with it is a floor for precision and recall. Wires are found well where a column of returns exists (41% of mapped poles have one within 3 m); the rest of the network is joined by geometry. Heights of added and bridged wire are modelled sag curves.")], fs=8.4)
    pl.legend(SX, y0 - 0.02, SW, [("line", C_CONF, "wire, observed"), ("dash", C_ADD, "wire added by the span model"), ("dash", C_BR, "line end bridged"), ("line", "#2E8B57", "observed piece no span explains"), ("dash", "#7B241C", "bridge between pieces"),
                                  ("dot", C_MAP, "mapped line"), ("square", C_SUP, "support from wire attachments"), ("tri", C_LEARN, "support from the learned detector"), ("ring", C_POLE, "mapped pole or tower"), ("dot", "#8E44AD", "substation structure")], fs=7.2, row_h=0.19)
    return pl.save("57_whiteriver_town_network_map.png")


# ------------------------------------------------------------------------------------------------ 58/59 diagnosis
def fig_diag_map(D):
    d = D["diag"]; dg = D["dg"]
    pl = ProPlate(58, KICK, "Why mapped power lines have no fitted wire", "Every 2 m along 40 km of mapped line, coloured by what the lidar shows at that point", src(D), "The map itself may be wrong where the lidar shows nothing")
    pl.panel_title(MX, 8.14, "A", "Mapped lines by lidar evidence", w=MW)
    a = pl.ax(MX, 1.0, MW, 6.95)
    base(a, D)
    for cat, col in CAT_COL.items():
        m = d.category == cat
        a.scatter(d.x[m], d.y[m], s=3.2, color=col, zorder=5, linewidths=0)
    sup = D["model"]["sup"]; sup = sup[sup.tier == "support"]
    a.scatter(sup.x, sup.y, s=10, marker="s", color="white", edgecolors=C_SUP, linewidths=0.7, zorder=9)
    scalebar(a, EXT, 500.0); north_arrow(a, EXT)
    sh = dg["share"]
    pl.panel_title(SX, 8.14, "B", "Key numbers", w=SW)
    y0 = pl.tiles(SX, 7.85, SW, [(f"{sh['elevated returns on the line, no wire fitted'] * 100:.0f}%", "of mapped length has returns on it but no wire fitted"), (f"{sh['no elevated returns, open ground'] * 100:.0f}%", "has no elevated returns, in open ground"), (f"{dg['median_distance_to_nearest_detected_support_m']:.0f} m", "median distance from a mapped line point to a detected support")])
    y0 = pl.notes(SX, y0 - 0.05, SW, [
        ("Reading the map", "Blue: a fitted wire lies within 6 m. Orange: returns that look like wire are on the mapped line but no span was fitted, because the supports at its ends were not found. Green and red: no elevated returns at all, under vegetation (photo NDVI of 0.3 or more within 4 m) or on open ground."),
        ("Main finding", f"{dg['km']['power_line']['elevated returns on the line, no wire fitted']:.1f} km of high-voltage line carries returns but no fitted wire. The missing piece is the supports, not the wires: where a wire was fitted a detected support is close, and where it was not the nearest one is tens of metres away."),
        ("Where the lidar shows nothing", f"{dg['km']['power_line']['no elevated returns, open ground']:.1f} km of mapped line has nothing above ground on open ground. That points at a line that is not there or is drawn in the wrong place; a further {sh['no elevated returns, under vegetation'] * 100:.0f}% lies under vegetation."),
        ("Limit", "A return 5 m up within 3 m of the line is not proof of a wire; sparse crowns count. The category boundaries (6 m, 15 m, 5 returns, NDVI 0.3) were fixed before the map's quality was examined.")], fs=8.3)
    pl.legend(SX, y0 - 0.02, SW, [("dot", CAT_COL[c], f"{CAT_SHORT[c]} ({sh[c] * 100:.0f}%)") for c in CAT_COL] + [("square", "#888", "detected support")], fs=7.3, row_h=0.19)
    return pl.save("58_whiteriver_town_line_diagnosis_map.png")


def fig_diag_fig(D):
    d, dg = D["diag"], D["dg"]
    pl = ProPlate(59, KICK, "Mapped lines without a wire: the evidence in numbers", "Length by category, vegetation under the line, and distance to the nearest detected support", src(D), "Samples every 2 m; vegetation from the photo's NDVI within 4 m")
    cats = list(CAT_COL)
    pl.panel_title(0.6, 8.14, "A", "Length by category", w=4.5)
    a = pl.ax(2.3, 4.75, 3.0, 3.1)
    left = np.zeros(len(cats))
    for kd, col in zip(("power_line", "minor_line"), ("#34495E", "#BDC3C7")):
        v = np.array([dg["km"][kd].get(c, 0) for c in cats])
        a.barh(range(len(cats)), v, left=left, color=col, label=kd.replace("_", " "))
        left += v
    a.set_yticks(range(len(cats))); a.set_yticklabels([CAT_SHORT[c] for c in cats], fontsize=7.5); a.invert_yaxis()
    for i, v in enumerate(left):
        a.text(v + 0.2, i, f"{v:.1f} km", va="center", fontsize=7.5, color=MUTED)
    a.set_xlabel("km of mapped line", fontsize=8, color=MUTED); a.tick_params(labelsize=8, colors=MUTED); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=7.5, frameon=False, loc="lower right")
    pl.panel_title(5.6, 8.14, "B", "Vegetation under the line", w=4.2)
    b = pl.ax(5.85, 4.75, 3.8, 3.1)
    b.bar(range(len(cats)), [dg["median_ndvi_by_category"][c] for c in cats], color=[CAT_COL[c] for c in cats]); b.axhline(0.3, color="k", lw=0.8, ls="--")
    b.set_xticks(range(len(cats))); b.set_xticklabels(["fitted", "6-15 m", "returns", "veg.", "open"], fontsize=7.5); b.set_ylabel("median NDVI under the line", fontsize=8, color=MUTED); b.tick_params(labelsize=8, colors=MUTED); b.spines[["top", "right"]].set_visible(False)
    pl.panel_title(0.6, 4.4, "C", "Distance from a mapped line point to the nearest detected support", w=9.0)
    c = pl.ax(1.0, 1.15, 8.6, 2.75)
    for cat in (cats[0], cats[2], cats[4]):
        v = d.dist_support_m[d.category == cat]
        c.hist(v.clip(0, 300), bins=40, histtype="step", color=CAT_COL[cat], lw=1.6, density=True, label=CAT_SHORT[cat])
    c.set_xlabel("distance, m", fontsize=8, color=MUTED); c.set_ylabel("density", fontsize=8, color=MUTED); c.tick_params(labelsize=8, colors=MUTED); c.spines[["top", "right"]].set_visible(False); c.legend(fontsize=8, frameon=False)
    pl.panel_title(10.1, 8.14, "D", "What it says", w=5.3)
    pl.notes(10.1, 7.85, 5.3, [
        ("Supports explain the gap", f"Where a wire was fitted, the median distance to a detected support is {np.median(d.dist_support_m[d.category == cats[0]]):.0f} m; where returns are present but no wire was fitted it is {np.median(d.dist_support_m[d.category == cats[2]]):.0f} m. Spans are only fitted between supports, so a missing support is a missing span."),
        ("Vegetation", "Lines with a fitted wire and lines with unfitted returns both run through vegetated ground (median NDVI about 0.3). Only the lines with no elevated returns split into vegetated and open ground."),
        ("What it does not say", "The map is not ground truth. Red and green lengths are where lidar and map disagree; telling a missing line from a misplaced one needs a survey or an imagery review.")], fs=8.4)
    return pl.save("59_whiteriver_town_line_diagnosis_figure.png")


# ------------------------------------------------------------------------------------------------ 60/61 supports
def fig_sup_map(D):
    sm, su = D["sm"], D["su"]
    LS = D["sp"]["LS"]
    ref = D["model"]["ref"]
    from scipy.spatial import cKDTree
    hit = cKDTree(ref).query(LS)[0] <= 5.0
    geo = D["model"]["sup"]; geo = geo[geo.tier == "support"]
    r5 = [x for x in su["pr_5m"] if abs(x["threshold"] - 0.12) < 0.02][0]
    pl = ProPlate(60, KICK, "Finding poles and towers: geometry against a learned detector", "Mapped poles and towers, supports from wire attachments, and detections of a model scored only on tiles it was not trained on", src(D), "Green triangles lie within 5 m of a mapped pole")
    pl.panel_title(MX, 8.14, "A", "Supports", w=MW)
    a = pl.ax(MX, 1.0, MW, 6.95)
    base(a, D)
    a.scatter(ref[:, 0], ref[:, 1], s=34, facecolors="none", edgecolors=C_POLE, linewidths=0.8, zorder=7)
    a.scatter(geo.x, geo.y, s=18, marker="s", color=C_SUP, zorder=8, edgecolors="white", linewidths=0.4)
    a.scatter(LS[hit, 0], LS[hit, 1], s=26, marker="^", color="#2E8B57", zorder=9, edgecolors="white", linewidths=0.4)
    a.scatter(LS[~hit, 0], LS[~hit, 1], s=26, marker="^", color=C_LEARN, zorder=9, edgecolors="white", linewidths=0.4)
    scalebar(a, EXT, 500.0); north_arrow(a, EXT)
    pl.panel_title(SX, 8.14, "B", "Key numbers", w=SW)
    y0 = pl.tiles(SX, 7.85, SW, [(f"{su['auc_mean']:.2f}", "held-out ranking quality (area under the ROC curve)"), (f"{r5['recall'] * 100:.0f}%", f"of mapped poles found within 5 m ({r5['precision'] * 100:.0f}% precision)"), ("41%", "of mapped poles have any column of returns within 3 m")])
    y0 = pl.notes(SX, y0 - 0.05, SW, [
        ("The detector", f"A gradient-boosted tree model scores {su['candidates']:,} column-shaped candidates from the point mass in height bands, column spread, isolation, nearby wires and tree cover. Mapped poles within 4 m label the {su['positives']} positives. Scores are out of fold: five folds, each holding out whole 500 m tiles. No GPU or deep-learning library is available here, so this is trees, not a neural network."),
        ("Result", f"The model ranks candidates well, but at a score of 0.12 it keeps {r5['detections']} detections with {r5['precision'] * 100:.0f}% within 5 m of a mapped pole, about what the geometric method achieves at the same recall. It is not an improvement; it is kept as an extra source of anchors for span fitting."),
        ("Why recall is capped", "Only 41% of mapped poles have a column of returns within 3 m, and 59% within 5 m, so no detector can pass those ceilings. Orange detections are unconfirmed rather than wrong: the map holds about 7 poles per km of line where a 40 to 60 m spacing would give 17 to 25.")], fs=8.3)
    pl.legend(SX, y0 - 0.02, SW, [("ring", C_POLE, f"mapped pole or tower ({len(ref)})"), ("square", C_SUP, f"geometric support ({len(geo)})"), ("tri", "#2E8B57", f"learned, near a mapped pole ({int(hit.sum())})"), ("tri", C_LEARN, f"learned, no mapped pole near ({int((~hit).sum())})")], fs=7.4)
    return pl.save("60_whiteriver_town_support_map.png")


def fig_sup_fig(D):
    su, sm = D["su"], D["sm"]
    from lidar3d import town_learn as TL
    from scipy.spatial import cKDTree
    ref = D["model"]["ref"]
    C, oof, y = sm["C"], sm["oof"], sm["y"]
    pl = ProPlate(61, KICK, "Pole and tower detection: held-out accuracy", "Precision and recall against the map, the ceiling the data sets, and how well the score separates mapped poles from other columns", src(D), "Scores are out of fold on whole 500 m tiles")
    thr = np.round(np.arange(0.02, 0.9, 0.02), 2)
    pl.panel_title(0.6, 8.14, "A", "Precision against recall", w=4.5)
    a = pl.ax(0.95, 4.75, 4.1, 3.1)
    for r_, col in ((3.0, "#D35400"), (5.0, "#1F4E79")):
        rows_ = TL.pr_curve(C, oof, ref, r_, thr)
        pr = [(rc, p) for t, n, p, rc in rows_ if n > 0]
        a.plot([p[0] for p in pr], [p[1] for p in pr], "-o", ms=2.5, color=col, label=f"learned, {r_:g} m")
        g = su["geometric"]
        for nm, mk in (("geometric_supports", "s"), ("raw_pole_detector", "D")):
            v = g[f"{nm}_{int(r_)}m"]
            a.scatter(v["recall"], v["precision"], marker=mk, s=45, color=col, edgecolors="k", zorder=5, label=f"{'geometric supports' if 'geo' in nm else 'raw pole detector'}, {r_:g} m")
    a.set_xlabel("recall", fontsize=8, color=MUTED); a.set_ylabel("precision against the map", fontsize=8, color=MUTED); a.set_xlim(0, 0.45); a.set_ylim(0, 1.0); a.tick_params(labelsize=8, colors=MUTED); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=6.6, frameon=False, loc="upper right")
    pl.panel_title(5.3, 8.14, "B", "Ranking quality by fold", w=4.3)
    b = pl.ax(5.6, 4.75, 3.9, 3.1)
    b.bar(range(5), su["auc_per_fold"], color="#1F4E79"); b.axhline(0.5, color="k", lw=0.7, ls=":"); b.set_ylim(0.4, 1.0)
    b.set_xticks(range(5)); b.set_xticklabels([f"{i + 1}" for i in range(5)], fontsize=8); b.set_xlabel("fold", fontsize=8, color=MUTED); b.set_ylabel("area under ROC curve", fontsize=8, color=MUTED); b.tick_params(labelsize=8, colors=MUTED); b.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(su["auc_per_fold"]):
        b.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    pl.panel_title(0.6, 4.4, "C", "The ceiling set by the data", w=4.5)
    c = pl.ax(0.95, 1.15, 4.1, 2.75)
    t = cKDTree(C)
    ceil = [float((t.query(ref)[0] <= r_).mean()) for r_ in (3, 5)]
    r3 = [x for x in su["pr_3m"] if abs(x["threshold"] - 0.12) < 0.02][0]["recall"]; r5 = [x for x in su["pr_5m"] if abs(x["threshold"] - 0.12) < 0.02][0]["recall"]
    xs = np.arange(2)
    c.bar(xs - 0.2, ceil, 0.38, color="#BDC3C7", label="mapped poles with a column near"); c.bar(xs + 0.2, [r3, r5], 0.38, color="#D35400", label="found at score 0.12")
    c.set_xticks(xs); c.set_xticklabels(["within 3 m", "within 5 m"], fontsize=8); c.set_ylim(0, 0.8); c.tick_params(labelsize=8, colors=MUTED); c.spines[["top", "right"]].set_visible(False); c.legend(fontsize=7.2, frameon=False, loc="upper left")
    pl.panel_title(5.3, 4.4, "D", "Score distributions", w=4.3)
    d_ = pl.ax(5.6, 1.15, 3.9, 2.75)
    d_.hist(oof[y == 0].clip(0, 1), bins=np.linspace(0, 1, 41), color="#BDC3C7", label="not near a mapped pole", log=True); d_.hist(oof[y == 1], bins=np.linspace(0, 1, 41), color="#D35400", alpha=0.85, label="near a mapped pole", log=True)
    d_.set_xlabel("out-of-fold score", fontsize=8, color=MUTED); d_.set_ylabel("candidates (log)", fontsize=8, color=MUTED); d_.tick_params(labelsize=8, colors=MUTED); d_.spines[["top", "right"]].set_visible(False); d_.legend(fontsize=7.4, frameon=False)
    pl.panel_title(10.1, 8.14, "E", "What the figures show", w=5.3)
    pl.notes(10.1, 7.85, 5.3, [
        ("A learned score that ranks well but does not beat geometry", f"Held-out ranking is good in every fold ({min(su['auc_per_fold']):.2f} to {max(su['auc_per_fold']):.2f}), yet the useful operating points reach only about 20 to 30% precision at 10 to 20% recall, level with the geometric supports (squares) and the raw pole detector (diamonds)."),
        ("The data set the limit", "Only 41% of mapped poles have a column-shaped candidate within 3 m (59% within 5 m), so recall cannot exceed those bars. At 25 returns per m\u00b2 a pole is a handful of returns."),
        ("Precision is a floor", "The map misses poles, so some detections counted as wrong are real. The same is true of every precision figure in this series.")], fs=8.4)
    return pl.save("61_whiteriver_town_support_figure.png")



# ------------------------------------------------------------------------------------------------ 62/63 equipment
def fig_eq_map(D):
    eq = D["eq"]
    big = max(eq["yards"], key=lambda y: y["area_m2"])
    sm_y = min(eq["yards"], key=lambda y: y["area_m2"])
    S = np.array(big["structure_xy"]); M = np.array(big["mapped_xy"])
    vm, pt = big["versus_map"], eq["pole_top"]
    pl = ProPlate(62, KICK, "Equipment: substation structures and pole-top candidates", "Structures found in the dense returns of the two yards against the mapped switches, transformers and portals, and pole-top candidates along the lines", src(D), "The 3DEP classes carry no equipment class")
    cx, cy = big["centre"]
    half = 130
    pl.panel_title(MX, 8.14, "A", "Main substation yard", w=MW)
    a = pl.ax(MX, 1.0, MW, 6.95)
    a.set_facecolor("#F4F4F2")
    a.imshow(D["hs"], origin="lower", extent=(-HALF, HALF, -HALF, HALF), cmap="gray", vmin=0.1, vmax=1.0, alpha=0.8, interpolation="bilinear")
    ext = (cx - half, cx + half, cy - half, cy + half)
    map_frame(a, ext, step=50.0, grid=False)
    a.scatter(S[:, 0], S[:, 1], s=16, color="#27AE60", zorder=6, edgecolors="white", linewidths=0.3)
    a.scatter(M[:, 0], M[:, 1], s=52, marker="x", color="#C0392B", linewidths=1.0, zorder=8)
    scalebar(a, ext, 50.0); north_arrow(a, ext)
    pl.panel_title(SX, 8.14, "B", "Key numbers", w=SW)
    y0 = pl.tiles(SX, 7.85, SW, [(f"{vm['within_3m'] * 100:.0f}%", f"of {vm['mapped_equipment']} mapped yard items have a structure within 3 m (random points: {vm['random_within_3m'] * 100:.0f}%)"), (f"{big['structures']}", f"structures found in the main yard, median {big['median_top_hag_m']:.1f} m tall"), (f"{pt['rate_geometric_supports'] * 100:.0f}%", f"of supports show a pole-top thickening (control columns: {pt['rate_control'] * 100:.0f}%)")])
    y0 = pl.notes(SX, y0 - 0.05, SW, [
        ("Substation yards", f"Within 1.5 m the match is {vm['within_1p5m'] * 100:.0f}% against {vm['random_within_1p5m'] * 100:.0f}% for random points, with a median distance of {vm['median_distance_m']:.1f} m. The position match is real, but one structure can be several devices and many structures are buswork, fences or frames."),
        ("Pole-top equipment", f"A thickening just below the top fires at {pt['candidates_at_geometric_supports']} of {pt['supports_tested']} supports, only a few points more often than at control columns. All 97 mapped switches, transformers and portals are inside yards, so these candidates have no ground truth and name nothing; at 25 returns per m\u00b2 a pole-mounted transformer is a few dozen returns.")], fs=8.3)
    pl.panel_title(SX, y0 - 0.02, "C", "Pole-top candidates", w=3.6)
    b = pl.ax(SX, 1.0, 3.6, max(y0 - 0.45 - 1.0, 1.5))
    pole = pt["candidates"]
    b.imshow(D["hs"], origin="lower", extent=(-HALF, HALF, -HALF, HALF), cmap="gray", vmin=0.1, vmax=1.0, alpha=0.8); map_frame(b, EXT, step=1000, grid=False, labels=False)
    if pole:
        pc = np.array([(p["x"], p["y"]) for p in pole]); b.scatter(pc[:, 0], pc[:, 1], s=26, marker="o", facecolors="none", edgecolors="#D35400", linewidths=1.0, zorder=6)
    b.scatter([cx], [cy], s=120, marker="s", facecolors="none", edgecolors="#C0392B", linewidths=1.2)
    b.set_xticks([]); b.set_yticks([])
    pl.panel_title(SX + 3.85, y0 - 0.02, "D", "Second, small yard", w=3.6)
    c = pl.ax(SX + 3.85, 1.0, 3.6, max(y0 - 0.45 - 1.0, 1.5))
    s2 = np.array(sm_y["structure_xy"]).reshape(-1, 2); cx2, cy2 = sm_y["centre"]
    c.imshow(D["hs"], origin="lower", extent=(-HALF, HALF, -HALF, HALF), cmap="gray", vmin=0.1, vmax=1.0, alpha=0.8)
    map_frame(c, (cx2 - 60, cx2 + 60, cy2 - 60, cy2 + 60), step=50.0, grid=False, labels=False)
    if len(s2):
        c.scatter(s2[:, 0], s2[:, 1], s=14, color="#27AE60", zorder=6)
    c.set_xticks([]); c.set_yticks([])
    pl.legend(MX + 0.2, 1.75, 4.8, [("dot", "#27AE60", f"structure found ({len(S)})"), ("dot", "#C0392B", f"mapped switch, transformer or portal ({len(M)})")], fs=7.3)
    return pl.save("62_whiteriver_town_equipment_map.png")


def fig_eq_fig(D):
    eq = D["eq"]
    big = max(eq["yards"], key=lambda y: y["area_m2"])
    vm, pt = big["versus_map"], eq["pole_top"]
    pl = ProPlate(63, KICK, "Equipment detection: how it checks out", "Substation structures against mapped equipment with a random-position control, and the pole-top rule against its control", src(D), "Yard: 97 mapped items. Pole tops: no ground truth")
    pl.panel_title(0.6, 8.14, "A", "Yard structures against the map", w=4.5)
    a = pl.ax(0.95, 4.75, 4.1, 3.1)
    xs = np.arange(2)
    a.bar(xs - 0.2, [vm["within_1p5m"], vm["within_3m"]], 0.38, color="#27AE60", label="mapped equipment"); a.bar(xs + 0.2, [vm["random_within_1p5m"], vm["random_within_3m"]], 0.38, color="#BDC3C7", label="random points")
    for i, (u, v) in enumerate(zip([vm["within_1p5m"], vm["within_3m"]], [vm["random_within_1p5m"], vm["random_within_3m"]])):
        a.text(i - 0.2, u + 0.01, f"{u * 100:.0f}%", ha="center", fontsize=8); a.text(i + 0.2, v + 0.01, f"{v * 100:.0f}%", ha="center", fontsize=8)
    a.set_xticks(xs); a.set_xticklabels(["within 1.5 m", "within 3 m"], fontsize=8); a.set_ylim(0, 0.9); a.set_ylabel("share with a structure nearby", fontsize=8, color=MUTED); a.tick_params(labelsize=8, colors=MUTED); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=7.5, frameon=False)
    pl.panel_title(5.3, 8.14, "B", "Pole-top rule against a control", w=4.3)
    b = pl.ax(5.6, 4.75, 3.9, 3.1)
    vals = [pt["rate_geometric_supports"], pt["rate_learned_supports"], pt["rate_control"]]
    b.bar(range(3), vals, color=["#1F4E79", "#D35400", "#BDC3C7"])
    for i, v in enumerate(vals):
        b.text(i, v + 0.005, f"{v * 100:.0f}%", ha="center", fontsize=8)
    b.set_xticks(range(3)); b.set_xticklabels([f"geometric\nsupports\n(n={pt['supports_tested']})", f"learned\nsupports\n(n={pt['learned_supports_tested']})", f"control\ncolumns\n(n={pt['control_column_candidates_tested']})"], fontsize=7.5)
    b.set_ylabel("share with a thickening", fontsize=8, color=MUTED); b.tick_params(labelsize=8, colors=MUTED); b.spines[["top", "right"]].set_visible(False)
    pl.panel_title(0.6, 4.4, "C", "Heights of the yard structures", w=9.0)
    c = pl.ax(0.95, 1.15, 8.5, 2.75)
    c.hist(np.array(big["structure_top_hag"]), bins=np.arange(0, 22, 1), color="#8E44AD"); c.set_xlabel("top of the structure above ground, m", fontsize=8, color=MUTED); c.set_ylabel("structures", fontsize=8, color=MUTED); c.tick_params(labelsize=8, colors=MUTED); c.spines[["top", "right"]].set_visible(False)
    pl.panel_title(10.1, 8.14, "D", "What the figures show", w=5.3)
    pl.notes(10.1, 7.85, 5.3, [
        ("Structures sit on mapped equipment", f"The median distance from a mapped item to a structure is {vm['median_distance_m']:.1f} m, and {vm['within_3m'] * 100:.0f}% are within 3 m against {vm['random_within_3m'] * 100:.0f}% for random points. A quarter of the mapped items have no structure within 3 m, and most structures are unmapped, because many are buswork, fences or frames."),
        ("Pole tops", "The thickening rule fires slightly more often on supports than on control columns, and about 85 to 90% of supports show nothing. It marks places to look at; it does not identify transformers."),
        ("Heights", "Yard structures cluster between 3 and 10 m, with a tail of gantries up to 20 m.")], fs=8.4)
    return pl.save("63_whiteriver_town_equipment_figure.png")


# ------------------------------------------------------------------------------------------------ 64/65 spans
def fig_span_map(D):
    spj = D["spj"]
    conf, add, br = line_layers(D)
    g, ad, bd = spj["geometric_rule"], spj["added_by_model"], spj["bridged"]
    cov, cn = spj["mapped_line_covered_within_6m"], spj["connectivity"]
    pl = ProPlate(64, KICK, "Joining the network: observed, added and bridged spans", "Spans between supports scored on held-out tiles, then line ends that face each other bridged where the line continues straight", src(D), "Dashed orange: weakly observed span accepted by the model")
    pl.panel_title(MX, 8.14, "A", "Spans", w=MW)
    a = pl.ax(MX, 1.0, MW, 6.95)
    base(a, D)
    mp = mapped_polyline_pts(D)
    a.scatter(mp[:, 0], mp[:, 1], s=1.2, color=C_MAP, zorder=3, linewidths=0)
    lc(a, conf, C_CONF, 1.3); lc(a, add, C_ADD, 1.3, "--", 6); lc(a, br, C_BR, 1.8, ":", 7)
    A = D["sp"]["A"]
    a.scatter(A[:, 0], A[:, 1], s=4, color="white", edgecolors="k", linewidths=0.3, zorder=9)
    scalebar(a, EXT, 500.0); north_arrow(a, EXT)
    pl.panel_title(SX, 8.14, "B", "Key numbers", w=SW)
    y0 = pl.tiles(SX, 7.85, SW, [(f"+{ad['length_m'] / 1000:.1f} km", f"of wire added by the span model, {ad['precision_by_length'] * 100:.0f}% within 6 m of a mapped line"), (f"{cov['confirmed'] * 100:.0f}% to {cov['plus_bridged'] * 100:.0f}%", "of mapped line with a wire within 6 m"), (f"{cn['confirmed']['components']} to {cn['plus_bridged']['components']}", "separate network pieces")])
    y0 = pl.notes(SX, y0 - 0.05, SW, [
        ("Candidate spans", f"{spj['anchors']} anchors (geometric supports and wire ends, plus {spj['anchors_added_from_learned_supports']} from the learned detector) give {spj['candidate_pairs']:,} neighbour pairs, of which {spj['pair_table']:,} have a wire fit. A tree model scores each from its length, returns, sag, end heights, the support score at each end and how straight it continues a confirmed neighbour."),
        ("What the model added", f"At score {spj['threshold']:.2f}, the lowest threshold whose held-out precision stays within 2 points of the first rule's {g['precision_by_length'] * 100:.0f}%, it accepts {spj['scored_spans']['spans']} spans, {ad['spans']} of them new. These are partly observed (from 20% of the span up) and partly modelled."),
        ("Bridging", f"{bd['spans']} line ends facing each other within {bd['gap_range_m'][0]} to {bd['gap_range_m'][1]} m and 15 degrees were joined, {bd['length_m']:.0f} m in all, {bd['fraction_of_length_within_6m_of_mapped_line'] * 100:.0f}% of it within 6 m of a mapped line. Bridging is conservative: most gaps have no facing end, because no support was found on the far side.")], fs=8.3)
    pl.legend(SX, y0 - 0.02, SW, [("line", C_CONF, f"observed span ({len(conf)})"), ("dash", C_ADD, f"added by the span model ({len(add)})"), ("dash", C_BR, f"bridged line end ({len(br)})"), ("dot", C_MAP, "mapped line"), ("ring", "k", "anchor")], fs=7.4)
    return pl.save("64_whiteriver_town_span_map.png")


def fig_span_fig(D):
    spj, st = D["spj"], D["stats"]
    pl = ProPlate(65, KICK, "Spans and gap filling: results", "How much of the mapped network is recovered, how connected the model becomes, how precise the added spans are, and how accurate a filled wire is", src(D), "Against Overture lines: lower bounds")
    cov = spj["mapped_line_covered_within_6m"]
    v = [cov["confirmed"], cov["plus_scored_spans"], cov["plus_bridged"]]
    pl.panel_title(0.6, 8.14, "A", "Share of the mapped network recovered", w=4.5)
    a = pl.ax(0.95, 4.75, 4.1, 3.1)
    a.bar(range(3), [x * 100 for x in v], color=[C_CONF, C_ADD, C_BR])
    for i, x in enumerate(v):
        a.text(i, x * 100 + 0.8, f"{x * 100:.1f}%", ha="center", fontsize=8.5)
    a.set_xticks(range(3)); a.set_xticklabels(["first rule", "+ model spans", "+ bridges"], fontsize=8); a.set_ylabel("mapped line with a wire within 6 m, %", fontsize=7.5, color=MUTED); a.set_ylim(0, 55); a.tick_params(labelsize=8, colors=MUTED); a.spines[["top", "right"]].set_visible(False)
    pl.panel_title(5.3, 8.14, "B", "Connectivity of the network", w=4.3)
    b = pl.ax(5.6, 4.75, 3.7, 3.1)
    cn = spj["connectivity"]
    names = ["confirmed", "plus_scored_spans", "plus_bridged"]
    b.bar(np.arange(3), [cn[n]["components"] for n in names], 0.5, color="#34495E")
    for i, n in enumerate(names):
        b.text(i, cn[n]["components"] + 0.8, str(cn[n]["components"]), ha="center", fontsize=8.5)
    b.set_ylabel("separate pieces", fontsize=8, color=MUTED); b.set_xticks(range(3)); b.set_xticklabels(["first rule", "+ model", "+ bridges"], fontsize=8)
    b2 = b.twinx(); b2.plot(range(3), [cn[n]["largest_component_share_of_nodes"] * 100 for n in names], "-o", color="#D35400"); b2.set_ylim(0, 100); b2.set_ylabel("largest piece, % of supports", fontsize=8, color="#D35400")
    b.tick_params(labelsize=8, colors=MUTED); b2.tick_params(labelsize=8, colors="#D35400"); b.spines[["top"]].set_visible(False); b2.spines[["top"]].set_visible(False)
    pl.panel_title(0.6, 4.4, "C", "Precision against the map, by length", w=4.5)
    c = pl.ax(0.95, 1.15, 4.1, 2.75)
    g, ad, bd = spj["geometric_rule"], spj["added_by_model"], spj["bridged"]
    pv = [g["precision_by_length"], ad["precision_by_length"], bd["fraction_of_length_within_6m_of_mapped_line"]]
    ln = [g["length_m"] / 1000, ad["length_m"] / 1000, bd["length_m"] / 1000]
    c.bar(range(3), [p * 100 for p in pv], color=[C_CONF, C_ADD, C_BR])
    for i, (p, l) in enumerate(zip(pv, ln)):
        c.text(i, p * 100 + 1.5, f"{p * 100:.0f}%\n({l:.1f} km)", ha="center", fontsize=8)
    c.set_xticks(range(3)); c.set_xticklabels(["first rule", "added", "bridged"], fontsize=8); c.set_ylim(0, 118); c.set_ylabel("length within 6 m of a mapped line, %", fontsize=7.5, color=MUTED); c.tick_params(labelsize=8, colors=MUTED); c.spines[["top", "right"]].set_visible(False)
    pl.panel_title(5.3, 4.4, "D", "Filling a gap in a wire", w=4.3)
    d_ = pl.ax(5.6, 1.15, 3.7, 2.75)
    h = st["interpolation_holdout"]
    x = np.arange(2)
    d_.bar(x - 0.2, [h["20"]["rms_parabola_m"] * 100, h["40"]["rms_parabola_m"] * 100], 0.38, color="#1F4E79", label="fitted sag curve"); d_.bar(x + 0.2, [h["20"]["rms_linear_m"] * 100, h["40"]["rms_linear_m"] * 100], 0.38, color="#BDC3C7", label="straight line")
    d_.set_xticks(x); d_.set_xticklabels(["20 m gap", "40 m gap"], fontsize=8); d_.set_ylabel("rms error, cm", fontsize=8, color=MUTED); d_.tick_params(labelsize=8, colors=MUTED); d_.spines[["top", "right"]].set_visible(False); d_.legend(fontsize=7.5, frameon=False)
    pl.panel_title(10.1, 8.14, "E", "What the figures show", w=5.3)
    pl.notes(10.1, 7.85, 5.3, [
        ("Recovery and connection", f"Scoring spans lifts the share of mapped line with a wire from {v[0] * 100:.0f}% to {v[1] * 100:.0f}%, bridging adds {(v[2] - v[1]) * 100:.1f} points, and the network falls from {cn['confirmed']['components']} pieces to {cn['plus_bridged']['components']}, the largest holding {cn['plus_bridged']['largest_component_share_of_nodes'] * 100:.0f}% of the supports."),
        ("Precision", f"The first rule's spans are {g['precision_by_length'] * 100:.0f}% within 6 m of a mapped line by length. Spans the model added are {ad['precision_by_length'] * 100:.0f}%, and bridges {bd['fraction_of_length_within_6m_of_mapped_line'] * 100:.0f}%. The map is incomplete, so these are floors."),
        ("Filling a gap", f"Hiding a 20 to 40 m stretch of a well-observed wire ({h['20']['n']} tests per size) and refitting gives {h['20']['rms_parabola_m'] * 100:.0f} to {h['40']['rms_parabola_m'] * 100:.0f} cm error with the sag curve against {h['20']['rms_linear_m'] * 100:.0f} to {h['40']['rms_linear_m'] * 100:.0f} cm for a straight line. Hidden stretches were inliers of the fit, so this is optimistic for noisy wires.")], fs=8.4)
    return pl.save("65_whiteriver_town_span_figure.png")
