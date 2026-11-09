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
from .plates import INK, MUTED, Plate, map_frame, north_arrow, scalebar

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
    D["stats"], D["dg"], D["su"], D["spj"], D["eq"] = J("stats.json"), J("extras_diagnosis.json"), J("extras_support.json"), J("extras_spans.json"), J("extras_equipment.json")
    G = D["model"]["G"]
    D["hs"] = hillshade(np.nan_to_num(G, nan=float(np.nanmedian(G))), 1.0, 315, 40)
    D["G"] = G
    return D


def src(D):
    s = D["stats"]
    return (f"Lidar: USGS 3DEP {s['project']}, flown {s['collected'][0]} (about 25 returns per m2); 36 tiles read with a 40 m buffer. Power features: Overture Maps (OpenStreetMap-derived), used only to check and to train the support and span scores. "
            "Imagery: NAIP 2019.")


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
    pl = Plate("Where the White River town model is", "A 3 km by 3 km square of the White River valley between Sumner, Auburn and Bonney Lake in Pierce County, Washington, with the White River substation at its centre", src(D), "Dashed grey: county lines. Red: freeways. Blue: water")
    a = pl.ax(0.6, 1.1, 7.0, 7.2)
    sitemap.draw_regional(a, L, bbox, (lon, lat), (dlon, dlat), label_places=("Seattle", "Tacoma", "Auburn", "Kent", "Puyallup", "Enumclaw", "Federal Way"))
    a.annotate("model area", xy=(lon, lat), xytext=(lon + 0.1, lat - 0.2), fontsize=9, fontweight="bold", color="#E74C3C", ha="center", arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2))
    pl.label(0.6, 8.35, "Region")
    b = pl.ax(8.2, 4.1, 3.8, 3.8)
    b.imshow(np.moveaxis(naip_img[:3], 0, -1), extent=(-HALF, HALF, -HALF, HALF)); b.set_xticks([]); b.set_yticks([])
    pl.label(8.2, 8.0, "Aerial imagery, 3 km")
    c = pl.ax(12.2, 4.1, 3.2, 3.8)
    c.imshow(D["G"], origin="lower", extent=EXT, cmap="terrain"); c.set_xticks([]); c.set_yticks([])
    pl.label(12.2, 8.0, "Bare-earth terrain")
    rows(pl, 8.2, 3.8, [("Why here", f"Of the Pierce County towns scanned, this flight is the densest and the area has a real transmission corridor (mapped: {s['overture']['power_poles_and_towers']} poles and towers, 38 km of high-voltage line) and a large substation. Density is about 25 returns per m2 over {s['area_km2']:.0f} km2 and {s['points_read_incl_buffers'] / 1e6:.0f} million points were read."),
                        ("What is here", "Flat valley floor with farmland, warehouses and housing; the terrain panel spans about 60 m of relief. Vertical datum NAVD88."),
                        ("How it was built", "The square was cut into 36 tiles of 500 m, read with a 40 m buffer, and wires, poles and elevated returns were extracted per tile and joined across the tile edges.")], 95)
    return pl.save("56_whiteriver_town_site_map.png")


# ------------------------------------------------------------------------------------------------ 57 full network
def fig_full(D):
    s, sp, eq = D["stats"], D["sp"], D["eq"]
    conf, add, br = line_layers(D)
    pl = Plate("The White River utility network, 3 km by 3 km", "Wires fitted between supports from the lidar, wires the model added where returns are weak, line ends bridged across gaps, supports and substation equipment",
               src(D), "Wire heights are fitted from lidar returns; dashed lines are inferred, not observed")
    a = pl.ax(0.6, 1.1, 7.2, 7.2)
    base(a, D)
    mp = mapped_polyline_pts(D)
    a.scatter(mp[:, 0], mp[:, 1], s=1.2, color=C_MAP, zorder=3, linewidths=0)
    lc(a, conf, C_CONF, 1.2); lc(a, add, C_ADD, 1.1, "--", 6); lc(a, br, C_BR, 1.4, ":", 7)
    sup = D["model"]["sup"]; sup = sup[sup.tier == "support"]
    a.scatter(sup.x, sup.y, s=14, marker="s", color=C_SUP, zorder=9, edgecolors="white", linewidths=0.4)
    LS, LSs = sp["LS"], sp["LSs"]
    a.scatter(LS[:, 0], LS[:, 1], s=16, marker="^", color=C_LEARN, zorder=9, edgecolors="white", linewidths=0.3)
    ref = D["model"]["ref"]
    a.scatter(ref[:, 0], ref[:, 1], s=9, facecolors="none", edgecolors=C_POLE, linewidths=0.5, zorder=8)
    for y in eq["yards"]:
        S = np.array(y["structure_xy"]).reshape(-1, 2)
        if len(S):
            a.scatter(S[:, 0], S[:, 1], s=2.5, color="#8E44AD", zorder=8)
    a.set_title("")
    scalebar(a, EXT, 500.0); north_arrow(a, EXT)
    legend(a, [Line2D([], [], color=C_CONF, lw=1.4, label="wire, observed returns"), Line2D([], [], color=C_ADD, lw=1.4, ls="--", label="wire added by the span model"), Line2D([], [], color=C_BR, lw=1.6, ls=":", label="line end bridged across a gap"),
               Line2D([], [], color=C_MAP, marker=".", ls="", label="mapped line (Overture)"), pts_legend(C_SUP, "support from wire attachments", "s", 5), pts_legend(C_LEARN, "support from the learned detector", "^", 5),
               pts_legend(C_POLE, "mapped pole or tower", "o", 5, "none"), pts_legend("#8E44AD", "substation structure", ".", 6)], "lower right", 7)
    pl.label(0.6, 8.35, "Full network")
    cov = D["spj"]["mapped_line_covered_within_6m"]
    cn = D["spj"]["connectivity"]
    rows(pl, 8.3, 8.2, [
        ("What is on the map", f"{len(conf)} observed wire spans, {len(add)} spans the model added, {len(br)} bridged line ends, {int(sup.shape[0])} supports from wire attachments, {len(LS)} learned supports and the structures of two substation yards."),
        ("How much of the mapped network is recovered", f"{cov['confirmed'] * 100:.0f}% of the mapped line length has a fitted wire within 6 m with the first rule, {cov['plus_scored_spans'] * 100:.0f}% after the model-added spans and {cov['plus_bridged'] * 100:.0f}% after bridging. The rest is mostly line with returns on it but no detected support (41%), line with no returns in open ground (15%, so the map may be wrong there) and line under vegetation (7%)."),
        ("Connectivity", f"Supports linked by spans form {cn['confirmed']['components']} separate pieces at first, {cn['plus_scored_spans']['components']} with the model's spans and {cn['plus_bridged']['components']} after bridging; the largest piece holds {cn['confirmed']['largest_component_share_of_nodes'] * 100:.0f}%, then {cn['plus_scored_spans']['largest_component_share_of_nodes'] * 100:.0f}%, then {cn['plus_bridged']['largest_component_share_of_nodes'] * 100:.0f}% of the supports."),
        ("Read with care", "The map is incomplete and sometimes wrong, so every agreement figure with it is a floor for precision and a floor for recall. Poles are found well when a column of returns exists (41% of mapped poles have one within 3 m); the rest of the network is joined by geometry. Heights of added and bridged wires are modelled sag curves.")], 80)
    return pl.save("57_whiteriver_town_network_map.png")


# ------------------------------------------------------------------------------------------------ 58/59 diagnosis
def fig_diag_map(D):
    d = D["diag"]; dg = D["dg"]
    pl = Plate("Why mapped power lines have no fitted wire", "Every 2 m along the 40 km of mapped line, coloured by what the lidar shows there", src(D), "Colour = category of the lidar evidence at that point of the mapped line; the map itself may be wrong")
    a = pl.ax(0.6, 1.1, 7.2, 7.2)
    base(a, D)
    for cat, col in CAT_COL.items():
        m = d.category == cat
        a.scatter(d.x[m], d.y[m], s=3.2, color=col, zorder=5, linewidths=0)
    sup = D["model"]["sup"]; sup = sup[sup.tier == "support"]
    a.scatter(sup.x, sup.y, s=10, marker="s", color="white", edgecolors=C_SUP, linewidths=0.7, zorder=9)
    scalebar(a, EXT, 500.0); north_arrow(a, EXT)
    legend(a, [pts_legend(CAT_COL[c], f"{CAT_SHORT[c]}  ({dg['share'][c] * 100:.0f}%)", ".", 9) for c in CAT_COL] + [pts_legend(C_SUP, "detected support", "s", 5, "white")], "lower right", 7.5)
    pl.label(0.6, 8.35, "Mapped lines by lidar evidence")
    rows(pl, 8.3, 8.2, [
        ("Reading the map", "Blue: a fitted wire lies within 6 m. Orange: returns that look like wire or crown are on the mapped line but no span was fitted, because the supports at its ends were not found. Green and red: no elevated returns at all, under vegetation (NAIP NDVI of 0.3 or more within 4 m) or on open ground."),
        ("Main finding", f"{dg['share']['elevated returns on the line, no wire fitted'] * 100:.0f}% of the mapped length ({dg['km']['power_line']['elevated returns on the line, no wire fitted']:.1f} km of high-voltage line) has elevated returns on it with no wire fitted. The nearest detected support is a median {dg['median_distance_to_nearest_detected_support_m']:.0f} m from a mapped line point, so the missing piece is the supports, not the wires."),
        ("Where the lidar shows nothing", f"{dg['share']['no elevated returns, open ground'] * 100:.0f}% of the mapped length ({dg['km']['power_line']['no elevated returns, open ground']:.1f} km) has no elevated returns on open ground, which points at a mapped line that is not there or is drawn in the wrong place; {dg['share']['no elevated returns, under vegetation'] * 100:.0f}% is under vegetation."),
        ("Limit", "A return within 3 m at 5 m height is not proof of a wire; sparse tree crowns count. Category boundaries (6, 15 m, 5 returns, NDVI 0.3) were fixed before looking at the map's quality.")], 80)
    return pl.save("58_whiteriver_town_line_diagnosis_map.png")


def fig_diag_fig(D):
    d, dg = D["diag"], D["dg"]
    pl = Plate("Mapped lines without a wire: the evidence in numbers", "Length by category, vegetation under the line, and the distance to the nearest detected support", src(D), "Spacing between samples 2 m; vegetation = NAIP 2019 NDVI within 4 m")
    cats = list(CAT_COL)
    a = pl.ax(1.0, 4.9, 6.6, 3.2)
    kinds = ("power_line", "minor_line")
    left = np.zeros(len(cats))
    for kd, col in zip(kinds, ("#34495E", "#BDC3C7")):
        v = np.array([dg["km"][kd].get(c, 0) for c in cats])
        a.barh(range(len(cats)), v, left=left, color=col, label=kd.replace("_", " "))
        left += v
    a.set_yticks(range(len(cats))); a.set_yticklabels([CAT_SHORT[c] for c in cats], fontsize=8); a.invert_yaxis()
    for i, v in enumerate(left):
        a.text(v + 0.2, i, f"{v:.1f} km", va="center", fontsize=8, color=MUTED)
    a.set_xlabel("km of mapped line", fontsize=8, color=MUTED); a.tick_params(labelsize=8); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=8, frameon=False, loc="lower right")
    pl.label(1.0, 8.35, "Length by category")
    b = pl.ax(9.0, 4.9, 6.0, 3.2)
    nd = [dg["median_ndvi_by_category"][c] for c in cats]
    b.bar(range(len(cats)), nd, color=[CAT_COL[c] for c in cats]); b.axhline(0.3, color="k", lw=0.8, ls="--"); b.text(len(cats) - 0.5, 0.31, "vegetation threshold 0.3", fontsize=7.5, ha="right", va="bottom")
    b.set_xticks(range(len(cats))); b.set_xticklabels([CAT_SHORT[c].replace(", ", ",\n").replace(" no wire fitted", "\nno wire fitted") for c in cats], fontsize=7); b.set_ylabel("median NDVI under the line", fontsize=8, color=MUTED); b.tick_params(labelsize=8); b.spines[["top", "right"]].set_visible(False)
    pl.label(9.0, 8.35, "Vegetation under the line")
    c = pl.ax(1.0, 1.3, 6.6, 2.7)
    for cat in (cats[0], cats[2], cats[4]):
        v = d.dist_support_m[d.category == cat]
        c.hist(v.clip(0, 300), bins=40, histtype="step", color=CAT_COL[cat], lw=1.6, density=True, label=CAT_SHORT[cat])
    c.set_xlabel("distance from the mapped line point to the nearest detected support, m", fontsize=8, color=MUTED); c.set_ylabel("density", fontsize=8, color=MUTED); c.tick_params(labelsize=8); c.spines[["top", "right"]].set_visible(False); c.legend(fontsize=8, frameon=False)
    pl.label(1.0, 4.35, "Distance to the nearest support")
    rows(pl, 9.0, 3.9, [("What it says", f"Where a wire was fitted, a detected support is close; where returns are present but no wire was fitted the supports are tens of metres away. Median distance to a support is {np.median(d.dist_support_m[d.category == cats[0]]):.0f} m for the fitted category and {np.median(d.dist_support_m[d.category == cats[2]]):.0f} m for the unfitted one."),
                        ("What it does not say", "The map is not ground truth. The red and green lengths are where the lidar and the map disagree; this diagnosis cannot tell a missing line from a misplaced one without a survey or imagery review.")], 70)
    return pl.save("59_whiteriver_town_line_diagnosis_figure.png")


# ------------------------------------------------------------------------------------------------ 60/61 supports
def fig_sup_map(D):
    sm, su = D["sm"], D["su"]
    C, oof = sm["C"], sm["oof"]
    LS, LSs = D["sp"]["LS"], D["sp"]["LSs"]
    ref = D["model"]["ref"]
    from scipy.spatial import cKDTree
    hit = cKDTree(ref).query(LS)[0] <= 5.0
    geo = D["model"]["sup"]; geo = geo[geo.tier == "support"]
    pl = Plate("Finding poles and towers: geometry against a learned detector", "Mapped poles and towers, supports from wire attachments and detections of a model scored only on tiles it was not trained on", src(D), "Green: detection within 5 m of a mapped pole. Orange: no mapped pole nearby (a true unmapped pole or a false detection)")
    a = pl.ax(0.6, 1.1, 7.2, 7.2)
    base(a, D)
    a.scatter(ref[:, 0], ref[:, 1], s=34, facecolors="none", edgecolors=C_POLE, linewidths=0.8, zorder=7)
    a.scatter(geo.x, geo.y, s=18, marker="s", color=C_SUP, zorder=8, edgecolors="white", linewidths=0.4)
    a.scatter(LS[hit, 0], LS[hit, 1], s=26, marker="^", color="#2E8B57", zorder=9, edgecolors="white", linewidths=0.4)
    a.scatter(LS[~hit, 0], LS[~hit, 1], s=26, marker="^", color=C_LEARN, zorder=9, edgecolors="white", linewidths=0.4)
    scalebar(a, EXT, 500.0); north_arrow(a, EXT)
    legend(a, [pts_legend(C_POLE, f"mapped pole or tower ({len(ref)})", "o", 7, "none"), pts_legend(C_SUP, f"geometric support ({len(geo)})", "s", 5), pts_legend("#2E8B57", f"learned, near a mapped pole ({int(hit.sum())})", "^", 6), pts_legend(C_LEARN, f"learned, no mapped pole near ({int((~hit).sum())})", "^", 6)], "lower right", 7.5)
    pl.label(0.6, 8.35, "Supports")
    r5 = [x for x in su["pr_5m"] if abs(x["threshold"] - 0.12) < 0.02][0]
    rows(pl, 8.3, 8.2, [
        ("The detector", f"A gradient-boosted tree model scores {su['candidates']:,} column-shaped candidate locations from the point mass in height bands, column spread, isolation, wires nearby and tree cover. Mapped poles and towers within 4 m label the positives ({su['positives']}). Scores are out of fold: five folds, each holding out whole 500 m tiles. (No GPU or deep-learning library is installed here, so this is trees, not a neural network.)"),
        ("Held-out result", f"Ranking quality is good (area under the ROC curve {su['auc_mean']:.2f}, folds {min(su['auc_per_fold']):.2f} to {max(su['auc_per_fold']):.2f}), but the useful operating point is modest: at a score of 0.12 it keeps {r5['detections']} detections with {r5['precision'] * 100:.0f}% within 5 m of a mapped pole and {r5['recall'] * 100:.0f}% of mapped poles found."),
        ("Why recall is capped", "Only 41% of mapped poles have any column of returns within 3 m and 59% within 5 m at this density, so no detector can pass those ceilings. Orange detections are unconfirmed, not necessarily wrong: the map holds about 7 poles per km of line, fewer than spacing of 40 to 60 m would give."),
        ("Verdict", "The learned detector is about as good as the geometric one at the same recall, which is not an improvement worth a claim. It is kept as an extra source of anchors for span fitting.")], 80)
    return pl.save("60_whiteriver_town_support_map.png")


def fig_sup_fig(D):
    su, sm = D["su"], D["sm"]
    from lidar3d import town_learn as TL
    ref = D["model"]["ref"]
    C, oof, y = sm["C"], sm["oof"], sm["y"]
    pl = Plate("Pole and tower detection: held-out accuracy", "Precision and recall against the map, the ceiling set by the data, and how well the score separates mapped poles from other columns", src(D), "All learned scores are out of fold on whole 500 m tiles; precision against the map is a lower bound because the map misses poles")
    thr = np.round(np.arange(0.02, 0.9, 0.02), 2)
    a = pl.ax(1.0, 4.9, 6.4, 3.2)
    for r_, col in ((3.0, "#D35400"), (5.0, "#1F4E79")):
        rows_ = TL.pr_curve(C, oof, ref, r_, thr)
        pr = [(rc, p) for t, n, p, rc in rows_ if n > 0]
        a.plot([p[0] for p in pr], [p[1] for p in pr], "-o", ms=2.5, color=col, label=f"learned, match within {r_:g} m")
        g = su["geometric"]
        for nm, mk in (("geometric_supports", "s"), ("raw_pole_detector", "D")):
            v = g[f"{nm}_{int(r_)}m"]
            a.scatter(v["recall"], v["precision"], marker=mk, s=55, color=col, edgecolors="k", zorder=5, label=f"{'geometric supports' if 'geo' in nm else 'raw pole detector'}, {r_:g} m")
    a.set_xlabel("recall (share of mapped poles found)", fontsize=8, color=MUTED); a.set_ylabel("precision against the map", fontsize=8, color=MUTED); a.set_xlim(0, 0.45); a.set_ylim(0, 1.0); a.tick_params(labelsize=8); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=7, frameon=False, loc="upper right")
    pl.label(1.0, 8.35, "Precision against recall")
    b = pl.ax(9.0, 4.9, 6.0, 3.2)
    b.bar(range(5), su["auc_per_fold"], color="#1F4E79"); b.axhline(0.5, color="k", lw=0.7, ls=":"); b.set_ylim(0.4, 1.0)
    b.set_xticks(range(5)); b.set_xticklabels([f"fold {i + 1}" for i in range(5)], fontsize=8); b.set_ylabel("area under ROC curve", fontsize=8, color=MUTED); b.tick_params(labelsize=8); b.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(su["auc_per_fold"]):
        b.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    pl.label(9.0, 8.35, "Held-out ranking quality")
    c = pl.ax(1.0, 1.3, 6.4, 2.7)
    from scipy.spatial import cKDTree
    t = cKDTree(C)
    ceil = [float((t.query(ref)[0] <= r_).mean()) for r_ in (3, 5)]
    r3 = [x for x in su["pr_3m"] if abs(x["threshold"] - 0.12) < 0.02][0]["recall"]; r5 = [x for x in su["pr_5m"] if abs(x["threshold"] - 0.12) < 0.02][0]["recall"]
    xs = np.arange(2)
    c.bar(xs - 0.2, ceil, 0.38, color="#BDC3C7", label="mapped poles with a column-shaped candidate near"); c.bar(xs + 0.2, [r3, r5], 0.38, color="#D35400", label="found by the learned detector at score 0.12")
    c.set_xticks(xs); c.set_xticklabels(["within 3 m", "within 5 m"], fontsize=8); c.set_ylim(0, 0.8); c.tick_params(labelsize=8); c.spines[["top", "right"]].set_visible(False); c.legend(fontsize=7.5, frameon=False, loc="upper left")
    pl.label(1.0, 4.35, "The ceiling set by the data")
    d_ = pl.ax(9.0, 1.3, 6.0, 2.7)
    d_.hist(oof[y == 0].clip(0, 1), bins=np.linspace(0, 1, 41), color="#BDC3C7", label="not near a mapped pole", log=True); d_.hist(oof[y == 1], bins=np.linspace(0, 1, 41), color="#D35400", alpha=0.85, label="near a mapped pole", log=True)
    d_.set_xlabel("out-of-fold score", fontsize=8, color=MUTED); d_.set_ylabel("candidates (log)", fontsize=8, color=MUTED); d_.tick_params(labelsize=8); d_.spines[["top", "right"]].set_visible(False); d_.legend(fontsize=8, frameon=False)
    pl.label(9.0, 4.35, "Score distributions")
    return pl.save("61_whiteriver_town_support_figure.png")


# ------------------------------------------------------------------------------------------------ 62/63 equipment
def fig_eq_map(D):
    eq = D["eq"]
    big = max(eq["yards"], key=lambda y: y["area_m2"])
    sm_y = min(eq["yards"], key=lambda y: y["area_m2"])
    S = np.array(big["structure_xy"]); M = np.array(big["mapped_xy"]); H = np.array(big["structure_top_hag"])
    from scipy.spatial import cKDTree
    d_m = cKDTree(S).query(M)[0] if len(S) and len(M) else np.array([])
    pl = Plate("Equipment: substation yard structures and pole-top candidates", "Structures found in the dense returns of the two substation yards, the mapped switches, transformers and portals, and pole-top equipment candidates along the lines", src(D), "Yard structures need no class codes; the 3DEP classes carry no equipment class")
    cx, cy = big["centre"]
    half = 130
    a = pl.ax(0.6, 1.1, 7.2, 7.2)
    a.set_facecolor("#F4F4F2")
    a.imshow(D["hs"], origin="lower", extent=(-HALF, HALF, -HALF, HALF), cmap="gray", vmin=0.1, vmax=1.0, alpha=0.8, interpolation="bilinear")
    ext = (cx - half, cx + half, cy - half, cy + half)
    map_frame(a, ext, step=50.0, grid=False)
    a.scatter(S[:, 0], S[:, 1], s=16, color="#27AE60", zorder=6, edgecolors="white", linewidths=0.3)
    a.scatter(M[:, 0], M[:, 1], s=52, marker="x", color="#C0392B", linewidths=1.0, zorder=8)
    scalebar(a, ext, 50.0); north_arrow(a, ext)
    legend(a, [pts_legend("#27AE60", f"structure found ({len(S)})", "o", 6), Line2D([], [], marker="x", ls="", color="#C0392B", ms=7, label=f"mapped switch, transformer or portal ({len(M)})")], "lower right", 8)
    pl.label(0.6, 8.35, "Main substation yard")
    b = pl.ax(8.2, 4.1, 3.4, 3.4)
    pole = D["eq"]["pole_top"]["candidates"]
    b.imshow(D["hs"], origin="lower", extent=(-HALF, HALF, -HALF, HALF), cmap="gray", vmin=0.1, vmax=1.0, alpha=0.8); map_frame(b, EXT, step=1000, grid=False, labels=False)
    if pole:
        pc = np.array([(p["x"], p["y"]) for p in pole]); b.scatter(pc[:, 0], pc[:, 1], s=26, marker="o", facecolors="none", edgecolors="#D35400", linewidths=1.0, zorder=6)
    b.scatter([cx], [cy], s=120, marker="s", facecolors="none", edgecolors="#C0392B", linewidths=1.2)
    b.set_xticks([]); b.set_yticks([])
    pl.label(8.2, 7.8, "Pole-top candidates, whole area")
    c = pl.ax(12.0, 4.1, 3.4, 3.4)
    s2 = np.array(sm_y["structure_xy"]).reshape(-1, 2); cx2, cy2 = sm_y["centre"]
    c.imshow(D["hs"], origin="lower", extent=(-HALF, HALF, -HALF, HALF), cmap="gray", vmin=0.1, vmax=1.0, alpha=0.8)
    ext2 = (cx2 - 60, cx2 + 60, cy2 - 60, cy2 + 60); map_frame(c, ext2, step=50.0, grid=False)
    if len(s2): c.scatter(s2[:, 0], s2[:, 1], s=14, color="#27AE60", zorder=6)
    pl.label(12.0, 7.8, "Second, small yard")
    vm = big["versus_map"]
    rows(pl, 8.2, 3.7, [
        ("Substation yard", f"{len(S)} compact structures with a median top {big['median_top_hag_m']:.1f} m above ground. {vm['within_3m'] * 100:.0f}% of the {vm['mapped_equipment']} mapped switches, transformers and portals have a structure within 3 m, against {vm['random_within_3m'] * 100:.0f}% for random points in the yard; within 1.5 m, {vm['within_1p5m'] * 100:.0f}% against {vm['random_within_1p5m'] * 100:.0f}%. Median distance {vm['median_distance_m']:.1f} m."),
        ("Pole-top equipment", f"{eq['pole_top']['candidates_at_geometric_supports']} of {eq['pole_top']['supports_tested']} geometric supports ({eq['pole_top']['rate_geometric_supports'] * 100:.0f}%) show a thickening just below the top. The control (column-shaped places the support model calls not mapped poles) shows it {eq['pole_top']['rate_control'] * 100:.0f}% of the time. That separation is slight."),
        ("Not claimed", "The map has no pole-mounted equipment (all 97 mapped switches, transformers and portals are in the yards), so the pole-top candidates are unverified. At 25 returns per m2 a pole-mounted transformer is a few dozen returns; naming it a transformer needs imagery.")], 78)
    return pl.save("62_whiteriver_town_equipment_map.png")


def fig_eq_fig(D):
    eq = D["eq"]
    big = max(eq["yards"], key=lambda y: y["area_m2"])
    vm = big["versus_map"]
    pl = Plate("Equipment detection: how it checks out", "Substation structures against mapped equipment with a random-position control, and the pole-top rule against its control", src(D), "Yard: 97 mapped items; pole-top: no ground truth")
    a = pl.ax(1.0, 4.9, 6.4, 3.2)
    xs = np.arange(2)
    a.bar(xs - 0.2, [vm["within_1p5m"], vm["within_3m"]], 0.38, color="#27AE60", label="mapped equipment"); a.bar(xs + 0.2, [vm["random_within_1p5m"], vm["random_within_3m"]], 0.38, color="#BDC3C7", label="random points in the yard")
    for i, (u, v) in enumerate(zip([vm["within_1p5m"], vm["within_3m"]], [vm["random_within_1p5m"], vm["random_within_3m"]])):
        a.text(i - 0.2, u + 0.01, f"{u * 100:.0f}%", ha="center", fontsize=8); a.text(i + 0.2, v + 0.01, f"{v * 100:.0f}%", ha="center", fontsize=8)
    a.set_xticks(xs); a.set_xticklabels(["structure within 1.5 m", "structure within 3 m"], fontsize=8); a.set_ylim(0, 0.9); a.set_ylabel("share with a structure nearby", fontsize=8, color=MUTED); a.tick_params(labelsize=8); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=8, frameon=False)
    pl.label(1.0, 8.35, "Yard structures against the map")
    b = pl.ax(9.0, 4.9, 6.0, 3.2)
    pt = eq["pole_top"]
    vals = [pt["rate_geometric_supports"], pt["rate_learned_supports"], pt["rate_control"]]
    b.bar(range(3), vals, color=["#1F4E79", "#D35400", "#BDC3C7"])
    for i, v in enumerate(vals):
        b.text(i, v + 0.005, f"{v * 100:.0f}%", ha="center", fontsize=8)
    b.set_xticks(range(3)); b.set_xticklabels([f"geometric supports\n(n={pt['supports_tested']})", f"learned supports\n(n={pt['learned_supports_tested']})", f"control columns\n(n={pt['control_column_candidates_tested']})"], fontsize=8)
    b.set_ylabel("share with a pole-top thickening", fontsize=8, color=MUTED); b.tick_params(labelsize=8); b.spines[["top", "right"]].set_visible(False)
    pl.label(9.0, 8.35, "Pole-top rule against a control")
    c = pl.ax(1.0, 1.3, 6.4, 2.7)
    H = np.array(big["structure_top_hag"])
    c.hist(H, bins=np.arange(0, 22, 1), color="#8E44AD"); c.set_xlabel("top of the structure above ground, m", fontsize=8, color=MUTED); c.set_ylabel("structures", fontsize=8, color=MUTED); c.tick_params(labelsize=8); c.spines[["top", "right"]].set_visible(False)
    pl.label(1.0, 4.35, "Heights of the yard structures")
    rows(pl, 9.0, 4.2, [("Yard", f"Structures sit {vm['median_distance_m']:.1f} m (median) from the mapped equipment. The excess over random is real ({vm['within_3m'] * 100:.0f}% against {vm['random_within_3m'] * 100:.0f}%) but a quarter of the mapped items have no structure within 3 m, and most structures are unmapped or are fences, buswork and conduit rather than equipment."),
                       ("Pole tops", "The rule fires more often on supports than on controls, but only by a few percentage points, and about 85 to 90% of supports show no thickening. This does not identify transformers; it marks places to look at.")], 70)
    return pl.save("63_whiteriver_town_equipment_figure.png")


# ------------------------------------------------------------------------------------------------ 64/65 spans
def fig_span_map(D):
    spj = D["spj"]
    conf, add, br = line_layers(D)
    pl = Plate("Joining the network: observed, model-added and bridged spans", "Spans between supports scored on held-out tiles, then line ends that face each other bridged where the line continues straight", src(D), "Dashed orange: the model accepted a weakly observed span. Dotted red: no returns, inferred from the straight continuation")
    a = pl.ax(0.6, 1.1, 7.2, 7.2)
    base(a, D)
    mp = mapped_polyline_pts(D)
    a.scatter(mp[:, 0], mp[:, 1], s=1.2, color=C_MAP, zorder=3, linewidths=0)
    lc(a, conf, C_CONF, 1.3); lc(a, add, C_ADD, 1.3, "--", 6); lc(a, br, C_BR, 1.8, ":", 7)
    A = D["sp"]["A"]
    a.scatter(A[:, 0], A[:, 1], s=4, color="white", edgecolors="k", linewidths=0.3, zorder=9)
    scalebar(a, EXT, 500.0); north_arrow(a, EXT)
    legend(a, [Line2D([], [], color=C_CONF, lw=1.5, label=f"observed span (first rule, {len(conf)})"), Line2D([], [], color=C_ADD, lw=1.5, ls="--", label=f"added by the span model ({len(add)})"), Line2D([], [], color=C_BR, lw=2, ls=":", label=f"bridged line end ({len(br)})"),
               Line2D([], [], color=C_MAP, marker=".", ls="", label="mapped line"), pts_legend("white", "anchor", "o", 4, "white")], "lower right", 7.5)
    pl.label(0.6, 8.35, "Spans")
    g, ad, bd = spj["geometric_rule"], spj["added_by_model"], spj["bridged"]
    rows(pl, 8.3, 8.2, [
        ("Candidate spans", f"{spj['anchors']} anchors (the geometric supports and wire ends plus {spj['anchors_added_from_learned_supports']} from the learned detector) give {spj['candidate_pairs']:,} neighbour pairs; {spj['pair_table']:,} have a wire fit of some coverage. A tree model scores each from its length, returns, sag, end heights, the support score at each end and how straight it continues a confirmed neighbour."),
        ("What the model added", f"At score {spj['threshold']:.2f} (the lowest threshold whose held-out precision against the map stays within 2 points of the first rule's {g['precision_by_length'] * 100:.0f}%) it accepts {spj['scored_spans']['spans']} spans, {ad['spans']} of them new: {ad['length_m'] / 1000:.1f} km, {ad['precision_by_length'] * 100:.0f}% of which lies within 6 m of a mapped line. These spans are partly observed (from 20% of the span up) and partly modelled."),
        ("Bridging", f"{bd['spans']} line ends facing each other within {bd['gap_range_m'][0]} to {bd['gap_range_m'][1]} m and 15 degrees were joined, {bd['length_m']:.0f} m in all, {bd['fraction_of_length_within_6m_of_mapped_line'] * 100:.0f}% within 6 m of a mapped line. Heights come from a sag curve (coefficient {bd['sag_coefficient']:.5f} per m) between the end heights of the adjacent spans."),
        ("Limits", "Bridging is conservative: most gaps have no facing end because no support was found on the far side. Precision is against an incomplete map.")], 80)
    return pl.save("64_whiteriver_town_span_map.png")


def fig_span_fig(D):
    spj, st = D["spj"], D["stats"]
    pl = Plate("Spans and gap filling: results", "How much of the mapped network is recovered, how connected the model becomes, how precise the added spans are, and how accurate the filled wire is", src(D), "Against Overture lines (lower bounds); interpolation error from hidden stretches")
    a = pl.ax(1.0, 4.9, 6.4, 3.2)
    cov = spj["mapped_line_covered_within_6m"]
    v = [cov["confirmed"], cov["plus_scored_spans"], cov["plus_bridged"]]
    a.bar(range(3), [x * 100 for x in v], color=[C_CONF, C_ADD, C_BR])
    for i, x in enumerate(v):
        a.text(i, x * 100 + 0.8, f"{x * 100:.1f}%", ha="center", fontsize=8.5)
    a.set_xticks(range(3)); a.set_xticklabels(["first rule\n(observed spans)", "+ spans the model\nadded", "+ bridged\nline ends"], fontsize=8); a.set_ylabel("mapped line length with a wire within 6 m, %", fontsize=8, color=MUTED); a.set_ylim(0, 55); a.tick_params(labelsize=8); a.spines[["top", "right"]].set_visible(False)
    pl.label(1.0, 8.35, "Share of the mapped network recovered")
    b = pl.ax(9.0, 4.9, 6.0, 3.2)
    cn = spj["connectivity"]
    names = ["confirmed", "plus_scored_spans", "plus_bridged"]
    b.bar(np.arange(3) - 0.2, [cn[n]["components"] for n in names], 0.38, color="#34495E", label="separate pieces")
    for i, n in enumerate(names):
        b.text(i - 0.2, cn[n]["components"] + 0.8, str(cn[n]["components"]), ha="center", fontsize=8.5)
    b.set_ylabel("separate pieces", fontsize=8, color=MUTED); b.set_xticks(range(3)); b.set_xticklabels(["first rule", "+ model spans", "+ bridged"], fontsize=8)
    b2 = b.twinx(); b2.plot(range(3), [cn[n]["largest_component_share_of_nodes"] * 100 for n in names], "-o", color="#D35400", label="largest piece, % of supports"); b2.set_ylim(0, 100); b2.set_ylabel("largest piece, % of supports", fontsize=8, color="#D35400")
    b.tick_params(labelsize=8); b2.tick_params(labelsize=8); b.spines[["top"]].set_visible(False); b2.spines[["top"]].set_visible(False)
    pl.label(9.0, 8.35, "Connectivity of the network")
    c = pl.ax(1.0, 1.3, 6.4, 2.7)
    g, ad, bd = spj["geometric_rule"], spj["added_by_model"], spj["bridged"]
    pv = [g["precision_by_length"], ad["precision_by_length"], bd["fraction_of_length_within_6m_of_mapped_line"]]
    ln = [g["length_m"] / 1000, ad["length_m"] / 1000, bd["length_m"] / 1000]
    c.bar(range(3), [p * 100 for p in pv], color=[C_CONF, C_ADD, C_BR])
    for i, (p, l) in enumerate(zip(pv, ln)):
        c.text(i, p * 100 + 1.5, f"{p * 100:.0f}%\n({l:.1f} km)", ha="center", fontsize=8)
    c.set_xticks(range(3)); c.set_xticklabels(["first rule", "added by the model", "bridged"], fontsize=8); c.set_ylim(0, 115); c.set_ylabel("length within 6 m of a mapped line, %", fontsize=8, color=MUTED); c.tick_params(labelsize=8); c.spines[["top", "right"]].set_visible(False)
    pl.label(1.0, 4.35, "Precision against the map, by length")
    d_ = pl.ax(9.0, 1.3, 3.2, 2.7)
    h = st["interpolation_holdout"]
    x = np.arange(2)
    d_.bar(x - 0.2, [h["20"]["rms_parabola_m"] * 100, h["40"]["rms_parabola_m"] * 100], 0.38, color="#1F4E79", label="fitted sag curve"); d_.bar(x + 0.2, [h["20"]["rms_linear_m"] * 100, h["40"]["rms_linear_m"] * 100], 0.38, color="#BDC3C7", label="straight line")
    d_.set_xticks(x); d_.set_xticklabels(["20 m gap", "40 m gap"], fontsize=8); d_.set_ylabel("rms error, cm", fontsize=8, color=MUTED); d_.tick_params(labelsize=8); d_.spines[["top", "right"]].set_visible(False); d_.legend(fontsize=7.5, frameon=False)
    pl.label(9.0, 4.35, "Filling a gap in a wire")
    rows(pl, 12.5, 4.2, [("Reading", f"Interpolation: hiding a stretch of well-observed wire ({h['20']['n']} tests per size) and refitting gives {h['20']['rms_parabola_m'] * 100:.0f} to {h['40']['rms_parabola_m'] * 100:.0f} cm error with the sag curve against {h['20']['rms_linear_m'] * 100:.0f} to {h['40']['rms_linear_m'] * 100:.0f} cm for a straight line. Hidden stretches were inliers of the original fit, so this is optimistic for noisy wires.")], 32)
    return pl.save("65_whiteriver_town_span_figure.png")
