"""Plates 71 to 74 for the White River scene (650 m by 530 m window): the same four views as the earlier 800 m study (overview against the map, profile and yard plan, joining broken wires, model against the map), drawn from the
town-scale network, which now keeps every detected wire piece."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from .plates import map_frame, north_arrow, scalebar
from .plates_pro import ACCENT2, MUTED, ProPlate

TAN = "#CFC8B6"
BLUE, CYAN, MAG, GREEN, RED = "#1F4E79", "#17A2B8", "#B0298A", "#2F6B3F", "#C0392B"


def _src(S):
    return (f"Lidar: USGS 3DEP project WA_PierceCounty_1_2020, collected 2020-04-13, about 25 returns per m2; the survey carries no wire, tower, building or vegetation classes, so every feature is derived from the points. Imagery: NAIP, {S['naip_date']}. "
            "Reference: Overture Maps power features (OpenStreetMap-derived), used only to check results, never to build them. Local grid in metres from the substation centre, UTM zone 10N.")


KICK = "White River scene  ·  650 m by 530 m"
SBX, SBW = 9.85, 5.55          # sidebar x and width


def _rows(pl, x, y0, items, width, fs=8.2, head=9.5):
    for k_, v_ in items:
        pl.text(x, y0, k_, fontsize=head, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v_, width))
        pl.text(x, y0 - 0.26, w, fontsize=fs, color=MUTED, linespacing=1.4, va="top")
        y0 -= 0.3 + 0.17 * (w.count("\n") + 1) + 0.32
    return y0


def _background(ax, S, win, res=1.0):
    d = S["d"]
    x0, x1, y0, y1 = win
    m = (S["hag"] >= 2.0) & (d["cls"] != 2)
    nx, ny = int((x1 - x0) / res), int((y1 - y0) / res)
    ix = np.clip(((d["x"][m] - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(((d["y"][m] - y0) / res).astype(int), 0, ny - 1)
    cnt = np.bincount(iy * nx + ix, minlength=nx * ny).reshape(ny, nx)
    img = np.where(cnt[..., None] >= 2, np.array([0.81, 0.78, 0.71, 1.0]), np.array([1, 1, 1, 0.0]))
    ax.imshow(img, origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest", zorder=1)


def _lines(feats, kinds=("power_line", "minor_line")):
    out = []
    for k in kinds:
        for g in feats.get(k, []):
            for p in (g.geoms if hasattr(g, "geoms") else [g]):
                if p.geom_type == "LineString":
                    out.append((k, np.asarray(p.coords)))
    return out


def _pts(feats, k):
    return np.array([(g.x, g.y) for g in feats.get(k, []) if g.geom_type == "Point"]).reshape(-1, 2)


def _inwin(xy, win, pad=0.0):
    return (xy[:, 0] >= win[0] - pad) & (xy[:, 0] <= win[1] + pad) & (xy[:, 1] >= win[2] - pad) & (xy[:, 1] <= win[3] + pad)


def _net_points(N, step=1.0):
    from . import spans, unified as U
    K, br = N["K"], N["br"]
    parts = [U.densify(w, step) for w in K["span_w"]] + [U.densify(a["xyz"], step) for a in K["adds"]] + [U.densify(b["xyz"], step) for b in br]
    return np.vstack(parts)


# ------------------------------------------------------------------------------------------------ 71 overview
def fig_overview(S, D, N, feats, win, eq, name="71_whiteriver_scene_overview.png"):
    from shapely.geometry import LineString
    d = S["d"]
    x, y = d["x"], d["y"]
    wm = D["wmask"] & _inwin(np.c_[x, y], win)
    W = np.c_[x[wm], y[wm]]
    lines = _lines(feats)
    towers, poles = _pts(feats, "power_tower"), _pts(feats, "power_pole")
    ref = np.vstack([towers, poles])
    ref = ref[_inwin(ref, win)]
    K = N["K"]
    cols = np.array([(p["x"], p["y"]) for p in K["poles"]] + [(p["x"], p["y"]) for p in N["Y"]["posts"]]).reshape(-1, 2)
    samp = []
    for k, c in lines:
        ls = LineString(c)
        for i in range(int(ls.length // 2) + 1):
            samp.append(ls.interpolate(i * 2.0).coords[0])
    samp = np.array(samp)
    samp = samp[_inwin(samp, win)]
    cov6 = float((cKDTree(W).query(samp)[0] <= 6).mean()) if len(W) and len(samp) else 0.0
    near5 = float((cKDTree(samp).query(W)[0] <= 5).mean()) if len(samp) else 0.0
    km = len(samp) * 2 / 1000
    chains = K["chain_l"]
    wlen = sum(np.hypot(*np.diff(c["xyz"][:, :2], axis=0).T).sum() for c in chains) / 1000
    dd = cKDTree(ref).query(cols)[0] if len(cols) else np.array([])
    matched = int((dd <= 3).sum())
    recall = float((cKDTree(cols).query(ref)[0] <= 3).mean()) if len(cols) else 0.0
    prec = matched / max(len(cols), 1)
    big = max(eq["yards"], key=lambda y_: y_["area_m2"])
    vm = big["versus_map"]
    pl = ProPlate(71, KICK, "Poles, wires and a substation from lidar", "What a 25 points-per-m\u00b2 survey shows around White River Substation, Pierce County, Washington, set against the mapped power features", _src(S))
    pl.panel_title(0.6, 8.14, "A", "Detected wires and columns on the mapped network", w=8.9)
    ax = pl.ax(0.6, 1.0, 8.9, 6.95)
    _background(ax, S, win)
    map_frame(ax, win, step=100.0, grid=False)
    for _, c in lines:
        ax.plot(c[:, 0], c[:, 1], color=RED, lw=0.7, zorder=3, alpha=0.9)
    ax.scatter(W[:, 0], W[:, 1], s=1.6, color=BLUE, zorder=5, linewidths=0)
    ax.scatter(towers[:, 0], towers[:, 1], marker="^", s=26, facecolors="none", edgecolors="k", linewidths=0.9, zorder=7)
    ax.scatter(poles[:, 0], poles[:, 1], marker="o", s=14, facecolors="none", edgecolors=GREEN, linewidths=0.9, zorder=7)
    ax.scatter(cols[:, 0], cols[:, 1], marker="o", s=70, facecolors="none", edgecolors=MAG, linewidths=1.3, zorder=8)
    yx, yy = N["Y"]["yard"].exterior.xy
    ax.plot(yx, yy, color=MAG, lw=1.4, zorder=6)
    ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
    scalebar(ax, win, 100.0); north_arrow(ax, win)
    pl.panel_title(SBX, 8.14, "B", "Key numbers", w=SBW)
    y0 = pl.tiles(SBX, 7.85, SBW, [(f"{wlen:.1f} km", f"of wire found, in {len(chains)} pieces"), (f"{near5 * 100:.0f}%", "of wire points lie within 5 m of a mapped line"), (f"{cov6 * 100:.0f}%", f"of the {km:.1f} km of mapped line has a wire within 6 m")])
    y0 = pl.notes(SBX, y0 - 0.05, SBW, [
        ("What the lidar finds", f"Wires show up as thin, near-horizontal lines of returns with open air around them. Almost everything found sits on a mapped line, so very little is invented. What is missing is wire the survey did not capture: lines under tree crowns, lines with no returns in open ground, and spans whose supports were not detected."),
        ("Poles and towers", f"{len(cols)} columns were found, {matched} of them within 3 m of one of {len(ref)} mapped poles and towers (precision {prec * 100:.0f}%, recall {recall * 100:.0f}%). A pole is only a few returns thick at this density, and only 41% of mapped poles have any column of returns within 3 m, which caps what any detector can find."),
        ("Substation yard", f"{big['structures']} elevated structures lie inside the {big['area_m2'] / 1e4:.1f} ha yard. {vm['within_3m'] * 100:.0f}% of its {vm['mapped_equipment']} mapped switches, transformers and portals have a structure within 3 m, against {vm['random_within_3m'] * 100:.0f}% for random points. The match checks position, not identity.")])
    pl.legend(SBX, y0 - 0.05, SBW, [("line", BLUE, "wire points found in the lidar"), ("line", RED, "mapped power and minor lines"), ("tri", "k", "mapped tower"), ("ring", GREEN, "mapped pole"), ("ring", MAG, "vertical column found in the lidar"), ("line", MAG, "mapped substation outline")], title="LEGEND")
    pl.save(name)
    return dict(wire_pieces=len(chains), wire_km=float(wlen), within5=near5, mapped_km=km, mapped_covered=cov6, columns=int(len(cols)), poles_ref=int(len(ref)), pole_precision=prec, pole_recall=recall)


# ------------------------------------------------------------------------------------------------ 72 profile and plan
def fig_profile(S, D, N, feats, win, eq, name="72_whiteriver_scene_profile_yard.png"):
    d = S["d"]
    x, y, z = d["x"], d["y"], d["z"]
    band = np.abs(y - 0.0) <= 30.0
    pl = ProPlate(72, KICK, "Wires and a substation yard in profile and plan", "A side view of the lidar through the substation with the wires found, and the elevated structures inside the yard", _src(S))
    pl.panel_title(0.6, 8.14, "A", "Side view through the yard (a 60 m wide band, wires in blue)", w=14.8)
    a = pl.ax(1.25, 5.35, 14.15, 2.55)
    g = band & (d["cls"] == 2)
    o = band & (d["cls"] != 2) & ~D["wmask"]
    w = band & D["wmask"]
    rng = np.random.default_rng(0)
    sub = lambda m, f: np.flatnonzero(m)[rng.random(m.sum()) < f]
    for idx, col, sz in ((sub(o, 0.12), "#B8B09B", 0.25), (sub(g, 0.12), "#7A6A55", 0.3), (sub(w, 1.0), BLUE, 1.6)):
        a.scatter(x[idx], z[idx], s=sz, color=col, linewidths=0, rasterized=True)
    a.set_xlim(win[0], win[1]); a.set_xlabel("east, m from the substation centre", fontsize=8, color=MUTED); a.set_ylabel("elevation, m", fontsize=8, color=MUTED)
    a.tick_params(labelsize=8, colors=MUTED); a.spines[["top", "right"]].set_visible(False)
    big = max(eq["yards"], key=lambda y_: y_["area_m2"])
    pl.panel_title(0.6, 4.55, "B", "Yard plan: height above ground, structures and mapped equipment", w=5.4)
    b = pl.ax(0.75, 1.0, 5.1, 3.3)
    m = (np.abs(x - big["centre"][0]) <= 120) & (np.abs(y - big["centre"][1]) <= 100) & (S["hag"] > 0.3)
    nx, ny = 240, 200
    gx0, gy0 = big["centre"][0] - 120, big["centre"][1] - 100
    ix = np.clip(((x[m] - gx0) / 1.0).astype(int), 0, nx - 1); iy = np.clip(((y[m] - gy0) / 1.0).astype(int), 0, ny - 1)
    top = np.full(nx * ny, np.nan)
    np.fmax.at(top, iy * nx + ix, np.minimum(S["hag"][m], 15))
    b.imshow(top.reshape(ny, nx), origin="lower", extent=(gx0, gx0 + nx, gy0, gy0 + ny), cmap="viridis", vmin=0, vmax=15, interpolation="nearest")
    S_ = np.array(big["structure_xy"]).reshape(-1, 2); M_ = np.array(big["mapped_xy"]).reshape(-1, 2)
    b.scatter(S_[:, 0], S_[:, 1], marker="x", s=14, color="#E74C3C", linewidths=0.8)
    b.scatter(M_[:, 0], M_[:, 1], marker="o", s=26, facecolors="none", edgecolors="white", linewidths=0.9)
    yx, yy = N["Y"]["yard"].exterior.xy
    b.plot(yx, yy, color=ACCENT2, lw=1.2)
    b.set_xlim(gx0, gx0 + nx); b.set_ylim(gy0, gy0 + ny); b.set_aspect("equal"); b.tick_params(labelsize=7, colors=MUTED)
    pl.legend(6.3, 4.3, 3.2, [("dot", "#E74C3C", "structure found"), ("ring", "#888", "mapped equipment")], fs=7.4)
    vm = big["versus_map"]
    pl.panel_title(6.3, 3.75, "C", "Reading the figure", w=9.1)
    pl.notes(6.3, 3.45, 4.35, [
        ("Side view", "Each span between supports sags in a catenary curve, visible in the blue points. Gantries and towers rise 15 to 25 m above the yard floor, and the yard is levelled about 6 m above the land to its west."),
        ("Wires", "Wires are found where returns form a thin, near-horizontal line with open space around it. Wires under tree crowns, under 4 m above ground, or too thin to return a pulse can be missed.")], fs=8.3)
    pl.notes(10.95, 3.45, 4.45, [
        ("Yard plan", f"Structures are connected groups of returns more than 1.5 m above ground that are not wires. One group can be one device or several that touch, so the count of {big['structures']} is not a count of equipment."),
        ("Check against the map", f"{vm['within_3m'] * 100:.0f}% of {vm['mapped_equipment']} mapped items have a structure within 3 m (random points: {vm['random_within_3m'] * 100:.0f}%). This tests position only.")], fs=8.3)
    pl.save(name)


# ------------------------------------------------------------------------------------------------ 73 joining
def fig_joining(S, N, name="73_whiteriver_scene_joining_gaps.png"):
    K = N["K"]
    conds = [c for c in K["cond"] if c["interpolated_bins"] >= 8 and c["observed_bins"] >= 20]
    conds = sorted(conds, key=lambda c: -c["interpolated_bins"] * np.sqrt(c["length"]))[:3]
    tot_obs = sum(c["observed_bins"] for c in K["cond"]) * 2.0
    tot_fil = sum(c["interpolated_bins"] for c in K["cond"]) * 2.0
    frag = sum(len(c["members"]) for c in K["cond"])
    pl = ProPlate(73, KICK, "Joining broken wires and filling the gaps", "Wire fragments are grouped into conductors, a sag curve bridges the breaks, and supports sit where the height profile peaks", _src(S))
    ys = [6.15, 4.35, 2.55]
    for k, (c, y0) in enumerate(zip(conds, ys)):
        pl.panel_title(0.6, y0 + 1.72, "ABC"[k], f"Conductor {'ABC'[k]}: {len(c['members'])} fragments, {c['length']:.0f} m long", w=10.6)
        a = pl.ax(1.2, y0, 10.0, 1.45)
        s = c["s"]
        fl = np.asarray(c["flag"])
        zo = c["z_fill"]
        a.plot(s, zo, color="#B5B5B5", lw=1.0, zorder=1)
        a.scatter(s[fl == 1], zo[fl == 1], s=7, color=BLUE, zorder=3, linewidths=0)
        a.scatter(s[fl == 2], zo[fl == 2], s=7, color=CYAN, zorder=3, linewidths=0)
        sup = np.asarray(c["supports"]).astype(int)
        a.scatter(s[sup], zo[sup], marker="^", s=55, facecolors="none", edgecolors=MAG, linewidths=1.4, zorder=5)
        a.set_ylabel("m above ground", fontsize=8, color=MUTED); a.tick_params(labelsize=8, colors=MUTED); a.spines[["top", "right"]].set_visible(False)
        if k == 2:
            a.set_xlabel("distance along the conductor, m", fontsize=8, color=MUTED)
    pl.legend(1.2, 1.55, 10.0, [("dot", BLUE, "observed in the lidar"), ("dot", CYAN, "filled by the sag curve"), ("tri", MAG, "support at a height peak")])
    pl.panel_title(SBX + 0.3, 8.14, "D", "Key numbers", w=SBW - 0.3)
    y0 = pl.tiles(SBX + 0.3, 7.85, SBW - 0.3, [(f"{frag}", f"wire fragments joined into {len(K['cond'])} conductors"), (f"{tot_fil / 1000:.2f} km", "of conductor filled by a sag curve"), (f"{tot_obs / 1000:.1f} km", "of conductor observed directly")], num_fs=19)
    pl.notes(SBX + 0.3, y0 - 0.05, SBW - 0.3, [
        ("How fragments are joined", "Pieces are chained when they are collinear within a few degrees, close in height and no more than 60 m apart, so parallel wires in a bundle stay separate. A gap is filled with a parabola only when the wire is seen on both sides."),
        ("What filling can and cannot do", "Filling restores continuity inside a span. It adds little coverage of the mapped network, because most of the missing length is whole spans with no observed wire, which a curve cannot invent."),
        ("Where the rest goes", "Wire pieces that no fitted span explains are kept as their own polylines, and pieces that line up are bridged (plates 57 and 82).")])
    pl.save(name)


# ------------------------------------------------------------------------------------------------ 74 model vs map
def fig_fused(S, D, N, feats, win, name="74_whiteriver_scene_model_vs_map.png"):
    from shapely.geometry import LineString
    d = S["d"]
    K, br = N["K"], N["br"]
    net = _net_points(N)
    tnet = cKDTree(net[:, :2])
    m = (S["hag"] >= 8.0) & ~D["wmask"] & _inwin(np.c_[d["x"], d["y"]], win)
    te = cKDTree(np.c_[d["x"][m], d["y"][m]])
    cats = []
    for k, c in _lines(feats):
        ls = LineString(c)
        n = max(int(ls.length // 2), 1)
        pts = np.array([ls.interpolate(i * 2.0).coords[0] for i in range(n + 1)])
        ok = _inwin(pts, win)
        if ok.sum() < 5:
            continue
        pts = pts[ok]
        cov = float((tnet.query(pts)[0] <= 6).mean())
        iso = float((np.array(te.query_ball_point(pts, 3.0, return_length=True)) >= 3).mean())
        cat = "confirmed" if cov >= 0.6 else ("partial" if cov >= 0.2 else ("elevated" if iso >= 0.5 else "none"))
        cats.append((cat, len(pts) * 2.0, c))
    colors = {"confirmed": "#E67E22", "partial": "#F1C40F", "elevated": "#C0392B", "none": "#7B241C"}
    labels = {"confirmed": "Confirmed", "partial": "Partial", "elevated": "Returns, no wire", "none": "No lidar evidence"}
    tot = sum(c[1] for c in cats)
    km = {k: sum(c[1] for c in cats if c[0] == k) for k in colors}
    pl = ProPlate(74, KICK, "The wire and support model against the mapped network", "Conductors and supports from the lidar over the mapped lines, each mapped line coloured by how much the lidar supports it", _src(S))
    pl.panel_title(0.6, 8.14, "A", "Model over the mapped features", w=8.9)
    ax = pl.ax(0.6, 1.0, 8.9, 6.95)
    _background(ax, S, win)
    map_frame(ax, win, step=100.0, grid=False)
    for cat, _, c in cats:
        ax.plot(c[:, 0], c[:, 1], color=colors[cat], lw=1.8, zorder=3, alpha=0.95, solid_capstyle="butt")
    obs, fil = [], []
    for w in K["span_w"]:
        obs.append(w)
    for a_ in K["adds"]:
        (fil if np.mean(a_["flag"]) > 1.5 else obs).append(a_["xyz"])
    for b in br:
        fil.append(b["xyz"])
    O = np.vstack([w[:, :2] for w in obs]); F = np.vstack([w[:, :2] for w in fil]) if fil else np.zeros((0, 2))
    ax.scatter(O[:, 0], O[:, 1], s=1.4, color=BLUE, zorder=5, linewidths=0)
    if len(F):
        ax.scatter(F[:, 0], F[:, 1], s=1.4, color=CYAN, zorder=5, linewidths=0)
    sup = K["sup"][K["sup"].tier == "support"][["x", "y"]].to_numpy()
    ax.scatter(sup[:, 0], sup[:, 1], marker="o", s=50, facecolors="none", edgecolors=MAG, linewidths=1.2, zorder=8)
    ref = np.vstack([_pts(feats, "power_tower"), _pts(feats, "power_pole")]); ref = ref[_inwin(ref, win)]
    ax.scatter(ref[:, 0], ref[:, 1], marker="+", s=30, color=GREEN, linewidths=1.2, zorder=7)
    ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3]); scalebar(ax, win, 100.0); north_arrow(ax, win)
    dsup = cKDTree(ref).query(sup)[0]
    prec = float((dsup <= 5).mean()); rec = float((cKDTree(sup).query(ref)[0] <= 5).mean())
    off = float(np.median(tnet.query(np.vstack([c[2] for cat, _, c in cats if cat == "confirmed"]))[0]))
    pl.panel_title(SBX, 8.14, "B", "Mapped line length by lidar evidence", w=SBW)
    y0 = pl.tiles(SBX, 7.85, SBW, [(f"{km['confirmed'] / 1000:.1f} km", f"confirmed: {km['confirmed'] / tot * 100:.0f}% of the mapped line"), (f"{km['partial'] / 1000:.1f} km", f"partial: {km['partial'] / tot * 100:.0f}%"), (f"{(km['elevated'] + km['none']) / 1000:.1f} km", f"no wire found: {(km['elevated'] + km['none']) / tot * 100:.0f}%")])
    y0 = pl.notes(SBX, y0 - 0.05, SBW, [
        ("How lines are graded", f"A mapped line is confirmed when the network lies within 6 m of 60% or more of it, and partial from 20% to 60%. Below 20% it is 'returns, no wire' ({km['elevated'] / tot * 100:.0f}%) when isolated returns 8 m or more above ground lie along half of it, and 'no lidar evidence' ({km['none'] / tot * 100:.0f}%) otherwise. These describe the lidar, not mapping errors."),
        ("Supports", f"{len(sup)} fused supports in the window against {len(ref)} mapped; {int((dsup <= 5).sum())} lie within 5 m of a mapped one (precision {prec * 100:.0f}%, recall {rec * 100:.0f}%). The map is incomplete and traced from imagery, so an unmatched support is not necessarily wrong."),
        ("Position", f"Where model and map agree, the wire network lies a median {off:.1f} m from the mapped line. Every wire piece the detector found is in the network.")])
    pl.legend(SBX, y0 - 0.05, SBW, [("dot", BLUE, "conductor, observed"), ("dot", CYAN, "conductor, filled or bridged"), ("ring", MAG, "fused support"), ("dot", GREEN, "mapped tower or pole")] + [("line", colors[k], "mapped line: " + labels[k].lower()) for k in colors], title="LEGEND")
    pl.save(name)
    return dict(km=km, total_km=tot / 1000, supports=int(len(sup)), mapped=int(len(ref)), precision=prec, recall=rec)
