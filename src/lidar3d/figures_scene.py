"""Plates 71 to 74 for the White River scene (650 m by 530 m window): the same four views as the earlier 800 m study (overview against the map, profile and yard plan, joining broken wires, model against the map), drawn from the
town-scale network, which now keeps every detected wire piece."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from .plates import MUTED, Plate, map_frame, north_arrow, scalebar

TAN = "#CFC8B6"
BLUE, CYAN, MAG, GREEN, RED = "#1F4E79", "#17A2B8", "#B0298A", "#2F6B3F", "#C0392B"


def _src(S):
    return (f"Lidar: USGS 3DEP WA_PierceCounty_1_2020 on AWS (usgs-lidar-public), collected 2020-04-13; no wire or tower classes are present, so everything here is found from the points. Imagery: NAIP {S['naip_date']} (colour only). "
            "Mapped features: Overture Maps infrastructure (power), which derives from OpenStreetMap, used only to compare with the model, never to build it.")


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
    # numbers
    LP = np.array([ln[0] for ln in [(np.array([s.interpolate(i * 2.0).coords[0] for i in range(int(s.length // 2) + 1)]), None) for s in [__import__("shapely.geometry", fromlist=["LineString"]).LineString(c) for _, c in lines]]] if False else [0.0])
    samp = []
    for k, c in lines:
        ls = __import__("shapely.geometry", fromlist=["LineString"]).LineString(c)
        for i in range(int(ls.length // 2) + 1):
            samp.append(ls.interpolate(i * 2.0).coords[0])
    samp = np.array(samp)
    samp = samp[_inwin(samp, win)]
    tW = cKDTree(W) if len(W) else None
    cov6 = float((tW.query(samp)[0] <= 6).mean()) if tW is not None and len(samp) else 0.0
    dline = cKDTree(samp).query(W)[0] if len(samp) else np.array([])
    near5 = float((dline <= 5).mean()) if len(dline) else 0.0
    km = len(samp) * 2 / 1000
    chains = N["K"]["chain_l"]
    wlen = sum(np.hypot(*np.diff(c["xyz"][:, :2], axis=0).T).sum() for c in chains) / 1000
    # poles vs map
    tr = cKDTree(ref)
    dd, _ = tr.query(cols) if len(cols) else (np.array([]), None)
    matched = int((dd <= 3).sum())
    d2 = cKDTree(cols).query(ref)[0] if len(cols) else np.array([])
    recall = float((d2 <= 3).mean()) if len(d2) else 0.0
    prec = matched / max(len(cols), 1)
    big = max(eq["yards"], key=lambda y_: y_["area_m2"])
    vm = big["versus_map"]
    pl = Plate("Poles, wires and a substation from lidar", "What a 25 points per m2 survey shows around White River Substation, Pierce County, Washington, against the mapped power features; this window is 650 m by 530 m", _src(S), "Local grid, metres from the substation centre (UTM 10N)")
    ax = pl.ax(0.6, 0.95, 8.55, 7.0)
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
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=3, label="wire points found in the lidar"), Line2D([], [], color=RED, lw=0.9, label="mapped power and minor lines"),
                       Line2D([], [], marker="^", ls="", mfc="none", mec="k", label="mapped tower"), Line2D([], [], marker="o", ls="", mfc="none", mec=GREEN, label="mapped pole"),
                       Line2D([], [], marker="o", ls="", mfc="none", mec=MAG, ms=8, label="vertical column found in the lidar"), Line2D([], [], color=MAG, lw=1.4, label="mapped substation outline")], loc="lower right", fontsize=7.5, framealpha=0.92)
    pl.label(0.6, 8.15, "Detected wires and columns on mapped lines, towers and poles")
    pl.label(9.6, 8.15, "How well the lidar and the map agree")
    _rows(pl, 9.7, 7.9, [
        ("Wires", f"{len(chains)} wire pieces, {wlen:.1f} km, median {np.median(S['hag'][wm]):.0f} m above ground. {near5 * 100:.0f}% of wire points lie within 5 m of a mapped line, so little is invented. But only {cov6 * 100:.0f}% of the {km:.1f} km of mapped line in the window has a detected wire within 6 m: the rest is a mix of lines with returns on them but no wire found, lines under vegetation, and lines with no elevated returns at all (plates 58 and 59 break this down)."),
        ("Poles and towers", f"{len(cols)} columns found (including posts in the yard); {matched} lie within 3 m of one of the {len(ref)} mapped poles and towers (precision {prec * 100:.0f}%, recall {recall * 100:.0f}%). A pole is a few returns thick at this density, so blind detection is weak: only 41% of mapped poles have any column-shaped returns within 3 m, which caps what a detector can find, and the mapped positions are the better guide."),
        ("Substation yard", f"{big['structures']} elevated structures found inside the yard ({big['area_m2'] / 1000:.0f} thousand m2), median top {big['median_top_hag_m']:.0f} m above ground. Of {vm['mapped_equipment']} mapped switches, transformers and portals, {vm['within_3m'] * 100:.0f}% have a lidar structure within 3 m, against {vm['random_within_3m'] * 100:.0f}% for random points in the yard ({vm['within_1p5m'] * 100:.0f}% against {vm['random_within_1p5m'] * 100:.0f}% within 1.5 m).")], 62)
    pl.save(name)
    return dict(wire_pieces=len(chains), wire_km=float(wlen), within5=near5, mapped_km=km, mapped_covered=cov6, columns=int(len(cols)), poles_ref=int(len(ref)), pole_precision=prec, pole_recall=recall)


# ------------------------------------------------------------------------------------------------ 72 profile and plan
def fig_profile(S, D, N, feats, win, eq, name="72_whiteriver_scene_profile_yard.png"):
    d = S["d"]
    x, y, z = d["x"], d["y"], d["z"]
    band = np.abs(y - 0.0) <= 30.0
    pl = Plate("Wires and a substation yard, in profile and in plan", "A side view of the lidar through the substation, with the wires the detector found, and the elevated structures in the yard", _src(S), "Local grid, metres from the substation centre (UTM 10N)")
    a = pl.ax(1.0, 5.2, 14.0, 2.95)
    g = band & (d["cls"] == 2)
    o = band & (d["cls"] != 2) & ~D["wmask"]
    w = band & D["wmask"]
    rng = np.random.default_rng(0)
    sub = lambda m, f: np.flatnonzero(m)[rng.random(m.sum()) < f]
    for idx, col, sz in ((sub(o, 0.12), "#B8B09B", 0.25), (sub(g, 0.12), "#7A6A55", 0.3), (sub(w, 1.0), BLUE, 1.6)):
        a.scatter(x[idx], z[idx], s=sz, color=col, linewidths=0, rasterized=True)
    a.set_xlim(win[0], win[1]); a.set_xlabel("east, m from the substation centre", fontsize=8, color=MUTED); a.set_ylabel("elevation, m", fontsize=8, color=MUTED)
    a.tick_params(labelsize=8); a.spines[["top", "right"]].set_visible(False)
    pl.label(1.0, 8.3, "Side view through the yard (a 60 m wide band): wires in blue")
    big = max(eq["yards"], key=lambda y_: y_["area_m2"])
    b = pl.ax(1.0, 0.95, 4.9, 3.35)
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
    b.plot(yx, yy, color=MAG, lw=1.2)
    b.set_xlim(gx0, gx0 + nx); b.set_ylim(gy0, gy0 + ny); b.set_aspect("equal"); b.tick_params(labelsize=7)
    pl.label(1.0, 4.42, "Yard: structures (red x), mapped equipment (white rings)")
    _rows(pl, 6.4, 4.25, [
        ("Reading the plan", "Height above ground in the yard runs from 0 to 15 m (viridis). Structures are connected groups of points more than 1.5 m above ground that are not wires, drawn as red crosses. A group can be one piece of equipment or several that touch, so the number of structures is not a count of equipment, and the match to the map is a position check, not an identification."),
        ("Wires", "Wires are found where the lidar points form a thin, near-horizontal line with open space around it. Wires hidden under tree crowns, under 4 m above ground, or very thin conductors can be missed."),
        ("The profile", "The side view shows the sag of each span between supports: a catenary curve in the blue points. Gantries and towers rise 15 to 25 m above the yard floor; the yard is levelled about 6 m above the land to its west.")], 120, fs=8.0)
    pl.save(name)


# ------------------------------------------------------------------------------------------------ 73 joining
def fig_joining(S, N, name="73_whiteriver_scene_joining_gaps.png"):
    K = N["K"]
    conds = [c for c in K["cond"] if c["interpolated_bins"] >= 8 and c["observed_bins"] >= 20]
    conds = sorted(conds, key=lambda c: -c["interpolated_bins"] * np.sqrt(c["length"]))[:3]
    pl = Plate("Joining broken wires and filling the gaps", "Wire fragments the lidar found are grouped into conductors; a sag curve bridges the breaks, and supports are placed where the height profile peaks", _src(S), "Heights above ground; supports are the peaks of each conductor's height profile")
    ys = [6.05, 4.1, 2.15]
    for k, (c, y0) in enumerate(zip(conds, ys)):
        a = pl.ax(1.2, y0, 9.8, 1.5)
        s = c["s"]
        fl = np.asarray(c["flag"])
        zo = c["z_fill"]
        a.plot(s, zo, color="#B5B5B5", lw=1.0, zorder=1)
        a.scatter(s[fl == 1], zo[fl == 1], s=7, color=BLUE, zorder=3, label="observed in the lidar", linewidths=0)
        a.scatter(s[fl == 2], zo[fl == 2], s=7, color=CYAN, zorder=3, label="filled by the sag curve", linewidths=0)
        sup = np.asarray(c["supports"]).astype(int)
        a.scatter(s[sup], zo[sup], marker="^", s=55, facecolors="none", edgecolors=MAG, linewidths=1.4, zorder=5, label="support at a height peak")
        a.set_ylabel("m above ground", fontsize=8, color=MUTED); a.tick_params(labelsize=8); a.spines[["top", "right"]].set_visible(False)
        a.text(-0.07, 1.0, "ABC"[k], transform=a.transAxes, fontsize=15, fontweight="bold", va="top")
        if k == 0:
            a.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=3)
        if k == 2:
            a.set_xlabel("distance along the conductor, m", fontsize=8, color=MUTED)
        obs, fil = c["observed_bins"] * 2.0, c["interpolated_bins"] * 2.0
        pl.text(11.3, y0 + 1.4, f"{len(c['members'])} fragments, {c['length']:.0f} m long\n{obs:.0f} m observed, {fil:.0f} m filled\n{len(c['supports'])} supports", fontsize=8.5, color=MUTED, va="top", linespacing=1.4)
    tot_obs = sum(c["observed_bins"] for c in K["cond"]) * 2.0
    tot_fil = sum(c["interpolated_bins"] for c in K["cond"]) * 2.0
    frag = sum(len(c["members"]) for c in K["cond"])
    pl.text(1.2, 1.55, "\n".join(textwrap.wrap(f"{frag} wire fragments became {len(K['cond'])} conductors; {tot_fil:.0f} m of conductor was filled in, against {tot_obs:.0f} m observed. The wire pieces no fitted span explained are kept as their own polylines (plate 68 and plate 57), and pieces that line up are bridged, so a gap only disappears when the curve is supported by wire on both sides.", 175)),
            fontsize=8.5, color=MUTED, va="top", linespacing=1.4)
    pl.save(name)


# ------------------------------------------------------------------------------------------------ 74 model vs map
def fig_fused(S, D, N, feats, win, name="74_whiteriver_scene_model_vs_map.png"):
    from shapely.geometry import LineString
    d = S["d"]
    K, br = N["K"], N["br"]
    net = _net_points(N)
    tnet = cKDTree(net[:, :2])
    # elevated returns not explained by wire: any return 8 m or more above ground that is not a wire point and sits in open space (few neighbours)
    m = (S["hag"] >= 8.0) & ~D["wmask"] & _inwin(np.c_[d["x"], d["y"]], win)
    Pe = np.c_[d["x"][m], d["y"][m]]
    te = cKDTree(Pe)
    cats = []
    segs = []
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
    tot = sum(c[1] for c in cats)
    km = {k: sum(c[1] for c in cats if c[0] == k) for k in colors}
    pl = Plate("Fused line and support model against OpenStreetMap", "Conductors and supports from the lidar (blue, cyan, magenta) over the mapped lines and supports (orange to red, green)", _src(S), "Local grid, metres from the substation centre (UTM 10N)")
    ax = pl.ax(0.6, 0.95, 8.55, 7.0)
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
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="conductor, observed"), Line2D([], [], marker="o", ls="", color=CYAN, ms=4, label="conductor, filled or bridged"), Line2D([], [], marker="o", ls="", mfc="none", mec=MAG, ms=8, label="fused support"),
                       Line2D([], [], marker="+", ls="", color=GREEN, label="mapped tower or pole"), Line2D([], [], color=colors["confirmed"], lw=2, label="mapped line: confirmed"), Line2D([], [], color=colors["partial"], lw=2, label="partial"),
                       Line2D([], [], color=colors["elevated"], lw=2, label="elevated returns, no wire found"), Line2D([], [], color=colors["none"], lw=2, label="no lidar evidence")], loc="lower right", fontsize=7.5, framealpha=0.92)
    pl.label(0.6, 8.15, "Model over the mapped features")
    pl.label(9.6, 8.15, "Mapped lines by what the lidar shows")
    ys = 7.75
    for k, txt in (("confirmed", "Confirmed: network within 6 m of 60% or more of the line"), ("partial", "Partial: 20% to 60%"), ("elevated", "Under 20% covered, but isolated returns at least 8 m up along half of it"), ("none", "Under 20% covered and no such returns")):
        pl.text(9.7, ys, f"{km[k] / 1000:.1f} km  ({km[k] / max(tot, 1) * 100:.0f}%)", fontsize=11, fontweight="bold", va="top")
        pl.text(11.9, ys - 0.03, "\n".join(textwrap.wrap(txt, 38)), fontsize=8.5, color=MUTED, va="top")
        ys -= 0.62
    dd = cKDTree(np.vstack([c[2] for c in cats if c[0] == "confirmed"] or [np.zeros((1, 2))]))
    prec_r = {}
    for r_ in (5.0,):
        tr = cKDTree(ref)
        dsup = tr.query(sup)[0]
        prec = float((dsup <= r_).mean()); rec = float((cKDTree(sup).query(ref)[0] <= r_).mean())
    _rows(pl, 9.7, 5.0, [
        ("Supports", f"{len(sup)} fused supports in the window against {len(ref)} mapped; {int((dsup <= 5).sum())} lie within 5 m of a mapped one (precision {prec * 100:.0f}%, recall {rec * 100:.0f}%). Mapped supports are traced from imagery or surveyed to varying accuracy and the map is incomplete, so an unmatched support is not necessarily wrong and a missed one is not necessarily absent."),
        ("Lines", f"The network lies within {np.median(tnet.query(np.vstack([c[2] for cat, _, c in cats if cat == 'confirmed']))[0]):.1f} m (median) of the mapped line where they agree. Every wire piece the detector found is in the network now, so 'no lidar evidence' means no wire piece within 6 m and nothing elevated above the line, which is a statement about the lidar, not a mapping error.")], 62)
    pl.save(name)
    return dict(km=km, total_km=tot / 1000, supports=int(len(sup)), mapped=int(len(ref)), precision=prec, recall=rec)
