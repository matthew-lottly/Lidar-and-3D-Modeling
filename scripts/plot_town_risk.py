"""Plates 83 and 84: how close the tree canopy is to the White River lines, poles and towers, in three dimensions.

Two measures, both from the lidar:
  * Fall reach. A tree is a rigid stem of its lidar height H pivoting at its base. It can strike a point of a wire at horizontal distance d from the base and
    height h above the base's ground if sqrt(d^2 + h^2) <= H. The margin H - sqrt(d^2 + h^2) says by how much it would reach. A support is reachable if d <= H
    (the stem lands somewhere along the pole or tower).
  * Clearance. The 3D distance from each 1 m sample of wire to the nearest vegetation return (grow-in, contact in wind).
Usage: xvfb-run -a python scripts/plot_town_risk.py [83|84|all]"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
from shapely import contains_xy

import build_town_scene as BS
import plot_town_scene as PS
from lidar3d import render_pv as R, unified as U
from lidar3d.plates_pro import ProPlate

PROC = ROOT / "data" / "processed" / "town"
KICK = "White River scene  ·  vegetation risk"
SBX, SBW = PS.SBX, PS.SBW
EXP_BINS = [0, 1, 3, 10, 10 ** 9]
EXP_COL = ["#9AA3AD", "#F2C14E", "#E8743B", "#B3261E"]
EXP_LAB = ["no tree can reach", "1 to 2 trees", "3 to 9 trees", "10 or more trees"]
CLR_BINS = [0, 1.5, 3.0, 5.0, 1e9]
CLR_COL = ["#B3261E", "#E8743B", "#F2C14E", "#9AA3AD"]
CLR_LAB = ["under 1.5 m", "1.5 to 3 m", "3 to 5 m", "5 m or more"]


def analyse(sc):
    S, D, N = sc.S, sc.D, sc.N
    d = S["d"]
    gz = U.ground_fn(S["G"])
    I = sc.info["classified"]
    wires = I["wires_final"]
    yard = N["Y"]["yard"]
    # 1 m wire samples with the wire they belong to
    smp, wid = [], []
    for i, w in enumerate(wires):
        P = U.densify(np.asarray(w, float), 1.0)
        smp.append(P); wid.append(np.full(len(P), i))
    P = np.vstack(smp); wid = np.concatenate(wid)
    in_yard = contains_xy(yard.buffer(3), P[:, 0], P[:, 1])
    gw = gz(P[:, 0], P[:, 1])
    # trees
    T = D["trees"]
    tx, ty, H = np.asarray(T["x"], float), np.asarray(T["y"], float), np.asarray(T["height"], float)
    tg = gz(tx, ty)
    wt = cKDTree(P[:, :2])
    n_strike = np.zeros(len(P), int)
    best = np.full(len(P), -np.inf)
    t_strike = np.zeros(len(tx), bool)
    t_margin = np.full(len(tx), -np.inf)
    t_n = np.zeros(len(tx), int)
    for k in range(len(tx)):
        idx = wt.query_ball_point([tx[k], ty[k]], H[k])
        if not idx:
            continue
        idx = np.asarray(idx)
        dd = np.hypot(P[idx, 0] - tx[k], P[idx, 1] - ty[k])
        hh = np.maximum(P[idx, 2] - tg[k], 0.0)
        m = H[k] - np.hypot(dd, hh)
        hit = m >= 0
        if hit.any():
            n_strike[idx[hit]] += 1
            best[idx] = np.maximum(best[idx], m)
            t_strike[k] = True
            t_margin[k] = m.max()
            t_n[k] = int(np.unique(wid[idx[hit]]).size)
    # clearance to vegetation returns (returns within 1 m of a model wire are taken as wire, not vegetation)
    hag = S["hag"]
    veg = ~D["bmask"] & ~D["wmask"] & ~D["inyard"] & (d["cls"] != 2) & (hag >= 1.5)
    V = np.c_[d["x"][veg], d["y"][veg], d["z"][veg]].astype(float)
    near_w = wt.query(V[:, :2])[0] <= 1.5
    if near_w.any():
        dv = cKDTree(P).query(V[near_w])[0]
        drop = np.flatnonzero(near_w)[dv <= 1.0]
        V = np.delete(V, drop, axis=0)
    clr = cKDTree(V).query(P)[0]
    # wires recovered from rows of canopy tops were traced from those tops, so their clearance would measure the trees against themselves: left out
    rows = D.get("wire_rows", [])
    from_rows = np.zeros(len(P), bool)
    if rows:
        RP = np.vstack([U.densify(np.asarray(r, float), 0.5) for r in rows])
        from_rows = cKDTree(RP[:, :2]).query(P[:, :2])[0] <= 2.0
    # supports outside the yard
    K = N["K"]
    sup = K["sup"][K["sup"].tier == "support"]
    SP = np.c_[sup.x, sup.y, sup.top_z].astype(float)
    E = N["E"]
    if len(E):
        SP = np.vstack([SP, np.c_[E.x, E.y, gz(E.x.to_numpy(float), E.y.to_numpy(float)) + E.top_hag]])
    SP = SP[~contains_xy(yard.buffer(3), SP[:, 0], SP[:, 1])]
    tt = cKDTree(np.c_[tx, ty])
    s_n = np.zeros(len(SP), int); s_top = np.zeros(len(SP), int)
    for j, (sx, sy, sz) in enumerate(SP):
        idx = np.asarray(tt.query_ball_point([sx, sy], H.max()))
        if len(idx) == 0:
            continue
        dd = np.hypot(tx[idx] - sx, ty[idx] - sy)
        s_n[j] = int((dd <= H[idx]).sum())
        s_top[j] = int((np.hypot(dd, np.maximum(sz - tg[idx], 0)) <= H[idx]).sum())
    out_ = ~in_yard
    oc = out_ & ~from_rows
    exp_cls = np.digitize(n_strike, EXP_BINS[1:-1])
    clr_cls = np.digitize(clr, CLR_BINS[1:-1])
    st = dict(
        wire_km_outside_yard=float(out_.sum() / 1000), exposed_km=float(((n_strike > 0) & out_).sum() / 1000),
        exposed_share=float(((n_strike > 0) & out_).sum() / max(out_.sum(), 1)),
        km_by_exposure={EXP_LAB[c]: float(((exp_cls == c) & out_).sum() / 1000) for c in range(4)},
        km_by_clearance={CLR_LAB[c]: float(((clr_cls == c) & oc).sum() / 1000) for c in range(4)}, km_from_canopy_rows=float((out_ & from_rows).sum() / 1000),
        trees=int(len(tx)), trees_can_strike=int(t_strike.sum()), trees_can_strike_two_or_more_wires=int((t_n >= 2).sum()),
        supports=int(len(SP)), supports_reachable=int((s_n > 0).sum()), supports_top_reachable=int((s_top > 0).sum()),
        median_strike_tree_height_m=float(np.median(H[t_strike])) if t_strike.any() else None,
        min_clearance_m=float(clr[oc].min()), share_clear_under_3m=float(((clr < 3) & oc).sum() / max(oc.sum(), 1)))
    # the most exposed wire outside the yard (by number of distinct trees that can strike it)
    per_wire = {}
    for i in np.unique(wid[out_]):
        m = (wid == i) & out_
        per_wire[int(i)] = dict(len=int(m.sum()), max_n=int(n_strike[m].max()), exposed_m=int((n_strike[m] > 0).sum()))
    worst = max(per_wire, key=lambda i: (per_wire[i]["exposed_m"] * (per_wire[i]["len"] >= 60), per_wire[i]["max_n"]))
    st["worst_wire"] = dict(index=int(worst), **per_wire[worst])
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "scene_vegetation_risk.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1), flush=True)
    return dict(from_rows=from_rows, P=P, wid=wid, in_yard=in_yard, gw=gw, n=n_strike, best=best, clr=clr, exp_cls=exp_cls, clr_cls=clr_cls, tx=tx, ty=ty, H=H, tg=tg,
                t_strike=t_strike, t_margin=t_margin, SP=SP, s_n=s_n, s_top=s_top, st=st, wires=wires, gz=gz)


def plan(a, sc, A, colour_by="exp"):
    D, S, N = sc.D, sc.S, sc.N
    x0, x1, y0, y1 = BS.WIN
    ch = np.where(np.isfinite(D["chm"]), D["chm"], 0.0)
    a.imshow(np.ma.masked_less(ch, 2.0), origin="lower", extent=(x0, x1, y0, y1), cmap=ListedColormap(["#DCE6D5", "#C3D6BA", "#A9C59E", "#8DB383"]), vmin=2, vmax=30, interpolation="nearest", zorder=1)
    for b in D["buildings"]:
        fx, fy = b.footprint.exterior.xy
        a.fill(np.asarray(fx) + D["bld_offset"][0], np.asarray(fy) + D["bld_offset"][1], fc="#E4E1DB", ec="#C9C4BA", lw=0.3, zorder=2)
    yx, yy = N["Y"]["yard"].exterior.xy
    a.fill(yx, yy, fc="#EEECE8", ec="#9C978C", lw=0.6, zorder=2)
    cls = A["exp_cls"].copy() if colour_by == "exp" else A["clr_cls"].copy()
    if colour_by == "clr":
        cls[A["from_rows"]] = 4
    cols = EXP_COL if colour_by == "exp" else CLR_COL + ["#C9CDD2"]
    P, wid = A["P"], A["wid"]
    for i in np.unique(wid):
        m = np.flatnonzero(wid == i)
        q = P[m]; c = cls[m]
        cut = np.flatnonzero(np.diff(c)) + 1
        for s0, s1 in zip(np.r_[0, cut], np.r_[cut, len(q)]):
            seg = q[max(s0 - 1, 0):s1]
            k = c[s0]
            a.plot(seg[:, 0], seg[:, 1], color=cols[k], lw=0.7 + 0.45 * k if colour_by == "exp" else 0.7 + 0.45 * max(3 - k, 0), zorder=4 + (k if colour_by == "exp" else max(3 - k, 0)), solid_capstyle="round", ls=(0, (2, 1.5)) if k == 4 else "-")
    a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color("#B7BCC3"); sp.set_linewidth(0.6)
    from lidar3d.plates import north_arrow, scalebar
    scalebar(a, (x0, x1, y0, y1), 100.0); north_arrow(a, (x0, x1, y0, y1))


def run_83(sc, A):
    st = A["st"]
    pl = ProPlate(83, KICK, "Which lines and supports a falling tree could reach",
                  "Every lidar tree tested against every metre of wire in three dimensions: tree height, the wire's height above the tree's base, and the distance between them",
                  PS.src(sc.S), "Fall reach: sqrt(d² + h²) ≤ tree height")
    pl.panel_title(0.6, 8.14, "A", "Number of trees that could strike each metre of wire, and the trees that can", w=9.0)
    a = pl.ax(0.6, 1.0, 9.0, 6.95)
    plan(a, sc, A, "exp")
    ts = A["t_strike"]
    sc_ = a.scatter(A["tx"][ts], A["ty"][ts], s=3 + A["H"][ts] * 0.5, c=np.clip(A["t_margin"][ts], 0, 15), cmap="YlOrRd", vmin=0, vmax=15, edgecolors="#5a1d0e", linewidths=0.15, zorder=8)
    s_n = A["s_n"]; SP = A["SP"]
    a.scatter(SP[:, 0], SP[:, 1], s=26, marker="s", c=["#B3261E" if n >= 10 else "#E8743B" if n >= 3 else "#F2C14E" if n >= 1 else "#9AA3AD" for n in s_n], edgecolors="#15191D", linewidths=0.6, zorder=9)
    pl.panel_title(SBX, 8.14, "B", "Key numbers", w=SBW)
    y = pl.tiles(SBX, 7.85, SBW, [(f"{st['exposed_share'] * 100:.0f}%", f"of {st['wire_km_outside_yard']:.1f} km of wire outside the yard is within fall reach"),
                                  (f"{st['trees_can_strike']:,}", f"of {st['trees']:,} trees could strike a wire"),
                                  (f"{st['supports_reachable']}", f"of {st['supports']} line supports within a tree's reach")])
    y = pl.legend(SBX, y - 0.05, SBW, [("line", c, l) for c, l in zip(EXP_COL, EXP_LAB)] + [("square", "#E8743B", "support (colour = trees in reach)"), ("dot", "#E0542C", "tree that can strike (size = height, darker = more overreach)")], title="WIRES, SUPPORTS AND TREES")
    y = pl.notes(SBX, y - 0.08, SBW, [
        ("How reach is tested", "A tree is a rigid stem of its lidar height pivoting at its base. It reaches a point of wire at plan distance d and height h above the tree's base when the square root of d² + h² is no more than the tree's height, so a tall wire on a slope above a tree is harder to reach, and a tree upslope of a line is easier."),
        ("What it is not", "This is strike potential, not likelihood: lean, species, health, wind direction and rooting are not known from the lidar, and a tree may break above its base. Crowns are found by watershed segmentation, so two trees can be counted as one."),
        ("Supports", f"A support counts when a tree's height reaches its base: {st['supports_reachable']} of {st['supports']}; {st['supports_top_reachable']} have a tree tall enough to reach the top as well.")], fs=8.2)
    pl.panel_title(SBX, y - 0.02, "C", "Wire length by trees in reach (outside the yard)", w=SBW)
    b = pl.ax(SBX + 1.55, max(1.15, y - 1.55), SBW - 1.75, max(0.8, min(1.15, y - 0.55 - 1.15)))
    v = [st["km_by_exposure"][l] for l in EXP_LAB]
    b.barh(range(4)[::-1], v, color=EXP_COL)
    for i, x_ in zip(range(4)[::-1], v):
        b.text(x_ + max(v) * 0.02, i, f"{x_:.2f} km", va="center", fontsize=7.5)
    b.set_yticks(range(4)[::-1]); b.set_yticklabels(EXP_LAB, fontsize=7.5); b.set_xlim(0, max(v) * 1.3); b.tick_params(labelsize=7.5, colors="#5B6470"); b.spines[["top", "right"]].set_visible(False)
    pl.save("83_whiteriver_vegetation_fall_reach.png")


def risk_layers(sc, A):
    """The classified scene with crowns and wires recoloured by risk."""
    L = dict(sc.L["classified"])
    S, D = sc.S, sc.D
    T = D["trees"]
    t = A["t_strike"]
    m = np.clip(A["t_margin"], 0, 15) / 15
    cm = plt.get_cmap("YlOrRd")
    col = np.where(t[:, None], cm(0.35 + 0.6 * m)[:, :3] * 255, np.array([[168, 190, 160]]))
    crowns, trunks = R.crowns_mesh(D["chm"], D["tree_lab"], np.asarray(T["id"]), np.c_[T["x"], T["y"]], np.asarray(T["base"]), S["G"]["z"], BS.WIN[0], BS.WIN[2], 0.5, col)
    L["crowns"] = crowns; L["trunks"] = trunks
    for k in ("wires", "wires_lines", "support_markers", "canopy"):
        L.pop(k, None)
    out = [dict(mesh=m_, smooth=(k == "crowns")) if k != "frames" else dict(mesh=m_, kind="points", point_size=3.0) for k, m_ in L.items()]
    P, wid, cls = A["P"], A["wid"], A["exp_cls"]
    for c in range(4):
        segs = []
        for i in np.unique(wid):
            mm = np.flatnonzero(wid == i)
            q = P[mm]; cc = cls[mm] == c
            cut = np.flatnonzero(np.diff(cc.astype(int))) + 1
            for s0, s1 in zip(np.r_[0, cut], np.r_[cut, len(q)]):
                if cc[s0] and s1 - s0 >= 1:
                    segs.append(q[max(s0 - 1, 0):min(s1 + 1, len(q))])
        lm = R.lines_mesh(segs)
        if lm is not None:
            out.append(dict(mesh=lm, kind="lines", color=EXP_COL[c], width=2.2 + 0.8 * c))
    return out


def run_84(sc, A):
    st = A["st"]
    pl = ProPlate(84, KICK, "Tree-fall risk in three dimensions, and a span in profile",
                  "Crowns that can reach a wire in orange to red (darker = more overreach); wires coloured by how many trees can reach them; clearance to the nearest vegetation return",
                  PS.src(sc.S), "Strike potential only: lean, species and health are not known")
    layers = risk_layers(sc, A)
    pl.panel_title(0.6, 8.14, "A", "Classified scene, looking north-north-east", w=9.0)
    f, dist, az, el = sc.views["overview"]
    pl.image(R.render(layers, f, dist, az, el, (1920, 1050), bg="white", ssao=True), 0.6, 3.05, 9.0, 4.9)
    # profile of the most exposed wire
    w = A["st"]["worst_wire"]["index"]
    m = np.flatnonzero(A["wid"] == w)
    q = A["P"][m]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(q[:, :2], axis=0).T))]
    gw = A["gw"][m]
    pl.panel_title(0.6, 2.75, "B", f"The most exposed span outside the yard in profile ({s[-1]:.0f} m)", w=9.0)
    b = pl.ax(1.1, 1.15, 8.4, 1.3)
    b.fill_between(s, gw.min() - 2, gw, color="#E4E1DB", lw=0)
    b.plot(s, gw, color="#8A847A", lw=0.8)
    b.plot(s, q[:, 2], color="#15191D", lw=1.4)
    tr = cKDTree(q[:, :2])
    dd, ii = tr.query(np.c_[A["tx"], A["ty"]])
    near = dd <= np.maximum(A["H"], 5)
    for k in np.flatnonzero(near):
        hit = A["t_strike"][k] and dd[k] <= A["H"][k]
        b.plot([s[ii[k]], s[ii[k]]], [A["tg"][k], A["tg"][k] + A["H"][k]], color="#E0542C" if hit else "#8DB383", lw=1.6 if hit else 1.0, alpha=0.9, solid_capstyle="butt")
    b.set_xlim(0, s[-1]); b.set_ylim(gw.min() - 2, max(q[:, 2].max(), (A["tg"][near] + A["H"][near]).max() if near.any() else 0) + 3)
    b.set_xlabel("distance along the wire, m", fontsize=7.5, color="#5B6470"); b.set_ylabel("elevation, m", fontsize=7.5, color="#5B6470")
    b.tick_params(labelsize=7.5, colors="#5B6470"); b.spines[["top", "right"]].set_visible(False)
    pl.panel_title(SBX, 8.14, "C", "Clearance to the nearest vegetation", w=SBW)
    a2 = pl.ax(SBX, 4.25, SBW, 3.65)
    plan(a2, sc, A, "clr")
    y = pl.legend(SBX, 4.1, SBW, [("line", c, l) for c, l in zip(CLR_COL, CLR_LAB)] + [("dash", "#C9CDD2", "traced from canopy rows: not measured")], title="3D DISTANCE FROM WIRE TO THE NEAREST VEGETATION RETURN")
    y = pl.notes(SBX, y - 0.06, SBW, [
        ("Clearance", f"{st['share_clear_under_3m'] * 100:.0f}% of the wire outside the yard has a vegetation return within 3 m in 3D (closest {st['min_clearance_m']:.1f} m). Returns within 1 m of a modelled wire are taken as wire, so contact closer than 1 m cannot be seen. The {st['km_from_canopy_rows']:.1f} km of wire traced from rows of canopy tops is left out (dashed)."),
        ("Profile", "Wire (black) over ground, with every tree whose base lies within its own height of the wire drawn at full lidar height: orange can reach the wire, green cannot because it is too short for the distance and the wire's height.")], fs=8.1)
    pl.save("84_whiteriver_vegetation_risk_3d.png")


if __name__ == "__main__":
    sc = PS.Scene()
    A = analyse(sc)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("83", "all"):
        run_83(sc, A)
    if which in ("84", "all"):
        run_84(sc, A)
