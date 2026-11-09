"""Close-up 3D models at the White River substation and its lines: the yard (posts, bus wires, equipment blocks), one transformer, and poles with their fitted wires. Plates 66 to 68 and GLB models in outputs/utility3d.
Usage: python scripts/build_town_closeups.py   (fetches a 580 m by 260 m window of 3DEP points once and caches it)"""
import json
import pickle
import sys
import textwrap
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
import trimesh
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely import contains_xy

import build_town as BT
import build_town_model as BM
from lidar3d import closeups as CU
from lidar3d import conductors as C
from lidar3d import cloud, infra3d, render3d, spans, utility
from lidar3d.ept import CACHE, EPT
from lidar3d.plates import MUTED, Plate

WIN = (-450.0, 130.0, -130.0, 130.0)
OUT3D = ROOT / "outputs" / "utility3d"
PROC = ROOT / "data" / "processed" / "town"
STEEL, WIRE, BLOCK = (140, 146, 152), (35, 38, 44), (74, 107, 140)


def load_window():
    p = CACHE / "town" / "closeup_points.pkl"
    if not p.exists():
        from pyproj import Transformer
        tr = Transformer.from_crs(4326, BT.CRS, always_xy=True); cx, cy = tr.transform(*BT.LONLAT)
        back = Transformer.from_crs(BT.CRS, 4326, always_xy=True)
        x0, x1, y0, y1 = WIN
        W, S = back.transform(cx + x0, cy + y0); E, N = back.transform(cx + x1, cy + y1)
        c = EPT(BT.PROJECT).fetch((W, S, E, N), max_depth=20, log=print)
        k = (c.x >= cx + x0) & (c.x < cx + x1) & (c.y >= cy + y0) & (c.y < cy + y1) & (c.cls != 7)
        p.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(dict(x=c.x[k] - cx, y=c.y[k] - cy, z=c.z[k], cls=c.cls[k], inten=c.intensity[k], ret=c.ret[k], nret=c.nret[k], cx=cx, cy=cy), open(p, "wb"))
    return pickle.load(open(p, "rb"))


def lit(gg, x0, y0, res=1.0):
    hs = cloud.hillshade(np.nan_to_num(gg, nan=float(np.nanmedian(gg))), res, 315, 40)
    ny, nx = gg.shape
    X, Y = np.meshgrid(np.arange(nx) * res + x0 + res / 2, np.arange(ny) * res + y0 + res / 2)
    g = (110 + 110 * hs).astype(float)
    return np.c_[X.ravel(), Y.ravel(), gg.ravel()], np.repeat(g.ravel()[:, None], 3, 1) * np.array([[1.0, 1.0, 0.96]])


def hag_colors(h, top=24.0):
    return (plt.get_cmap("turbo")(np.clip(h / top, 0, 1))[:, :3] * 255).astype(np.uint8)


def mesh_points(m, density=300.0):
    p, c = infra3d.sample_mesh(m, density)
    return p, c.astype(float)


def panel(pl, im, l, b, w, h, title=None, tl=None):
    a = pl.ax(l, b, w, h)
    a.imshow(im); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color("#999999")
    if title:
        pl.label(l, b + h + 0.15, title)
    return a


def rows(pl, x, y0, items, width, fs=8.2):
    for k_, v_ in items:
        pl.text(x, y0, k_, fontsize=9, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v_, width))
        pl.text(x, y0 - 0.24, w, fontsize=fs, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.27 + 0.16 * (w.count("\n") + 1) + 0.14
    return y0


def main():
    OUT3D.mkdir(parents=True, exist_ok=True)
    d = load_window()
    x, y, z, cls = d["x"], d["y"], d["z"], d["cls"]
    xyz = np.c_[x, y, z]
    x0, x1, y0, y1 = WIN
    hag, gg = utility.height_above_ground(x, y, z, cls, (x0, y0, x1, y1), 1.0)
    gpts, gcol = lit(gg, x0, y0)
    wmask, wlab, winfo = utility.detect_wires(xyz, hag)
    feats = BM.overture(d["cx"], d["cy"])
    yard = max(feats["substation"], key=lambda g: g.area)
    tmap = np.array([(g.x, g.y) for g in feats["transformer"]])
    src = (f"Lidar: USGS 3DEP {BT.PROJECT}, flown 2020-04-13 (about 25 returns per m2). Models are fitted to the returns: cylinders for posts, tubes for wires, oriented boxes for equipment blocks; sizes are measured, shapes assumed. "
           "Mapped features: Overture Maps.")
    out = {}
    # ------------------------------------------------------------------ facility
    posts = utility.detect_poles(xyz, hag, exclude=wmask, min_hag=1.5, min_height=2.5, max_radius=0.8, ring_ratio=None)
    inyard = lambda px, py: contains_xy(yard.buffer(6), np.atleast_1d(px), np.atleast_1d(py))
    posts = [p for p in posts if inyard(p["x"], p["y"])[0] and p["top_hag"] <= 32.0]
    near_post = cKDTree(np.array([(p["x"], p["y"]) for p in posts])).query(xyz[:, :2])[0] <= 0.9
    ex = wmask | near_post
    cx_, cy_ = yard.centroid.x, yard.centroid.y
    blocks = CU.equipment_blocks(xyz, hag, ex, (cx_, cy_), half=110.0, min_pts=120, min_dim=1.2, max_h=7.0)
    blocks = [b for b in blocks if inyard(b["centre"][0], b["centre"][1])[0] and b["size"][0] < 25 and b["size"][1] < 25]
    wires_y = [p for p in CU.wire_polylines(xyz[wmask & contains_xy(yard.buffer(6), x, y)], wlab[wmask & contains_xy(yard.buffer(6), x, y)]) if len(p) >= 3]
    ginv = RegularGridInterpolator((np.arange(gg.shape[0]) + 0.5 + y0, np.arange(gg.shape[1]) + 0.5 + x0), gg, bounds_error=False, fill_value=None)
    gz = lambda px, py: ginv(np.c_[np.atleast_1d(py), np.atleast_1d(px)])
    post_m = [infra3d.tube(np.array([[p["x"], p["y"], float(gz(p["x"], p["y"])[0])], [p["x"], p["y"], float(gz(p["x"], p["y"])[0]) + p["top_hag"]]]), 0.22, 6) for p in posts]
    for b in blocks:
        b["centre"] = b["centre"] + np.array([0, 0, float(gz(b["centre"][0], b["centre"][1])[0])])
    meshes = dict(posts=infra3d._merge(post_m, STEEL), wires=CU.tube_mesh(wires_y, 0.05, WIRE), blocks=infra3d._merge([CU.box_part(b) for b in blocks], BLOCK) if blocks else None)
    meshes = {k: v for k, v in meshes.items() if v is not None}
    sc = trimesh.Scene()
    for k, m in meshes.items():
        sc.add_geometry(m, geom_name=k)
    sc.export(OUT3D / "white_river_yard_model.glb")
    win = (np.abs(x - cx_) <= 110) & (np.abs(y - cy_) <= 110)
    gwin = (np.abs(gpts[:, 0] - cx_) <= 110) & (np.abs(gpts[:, 1] - cy_) <= 110)
    ep = win & (hag >= 0.5)
    pts_raw = np.vstack([gpts[gwin], xyz[ep]]); col_raw = np.vstack([gcol[gwin], hag_colors(hag[ep], 24)])
    mp, mc = [], []
    for k, m in meshes.items():
        p, c = mesh_points(m, 120.0); mp.append(p); mc.append(c)
    pts_mod = np.vstack([gpts[gwin]] + mp); col_mod = np.vstack([gcol[gwin]] + mc)
    ims = []
    for az, el in ((215, 30), (40, 24)):
        a, info = render3d.render(pts_raw, col_raw, az=az, el=el, width=1700, height=900, splat=2)
        b, _ = render3d.render(pts_mod, np.clip(col_mod, 0, 255).astype(np.uint8), az=az, el=el, width=1700, height=900, splat=2, center=info["center"], scale=info["scale"], off=info["off"])
        ims.append((a, b))
    out["yard"] = dict(area_m2=float(yard.area), posts=len(posts), bus_wire_polylines=len(wires_y), equipment_blocks=len(blocks), block_sizes_m=[[round(float(v), 1) for v in b["size"]] for b in blocks[:12]], median_post_height_m=float(np.median([p["top_hag"] for p in posts])), max_post_height_m=float(max(p["top_hag"] for p in posts)))
    pl = Plate("The substation yard in 3D", "Lidar returns coloured by height above ground, and the model fitted to them: posts and gantry legs, bus wires and equipment blocks", src, "Model view uses the same camera as the returns view")
    panel(pl, ims[0][0], 0.6, 4.6, 7.2, 3.6, "Lidar returns, from the south-west")
    panel(pl, ims[0][1], 8.2, 4.6, 7.2, 3.6, "Fitted model, same view")
    panel(pl, ims[1][1], 0.6, 0.95, 7.2, 3.3, "Fitted model, from the north-east")
    y_ = out["yard"]
    rows(pl, 8.3, 4.0, [
        ("What is in the model", f"{y_['posts']} posts and gantry legs (cylinders, median top {y_['median_post_height_m']:.1f} m, tallest {y_['max_post_height_m']:.1f} m above ground), {y_['bus_wire_polylines']} bus wires and conductors (tubes) and {y_['equipment_blocks']} equipment blocks (boxes of at least 1.2 m in every dimension and 120 returns) inside a {y_['area_m2'] / 1000:.0f} thousand m2 yard."),
        ("How to read it", "Colours in the first view are height above ground, blue near the ground and red at the top of the gantries. The model drops everything the fitted shapes do not explain, so small apparatus and fences are missing from it and it is sparser than the yard really is."),
        ("Limits", "A box is the bounding volume of connected returns, not the shape of a transformer or breaker: neighbouring apparatus that touches in the 0.5 m voxel grid is merged into one block, so some boxes are larger than any single device, and thin lattice frames are only partly captured by the cylinders. Insulators, cross-arms and cable are not modelled here. No imagery colours the model yet.")], 90)
    pl.save("66_whiteriver_yard_3d.png")
    # ------------------------------------------------------------------ transformer
    t_rows = []
    best = None
    for q in tmap:
        m6 = CU.equipment_blocks(xyz, hag, wmask | near_post, q, half=14.0, min_pts=60, min_dim=1.2, max_h=6.0)
        m6 = [b for b in m6 if b["size"][0] < 20 and b["size"][1] < 20]
        if not m6:
            t_rows.append(dict(mapped_xy=q.tolist(), block=None)); continue
        b = min(m6, key=lambda b: np.hypot(*(b["centre"][:2] - q)))
        t_rows.append(dict(mapped_xy=[float(q[0]), float(q[1])], block=dict(size_m=[round(float(v), 1) for v in b["size"]], points=b["n"], offset_m=round(float(np.hypot(*(b["centre"][:2] - q))), 1), top_hag_m=round(float(b["z1"]), 1))))
        if b["n"] >= 800 and (best is None or b["size"][0] * b["size"][1] < best[1]["size"][0] * best[1]["size"][1]):
            best = (q, b)
    q, tb = best
    w = (np.abs(x - q[0]) <= 15) & (np.abs(y - q[1]) <= 15)
    tw = w & (hag >= 0.4)
    gw = (np.abs(gpts[:, 0] - q[0]) <= 15) & (np.abs(gpts[:, 1] - q[1]) <= 15)
    g2 = w & (cls == 2)
    gp = np.c_[xyz[g2]]; gc = np.full((len(gp), 3), 170.0)
    tb["centre"] = tb["centre"] + np.array([0, 0, float(gz(tb["centre"][0], tb["centre"][1])[0])])
    bm = infra3d._merge([CU.box_part(tb)], BLOCK)
    near_t = (np.abs(x - q[0]) <= 15) & (np.abs(y - q[1]) <= 15)
    posts_t = [p for p in posts if abs(p["x"] - q[0]) <= 15 and abs(p["y"] - q[1]) <= 15]
    legs = infra3d._merge([infra3d.tube(np.array([[p["x"], p["y"], float(gz(p["x"], p["y"])[0])], [p["x"], p["y"], float(gz(p["x"], p["y"])[0]) + p["top_hag"]]]), 0.22, 6) for p in posts_t], STEEL) if posts_t else None
    wt = [pp for pp in CU.wire_polylines(xyz[wmask & near_t], wlab[wmask & near_t]) if len(pp) >= 3]
    wm = CU.tube_mesh(wt, 0.05, WIRE) if wt else None
    sc = trimesh.Scene(); sc.add_geometry(bm, geom_name="transformer_block")
    for nm, m in (("legs", legs), ("wires", wm)):
        if m is not None:
            sc.add_geometry(m, geom_name=nm)
    sc.export(OUT3D / "white_river_transformer_model.glb")
    yaw_ = tb["yaw"]; Rz = np.array([[np.cos(yaw_), -np.sin(yaw_)], [np.sin(yaw_), np.cos(yaw_)]])
    half_ = tb["size"][:2] / 2
    gcen = float(gz(tb["centre"][0], tb["centre"][1])[0])
    corners = [np.r_[tb["centre"][:2] + Rz @ (np.array([sx, sy]) * half_), gcen + zz] for sx in (-1, 1) for sy in (-1, 1) for zz in (tb["z0"], tb["z1"])]
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    frame = infra3d._merge([infra3d.tube(np.array([corners[i], corners[j]]), 0.08, 4) for i, j in edges], (20, 20, 25))
    mp2, mc2 = [gp], [gc]
    for m, dens in ((frame, 900.0), (legs, 200.0), (wm, 200.0)):
        if m is not None:
            p, c = mesh_points(m, dens); mp2.append(p); mc2.append(c)
    pts_a = np.vstack([gp, xyz[tw & (hag < 26)]]); col_a = np.vstack([gc, hag_colors(hag[tw & (hag < 26)], 26)])
    pts_b = np.vstack(mp2 + [xyz[tw & (hag < 26)]]); col_b = np.clip(np.vstack(mc2 + [np.full((int((tw & (hag < 26)).sum()), 3), 150.0)]), 0, 255).astype(np.uint8)
    imgs = []
    for az, el in ((200, 24), (120, 18)):
        a, info = render3d.render(pts_a, col_a.astype(np.uint8), az=az, el=el, width=1700, height=1000, splat=3)
        b, _ = render3d.render(pts_b, col_b, az=az, el=el, width=1700, height=1000, splat=3, center=info["center"], scale=info["scale"], off=info["off"])
        imgs.append((a, b))
    # orthographic plan and side with the box outline
    fig2 = plt.figure(figsize=(7.2, 3.4), dpi=150)
    pp = xyz[w & (hag > 0.5) & (hag < 7.5)]; hh = hag[w & (hag > 0.5) & (hag < 7.5)]
    a1 = fig2.add_axes((0.03, 0.12, 0.40, 0.8)); a2 = fig2.add_axes((0.56, 0.12, 0.42, 0.8))
    a1.scatter(pp[:, 0], pp[:, 1], c=hh, s=2.5, cmap="turbo", vmin=0, vmax=8); a1.set_aspect("equal"); a1.set_title("plan, metres", fontsize=8)
    c0, sz, yaw = tb["centre"], tb["size"], tb["yaw"]
    cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * sz[:2] / 2
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    ob = cs @ R.T + c0[:2]
    a1.plot(ob[:, 0], ob[:, 1], color="k", lw=1.2); a1.scatter([q[0]], [q[1]], marker="x", color="red", s=40, zorder=9)
    a1.set_xlim(q[0] - 12, q[0] + 12); a1.set_ylim(q[1] - 12, q[1] + 12); a1.tick_params(labelsize=7)
    u = np.array([np.cos(yaw), np.sin(yaw)])
    s_ = (pp[:, :2] - c0[:2]) @ u
    a2.scatter(s_, hh, c=hh, s=2.5, cmap="turbo", vmin=0, vmax=8); a2.plot([-sz[0] / 2, sz[0] / 2, sz[0] / 2, -sz[0] / 2, -sz[0] / 2], [tb["z0"], tb["z0"], tb["z1"], tb["z1"], tb["z0"]], color="k", lw=1.2)
    a2.set_aspect("equal"); a2.set_title("side along the long axis, metres above ground", fontsize=8); a2.set_xlim(-12, 12); a2.set_ylim(-0.5, 8.5); a2.tick_params(labelsize=7)
    fig2.canvas.draw(); ortho = np.asarray(fig2.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig2)
    out["transformer"] = dict(mapped_xy=q.tolist(), box_size_m=[round(float(v), 1) for v in tb["size"]], top_above_ground_m=round(float(tb["z1"]), 1), points=tb["n"], offset_from_mapped_point_m=round(float(np.hypot(*(tb["centre"][:2] - q))), 1), all_mapped=t_rows)
    tr_ = out["transformer"]
    pl = Plate("A transformer in 3D", "The most compact of the equipment blocks at the three mapped yard transformer positions: the lidar returns around it and the box fitted to them", src, "Red cross in the plan view: the mapped transformer position")
    panel(pl, imgs[0][0], 0.6, 4.6, 7.2, 3.6, "Lidar returns, height above ground")
    panel(pl, imgs[0][1], 8.2, 4.6, 7.2, 3.6, "Fitted box (black edges), legs and wires over the returns")
    panel(pl, ortho, 0.6, 0.95, 7.2, 3.4, "Plan and side with the box outline")
    rows(pl, 8.3, 4.0, [
        ("Measured", f"The block is {tr_['box_size_m'][0]:.1f} m by {tr_['box_size_m'][1]:.1f} m in plan and {tr_['box_size_m'][2]:.1f} m tall (top {tr_['top_above_ground_m']:.1f} m above ground), from {tr_['points']:,} returns, {tr_['offset_from_mapped_point_m']:.1f} m from the mapped position. Of the three mapped transformers, " + "; ".join(f"{'none found' if r['block'] is None else str(r['block']['size_m'][0]) + ' x ' + str(r['block']['size_m'][1]) + ' x ' + str(r['block']['size_m'][2]) + ' m, ' + str(r['block']['points']) + ' returns'}" for r in t_rows) + "."),
        ("What it is and is not", "A bulky block of returns about 4.5 m tall at the foot of a steel gantry, around the mapped transformer position. The lidar gives an envelope, not the tank, radiators or bushings, and the footprint is that of an equipment bay (a transformer with its radiators and neighbouring apparatus, or a bank of units), so the box is not the dimensions of one transformer."),
        ("Limits", "Ground returns inside the footprint are few, and the top of a transformer is only a few returns thick at this density. The size is the 1st to 99th percentile of height in the block. The type and rating cannot be read from the lidar.")], 90)
    pl.save("67_whiteriver_transformer_3d.png")
    # ------------------------------------------------------------------ poles and lines
    cw = (x >= -440) & (x <= -50) & (y >= -70) & (y <= 70)
    xyzc, hagc = xyz[cw], hag[cw]
    wm_c, wl_c, wi_c = wmask[cw], wlab[cw], None
    poles = utility.detect_poles(xyzc, hagc, exclude=wm_c)
    chains = [xyzc[wl_c == c] for c in np.unique(wl_c[wl_c >= 0]) if (wl_c == c).sum() >= 8]
    hags = [hagc[wl_c == c] for c in np.unique(wl_c[wl_c >= 0]) if (wl_c == c).sum() >= 8]
    cond = C.build_conductors(chains, hags)
    sag = C.merge_supports([c["support_xy"] for c in cond], 6.0)
    ends = []
    for ch in chains:
        qd = ch[:, :2] - ch[:, :2].mean(0); pr = qd @ np.linalg.svd(qd, full_matrices=False)[2][0]; ends += [ch[pr.argmin(), :2], ch[pr.argmax(), :2]]
    A = spans.anchor_candidates(sag[:, :2] if len(sag) else np.zeros((0, 2)), np.array([(p["x"], p["y"]) for p in poles]).reshape(-1, 2), np.array(ends))
    P, _ = spans.candidate_points(xyzc, hagc)
    prs = spans.pairs(A, max_neighbours=7)
    tree = cKDTree(P[:, :2])
    res = []
    for i, j in prs:
        a, b = A[i], A[j]
        idx = tree.query_ball_point((a + b) / 2, np.hypot(*(b - a)) / 2 + 12.0)
        if len(idx) < 25:
            continue
        for wv in spans.fit_pair(P[idx], a, b, None, corridor=8.0, max_wires=14, rng=np.random.default_rng(1), min_cov=0.5, min_n=25):
            res.append((i, j, wv))
    ground_s = lambda a, b: gz(a, b)[0] if np.ndim(a) == 0 else gz(a, b)
    sup, tiers = infra3d.supports_from_spans(A, res, ground_s)
    sup["through_wires"] = infra3d.through_wire_counts(res, sup)
    sup["tier"] = np.where(sup.through_wires == 0, "support", "attachment")
    sm = infra3d.support_meshes(sup, tiers)
    wmsh = infra3d.wire_mesh(res, 2.0, 0.05)
    sc = trimesh.Scene()
    for k, m in sm.items():
        sc.add_geometry(m, geom_name=k)
    sc.add_geometry(wmsh, geom_name="wires")
    sc.export(OUT3D / "white_river_poles_lines_model.glb")
    # choose the stretch with the most supports in a 170 m by 80 m box
    S_ = sup[sup.tier == "support"][["x", "y"]].to_numpy()
    cand = [(np.sum((np.abs(S_[:, 0] - a_) <= 85) & (np.abs(S_[:, 1] - b_) <= 40)), a_, b_) for a_ in np.arange(-380, -110, 20) for b_ in np.arange(-30, 31, 10)]
    n_s, ax_, ay_ = max(cand)
    box = (np.abs(xyzc[:, 0] - ax_) <= 90) & (np.abs(xyzc[:, 1] - ay_) <= 45)
    gsel = (np.abs(gpts[:, 0] - ax_) <= 90) & (np.abs(gpts[:, 1] - ay_) <= 45)
    ep = box & (hagc >= 0.4) & (hagc < 40)
    pts_a = np.vstack([gpts[gsel], xyzc[ep]]); col_a = np.vstack([gcol[gsel], hag_colors(hagc[ep], 20)])
    mp3, mc3 = [gpts[gsel]], [gcol[gsel]]
    for m in list(sm.values()) + [wmsh]:
        p, c = mesh_points(m, 200.0)
        k_ = (np.abs(p[:, 0] - ax_) <= 90) & (np.abs(p[:, 1] - ay_) <= 45)
        mp3.append(p[k_]); mc3.append(c[k_])
    pts_b = np.vstack(mp3); col_b = np.clip(np.vstack(mc3), 0, 255).astype(np.uint8)
    views = []
    for az, el, wdt in ((200, 20, 1700), (110, 16, 1700)):
        a, info = render3d.render(pts_a, col_a.astype(np.uint8), az=az, el=el, width=wdt, height=900, splat=2)
        b, _ = render3d.render(pts_b, col_b, az=az, el=el, width=wdt, height=900, splat=2, center=info["center"], scale=info["scale"], off=info["off"])
        views.append((a, b))
    # a profile of the best-observed span
    best_w = max(res, key=lambda r: r[2]["coverage"] * r[2]["L"])[2] if res else None
    prof = None
    if best_w is not None:
        fig3 = plt.figure(figsize=(7.2, 3.4), dpi=150); ax3 = fig3.add_axes((0.08, 0.15, 0.9, 0.78))
        xy, fl = spans.wire_xyz(best_w, 1.0)
        ss = np.hypot(*(xy[:, :2] - xy[0, :2]).T)
        ax3.plot(ss[fl == 1], xy[fl == 1, 2], ".", ms=2.5, color="#1F4E79", label="observed (returns support it)"); ax3.plot(ss[fl != 1], xy[fl != 1, 2], ".", ms=2.5, color="#17A2B8", label="filled by the fitted curve")
        ax3.set_xlabel("distance along the span, m", fontsize=8); ax3.set_ylabel("wire elevation, m (NAVD88)", fontsize=8); ax3.tick_params(labelsize=7); ax3.legend(fontsize=7, frameon=False); ax3.spines[["top", "right"]].set_visible(False)
        fig3.canvas.draw(); prof = np.asarray(fig3.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig3)
    sx = sup[(sup.tier == "support") & (np.abs(sup.x - ax_) <= 90) & (np.abs(sup.y - ay_) <= 45)]
    out["poles_lines"] = dict(window_centre=[float(ax_), float(ay_)], supports_in_view=int(len(sx)), poles_in_view=int((sx.kind == "pole").sum()), towers_in_view=int((sx.kind == "tower").sum()), spans_total=len(res), median_support_height_m=float(sx.height.median()) if len(sx) else None,
                              span_lengths_m=[round(float(w["L"])) for _, _, w in res][:20], best_span_coverage=float(best_w["coverage"]) if best_w else None, best_span_length_m=float(best_w["L"]) if best_w else None)
    pl_ = out["poles_lines"]
    pl = Plate("Poles and lines in 3D", "A stretch of the lines west of the yard: lidar returns coloured by height, and the supports, cross-arms, insulators and wires fitted to them", src, "Cross-arm and insulator sizes are assumed typical values; their heights and widths come from where the wires attach")
    panel(pl, views[0][0], 0.6, 4.6, 7.2, 3.6, "Lidar returns, from the south-south-west")
    panel(pl, views[0][1], 8.2, 4.6, 7.2, 3.6, "Fitted model, same view")
    if prof is not None:
        panel(pl, prof, 0.6, 0.95, 7.2, 3.4, "One span: observed wire and the filled part")
    rows(pl, 8.3, 4.0, [
        ("In the view", f"{pl_['supports_in_view']} supports ({pl_['poles_in_view']} wood poles, {pl_['towers_in_view']} steel structures) with a median height of {pl_['median_support_height_m']:.1f} m, joined by wires fitted with a sag curve between their attachment points. Wires are drawn as 10 cm tubes."),
        ("The profile", f"The span shown is {pl_['best_span_length_m']:.0f} m long with {pl_['best_span_coverage'] * 100:.0f}% of its length supported by returns; the rest is the fitted curve. In the held-out test a hidden 20 to 40 m stretch is recovered to 4 to 5 cm rms."),
        ("Limits", "Pole radius, cross-arm cross-section and insulator length are assumed, not measured. Transformers, switches and cables on poles are not modelled. Only the fitted spans are drawn; a line the model did not fit does not appear. No imagery colours the model yet.")], 90)
    pl.save("68_whiteriver_poles_lines_3d.png")
    # a close view of one wood pole and its wires
    wood = sup[(sup.tier == "support") & (sup.kind == "pole") & (np.abs(sup.x - ax_) <= 120)]
    if len(wood):
        pr_ = wood.sort_values("n_wires", ascending=False).iloc[0]
        px_, py_ = float(pr_.x), float(pr_.y)
        hw = 14.0
        bx = (np.abs(xyzc[:, 0] - px_) <= hw) & (np.abs(xyzc[:, 1] - py_) <= hw) & (hagc >= 0.3) & (hagc < 40)
        gb = (np.abs(gpts[:, 0] - px_) <= hw) & (np.abs(gpts[:, 1] - py_) <= hw)
        pa = np.vstack([gpts[gb], xyzc[bx]]); ca = np.vstack([gcol[gb], hag_colors(hagc[bx], 16)])
        mpp, mcc = [gpts[gb]], [gcol[gb]]
        for m in list(sm.values()) + [wmsh]:
            p_, c_ = mesh_points(m, 1500.0)
            k_ = (np.abs(p_[:, 0] - px_) <= hw) & (np.abs(p_[:, 1] - py_) <= hw)
            mpp.append(p_[k_]); mcc.append(c_[k_])
        pb = np.vstack(mpp); cb = np.clip(np.vstack(mcc), 0, 255).astype(np.uint8)
        ims2 = []
        for az, el in ((200, 12), (110, 12)):
            a, info = render3d.render(pa, ca.astype(np.uint8), az=az, el=el, width=1500, height=1100, splat=3)
            b, _ = render3d.render(pb, cb, az=az, el=el, width=1500, height=1100, splat=3, center=info["center"], scale=info["scale"], off=info["off"])
            ims2.append((a, b))
        out["pole_closeup"] = dict(x=px_, y=py_, height_m=float(pr_.height), wires_attached=int(pr_.n_wires), cross_arm_tiers=len(tiers[list(sup.index).index(pr_.name)]) if len(tiers) == len(sup) else None, nearby_returns=int(bx.sum()))
        pc = out["pole_closeup"]
        pl = Plate("A pole and its wires up close", "The wood pole in the corridor with the most attached wires: lidar returns and the fitted pole, cross-arm tiers, insulators and wires, from two directions", src, "Cross-arm, insulator and pole sizes are assumed typical values; heights and widths come from where the wires attach")
        panel(pl, ims2[0][0], 0.6, 4.6, 7.2, 3.6, "Lidar returns, from the south-south-west")
        panel(pl, ims2[0][1], 8.2, 4.6, 7.2, 3.6, "Fitted model, same view")
        panel(pl, ims2[1][0], 0.6, 0.95, 7.2, 3.4, "Lidar returns, from the east")
        panel(pl, ims2[1][1], 8.2, 0.95, 7.2, 3.4, "Fitted model, same view")
        pl.text(0.6, 0.85, f"Pole {pc['height_m']:.1f} m tall with {pc['wires_attached']} wires attached; {pc['nearby_returns']:,} returns inside the 28 m box.", fontsize=8.5, color=MUTED)
        pl.save("69_whiteriver_pole_closeup_3d.png")
    (PROC / "closeups.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
