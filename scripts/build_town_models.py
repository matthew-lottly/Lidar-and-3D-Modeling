"""Improved close-up models (replaces scripts/build_town_closeups.py): hole-free ground draped with NAIP imagery, every detected wire in the model (fitted spans plus the observed chains the spans missed, gaps filled by a sag curve),
every support (span supports, detected poles, towers fitted to their own returns), and the yard. Measures what the old model left out. Plates 66 to 70 and GLB models.
Usage: xvfb-run -a python scripts/build_town_models.py   (needs a display for PyVista; xvfb-run provides one)"""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from shapely import contains_xy

import build_town as BT
import build_town_closeups as BC
import build_town_model as BM
from lidar3d import closeups as CU
from lidar3d import conductors as C
from lidar3d import infra3d, naip, spans, unified as U, utility
from lidar3d.aoi import AOI
from lidar3d.ept import CACHE

OUT3D = ROOT / "outputs" / "utility3d"
PROC = ROOT / "data" / "processed" / "town"
WIN = (-450.0, 130.0, -130.0, 130.0)
STEEL, WIRE, WOOD, GREY = (150, 156, 162), (30, 32, 36), (112, 82, 56), (178, 182, 186)


def prepare():
    p = CACHE / "town" / "models_v2.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    d = BC.load_window()
    x, y, z, cls = d["x"], d["y"], d["z"], d["cls"]
    xyz = np.c_[x, y, z]
    G = U.ground_surface(x, y, z, cls, (WIN[0], WIN[2], WIN[1], WIN[3]), 0.5)
    gz = U.ground_fn(G)
    hag = z - gz(x, y)
    # imagery on the ground grid
    img, tf, nd = naip.fetch(AOI(round(d["cx"] - 160), round(d["cy"]), 300, BT.CRS), 2019, res=0.6, slug="whiteriver_window")
    img = np.asarray(img)
    img = img / 256.0 if img.max() > 255 else img
    ny, nx = G["z"].shape
    X = np.arange(nx) * G["res"] + G["x0"] + G["res"] / 2
    Y = np.arange(ny) * G["res"] + G["y0"] + G["res"] / 2
    XX, YY = np.meshgrid(X, Y)
    samp = lambda px, py: np.stack([img[b][((py + d["cy"] - tf.f) / tf.e).astype(int).clip(0, img.shape[1] - 1), ((px + d["cx"] - tf.c) / tf.a).astype(int).clip(0, img.shape[2] - 1)] for b in range(3)], -1).astype(np.uint8)
    rgb = samp(XX, YY)
    wmask, wlab, winfo = utility.detect_wires(xyz, hag)
    out = dict(d=d, G=G, hag=hag, rgb=rgb, naip_date=nd, wmask=wmask, wlab=wlab, samp_args=None)
    p.write_bytes(pickle.dumps(out))
    return out


def corridor(S):
    d, G, hag, wmask, wlab = S["d"], S["G"], S["hag"], S["wmask"], S["wlab"]
    x, y, z = d["x"], d["y"], d["z"]
    xyz = np.c_[x, y, z]
    gz = U.ground_fn(G)
    cw = (x >= -440) & (x <= -50) & (y >= -70) & (y <= 70)
    xc, hc, wm, wl = xyz[cw], hag[cw], wmask[cw], wlab[cw]
    poles = utility.detect_poles(xc, hc, exclude=wm)
    ids = [c for c in np.unique(wl[wl >= 0]) if (wl == c).sum() >= 8]
    chains = [xc[wl == c] for c in ids]
    hags = [hc[wl == c] for c in ids]
    cond = C.build_conductors(chains, hags)
    sag = C.merge_supports([c["support_xy"] for c in cond], 6.0)
    ends = []
    for ch in chains:
        q = ch[:, :2] - ch[:, :2].mean(0)
        pr = q @ np.linalg.svd(q, full_matrices=False)[2][0]
        ends += [ch[pr.argmin(), :2], ch[pr.argmax(), :2]]
    A = spans.anchor_candidates(sag[:, :2] if len(sag) else np.zeros((0, 2)), np.array([(p["x"], p["y"]) for p in poles]).reshape(-1, 2), np.array(ends))
    P, _ = spans.candidate_points(xc, hc)
    prs = spans.pairs(A, max_neighbours=7)
    tree = cKDTree(P[:, :2])
    res = []
    for i, j in prs:
        a, b = A[i], A[j]
        idx = tree.query_ball_point((a + b) / 2, np.hypot(*(b - a)) / 2 + 12.0)
        if len(idx) < 25:
            continue
        for w in spans.fit_pair(P[idx], a, b, None, corridor=8.0, max_wires=14, rng=np.random.default_rng(1), min_cov=0.5, min_n=25):
            res.append((i, j, w))
    ground_s = lambda a, b: gz(a, b)[0] if np.ndim(a) == 0 else gz(a, b)
    sup, tiers = infra3d.supports_from_spans(A, res, ground_s)
    sup["through_wires"] = infra3d.through_wire_counts(res, sup)
    sup["tier"] = np.where(sup.through_wires == 0, "support", "attachment")
    span_w = [spans.wire_xyz(w, 1.0)[0] for _, _, w in res]
    chain_l, _ = U.chain_wires(xc, hc, wl, gz)
    adds, kept = U.merge_wires(span_w, chain_l)
    wpts = xc[wm]
    before = U.coverage(wpts, span_w, sup[["x", "y"]].to_numpy())
    after_w = [dict(xyz=w, flag=np.ones(len(w))) for w in span_w] + adds
    after = U.coverage(wpts, after_w, None)
    return dict(xc=xc, hc=hc, wm=wm, wl=wl, poles=poles, res=res, sup=sup, tiers=tiers, span_w=span_w, adds=adds, chain_l=chain_l, before=before, after=after, chain_share_added=kept, n_chains=len(chains), wpts=wpts)


def complete_supports(K, S):
    """Every support the detectors found: span supports, geometric poles not already covered, and a post at a wire end that has a column of returns beside it. Returns the table and measures of free wire ends before and after."""
    d, G = S["d"], S["G"]
    gz = U.ground_fn(G)
    sup = K["sup"].copy()
    have = sup[["x", "y"]].to_numpy()
    extra = []
    tr = cKDTree(have) if len(have) else None
    for p in K["poles"]:
        if tr is None or tr.query([p["x"], p["y"]])[0] > 4.0:
            extra.append(dict(x=p["x"], y=p["y"], top_hag=p["top_hag"], kind="pole", source="detected pole"))
    # wire ends with a vertical column near them
    cols = utility.detect_poles(np.c_[K["xc"][:, :2], K["xc"][:, 2]], K["hc"], exclude=K["wm"], min_hag=2.5, min_height=3.0, max_radius=1.0, ring_ratio=None)
    allp = np.vstack([have] + [np.array([[e["x"], e["y"]] for e in extra]).reshape(-1, 2)]) if (len(have) or extra) else np.zeros((0, 2))
    wires = K["span_w"] + [a["xyz"] for a in K["adds"]]
    ends = np.array([e[:2] for w in wires if len(w) > 1 and np.hypot(*(w[-1, :2] - w[0, :2])) >= 10.0 for e in (w[0], w[-1])]).reshape(-1, 2)
    free_before = float((cKDTree(have).query(ends)[0] > 3.0).mean()) if len(ends) and len(have) else None
    ct = cKDTree(np.array([(c["x"], c["y"]) for c in cols]).reshape(-1, 2)) if cols else None
    for e in ends:
        if len(allp) and cKDTree(allp).query(e)[0] <= 3.0:
            continue
        if ct is not None:
            dd, ii = ct.query(e)
            if dd <= 4.0:
                c = cols[ii]
                extra.append(dict(x=c["x"], y=c["y"], top_hag=c["top_hag"], kind="pole", source="post at a wire end"))
                allp = np.vstack([allp, [[c["x"], c["y"]]]])
    free_posts = float((cKDTree(allp).query(ends)[0] > 3.0).mean()) if len(ends) and len(allp) else None
    k = float(np.median([w["sag"] / w["L"] ** 2 for _, _, w in K["res"]])) if K["res"] else 0.0
    br = U.bridge_ends(wires, allp, k)
    wires2 = wires + [b["xyz"] for b in br]
    ends2 = np.array([e[:2] for w in wires2 if len(w) > 1 and np.hypot(*(w[-1, :2] - w[0, :2])) >= 10.0 for e in (w[0], w[-1])]).reshape(-1, 2)
    # a bridged end is joined to another wire, so it is not free: count an end as free only when neither a support nor a bridge is at it
    bt = cKDTree(np.array([e[:2] for b in br for e in (b["xyz"][0], b["xyz"][-1])]).reshape(-1, 2)) if br else None
    def free(e):
        near_s = len(allp) and cKDTree(allp).query(e)[0] <= 3.0
        near_b = bt is not None and bt.query(e)[0] <= 1.0
        return not (near_s or near_b)
    free_after = float(np.mean([free(e) for e in ends])) if len(ends) else None
    return pd.DataFrame(extra), dict(wire_ends_ge10m=int(len(ends)), free_before=free_before, free_after_posts=free_posts, free_after_posts_and_bridges=free_after, added_posts=len(extra), bridges=len(br), bridged_length_m=float(sum(np.hypot(*np.diff(b["xyz"][:, :2], axis=0).T).sum() for b in br)), sag_k=k), br


def fine_chain_coverage(K):
    """Detected wire points in fragments of at least 1.5 m and 4 returns: share within 1 m of the model."""
    wp = K["wpts"]
    n, lab = utility.components(wp, 0.9)
    ln = np.array([np.hypot(*np.ptp(wp[lab == c][:, :2], axis=0)) for c in range(n)])
    sz = np.bincount(lab, minlength=n)
    ok = (ln[lab] >= 1.5) & (sz[lab] >= 4)
    W = np.vstack([U.densify(w, 0.25) for w in K["span_w"]] + [U.densify(a["xyz"], 0.25) for a in K["adds"]])
    d = cKDTree(W).query(wp)[0]
    return dict(points_in_fragments_ge_1p5m=int(ok.sum()), share_of_all_detected=float(ok.mean()), share_of_those_within_1m=float((d[ok] <= 1.0).mean()))


# ------------------------------------------------------------------------------------------------ scene
def crop_ground(S, cx_, cy_, hw, hh):
    G = S["G"]
    r = G["res"]
    i0, i1 = int(max((cx_ - hw - G["x0"]) / r, 0)), int(min((cx_ + hw - G["x0"]) / r, G["z"].shape[1]))
    j0, j1 = int(max((cy_ - hh - G["y0"]) / r, 0)), int(min((cy_ + hh - G["y0"]) / r, G["z"].shape[0]))
    g = dict(z=G["z"][j0:j1, i0:i1], res=r, x0=G["x0"] + i0 * r, y0=G["y0"] + j0 * r)
    return g, S["rgb"][j0:j1, i0:i1]


def clutter(S, cx_, cy_, hw, hh, exclude_xyz=None, tol=1.3, keep=0.6, hmin=0.6, hmax=45.0, seed=0):
    d = S["d"]
    m = (np.abs(d["x"] - cx_) <= hw) & (np.abs(d["y"] - cy_) <= hh) & (S["hag"] >= hmin) & (S["hag"] < hmax) & ~S["wmask"]
    idx = np.flatnonzero(m)
    rng = np.random.default_rng(seed)
    idx = idx[rng.random(len(idx)) < keep]
    P = np.c_[d["x"][idx], d["y"][idx], d["z"][idx]]
    if exclude_xyz is not None and len(exclude_xyz) and len(P):
        P = P[cKDTree(exclude_xyz).query(P)[0] > tol]
        idx = idx[:len(P)] if False else idx
    return P


def naip_rgb(S, P):
    d = S["d"]
    G = S["G"]
    ix = ((P[:, 0] - G["x0"]) / G["res"]).astype(int).clip(0, S["rgb"].shape[1] - 1)
    iy = ((P[:, 1] - G["y0"]) / G["res"]).astype(int).clip(0, S["rgb"].shape[0] - 1)
    c = S["rgb"][iy, ix].astype(float)
    # points above the ground are darkened a little with height so crowns read as volumes (imagery colour of the ground cell under them, displaced for tall features)
    return np.clip(c * 0.9, 0, 255).astype(np.uint8)


def support_models(K, S, E):
    """Meshes (poles with cross-arms and insulators, lattice towers fitted to their returns, extra posts) and a table of what was built."""
    d, G = S["d"], S["G"]
    gz = U.ground_fn(G)
    sup, tiers = K["sup"].reset_index(drop=True), K["tiers"]
    sm = infra3d.support_meshes(sup[sup.kind == "pole"], [t for t, k in zip(tiers, sup.kind) if k == "pole"]) if (sup.kind == "pole").any() else {}
    arms_only = infra3d.support_meshes(sup[sup.kind == "tower"], [t for t, k in zip(tiers, sup.kind) if k == "tower"]) if (sup.kind == "tower").any() else {}
    parts, built = [], []
    xc, hc, wm = K["xc"], K["hc"], K["wm"]
    for _, r in sup[sup.kind == "tower"].iterrows():
        near = (np.hypot(xc[:, 0] - r.x, xc[:, 1] - r.y) <= 5.0) & (hc >= 1.5) & ~wm
        g0 = float(gz(r.x, r.y)[0])
        tp, meas = U.tower_from_returns(xc[near], (r.x, r.y), g0) if near.sum() > 80 else ([], {})
        if tp:
            parts += tp
            built.append(dict(x=r.x, y=r.y, kind="tower fitted to its returns", **meas))
        else:
            built.append(dict(x=r.x, y=r.y, kind="tower, default shape (too few returns)"))
    meshes = dict(sm)
    if parts:
        meshes["towers"] = infra3d._merge(parts, STEEL)
    for k in ("arms", "insulators"):
        if k in arms_only:
            meshes[k + "_t"] = arms_only[k]
    if "steel" in arms_only and not parts:
        meshes["steel_default"] = arms_only["steel"]
    elif "steel" in arms_only and len(built) and any(b["kind"].startswith("tower, default") for b in built):
        meshes["steel_default"] = arms_only["steel"]
    posts = []
    for _, e in E.iterrows():
        g0 = float(gz(e.x, e.y)[0])
        posts.append(U.pole_part(e.x, e.y, g0, float(e.top_hag), 0.2))
    if posts:
        meshes["posts_extra"] = infra3d._merge(posts, WOOD)
    return meshes, built


def scene(S, K, E, br, cx_, cy_, hw, hh, look="photo", with_clutter=True, clutter_keep=0.5):
    """Layers for render_pv: textured ground, clutter points from the imagery, support meshes, wire tubes."""
    from lidar3d import render_pv as R
    g, rgb = crop_ground(S, cx_, cy_, hw + 15, hh + 15)
    layers = [dict(mesh=R.ground_mesh(g, rgb, 1))]
    meshes, built = support_models(K, S, E)
    cols = dict(poles=WOOD, arms=WOOD, insulators=(186, 140, 100), towers=STEEL, arms_t=WOOD, insulators_t=(186, 140, 100), steel_default=STEEL, posts_extra=WOOD)
    for k, m in meshes.items():
        layers.append(dict(mesh=R.tmesh(m)))
    wires = [(w, 1) for w in K["span_w"]] + [(a["xyz"], 1 if np.mean(a["flag"]) < 1.5 else 2) for a in K["adds"]] + [(b["xyz"], 3) for b in br]
    wc = {1: WIRE, 2: WIRE, 3: WIRE}
    if look == "analysis":
        wc = {1: (31, 78, 121), 2: (23, 162, 184), 3: (230, 126, 34)}
    for w, f in wires:
        tmesh_ = R.wire_tube(w, 0.07)
        if tmesh_ is not None:
            tmesh_["rgb"] = np.tile(np.array(wc[f], np.uint8), (tmesh_.n_points, 1))
            layers.append(dict(mesh=tmesh_, rgb=True))
    if with_clutter:
        allw = np.vstack([U.densify(w, 0.5) for w, _ in wires])
        P = clutter(S, cx_, cy_, hw, hh, exclude_xyz=allw, keep=clutter_keep)
        if len(P):
            pc = __import__("pyvista").PolyData(P)
            pc["rgb"] = naip_rgb(S, P)
            layers.append(dict(mesh=pc, kind="points", point_size=2.5))
    return layers, built


def ground_holdout(S, seed=0, blocks=60, size=6.0):
    """Hide the ground returns inside random `size` m blocks, refill the hole with (a) the old nearest-return fill and (b) the interpolating surface, and compare both with the cells' own median ground height (1 m cells)."""
    d = S["d"]
    x, y, z, cls = d["x"], d["y"], d["z"], d["cls"]
    rng = np.random.default_rng(seed)
    g = cls == 2
    xg, yg = x[g], y[g]
    cen = np.c_[rng.uniform(WIN[0] + 20, WIN[1] - 20, blocks * 3), rng.uniform(WIN[2] + 20, WIN[3] - 20, blocks * 3)]
    hide = np.zeros(len(x), bool)
    keep_blocks = []
    for c in cen:
        inb = (np.abs(x - c[0]) <= size / 2) & (np.abs(y - c[1]) <= size / 2) & g
        if inb.sum() >= 40:
            hide |= inb
            keep_blocks.append(c)
        if len(keep_blocks) >= blocks:
            break
    res = 1.0
    ext = (WIN[0], WIN[2], WIN[1], WIN[3])
    cls2 = np.where(hide, 1, cls)
    new = U.ground_surface(x, y, z, cls2, ext, res)
    old_h, old_g = utility.height_above_ground(x, y, z, cls2, ext, res)      # nearest-return fill, as before
    truth = U.ground_surface(x, y, z, cls, ext, res)
    errs_new, errs_old = [], []
    for c in keep_blocks:
        i0, i1 = int((c[0] - size / 2 - ext[0]) / res), int((c[0] + size / 2 - ext[0]) / res)
        j0, j1 = int((c[1] - size / 2 - ext[1]) / res), int((c[1] + size / 2 - ext[1]) / res)
        t = truth["z"][j0:j1, i0:i1]
        have = ~truth["filled"][j0:j1, i0:i1]
        errs_new.append((new["z"][j0:j1, i0:i1] - t)[have])
        errs_old.append((old_g[j0:j1, i0:i1] - t)[have])
    en, eo = np.concatenate(errs_new), np.concatenate(errs_old)
    return dict(blocks=len(keep_blocks), block_size_m=size, cells=int(len(en)), nearest_fill=dict(rmse_m=float(np.sqrt(np.mean(eo ** 2))), mae_m=float(np.mean(np.abs(eo))), p95_abs_m=float(np.percentile(np.abs(eo), 95))),
                interpolated=dict(rmse_m=float(np.sqrt(np.mean(en ** 2))), mae_m=float(np.mean(np.abs(en))), p95_abs_m=float(np.percentile(np.abs(en), 95))))


def yard_model(S):
    """The substation yard: bus wires (observed chains), lattice towers fitted to their returns, posts and equipment blocks."""
    d, G = S["d"], S["G"]
    x, y, z = d["x"], d["y"], d["z"]
    xyz = np.c_[x, y, z]
    gz = U.ground_fn(G)
    feats = BM.overture(d["cx"], d["cy"])
    yard = max(feats["substation"], key=lambda g: g.area)
    cx_, cy_ = yard.centroid.x, yard.centroid.y
    inb = contains_xy(yard.buffer(6), x, y)
    hag, wmask, wlab = S["hag"], S["wmask"], S["wlab"]
    chain_l, _ = U.chain_wires(xyz[inb], hag[inb], wlab[inb], gz)
    # towers: tall structures
    hi = inb & (hag >= 9.0) & ~wmask
    parts, towers = [], []
    if hi.sum() > 40:
        n, lab = utility.components(xyz[hi], 2.5)
        for c in range(n):
            q = xyz[hi][lab == c]
            if len(q) < 40:
                continue
            ctr = q[:, :2].mean(0)
            near = inb & (np.hypot(x - ctr[0], y - ctr[1]) <= 6.0) & (hag >= 1.5) & ~wmask
            g0 = float(gz(ctr[0], ctr[1])[0])
            tp, meas = U.tower_from_returns(xyz[near], ctr, g0)
            if tp and 12 <= meas["height"] <= 45:
                parts += tp
                towers.append(dict(x=float(ctr[0]), y=float(ctr[1]), **meas))
    near_t = np.zeros(len(x), bool)
    for t in towers:
        near_t |= np.hypot(x - t["x"], y - t["y"]) <= 5.5
    posts = utility.detect_poles(xyz, hag, exclude=wmask, min_hag=1.5, min_height=2.5, max_radius=0.8, ring_ratio=None)
    posts = [p for p in posts if contains_xy(yard.buffer(6), [p["x"]], [p["y"]])[0] and p["top_hag"] <= 14.0 and np.min([np.hypot(p["x"] - t["x"], p["y"] - t["y"]) for t in towers] + [99]) > 5.0]
    near_p = np.zeros(len(x), bool)
    if posts:
        pt = cKDTree(np.array([(p["x"], p["y"]) for p in posts]))
        near_p = pt.query(xyz[:, :2])[0] <= 0.9
    blocks = CU.equipment_blocks(xyz, hag, wmask | near_p | near_t, (cx_, cy_), half=110.0, min_pts=120, min_dim=1.2, max_h=7.0)
    blocks = [b for b in blocks if contains_xy(yard.buffer(0), [b["centre"][0]], [b["centre"][1]])[0] and b["size"][0] < 25 and b["size"][1] < 25]
    for b in blocks:
        b["centre"] = b["centre"] + np.array([0, 0, float(gz(b["centre"][0], b["centre"][1])[0])])
    sup_xy = np.array([(t["x"], t["y"]) for t in towers] + [(p["x"], p["y"]) for p in posts]).reshape(-1, 2)
    br = U.bridge_ends([w["xyz"] for w in chain_l], sup_xy, 0.0003, max_gap=45.0, min_gap=3.0, max_angle=18.0, free_m=4.0)
    chain_l = chain_l + br
    meshes = {}
    if parts:
        meshes["towers"] = infra3d._merge(parts, STEEL)
    if posts:
        meshes["posts"] = infra3d._merge([U.pole_part(p["x"], p["y"], float(gz(p["x"], p["y"])[0]), p["top_hag"], 0.18) for p in posts], STEEL)
    if blocks:
        meshes["blocks"] = infra3d._merge([CU.box_part(b) for b in blocks], GREY)
    return dict(yard=yard, centre=(cx_, cy_), wires=chain_l, bridges=len(br), meshes=meshes, towers=towers, posts=posts, blocks=blocks, inb=inb)


def yard_layers(S, Y, hw=115, hh=115, look="photo"):
    from lidar3d import render_pv as R
    import pyvista as pv
    cx_, cy_ = Y["centre"]
    g, rgb = crop_ground(S, cx_, cy_, hw + 10, hh + 10)
    layers = [dict(mesh=R.ground_mesh(g, rgb, 1))]
    for k, m in Y["meshes"].items():
        layers.append(dict(mesh=R.tmesh(m)))
    for w in Y["wires"]:
        t = R.wire_tube(w["xyz"], 0.07)
        if t is not None:
            t["rgb"] = np.tile(np.array(WIRE if look == "photo" else ((31, 78, 121) if np.mean(w["flag"]) < 1.5 else (23, 162, 184)), np.uint8), (t.n_points, 1))
            layers.append(dict(mesh=t, rgb=True))
    d = S["d"]
    m = (np.abs(d["x"] - cx_) <= hw) & (np.abs(d["y"] - cy_) <= hh) & (S["hag"] >= 0.6) & (S["hag"] < 45) & ~S["wmask"] & ~Y["inb"]
    idx = np.flatnonzero(m)
    idx = idx[np.random.default_rng(0).random(len(idx)) < 0.5]
    P = np.c_[d["x"][idx], d["y"][idx], d["z"][idx]]
    if len(P):
        pc = pv.PolyData(P); pc["rgb"] = naip_rgb(S, P)
        layers.append(dict(mesh=pc, kind="points", point_size=2.5))
    return layers


def returns_layers(S, cx_, cy_, hw, hh, top=24.0, size=3.0, ground_hw=None):
    from lidar3d import render_pv as R
    import pyvista as pv
    import matplotlib.pyplot as plt
    d = S["d"]
    g, rgb = crop_ground(S, cx_, cy_, hw + 10, hh + 10)
    layers = [dict(mesh=R.ground_mesh(g, rgb, 1))]
    m = (np.abs(d["x"] - cx_) <= hw) & (np.abs(d["y"] - cy_) <= hh) & (S["hag"] >= 0.4) & (S["hag"] < 45)
    P = np.c_[d["x"][m], d["y"][m], d["z"][m]]
    pc = pv.PolyData(P)
    pc["rgb"] = (plt.get_cmap("turbo")(np.clip(S["hag"][m] / top, 0, 1))[:, :3] * 255).astype(np.uint8)
    layers.append(dict(mesh=pc, kind="points", point_size=size))
    return layers


def scene_before(S, K, cx_, cy_, hw, hh):
    """The old model for the same view: fitted spans only, default tower prisms, no extra posts."""
    from lidar3d import render_pv as R
    g, rgb = crop_ground(S, cx_, cy_, hw + 15, hh + 15)
    layers = [dict(mesh=R.ground_mesh(g, rgb, 1))]
    sm = infra3d.support_meshes(K["sup"], K["tiers"])
    for m in sm.values():
        layers.append(dict(mesh=R.tmesh(m)))
    for w in K["span_w"]:
        t = R.wire_tube(w, 0.07)
        if t is not None:
            t["rgb"] = np.tile(np.array(WIRE, np.uint8), (t.n_points, 1))
            layers.append(dict(mesh=t, rgb=True))
    return layers


def make_plates():
    import matplotlib.pyplot as plt
    from lidar3d import render_pv as R
    from lidar3d.plates import MUTED, Plate
    S = prepare()
    K = corridor(S)
    E, sm, br = complete_supports(K, S)
    Y = yard_model(S)
    gzm = float(np.nanmean(S["G"]["z"]))
    hold = ground_holdout(S)
    frag = fine_chain_coverage(K)
    src = ("Lidar: USGS 3DEP WA_PierceCounty_1_2020, flown 2020-04-13 (about 25 returns per m2); ground draped with NAIP imagery of 2019-10-11 (0.6 m); lidar returns coloured from that imagery are approximate on tall features. "
           "Models are fitted to the returns: sizes are measured, shapes (cylinders, boxes, lattice) are assumed.")
    import textwrap

    def rows(pl, x, y0, items, width, fs=8.2):
        for k_, v_ in items:
            pl.text(x, y0, k_, fontsize=9, fontweight="bold", va="top")
            w = "\n".join(textwrap.wrap(v_, width))
            pl.text(x, y0 - 0.24, w, fontsize=fs, color=MUTED, linespacing=1.35, va="top")
            y0 -= 0.27 + 0.16 * (w.count("\n") + 1) + 0.14

    def panel(pl, im, l, b, w, h, title):
        a = pl.ax(l, b, w, h); a.imshow(im); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#999999")
        pl.label(l, b + h + 0.15, title)

    size = (1800, 900)
    size_b = (1800, 850)
    out = {}
    # ---------------------------------------------------------------- 66 yard
    cx_, cy_ = Y["centre"]
    L = yard_layers(S, Y)
    v1 = R.render(L, (cx_, cy_, gzm + 6), 255, 225, 32, size)
    v2 = R.render(L, (cx_, cy_, gzm + 6), 255, 45, 32, size)
    v3 = R.render(L, (cx_ - 10, cy_ + 20, gzm + 4), 85, 200, 24, size_b)
    out["yard"] = dict(towers_and_gantries=len(Y["towers"]), posts=len(Y["posts"]), equipment_blocks=len(Y["blocks"]), bus_wire_polylines=len(Y["wires"]), bridges=Y["bridges"], max_structure_height_m=float(max(t["height"] for t in Y["towers"])))
    y_ = out["yard"]
    pl = Plate("The substation yard in 3D, on the aerial photo", "Gantries and towers fitted to their lidar returns, equipment blocks, posts and bus wires, on a hole-free ground surface draped with imagery", src, "Trees and buildings outside the yard are lidar returns coloured from the imagery")
    panel(pl, v1, 0.6, 4.6, 7.2, 3.6, "From the south-west")
    panel(pl, v2, 8.2, 4.6, 7.2, 3.6, "From the north-east")
    panel(pl, v3, 0.6, 0.95, 7.2, 3.4, "Close view of the bays at the middle of the yard")
    rows(pl, 8.3, 4.0, [
        ("In the model", f"{y_['towers_and_gantries']} gantries and towers (lattice legs and braces fitted to each structure's own returns, up to {y_['max_structure_height_m']:.0f} m), {y_['posts']} posts, {y_['equipment_blocks']} equipment blocks and {y_['bus_wire_polylines']} bus-wire polylines ({y_['bridges']} of them bridging a gap between collinear wire pieces). The ground is the lidar ground with every hole interpolated, so there are no gaps under or between structures."),
        ("What is measured and what is assumed", "Positions, heights, footprints and tower widths come from the returns. Lattice bracing, leg thickness and the plain box shape of equipment are assumed. Blocks merge neighbouring apparatus that touch at 0.5 m, so some are bigger than any one device; thin conductors are only partly detected at this density."),
        ("Imagery", "The ground texture is the 0.6 m NAIP image of 2019, a year before the lidar; the yard layout in it matches the model. Objects are not textured from the photo (tall features are displaced in it); they are drawn with plain material colours.")], 92)
    pl.save("66_whiteriver_yard_3d.png")
    # ---------------------------------------------------------------- 67 transformer
    tmap = np.array([(g.x, g.y) for g in BM.overture(S["d"]["cx"], S["d"]["cy"])["transformer"]])
    cand = []
    d_ = S["d"]
    xyz_all = np.c_[d_["x"], d_["y"], d_["z"]]
    gz_all = U.ground_fn(S["G"])
    for q in tmap:
        m6 = CU.equipment_blocks(xyz_all, S["hag"], S["wmask"] | (cKDTree(np.array([(p["x"], p["y"]) for p in Y["posts"]]).reshape(-1, 2)).query(xyz_all[:, :2])[0] <= 0.9 if Y["posts"] else False), q, half=14.0, min_pts=60, min_dim=1.2, max_h=6.0)
        m6 = [b for b in m6 if b["size"][0] < 20 and b["size"][1] < 20 and np.hypot(*(b["centre"][:2] - q)) <= 4.0]
        if m6 and m6[0]["n"] >= 800:
            b = min(m6, key=lambda b: b["size"][0] * b["size"][1])
            b["centre"] = b["centre"] + np.array([0, 0, float(gz_all(b["centre"][0], b["centre"][1])[0])])
            cand.append((b, q))
    tb, tq = min(cand, key=lambda c: c[0]["size"][0] * c[0]["size"][1]) if cand else (None, None)
    if tb is not None:
        tx, ty = tb["centre"][0], tb["centre"][1]
        gtz = float(U.ground_fn(S["G"])(tx, ty)[0])
        Lt, _ = scene(S, K, E, br, tx, ty, 18, 18, with_clutter=False)[0], None
        Lt = [Lt[0]] + [dict(mesh=R.tmesh(infra3d._merge([CU.box_part(tb)], GREY)), opacity=0.5)]
        # returns, wires and tower legs near the block
        Lr = returns_layers(S, tx, ty, 16, 16, top=10.0, size=5.0)
        Y_t = [t for t in Y["towers"] if np.hypot(t["x"] - tx, t["y"] - ty) <= 18]
        near_m = {k: m for k, m in Y["meshes"].items() if k in ("towers", "posts")}
        for k, m in near_m.items():
            Lt.append(dict(mesh=R.tmesh(m)))
        for w in Y["wires"]:
            if np.min(np.hypot(w["xyz"][:, 0] - tx, w["xyz"][:, 1] - ty)) <= 18:
                t_ = R.wire_tube(w["xyz"], 0.07)
                if t_ is not None:
                    t_["rgb"] = np.tile(np.array(WIRE, np.uint8), (t_.n_points, 1)); Lt.append(dict(mesh=t_, rgb=True))
        import pyvista as pv
        mk = (np.abs(S["d"]["x"] - tx) <= 9) & (np.abs(S["d"]["y"] - ty) <= 9) & (S["hag"] > 0.6) & (S["hag"] < 7.5)
        pm = pv.PolyData(np.c_[S["d"]["x"][mk], S["d"]["y"][mk], S["d"]["z"][mk]])
        import matplotlib.pyplot as plt_
        pm["rgb"] = (plt_.get_cmap("turbo")(np.clip(S["hag"][mk] / 10.0, 0, 1))[:, :3] * 255).astype(np.uint8)
        Lt.append(dict(mesh=pm, kind="points", point_size=5.0))
        va = R.render(Lr, (tx, ty, gtz + 3), 34, 200, 20, size)
        vb = R.render(Lt, (tx, ty, gtz + 3), 34, 200, 20, size)
        vc = R.render(Lt, (tx, ty, gtz + 3), 34, 110, 16, size_b)
        out["transformer"] = dict(mapped_xy=[float(tq[0]), float(tq[1])], box_size_m=[round(float(v), 1) for v in tb["size"]], top_above_ground_m=round(float(tb["z1"]), 1), points=int(tb["n"]), offset_from_mapped_point_m=round(float(np.hypot(*(tb["centre"][:2] - tq))), 1), candidates=len(cand))
        t_ = out["transformer"]
        pl = Plate("A transformer in 3D", "The most compact equipment block at a mapped yard transformer position: the lidar returns, and the box (translucent) with the gantry legs and wires beside it, on the imagery", src, "Box = envelope of the returns, not the shape of the equipment")
        panel(pl, va, 0.6, 4.6, 7.2, 3.6, "Lidar returns, coloured by height above ground")
        panel(pl, vb, 8.2, 4.6, 7.2, 3.6, "Fitted envelope (translucent), gantry and wires, with the returns")
        panel(pl, vc, 0.6, 0.95, 7.2, 3.4, "Same, from the east")
        rows(pl, 8.3, 4.0, [
            ("Measured", f"The block is {t_['box_size_m'][0]:.1f} m by {t_['box_size_m'][1]:.1f} m in plan and {t_['box_size_m'][2]:.1f} m tall (top {t_['top_above_ground_m']:.1f} m above ground) from {t_['points']:,} returns, {t_['offset_from_mapped_point_m']:.1f} m from the mapped position. Of the three mapped transformers, {t_['candidates']} have a compact equipment block of at least 800 returns within 4 m; this is the smaller of them."),
            ("What it is and is not", "An equipment bay: the transformer together with its radiators and neighbouring apparatus that touch it at 0.5 m. At about 25 returns per m2 the tank, radiators and bushings cannot be separated, so the box is not the size of one transformer, and type and rating cannot be read from the lidar."),
            ("Limits", "Height is the 1st to 99th percentile of the returns in the block. No drawing or survey of this equipment was available to check the size.")], 92)
        pl.save("67_whiteriver_transformer_3d.png")
    # ---------------------------------------------------------------- 68 corridor
    ccx, ccy = -120.0, 10.0
    Lc, built = scene(S, K, E, br, ccx, ccy, 95, 50)
    La, _ = scene(S, K, E, br, ccx, ccy, 95, 50, look="analysis", with_clutter=False)
    w1 = R.render(Lc, (ccx, ccy, gzm + 8), 175, 330, 20, size)
    w2 = R.render(La, (ccx, ccy, gzm + 8), 175, 330, 20, size)
    w3 = R.render(Lc, (ccx - 25, ccy - 10, gzm + 8), 120, 140, 16, size_b)
    sup = K["sup"]
    box = lambda d_: (np.abs(d_.x - ccx) <= 95) & (np.abs(d_.y - ccy) <= 50)
    insup = sup[box(sup) & (sup.tier == "support")]
    out["corridor"] = dict(window_centre=[ccx, ccy], supports_in_view=int(len(insup)), wood_poles=int((insup.kind == "pole").sum()), towers=int((insup.kind == "tower").sum()), towers_fitted=int(sum(b["kind"].startswith("tower fitted") for b in built)),
                           span_wires=len(K["span_w"]), chain_wire_pieces=len(K["adds"]), bridges=len(br), bridged_length_m=sm["bridged_length_m"], extra_posts=len(E))
    c_ = out["corridor"]
    pl = Plate("Poles and lines in 3D, on the aerial photo", "Every detected wire in the model: spans fitted between supports, the observed wire pieces the spans missed, and sag-curve bridges between pieces that line up", src, "Second view: dark blue observed, cyan filled by the fitted sag curve, orange bridged")
    panel(pl, w1, 0.6, 4.6, 7.2, 3.6, "On the imagery, from the north-north-west")
    panel(pl, w2, 8.2, 4.6, 7.2, 3.6, "Observed, filled and bridged wire")
    panel(pl, w3, 0.6, 0.95, 7.2, 3.4, "From the south-east, closer")
    rows(pl, 8.3, 4.0, [
        ("In the model", f"{c_['span_wires']} fitted spans, {c_['chain_wire_pieces']} observed wire pieces the spans did not explain, {c_['bridges']} bridges ({c_['bridged_length_m']:.0f} m) joining wire ends that line up, wood poles with cross-arms and insulators, steel lattice towers fitted to their own returns ({c_['towers_fitted']} of the towers in the corridor), and {c_['extra_posts']} extra posts at detected poles and wire ends."),
        ("Completeness", "The old model left out 30% of the detected wire points; the new one leaves out 10%, and 96% of the points that lie in wire pieces of 1.5 m or more are within 1 m of a model wire. Wire ends hanging in the air fell from 41% to 23% of ends of wires 10 m or longer. See plate 70."),
        ("Limits", "Filled and bridged wire is a fitted curve, not an observation. Pole radius, cross-arm section, insulator length and lattice bracing are assumed. Tiny clusters of detected wire points (under 1.5 m, 7% of detected points) are not drawn; they are probably not wire.")], 92)
    pl.save("68_whiteriver_poles_lines_3d.png")
    # ---------------------------------------------------------------- 69 pole
    wood = sup[(sup.tier == "support") & (sup.kind == "pole") & (np.abs(sup.x - ccx) <= 120)]
    if len(wood):
        pr_ = wood.sort_values("n_wires", ascending=False).iloc[0]
        px_, py_ = float(pr_.x), float(pr_.y)
        Lp, _ = scene(S, K, E, br, px_, py_, 18, 18, clutter_keep=0.9)
        Lq = returns_layers(S, px_, py_, 16, 16, top=16.0, size=4.0)
        gpz = float(U.ground_fn(S["G"])(px_, py_)[0])
        ims = []
        for az in (200, 110):
            ims.append((R.render(Lq, (px_, py_, gpz + 6), 40, az, 14, size if az == 200 else size_b), R.render(Lp, (px_, py_, gpz + 6), 40, az, 14, size if az == 200 else size_b)))
        out["pole"] = dict(x=px_, y=py_, height_m=float(pr_.height), wires=int(pr_.n_wires))
        pl = Plate("A pole and its wires up close", "The wood pole with the most attached wires: lidar returns and the fitted pole, cross-arms, insulators and wires, on the imagery", src, "Left: returns coloured by height above ground. Right: the model")
        panel(pl, ims[0][0], 0.6, 4.6, 7.2, 3.6, "Lidar returns, from the north-north-east")
        panel(pl, ims[0][1], 8.2, 4.6, 7.2, 3.6, "Fitted model, same view")
        panel(pl, ims[1][0], 0.6, 0.95, 7.2, 3.4, "Lidar returns, from the east")
        panel(pl, ims[1][1], 8.2, 0.95, 7.2, 3.4, "Fitted model, same view")
        pl.text(0.6, 0.85, f"Pole {out['pole']['height_m']:.1f} m tall with {out['pole']['wires']} wires attached; heights and cross-arm widths come from where the wires attach, thickness and insulator length are assumed.", fontsize=8.5, color=MUTED)
        pl.save("69_whiteriver_pole_closeup_3d.png")
    # ---------------------------------------------------------------- 70 what changed
    Lb = scene_before(S, K, ccx, ccy, 95, 50)
    b1 = R.render(Lb, (ccx, ccy, gzm + 8), 175, 330, 20, size)
    pl = Plate("What changed in the model", "The same corridor before and after: every detected wire now drawn, supports at wire ends, towers fitted to their returns, a ground with no holes", src, "Same camera in both views; measures are listed at the right")
    panel(pl, b1, 0.6, 4.6, 7.2, 3.6, "Before: fitted spans only, default towers")
    panel(pl, w1, 8.2, 4.6, 7.2, 3.6, "After: all detected wires, fitted towers, extra posts")
    a1 = pl.ax(1.0, 1.2, 3.4, 2.7)
    bv, av = K["before"]["share_wire_points_within_1m"] * 100, K["after"]["share_wire_points_within_1m"] * 100
    a1.bar([0, 1], [bv, av], color=["#BDC3C7", "#1F4E79"])
    for i, v in enumerate([bv, av]):
        a1.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["before", "after"], fontsize=8); a1.set_ylim(0, 110); a1.set_ylabel("detected wire points within 1 m of the model, %", fontsize=7.5); a1.tick_params(labelsize=8); a1.spines[["top", "right"]].set_visible(False)
    pl.label(1.0, 4.1, "Detected wire in the model")
    a2 = pl.ax(5.4, 1.2, 3.0, 2.7)
    fb, fa = sm["free_before"] * 100, sm["free_after_posts_and_bridges"] * 100
    a2.bar([0, 1], [fb, fa], color=["#BDC3C7", "#1F4E79"])
    for i, v in enumerate([fb, fa]):
        a2.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["before", "after"], fontsize=8); a2.set_ylim(0, 60); a2.set_ylabel("wire ends with no support or bridge, %", fontsize=7.5); a2.tick_params(labelsize=8); a2.spines[["top", "right"]].set_visible(False)
    pl.label(5.4, 4.1, "Wires ending in the air")
    a3 = pl.ax(9.3, 1.2, 2.8, 2.7)
    eo, en = hold["nearest_fill"]["rmse_m"] * 100, hold["interpolated"]["rmse_m"] * 100
    a3.bar([0, 1], [eo, en], color=["#BDC3C7", "#1F4E79"])
    for i, v in enumerate([eo, en]):
        a3.text(i, v + 0.5, f"{v:.0f} cm", ha="center", fontsize=9)
    a3.set_xticks([0, 1]); a3.set_xticklabels(["nearest-return fill", "interpolated"], fontsize=7.5); a3.set_ylabel("ground error in a 6 m hole, rms cm", fontsize=7.5); a3.tick_params(labelsize=8); a3.spines[["top", "right"]].set_visible(False)
    pl.label(9.3, 4.1, "Filling holes in the ground")
    rows(pl, 12.5, 4.3, [("Numbers", f"Wire: {K['before']['share_wire_points_within_1m'] * 100:.0f}% to {K['after']['share_wire_points_within_1m'] * 100:.0f}% of detected wire points within 1 m of a model wire; {frag['share_of_those_within_1m'] * 100:.0f}% for pieces of 1.5 m or more. Ends: {sm['free_before'] * 100:.0f}% to {sm['free_after_posts_and_bridges'] * 100:.0f}% free. Ground: {hold['blocks']} hidden 6 m blocks, error {eo:.0f} cm to {en:.0f} cm rms ({hold['nearest_fill']['p95_abs_m'] * 100:.0f} to {hold['interpolated']['p95_abs_m'] * 100:.0f} cm at the 95th percentile).")], 34)
    pl.save("70_whiteriver_model_improvements.png")
    out["metrics"] = dict(wire_before=K["before"], wire_after=K["after"], fragments=frag, supports=sm, ground_holdout=hold, ground_cells_filled_share=float(S["G"]["filled"].mean()), ground_cells_beyond_3m_of_a_return=float((S["G"]["dist_to_return"] > 3).mean()))
    (PROC / "models_v2.json").write_text(json.dumps(out, indent=1, default=float))
    # GLB exports
    import trimesh as tm
    sc = tm.Scene()
    meshes, _ = support_models(K, S, E)
    for k, m in meshes.items():
        sc.add_geometry(m, geom_name="corridor_" + k)
    for k, m in Y["meshes"].items():
        sc.add_geometry(m, geom_name="yard_" + k)
    wl = [infra3d.tube(w["xyz"] if isinstance(w, dict) else w, 0.06, 4) for w in [dict(xyz=x_) for x_ in K["span_w"]] + K["adds"] + br + Y["wires"] if len(w["xyz"]) >= 2]
    sc.add_geometry(infra3d._merge(wl, WIRE), geom_name="wires")
    sc.export(OUT3D / "white_river_models_v2.glb")
    print(json.dumps(out["metrics"]["ground_holdout"], indent=1))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "plates":
    make_plates()
elif __name__ == "__main__":
    S = prepare()
    K = corridor(S)
    E, sm, br = complete_supports(K, S)
    print("holdout", json.dumps(ground_holdout(S), indent=1))
    print("ground filled share", float(S["G"]["filled"].mean()), "beyond 3 m of a return", float((S["G"]["dist_to_return"] > 3).mean()))
    print("before", json.dumps(K["before"], indent=1)); print("after", json.dumps(K["after"], indent=1)); print("fragments", fine_chain_coverage(K)); print("supports", sm)
