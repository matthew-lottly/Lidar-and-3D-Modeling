"""Classified and realistic 3D scene of the White River substation, its lines and the warehouses south of it (650 m by 530 m): ground, trees (individual crowns), buildings (roof planes), supports, wires and the yard,
drawn once in plain classified colours (no imagery) and once on the aerial photo with imagery-coloured features. Usage: xvfb-run -a python scripts/build_town_scene.py [data|plates]"""
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
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import cKDTree

import build_town as BT
from lidar3d import naip, unified as U, utility
from lidar3d.aoi import AOI
from lidar3d.ept import CACHE, EPT

WIN = (-450.0, 200.0, -400.0, 130.0)
CACHE_D = CACHE / "town"


def load_points():
    p = CACHE_D / "scene_points.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    tr = Transformer.from_crs(4326, BT.CRS, always_xy=True)
    cx, cy = tr.transform(*BT.LONLAT)
    back = Transformer.from_crs(BT.CRS, 4326, always_xy=True)
    x0, x1, y0, y1 = WIN
    W, S = back.transform(cx + x0, cy + y0)
    E, N = back.transform(cx + x1, cy + y1)
    c = EPT(BT.PROJECT).fetch((W, S, E, N), max_depth=20, log=print)
    k = (c.x >= cx + x0) & (c.x < cx + x1) & (c.y >= cy + y0) & (c.y < cy + y1) & (c.cls != 7)
    d = dict(x=(c.x[k] - cx).astype("float32"), y=(c.y[k] - cy).astype("float32"), z=c.z[k].astype("float32"), cls=c.cls[k], ret=c.ret[k], nret=c.nret[k], inten=c.intensity[k], cx=cx, cy=cy)
    CACHE_D.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(d))
    return d


def prepare():
    p = CACHE_D / "scene_prep.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    d = load_points()
    x, y, z, cls = d["x"].astype(float), d["y"].astype(float), d["z"].astype(float), d["cls"]
    G = U.ground_surface(x, y, z, cls, (WIN[0], WIN[2], WIN[1], WIN[3]), 0.5)
    gz = U.ground_fn(G)
    hag = z - gz(x, y)
    ccx, ccy = (WIN[0] + WIN[1]) / 2, (WIN[2] + WIN[3]) / 2
    img, tf, nd = naip.fetch(AOI(round(d["cx"] + ccx), round(d["cy"] + ccy), 420, BT.CRS), 2019, res=0.6, slug="whiteriver_scene")
    img = np.asarray(img)
    img = img / 256.0 if img.max() > 255 else img
    out = dict(d=d, G=G, hag=hag, img=img.astype(np.float32), tf=tf, naip_date=nd)
    p.write_bytes(pickle.dumps(out))
    return out


def sample_naip(S, px, py):
    """RGB (uint8) and NDVI (float) of the aerial image at local metres (px, py)."""
    d, img, tf = S["d"], S["img"], S["tf"]
    c = ((px + d["cx"] - tf.c) / tf.a).astype(int).clip(0, img.shape[2] - 1)
    r = ((py + d["cy"] - tf.f) / tf.e).astype(int).clip(0, img.shape[1] - 1)
    rgb = np.stack([img[b][r, c] for b in range(3)], -1).astype(np.uint8)
    nir, red = img[3][r, c].astype(float), img[0][r, c].astype(float)
    return rgb, (nir - red) / np.maximum(nir + red, 1.0)


def detect(S):
    """Wires, rooftops and trees in the window, with the measures each can be checked by."""
    p = CACHE_D / "scene_detect.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    from lidar3d import classify as CL
    from lidar3d import rooftop as RT
    import build_town_model as BM
    from shapely import contains_xy, wkb as wkb_
    from shapely.geometry import box as sbox
    d, G, hag = S["d"], S["G"], S["hag"]
    x, y, z, cls = d["x"].astype(float), d["y"].astype(float), d["z"].astype(float), d["cls"]
    xyz = np.c_[x, y, z]
    out = {}
    # ---- yard polygon and wires
    feats = BM.overture(d["cx"], d["cy"])
    yard = max(feats["substation"], key=lambda g: g.area)
    inyard = contains_xy(yard.buffer(6), x, y)
    wp = CACHE_D / "scene_wires.pkl"
    if wp.exists():
        wmask, wlab = pickle.loads(wp.read_bytes())
    else:
        wmask, wlab, winfo = utility.detect_wires(xyz, hag)
        wp.write_bytes(pickle.dumps((wmask, wlab)))
    out.update(yard=yard, inyard=inyard, wmask=wmask, wlab=wlab)
    print("wires", int(wmask.sum()), flush=True)
    # ---- imagery products
    img = S["img"]
    ndvi_img = (img[3] - img[0]) / np.maximum(img[3] + img[0], 1.0)
    fit = CL.fit_relief_displacement(ndvi_img, S["tf"], d["cx"], d["cy"], x, y, hag)
    out["displacement_canopy_fit_not_used"] = fit
    print("canopy-colour displacement fit (not used, unstable between window halves):", {k: round(v, 4) if isinstance(v, float) else v for k, v in fit.items()}, flush=True)
    # ---- buildings: planar returns outside the yard
    ccx, ccy = (WIN[0] + WIN[1]) / 2, (WIN[2] + WIN[3]) / 2
    rawp = CACHE_D / "scene_bld_raw.pkl"
    if rawp.exists():
        blds, polys = pickle.loads(rawp.read_bytes())
    else:
        C = RT.Cloud((x - ccx).astype("float32"), (y - ccy).astype("float32"), z.astype("float32"), cls, d["nret"], "2020-04-13", 335.0, d["cx"] + ccx, d["cy"] + ccy)
        dtm, hagC, dsm = RT.surfaces(C, 1.0)
        use = RT.planar_points(C, hagC) & ~inyard & ~wmask
        lab, polys = RT.footprints(C, hagC, use=use, min_area=30.0)
        blds = RT.build_buildings(C, hagC, dtm, lab, polys, use=use)
        rawp.write_bytes(pickle.dumps((blds, polys)))
    # ---- keep real roofs: planar returns also form on dense crowns, which show up as vegetated (high NDVI) in the imagery
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    ndvi_img = (img[3] - img[0]) / np.maximum(img[3] + img[0], 1.0)
    tfm = S["tf"]
    feats_b = []
    out["bld_offset"] = (ccx, ccy)
    out["buildings"] = list(blds)
    out["displacement_roofs"] = roof_displacement(S, out)
    fitr = out["displacement_roofs"]
    for b in blds:
        cells = b.cells + np.array([ccx, ccy])
        if len(cells):
            h = float(max(b.height, 1.0))
            cx_ = cells[:, 0] + h * (fitr["k"] * cells[:, 0] + fitr["bx"])
            cy_ = cells[:, 1] + h * (fitr["k"] * cells[:, 1] + fitr["by"])
            nd = float(np.median(ndvi_img[((cy_ + d["cy"] - tfm.f) / tfm.e).astype(int).clip(0, ndvi_img.shape[0] - 1), ((cx_ + d["cx"] - tfm.c) / tfm.a).astype(int).clip(0, ndvi_img.shape[1] - 1)]))
        else:
            nd = 1.0
        pshare = float((b.cell_plane >= 0).mean()) if len(b.cell_plane) else 0.0
        feats_b.append(dict(area=float(b.footprint.area), height=float(b.height), ndvi=nd, plane_share=pshare))
    def edge_clear(b):
        q = np.array(b.footprint.exterior.coords) + [ccx, ccy]
        return bool(q[:, 0].min() > WIN[0] + 6 and q[:, 0].max() < WIN[1] - 6 and q[:, 1].min() > WIN[2] + 6 and q[:, 1].max() < WIN[3] - 6)
    # roofs are not vegetated (NDVI under 0.25 at the displaced roof position), at least 40 m2, 2.5 to 25 m tall (a tall footprint must be big: small tall planar patches are crowns or towers), and not cut by the window edge
    keep = [b for b, f in zip(blds, feats_b) if f["ndvi"] < 0.25 and f["area"] >= 40.0 and 2.5 <= b.height <= 25.0 and (b.height < 12.0 or f["area"] >= 300.0) and edge_clear(b)]
    out["building_features"] = feats_b
    # every cell of a footprint belongs to its nearest roof plane (the stock assignment leaves cells more than 2 m from a plane's returns bare, which draws holes in the roof)
    for b in blds:
        allp = np.vstack([p.pts for p in b.planes])
        pid = np.concatenate([np.full(len(p.pts), j) for j, p in enumerate(b.planes)])
        _, kk = cKDTree(allp[:, :2]).query(b.cells)
        b.cell_plane = pid[kk]
    out["buildings_all"] = len(blds)
    out["buildings_raw_list"] = list(blds)
    out["buildings"] = keep
    # independent check against Overture's footprints (area overlap; rows of houses merge, so IoU per building would understate it)
    import pyarrow.dataset as pads
    from lidar3d import sitemap
    trf = Transformer.from_crs(4326, BT.CRS, always_xy=True)
    bk = Transformer.from_crs(BT.CRS, 4326, always_xy=True)
    Wl, Sl = bk.transform(d["cx"] + WIN[0], d["cy"] + WIN[2])
    El, Nl = bk.transform(d["cx"] + WIN[1], d["cy"] + WIN[3])
    tb = sitemap._table("buildings", "building", ["geometry"], (Wl, Sl, El, Nl))
    ov = unary_union([Polygon([(trf.transform(a, b)[0] - d["cx"], trf.transform(a, b)[1] - d["cy"]) for a, b in wkb_.loads(g).exterior.coords]) for g in tb.geometry]) if len(tb) else Polygon()
    def agree(bl):
        u = unary_union([Polygon(np.array(b.footprint.exterior.coords) + [ccx, ccy]) for b in bl]) if bl else Polygon()
        inter = u.intersection(ov).area
        return dict(buildings=len(bl), footprint_area_m2=float(u.area), share_of_lidar_area_inside_overture=float(inter / max(u.area, 1e-9)), share_of_overture_area_covered=float(inter / max(ov.area, 1e-9)))
    out["building_check"] = dict(all_planar_footprints=agree(list(blds)), kept=agree(keep), overture_buildings=int(len(tb)), overture_area_m2=float(ov.area))
    print("buildings kept", len(keep), "of", len(blds), out["building_check"], flush=True)
    blds = keep
    polys = [b.footprint for b in blds]
    out["bld_offset"] = (ccx, ccy)
    # ---- trees from what is left: not ground, not wire, not building, not yard
    bmask = np.zeros(len(x), bool)
    # returns of every roof-like footprint (not vegetated) are kept out of the trees, including roofs dropped from the drawn set for being tall, small or cut by the window edge
    roofish = [b.footprint for b, f in zip(out["buildings_raw_list"], feats_b) if f["ndvi"] < 0.25 and f["area"] >= 40.0]
    if len(roofish):
        for q in roofish:
            qq = q.buffer(1.5)
            minx, miny, maxx, maxy = qq.bounds
            sel = np.flatnonzero((x - ccx >= minx) & (x - ccx <= maxx) & (y - ccy >= miny) & (y - ccy <= maxy))
            if len(sel):
                bmask[sel[contains_xy(qq, x[sel] - ccx, y[sel] - ccy)]] = True
    out["bmask"] = bmask
    veg = ~bmask & ~wmask & ~inyard & (cls != 2)
    ext = (WIN[0], WIN[2], WIN[1], WIN[3])
    chm = CL.canopy_height_model(x[veg], y[veg], hag[veg], ext, 0.5)
    tlab, T = CL.segment_trees(chm, ext, 0.5)
    base = CL.crown_bases(tlab, T["id"], x[veg], y[veg], hag[veg], ext, 0.5)
    T["base"] = np.minimum(base, 0.7 * T["height"])
    out.update(chm=chm, tree_lab=tlab, trees=T)
    print("trees", len(T["x"]), "median height", float(np.median(T["height"])), flush=True)
    p.write_bytes(pickle.dumps(out))
    return out


def network(S, D):
    """The utility network over the whole window outside the yard: fitted spans, observed wire pieces the spans missed, bridges, supports (same method as the close-up models), and the yard model."""
    p = CACHE_D / "scene_network.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    import build_town_models as BMo
    S2 = dict(d=dict(S["d"], x=S["d"]["x"].astype(float), y=S["d"]["y"].astype(float), z=S["d"]["z"].astype(float)), G=S["G"], hag=S["hag"], wmask=D["wmask"], wlab=D["wlab"])
    K = BMo.corridor(S2, bounds=WIN, exclude=D["inyard"])
    E, sm, br = BMo.complete_supports(K, S2)
    # the yard model needs imagery-free inputs only
    Y = BMo.yard_model(S2)
    out = dict(K=K, E=E, sm=sm, br=br, Y=Y)
    p.write_bytes(pickle.dumps(out))
    print("network:", len(K["span_w"]), "spans", len(K["adds"]), "chain pieces", len(br), "bridges", len(K["sup"]), "supports; yard towers", len(Y["towers"]), flush=True)
    return out


def make_maps():
    import build_town_model as BM
    from lidar3d import figures_scene as FS
    S = prepare()
    D = detect(S)
    N = network(S, D)
    feats = BM.overture(S["d"]["cx"], S["d"]["cy"])
    eq = json.loads((ROOT / "data" / "processed" / "town" / "extras_equipment.json").read_text())
    out = {}
    out["overview"] = FS.fig_overview(S, D, N, feats, WIN, eq)
    FS.fig_profile(S, D, N, feats, WIN, eq)
    FS.fig_joining(S, N)
    out["fused"] = FS.fig_fused(S, D, N, feats, WIN)
    (ROOT / "data" / "processed" / "town" / "scene_maps.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def roof_displacement(S, D):
    """Relief displacement of the aerial image from building roofs (stable across the window, unlike a canopy-colour fit), with split-window repeats."""
    from lidar3d import classify as CL
    img, tf, d = S["img"], S["tf"], S["d"]
    gray = img[:3].mean(0)
    ccx, ccy = D["bld_offset"]
    outs, cen = [], []
    for b in D["buildings"]:
        if b.height < 3:
            continue
        ring = np.array(b.footprint.exterior.coords)
        seg = [a + (c - a) * np.linspace(0, 1, max(int(np.hypot(*(c - a))), 1))[:, None] for a, c in zip(ring[:-1], ring[1:])]
        xy = np.vstack(seg) + [ccx, ccy]
        outs.append((xy, float(b.height)))
        cen.append(xy.mean(0))
    fit = CL.fit_displacement_from_roofs(gray, tf, d["cx"], d["cy"], outs)
    cen = np.array(cen)
    split = {}
    for nm, m in (("west", cen[:, 0] < -125), ("east", cen[:, 0] >= -125), ("north", cen[:, 1] >= -250), ("south", cen[:, 1] < -250)):
        sel = [o for o, mm in zip(outs, m) if mm]
        if len(sel) >= 8:
            r = CL.fit_displacement_from_roofs(gray, tf, d["cx"], d["cy"], sel)
            split[nm] = dict(buildings=len(sel), bx=r["bx"], by=r["by"], k=r["k"], edge_before=r["edge_before"], edge_after=r["edge_after"])
    fit["split"] = split
    return fit


def assemble(S, D, N, mode="classified", step=2):
    """Layers for render_pv: ground, trees, buildings, supports, wires, yard equipment and fence, in classified colours (no imagery) or imagery-coloured."""
    import matplotlib.pyplot as plt
    import pyvista as pv
    import build_town_models as BMo
    from shapely import contains_xy
    from shapely.geometry import Point
    from lidar3d import classify as CL, infra3d, render_pv as R, scene3d as SC
    d, G = S["d"], S["G"]
    x0, x1, y0, y1 = WIN
    gz = U.ground_fn(G)
    fit = roof_displacement(S, D)
    photo = mode == "realistic"
    K, E, br, Y = N["K"], N["E"], N["br"], N["Y"]
    S2 = dict(d=dict(d, x=d["x"].astype(float), y=d["y"].astype(float), z=d["z"].astype(float)), G=G, hag=S["hag"])
    layers = {}
    drop_wire_trees(D, N, gz)
    # ground
    ny, nx = G["z"].shape
    X = np.arange(nx) * G["res"] + G["x0"] + G["res"] / 2
    Y_ = np.arange(ny) * G["res"] + G["y0"] + G["res"] / 2
    XX, YY = np.meshgrid(X, Y_)
    rgb_g, nd_g = sample_naip(S, XX, YY)
    if photo:
        col = rgb_g
    else:
        veg = nd_g >= 0.25
        col = np.where(veg[..., None], np.array([196, 210, 180], np.uint8), np.array([214, 214, 210], np.uint8))
        yard_m = contains_xy(Y["yard"], XX, YY)
        col = np.where(yard_m[..., None], np.array([200, 196, 186], np.uint8), col)
    layers["ground"] = R.ground_mesh(dict(z=G["z"], res=G["res"], x0=G["x0"], y0=G["y0"]), col, step)
    # trees: the canopy surface from the lidar canopy height model (not templates), coloured from the photo or by height
    T = D["trees"]
    n = len(T["x"])
    chm = D["chm"]
    from scipy import ndimage as ndi
    mask = np.isfinite(chm) & (chm >= 2.0)
    mask = ndi.binary_closing(mask, iterations=1) | mask
    filled = np.where(np.isfinite(chm), chm, 0.0)
    smooth = ndi.gaussian_filter(filled, 0.9)
    top = np.where(mask, np.maximum(smooth, 1.0), 0.0) + G["z"]
    if photo:
        h_c = np.where(mask, smooth, 0.0)
        de = h_c * (fit["k"] * XX + fit["bx"]); dn = h_c * (fit["k"] * YY + fit["by"])
        cc, _ = sample_naip(S, XX + de, YY + dn)
        _, nd_c = sample_naip(S, XX + de, YY + dn)
        c0, _ = sample_naip(S, XX, YY)
        cc = np.where((nd_c < 0.12)[..., None], (c0 * 0.8 + np.array([18, 38, 16])).clip(0, 255).astype(np.uint8), cc)
        tone = np.clip(0.9 + 0.012 * np.minimum(smooth, 25.0), 0.9, 1.2)[..., None]
        cc = np.clip(cc.astype(float) * tone, 0, 255).astype(np.uint8)
    else:
        t = np.clip((smooth - 3.0) / 32.0, 0, 1)
        cc = (plt.get_cmap("Greens")(0.3 + 0.62 * t)[..., :3] * 255).astype(np.uint8)
    # one crown per segmented tree, its shape from the tree's own canopy-height cells, its colour the median of its cells
    lab = D["tree_lab"]
    ids = np.asarray(T["id"])
    tcol = np.stack([ndi.median(cc[..., k].astype(float), lab, ids) for k in range(3)], 1)
    tcol = np.where(np.isfinite(tcol), tcol, 90.0)
    tcol = 0.72 * tcol + 0.28 * np.array([46.0, 84.0, 42.0])  # the photo is hazy and late in the season; pull crowns toward foliage green
    if not photo:
        th = np.clip((np.asarray(T["height"]) - 3.0) / 32.0, 0, 1)
        tcol = plt.get_cmap("Greens")(0.45 + 0.45 * th)[:, :3] * 255
    crowns, trunks = R.crowns_mesh(chm, lab, ids, np.c_[T["x"], T["y"]], np.asarray(T["base"]), G["z"], WIN[0], WIN[2], 0.5, tcol)
    layers["crowns"] = crowns
    layers["trunks"] = trunks
    # buildings
    def roof_color(b, j):
        if not photo:
            return (222, 168, 98)
        cells = b.cells[b.cell_plane == j]
        if len(cells) == 0:
            return (150, 150, 150)
        pts = cells + np.array(D["bld_offset"])
        h = float(np.mean(b.planes[j].pts[:, 2]) - b.ground_z)
        cc = CL.displaced_rgb(S["img"][:3], S["tf"], d["cx"], d["cy"], pts[:, 0], pts[:, 1], np.full(len(pts), h), dict(k=fit["k"], bx=fit["bx"], by=fit["by"]))
        return tuple(int(v) for v in np.median(cc, axis=0))
    layers["buildings"] = SC.buildings_mesh(D["buildings"], D["bld_offset"], roof_color, wall_color=(214, 208, 198) if photo else (200, 196, 190))
    # supports and wires
    meshes, built = BMo.support_models(K, S2, E)
    pal = dict(poles=(140, 92, 48), arms=(120, 80, 44), insulators=(200, 170, 130), towers=(70, 96, 130), arms_t=(120, 80, 44), insulators_t=(200, 170, 130), steel_default=(70, 96, 130), posts_extra=(140, 92, 48))
    for k, m in meshes.items():
        p = R.tmesh(m)
        if not photo:
            p["rgb"] = np.tile(np.array(pal.get(k, (120, 120, 120)), np.uint8), (p.n_points, 1))
        layers["sup_" + k] = p
    for k, m in Y["meshes"].items():
        if k == "blocks":
            continue
        p = R.tmesh(m)
        if not photo:
            p["rgb"] = np.tile(np.array((70, 96, 130), np.uint8), (p.n_points, 1))
        layers["yard_" + k] = p
    boxy = [b for b in Y["blocks"] if b.get("fill", 1.0) >= 0.22]
    frames = [b for b in Y["blocks"] if b.get("fill", 1.0) < 0.22]
    if frames:
        fp = np.vstack([b["points"] for b in frames])
        pc_ = pv.PolyData(fp)
        pc_["rgb"] = np.tile(np.array((150, 156, 162) if photo else (70, 96, 130), np.uint8), (len(fp), 1))
        layers["frames"] = pc_
    eqp, kinds = SC.equipment_templates(boxy)
    if eqp is not None:
        if not photo:
            eqp["rgb"] = np.tile(np.array((60, 110, 160), np.uint8), (eqp.n_points, 1))
        layers["equipment"] = eqp
    ring = np.asarray(Y["yard"].exterior.coords)
    layers["fence"] = SC.fence_mesh(ring, lambda a, b: gz(a, b))
    wires_raw = [w for w in K["span_w"]] + [a["xyz"] for a in K["adds"]] + [b["xyz"] for b in br] + [w["xyz"] for w in Y["wires"]] + list(D.get("wire_rows", []))
    wires, clean_stats = U.clean_wires(wires_raw)
    # every end must land on a support, another wire or yard equipment
    sx_all = K["sup"]
    anch = [np.c_[sx_all.x, sx_all.y, sx_all.top_z]]
    if len(E):
        anch.append(np.c_[E.x, E.y, gz(E.x.to_numpy(float), E.y.to_numpy(float)) + E.top_hag])
    for t in Y["towers"]:
        anch.append([[t["x"], t["y"], float(gz(t["x"], t["y"])[0]) + t["height"]]])
    for q in Y["posts"]:
        anch.append([[q["x"], q["y"], float(gz(q["x"], q["y"])[0]) + q["top_hag"]]])
    for b in Y["blocks"]:
        c = b["centre"]
        anch.append([[c[0], c[1], float(gz(c[0], c[1])[0]) + b["z1"]]])
    anchors = np.vstack([np.asarray(a, float).reshape(-1, 3) for a in anch])
    free0 = U.free_ends(wires, anchors)
    ksag = float(np.median([w["sag"] / w["L"] ** 2 for _, _, w in K["res"]])) if K["res"] else 0.0
    wires, inferred, conn = U.connect_wires(wires, anchors, gz, ksag)
    # a weld or snap can move an end another end had already joined; a second pass re-joins those
    wires, inf2, conn2 = U.connect_wires(wires, np.vstack([anchors, inferred]) if len(inferred) else anchors, gz, ksag)
    inferred = np.vstack([inferred, inf2]) if len(inf2) else inferred
    _, left = U.free_ends(wires, np.vstack([anchors, inferred]) if len(inferred) else anchors, points=True)
    if len(left):  # whatever is still open ends on an assumed pole
        inferred = np.vstack([inferred, left + [0, 0, 0.3]]) if len(inferred) else left + [0, 0, 0.3]
        conn["inferred_pole"] += len(left)
    conn["second_pass"] = {k: conn2[k] for k in ("span_ahead", "drop_near", "inferred_pole", "dropped_wires")}
    conn["free_ends_before"] = free0
    conn["free_ends_after"] = U.free_ends(wires, np.vstack([anchors, inferred]) if len(inferred) else anchors)
    clean_stats["connect"] = conn
    print("wire connection:", conn, flush=True)
    if len(inferred):
        parts = [U.pole_part(p[0], p[1], float(gz(p[0], p[1])[0]), p[2] - float(gz(p[0], p[1])[0]) + 0.3, 0.2) for p in inferred]
        from lidar3d import infra3d as I3
        pm_ = R.tmesh(I3._merge(parts, (150, 110, 70)))
        if not photo:
            pm_["rgb"] = np.tile(np.array((200, 120, 60), np.uint8), (pm_.n_points, 1))
        layers["sup_inferred"] = pm_
    clean_stats["roughness_before_m"] = float(np.nanmedian([U.roughness(w) for w in wires_raw if len(w) > 3]))
    clean_stats["roughness_after_m"] = float(np.nanmedian([U.roughness(w) for w in wires if len(w) > 3]))
    tubes = []
    for w in wires:
        t = R.wire_tube(w, 0.07)
        if t is not None:
            t["rgb"] = np.tile(np.array((28, 30, 34) if photo else (20, 20, 20), np.uint8), (t.n_points, 1))
            tubes.append(t)
    layers["wires"] = SC.merge(tubes)
    layers["wires_lines"] = R.lines_mesh(wires)
    sx = list(N["K"]["sup"][N["K"]["sup"].tier == "support"].itertuples())
    tops = [(r.x, r.y, r.top_z) for r in sx if not Y["yard"].contains(Point(r.x, r.y))] + [tuple(p) for p in inferred] + [(t["x"], t["y"], float(gz(t["x"], t["y"])[0]) + t["height"]) for t in Y["towers"]]
    if tops:
        tops = [t for t in tops if not Y["yard"].buffer(3).contains(Point(t[0], t[1]))]
        pm = pv.PolyData(np.array(tops, float))
        layers["support_markers"] = pm.glyph(geom=pv.Sphere(radius=2.2, theta_resolution=10, phi_resolution=8), scale=False, orient=False)
        layers["support_markers"]["rgb"] = np.tile(np.array((235, 70, 50), np.uint8), (layers["support_markers"].n_points, 1))
    info = dict(tree_colour_fallbacks=int(locals().get("layers_tree_fallback", 0)), open_frames=len(frames), boxy_blocks=len(boxy), clean=clean_stats, wires_final=wires, wires_raw=wires_raw, displacement=fit, equipment_kinds={k: kinds.count(k) for k in set(kinds)}, trees=int(n), buildings=len(D["buildings"]))
    return layers, info


def drop_wire_trees(D, N, gz, tol=2.5):
    """Wire and tower returns the wire detector missed form narrow ridges in the canopy height model, and the watershed puts a 'tree top' on them. A top within `tol` m (3D) of a modelled wire,
    or within `tol` m (plan) and 4 m (height) of a support top, is not a tree: it is removed from the trees and its crown cells from the label grid. Runs once per detection."""
    if D.get("trees_wire_filtered"):
        return
    from scipy.spatial import cKDTree
    K, Y = N["K"], N["Y"]
    T = D["trees"]
    ws = list(K["span_w"]) + [a["xyz"] for a in K["adds"]] + [b["xyz"] for b in N["br"]] + [w["xyz"] for w in Y["wires"]]
    P = np.vstack([U.densify(np.asarray(w, float), 0.5) for w in ws if len(w) >= 2])
    tz = gz(np.asarray(T["x"], float), np.asarray(T["y"], float)) + T["height"]
    top = np.c_[T["x"], T["y"], tz]
    # plan distance with a height window: the smoothed canopy model sits a little under the wire, so a 3D test misses many
    dp, kp = cKDTree(P[:, :2]).query(top[:, :2])
    bad = (dp <= 2.0) & (np.abs(P[kp, 2] - tz) <= 4.0)
    bad |= cKDTree(P).query(top)[0] <= tol
    sup = np.r_[np.c_[K["sup"].x, K["sup"].y, K["sup"].top_z], np.array([[t["x"], t["y"], float(gz(t["x"], t["y"])[0]) + t["height"]] for t in Y["towers"]]).reshape(-1, 3)]
    d, k = cKDTree(sup[:, :2]).query(top[:, :2])
    bad |= (d <= tol) & (np.abs(sup[k, 2] - tz) <= 4.0)
    # rows of narrow, equal-height tops in a straight line are missed conductors: they leave the trees and join the wire model
    keep_ = ~bad
    rmask, rows = U.wire_rows(np.asarray(T["x"], float)[keep_], np.asarray(T["y"], float)[keep_], tz[keep_], np.asarray(T["radius"], float)[keep_])
    # a row hugging the window border is crowns cut by the border, not a conductor
    def edge_d(r):
        return np.median(np.minimum.reduce([r[:, 0] - WIN[0], WIN[1] - r[:, 0], r[:, 1] - WIN[2], WIN[3] - r[:, 1]]))
    xs_, ys_ = np.asarray(T["x"], float)[keep_], np.asarray(T["y"], float)[keep_]
    ok_rows = []
    for r in rows:
        if edge_d(r) < 6.0:
            near = cKDTree(r[:, :2]).query(np.c_[xs_, ys_])[0] <= 3.0
            rmask &= ~near
        else:
            ok_rows.append(r)
    rows = ok_rows
    rowbad = np.zeros(len(bad), bool); rowbad[np.flatnonzero(keep_)[rmask]] = True
    D["wire_rows"] = rows
    D["trees_removed_as_wire_rows"] = int(rowbad.sum())
    bad |= rowbad
    print("missed conductors from rows of canopy tops:", len(rows), "rows,", int(rowbad.sum()), "tops", flush=True)
    rm = np.asarray(T["id"])[bad]
    D["tree_lab"] = np.where(np.isin(D["tree_lab"], rm), 0, D["tree_lab"])
    D["trees"] = {kk: (np.asarray(v)[~bad] if np.ndim(v) == 1 and len(v) == len(bad) else v) for kk, v in T.items()}
    D["trees_removed_on_wires"] = int(bad.sum())
    D["trees_wire_filtered"] = True
    print("tree tops on wires or supports removed:", int(bad.sum()), "of", len(bad), flush=True)


def fit_h(T):
    """Height used to look up a crown's image colour: two thirds of the way up the crown, where most of the visible foliage is."""
    return T["base"] + 0.66 * (T["height"] - T["base"])


if __name__ == "__main__":
    S = prepare()
    if len(sys.argv) > 1 and sys.argv[1] == "maps":
        make_maps()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] in ("detect", "network"):
        D = detect(S)
        if sys.argv[1] == "network":
            network(S, D)
        sys.exit(0)
    print("points", len(S["d"]["x"]), "ground cells", S["G"]["z"].shape, "filled share", float(S["G"]["filled"].mean()), "naip", S["naip_date"])
