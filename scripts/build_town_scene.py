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
    from shapely import contains_xy
    from shapely.geometry import box as sbox
    d, G, hag = S["d"], S["G"], S["hag"]
    x, y, z, cls = d["x"].astype(float), d["y"].astype(float), d["z"].astype(float), d["cls"]
    xyz = np.c_[x, y, z]
    out = {}
    # ---- yard polygon and wires
    feats = BM.overture(d["cx"], d["cy"])
    yard = max(feats["substation"], key=lambda g: g.area)
    inyard = contains_xy(yard.buffer(6), x, y)
    wmask, wlab, winfo = utility.detect_wires(xyz, hag)
    out.update(yard=yard, inyard=inyard, wmask=wmask, wlab=wlab)
    print("wires", int(wmask.sum()), flush=True)
    # ---- imagery products
    img = S["img"]
    ndvi_img = (img[3] - img[0]) / np.maximum(img[3] + img[0], 1.0)
    fit = CL.fit_relief_displacement(ndvi_img, S["tf"], d["cx"], d["cy"], x, y, hag)
    out["displacement"] = fit
    print("displacement", fit, flush=True)
    # ---- buildings: planar returns outside the yard
    ccx, ccy = (WIN[0] + WIN[1]) / 2, (WIN[2] + WIN[3]) / 2
    C = RT.Cloud((x - ccx).astype("float32"), (y - ccy).astype("float32"), z.astype("float32"), cls, d["nret"], "2020-04-13", 335.0, d["cx"] + ccx, d["cy"] + ccy)
    dtm, hagC, dsm = RT.surfaces(C, 1.0)
    use = RT.planar_points(C, hagC) & ~inyard & ~wmask
    lab, polys = RT.footprints(C, hagC, use=use, min_area=30.0)
    blds = RT.build_buildings(C, hagC, dtm, lab, polys, use=use)
    out["buildings"] = blds
    out["bld_offset"] = (ccx, ccy)
    print("buildings", len(blds), "of", len(polys), "footprints", flush=True)
    # ---- trees from what is left: not ground, not wire, not building, not yard
    bmask = np.zeros(len(x), bool)
    if len(polys):
        for q in polys:
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
