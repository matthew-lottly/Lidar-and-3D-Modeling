"""3D scene of the White River utility model: lidar ground and objects coloured from NAIP imagery, with fitted supports, cross-arms, insulators and wires as geometry.
Usage: python scripts/build_spans_3d.py   (after scripts/build_spans.py). Writes the GLB, the HTML viewer and plates 25 to 27 via lidar3d.figures_spans."""
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
import trimesh
from scipy.spatial import cKDTree

import build_utility as BU
from lidar3d import exports, infra3d, naip, render3d
from lidar3d.aoi import AOI
from lidar3d.ept import CACHE

HALF = BU.HALF


def sample_rgb(img, tf, cx, cy, x, y):
    c = ((x + cx - tf.c) / tf.a).astype(int).clip(0, img.shape[2] - 1)
    r = ((y + cy - tf.f) / tf.e).astype(int).clip(0, img.shape[1] - 1)
    return np.stack([img[b, r, c] for b in range(3)], 1).astype(float)


def build(F=None):
    x, y, z, cls, date, cx, cy, tr = BU.load_points()
    F = F or pickle.loads((CACHE / "utility" / "spans_final.pkl").read_bytes())
    sup, tiers, res, gg, hag = F["sup"], F["tiers"], F["res"], F["gg"], F["hag"]
    img, tf, ndate = naip.fetch(AOI(round(cx), round(cy), HALF + 20, "EPSG:32610"), 2019, slug="whiteriver")
    img = np.asarray(img)
    if img.max() > 255:
        img = img / 256.0
    # ground from the lidar ground grid, imagery draped and lightly shaded
    n = gg.shape[0]
    gx, gy = np.meshgrid(np.arange(n) + 0.5 - HALF, np.arange(n) + 0.5 - HALF)
    gcol = sample_rgb(img, tf, cx, cy, gx.ravel(), gy.ravel())
    hs = render3d.shade_texture(gg, 1.0, gcol.reshape(n, n, 3).astype(np.uint8), strength=0.35)
    ground = np.c_[gx.ravel(), gy.ravel(), gg.ravel()]
    gcol = hs.reshape(-1, 3).astype(float)
    # a 0.5 m version for close views: heights interpolated from the 1 m grid, colours from the 0.6 m imagery
    from scipy.interpolate import RegularGridInterpolator as RGI
    fx = np.arange(0.25, 2 * HALF, 0.5) - HALF
    fgx, fgy = np.meshgrid(fx, fx)
    fz = RGI((gy[:, 0], gx[0]), gg, bounds_error=False, fill_value=None)(np.c_[fgy.ravel(), fgx.ravel()])
    hsf = render3d.shade_texture(fz.reshape(fgx.shape), 0.5, sample_rgb(img, tf, cx, cy, fgx.ravel(), fgy.ravel()).reshape(*fgx.shape, 3).astype(np.uint8), strength=0.35)
    ground_fine = np.c_[fgx.ravel(), fgy.ravel(), fz]
    gcol_fine = hsf.reshape(-1, 3).astype(float)
    # everything else the lidar saw above the ground, minus returns that the wire and support models already explain
    wire_pts, wflag = infra3d.sample_wires(res, 0.5)
    keep = (hag >= 0.4) & (cls != 7)
    rng = np.random.default_rng(3)
    idx = np.flatnonzero(keep)
    idx = idx[(rng.random(len(idx)) < 0.35) & (hag[idx] < 45)]
    obj = np.c_[x[idx], y[idx], z[idx]]
    near_w = cKDTree(wire_pts).query(obj)[0] <= 1.2
    near_s = cKDTree(sup[["x", "y"]].to_numpy()).query(obj[:, :2])[0] <= 4.0
    gone = near_w | (near_s & (hag[idx] > 3))
    # points high above the ground and not part of a mass are raw wire or tower returns: the model draws those
    obj, hg = obj[~gone], hag[idx][~gone]
    ocol = sample_rgb(img, tf, cx, cy, obj[:, 0], obj[:, 1])
    shade = np.clip(0.78 + 0.22 * np.minimum(hg, 25) / 25, 0, 1.15)[:, None]
    ocol = np.clip(ocol * shade, 0, 255)
    meshes = infra3d.support_meshes(sup, tiers)
    wires = infra3d.wire_mesh(res, 2.0, 0.045)
    spts, scol = [], []
    for m in meshes.values():
        p, c = infra3d.sample_mesh(m, 260.0)
        spts.append(p); scol.append(c)
    return dict(ground=ground, gcol=gcol, ground_fine=ground_fine, gcol_fine=gcol_fine, obj=obj, ocol=ocol, meshes=meshes, wires=wires, wire_pts=wire_pts, wflag=wflag, sup_pts=np.vstack(spts), sup_col=np.vstack(scol).astype(float), cx=cx, cy=cy, img=img, tf=tf, ndate=ndate, date=date)


def scene_cloud(S, mode="photo", extent=None):
    """(points, colours) for the software renderer. mode 'photo': imagery colours, wires dark. 'analysis': wires blue where observed and cyan where filled, scene muted."""
    fine = extent is not None
    g, gc = (S["ground_fine"], S["gcol_fine"]) if fine else (S["ground"], S["gcol"])
    o, oc = S["obj"], S["ocol"]
    if not fine:
        k = np.arange(len(o)) % 3 == 0
        o, oc = o[k], oc[k]
    if mode == "analysis":
        gc = 0.35 * gc + 0.65 * 235; oc = 0.35 * oc + 0.65 * 235
    wc = np.where(S["wflag"][:, None] == 1, np.array([[28, 78, 140]]), np.array([[23, 162, 184]])) if mode == "analysis" else np.tile(np.array([[40, 40, 45]]), (len(S["wflag"]), 1))
    pts = np.vstack([g, o, S["wire_pts"], S["sup_pts"]])
    col = np.vstack([gc, oc, wc, S["sup_col"] if mode == "photo" else np.tile(np.array([[178, 43, 138]]), (len(S["sup_pts"]), 1))])
    if extent is not None:
        x0, x1, y0, y1 = extent
        m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        pts, col = pts[m], col[m]
    return pts, np.clip(col, 0, 255).astype(np.uint8)


def export(S, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = trimesh.Scene()
    for nm, m in S["meshes"].items():
        sc.add_geometry(m, geom_name=nm)
    sc.add_geometry(S["wires"], geom_name="wires")
    sc.export(out_dir / "white_river_lines_and_supports.glb")
    n = int(np.sqrt(len(S["ground"])))
    exports.terrain_mesh(S["ground"][:, 2].reshape(n, n)[::-1], 0, 0, 2, S["gcol"].reshape(n, n, 3).astype(np.uint8)[::-1], path_glb=out_dir / "white_river_ground_naip.glb")
    pts, col = scene_cloud(S, "photo")
    sub = np.random.default_rng(0).random(len(pts)) < 0.5
    exports.html_viewer(pts[sub], col[sub], out_dir / "white_river_utility_viewer.html", len(pts), S["date"], S["date"], "model + lidar, NAIP colours", title="White River Substation: fitted lines and supports in the lidar scene")


if __name__ == "__main__":
    S = build()
    export(S, ROOT / "outputs" / "utility3d")
    from lidar3d import figures_spans
    st = json.loads((ROOT / "data" / "processed" / "spans" / "stats.json").read_text())
    F = pickle.loads((CACHE / "utility" / "spans_final.pkl").read_bytes())
    site = dict(project=BU.PROJECT, collected=S["date"], naip=S["ndate"])
    pts, col = scene_cloud(S, "photo")
    over, _ = render3d.render(pts, col, az=215, el=32, width=2400, height=1350, splat=2)
    figures_spans.fig_model(site, st, F)
    figures_spans.fig_scene(site, st, over)
    ims, caps = [], []
    for ext, az, nm in (((-300, -110, -160, 10), 215, "west corridors"), ((190, 400, -25, 115), 150, "east corridor")):
        p1, c1 = scene_cloud(S, "photo", ext)
        im1, info = render3d.render(p1, c1, az=az, el=24, width=1600, height=1350, splat=2)
        p2, c2 = scene_cloud(S, "analysis", ext)
        im2, _ = render3d.render(p2, c2, az=az, el=24, width=1600, height=1350, splat=2, center=info["center"], scale=info["scale"], off=info["off"])
        ims += [im1, im2]; caps += [(f"{nm}: imagery colours", "Lidar returns coloured from NAIP imagery, with the fitted supports, cross-arms, insulators and wires drawn as geometry."),
                                    (f"{nm}: observed or filled", "Same view. Wires are dark blue where lidar returns support them and cyan where the fitted curve fills the gap; supports are magenta; the rest is faded.")]
    figures_spans.fig_closeups(site, st, ims, caps)
