"""Plates 39 to 45 and 3D views for the rooftop study (run after scripts/build_rooftops.py all)."""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import trimesh

from lidar3d import figures_rooftop as FR
from lidar3d import naip, render3d
from lidar3d import rooftop as RT
from lidar3d.ept import CACHE

NAIP_YEAR = 2018


def scene_image(key, az=205, el=38, width=1800, height=1300):
    sc = trimesh.load(ROOT / "outputs" / "rooftops" / key / "block.glb")
    pts, cols = [], []
    for g in sc.geometry.values():
        n = int(max(g.area * 40, 100))
        P, fi = trimesh.sample.sample_surface(g, n)
        vc = np.asarray(g.visual.vertex_colors)[g.faces[fi, 0]][:, :3]
        pts.append(P); cols.append(vc)
    P, Cc = np.vstack(pts), np.vstack(cols).astype(np.uint8)
    im, _ = render3d.render(P, Cc, az=az, el=el, width=width, height=height, splat=2)
    return im


def main():
    sts, Fs, imgs, scenes = {}, {}, {}, {}
    for key in ("westu", "downtown"):
        st = json.loads((ROOT / "data" / "processed" / "rooftops" / f"{key}.json").read_text())
        F = pickle.loads((CACHE / "rooftop" / f"{key}_final.pkl").read_bytes())
        F["polys"] = {int(k): v for k, v in F["polys"].items()}
        sts[key], Fs[key] = st, F
        site = RT.SITES[key]
        img, _, nd = naip.fetch(RT.make_aoi(site, site["half"]), NAIP_YEAR, res=1.0, slug=f"rooftop_{key}")
        imgs[key] = img
        scenes[key] = scene_image(key)
    FR.fig_sites(sts, imgs)
    FR.fig_extraction("westu", sts["westu"], Fs["westu"])
    FR.fig_solar("westu", sts["westu"], Fs["westu"], "41")
    FR.fig_extraction("downtown", sts["downtown"], Fs["downtown"])
    FR.fig_solar("downtown", sts["downtown"], Fs["downtown"], "43")
    FR.fig_3d(sts, scenes)
    FR.fig_tests(sts)


if __name__ == "__main__":
    main()
