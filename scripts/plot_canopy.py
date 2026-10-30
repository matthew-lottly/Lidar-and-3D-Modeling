"""Plates and interactive maps for the canopy study (run after scripts/build_canopy.py has produced both sites).
Usage: python scripts/plot_canopy.py"""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from lidar3d import canopy_map, naip
from lidar3d import figures_canopy as FC
from lidar3d import forestry as F
from lidar3d.ept import CACHE

NUM = dict(houston="33", tahoe="35")


def interactive(key, st, Fz):
    n = Fz["chm"].shape[0]
    chm = Fz["chm"]
    vmax = float(np.percentile(chm[chm >= 2], 99.5))
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    rgba = (cm.YlGn(Normalize(0, vmax)(np.clip(chm, 0, vmax))) * 255).astype(np.uint8)
    rgba[chm < 0.5] = (242, 239, 228, 255)
    dens = np.kron(Fz["dens"], np.ones((50, 50)))[:n, :n]
    drgba = (cm.magma_r(Normalize(0, max(np.percentile(Fz["dens"], 99), 1))(dens)) * 255).astype(np.uint8)
    img = np.zeros((n, n, 4), np.uint8); img[..., :3] = (242, 239, 228); img[..., 3] = 255
    img[chm >= 2, :3] = (46, 125, 50)
    sizes = np.concatenate([[0], np.bincount(Fz["gl"].ravel())[1:]])[Fz["gl"]] if Fz["gl"].max() else np.zeros_like(Fz["gl"])
    for lo, hi, col in ((10, 100, (252, 217, 89)), (100, 1000, (242, 140, 38)), (1000, 1e4, (204, 64, 38)), (1e4, 1e12, (115, 26, 64))):
        img[(Fz["gl"] > 0) & (sizes >= lo) & (sizes < hi), :3] = col
    g = Fz["graph"]
    cimg = FC._cluster_image(g, Fz["cl10"], Fz["ca10"], (chm >= 2) & (g["labels"] == 0))
    crgba = np.concatenate([(cimg * 255).astype(np.uint8), np.full((n, n, 1), 255, np.uint8)], 2)
    from matplotlib.colors import LogNorm
    reach = Fz["reach10"] / 1e4
    nrm = LogNorm(max(reach.min(), 1e-3), reach.max())
    rr = (cm.viridis(nrm(np.clip(reach, nrm.vmin, nrm.vmax))) * 255).astype(np.uint8)
    rimg = np.zeros((n, n, 4), np.uint8); rimg[..., :3] = (247, 245, 238); rimg[..., 3] = 255
    rimg[(chm >= 2) & (g["labels"] == 0), :3] = (204, 204, 204)
    mm = g["labels"] > 0
    rimg[mm] = np.concatenate([[[247, 245, 238, 255]], rr])[g["labels"][mm]]
    t = Fz["trees"]
    pts = list(zip(t["col"] + 0.5, t["row"] + 0.5, t["height"], t["crown_diameter_m"]))
    out = ROOT / "outputs" / "canopy" / key / "canopy_map.html"
    canopy_map.write_map(out, st["site"], float(n), [("Canopy height", F.rgba_png_b64(rgba)), ("Tree density (per ha)", F.rgba_png_b64(drgba)), ("Canopy openings by size", F.rgba_png_b64(img)), ("Connectivity: clusters across 10 m gaps", F.rgba_png_b64(crgba)), ("Connectivity: reachable canopy per patch", F.rgba_png_b64(rimg))], pts, vmax,
                         note=f"Lidar {st['collected']}, {st['area_ha']:.0f} ha, 1 m cells. Trees: local maxima of a smoothed canopy height model; crowded trees are undercounted.")
    return out


def main():
    sts, Fzs = {}, {}
    for key in ("houston", "tahoe"):
        st = json.loads((ROOT / "data" / "processed" / "canopy" / f"{key}.json").read_text())
        Fz = pickle.loads((CACHE / "forestry" / f"{key}_final.pkl").read_bytes())
        site = F.SITES[key]
        st["site"] = site["name"]
        img, _, _ = naip.fetch(F.make_aoi(site), {"houston": 2018, "tahoe": 2022}[key], res=1.0, slug=site["slug"])
        _, county, land = FC.fig_site(key, st, img)
        st["location"] = dict(county=county, public_land=land)
        (ROOT / "data" / "processed" / "canopy" / f"{key}.json").write_text(json.dumps(st, indent=1, default=float))
        FC.fig_results(key, st, Fz, NUM[key])
        interactive(key, st, Fz)
        sts[key] = st
        Fzs[key] = Fz
    FC.fig_tests(sts)
    FC.fig_conn_maps(sts, Fzs)
    FC.fig_conn_graph(sts, Fzs)
    print({k: v["location"] for k, v in sts.items()})


if __name__ == "__main__":
    main()
