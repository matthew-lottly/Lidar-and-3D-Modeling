"""Canopy study checked against independent real data (run after scripts/build_canopy.py has been run; reads the lidar again from EPT):
 - the USGS-processed DTM of the same project (2 m): a second processing of the same points, for the ground surface;
 - ESA WorldCover 2021 and Esri 10 m land cover 2022 (satellite-derived, independent of lidar): tree cover at 10 m, as agreement of 'canopy' against 'trees'.
Usage: python scripts/validate_remote.py houston|tahoe"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

import build_canopy as BC
from lidar3d import forestry as F
from lidar3d import realref as RR
from lidar3d import validation as V

DATA = ROOT / "data" / "processed" / "validation"


def block(a, k):
    n = (a.shape[0] // k) * k
    return a[:n, :n].reshape(n // k, k, n // k, k).mean(axis=(1, 3))


def main(slug):
    DATA.mkdir(parents=True, exist_ok=True)
    site = dict(F.SITES[slug])
    half = site["half"]
    n = int(2 * half)
    aoi = F.make_aoi(site)
    W, S, E, N = aoi.bbox_ll
    x0, y1 = aoi.cx - half, aoi.cy + half
    R = BC.run_pipeline(F.load_points(site, log=lambda s: None), half, max_h=site["max_height"])
    dtm, chm = R["dtm"], R["chm"]
    out = dict(analysis=f"Canopy structure ({slug})", project=site["project"])
    ref = RR.lidar_product("dtm", site["project"], (W, S, E, N), x0, y1, 1.0, (n, n), aoi.crs)
    if ref is not None and np.isfinite(ref).any():
        best = None
        for flip in (True, False):
            r = ref[::-1] if flip else ref
            ok = np.isfinite(r) & np.isfinite(dtm)
            c = np.corrcoef(r[ok], dtm[ok])[0, 1] if ok.sum() > 100 else -1
            if best is None or c > best[0]:
                best = (c, flip, r, ok)
        c, flip, r, ok = best
        open_ = ok & (chm < 1.0)
        out["usgs_dtm"] = dict(V.error_summary((dtm - r)[open_]), grid_flipped=bool(flip), correlation=float(c), n_cells_open_ground=int(open_.sum()), under_canopy=V.error_summary((dtm - r)[ok & (chm >= 2)]),
                               note="USGS's own DTM of the same survey, 2 m resampled to 1 m: a second processing of the same points, so it checks the ground model and not the flight")
    else:
        out["usgs_dtm"] = dict(status="no USGS DTM product found for this project over the window")
    cover = block((chm >= 2.0).astype("float32")[::-1] if out.get("usgs_dtm", {}).get("grid_flipped", True) else (chm >= 2.0).astype("float32"), 10)
    m10 = cover.shape[0]
    args = (x0, y1, 10.0, (m10, m10), aoi.crs)
    for key, fn, tree, nm in (("worldcover_2021", lambda: RR.worldcover((W, S, E, N), *args), 10, "ESA WorldCover 2021 (class tree cover)"), ("esri_landcover_2022", lambda: RR.esri_lulc((W, S, E, N), *args), 2, "Esri 10 m land cover 2022 (class trees)")):
        try:
            lc = fn()
            if lc is None:
                raise RuntimeError("no tiles")
            valid = np.isfinite(lc)
            reft = np.where(valid, lc == tree, False)
            pred = cover >= 0.5
            out[key] = dict(reference=nm, cells_10m=int(valid.sum()), canopy_cover_lidar=float(cover[valid].mean()), tree_share_reference=float(reft[valid].mean()), agreement=V.binary_agreement(pred, reft, valid),
                            note="our canopy = cells whose 1 m CHM is at least 2 m, aggregated to 10 m and called tree when at least half the cell is canopy; satellite land cover has its own error of roughly 10 percent")
        except Exception as e:
            out[key] = dict(status=f"not run: {str(e)[:100]}")
    (DATA / f"canopy_{slug}_remote.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3500])


if __name__ == "__main__":
    main(sys.argv[1])
