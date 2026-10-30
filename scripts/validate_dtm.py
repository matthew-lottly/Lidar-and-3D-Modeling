"""Ground surface of the other studies checked against the USGS-processed 3DEP DTM of the same project (2 m), read from the Planetary Computer (realref.lidar_product). The DTM comes from USGS's own ground
classification and gridding of the same points, so this catches gridding and ground-class errors, not a flaw in the flight. Our grid here is the mean of class-2 returns per 2 m cell, with no filling.
Usage: python scripts/validate_dtm.py"""
import json
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from pyproj import Transformer
from scipy import ndimage

from lidar3d import realref as RR
from lidar3d import validation as V
from lidar3d.aoi import AOI
from lidar3d.ept import CACHE, EPT

DATA = ROOT / "data" / "processed" / "validation"
SITES = [
    ("seattle_center", "WA_KingCo_1_2021", (-122.3510, 47.6215), 350.0, "EPSG:32610"),
    ("capecod_2014_ref_Sandy_Z19_2013", "MA_NE_CMGP_Sandy_Z19_A1_2015", (-69.97151507, 41.91278736), 300.0, "EPSG:32619"),
    ("capecod_2014_ref_Sndy_CMPG_2013", "MA_NE_CMGP_Sandy_Z19_A1_2015", (-69.97151507, 41.91278736), 300.0, "EPSG:32619"),
    ("capecod_2021", "MA_CentralEastern_2_2021", (-69.97151507, 41.91278736), 300.0, "EPSG:32619"),
    ("wabash_mtcarmel", "IL_WabashRiver_2010", (-87.93, 38.31), 1750.0, "EPSG:32616"),
    ("whiteriver_utility", "WA_PierceCounty_1_2020", (-122.21035, 47.23898), 400.0, "EPSG:32610"),
    ("tatoosh_rainier", "WA_CentralWildfire_2_D22", None, None, None),
]


def one(slug, project, lonlat, half, crs, ref_project=None):
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*lonlat)
    aoi = AOI(round(cx), round(cy), half, crs)
    W, S, E, N = aoi.bbox_ll
    res = 2.0
    n = int(2 * half / res)
    ref = RR.lidar_product("dtm", ref_project or project, (W, S, E, N), aoi.cx - half, aoi.cy + half, res, (n, n), crs)
    if ref is None or not np.isfinite(ref).any():
        return dict(project=project, reference_project=ref_project or project, status="no USGS DTM product found over the window (USGS publishes these products for older projects only)")
    c = EPT(project).fetch(aoi.bbox_ll, max_depth=20, log=lambda s: None)
    x0, y0, x1, y1 = aoi.bounds
    m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & (c.cls == 2)
    ix = np.clip(((c.x[m] - x0) / res).astype(int), 0, n - 1)
    iy = np.clip(((c.y[m] - y0) / res).astype(int), 0, n - 1)
    k = iy * n + ix
    s = np.bincount(k, c.z[m].astype("float64"), minlength=n * n)
    cnt = np.bincount(k, minlength=n * n)
    mine = np.where(cnt >= 2, s / np.maximum(cnt, 1), np.nan).reshape(n, n)
    r = ref[::-1]
    d = mine - r
    ok = np.isfinite(d)
    gy, gx = np.gradient(np.where(np.isfinite(r), r, np.nanmedian(r)), res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    out = dict(project=project, reference_project=ref_project or project, cells=int(ok.sum()), ground_points=int(m.sum()), all=V.error_summary(d[ok]), correlation=float(np.corrcoef(mine[ok], r[ok])[0, 1]))
    for lo, hi in ((0, 5), (5, 15), (15, 90)):
        sel = ok & (slope >= lo) & (slope < hi)
        out[f"slope_{lo}_{hi}"] = V.error_summary(d[sel])
    out["note"] = "difference = ours minus USGS, class-2 mean per 2 m cell against USGS's 2 m DTM; a second processing of the same survey"
    return out


REFS = {"capecod_2014_ref_Sandy_Z19_2013": "USGS_LPC_MA_NE_CMGP_Sandy_Z19_2013", "capecod_2014_ref_Sndy_CMPG_2013": "USGS_LPC_MA_Sndy_CMPG_2013_LAS_2015"}


def SITES_LEFT(slug):
    names = [x[0] for x in SITES]
    return names[names.index(slug) + 1:]


def main():
    only = sys.argv[1:]
    DATA.mkdir(parents=True, exist_ok=True)
    f = DATA / "dtm_other_studies.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    for slug, project, lonlat, half, crs in SITES:
        if only and slug not in only:
            continue
        if lonlat is None:
            out[slug] = dict(project=project, status="not run: the Rainier window is a snow and rock slope and USGS publishes no DTM product for this project")
            continue
        try:
            out[slug] = one(slug, project, lonlat, half, crs, REFS.get(slug))
        except Exception as e:
            out[slug] = dict(project=project, status=f"not run: {str(e)[:120]}")
        print(slug, json.dumps(out[slug], default=float)[:400], flush=True)
        if not any(k.startswith("capecod_2014") for k in SITES_LEFT(slug)):
            shutil.rmtree(CACHE / "ept" / project, ignore_errors=True)
        (DATA / "dtm_other_studies.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
