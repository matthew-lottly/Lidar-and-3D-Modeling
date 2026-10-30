"""Accuracy assessment of the Pacifica differencing against real data: two independent surveys from 2010 (ARRA-CA_GoldenGate_2010 and ARRA-CA_SanFranCoast_2010) of the same ground. Nothing moved between them, so
running the whole pipeline (registration, DEM of difference, level of detection, volumes, M3C2) on that pair measures what the pipeline reports when the true change is zero: the false-alarm volume and area.
Usage: python scripts/validate_topodiff.py   (after scripts/build_topodiff.py)"""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from lidar3d import topodiff as TD
from lidar3d import validation as V
from lidar3d.ept import CACHE

SITE = dict(slug="pacifica", lonlat=(-122.4985, 37.6175), half=450.0, crs="EPSG:32610")
HALF, RES = SITE["half"], 2.0
DATA = ROOT / "data" / "processed" / "validation"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    st = json.loads((ROOT / "data" / "processed" / "topodiff" / "stats.json").read_text())
    F = pickle.loads((CACHE / "topodiff" / "final.pkl").read_bytes())
    a = TD.load_epoch(SITE, "2010", "ARRA-CA_SanFranCoast_2010", log=lambda s: None)
    b = TD.load_epoch(SITE, "2010gg", "ARRA-CA_GoldenGate_2010", log=lambda s: None)
    out = dict(analysis="Pacifica topographic differencing", pair=dict(a="ARRA-CA_SanFranCoast_2010", b="ARRA-CA_GoldenGate_2010", points=[int(len(a.x)), int(len(b.x))], ground=[int((a.cls == 2).sum()), int((b.cls == 2).sum())]))
    Da, _ = TD.ground_dem(a, HALF, RES, 1)
    Db, _ = TD.ground_dem(b, HALF, RES, 1)
    both = np.isfinite(Da) & np.isfinite(Db)
    out["pair"]["cells_with_both"] = int(both.sum())
    out["pair"]["coverage_of_study_land"] = float(both.sum() / max(np.isfinite(F["D"]["2023"]).sum(), 1))
    stable = F["stable"] & both
    out["pair"]["stable_cells_with_both"] = int(stable.sum())
    if both.sum() < 2000:
        out["status"] = "the two 2010 surveys barely overlap; no null test possible"
        (DATA / "topodiff.json").write_text(json.dumps(out, indent=1, default=float))
        print(out)
        return
    dz_raw = Db - Da
    raw = dz_raw[both]
    out["raw_difference_all_overlap"] = V.error_summary(raw)
    # the two surveys can sit on different vertical datums (ellipsoid against NAVD88 differs by tens of metres on this coast); measure the offset on stable ground and remove it, as the real analysis does with ICP
    off = float(np.median(dz_raw[F["stable"] & both])) if (F["stable"] & both).any() else float(np.median(raw))
    out["vertical_offset_between_surveys_m"] = off
    b = TD.Epoch(b.tag, b.project, b.date, b.x, b.y, (b.z - off).astype("float32"), b.cls, b.nret)
    Db = Db - off
    dz_raw = Db - Da
    out["difference_after_removing_offset_stable"] = V.error_summary(dz_raw[F["stable"] & both])
    # null test with the same procedure as the real intervals: sigma_z from stable ground, slope-dependent level of detection, volumes in the coastal band
    slope = F["slope"]
    dist = F["dist_sea"]
    coast = both & (dist <= 60.0)
    if stable.sum() > 50:
        sz = TD.error_sd(dz_raw[stable])
        # the same registration step as in the real analysis: ICP of b onto a on stable ground
        iy, ix, _ = TD._cell_index(b.x, b.y, HALF, RES)
        iy2, ix2, _ = TD._cell_index(a.x, a.y, HALF, RES)
        src = b.xyz[(b.cls == 2) & F["stable"][iy, ix]].astype("float64")
        dst = a.xyz[(a.cls == 2) & F["stable"][iy2, ix2]].astype("float64")
        reg = None
        if len(src) > 500 and len(dst) > 500:
            rng = np.random.default_rng(0)
            R, t, _ = TD.icp_point_to_plane(src[rng.choice(len(src), min(len(src), 100000), replace=False)], dst[rng.choice(len(dst), min(len(dst), 200000), replace=False)])
            reg = dict(translation_m=[float(v) for v in t])
            Db2, _ = TD.ground_dem(TD.apply(R, t, b), HALF, RES, 1)
        else:
            Db2 = Db
        out["registration"] = reg
        dz = Db2 - Da
        sz2 = TD.error_sd(dz[stable & np.isfinite(dz)])
        lod = TD.level_of_detection(slope, sz2, 0.07)
        v = TD.dod_volumes(dz, lod, RES, coast)
        v_all = TD.dod_volumes(dz, lod, RES, both)
        out["null_test"] = dict(sigma_z_stable_m=sz2, sigma_z_before_registration_m=sz, coastal_band=v, whole_overlap=v_all,
                                false_alarm_share_of_coastal_cells=float(v["significant_cells"] / max(v["cells"], 1)),
                                compare_real_2016_2023_coastal_net_m3=st["dod"]["2016-2023"]["coastal_band_60m"]["lod95"]["net_m3"], compare_real_2010_2016_coastal_net_m3=st["dod"]["2010-2016"]["coastal_band_60m"]["lod95"]["net_m3"])
        # errors by slope class on the null pair, as a check on the slope-dependent level of detection
        bys = {}
        for lo, hi in ((0, 5), (5, 15), (15, 30), (30, 90)):
            m = both & (slope >= lo) & (slope < hi) & np.isfinite(dz)
            if m.sum() > 30:
                bys[f"{lo}-{hi}"] = dict(cells=int(m.sum()), nmad_m=V.nmad(dz[m]), share_above_lod=float((np.abs(dz[m]) > lod[m]).mean()), median_lod_m=float(np.median(lod[m])))
        out["null_test"]["by_slope"] = bys
    (DATA / "topodiff.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3500])


if __name__ == "__main__":
    main()
