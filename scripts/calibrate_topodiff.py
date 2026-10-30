"""Re-test the Pacifica real intervals with a level of detection calibrated on the null test (run after validate_topodiff.py).
The null test (two independent 2010 surveys, no real change) showed the formula level of detection leaves about 47% of coastal cells 'significant'. Here the detection threshold per slope class is set to 1.96 times the
null test's robust spread (NMAD) in that class, and the real intervals' coastal volumes are recomputed with it. Volumes below that threshold are not claimed."""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from lidar3d import topodiff as TD
from lidar3d.ept import CACHE

DATA = ROOT / "data" / "processed" / "validation"
RES = 2.0


def main():
    nul = json.loads((DATA / "topodiff.json").read_text())["null_test"]["by_slope"]
    edges = [(0, 5), (5, 15), (15, 30), (30, 90)]
    sig = {f"{lo}-{hi}": 1.96 * nul[f"{lo}-{hi}"]["nmad_m"] for lo, hi in edges}
    F = pickle.loads((CACHE / "topodiff" / "final.pkl").read_bytes())
    slope = F["slope"]
    lod = np.full(slope.shape, np.nan, "float32")
    for lo, hi in edges:
        lod[(slope >= lo) & (slope < hi)] = sig[f"{lo}-{hi}"]
    lod[~np.isfinite(lod)] = max(sig.values())
    coast = F["dist_sea"] <= 60.0
    out = dict(calibrated_lod_m=sig, intervals={})
    for (a, b), dz in F["dods"].items():
        m = coast & np.isfinite(dz)
        formula = TD.dod_volumes(dz, F["lods"][(a, b)], RES, m)
        calib = TD.dod_volumes(dz, lod, RES, m)
        out["intervals"][f"{a}-{b}"] = dict(formula_lod=dict(net_m3=formula["net_m3"], erosion_m3=formula["erosion_m3"], deposition_m3=formula["deposition_m3"], significant_area_m2=formula["significant_area_m2"]),
                                          null_calibrated_lod=dict(net_m3=calib["net_m3"], erosion_m3=calib["erosion_m3"], deposition_m3=calib["deposition_m3"], significant_area_m2=calib["significant_area_m2"]))
    (DATA / "topodiff_calibrated.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
