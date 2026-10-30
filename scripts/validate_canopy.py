"""Accuracy assessment of the canopy study against real data: the Tahoe forest was flown twice (CA_NoCAL_Wildfires_B1_2018 and the analysed CA_SierraNevada_5_2022) with different sensors and point densities.
The same pipeline run on the 2018 survey gives an independent measurement of the same trees: the share of trees found in both, their height difference (expected: growth of a few decimetres in 4 years, not noise),
canopy cover and CHM agreement. Houston has no second survey and is checked against NAIP, WorldCover and others (see scripts/validate_remote.py).
Usage: python scripts/validate_canopy.py   (after scripts/build_canopy.py tahoe)"""
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
from lidar3d import validation as V

DATA = ROOT / "data" / "processed" / "validation"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    site = dict(F.SITES["tahoe"])
    half = site["half"]
    P22 = F.load_points(site, log=lambda s: None)
    alt = dict(site, slug="tahoe_nf_2018", project="USGS_LPC_CA_NoCAL_Wildfires_B1_2018")
    P18 = F.load_points(alt, log=lambda s: None)
    R22 = BC.run_pipeline(P22, half, max_h=site["max_height"])
    R18 = BC.run_pipeline(P18, half, max_h=site["max_height"])
    out = dict(analysis="Canopy structure (Tahoe forest)", pair=dict(analysed=site["project"], reference="USGS_LPC_CA_NoCAL_Wildfires_B1_2018", dates=[P22.date, P18.date], density_per_m2=[float(len(P22.x) / (2 * half) ** 2), float(len(P18.x) / (2 * half) ** 2)],
                                                                    ground_density_per_m2=[float((P22.cls == 2).sum() / (2 * half) ** 2), float((P18.cls == 2).sum() / (2 * half) ** 2)], vegetation_classes_present_2018=bool(R18["masks"][3])))
    c22, c18 = R22["chm"], R18["chm"]
    out["canopy_cover"] = dict(cover_2022=float((c22 >= 2).mean()), cover_2018=float((c18 >= 2).mean()), agreement=V.binary_agreement(c22 >= 2, c18 >= 2))
    ok = (c22 >= 2) & (c18 >= 2)
    d = (c22 - c18)[ok]
    out["chm_where_both_canopy"] = dict(V.error_summary(d), pearson_r=float(np.corrcoef(c22[ok], c18[ok])[0, 1]), note="2018 minus 2022 would include 4 years of growth, fire or harvest; the median shows whether the two surveys agree in level")
    # terrain: the two ground models on open ground
    dt22, dt18 = R22["dtm"], R18["dtm"]
    open_ = (c22 < 1.0) & (c18 < 1.0)
    out["terrain"] = V.error_summary((dt22 - dt18)[open_])
    out["terrain_all"] = V.error_summary((dt22 - dt18).ravel())
    # trees
    t22, t18 = R22["trees"], R18["trees"]
    xy22 = np.c_[t22["col"], t22["row"]]
    xy18 = np.c_[t18["col"], t18["row"]]
    j, dist = V.match_points(xy22, xy18, 2.0)
    m = j >= 0
    h22, h18 = t22["height"], t18["height"]
    dh = h22[m] - h18[j[m]]
    out["trees"] = dict(n_2022=int(len(xy22)), n_2018=int(len(xy18)), matched_within_2m=int(m.sum()), share_of_2022_matched=float(m.mean()), share_of_2018_matched=float(m.sum() / max(len(xy18), 1)), height_2022_minus_2018=V.error_summary(dh),
                        median_match_distance_m=float(np.nanmedian(dist[m])) if m.any() else None, tall_trees_30m_plus=dict(n_2022=int((h22 >= 30).sum()), matched_share=float(m[h22 >= 30].mean()) if (h22 >= 30).any() else None),
                        short_trees_under_10m=dict(n_2022=int((h22 < 10).sum()), matched_share=float(m[h22 < 10].mean()) if (h22 < 10).any() else None))
    # the tree-count difference between the two surveys, against the planted-tree recall the analysis reported
    s22 = json.loads((ROOT / "data" / "processed" / "canopy" / "tahoe.json").read_text())
    out["compare_with_planted_tree_test"] = dict(planted_recall_regular_100_per_ha=s22["synthetic_stands"]["regular_100"]["recall"], planted_recall_regular_200_per_ha=s22["synthetic_stands"]["regular_200"]["recall"])
    (DATA / "canopy_tahoe.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
