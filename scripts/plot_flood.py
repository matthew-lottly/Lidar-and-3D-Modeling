"""Plates 51 to 55 and the interactive stage map for the Norfolk flood study (run after scripts/build_flood.py)."""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lidar3d import figures_flood as FF
from lidar3d import flood as FL
from lidar3d import naip
from lidar3d.ept import CACHE


def main():
    st = json.loads((ROOT / "data" / "processed" / "flood" / "stats.json").read_text())
    F = pickle.loads((CACHE / "flood" / "final.pkl").read_bytes())
    img, _, nd = naip.fetch(FL.make_aoi(), 2014, res=1.0, slug="norfolk")
    st["naip_date"] = nd
    FF.fig_site(st, F, img)
    FF.fig_hydrology(st, F)
    FF.fig_stages(st, F)
    FF.fig_vulnerability(st, F)
    FF.fig_tests(st, F)


if __name__ == "__main__":
    main()
