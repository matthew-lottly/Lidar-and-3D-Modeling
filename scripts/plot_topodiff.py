"""Plates 46 to 50 for the Pacifica bluff study (run after scripts/build_topodiff.py)."""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lidar3d import figures_topodiff as FT
from lidar3d import naip
from lidar3d import topodiff as TD
from lidar3d.ept import CACHE

SITE = dict(slug="pacifica", lonlat=(-122.4985, 37.6175), half=450.0, crs="EPSG:32610")


def main():
    st = json.loads((ROOT / "data" / "processed" / "topodiff" / "stats.json").read_text())
    F = pickle.loads((CACHE / "topodiff" / "final.pkl").read_bytes())
    img, _, nd = naip.fetch(TD.make_aoi(SITE), 2020, res=1.0, slug="pacifica")
    st["naip_date"] = nd
    FT.fig_site(st, img)
    FT.fig_dod(st, F)
    FT.fig_hotspots(st, F)
    FT.fig_volumes(st, F)
    FT.fig_tests(st, F)


if __name__ == "__main__":
    main()
