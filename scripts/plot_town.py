"""Plates 56 to 65 for the White River town model (run after scripts/build_town_extras.py all)."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lidar3d import figures_town as FT
from lidar3d import naip
from lidar3d.aoi import AOI


def main():
    D = FT.load()
    img, _, _ = naip.fetch(AOI(round(D["model"]["cx"]), round(D["model"]["cy"]), FT.HALF, "EPSG:32610"), 2019, res=1.0, slug="whiteriver_town")
    FT.fig_site(D, img)
    FT.fig_full(D)
    FT.fig_diag_map(D); FT.fig_diag_fig(D)
    FT.fig_sup_map(D); FT.fig_sup_fig(D)
    FT.fig_eq_map(D); FT.fig_eq_fig(D)
    FT.fig_span_map(D); FT.fig_span_fig(D)


if __name__ == "__main__":
    main()
