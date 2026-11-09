"""Stage 1 of the White River town-scale utility model: tiled detection of wires, poles and elevated returns over a 3 km square (3DEP WA_PierceCounty_1_2020, about 25 points per m2).
Usage: python scripts/build_town.py [workers]   Output: data/cache/town/tile_*.npz (not committed) then data/processed/town/stage1.json"""
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lidar3d import town
from lidar3d.ept import CACHE

PROJECT = "WA_PierceCounty_1_2020"
LONLAT = (-122.21035, 47.23898)
CRS = "EPSG:32610"
HALF = 1500.0
TILE = 500.0
TILES = CACHE / "town"


def run(t):
    t0 = time.time()
    p = town.process_tile(PROJECT, LONLAT, CRS, t, TILES)
    return t[:2], round(time.time() - t0)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tiles = town.tile_grid(HALF, TILE)
    print(len(tiles), "tiles", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for ij, sec in ex.map(run, tiles):
            print("tile", ij, f"{sec}s", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
