import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lidar3d import buildings, exports


def test_fit_plane_recovers_a_roof_pitch_with_outliers():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 20, 800), rng.uniform(0, 10, 800)
    z = 10 + 0.5 * x + rng.normal(0, 0.03, 800)          # 26.6 degrees pitch
    z[:100] += 6                                           # rooftop clutter
    co, inl = buildings.fit_plane(np.c_[x, y, z])
    assert abs(np.degrees(np.arctan(np.hypot(*co[:2]))) - 26.57) < 1.0
    assert 0.8 < inl.mean() < 0.95


def _row(geom):
    return dict(status="ok", geometry=geom, base_z=100.0, roof_z=112.0, height_m=12.0, height_max_m=13.0, ground_z=100.0, roof_type="flat", area_m2=geom.area, name="t", id="x", height=None)


def test_cityjson_solids_are_closed_for_holes_and_concave_shapes(tmp_path):
    ring = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)], holes=[[(5, 5), (5, 10), (10, 10), (10, 5)]])
    ell = Polygon([(30, 0), (50, 0), (50, 10), (40, 10), (40, 20), (30, 20)])
    g = gpd.GeoDataFrame([_row(ring), _row(ell)], geometry="geometry", crs="EPSG:32610")
    p = tmp_path / "t.city.json"
    assert exports.cityjson_lod1(g, p) == 2
    import json
    doc = json.loads(p.read_text())
    assert exports.cityjson_open_edges(doc) == {}
    assert doc["version"] == "1.1" and len(doc["CityObjects"]) == 2


def test_displacement_fit_recovers_a_known_camera():
    rng = np.random.default_rng(2)
    n = 60
    h = rng.uniform(10, 120, n); x = rng.uniform(-300, 300, n); y = rng.uniform(-300, 300, n)
    k, bx, by = 1 / 2800, -0.35, 0.02
    de = h * (k * x + bx) + rng.normal(0, 0.3, n); dn = h * (k * y + by) + rng.normal(0, 0.3, n)
    de[:5] += 20                                                  # bad matches
    from lidar3d import naip
    m = naip.fit_displacement(np.c_[h, x, y, de, dn, np.ones(n)])
    assert abs(m["bx"] - bx) < 0.02 and abs(m["by"] - by) < 0.02 and m["rms_after"] < m["rms_before"]


def test_true_ortho_moves_a_leaning_roof_back_over_its_footprint():
    from lidar3d import naip
    from rasterio.transform import from_origin
    n, res = 120, 0.6
    tf = from_origin(0, n * res, res, res)
    dsm = np.zeros((n, n)); dsm[50:70, 50:70] = 30.0              # a 30 m block
    ground = np.zeros((n, n))
    model = dict(k=0.0, bx=-0.2, by=0.0)                          # roof appears 6 m (10 cells) west of its footprint
    img = np.zeros((4, n, n), np.uint8); img[:, 50:70, 40:60] = 200   # image shows the roof at columns 40..60
    out, occ = naip.true_ortho(img, tf, dsm, ground, model, 0.0, 0.0)
    assert out[0, 60, 55] == 200 and out[0, 60, 65] == 200       # roof cells now take the roof colour
    assert out[0, 60, 45] != 200 or occ[60, 45]                  # the ground where the roof leaned is not roof-coloured, or flagged as hidden


def test_needle_profile_and_lathe_recover_a_known_hourglass():
    from lidar3d import needle
    rng = np.random.default_rng(4)
    z = rng.uniform(18, 100, 40000)
    r_true = 12 - 0.1 * (z - 20)                                    # 12 m at 20 m height narrowing to 4 m at 100 m
    th = rng.uniform(0, 2 * np.pi, z.size)
    rr = r_true * np.sqrt(rng.uniform(0.6, 1.0, z.size))            # returns inside the envelope
    pts = np.c_[rr * np.cos(th), rr * np.sin(th), z]
    pr = needle.profile(pts, shaft_top=200)
    mid = np.interp(60.0, pr["z"], pr["r"])
    assert abs(mid - (12 - 4)) < 0.6                                # about 8 m at 60 m
    assert pr["interpolated"][1:-1].sum() == 0
    m = needle.lathe(pr, (0.0, 0.0), 0.0)
    assert m.is_watertight and m.volume > 0
    assert abs(m.bounds[1][2] - pr["top"]) < 1e-6


def test_cityjson_accepts_a_modelled_solid_and_stays_closed(tmp_path):
    import json
    from lidar3d import needle
    pts = np.c_[np.zeros(200), np.zeros(200), np.linspace(20, 80, 200)] + np.random.default_rng(1).normal(0, 0.05, (200, 3))
    pr = needle.profile(pts, nmin=1, shaft_nmin=1)
    sq = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
    g = gpd.GeoDataFrame([{**_row(sq), "name": "Tower"}], geometry="geometry", crs="EPSG:32610")
    p = tmp_path / "s.city.json"
    exports.cityjson_lod1(g, p, solids={"Tower": needle.lathe(pr, (0.0, 0.0), 100.0)})
    doc = json.loads(p.read_text())
    assert exports.cityjson_open_edges(doc) == {}
    assert next(iter(doc["CityObjects"].values()))["geometry"][0]["lod"] == "2"
