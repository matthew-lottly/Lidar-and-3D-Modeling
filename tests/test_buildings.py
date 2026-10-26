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
