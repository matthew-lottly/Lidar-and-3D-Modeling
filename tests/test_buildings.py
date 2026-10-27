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


def test_rigid_roof_heights_make_a_footprint_one_height():
    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from lidar3d import naip
    tf = from_origin(0, 60, 1.0, 1.0)
    hh = np.zeros((60, 60)); hh[10:30, 10:30] = np.random.default_rng(0).uniform(5, 25, (20, 20)); hh[45:50, 45:50] = 7.0
    fp = gpd.GeoDataFrame([dict(status="ok", height_m=18.0, geometry=box(10, 30, 30, 50))], geometry="geometry", crs="EPSG:32610")
    out = naip.rigid_roof_heights(hh, fp, tf)
    assert np.allclose(out[10:30, 10:30], 18.0) and np.all(out[45:50, 45:50] == 7.0)


def test_roof_heightfield_covers_the_footprint_and_closes_with_walls():
    from rasterio.transform import from_origin
    from lidar3d import exports
    tf = from_origin(0, 60, 0.6, 0.6)
    z = np.add.outer(np.linspace(0, 10, 100), np.zeros(100)) + 20          # ramp
    pg = Polygon([(5, 5), (45, 5), (45, 40), (25, 40), (25, 25), (5, 25)])  # L shape
    rv, rf, wv, wf = exports.roof_heightfield(pg, (z, tf), 10.0)
    def area2d(f):
        u, v = rv[f[1], :2] - rv[f[0], :2], rv[f[2], :2] - rv[f[0], :2]
        return abs(u[0] * v[1] - u[1] * v[0]) / 2
    plan = sum(area2d(f) for f in rf)
    assert abs(plan - pg.area) / pg.area < 0.02                              # roof triangles tile the footprint, not its bounding box
    assert all(np.cross(rv[f[1]] - rv[f[0]], rv[f[2]] - rv[f[0]])[2] > 0 for f in rf)   # all facing up
    assert wv[:, 2].min() == 10.0 and len(wf) == len(wv)                      # two wall triangles per outline segment, one segment per outline point


def _plane(rng, n, dz, slope=0.05):
    x, y = rng.uniform(0, 40, (2, n))
    return np.c_[x, y, slope * x + rng.normal(0, 0.03, n) + dz]


def test_m3c2_recovers_a_known_displacement_and_flags_nothing_on_stable_ground():
    from lidar3d import change
    rng = np.random.default_rng(0)
    a, b = _plane(rng, 30000, 0.0), _plane(rng, 30000, 0.30)
    core = np.c_[rng.uniform(5, 35, (500, 2)), np.zeros(500)]
    core[:, 2] = 0.05 * core[:, 0]
    d, lod, *_ = change.m3c2(a, b, core, normal_radii=(2.0,), cyl_radius=1.0, max_distance=3.0)
    assert abs(np.nanmedian(d) - 0.30 * np.cos(np.arctan(0.05))) < 0.01
    d0, lod0, *_ = change.m3c2(a, _plane(rng, 30000, 0.0), core, normal_radii=(2.0,), cyl_radius=1.0, max_distance=3.0)
    assert change.significant(d0, lod0).mean() < 0.15


def test_surface_points_rejects_a_canopy_floating_above_ground():
    from lidar3d import change
    rng = np.random.default_rng(1)
    g = np.c_[rng.uniform(-100, 100, (20000, 2)), rng.normal(5, 0.03, 20000)]
    canopy = np.c_[rng.uniform(-100, 0, (4000, 2)), 12 + rng.uniform(0, 2, 4000)]
    xyz = np.vstack([g, canopy])
    e = change.Epoch("t", "p", "2020-01-01", xyz[:, 0], xyz[:, 1], xyz[:, 2], np.ones(len(xyz), np.uint8), np.ones(len(xyz), np.uint8), np.ones(len(xyz), np.uint8))
    m = change.surface_points(e)
    assert m[:20000].mean() > 0.9 and m[20000:].mean() < 0.05


def test_volume_from_normals_and_helpers():
    from lidar3d import change
    d = np.array([-1.0, -1.0, 0.5, 2.0])
    nrm = np.array([[0, 0, 1.0], [0, 0, 0.5], [0, 0, 1.0], [0, 0, 1.0]])
    sig = np.array([True, True, True, False])
    loss, gain = change.volume_from_normals(d, nrm, sig, 4.0)
    assert loss == 4.0 + 8.0 and gain == 2.0                         # a steeper cell has more surface area per plan cell
    assert abs(change.robust_sd(np.random.default_rng(0).normal(0, 0.05, 20000)) - 0.05) < 0.003
    assert change.gps_date(np.full(10, 1.3e9 - 1e9 + 1e9 - 1e9)) == change.gps_date(np.full(10, 3.0e8))


def test_regular_cores_one_point_per_cell():
    from lidar3d import change
    rng = np.random.default_rng(2)
    p = np.c_[rng.uniform(-50, 50, (5000, 2)), rng.uniform(0, 1, 5000)]
    c = change.regular_cores(p, 10.0)
    assert len(c) == 100 and np.all(np.abs((c[:, 0] + 300) % 10 - 5) < 1e-6)
