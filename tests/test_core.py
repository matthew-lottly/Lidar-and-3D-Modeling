"""Pure-logic tests: no network, no cached data."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lidar3d import landcover, products, radar
from lidar3d.cloud import hillshade
from lidar3d.ept import EPT


def test_otsu_separates_two_modes():
    rng = np.random.default_rng(0)
    v = np.r_[rng.normal(20, 3, 5000), rng.normal(80, 5, 5000)]
    t = landcover.otsu(v)          # any threshold in the empty gap between the modes is optimal
    assert (v[:5000] < t).mean() > 0.99 and (v[5000:] >= t).mean() > 0.99


def test_hillshade_flat_and_slope():
    flat = hillshade(np.zeros((20, 20)), 1.0, 315, 40)
    assert np.allclose(flat, np.sin(np.radians(40)))
    ramp = np.tile(np.arange(20, dtype=float), (20, 1)) * 0.5   # rises to the east, so the slope faces west
    lit_from_west = hillshade(ramp, 1.0, 270, 40)[10, 10]
    lit_from_east = hillshade(ramp, 1.0, 90, 40)[10, 10]
    assert lit_from_east > lit_from_west


def test_ept_node_bounds_halves_the_cube():
    e = EPT.__new__(EPT)
    e.cube = np.array([0, 0, 0, 8, 8, 8], float)
    assert np.allclose(e.node_bounds("0-0-0-0"), [0, 0, 0, 8, 8, 8])
    assert np.allclose(e.node_bounds("1-1-0-1"), [4, 0, 4, 8, 4, 8])
    assert np.allclose(e.node_bounds("3-7-7-7"), [7, 7, 7, 8, 8, 8])


def test_gps_time_to_calendar_date():
    # adjusted standard GPS time 1e9 offset: 2022-09-06 00:00:00 UTC is 1346544000 GPS-ish seconds since 1980-01-06 plus leap seconds (18 s)
    import datetime as dt
    target = dt.datetime(2022, 9, 6, 12)
    secs = (target - dt.datetime(1980, 1, 6)).total_seconds() - 1e9
    d0, d1 = products.collection_dates(np.full(1000, secs))
    assert d0 == d1 == target.date()


def test_majority_filter_removes_isolated_pixel():
    a = np.full((9, 9), 4, np.int8)
    a[4, 4] = 3
    assert landcover.majority(a, 3)[4, 4] == 4


def test_remove_small_merges_tiny_patch():
    a = np.full((20, 20), 4, np.int8)
    a[2:4, 2:4] = 6                                    # 4 cells
    a[10:16, 10:16] = 3                                # 36 cells
    out = landcover.remove_small(a, 9)
    assert out[2, 2] == 4 and out[12, 12] == 3


def test_to_db():
    assert np.isclose(radar.to_db(np.array([0.1]))[0], -10.0)
    assert np.isnan(radar.to_db(np.array([0.0]))[0])


def test_renderer_nearest_point_wins():
    from lidar3d import render3d
    R = render3d.rotation(215, 42)
    toward_viewer = R[2]                                   # world vector that maps to +depth in the view
    pts = np.array([[0.0, 0.0, 0.0], *(10 * toward_viewer)[None, :]])
    cols = np.array([[255, 0, 0], [0, 0, 255]], np.uint8)  # red is far, blue is near, same screen position
    img, info = render3d.render(pts, cols, az=215, el=42, width=40, height=40, scale=1.0, center=np.zeros(3), splat=1)
    assert tuple(img[20, 20]) == (0, 0, 255)
