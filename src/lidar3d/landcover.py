"""Land cover from the lidar point cloud itself.

Method (rule-based, every threshold is a named constant below):
  1. Height above ground (HAG) for every point = z - bare-earth DTM at the point's x,y (bilinear).
  2. Per 1 m cell, structural features from the points: highest return (canopy height), share of returns above 0.3 / 2 / 5 m,
     ground-return share, mean first-return intensity, point class flags (water = ASPRS 9, snow = ASPRS 21).
  3. A decision tree assigns one of eight classes. Cells are then cleaned with a majority filter and a minimum mapping unit.
No imagery or external land-cover product is used to make the map; WorldCover and Sentinel-2 NDVI are used only afterwards to check it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .cloud import Rasters

CLASSES = {1: ("Water", "#2B6CB0"), 2: ("Snow / ice", "#E2E8F0"), 3: ("Bare ground / rock", "#B7A99A"), 4: ("Short vegetation (< 1 m)", "#C6D98B"),
           5: ("Shrub (1 to 3 m)", "#8FB36B"), 6: ("Low tree (3 to 10 m)", "#3E8E5A"), 7: ("Tall tree (10 to 25 m)", "#1F5F3F"), 8: ("Very tall tree (> 25 m)", "#0B3B2A")}
# thresholds (metres, shares)
H_VEG, H_SHRUB, H_TREE, H_TALL, H_VTALL = 0.3, 1.0, 3.0, 10.0, 25.0
BARE_MAX_VEG_SHARE = 0.10        # at most this share of returns above H_VEG for a cell to count as bare
LOW_MIN_VEG_SHARE = 0.10
SNOW_MIN_SHARE = 0.5             # share of returns flagged snow
WATER_MIN_SHARE = 0.5
MMU_CELLS = 9                    # minimum mapping unit: 9 m2


def point_hag(R: Rasters, dtm: np.ndarray) -> np.ndarray:
    """Height above ground per point, from bilinear sampling of the DTM at the point location."""
    c = R.c
    fr = (R.y1 - c.y) / R.res - 0.5
    fc = (c.x - R.x0) / R.res - 0.5
    g = ndimage.map_coordinates(dtm, [fr, fc], order=1, mode="nearest")
    return c.z - g


@dataclass
class LandCover:
    cls: np.ndarray                 # int8 class codes, 0 = no data
    features: dict
    hag: np.ndarray                 # per-point height above ground

    def areas(self, res=1.0) -> dict[int, float]:
        return {k: float((self.cls == k).sum() * res * res) for k in CLASSES}


def classify(R: Rasters, dtm: np.ndarray, chm: np.ndarray, hag: np.ndarray | None = None) -> LandCover:
    c, shape = R.c, R.shape
    n = shape[0] * shape[1]
    idx = R.row * shape[1] + R.col
    hag = point_hag(R, dtm) if hag is None else hag
    tot = np.bincount(idx, minlength=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = lambda m: (np.bincount(idx[m], minlength=n) / tot).reshape(shape)
        above03, above2, above5 = share(hag > H_VEG), share(hag > 2.0), share(hag > 5.0)
        ground = share(c.cls == 2)
        snow = share(c.cls == 21)
        water = share(c.cls == 9)
    tot = tot.reshape(shape)
    f1 = c.ret == 1
    s = np.bincount(idx[f1], weights=c.intensity[f1].astype(float), minlength=n)
    k = np.bincount(idx[f1], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        inten = (s / k).reshape(shape)
    # smooth the structural shares slightly so 1 m cells with few returns are not noisy
    sm = lambda a: ndimage.uniform_filter(np.nan_to_num(a), 3, mode="nearest")
    above03s = sm(above03)
    out = np.zeros(shape, np.int8)
    out[:] = 3
    # Cells whose highest return is under 1 m are bare ground or short herbaceous vegetation. Height alone cannot separate them (and boulders add
    # spurious height), but near-infrared first-return intensity can: green vegetation reflects far more than rock or soil. Split with Otsu's
    # threshold computed from the lidar intensities alone.
    inten_s = ndimage.uniform_filter(np.nan_to_num(inten, nan=np.nanmedian(inten)), 5, mode="nearest")
    ground_level = chm < H_SHRUB
    thr = otsu(inten_s[ground_level])
    green = inten_s >= thr
    out[ground_level & green] = 4
    out[(chm >= H_SHRUB) & (chm < H_TREE)] = np.where(green, 5, 3)[(chm >= H_SHRUB) & (chm < H_TREE)]     # a 1 to 3 m return over non-green ground is rock, not shrub
    out[(chm >= H_TREE) & (chm < H_TALL)] = 6
    out[(chm >= H_TALL) & (chm < H_VTALL)] = 7
    out[chm >= H_VTALL] = 8
    # a tall single return over open ground (rocks, cornices, a boulder) is not a tree: require vegetation-like vertical spread
    spread = sm(above2) > 0.02
    not_tree = (out >= 6) & ~spread
    out[not_tree] = 5
    out[sm(snow) >= SNOW_MIN_SHARE] = 2
    out[sm(water) >= WATER_MIN_SHARE] = 1
    # clean-up: majority filter then remove regions smaller than the minimum mapping unit by absorbing them into the majority neighbour
    out = majority(out, 5)
    out = remove_small(out, MMU_CELLS)
    return LandCover(out, dict(otsu_threshold=thr, above03=above03, above2=above2, above5=above5, ground=ground, snow=snow, water=water, intensity=inten, count=tot), hag)


def otsu(v: np.ndarray, bins: int = 256) -> float:
    h, e = np.histogram(v, bins=bins)
    p = h / h.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * (e[:-1] + e[1:]) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    return float(e[:-1][np.nanargmax(sb)])


def majority(a: np.ndarray, size: int) -> np.ndarray:
    classes = sorted(CLASSES)
    votes = np.stack([ndimage.uniform_filter((a == k).astype(np.float32), size, mode="nearest") for k in classes])
    return np.array(classes, np.int8)[votes.argmax(0)]


def remove_small(a: np.ndarray, mmu: int) -> np.ndarray:
    out = a.copy()
    for k in CLASSES:
        lab, n = ndimage.label(a == k)
        if n == 0:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        small = np.isin(lab, np.where(sizes < mmu)[0] + 1)
        if small.any():
            out[small] = 0
    holes = out == 0
    if holes.any():
        idx = ndimage.distance_transform_edt(holes, return_distances=False, return_indices=True)
        out = out[tuple(idx)]
    return out


# ---------------- independent checks ----------------
WORLDCOVER_MAP = {1: 80, 2: 70, 3: 60, 4: 30, 5: 20, 6: 10, 7: 10, 8: 10}      # lidar class -> WorldCover class code


def compare_worldcover(lc: LandCover, aoi, wc: np.ndarray, wc_extent: tuple):
    """Agreement between the lidar map (resampled to WorldCover's 10 m cells by majority) and ESA WorldCover. Returns confusion matrix and overall agreement."""
    from rasterio.transform import from_bounds

    x0, y0, x1, y1 = aoi.bounds
    n = wc.shape[0]
    cell = (x1 - x0) / n
    mapped = np.vectorize(WORLDCOVER_MAP.get)(lc.cls)
    blk = int(round(cell / 1.0))
    h = (mapped.shape[0] // blk) * blk
    m = mapped[:h, :h].reshape(h // blk, blk, h // blk, blk).transpose(0, 2, 1, 3).reshape(h // blk, h // blk, -1)
    codes = np.array([10, 20, 30, 60, 70, 80])
    cnt = np.stack([(m == k).sum(-1) for k in codes], -1)
    lid = codes[cnt.argmax(-1)]
    w = wc[: lid.shape[0], : lid.shape[1]]
    w = np.where(np.isin(w, codes), w, 0)
    ok = w > 0
    cm = np.array([[int(((lid == a) & (w == b) & ok).sum()) for b in codes] for a in codes])     # rows: lidar, cols: WorldCover
    return codes, cm, float((lid == w)[ok].mean())
