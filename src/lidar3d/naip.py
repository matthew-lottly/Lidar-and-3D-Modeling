"""NAIP aerial imagery (USDA, 0.6 m, RGB + near-infrared) for an AOI, from the Microsoft Planetary Computer mirror. Cached as a GeoTIFF under data/cache/naip."""
from __future__ import annotations

import os

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

from .aoi import AOI
from .ept import CACHE

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"


def items(aoi: AOI, year: int):
    W, S, E, N = aoi.bbox_ll
    r = requests.post(STAC, json={"collections": ["naip"], "bbox": [W, S, E, N], "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", "limit": 20}, timeout=60)
    r.raise_for_status()
    return r.json()["features"]


def fetch(aoi: AOI, year: int, res: float = 0.6, slug: str = "seattle_center"):
    """Return (rgbn uint8 array (4, h, w), transform, dates) on a grid in the AOI's CRS covering the AOI at `res` metres. Tiles are read as windows, not downloaded whole."""
    out = CACHE / "naip" / f"{slug}_{year}.tif"
    if out.exists():
        with rasterio.open(out) as d:
            return d.read(), d.transform, d.tags().get("dates", "")
    feats = items(aoi, year)
    if not feats:
        raise RuntimeError(f"no NAIP {year} items over the area")
    tok = requests.get(SAS, timeout=60).json()["token"]
    os.environ.setdefault("CURL_CA_BUNDLE", os.environ.get("AWS_CA_BUNDLE", "/root/.ccr/ca-bundle.crt"))
    x0, y0, x1, y1 = aoi.bounds
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    from rasterio.transform import from_origin
    tf = from_origin(x0, y1, res, res)
    mosaic = np.zeros((4, h, w), np.uint8)
    filled = np.zeros((h, w), bool)
    dates = []
    for f in feats:
        href = f["assets"]["image"]["href"] + "?" + tok
        with rasterio.open(href) as src, WarpedVRT(src, crs=aoi.crs, resampling=Resampling.bilinear, transform=tf, width=w, height=h) as v:
            a = v.read(masked=True)
        ok = ~np.ma.getmaskarray(a).any(0) & (a.filled(0).max(0) > 0)
        new = ok & ~filled
        mosaic[:, new] = a.filled(0)[:, new]
        filled |= new
        dates.append(f["properties"]["datetime"][:10])
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", height=h, width=w, count=4, dtype="uint8", crs=aoi.crs, transform=tf, compress="deflate", photometric="RGB") as d:
        d.write(mosaic)
        d.update_tags(dates=",".join(sorted(set(dates))), year=str(year))
    print(f"NAIP {year}: {filled.mean() * 100:.1f}% of the area covered by {len(feats)} tiles")
    return mosaic, tf, ",".join(sorted(set(dates)))


# ------------------------------------------------------------------------------------------------ relief displacement
def fit_displacement(rows: np.ndarray):
    """rows: (n, 6) = height m, x m, y m (relative to the AOI centre), measured east shift m, north shift m, match score.
    A nadir camera displaces a point at height h above the bare ground by h * (p - n) / H (n = ground point under the camera, H = flying height), which over a small area is
    shift = h * (k * p + b) with k = 1/H. Fits k and b (east, north) by robust least squares. The camera position and height are not separable over a 700 m area, so only the fitted shifts are reported."""
    from scipy.optimize import least_squares
    h, x, y, de, dn = (rows[:, i] for i in range(5))

    def resid(p):
        k, bx, by = p
        return np.r_[h * (k * x + bx) - de, h * (k * y + by) - dn]

    sol = least_squares(resid, [0.0, -0.3, 0.0], loss="soft_l1", f_scale=3.0)
    k, bx, by = sol.x
    r = resid(sol.x)
    return dict(k=float(k), bx=float(bx), by=float(by), rms_before=float(np.sqrt(np.mean(np.r_[de, dn] ** 2))), rms_after=float(np.sqrt(np.mean(r ** 2))),
                median_abs_before=float(np.median(np.abs(np.r_[de, dn]))), median_abs_after=float(np.median(np.abs(r))), n=len(h))


def true_ortho(rgbn, tf, dsm, ground, model, aoi_cx, aoi_cy, min_h=2.0):
    """Re-sample the image so that every surface cell takes its colour from where the camera actually saw it (surface point shifted by the fitted displacement), using the lidar surface heights.
    dsm, ground: lidar surface and bare-earth elevation on the SAME grid as rgbn. Returns (corrected rgbn, occluded mask). Occluded cells (hidden behind a taller structure in the photo) are filled from the nearest visible ground and flagged."""
    from scipy import ndimage
    h_, w_ = dsm.shape
    res = tf.a
    hh = np.clip(dsm - ground, 0, None)
    rr, cc = np.mgrid[0:h_, 0:w_]
    x = tf.c + (cc + 0.5) * res - aoi_cx
    y = tf.f - (rr + 0.5) * tf.a - aoi_cy
    dx = hh * (model["k"] * x + model["bx"])
    dy = hh * (model["k"] * y + model["by"])
    src_c, src_r = cc + dx / res, rr - dy / res
    out = np.stack([ndimage.map_coordinates(rgbn[b].astype(float), [src_r, src_c], order=1, mode="nearest") for b in range(rgbn.shape[0])]).clip(0, 255).astype(np.uint8)
    inside = (src_c >= 0) & (src_c <= w_ - 1) & (src_r >= 0) & (src_r <= h_ - 1)
    out = np.where(inside[None], out, rgbn)                    # displaced position falls outside the image: keep the original pixel instead of smearing the edge
    # occlusion: an image pixel that is the displaced position of some elevated cell cannot also show the ground beneath it
    roof = hh > min_h
    claimed = np.zeros((h_, w_), bool)
    pc, pr = np.round(src_c[roof]).astype(int), np.round(src_r[roof]).astype(int)
    okp = (pc >= 0) & (pc < w_) & (pr >= 0) & (pr < h_)
    claimed[pr[okp], pc[okp]] = True
    claimed = ndimage.binary_closing(claimed, iterations=2)
    occluded = claimed & ~roof
    if occluded.any():
        idx = ndimage.distance_transform_edt(occluded | roof, return_distances=False, return_indices=True)     # nearest visible ground cell
        fill = out[:, idx[0], idx[1]]
        out = np.where(occluded[None], fill, out)
    return out, occluded


def _edge_maps(rgbn):
    g = rgbn[:3].astype(float).mean(0)
    return ndimage_gauss(np.hypot(*np.gradient(ndimage_gauss(g, 1.0))), 0.7)


def ndimage_gauss(a, s):
    from scipy import ndimage
    return ndimage.gaussian_filter(a, s)


def measure_shifts(footprints, rgbn, tf, aoi, search_per_m=0.5, min_h=8.0, min_area=250.0, sizes_only=False):
    """For each tall footprint, find how far its roof outline appears displaced in the image: slide the outline's edge map (from the lidar footprint) over the image's edge strength and
    take the best normalised cross-correlation within a search window proportional to the building height. Returns an (n, 6) array: height, x, y (m from the AOI centre), east shift m, north shift m, score."""
    import rasterio.features
    from skimage.feature import match_template
    res = tf.a
    h_, w_ = rgbn.shape[1:]
    gn = _edge_maps(rgbn)
    rows = []
    for _, r in footprints.iterrows():
        if r.status != "ok" or r.clipped or r.height_m < min_h or r.area_m2 < min_area:
            continue
        g = r.geometry
        x0, y0, x1, y1 = g.bounds
        mask = rasterio.features.rasterize([(g, 1)], out_shape=(h_, w_), transform=tf, fill=0, dtype="uint8").astype(float)
        edge = np.hypot(*np.gradient(ndimage_gauss(mask, 1.2)))
        ms = int(r.height_m * search_per_m / res) + 8
        pad = 6
        c0, c1 = int((x0 - tf.c) / res) - pad, int((x1 - tf.c) / res) + pad
        r0, r1 = int((tf.f - y1) / res) - pad, int((tf.f - y0) / res) + pad
        if c0 - ms < 0 or r0 - ms < 0 or c1 + ms > w_ or r1 + ms > h_:
            continue
        cc = match_template(gn[r0 - ms:r1 + ms, c0 - ms:c1 + ms], edge[r0:r1, c0:c1])
        iy, ix = np.unravel_index(cc.argmax(), cc.shape)
        rows.append((r.height_m, g.centroid.x - aoi.cx, g.centroid.y - aoi.cy, (ix - ms) * res, -(iy - ms) * res, cc.max()))
    return np.array(rows)


def alignment_check(footprints, before, after, tf, min_h=8.0, min_area=250.0):
    """Does the correction move roofs onto their footprints? For each tall footprint (not cut by the AOI edge) slide the outline over the image within +-6 cells and record the outline's match
    score with no shift and the offset to the best match, for the image before and after. Returns a dict of medians."""
    import rasterio.features
    from skimage.feature import match_template
    res = tf.a
    h_, w_ = before.shape[1:]
    G = [_edge_maps(before), _edge_maps(after)]
    out = []
    for _, r in footprints.iterrows():
        if r.status != "ok" or r.clipped or r.height_m < min_h or r.area_m2 < min_area:
            continue
        g = r.geometry
        x0, y0, x1, y1 = g.bounds
        edge = np.hypot(*np.gradient(ndimage_gauss(rasterio.features.rasterize([(g, 1)], out_shape=(h_, w_), transform=tf, fill=0, dtype="uint8").astype(float), 1.2)))
        ms, pad = 6, 6
        c0, c1 = int((x0 - tf.c) / res) - pad, int((x1 - tf.c) / res) + pad
        r0, r1 = int((tf.f - y1) / res) - pad, int((tf.f - y0) / res) + pad
        if c0 - ms < 0 or r0 - ms < 0 or c1 + ms > w_ or r1 + ms > h_:
            continue
        row = []
        for gm in G:
            cc = match_template(gm[r0 - ms:r1 + ms, c0 - ms:c1 + ms], edge[r0:r1, c0:c1])
            iy, ix = np.unravel_index(cc.argmax(), cc.shape)
            row += [cc[ms, ms], np.hypot(iy - ms, ix - ms) * res]
        out.append(row)
    A = np.array(out)
    return dict(n=len(A), score_before=float(np.median(A[:, 0])), score_after=float(np.median(A[:, 2])), offset_before_m=float(np.median(A[:, 1])), offset_after_m=float(np.median(A[:, 3])), improved_share=float((A[:, 2] > A[:, 0]).mean()))
