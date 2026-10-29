"""Scroll bars, channel and meander geometry from one lidar acquisition (no repeat coverage needed).

A bend that migrates leaves a record in the floodplain: curved ridges (scroll bars) and swales that follow the former bank. From one bare-earth surface we can map them, read the direction and
the increments of migration, and measure the present channel and its curvature. Dating needs something else (imagery in several years), and every step here has an error test.
Coordinates are local metres (UTM minus the area centre).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from scipy import ndimage

from .aoi import AOI
from .ept import CACHE, EPT

SITE = dict(slug="wabash_mtcarmel", name="Wabash River south of Mount Carmel, Illinois", lonlat=(-87.93, 38.31), half=1750.0, crs="EPSG:32616", project="IL_WabashRiver_2010")


def make_aoi(site=SITE) -> AOI:
    cx, cy = Transformer.from_crs("EPSG:4326", site["crs"], always_xy=True).transform(*site["lonlat"])
    return AOI(round(cx), round(cy), site["half"], site["crs"])


@dataclass
class Points:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    cls: np.ndarray
    ret: np.ndarray
    nret: np.ndarray
    inten: np.ndarray
    gps: np.ndarray
    psid: np.ndarray
    date: str
    cx: float
    cy: float


def load_points(site=SITE, log=print) -> Points:
    """All returns inside the AOI (low noise class 7 dropped), cached as float32 local metres."""
    npz = CACHE / "meander" / f"{site['slug']}_v2.npz"
    aoi = make_aoi(site)
    if not npz.exists():
        c = EPT(site["project"]).fetch(aoi.bbox_ll, max_depth=20, log=log, full_gps=True)
        x0, y0, x1, y1 = aoi.bounds
        m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & (c.cls != 7)
        g = c.gps[c.gps > 0]
        v = float(np.median(g))
        v += 1e9 if v < 1e9 else 0.0
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, x=(c.x[m] - aoi.cx).astype("float32"), y=(c.y[m] - aoi.cy).astype("float32"), z=c.z[m].astype("float32"), cls=c.cls[m], ret=c.ret[m], nret=c.nret[m],
                 inten=c.intensity[m].astype("uint16"), gps=c.gps[m], psid=c.psid[m], date=str((dt.datetime(1980, 1, 6) + dt.timedelta(seconds=v)).date()))
    d = np.load(npz)
    return Points(d["x"], d["y"], d["z"], d["cls"], d["ret"], d["nret"], d["inten"], d["gps"], d["psid"], str(d["date"]), float(aoi.cx), float(aoi.cy))


def cell_index(x, y, half, res):
    n = int(round(2 * half / res))
    ix = np.floor((x + half) / res).astype(int).clip(0, n - 1)
    iy = np.floor((y + half) / res).astype(int).clip(0, n - 1)
    return ix, iy, n


def dem_from_points(x, y, z, half, res=1.0, min_sigma=1.5, max_fill=60.0):
    """Bare-earth DEM from ground returns: median per cell, empty cells filled by normalised Gaussian convolution of the filled cells (a smooth fill that does not invent relief beyond the data's own scale).
    Returns (dem, count). Cells farther than `max_fill` metres from any return stay NaN (water dropouts, large voids)."""
    ix, iy, n = cell_index(x, y, half, res)
    key = iy * n + ix
    o = np.argsort(key, kind="stable")
    ks, zs = key[o], z[o]
    u, st, cnt = np.unique(ks, return_index=True, return_counts=True)
    med = np.array([np.median(zs[a:a + c]) for a, c in zip(st, cnt)], dtype="float64") if len(u) < 4_000_000 and False else np.add.reduceat(zs.astype("float64"), st) / cnt
    dem = np.full(n * n, np.nan)
    dem[u] = med
    cn = np.zeros(n * n); cn[u] = cnt
    dem, cn = dem.reshape(n, n), cn.reshape(n, n)
    known = np.isfinite(dem)
    w, v = known.astype(float), np.where(known, dem, 0.0)
    out = dem.copy()
    todo = ~known
    for s in (min_sigma, 3.0, 6.0, 12.0, 25.0):
        if not todo.any():
            break
        num, den = ndimage.gaussian_filter(v, s / res), ndimage.gaussian_filter(w, s / res)
        ok = todo & (den > 0.15)
        out[ok] = num[ok] / den[ok]
        v = np.where(np.isfinite(out), out, 0.0); w = np.isfinite(out).astype(float)
        todo = ~np.isfinite(out)
    far = ndimage.distance_transform_edt(~known, sampling=res) > max_fill
    out[far] = np.nan
    return out, cn


def sample_dem(dem, half, res, x, y):
    """Bilinear DEM value at points; NaN outside the grid or wherever any of the four surrounding cells is NaN."""
    fx, fy = (x + half) / res - 0.5, (y + half) / res - 0.5
    v = ndimage.map_coordinates(np.nan_to_num(dem, nan=0.0), [fy, fx], order=1, mode="nearest")
    ok = ndimage.map_coordinates(np.isfinite(dem).astype("float32"), [fy, fx], order=1, mode="nearest")
    return np.where(ok > 0.999, v, np.nan)


def channel_mask(P: Points, half, res=2.0, min_area_m2=40_000.0, close_m=6.0):
    """Wetted channel at the time of the survey: cells with no returns of any class (water absorbs the laser pulse), closed over small gaps, largest connected body.
    Ponds and oxbow lakes are voids too, so only the largest body is the river. Returns (mask, all_return_count)."""
    ix, iy, n = cell_index(P.x, P.y, half, res)
    cnt = np.zeros((n, n), np.int32)
    np.add.at(cnt, (iy, ix), 1)
    void = cnt == 0
    r = max(int(round(close_m / res)), 1)
    void = ndimage.binary_closing(void, structure=np.ones((3, 3)), iterations=r)
    lab, k = ndimage.label(void)
    if k == 0:
        return void, cnt
    sizes = ndimage.sum(void, lab, range(1, k + 1))
    keep = 1 + int(np.argmax(sizes))
    m = lab == keep
    m = ndimage.binary_fill_holes(m)
    assert m.sum() * res * res >= min_area_m2, "no river-sized void found"
    return m, cnt


def hand(dem, channel, res):
    """Height above the nearest channel cell (nearest in plan, not along flow): a quick floodplain/terrace separator."""
    d = np.where(np.isfinite(dem), dem, np.nan)
    edge = ndimage.binary_dilation(channel, iterations=2) & ~channel
    # elevation of the nearest bank cell
    lab_idx = ndimage.distance_transform_edt(~edge, return_distances=False, return_indices=True)
    return d - d[tuple(lab_idx)]


def ridge_swale(dem, sigma_m, res):
    """Principal curvatures of the DEM smoothed at `sigma_m` (m^-1; crest = most negative, swale = most positive) and the direction (radians, east = 0) along the crest."""
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    s = sigma_m / res
    d = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    H = hessian_matrix(d, sigma=s, order="rc", use_gaussian_derivatives=True)
    ev = hessian_matrix_eigvals(H) / res ** 2        # ev[0] >= ev[1]
    Hrr, Hrc, Hcc = H
    theta = 0.5 * np.arctan2(2 * Hrc, Hcc - Hrr)      # angle of the principal axis of largest curvature
    return ev[1], ev[0], theta


def extract_ridges(dem, res, valid=None, sigma_m=4.0, strong=0.0022, weak=0.0012, min_len_m=60.0, kind="crest"):
    """Crest lines (or swale lines with kind='swale') from the DEM: curvature across the feature at scale `sigma_m` (m^-1), non-maximum suppression across it, hysteresis between `weak` and `strong`,
    thinning to one pixel and removal of pieces shorter than `min_len_m`. Returns (line mask, strength grid, labelled lines)."""
    from skimage.filters import apply_hysteresis_threshold
    from skimage.morphology import skeletonize
    lo, hi, theta = ridge_swale(dem, sigma_m, res)
    c = -lo if kind == "crest" else hi
    c = np.where(np.isfinite(c), c, 0.0)
    if valid is not None:
        c = np.where(valid, c, 0.0)
    # theta is the axis of largest curvature in (col, row) terms for the Hessian eigen-decomposition; the crest normal is the axis of the most negative curvature
    th = theta + (0.0 if kind == "swale" else np.pi / 2)
    dc, dr = np.cos(th), np.sin(th)
    rr, cc = np.mgrid[0:c.shape[0], 0:c.shape[1]].astype(float)
    n1 = ndimage.map_coordinates(c, [rr + dr, cc + dc], order=1, mode="nearest")
    n2 = ndimage.map_coordinates(c, [rr - dr, cc - dc], order=1, mode="nearest")
    nms = (c >= n1) & (c >= n2)
    m = apply_hysteresis_threshold(np.where(nms, c, 0.0), weak, strong)
    m = ndimage.binary_dilation(m, iterations=1)           # close one-pixel gaps in the NMS line before thinning
    sk = skeletonize(m)
    lab, k = ndimage.label(sk, structure=np.ones((3, 3)))
    sizes = ndimage.sum(sk, lab, range(1, k + 1))
    keep = np.isin(lab, 1 + np.flatnonzero(sizes * res >= min_len_m))
    lab2, k2 = ndimage.label(keep, structure=np.ones((3, 3)))
    return keep, c, lab2


def ndwi(rgbn):
    g, n = rgbn[1].astype("float32"), rgbn[3].astype("float32")
    return (g - n) / np.maximum(g + n, 1.0)


def water_from_naip(rgbn, near, clean_px=3):
    """River water from a NAIP RGB+NIR image: Otsu threshold of NDWI computed inside the `near` mask (a band around the known river, so fields do not dominate the histogram), opened to drop
    speckle, holes filled, and the connected body that overlaps most of `near`'s core kept. Returns (mask, threshold)."""
    from skimage.filters import threshold_otsu
    w = ndwi(rgbn)
    t = float(threshold_otsu(w[near]))
    m = (w > t) & near
    m = ndimage.binary_opening(m, iterations=clean_px)
    lab, k = ndimage.label(m)
    if k:
        sizes = ndimage.sum(m, lab, range(1, k + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(m), t


def active_channel_from_naip(rgbn, near, ndvi_max=0.25, clean_px=3):
    """Active channel (open water plus bare bars) from NAIP: pixels without vegetation (NDVI below `ndvi_max`) in the `near` band, opened, largest connected body, holes filled.
    Unlike the wetted channel it does not move with river stage, but it does depend on the season's vegetation."""
    r, n = rgbn[0].astype("float32"), rgbn[3].astype("float32")
    v = (n - r) / np.maximum(n + r, 1.0)
    m = (v < ndvi_max) & near
    m = ndimage.binary_opening(m, iterations=clean_px)
    lab, k = ndimage.label(m)
    if k:
        sizes = ndimage.sum(m, lab, range(1, k + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(m)


def fit_arc_center(dem, res, box, x0=(0.0, 0.0), sigma_m=5.0, min_strength=0.0015, half=1750.0):
    """Centre of the circular arcs that the scroll bars (and swales) form inside `box` = (xmin, xmax, ymin, ymax) in local metres.
    At every pixel with strong ridge or swale curvature the feature direction is the direction of least curvature; the centre minimises the weighted squared sine of the angle between that
    direction and the tangent of the circle through the pixel about the centre (a Hough-like alignment fit). Returns (cx, cy, rms angular misfit in degrees)."""
    from scipy.optimize import minimize
    lo, hi, theta = ridge_swale(dem, sigma_m, res)
    s = np.maximum(-lo, hi)
    n = dem.shape[0]
    cc, rr = np.meshgrid(np.arange(n), np.arange(n))
    X, Y = cc * res - half + res / 2, rr * res - half + res / 2
    m = (s > min_strength) & (X > box[0]) & (X < box[1]) & (Y > box[2]) & (Y < box[3]) & np.isfinite(dem)
    # theta is the angle (in the row/col frame) of the largest-curvature axis; the feature runs perpendicular to the dominant curvature axis
    along = theta[m] + np.pi / 2
    tx, ty = np.cos(along), np.sin(along)           # unit feature direction in (col=x, row=y) axes
    px, py, w = X[m], Y[m], np.sqrt(s[m])

    def cost(c):
        dx, dy = px - c[0], py - c[1]
        r = np.hypot(dx, dy) + 1e-9
        radial = (dx * tx + dy * ty) / r             # cos(angle between the feature and the radial direction); zero when the feature is tangent
        return float(np.sum(w * radial ** 2) / np.sum(w))
    best = None
    for gx in np.linspace(box[0], box[1], 5):
        for gy in np.linspace(box[2] - 1500, box[3] - 300, 5):
            r = minimize(cost, [gx, gy], method="Nelder-Mead", options=dict(xatol=1.0, fatol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
    return float(best.x[0]), float(best.x[1]), float(np.degrees(np.arcsin(np.sqrt(best.fun))))


def bank_line(channel, res, half, x_min=-720.0, smooth_m=10.0, spacing_m=10.0):
    """Smooth line along the channel's outer (northern) edge: the northernmost channel cell in every column east of `x_min`, then a parametric smoothing spline (`smooth_m` = rms smoothing).
    Returns (points (n,2), unit normals (n,2) pointing away from the channel, arc length (n,))."""
    from scipy.interpolate import splev, splprep
    pts = []
    for c in range(channel.shape[1]):
        r = np.flatnonzero(channel[:, c])
        x = c * res - half + res / 2
        if len(r) and x >= x_min:
            pts.append((x, r.max() * res - half + res / 2))
    pts = np.array(pts, float)
    tck, _ = splprep(pts.T, s=len(pts) * smooth_m ** 2, k=3)
    u = np.linspace(0, 1, 4000)
    x, y = splev(u, tck)
    L = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    s = np.arange(0, L[-1], spacing_m)
    uu = np.interp(s, L, u)
    x, y = splev(uu, tck)
    dx, dy = splev(uu, tck, der=1)
    n = np.hypot(dx, dy)
    nx, ny = -dy / n, dx / n
    flip = np.sign(np.mean(ny)) or 1.0                  # the outer bank is the northern edge: normals point north on average
    return np.c_[x, y], np.c_[nx, ny] * flip, s


def rectify(dem, res, half, line, normals, d_min=-1400.0, d_max=2600.0, step=2.0):
    """Sample the DEM on transects normal to the bank line: rows = stations along the bank, columns = signed distance from it (positive away from the channel, i.e. northwards)."""
    d = np.arange(d_min, d_max + step, step)
    X = line[:, 0][:, None] + normals[:, 0][:, None] * d[None]
    Y = line[:, 1][:, None] + normals[:, 1][:, None] * d[None]
    fx, fy = (X + half) / res - 0.5, (Y + half) / res - 0.5
    v = ndimage.map_coordinates(np.nan_to_num(dem, nan=-9999.0), [fy, fx], order=1, mode="nearest")
    out = np.where((v < -9000) | (np.abs(X) > half) | (np.abs(Y) > half), np.nan, v)
    return d, out, X, Y


def coarsen(a, k=2):
    n = a.shape[0] // k * k
    b = a[:n, :n].reshape(n // k, k, n // k, k)
    with np.errstate(all="ignore"):
        return np.nanmean(b, axis=(1, 3))


def synthetic_surface(x, y, amp, wavelength, radius=900.0, tilt=0.0005, centre=(0.0, -1.0)):
    """A planted scroll-bar surface: cosine ridges of amplitude `amp` and wavelength `wavelength` on arcs of `radius` m about a centre `radius` m to the south of the patch centre, plus a gentle tilt.
    Returns (z, crest_distance_error function inputs): z at (x, y) and the radial distance of each point from the arc centre."""
    cx, cy = centre[0] * radius, centre[1] * radius
    d = np.hypot(x - cx, y - cy)
    return amp * np.cos(2 * np.pi * d / wavelength) + tilt * y, d


def synthetic_detection(x, y, amp, wavelength, noise, half, res=2.0, seed=0, **kw):
    """Plant ridges in the sampling pattern of real ground returns (positions x, y), add `noise` (m, per point), build the DEM and run the detector.
    Returns recall of planted crests (share of crest pixels with a detected ridge within 3 m), the share of detected pixels farther than 5 m from any planted crest, and the detected length (m)."""
    rng = np.random.default_rng(seed)
    z, _ = synthetic_surface(x, y, amp, wavelength) if amp > 0 else (np.zeros(len(x)), None)
    z = z + rng.normal(0, noise, len(x))
    dem, _ = dem_from_points(x, y, z, half, 1.0)
    dem2 = coarsen(dem, 2)
    keep, _, _ = extract_ridges(dem2, res, **kw)
    n = dem2.shape[0]
    cc, rr = np.meshgrid(np.arange(n), np.arange(n))
    X, Y = cc * res - half + res / 2, rr * res - half + res / 2
    inside = (np.abs(X) < half - 25) & (np.abs(Y) < half - 25)
    det = keep & inside
    if amp > 0:
        _, d = synthetic_surface(X, Y, amp, wavelength)
        off = (d / wavelength + 0.0) % 1.0
        dist = np.minimum(off, 1 - off) * wavelength            # distance (m) to the nearest planted crest along the radial direction
        crest = (dist < res / 2) & inside
        near_det = ndimage.binary_dilation(det, iterations=int(round(3 / res)))
        recall = float((near_det & crest).sum() / max(crest.sum(), 1))
        far = float(((dist > 5) & det).sum() / max(det.sum(), 1)) if det.any() else 0.0
    else:
        recall, far = float("nan"), float("nan")
    return recall, far, float(det.sum() * res)


def relief_strip(R, d, step_d, sigma_s=2.0, sigma_d_m=3.0, hp_m=30.0):
    """Band-pass of the rectified DEM (rows = stations, columns = distance): smoothed along the bank (`sigma_s` stations) and across it (`sigma_d_m`), minus the same strip smoothed over `hp_m`.
    NaN stays NaN. Positive = ridge crest relative to the surroundings, in metres of relief."""
    ok = np.isfinite(R)
    f = np.where(ok, R, 0.0)

    def sm(a, w, sd, ss):
        num = ndimage.gaussian_filter(a * w, (ss, sd / step_d)); den = ndimage.gaussian_filter(w, (ss, sd / step_d))
        return np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
    lo = sm(f, ok.astype(float), sigma_d_m, sigma_s)
    base = sm(np.where(np.isfinite(lo), lo, 0.0), np.isfinite(lo).astype(float), hp_m, 0.0)
    return lo - base


def track_ridges(V, d, station_m, prominence=0.06, min_sep_m=10.0, link_m=6.0, min_len_m=150.0, max_slope=0.8, max_gap=2, kind="crest"):
    """Pick the crest (or swale) in every row of the relief strip `V` and link picks between neighbouring stations into continuous lines.
    A pick must stand `prominence` m above (below) its surroundings; links join picks within `link_m` in distance at the next station; a line must be `min_len_m` long and not steeper than `max_slope` (distance per metre along the bank),
    which removes field boundaries and ditches that run across the arcs. Returns a list of dicts with 'si' (station index), 'd' (distance), 'amp' (relief, positive for both kinds)."""
    from scipy.signal import find_peaks
    sd = d[1] - d[0]
    sgn = 1.0 if kind == "crest" else -1.0
    active, done = [], []
    for i in range(V.shape[0]):
        row = sgn * np.where(np.isfinite(V[i]), V[i], 0.0)
        pk, pr = find_peaks(row, prominence=prominence, distance=max(int(min_sep_m / sd), 1))
        picks = [(float(d[j]), float(row[j])) for j in pk]
        used = set()
        for ch in active:
            best, bd = None, link_m
            for k, (pd_, _) in enumerate(picks):
                if k in used:
                    continue
                if abs(pd_ - ch["d"][-1]) <= bd:
                    best, bd = k, abs(pd_ - ch["d"][-1])
            if best is not None:
                used.add(best); ch["si"].append(i); ch["d"].append(picks[best][0]); ch["amp"].append(picks[best][1]); ch["miss"] = 0
            else:
                ch["miss"] += 1
        still = []
        for ch in active:
            (still if ch["miss"] <= max_gap else done).append(ch)
        active = still
        for k, (pd_, pa) in enumerate(picks):
            if k not in used:
                active.append(dict(si=[i], d=[pd_], amp=[pa], miss=0))
    done += active
    out = []
    for ch in done:
        si, dd = np.array(ch["si"]), np.array(ch["d"])
        if (si[-1] - si[0]) * station_m < min_len_m or len(si) < 5:
            continue
        slope = np.polyfit(si * station_m, dd, 1)[0]
        if abs(slope) > max_slope:
            continue
        out.append(dict(si=si, d=dd, amp=np.array(ch["amp"])))
    return out


# ------------------------------------------------------------------ pipeline pieces used by the build script and the tests
def strip_chains(dem, line, normals, half, prominence=0.05, station_m=10.0, step_d=2.0, d_range=(-1400.0, 2600.0), **kw):
    """DEM -> rectified strip -> band-pass -> tracked crest and swale lines. Returns (d, V, crests, swales, X, Y)."""
    d, R, X, Y = rectify(dem, 1.0, half, line, normals, d_min=d_range[0], d_max=d_range[1], step=step_d)
    V = relief_strip(R, d, step_d)
    return d, V, track_ridges(V, d, station_m, prominence=prominence, **kw), track_ridges(V, d, station_m, prominence=prominence, kind="swale", **kw), X, Y


def pick_positions(chains):
    """Set of (station, distance) picks of a list of chains."""
    return np.array([(si, dd) for c in chains for si, dd in zip(c["si"], c["d"])]) if chains else np.zeros((0, 2))


def pick_agreement(a, b, tol_m=4.0):
    """Share of picks in `a` that have a pick of `b` at the same station within `tol_m` metres in distance. NaN if `a` is empty."""
    if len(a) == 0:
        return float("nan")
    from collections import defaultdict
    by = defaultdict(list)
    for si, dd in b:
        by[int(si)].append(dd)
    hit = sum(1 for si, dd in a if any(abs(dd - e) <= tol_m for e in by.get(int(si), ())))
    return hit / len(a)


def interswath_differences(P: Points, half, res=2.0, max_slope_deg=1.5, min_pts=3):
    """Ground-surface agreement between overlapping flight lines: per `res` cell and per pair of lines with at least `min_pts` ground returns each, the difference of mean heights on flat ground
    (slope below `max_slope_deg` in the all-lines DEM). Returns {pair: array of differences (m)}; lines with fewer than 2e5 ground returns are ignored."""
    g = P.cls == 2
    ix, iy, n = cell_index(P.x[g], P.y[g], half, res)
    key, z, ps = iy * n + ix, P.z[g].astype("float64"), P.psid[g]
    lines = [int(p) for p in np.unique(ps) if (ps == p).sum() > 200_000]
    mean, cnt = {}, {}
    for p in lines:
        m = ps == p
        s = np.bincount(key[m], z[m], minlength=n * n); c = np.bincount(key[m], minlength=n * n)
        mean[p], cnt[p] = np.where(c >= min_pts, s / np.maximum(c, 1), np.nan), c
    allm = np.bincount(key, z, minlength=n * n) / np.maximum(np.bincount(key, minlength=n * n), 1)
    a2 = allm.reshape(n, n)
    gy, gx = np.gradient(ndimage.gaussian_filter(a2, 1.0), res)
    flat = (np.degrees(np.arctan(np.hypot(gx, gy))) < max_slope_deg).ravel()
    out = {}
    for i, p in enumerate(lines):
        for q in lines[i + 1:]:
            d = mean[p] - mean[q]
            ok = np.isfinite(d) & flat
            if ok.sum() > 100:
                out[(p, q)] = d[ok]
    return out


def strip_detection_test(P: Points, half, line, normals, amp, wavelength, noise, centre, prominence=0.05, seed=0, dem_noise=(0.0, 8.0)):
    """End-to-end detection test of the strip method. Cosine ridges (`amp`, `wavelength`) about `centre` are planted at the positions of the real ground returns with Gaussian `noise` per return,
    the DEM is rebuilt, rectified, band-passed and tracked exactly as for the real data, and the tracked crests are compared with those of the noise-free surface put through the same filters.
    Returns dict(recall, precision, picks_true, picks_detected, median_abs_offset_m)."""
    rng = np.random.default_rng(seed)
    g = P.cls == 2
    x, y = P.x[g].astype("float64"), P.y[g].astype("float64")
    cx, cy = centre
    z = amp * np.cos(2 * np.pi * np.hypot(x - cx, y - cy) / wavelength) + rng.normal(0, noise, len(x))
    dem, _ = dem_from_points(x, y, z, half, 1.0)
    if dem_noise[0] > 0:                           # spatially correlated surface error (classification, interpolation), which white per-return noise does not represent
        f = ndimage.gaussian_filter(rng.normal(size=dem.shape), dem_noise[1])
        dem = dem + f / f.std() * dem_noise[0]
    _, _, C, _, _, _ = strip_chains(dem, line, normals, half, prominence=prominence)
    if amp <= 0:
        return dict(recall=float("nan"), precision=float("nan"), picks_true=0, picks_detected=int(len(pick_positions(C))), median_abs_offset_m=float("nan"))
    # truth: analytic surface on the 1 m grid through the same chain
    n = dem.shape[0]
    gx = np.arange(n) - half + 0.5
    GX, GY = np.meshgrid(gx, gx)
    truth = amp * np.cos(2 * np.pi * np.hypot(GX - cx, GY - cy) / wavelength)
    truth = np.where(np.isfinite(dem), truth, np.nan)
    _, _, Ct, _, _, _ = strip_chains(truth, line, normals, half, prominence=prominence)
    a, b = pick_positions(C), pick_positions(Ct)
    off = []
    if len(a) and len(b):
        from collections import defaultdict
        by = defaultdict(list)
        for si, dd in b:
            by[int(si)].append(dd)
        for si, dd in a:
            c = by.get(int(si))
            if c:
                off.append(min(abs(dd - e) for e in c))
    return dict(recall=pick_agreement(b, a), precision=pick_agreement(a, b), picks_true=int(len(b)), picks_detected=int(len(a)), median_abs_offset_m=float(np.median(off)) if off else float("nan"))


# ------------------------------------------------------------------ imagery: registration, vegetation edges, migration rates
def ndvi_image(rgbn):
    r, n = rgbn[0].astype("float32"), rgbn[3].astype("float32")
    return (n - r) / np.maximum(n + r, 1.0)


def sample_image(img, half, X, Y, res=1.0):
    """Bilinear sample of a north-up image covering the AOI (row 0 = north edge) at local-metre coordinates."""
    fx, fy = (X + half) / res - 0.5, (half - Y) / res - 0.5
    return ndimage.map_coordinates(img.astype("float32"), [fy, fx], order=1, mode="nearest")


def estimate_shift(ref, mov, mask, tile=700):
    """Horizontal shift (rows, cols, in pixels) that registers image `mov` to `ref`: phase correlation on high-passed tiles that lie wholly inside `mask` (stable ground away from the river).
    Returns (median shift, median absolute deviation, number of tiles)."""
    from skimage.registration import phase_cross_correlation
    hp = lambda a: ndimage.gaussian_filter(a, 1.0) - ndimage.gaussian_filter(a, 10.0)
    out = []
    for r0 in range(0, ref.shape[0] - tile + 1, tile // 2):
        for c0 in range(0, ref.shape[1] - tile + 1, tile // 2):
            if mask[r0:r0 + tile, c0:c0 + tile].all():
                a, b = hp(ref[r0:r0 + tile, c0:c0 + tile]), hp(mov[r0:r0 + tile, c0:c0 + tile])
                s, _, _ = phase_cross_correlation(a, b, upsample_factor=10)
                out.append(s)
    if not out:
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), 0
    o = np.array(out)
    return np.median(o, 0), np.median(np.abs(o - np.median(o, 0)), 0) * 1.4826, len(o)


def channel_edges(V, d, seed_d=-120.0, thr=0.20, gap_m=6.0, smooth_rows=3, max_north=120.0):
    """North and south edge of the active channel (open water plus bare bars) in each row of an NDVI strip: the contiguous run of low-NDVI ground (below `thr`) that contains the seed distance
    `seed_d` (the middle of the channel in the strip), bridged over gaps shorter than `gap_m`. Rows are median-smoothed along the bank (`smooth_rows`). Returns (north_edge, south_edge) in metres from
    the lidar bank line; NaN where the seed is not on low-NDVI ground or the run reaches `max_north` metres north of the bank line (bare fields behind the bank would otherwise be taken as channel)."""
    step = abs(d[1] - d[0])
    W = ndimage.median_filter(np.nan_to_num(V, nan=1.0), size=(smooth_rows, 1)) < thr
    k0 = int(np.argmin(np.abs(d - seed_d)))
    g = max(int(gap_m / step), 1)
    n_e, s_e = np.full(V.shape[0], np.nan), np.full(V.shape[0], np.nan)
    for i in range(V.shape[0]):
        w = W[i]
        if not w[k0]:
            continue
        j = k0
        miss = 0
        while j + 1 < len(d) and miss <= g:
            j += 1
            miss = 0 if w[j] else miss + 1
        hi = j - miss
        j, miss = k0, 0
        while j - 1 >= 0 and miss <= g:
            j -= 1
            miss = 0 if w[j] else miss + 1
        lo = j + miss
        n_e[i], s_e[i] = (d[hi] if d[hi] < max_north else np.nan), d[lo]
    return n_e, s_e


def slope_ci(t, y, n_boot=500, block=10, seed=0):
    """Median over stations of the per-station OLS slope of y on t, with a block-bootstrap 95% interval over stations (blocks of `block` neighbouring stations).
    `y` has shape (stations, years), `t` (years,). Returns (median slope, lo, hi, per-station slopes)."""
    rng = np.random.default_rng(seed)
    sl = np.full(y.shape[0], np.nan)
    for i in range(y.shape[0]):
        ok = np.isfinite(y[i])
        if ok.sum() >= 4:
            sl[i] = np.polyfit(t[ok], y[i, ok], 1)[0]
    nb = int(np.ceil(len(sl) / block))
    bl = [sl[k * block:(k + 1) * block] for k in range(nb)]
    meds = []
    for _ in range(n_boot):
        pick = np.concatenate([bl[j] for j in rng.integers(0, nb, nb)])
        meds.append(np.nanmedian(pick))
    return float(np.nanmedian(sl)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)), sl
