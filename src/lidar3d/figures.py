"""The plates. Every number printed comes from the data via Products / LandCover / RadarSet."""
from __future__ import annotations

import numpy as np
import rasterio
from matplotlib import cm
from matplotlib.colors import LightSource, ListedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy import ndimage

from . import render3d
from .aoi import AOI, RAINIER_SUMMIT
from .cloud import hillshade
from .ept import CACHE
from .landcover import CLASSES
from .plates import (BLUE, FAINT, GREEN, INK, MUTED, RULE, RUST, Plate, colorbar, map_frame, north_arrow, scalebar)

SRC_LIDAR = "Lidar: USGS 3D Elevation Program (3DEP) point cloud WA_CentralWildfire_2_D22, public on AWS (usgs-lidar-public), collected {c0} to {c1}; classification as published (ASPRS: 2 ground, 9 water, 21 snow)."
SRC_DEM = "Regional context: USGS 3DEP 1 arc-second DEM."
SRC_S2 = "Imagery: Copernicus Sentinel-2 L2A, scene {s2}."
SRC_S1 = "Radar: Copernicus Sentinel-1 C-band, radiometrically terrain-corrected backscatter (Gamma0), public archive on AWS (sentinel-s1-rtc-indigo), 20 m, 2017 to April 2021."
SRC_WC = "Check data: ESA WorldCover 2021 v200 (CC BY 4.0)."


def hypsometric(n=256):
    return ListedColormap(cm.terrain(np.linspace(0.22, 0.98, n)))


def srcs(P, *extra):
    s = P.stats
    return " ".join([SRC_LIDAR.format(c0=s["collected"][0], c1=s["collected"][1]), *extra])


def to_extent_rc(P, x, y):
    x0, x1, y0, y1 = P.extent
    return (x - x0), (y1 - y)


# --------------------------------------------------------------------------------------------- 01 overview
def regional_hillshade(P, half_x=17000.0, half_y=11000.0, res=30.0):
    """3DEP 1 arc-second DEM reprojected to the study area's UTM grid, centred between the study area and Mount Rainier."""
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    sx, sy = Transformer.from_crs("EPSG:4326", P.aoi.crs, always_xy=True).transform(*RAINIER_SUMMIT)
    cx, cy = (sx + P.aoi.cx) / 2, (sy + P.aoi.cy) / 2
    x0, y1 = cx - half_x, cy + half_y
    shape = (int(2 * half_y / res), int(2 * half_x / res))
    dst = np.full(shape, np.nan, np.float32)
    with rasterio.open(CACHE / "dem" / "USGS_1_n47w122.tif") as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=from_origin(x0, y1, res, res), dst_crs=P.aoi.crs, resampling=Resampling.bilinear, dst_nodata=np.nan)
    return dst, (x0, x0 + 2 * half_x, y1 - 2 * half_y, y1), (sx, sy)


def fig_overview(P, names, S):
    s = P.stats
    pl = Plate("Study area: a low-canopy cirque in the Tatoosh Range", f"Lidar, radar and 3D models of a {s['area_km2']:.1f} km² area about {s['km_to_rainier']:.0f} km south-southeast of the summit of Mount Rainier, Washington",
               srcs(P, SRC_DEM, "Place names: Overture Maps (release 2026-09-23.1)."), "Grid: UTM zone 10N (EPSG:32610), metres")
    z, ext, (sx, sy) = regional_hillshade(P)
    hs = hillshade(np.nan_to_num(z, nan=np.nanmean(z)), 30.0, 315, 40, 1.4)
    a = pl.ax(0.6, 4.1, 5.3, 3.9)
    a.imshow(hs, extent=ext, cmap="gray", vmin=0.1, vmax=1.0, origin="upper")
    a.imshow(np.ma.masked_invalid(z), extent=ext, cmap=hypsometric(), alpha=0.35, origin="upper")
    x0, x1, y0, y1 = P.extent
    a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=RUST, lw=1.6))
    a.plot([sx], [sy], marker="^", ms=7, color=INK, mec="white", mew=0.8)
    a.annotate("Mount Rainier summit", (sx, sy), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=7.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    a.annotate("study area", (P.aoi.cx, P.aoi.cy), xytext=(20, -18), textcoords="offset points", fontsize=7.5, color=RUST, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), arrowprops=dict(arrowstyle="-", color=RUST, lw=0.8))
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
    for sp in a.spines.values():
        sp.set_color(MUTED)
    pl.text(0.6, 3.95, "Regional context (3DEP 1 arc-second DEM, 30 m hillshade). The red box is the study area.", fontsize=8, color=MUTED)
    # main map
    ex = P.extent
    m = pl.ax(6.6, 1.3, 7.2, 7.2)
    sm = ndimage.gaussian_filter(P.dtm, 1.0)
    m.imshow(P.hs, extent=ex, cmap="gray", vmin=0.15, vmax=1.0)
    m.imshow(sm, extent=ex, cmap=hypsometric(), alpha=0.38, vmin=P.dtm.min(), vmax=P.dtm.max())
    xs = np.linspace(ex[0], ex[1], P.dtm.shape[1]); ys = np.linspace(ex[3], ex[2], P.dtm.shape[0])
    cs = m.contour(xs, ys, sm, levels=np.arange(1450, 1950, 50), colors="#5a4a3a", linewidths=0.45, alpha=0.8)
    m.contour(xs, ys, sm, levels=np.arange(1500, 1950, 100), colors="#3b2f25", linewidths=0.8)
    m.clabel(m.contour(xs, ys, sm, levels=np.arange(1500, 1950, 100), colors="none"), fmt="%d", fontsize=6.5, colors="#3b2f25", inline=True, inline_spacing=6)
    map_frame(m, ex, 200)
    for nm in names:
        if nm["cls"] == "peak" or nm["kind"] == "water":
            if ex[0] < nm["x"] < ex[1] and ex[2] < nm["y"] < ex[3]:
                m.plot(nm["x"], nm["y"], marker="^" if nm["cls"] == "peak" else "o", ms=6, color=INK, mec="white", mew=0.7, zorder=30)
                m.annotate(nm["name"], (nm["x"], nm["y"]), xytext=(7, 5), textcoords="offset points", fontsize=8, fontweight="bold", color=INK, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85), zorder=31)
    scalebar(m, ex, 200); north_arrow(m, ex)
    pl.text(6.6, 0.98, f"Bare-earth hillshade from the lidar, 50 m contours (labelled every 100 m), elevation {s['dtm_min']:.0f} to {s['dtm_max']:.0f} m.", fontsize=8, color=MUTED)
    colorbar(pl, hypsometric(), s["dtm_min"], s["dtm_max"], 14.2, 4.6, 0.25, 3.6, "Elevation, m", orient="v")
    # facts table
    pl.label(0.6, 3.55, "Why this site")
    rows = [(f"{s['canopy_cover_3m'] * 100:.0f}%", "of the area is covered by vegetation taller than 3 m (lidar canopy height model)"),
            (f"{s['pts_per_m2']:.1f}", f"lidar points per m² ({s['points'] / 1e6:.1f} million points in {s['area_km2']:.1f} km²)"),
            (f"{s['relief_m']:.0f} m", f"of relief, {s['dtm_min']:.0f} to {s['dtm_max']:.0f} m, including a small lake in a bowl-shaped basin"),
            (f"{s['km_to_rainier']:.0f} km", "from the Mount Rainier summit; inside the national park, 3DEP point clouds were not available")]
    for i, (v, t) in enumerate(rows):
        y = 3.1 - i * 0.62
        pl.text(0.6, y, v, fontsize=15, fontweight="bold", color=BLUE)
        pl.text(2.0, y - 0.03, "\n".join(__import__("textwrap").wrap(t, 38)), fontsize=8.3, color=INK, linespacing=1.25)
    return pl.save("01_study_area.png")


# --------------------------------------------------------------------------------------------- 02 terrain
def fig_terrain(P):
    s = P.stats
    pl = Plate("Bare-earth terrain from lidar", "Ground-classified returns interpolated to a 1 m digital terrain model; slope and topographic position derived from it", srcs(P), "Grid: UTM zone 10N, metres")
    ex = P.extent
    # large hillshade with contours
    a = pl.ax(0.6, 1.2, 6.9, 6.8)
    a.imshow(P.hs, extent=ex, cmap="gray", vmin=0.1, vmax=1)
    xs = np.linspace(ex[0], ex[1], P.dtm.shape[1]); ys = np.linspace(ex[3], ex[2], P.dtm.shape[0])
    sm = ndimage.gaussian_filter(P.dtm, 1.0)
    a.contour(xs, ys, sm, levels=np.arange(1450, 1950, 25), colors="#7a5c3c", linewidths=0.3, alpha=0.7)
    a.contour(xs, ys, sm, levels=np.arange(1500, 1950, 100), colors="#3b2f25", linewidths=0.8)
    map_frame(a, ex, 200); scalebar(a, ex, 200); north_arrow(a, ex)
    pl.label(0.6, 8.25, "Hillshade, 25 m and 100 m contours")
    # slope
    b = pl.ax(7.9, 4.45, 3.7, 3.7)
    sc = ListedColormap(["#F7F4E9", "#F2D9A6", "#E8A86A", "#C86B3C", "#7B2E1B"])
    sl = np.digitize(P.slope, [10, 20, 30, 40]) 
    b.imshow(sl, extent=ex, cmap=sc, vmin=0, vmax=4, interpolation="nearest"); b.imshow(P.hs, extent=ex, cmap="gray", alpha=0.25)
    map_frame(b, ex, 400, grid=False, labels=False); b.set_xticks([]); b.set_yticks([])
    pl.label(7.9, 8.25, "Slope classes")
    labs = ["0-10°", "10-20°", "20-30°", "30-40°", "> 40°"]
    pct = [sum(v for k, v in s["slope_pct"].items() if k.split("-")[0] in ("0",)), s["slope_pct"]["10-20"], s["slope_pct"]["20-30"], s["slope_pct"]["30-40"], s["steep_over_40"]]
    pct[0] = s["slope_pct"]["0-10"]
    for i, (c, l, p) in enumerate(zip(sc.colors, labs, pct)):
        pl.bg.add_patch(Rectangle((7.9, 4.12 - i * 0.28), 0.3, 0.2, fc=c, ec=MUTED, lw=0.4))
        pl.text(8.35, 4.2 - i * 0.28, f"{l}   {p:.0f}% of area", fontsize=8.5, va="center")
    pl.text(7.9, 2.72, f"Mean slope {s['slope_mean']:.0f}°; {s['steep_over_40']:.0f}% of the area is steeper than 40°.", fontsize=8.5, color=MUTED)
    # TPI
    c = pl.ax(12.0, 4.45, 3.4, 3.4)
    tp = np.clip(P.tpi_s, -30, 30)
    c.imshow(tp, extent=ex, cmap="BrBG_r", vmin=-30, vmax=30); c.imshow(P.hs, extent=ex, cmap="gray", alpha=0.2)
    map_frame(c, ex, 400, grid=False, labels=False); c.set_xticks([]); c.set_yticks([])
    pl.label(12.0, 8.25, "Topographic position")
    colorbar(pl, "BrBG_r", -30, 30, 12.0, 3.85, 3.4, 0.16, "Height above (+) or below (−) the 50 m neighbourhood mean, m", ticks=[-30, -15, 0, 15, 30])
    # hypsometry
    h = pl.ax(8.4, 1.5, 3.2, 0.95)
    z = np.sort(P.dtm.ravel())
    h.plot(np.linspace(0, 100, z.size), z[::-1], color=INK, lw=1.4)
    h.set_xlabel("% of area above elevation", fontsize=7.5, color=MUTED); h.set_ylabel("m", fontsize=7.5, color=MUTED)
    h.tick_params(labelsize=7, length=2); h.set_xlim(0, 100); h.spines[["top", "right"]].set_visible(False)
    pl.label(8.4, 2.6 - 0.35, "")
    pl.text(12.0, 2.35, f"Elevation {s['dtm_min']:.0f} to {s['dtm_max']:.0f} m (relief {s['relief_m']:.0f} m).\nGround returns cover {s['ground_cell_cover'] * 100:.0f}% of 1 m cells; the rest\nis interpolated (never beyond a few metres in open terrain).", fontsize=8.3, color=INK, linespacing=1.35)
    return pl.save("02_terrain.png")


# --------------------------------------------------------------------------------------------- 03 point cloud
def cloud_arrays(P, n=2_500_000, seed=3):
    c = P.R.c
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(c), size=min(n, len(c)), replace=False))
    return idx, np.c_[c.x[idx], c.y[idx], c.z[idx]]


def fig_pointcloud(P, LC):
    s = P.stats
    c = P.R.c
    pl = Plate("The lidar point cloud", f"{s['points'] / 1e6:.1f} million points over {s['area_km2']:.1f} km²; 3D views are 2.5 million point subsets, projected without a GPU", srcs(P), "Grid: UTM zone 10N, metres")
    idx, pts = cloud_arrays(P)
    hag = LC.hag[idx]
    ctr = np.array([P.aoi.cx, P.aoi.cy, float(np.median(P.dtm))])
    # view 1: coloured by elevation
    cmap = hypsometric()
    zn = (pts[:, 2] - s["dtm_min"]) / (s["dtm_max"] - s["dtm_min"])
    col1 = (cmap(np.clip(zn, 0, 1))[:, :3] * 255).astype(np.uint8)
    img1, _ = render3d.render(pts, col1, az=205, el=32, width=1500, height=900, zexag=1.3, center=ctr, scale=None)
    a = pl.ax(0.6, 4.1, 7.4, 4.1); a.imshow(img1); a.axis("off")
    pl.label(0.6, 8.35, "Coloured by elevation")
    # view 2: by class / height above ground
    base = np.full((len(pts), 3), 165, np.uint8)
    cl = c.cls[idx]
    col2 = base.copy()
    gnd = cl == 2
    col2[gnd] = (170, 150, 125)
    col2[~gnd] = (200, 190, 170)
    veg = hag > 1.0
    t = np.clip(hag / 25, 0, 1)
    gcm = cm.YlGn(0.35 + 0.65 * t)[:, :3] * 255
    col2[veg] = gcm[veg].astype(np.uint8)
    col2[cl == 9] = (43, 108, 176); col2[cl == 21] = (226, 232, 240)
    img2, _ = render3d.render(pts, col2, az=205, el=32, width=1500, height=900, zexag=1.3, center=ctr, scale=None)
    b = pl.ax(8.2, 4.1, 7.2, 4.1); b.imshow(img2); b.axis("off")
    pl.label(8.2, 8.35, "Height above ground: vegetation green, ground tan")
    # density maps
    ex = P.extent
    d = pl.ax(0.6, 0.95, 2.9, 2.9)
    im = d.imshow(np.clip(P.density, 0, 80), extent=ex, cmap="magma", vmin=0, vmax=80); d.set_xticks([]); d.set_yticks([])
    pl.label(0.6, 4.0 - 0.1, "")
    pl.text(0.6, 3.95, "All points per m² (1 m cells)", fontsize=8, color=MUTED, va="bottom")
    e = pl.ax(3.7, 0.95, 2.9, 2.9)
    e.imshow(np.clip(P.ground_density, 0, 40), extent=ex, cmap="magma", vmin=0, vmax=40); e.set_xticks([]); e.set_yticks([])
    pl.text(3.7, 3.95, "Ground points per m²", fontsize=8, color=MUTED, va="bottom")
    # cross-section: from the tallest tree to the highest ground
    ti = np.unravel_index(np.argmax(P.chm), P.chm.shape); hi = np.unravel_index(np.argmax(P.dtm), P.dtm.shape)
    x0, _, _, y1 = ex
    A = np.array([x0 + ti[1] + .5, y1 - ti[0] - .5]); B = np.array([x0 + hi[1] + .5, y1 - hi[0] - .5])
    v = B - A; L = np.hypot(*v); u = v / L; nrm = np.array([-u[1], u[0]])
    rel = np.c_[c.x - A[0], c.y - A[1]]
    al, ac = rel @ u, rel @ nrm
    m = (al >= 0) & (al <= L) & (np.abs(ac) <= 5)
    f = pl.ax(7.4, 1.25, 8.0, 2.5)
    cs = c.cls[m]; hh = LC.hag[m]
    f.scatter(al[m][cs != 2], c.z[m][cs != 2], s=0.25, c=np.clip(hh[cs != 2], 0, 30), cmap="YlGn", vmin=-6, vmax=30, linewidths=0, rasterized=True)
    f.scatter(al[m][cs == 2], c.z[m][cs == 2], s=0.25, c="#8a6d4b", linewidths=0, rasterized=True)
    f.set_xlabel("distance along line, m", fontsize=8, color=MUTED); f.set_ylabel("elevation, m", fontsize=8, color=MUTED); f.tick_params(labelsize=7.5, length=2)
    f.spines[["top", "right"]].set_visible(False)
    pl.text(7.4, 3.95, f"Cross-section, 10 m wide slab, from the tallest tree ({s['chm_max']:.0f} m) to the highest ground: ground in brown, other returns green by height", fontsize=8, color=MUTED, va="bottom")
    return pl.save("03_point_cloud.png")


# --------------------------------------------------------------------------------------------- 04 canopy
def fig_canopy(P):
    s = P.stats
    pl = Plate("Canopy height from lidar", "Canopy height model = highest return minus bare-earth terrain, 1 m cells; tree tops are local maxima of the smoothed model", srcs(P), "Grid: UTM zone 10N, metres")
    ex = P.extent
    a = pl.ax(0.6, 1.2, 7.0, 6.85)
    a.imshow(P.hs, extent=ex, cmap="gray", vmin=0.2, vmax=1.0)
    chm = np.ma.masked_less(P.chm, 2.0)
    im = a.imshow(chm, extent=ex, cmap="YlGn", vmin=2, vmax=35, alpha=0.95)
    r, c, h = P.tops
    x0, y1 = ex[0], ex[3]
    big = h > 12
    a.scatter(x0 + c[big] + .5, y1 - r[big] - .5, s=5, c="none", edgecolors="#0b3b2a", linewidths=0.35)
    map_frame(a, ex, 200); scalebar(a, ex, 200); north_arrow(a, ex)
    pl.label(0.6, 8.25, "Vegetation taller than 2 m; circles mark trees taller than 12 m")
    colorbar(pl, "YlGn", 2, 35, 8.0, 7.55, 3.2, 0.16, "Canopy height, m")
    # histogram of tree heights
    hh = pl.ax(8.0, 4.7, 3.3, 2.1)
    hh.hist(h, bins=np.arange(3, 46, 1), color="#3E8E5A", edgecolor="white", linewidth=0.3)
    hh.set_xlabel("tree top height, m", fontsize=8, color=MUTED); hh.set_ylabel("trees", fontsize=8, color=MUTED); hh.tick_params(labelsize=7.5, length=2); hh.spines[["top", "right"]].set_visible(False)
    pl.text(8.0, 6.95, f"{s['trees']:,} detected tree tops; mean {s['tree_h_mean']:.1f} m, 90th percentile {s['tree_h_p90']:.0f} m, tallest {s['tree_h_max']:.0f} m", fontsize=8, color=MUTED, wrap=True)
    # canopy cover by elevation band (treeline)
    bands = np.arange(1400, 1951, 50)
    cov = []; 
    for lo, hi in zip(bands[:-1], bands[1:]):
        m = (P.dtm >= lo) & (P.dtm < hi)
        cov.append((float((P.chm[m] > 3).mean() * 100) if m.sum() > 2000 else np.nan, int(m.sum())))
    bb = pl.ax(12.0, 4.7, 3.4, 2.1)
    mids = (bands[:-1] + bands[1:]) / 2
    bb.barh(mids, [c[0] for c in cov], height=40, color="#3E8E5A")
    bb.set_xlabel("% of cells with vegetation > 3 m", fontsize=8, color=MUTED); bb.set_ylabel("elevation, m", fontsize=8, color=MUTED); bb.tick_params(labelsize=7.5, length=2); bb.spines[["top", "right"]].set_visible(False)
    pl.text(12.0, 6.95, "Canopy cover by 50 m elevation band: the tree line", fontsize=8, color=MUTED)
    pl.label(8.0, 3.5, "Summary")
    rows = [("Canopy cover > 3 m", f"{s['canopy_cover_3m'] * 100:.1f}% of the area"), ("Canopy cover > 8 m", f"{s['canopy_cover_8m'] * 100:.1f}%"), ("Tree density", f"{s['tree_density_per_ha']:.0f} tops per hectare (whole area)"),
            ("Tallest vegetation", f"{s['chm_max']:.0f} m")]
    for i, (k, v) in enumerate(rows):
        pl.text(8.0, 3.2 - i * 0.4, k, fontsize=9, color=MUTED); pl.text(10.6, 3.2 - i * 0.4, v, fontsize=9, color=INK)
    pl.text(8.0, 1.45, "Limits: the model cannot tell a tree from a boulder or a cornice of the same height; heights are the highest return per 1 m cell,\nso they run slightly low for narrow crowns. Tree tops can split broad crowns and merge dense stands.", fontsize=7.8, color=MUTED, linespacing=1.4)
    return pl.save("04_canopy.png")


# --------------------------------------------------------------------------------------------- 05 land cover
def landcover_cmap():
    return ListedColormap([CLASSES[k][1] for k in sorted(CLASSES)])


def fig_landcover(P, LC, check):
    s = P.stats
    pl = Plate("Land cover classified from the lidar point cloud", "A rule-based classifier on per-point height above ground, return structure and first-return intensity; no imagery was used to make the map",
               srcs(P, SRC_S2.format(s2=check["s2_scene"]), SRC_WC), "Grid: UTM zone 10N, metres")
    ex = P.extent
    a = pl.ax(0.6, 1.2, 7.0, 6.85)
    a.imshow(P.hs, extent=ex, cmap="gray", vmin=0.15, vmax=1.0)
    a.imshow(np.ma.masked_equal(LC.cls, 0), extent=ex, cmap=landcover_cmap(), vmin=1, vmax=8, alpha=0.82, interpolation="nearest")
    map_frame(a, ex, 200, grid=False); scalebar(a, ex, 200); north_arrow(a, ex)
    pl.label(0.6, 8.25, "Land cover, 1 m cells, over lidar hillshade")
    # legend with areas
    areas = LC.areas(1.0)
    tot = sum(areas.values())
    pl.label(8.0, 8.25, "Classes and area")
    for i, k in enumerate(sorted(CLASSES)):
        y = 7.8 - i * 0.36
        pl.bg.add_patch(Rectangle((8.0, y - 0.11), 0.32, 0.22, fc=CLASSES[k][1], ec=MUTED, lw=0.4))
        pl.text(8.45, y, CLASSES[k][0], fontsize=8.8, va="center")
        pl.text(11.0, y, f"{areas[k] / 1e4:.1f} ha  {areas[k] / tot * 100:.1f}%", fontsize=8.8, va="center", ha="right")
    # method
    pl.label(11.7, 8.25, "How it works")
    steps = ["1  Height above ground for every point, from the lidar terrain model.", "2  Per 1 m cell: highest return, share of returns above 0.3, 2 and 5 m, ground, snow and water flags.",
             f"3  Cells under 1 m: bare ground vs short vegetation by first-return intensity (Otsu threshold {LC.features['otsu_threshold']:.0f}); green vegetation reflects the near-infrared pulse strongly.",
             "4  Taller cells split by height into shrub and three tree classes; tall returns without vertical spread are not trees.", "5  Majority filter and 9 m\u00b2 minimum mapping unit."]
    y = 7.85
    for t in steps:
        w = "\n".join(__import__("textwrap").wrap(t, 46))
        pl.text(11.7, y, w, fontsize=7.8, linespacing=1.3)
        y -= 0.2 * (w.count("\n") + 1) + 0.2
    # validation
    pl.label(8.0, 4.75, "Independent checks (made after the map)")
    ax = pl.ax(9.4, 1.55, 2.2, 2.2)
    ks = [k for k in sorted(CLASSES) if check["ndvi"].get(k) is not None]
    ax.barh([CLASSES[k][0].split(" (")[0].replace(" / rock", "") for k in ks][::-1], [check["ndvi"][k] for k in ks][::-1], color=[CLASSES[k][1] for k in ks][::-1], edgecolor="#777777", linewidth=0.4)
    ax.axvline(0, color=INK, lw=0.6); ax.set_xlim(-0.1, 0.95)
    for i, k in enumerate(ks[::-1]):
        ax.text(max(check["ndvi"][k], 0) + 0.02, i, f"{check['ndvi'][k]:.2f}", va="center", fontsize=7.5)
    ax.tick_params(labelsize=7.5, length=2); ax.spines[["top", "right"]].set_visible(False)
    pl.text(8.0, 3.85, f"Sentinel-2 NDVI inside each class\n(scene {check['s2_date']}, 5 days after the lidar;\ncells that are >80% one class)", fontsize=7.6, color=MUTED, va="bottom", linespacing=1.3)
    # confusion vs worldcover
    codes, cmx, agree = check["wc"]
    names_ = {10: "Tree", 20: "Shrub", 30: "Grass", 60: "Bare", 70: "Snow", 80: "Water"}
    cax = pl.ax(12.6, 1.55, 2.6, 2.2)
    frac = cmx / np.maximum(cmx.sum(1, keepdims=True), 1)
    cax.imshow(frac, cmap="Greens", vmin=0, vmax=1)
    cax.set_xticks(range(6)); cax.set_xticklabels([names_[c] for c in codes], fontsize=7, rotation=45, ha="right"); cax.set_yticks(range(6)); cax.set_yticklabels([names_[c] for c in codes], fontsize=7)
    for i in range(6):
        for j in range(6):
            if cmx[i, j]:
                cax.text(j, i, f"{frac[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5, color="white" if frac[i, j] > 0.55 else INK)
    cax.tick_params(length=0)
    pl.text(12.0, 3.85, f"Lidar class (rows) vs ESA WorldCover (columns),\n% of each lidar class at 10 m.\nOverall agreement {agree * 100:.0f}%", fontsize=7.6, color=MUTED, va="bottom", linespacing=1.3)
    pl.text(8.0, 1.2, "WorldCover maps most sparsely vegetated alpine ground as grassland. Where the two\ndisagree, the NDVI check supports the lidar: its bare class has a mean NDVI of {:.2f},\nfar below short vegetation ({:.2f}).".format(check["ndvi"][3], check["ndvi"][4]), fontsize=7.6, color=MUTED, va="top", linespacing=1.3)
    return pl.save("05_land_cover.png")


# --------------------------------------------------------------------------------------------- 06 radar maps
SEASONS = {"snow-free (Aug–Sep)": (8, 9), "winter (Jan–Mar)": (1, 2, 3), "spring melt (Apr–May)": (4, 5)}


def fig_radar(P, rs):
    from . import radar
    ex = rs.extent
    pl = Plate("Radar: Sentinel-1 backscatter", f"C-band synthetic aperture radar, {len(rs.dates)} {rs.direction} acquisitions from {rs.dates[0]} to {rs.dates[-1]} with the same viewing geometry; 20 m pixels, so the 1.4 km area is 70 × 70 pixels",
               srcs(P) + " " + SRC_S1, "Grid: UTM zone 10N, metres. dB = 10 log10 of Gamma0")
    vv_sf, n_sf = radar.season_median(rs, rs.vv_db, SEASONS["snow-free (Aug–Sep)"])
    vv_w, n_w = radar.season_median(rs, rs.vv_db, SEASONS["winter (Jan–Mar)"])
    vv_sp, n_sp = radar.season_median(rs, rs.vv_db, SEASONS["spring melt (Apr–May)"])
    vh_sf, _ = radar.season_median(rs, rs.vh_db, SEASONS["snow-free (Aug–Sep)"])
    lia = np.nanmean(rs.lia, axis=0)
    panels = [("Lidar hillshade (for orientation)", None, None, None, None), (f"VV, snow-free, median of {n_sf}", vv_sf, "gray", -18, -2), (f"VV, winter, median of {n_w}", vv_w, "gray", -18, -2),
              (f"VH, snow-free, median of {n_sf}", vh_sf, "gray", -28, -10), (f"VV spring minus snow-free ({n_sp} scenes)", vv_sp - vv_sf, "RdBu", -6, 6), ("Local incidence angle (mean), degrees", lia, "viridis", 20, 70)]
    P_ = 2.6
    for i, (t, arr, cmap, lo, hi) in enumerate(panels):
        l = 0.6 + (i % 3) * 3.3
        b = 5.2 if i < 3 else 1.5
        a = pl.ax(l, b, P_, P_)
        if arr is None:
            a.imshow(P.hs, extent=P.extent, cmap="gray", vmin=0.1, vmax=1)
        else:
            a.imshow(arr, extent=ex, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([]); a.set_xlim(P.extent[0], P.extent[1]); a.set_ylim(P.extent[2], P.extent[3])
        pl.text(l, b + P_ + 0.08, "\n".join(__import__("textwrap").wrap(t, 46)), fontsize=8.5, color=INK, va="bottom", fontweight="bold")
        if arr is not None:
            colorbar(pl, cmap, lo, hi, l, b - 0.48, P_, 0.1, "")
    pl.label(10.6, 7.8, "How to read")
    notes = ["Bright = strong return: rough surfaces and slopes facing the sensor. Dark = water, smooth ground, or radar shadow.",
             "Change map: red = weaker return in spring (Apr\u2013May) than in late summer, which is what wet snow does to C-band; blue = stronger.",
             "Dry snow is nearly transparent to C-band. Layover and shadow remain on steep slopes even after terrain correction.",
             "Each 20 m pixel averages 400 of the 1 m lidar cells, so fine features visible in the hillshade are not resolved in radar."]
    y = 7.4
    for t in notes:
        w = "\n".join(__import__("textwrap").wrap(t, 50))
        pl.text(10.6, y, w, fontsize=8.6, color=MUTED, linespacing=1.35, va="top")
        y -= 0.17 * (w.count("\n") + 1) + 0.25
    return pl.save("06_radar_maps.png")


# --------------------------------------------------------------------------------------------- 07 radar time series and lidar-radar link
def fig_radar_series(P, LC, rs):
    from . import radar
    x0, y1 = P.aoi.bounds[0], P.aoi.bounds[3]
    elev = radar.block_mean(P.dtm, x0, y1, 1.0, rs)
    canopy = radar.block_mean((P.chm > 3).astype(float), x0, y1, 1.0, rs)
    ok = np.isfinite(elev)
    q = np.nanpercentile(elev[ok], [0, 33.3, 66.7, 100])
    pl = Plate("Radar time series and the lidar link", "Backscatter by elevation band through the year, and how lidar canopy cover relates to the radar signal",
               srcs(P) + " " + SRC_S1, "Elevation bands: equal-area thirds of the lidar terrain model")
    months = np.array([int(str(d)[5:7]) for d in rs.dates])
    cols = ["#2B6CB0", "#7A9E3B", "#9A3B12"]
    a = pl.ax(0.9, 4.7, 7.2, 3.3)
    for bi in range(3):
        m = ok & (elev >= q[bi]) & (elev <= q[bi + 1])
        per_scene = np.nanmean(rs.vv_db[:, m], axis=1)
        med = [np.nanmedian(per_scene[months == k]) for k in range(1, 13)]
        lo = [np.nanpercentile(per_scene[months == k], 25) for k in range(1, 13)]; hi = [np.nanpercentile(per_scene[months == k], 75) for k in range(1, 13)]
        a.plot(range(1, 13), med, color=cols[bi], lw=1.8, label=f"{q[bi]:.0f}–{q[bi + 1]:.0f} m")
        a.fill_between(range(1, 13), lo, hi, color=cols[bi], alpha=0.15, lw=0)
    a.set_xticks(range(1, 13)); a.set_xticklabels(list("JFMAMJJASOND"), fontsize=8)
    a.set_ylabel("mean VV backscatter, dB", fontsize=8.5, color=MUTED); a.tick_params(labelsize=8, length=2); a.spines[["top", "right"]].set_visible(False)
    a.legend(fontsize=8, frameon=False, loc="lower left", title="Elevation band", title_fontsize=8)
    pl.text(0.9, 8.1, "VV by month of year, 2017–2021: median and inter-quartile range across years", fontsize=8.5, color=INK, fontweight="bold", va="bottom")
    # date series for top & bottom band
    b = pl.ax(8.8, 4.7, 6.6, 3.3)
    import matplotlib.dates as mdates
    dts = rs.dates.astype("datetime64[D]").astype(object)
    for bi in (0, 2):
        m = ok & (elev >= q[bi]) & (elev <= q[bi + 1])
        b.plot(dts, np.nanmean(rs.vv_db[:, m], axis=1), ".-", color=cols[bi], ms=3, lw=0.6, label=f"{q[bi]:.0f}–{q[bi + 1]:.0f} m")
    b.xaxis.set_major_locator(mdates.YearLocator()); b.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False)
    b.legend(fontsize=8, frameon=False, loc="lower left")
    pl.text(8.8, 8.1, "Every acquisition, lowest and highest band", fontsize=8.5, color=INK, fontweight="bold", va="bottom")
    # lidar link: canopy cover vs VH snow-free, and vs VV
    vh_sf, _ = radar.season_median(rs, rs.vh_db, (8, 9)); vv_sf, _ = radar.season_median(rs, rs.vv_db, (8, 9))
    c = pl.ax(0.9, 1.2, 3.4, 2.8)
    m = np.isfinite(canopy) & np.isfinite(vh_sf)
    r_vh = float(np.corrcoef(canopy[m], vh_sf[m])[0, 1]); r_vv = float(np.corrcoef(canopy[m], vv_sf[m])[0, 1])
    c.scatter(canopy[m] * 100, vh_sf[m], s=9, c="#3E8E5A", alpha=0.65, linewidths=0)
    k = np.polyfit(canopy[m] * 100, vh_sf[m], 1); xx = np.array([0, canopy[m].max() * 100]); c.plot(xx, np.polyval(k, xx), color=INK, lw=1)
    c.set_xlabel("lidar canopy cover > 3 m in the 20 m pixel, %", fontsize=8, color=MUTED); c.set_ylabel("VH snow-free, dB", fontsize=8, color=MUTED); c.tick_params(labelsize=7.5, length=2); c.spines[["top", "right"]].set_visible(False)
    pl.text(0.9, 4.15, f"Canopy cover vs VH backscatter: r = {r_vh:.2f} (VV: r = {r_vv:.2f}), n = {int(m.sum())} pixels", fontsize=8.3, color=INK, fontweight="bold", va="bottom")
    d = pl.ax(5.4, 1.2, 3.4, 2.8)
    sl = radar.block_mean(P.slope, x0, y1, 1.0, rs)
    m2 = np.isfinite(sl) & np.isfinite(vv_sf)
    d.scatter(sl[m2], vv_sf[m2], s=9, c="#555555", alpha=0.55, linewidths=0)
    r_sl = float(np.corrcoef(sl[m2], vv_sf[m2])[0, 1])
    d.set_xlabel("mean lidar slope in the pixel, degrees", fontsize=8, color=MUTED); d.set_ylabel("VV snow-free, dB", fontsize=8, color=MUTED); d.tick_params(labelsize=7.5, length=2); d.spines[["top", "right"]].set_visible(False)
    pl.text(5.4, 4.15, f"Slope vs VV backscatter: r = {r_sl:.2f}", fontsize=8.3, color=INK, fontweight="bold", va="bottom")
    # seasonal contrast numbers
    lo_m = ok & (elev <= q[1]); hi_m = ok & (elev >= q[2])
    def mm(m, ks): return float(np.nanmean(np.nanmean(rs.vv_db[np.isin(months, ks)][:, m], axis=1)))
    pl.label(9.4, 4.2, "What the series shows")
    lines = [f"Lowest third ({q[0]:.0f}–{q[1]:.0f} m): VV {mm(lo_m, (8, 9)):.1f} dB in Aug–Sep, {mm(lo_m, (1, 2, 3)):.1f} dB in Jan–Mar.",
             f"Highest third ({q[2]:.0f}–{q[3]:.0f} m): VV {mm(hi_m, (8, 9)):.1f} dB in Aug–Sep, {mm(hi_m, (1, 2, 3)):.1f} dB in Jan–Mar, {mm(hi_m, (4, 5)):.1f} dB in Apr–May.",
             f"Spring (Apr\u2013May) is the low point of the year in all three bands, consistent with wet snow absorbing the signal. That is an interpretation: no snow-cover data was used to test it.",
             "Radar pixels are 20 m, lidar cells 1 m: lidar values are averaged inside each radar pixel."]
    y = 3.75
    for t in lines:
        w = "\n".join(__import__("textwrap").wrap(t, 66))
        pl.text(9.4, y, w, fontsize=8.3, linespacing=1.3)
        y -= 0.22 * (w.count("\n") + 1) + 0.17
    return pl.save("07_radar_time_series.png"), dict(r_vh=r_vh, r_vv=r_vv, r_slope=r_sl)


# --------------------------------------------------------------------------------------------- 08 3D terrain models
def _drape(P, LC, check):
    """Return (dtm, three 1 m RGB textures): Sentinel-2 true colour, land cover, elevation tint, all shaded by lidar relief."""
    z = np.nan_to_num(P.dtm, nan=float(np.nanmedian(P.dtm)))
    h, w = z.shape
    tci = check["tci"]
    zoom = (h / tci.shape[0], w / tci.shape[1], 1)
    tci1 = np.clip(ndimage.zoom(tci.astype(float), zoom, order=1), 0, 255).astype(np.uint8)
    lcmap = landcover_cmap()
    lc = (lcmap((LC.cls.astype(float) - 1) / 7.0)[..., :3] * 255).astype(np.uint8)
    lc[LC.cls == 0] = 200
    zn = (z - P.stats["dtm_min"]) / (P.stats["dtm_max"] - P.stats["dtm_min"])
    el = (hypsometric()(np.clip(zn, 0, 1))[..., :3] * 255).astype(np.uint8)
    return z, [render3d.shade_texture(z, 1.0, t, strength=s) for t, s in ((tci1, 0.55), (lc, 0.5), (el, 0.6))]


def fig_model3d(P, LC, check):
    s = P.stats
    pl = Plate("3D terrain model from the lidar ground surface", "The 1 m lidar terrain model as a 3D surface, draped with satellite colour, the lidar land-cover classes and elevation. This is a lidar model, not a photogrammetric one",
               srcs(P, SRC_S2.format(s2=check["s2_scene"])), "Vertical exaggeration 1.3; rendered in software, no GPU")
    z, tex = _drape(P, LC, check)
    x0, y1 = P.aoi.bounds[0], P.aoi.bounds[3]
    ctr = np.array([P.aoi.cx, P.aoi.cy, float(np.median(z))])
    titles = ["Sentinel-2 true colour (5 days after the lidar)", "Lidar land cover", "Elevation tint"]
    for i, (t, rgb) in enumerate(zip(titles, tex)):
        pts, col = render3d.terrain_points(z, x0, y1, 1.0, rgb, step=1)
        img, _ = render3d.render(pts, col, az=205, el=30, width=1500, height=1050, zexag=1.3, center=ctr, splat=2)
        if i == 0:
            a = pl.ax(0.6, 2.55, 7.6, 5.32); lx, ly = 0.6, 7.98
        else:
            a = pl.ax(9.4, 4.65 if i == 1 else 1.15, 4.57, 3.2); lx, ly = 9.4, 7.98 if i == 1 else 4.5
        a.imshow(img); a.axis("off")
        pl.label(lx, ly, t)
    pl.text(0.6, 2.25, f"Relief {s['dtm_max'] - s['dtm_min']:.0f} m ({s['dtm_min']:.0f} to {s['dtm_max']:.0f} m). The surface is the lidar ground model; trees are not drawn.\n"
                      "Photogrammetry needs overlapping photographs of the site. None exist in the public archives used here,\nso no photogrammetric model is shown; see docs/photogrammetry.md.", fontsize=8.5, color=MUTED, linespacing=1.4, va="top")
    return pl.save("08_terrain_model.png")
