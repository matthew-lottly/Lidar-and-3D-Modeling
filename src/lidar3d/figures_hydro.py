"""Plates 18 and 19: hydro-flattening, depression filling and D8 drainage of the Tatoosh cirque."""
from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from .cloud import hillshade
from .plates import MUTED, Plate, colorbar, map_frame, north_arrow, scalebar

SRC = ("Lidar: USGS 3D Elevation Program (3DEP) WA_CentralWildfire_2_D22 on AWS (usgs-lidar-public), collected {c0} to {c1}; published classes 2 (ground) and 9 (water) used. "
       "Mapped streams: Overture Maps water, release 2026-09-23.1.")


def _ex(aoi):
    x0, y0, x1, y1 = aoi.bounds
    return (x0, x1, y0, y1)


def fig_flatten(S, A, aoi, dates):
    ex = _ex(aoi)
    pl = Plate("Hydro-flattened terrain and depression filling", "A lake raised to one level, closed pits filled so water can leave, and a comparison of a simple ground filter with the published ground class",
               SRC.format(c0=dates[0], c1=dates[1]), "Grid: UTM zone 10N, metres")
    L, G = S["lake"], S["ground_filter"]
    lake = A["lake"]
    rr, cc = np.nonzero(lake)
    r0, r1, c0, c1 = max(rr.min() - 25, 0), rr.max() + 25, max(cc.min() - 25, 0), cc.max() + 25
    zx = (aoi.bounds[0] + c0, aoi.bounds[0] + c1, aoi.bounds[3] - r1, aoi.bounds[3] - r0)
    sl = (slice(r0, r1), slice(c0, c1))
    dtm = A["dtm"]
    hs = hillshade(dtm, 1.0, 315, 40, 1.0)
    # raw DTM in the lake: height relative to the water level
    a = pl.ax(0.7, 4.5, 3.6, 3.6)
    a.imshow(hs[sl], extent=zx, cmap="gray", vmin=0.2, vmax=1.0)
    rel = np.where(lake, dtm - L["level_m"], np.nan)
    a.imshow(rel[sl], extent=zx, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    a.set_xticks([]); a.set_yticks([])
    pl.label(0.7, 8.35, "Lake in the raw terrain model")
    colorbar(pl, "RdBu_r", -0.3, 0.3, 0.7, 4.0, 3.6, 0.12, "Raw model minus water level, m", ticks=[-0.3, 0, 0.3])
    b = pl.ax(4.5, 4.5, 3.6, 3.6)
    b.imshow(hs[sl], extent=zx, cmap="gray", vmin=0.2, vmax=1.0)
    b.imshow(np.where(lake, 1.0, np.nan)[sl], extent=zx, cmap="Blues", vmin=0, vmax=2, alpha=0.85)
    b.set_xticks([]); b.set_yticks([])
    pl.label(4.5, 8.35, "After hydro-flattening")
    pl.text(4.5, 4.05, f"One level, {L['level_m']:.2f} m", fontsize=8.5, color=MUTED, va="top")
    # depression fill map, whole area, log depth
    dep = A["filled"] - A["flat"]
    d = pl.ax(8.4, 4.5, 3.6, 3.6)
    d.imshow(hillshade(dtm, 1.0, 315, 40, 1.0), extent=ex, cmap="gray", vmin=0.2, vmax=1.0)
    d.imshow(np.where(dep > 0.02, dep, np.nan), extent=ex, cmap="YlOrRd", vmin=0, vmax=1.0, interpolation="nearest")
    d.set_xticks([]); d.set_yticks([])
    pl.label(8.4, 8.35, "Closed pits that were filled")
    colorbar(pl, "YlOrRd", 0, 1.0, 8.4, 4.0, 3.6, 0.12, "Fill depth, m (above 1 m shares the end colour)", ticks=[0, 0.5, 1.0])
    # ground filter disagreement
    g = pl.ax(12.2, 4.5, 3.2, 3.6)
    zmin, dtm0, ours = A["zmin"], A["dtm"], A["ours"]
    dz = zmin - dtm0
    cls = np.zeros(dz.shape, np.uint8)
    cls[(dz > 0.5) & ours] = 1                      # we call ground, but the lowest return is well above the published terrain
    cls[(np.abs(dz) <= 0.15) & ~ours] = 2           # we call non-ground, but the return is on the terrain
    from matplotlib.colors import ListedColormap
    g.imshow(hillshade(dtm0, 1.0, 315, 40, 1.0), extent=ex, cmap="gray", vmin=0.2, vmax=1.0)
    g.imshow(np.ma.masked_equal(cls, 0), extent=ex, cmap=ListedColormap(["#C0392B", "#2B6CB0"]), vmin=1, vmax=2, interpolation="nearest")
    g.set_xticks([]); g.set_yticks([])
    pl.label(12.2, 8.35, "Own ground filter vs published")
    # text
    dp, d8 = S["depressions"], S["d8"]
    t1 = (f"The lake. Water has no ground returns, so the terrain model interpolates across it from the shoreline: the streaks on the left. The surface varies by {L['dtm_std_before_m'] * 100:.0f} cm (5th to 95th percentile {L['dtm_p5']:.2f} to {L['dtm_p95']:.2f} m), "
          f"enough for water to appear to run uphill. It is replaced by the median of {L['water_points']:,} water-classified returns, {L['level_m']:.2f} m. The lake was already at its spill level: filling raised it by {dp['in_lake']:.2f} m.")
    t2 = (f"Depression filling. {dp['count']:,} closed pits (mostly boulders, hollows and single-cell noise) hold {dp['volume_m3']:,.0f} m³ in total; the deepest is {dp['max_depth_m']:.2f} m. Filling lets every cell drain off the grid; "
          f"{d8['interior_sinks']} interior sink cells remain after flat routing, from flats with no outlet.")
    t3 = (f"Ground filter. A progressive morphological filter on the lowest return per cell keeps {G['ground_share'] * 100:.0f}% of cells as ground and misses {G['omission'] * 100:.1f}% of cells that sit on the terrain, but it also keeps "
          f"{G['commission'] * 100:.0f}% of the cells whose lowest return is more than 0.5 m above the published terrain (red on the map; {G['canopy_commission'] * 100:.0f}% inside canopy). These cells have no ground return, only shrubs and boulders, "
          "which a grid of lowest returns cannot separate from ground. The terrain model here therefore uses the published ground class; blue marks terrain cells the filter drops.")
    y = 3.55
    for t in (t1, t2, t3):
        w = "\n".join(textwrap.wrap(t, 118))
        pl.text(0.7, y, w, fontsize=8.5, color=MUTED, linespacing=1.35)
        y -= 0.19 * (w.count("\n") + 1) + 0.2
    return pl.save("18_hydro_flattening.png")


def fig_drainage(S, A, aoi, dates):
    ex = _ex(aoi)
    ch, cd, hh = S["channels"], S["defined"], S["hand"]
    pl = Plate("Micro-drainage from lidar: D8 flow paths", "Each 1 m cell drains to its steepest lower neighbour; channels are cells with a large upstream area that also sit in a hollow",
               SRC.format(c0=dates[0], c1=dates[1]), "Grid: UTM zone 10N, metres")
    dtm = A["dtm"]
    hs = hillshade(dtm, 1.0, 315, 40, 1.0)
    a = pl.ax(0.6, 1.2, 7.0, 7.0)
    a.imshow(hs, extent=ex, cmap="gray", vmin=0.2, vmax=1.0)
    from scipy import ndimage
    dil = ndimage.maximum_filter(np.where(A["defined"], A["acc"], 0), size=3)        # drawn 3 cells wide for legibility; the analysis uses 1 m cells
    acc = np.where(dil > 0, dil, np.nan)
    a.imshow(acc, extent=ex, cmap="Blues", norm=LogNorm(vmin=1000, vmax=2e5), interpolation="nearest")
    a.imshow(np.where(A["lake"], 1.0, np.nan), extent=ex, cmap="Blues", vmin=0, vmax=1.2, alpha=0.9)
    sm = A["smask"]
    ys, xs = np.nonzero(sm)
    if len(xs):
        a.scatter(ex[0] + xs + 0.5, ex[3] - ys - 0.5, s=3, c="#C0392B", lw=0)
    map_frame(a, ex, 200, grid=False); scalebar(a, ex, 200); north_arrow(a, ex)
    pl.label(0.6, 8.35, "Defined channels (drawn 3 m wide), shaded by upstream area")
    colorbar(pl, "Blues", 1000, 2e5, 8.1, 7.55, 3.6, 0.14, "Upstream area, m² (log scale)", ticks=[1e4, 1e5])
    if ch["mapped_streams_in_area"]:
        pl.bg.add_patch(Rectangle((12.0, 7.55), 0.2, 0.15, fc="#C0392B", ec="none")); pl.text(12.3, 7.62, "mapped stream (Overture)", fontsize=8.5, va="center")
    else:
        pl.text(12.0, 7.62, "No stream is mapped in this area", fontsize=8.5, va="center", color=MUTED)
    # threshold table
    pl.label(8.1, 6.75, "How much channel depends on the threshold")
    y = 6.45
    pl.text(8.1, y, "Upstream area", fontsize=8.3, color=MUTED, fontweight="bold"); pl.text(10.5, y, "All flow paths", fontsize=8.3, color=MUTED, fontweight="bold"); pl.text(12.6, y, "Per km²", fontsize=8.3, color=MUTED, fontweight="bold")
    for k, (t, v) in enumerate(ch["by_threshold"].items()):
        yy = y - 0.27 * (k + 1)
        pl.text(8.1, yy, f"{int(t):,} m²" + ("   (used)" if float(t) == ch["threshold_m2"] else ""), fontsize=8.5)
        pl.text(10.5, yy, f"{v / 1000:,.1f} km", fontsize=8.5)
        pl.text(12.6, yy, f"{v / 1000 / S['area_km2']:,.0f} km", fontsize=8.5)
    # HAND bars
    b = pl.ax(8.4, 3.0, 3.4, 1.5)
    ts = list(hh)
    xs_ = np.arange(len(ts)); w = 0.38
    b.bar(xs_ - w / 2, [hh[t]["share_main"] * 100 for t in ts], w, color="#8A8A8A", label="streams (upstream area ≥ 5,000 m²)")
    b.bar(xs_ + w / 2, [hh[t]["share_defined"] * 100 for t in ts], w, color="#2B6CB0", label="all defined channels")
    b.set_xticks(xs_); b.set_xticklabels([f"≤ {t} m" for t in ts], fontsize=8)
    b.set_ylabel("% of the area", fontsize=8, color=MUTED); b.tick_params(labelsize=7.5, length=2); b.spines[["top", "right"]].set_visible(False); b.legend(fontsize=7.5, frameon=False, loc="upper left")
    pl.label(8.4, 4.8, "Ground close to the drainage (height above nearest drainage)")
    mp = (f"Overture maps {ch['mapped_streams_in_area']} stream line here ({ch['mapped_length_m']:.0f} m); the derived network recovers {cd['mapped_stream_recall'] * 100:.0f}% of it within 15 m."
          if ch["mapped_streams_in_area"] else "Overture maps no stream or river line anywhere in this area, so there is nothing to match the derived network against.")
    notes = [f"Defined channels: {cd['length_m'] / 1000:,.1f} km, {cd['density_km_per_km2']:.0f} km per km². Without the hollow test, flow paths above {ch['threshold_m2']:,.0f} m² total {ch['length_m'] / 1000:,.1f} km; {ch['share_unmapped_under_0p5m'] * 100:.0f}% of the unmapped ones sit less than 0.5 m below their surroundings.",
             mp + f" All {cd['unmapped_length_m'] / 1000:,.1f} km of defined channel are absent from that map. This shows the map is sparse, not that satellite imagery cannot see them: no imagery was compared, and a 10 m pixel is wider than most of these channels.",
             f"Lake catchment {S['lake']['catchment_km2'] * 100:.1f} ha. Height above drainage is a terrain proxy for flood-prone ground, not a hydraulic model: no rainfall, snowmelt, roughness or culverts."]
    y = 2.55
    for t in notes:
        w_ = "\n".join(textwrap.wrap(t, 100))
        pl.text(8.1, y, w_, fontsize=8.2, color=MUTED, linespacing=1.3, va="top"); y -= 0.165 * (w_.count("\n") + 1) + 0.14
    return pl.save("19_hydro_drainage.png")
