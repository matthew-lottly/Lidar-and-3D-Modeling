"""Plates 51 to 55 for the Norfolk drainage and flood-stage road vulnerability study."""
from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from . import flood as FL
from . import sitemap
from .cloud import hillshade
from .plates import MUTED, Plate

STAGES = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
RED, ORANGE, GREEN, GREY = "#C0392B", "#E67E22", "#2E8B57", "#9AA0A6"


def _src(st):
    return (f"Lidar: USGS 3DEP {st['project']}, flown {st['collected']} (vertical datum {st['vertical_datum']}). Roads: Overture Maps (OpenStreetMap-derived). "
            "Stages are water-surface heights.")


def _halo(w=2.0, c="white"):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=w, foreground=c)]


def _scalebar(ax, x0, y0, length, color="k"):
    import matplotlib.patheffects as pe
    halo = "black" if color == "white" else "white"
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.5, solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=4.5, foreground=halo)], zorder=20)
    ax.text(x0 + length / 2, y0 + 12, f"{length:g} m", ha="center", va="bottom", fontsize=7.5, color=color, path_effects=_halo(2.0, halo), zorder=21)


def _rows(pl, x, y0, rows, width, fs=8.2):
    for k_, v_ in rows:
        pl.text(x, y0, k_, fontsize=9, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v_, width))
        pl.text(x, y0 - 0.24, w, fontsize=fs, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.27 + 0.16 * (w.count("\n") + 1) + 0.14
    return y0


def _hs(F):
    return hillshade(np.nan_to_num(F["dtm"], nan=float(np.nanmedian(F["dtm"]))), 1.0, 315, 40)


def _vcbar(pl, x, y, h, cmap, vmin, vmax, label, norm=None, ticks=None):
    import matplotlib.pyplot as plt
    cb = pl.ax(x, y, 0.1, h)
    v = np.linspace(vmax, vmin, 256)
    cols = plt.get_cmap(cmap)((norm(v) if norm is not None else (v - vmin) / (vmax - vmin)))
    cb.imshow(cols[:, None, :], aspect="auto", extent=(0, 1, vmin, vmax))
    cb.set_xticks([]); cb.yaxis.tick_right(); cb.yaxis.set_label_position("right"); cb.tick_params(labelsize=7, length=2)
    if ticks is not None:
        cb.set_yticks(ticks)
    cb.set_ylabel(label, fontsize=7.5, color=MUTED, labelpad=4)


def fig_site(st, F, naip_img):
    bbox = (-76.55, 36.7, -76.1, 37.0)
    L = sitemap.regional_layers("norfolk", bbox, "Virginia", pop_min=30000)
    lon, lat = st["lonlat"]
    h = 700
    dlon = h / (111000 * np.cos(np.radians(lat))); dlat = h / 111000
    pl = Plate("Where the Norfolk study area is", "A 1.4 km square of Ghent, the Hague and Hampton Boulevard in the city of Norfolk, Virginia, a flat coastal-plain neighbourhood",
               _src(st), "Dashed grey: city and county lines. Red: freeways. Blue: water")
    a = pl.ax(0.6, 1.1, 7.0, 7.2)
    sitemap.draw_regional(a, L, bbox, (lon, lat), (dlon, dlat), label_places=("Norfolk", "Portsmouth", "Virginia Beach", "Chesapeake", "Hampton", "Newport News"))
    a.annotate("study area", xy=(lon, lat), xytext=(lon + 0.1, lat + 0.1), fontsize=9, fontweight="bold", color="#E74C3C", ha="center", arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2), path_effects=_halo())
    pl.label(0.6, 8.35, "Region")
    b = pl.ax(8.2, 4.1, 3.8, 3.8)
    b.imshow(np.moveaxis(naip_img[:3], 0, -1), extent=(-h, h, -h, h)); b.set_xticks([]); b.set_yticks([])
    _scalebar(b, -h + 60, -h + 50, 200.0, "white")
    pl.label(8.2, 8.0, "Aerial imagery")
    c = pl.ax(12.2, 4.1, 3.0, 3.8)
    pl.label(12.2, 8.0, "Bare-earth terrain")
    im = c.imshow(F["dtm"], origin="lower", extent=(-h, h, -h, h), cmap="terrain", vmin=0, vmax=6)
    c.set_xticks([]); c.set_yticks([])
    del im
    e = st["dtm_holdout"]
    rows = [("Why here", "Ghent and the Hague lie low, about 2 to 4 m above the vertical datum in the lidar, in a coastal-plain city where street flooding is a recognised problem; this study does not model any particular tide or storm."),
            ("Lidar", f"{st['density_per_m2']:.1f} returns per m2 in 2013; about {st['ground_and_road_density_per_m2']:.1f} per m2 reach the ground or road surface, the rest hit roofs and trees. Terrain is built at 1 m; held-out ground returns are matched within {e['rmse_m'] * 100:.1f} cm rms."),
            ("Elevations", "Terrain elevations are in the lidar's vertical datum (NAVD88). Tide levels are reported against other datums; converting a tide level to NAVD88 needs the local datum offset, which was not available offline here, so stages are given directly as NAVD88 heights.")]
    _rows(pl, 8.2, 3.8, rows, 86)
    return pl.save("51_norfolk_site_map.png")


def fig_hydrology(st, F):
    h = 700
    ext = (-h, h, -h, h)
    pl = Plate("Drainage paths and depressions in the bare-earth terrain", "Where water would run, where it would pool, and where a road embankment may be holding it back",
               _src(st), "D8 flow (pysheds); accumulation = 1 m cells draining through")
    hs = _hs(F)
    W = 3.7
    a = pl.ax(0.6, 4.4, W, W)
    a.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95)
    im = a.imshow(np.ma.masked_less(F["dtm"], -50), origin="lower", extent=ext, cmap="terrain", vmin=1, vmax=5, alpha=0.5)
    a.set_xticks([]); a.set_yticks([])
    _scalebar(a, -h + 60, -h + 60, 200.0)
    pl.label(0.6, 8.25, "Terrain height (m)")
    _vcbar(pl, 4.4, 4.4, W, "terrain", 1, 5, "m above datum", ticks=[1, 2, 3, 4, 5])
    del im
    b = pl.ax(5.0, 4.4, W, W)
    b.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95)
    acc = np.where(F["acc"] >= 100, F["acc"], np.nan)
    b.imshow(acc, origin="lower", extent=ext, cmap="plasma_r", norm=LogNorm(100, 50000), interpolation="nearest")
    b.set_xticks([]); b.set_yticks([])
    pl.label(5.0, 8.25, "Flow accumulation (log)")
    _vcbar(pl, 8.8, 4.4, W, "plasma_r", np.log10(100), np.log10(50000), "cells draining", norm=lambda v: (v - np.log10(100)) / (np.log10(50000) - np.log10(100)), ticks=[np.log10(v) for v in (100, 1000, 10000)])
    c = pl.ax(9.5, 4.4, W, W)
    c.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95)
    d = np.where(F["depth"] >= 0.15, F["depth"], np.nan)
    c.imshow(d, origin="lower", extent=ext, cmap="YlOrRd", vmin=0.15, vmax=2.0, interpolation="nearest")
    for r in F["roads"]:
        c.plot(r["coords"][:, 0], r["coords"][:, 1], color="#2C3E50", lw=0.3, alpha=0.6)
    for cd in F["cands"]:
        c.plot(cd["local_x"], cd["local_y"], "o", ms=6, mfc="none", mec="#1F6FB2", mew=1.6)
    c.set_xticks([]); c.set_yticks([])
    pl.label(9.5, 8.25, "Closed depressions and culvert candidates")
    _vcbar(pl, 13.3, 4.4, W, "YlOrRd", 0.15, 2.0, "depth to spill, m")
    hy = st["hydrology"]
    sv = st["synthetic_culvert_test"]["by_embankment_height"]
    rows = [("Drainage", f"{hy['stream_length_km_raw']:.0f} km of cells draining 3,000 m2 or more (0.3 ha). On a flat plain these are the street gutters and low yards, not natural channels; most of the water is routed along the streets and towards the low ground in the south-west and the underpasses of the freeway and rail line."),
            ("Depressions", f"{hy['depressions']} closed depressions of 60 m2 or more and 15 cm or deeper hold {hy['depression_volume_m3'] / 1000:.0f} thousand m3 over {hy['depression_area_ha']:.0f} ha; the deepest ({hy['max_depression_depth_m']:.0f} m) are underpass and tunnel approach trenches, which drain through pumps in reality."),
            ("Culvert candidates (blue rings)", f"{hy['culvert_candidates']} depressions spill at a road and are held back by at least 10 cm of embankment; together they would hold {hy['candidate_storage_m3'] / 1000:.0f} thousand m3. The strongest barrier is {max(x['barrier_m'] for x in F['cands']):.2f} m (Hampton Boulevard). Candidates with under 20 cm of barrier are within about 3 times the terrain's vertical error and are weak evidence."),
            ("Detection test", f"Planted embankments with terrain-level noise: found {sv['0.5']['blocked_detected'] * 100:.0f}% of the time at 0.5 m, {sv['1']['blocked_detected'] * 100:.0f}% at 1 m and {sv['2']['blocked_detected'] * 100:.0f}% at 2 m, 0% at 0.3 m or less, and never flagged when a culvert was open.")]
    _rows(pl, 0.6, 4.0, rows, 120, 8.0)
    return pl.save("52_norfolk_drainage.png")


def _road_colors(F, lv):
    S = F["S"]
    conn = S[f"connected_depth_ge_0.3_{lv:g}"].to_numpy()
    deep = (S[f"max_depth_{lv:g}"] >= 0.30).to_numpy()
    el = S["elevated"].to_numpy()
    return np.where(el, GREY, np.where(conn, RED, np.where(deep, ORANGE, GREEN)))


def fig_stages(st, F):
    h = 700
    ext = (-h, h, -h, h)
    pl = Plate("Which streets go under at each water level", "Bathtub flooding connected to the open water in the window, with each road coloured by whether 30 cm or more of water would stand on it",
               _src(st), "Static bathtub model; no rain, drains or pumps")
    hs = _hs(F)
    dtm = F["dtm"]
    src = F["water"]
    roads = F["roads"]
    W = 3.0
    for k, lv in enumerate(STAGES):
        r_, c_ = divmod(k, 3)
        x0 = 0.6 + c_ * 3.4
        y0 = 4.8 - r_ * 3.75
        ax = pl.ax(x0, y0, W, W)
        ax.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95)
        flood = FL.bathtub(dtm, lv, src, True)
        dep = np.where(flood, lv - dtm, np.nan)
        ax.imshow(dep, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=2.5, interpolation="nearest")
        cols = _road_colors(F, lv)
        for rd, cc in zip(roads, cols):
            ax.plot(rd["coords"][:, 0], rd["coords"][:, 1], color=cc, lw=0.9 if cc != GREEN else 0.5, alpha=0.95 if cc != GREEN else 0.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(-h + 25, h - 30, f"{lv:g} m", fontsize=11, fontweight="bold", color="black", va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
        net = st["road_vulnerability"][f"{lv:g}"]["connected_to_tidal_water"]
        pl.text(x0, y0 - 0.06, f"{net['km_impassable']:.1f} km impassable;  {st['stages'][f'{lv:g}']['raw']['connected_ha']:.0f} ha flooded", fontsize=7.8, color=MUTED, va="top")
    leg = [Line2D([], [], color=RED, lw=2, label="impassable (30 cm or more), connected"), Line2D([], [], color=ORANGE, lw=2, label="would pond (30 cm or more), not connected"), Line2D([], [], color=GREEN, lw=2, label="clear"),
           Line2D([], [], color=GREY, lw=2, label="bridge, tunnel or overpass")]
    lax = pl.ax(10.9, 6.5, 4.4, 1.4); lax.axis("off"); lax.legend(handles=leg, loc="upper left", fontsize=8, frameon=False, labelspacing=0.7)
    _vcbar(pl, 10.9, 4.1, 2.1, "Blues", 0, 2.5, "water depth, m")
    rows = [("Reading the maps", "A stage is the height of the water surface above the vertical datum. Terrain below it that is connected to the open water seen in the lidar (areas with no returns, in the south-west of the window) floods; other low ground (orange roads) only holds rain. Water arriving from outside the window by other routes is not modelled, and the open water here is a set of ponds and basins, not a mapped tidal creek.")]
    _rows(pl, 12.1, 4.0, rows, 36, 7.8)
    return pl.save("53_norfolk_flood_stages.png")


def fig_vulnerability(st, F):
    pl = Plate("How much of the road network is lost as the water rises", "Kilometres of drivable road with 30 cm or more of water, and how much of the network stays connected",
               _src(st), "Drivable roads; bridges and tunnels excluded; 30 cm = impassable (scenario)")
    rv = st["road_vulnerability"]
    tot = st["roads"]["drivable_km"]
    x = np.arange(len(STAGES))
    a = pl.ax(0.9, 4.6, 4.4, 3.2)
    low = [rv[f"{lv:g}"]["any_low_ground"]["km_impassable"] for lv in STAGES]
    con = [rv[f"{lv:g}"]["connected_to_tidal_water"]["km_impassable"] for lv in STAGES]
    a.bar(x - 0.19, low, 0.36, color=ORANGE, label="any low ground (ponding)")
    a.bar(x + 0.19, con, 0.36, color=RED, label="connected to open water")
    a.set_xticks(x); a.set_xticklabels([f"{lv:g}" for lv in STAGES]); a.set_xlabel("water-surface height, m above datum", fontsize=8, color=MUTED); a.set_ylabel(f"km impassable (of {tot:.0f} km drivable)", fontsize=8, color=MUTED)
    a.tick_params(labelsize=8, length=2); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=7.8, frameon=False, loc="upper left")
    pl.label(0.9, 8.35, "Road length under water")
    b = pl.ax(6.1, 4.6, 4.0, 3.2)
    b.plot(x, [rv[f"{lv:g}"]["any_low_ground"]["largest_connected_share_of_km"] * 100 for lv in STAGES], "-o", color=ORANGE, label="any low ground")
    b.plot(x, [rv[f"{lv:g}"]["connected_to_tidal_water"]["largest_connected_share_of_km"] * 100 for lv in STAGES], "-o", color=RED, label="connected to open water")
    b.set_xticks(x); b.set_xticklabels([f"{lv:g}" for lv in STAGES]); b.set_xlabel("water-surface height, m", fontsize=8, color=MUTED); b.set_ylabel("largest connected piece, % of network km", fontsize=8, color=MUTED); b.set_ylim(0, 100)
    b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False); b.legend(fontsize=7.8, frameon=False)
    pl.label(6.1, 8.35, "Network that stays connected")
    c = pl.ax(11.1, 4.6, 4.0, 3.2)
    S = F["S"]
    first = S.first_impassable_stage.to_numpy()
    bins = list(STAGES) + [np.nan]
    cls_order = ["motorway", "primary", "secondary", "tertiary", "residential", "service", "unclassified"]
    bottom = np.zeros(len(bins))
    cmap = {"motorway": "#6C3483", "primary": "#C0392B", "secondary": "#E67E22", "tertiary": "#F1C40F", "residential": "#2E86C1", "service": "#7F8C8D", "unclassified": "#1ABC9C"}
    for k in cls_order:
        v = []
        for lv in bins:
            m = (S.cls == k) & ((S.first_impassable_stage == lv) if lv == lv else S.first_impassable_stage.isna()) & ~S.elevated
            v.append(S.length_m[m].sum() / 1000)
        c.bar(range(len(bins)), v, bottom=bottom, color=cmap[k], label=k)
        bottom += np.array(v)
    c.set_xticks(range(len(bins))); c.set_xticklabels([f"{b:g}" for b in STAGES] + ["never"]); c.set_xlabel("first stage at which 30 cm of water stands on the road", fontsize=8, color=MUTED); c.set_ylabel("km of road", fontsize=8, color=MUTED)
    c.tick_params(labelsize=8, length=2); c.spines[["top", "right"]].set_visible(False); c.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    del first
    pl.label(11.1, 8.35, "When each road class first goes under")
    d = pl.ax(0.9, 0.95, 5.2, 3.0)
    d.imshow(_hs(F), origin="lower", extent=(-700, 700, -700, 700), cmap="gray", vmin=0.2, vmax=0.95, alpha=0.6)
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("plasma_r")
    for rd, f_ in zip(F["roads"], S.first_impassable_stage.to_numpy()):
        d.plot(rd["coords"][:, 0], rd["coords"][:, 1], color=cm((f_ - 1.5) / 2.5) if f_ == f_ else "#BBBBBB", lw=1.1 if f_ == f_ else 0.4)
    d.set_xticks([]); d.set_yticks([]); d.set_aspect("equal")
    pl.label(0.9, 4.15, "First stage at which each road goes under (dark = early)")
    _vcbar(pl, 6.15, 0.95, 3.0, "plasma_r", 1.5, 4.0, "stage, m", ticks=[1.5, 2, 2.5, 3, 3.5, 4])
    r2, r3, r4 = rv["2.5"], rv["3"], rv["4"]
    rows = [("Reading it", f"Of {tot:.0f} km of drivable road, {r2['any_low_ground']['km_impassable']:.0f} km has 30 cm of water on it at a stage of 2.5 m if every low spot floods, {r3['any_low_ground']['km_impassable']:.0f} km at 3 m and {r4['any_low_ground']['km_impassable']:.0f} km at 4 m; with only water connected to open water the figures are {r2['connected_to_tidal_water']['km_impassable']:.0f}, {r3['connected_to_tidal_water']['km_impassable']:.0f} and {r4['connected_to_tidal_water']['km_impassable']:.0f} km."),
            ("Network", f"At 3 m the largest connected piece of the drivable network holds {r3['connected_to_tidal_water']['largest_connected_share_of_km'] * 100:.0f}% of its length ({r3['connected_to_tidal_water']['pieces']} pieces) if only water connected to the open water is counted, {r3['any_low_ground']['largest_connected_share_of_km'] * 100:.0f}% if every low spot floods; at 3.5 m the network is almost entirely cut."),
            ("Not modelled", "Storm drains and pumps, rainfall, wave run-up, and the sub-street flow of water through the sewer system. Real streets flood more slowly than a bathtub fills, so these are upper bounds for a flood that lasts hours, and nearer to the truth for a slow rise.")]
    _rows(pl, 9.7, 3.95, rows, 66, 7.9)
    return pl.save("54_norfolk_road_vulnerability.png")


def fig_tests(st, F):
    pl = Plate("How far can the flood and drainage numbers be trusted?", "Terrain accuracy, the effect of vertical error on flooded area, planted embankments, and sensitivity to resolution and thresholds",
               _src(st), "Terrain error measured on held-out lidar ground and road returns")
    e = st["dtm_holdout"]
    a = pl.ax(0.9, 5.2, 3.2, 2.65)
    a.bar([0, 1, 2], [e["ground_rmse_m"] * 100, e["road_surface_rmse_m"] * 100, e["p95_abs_m"] * 100], color=["#2E86C1", "#7F8C8D", "#1B4F72"])
    for i, v in enumerate([e["ground_rmse_m"] * 100, e["road_surface_rmse_m"] * 100, e["p95_abs_m"] * 100]):
        a.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=8)
    a.set_xticks(range(3)); a.set_xticklabels(["ground\nrms", "road surface\nrms", "all, 95th\npercentile"], fontsize=7.5); a.set_ylabel("held-out error, cm", fontsize=8, color=MUTED); a.tick_params(labelsize=8, length=2); a.spines[["top", "right"]].set_visible(False)
    pl.label(0.9, 8.35, "Terrain error (10% of returns held out)")
    b = pl.ax(4.9, 5.2, 4.4, 2.65)
    lv = np.array(STAGES)
    base = np.array([st["stages"][f"{v:g}"]["raw"]["connected_ha"] for v in lv])
    lo1 = np.array([st["stages"][f"{v:g}"]["raw_stage_minus_0.1"] for v in lv]); hi1 = np.array([st["stages"][f"{v:g}"]["raw_stage_plus_0.1"] for v in lv])
    lo2 = np.array([st["stages"][f"{v:g}"]["raw_stage_minus_0.2"] for v in lv]); hi2 = np.array([st["stages"][f"{v:g}"]["raw_stage_plus_0.2"] for v in lv])
    b.fill_between(lv, lo2, hi2, color="#AED6F1", label="+-20 cm"); b.fill_between(lv, lo1, hi1, color="#5DADE2", label="+-10 cm"); b.plot(lv, base, "k-o", ms=4, label="terrain as measured")
    b.set_xlabel("stage, m", fontsize=8, color=MUTED); b.set_ylabel("flooded area connected to open water, ha", fontsize=8, color=MUTED); b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False); b.legend(fontsize=7.5, frameon=False, loc="upper left")
    pl.label(4.9, 8.35, "Flooded area if the stage is off by 10 or 20 cm")
    c = pl.ax(10.2, 5.2, 4.8, 2.65)
    sv = st["synthetic_culvert_test"]["by_embankment_height"]
    hs_ = list(sv)
    c.bar(np.arange(len(hs_)) - 0.19, [sv[k]["blocked_detected"] for k in hs_], 0.36, color="#1F6FB2", label="blocked: found")
    c.bar(np.arange(len(hs_)) + 0.19, [sv[k]["open_culvert_flagged"] for k in hs_], 0.36, color="#C0392B", label="culvert open: wrongly flagged")
    c.set_xticks(range(len(hs_))); c.set_xticklabels([f"{float(k):g}" for k in hs_]); c.set_xlabel("height of a planted road embankment, m", fontsize=8, color=MUTED); c.set_ylabel("share of trials", fontsize=8, color=MUTED); c.set_ylim(0, 1.15)
    c.tick_params(labelsize=8, length=2); c.spines[["top", "right"]].set_visible(False); c.legend(fontsize=7.5, frameon=False, loc="upper left")
    pl.label(10.2, 8.35, "Planted embankments (noise as measured)")
    d = pl.ax(0.9, 1.3, 4.2, 2.6)
    sr = st["sensitivity_resolution"]
    x = np.arange(2)
    d.bar(x - 0.19, [sr["stage_2.5"]["connected_1m_ha"], sr["stage_3.5"]["connected_1m_ha"]], 0.36, color="#1B4F72", label="1 m cells")
    d.bar(x + 0.19, [sr["stage_2.5"]["connected_2m_ha"], sr["stage_3.5"]["connected_2m_ha"]], 0.36, color="#5DADE2", label="2 m cells")
    d.set_xticks(x); d.set_xticklabels(["stage 2.5 m", "stage 3.5 m"]); d.set_ylabel("connected flooded area, ha", fontsize=8, color=MUTED); d.tick_params(labelsize=8, length=2); d.spines[["top", "right"]].set_visible(False); d.legend(fontsize=7.5, frameon=False)
    pl.label(0.9, 4.1, "Cell size")
    f = pl.ax(5.8, 1.3, 4.2, 2.6)
    rawh = [st["stages"][f"{v:g}"]["raw"]["connected_ha"] for v in lv]
    ob = [st["stages"][f"{v:g}"]["raw"]["connected_outside_buildings_ha"] for v in lv]
    f.plot(lv, rawh, "-o", color="#1B4F72", label="all connected cells"); f.plot(lv, ob, "-o", color="#E67E22", label="outside buildings and trees")
    f.set_xlabel("stage, m", fontsize=8, color=MUTED); f.set_ylabel("ha", fontsize=8, color=MUTED); f.tick_params(labelsize=8, length=2); f.spines[["top", "right"]].set_visible(False); f.legend(fontsize=7.5, frameon=False, loc="upper left")
    pl.label(5.8, 4.1, "Buildings and tree crowns are filled terrain")
    g = st["stages"]
    rows = [("Terrain", f"Held-out ground returns are matched within {e['ground_rmse_m'] * 100:.1f} cm rms and road-surface returns within {e['road_surface_rmse_m'] * 100:.1f} cm (95th percentile of all: {e['p95_abs_m'] * 100:.0f} cm). This is interpolation and point error; the lidar's absolute vertical accuracy against tide gauges or benchmarks was not checked."),
            ("Vertical error", f"A 10 cm error in the stage changes the connected flooded area at 3 m from {g['3']['raw_stage_minus_0.1']:.0f} to {g['3']['raw_stage_plus_0.1']:.0f} ha (central value {g['3']['raw']['connected_ha']:.0f}), and at 2.5 m from {g['2.5']['raw_stage_minus_0.1']:.0f} to {g['2.5']['raw_stage_plus_0.1']:.0f} ha (central {g['2.5']['raw']['connected_ha']:.0f}): on a plain this flat the answer is very sensitive to the datum."),
            ("Culverts", "Opening the culvert candidates changes the connected flooded area by 0 ha at every stage: culverts matter for how fast rain drains, not for where a still water level ends up.")]
    _rows(pl, 10.4, 4.1, rows, 60, 7.9)
    return pl.save("55_norfolk_accuracy_tests.png")
