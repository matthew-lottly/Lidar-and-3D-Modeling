"""Plates 39 to 45: rooftop solar study (site map, roof extraction, irradiance maps, accuracy tests, 3D view)."""
from __future__ import annotations

import textwrap

import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from . import sitemap
from .cloud import hillshade
from .plates import MUTED, Plate

KW = (600.0, 1500.0)
CMAP = "inferno"
NAMES = {"westu": "West University Place", "downtown": "Downtown Houston"}
COL = {"westu": "#B9770E", "downtown": "#1F4E79"}


def _src(st):
    return (f"Lidar: USGS 3DEP {st['lidar_project']}, flown {st['collected']}. Irradiance: NREL NSRDB typical meteorological year (4 km grid, hourly DNI and DHI). Orientation layers: Overture Maps (OpenStreetMap-derived); "
            "never used to build the model.")


def _halo(w=2.0, c="white"):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=w, foreground=c)]


def _scalebar(ax, x0, y0, length, color="k"):
    import matplotlib.patheffects as pe
    halo = "black" if color == "white" else "white"
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.5, solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=4.5, foreground=halo)], zorder=20)
    ax.text(x0 + length / 2, y0, f"{length:g} m", ha="center", va="bottom", fontsize=7.5, color=color, path_effects=_halo(2.0, halo), zorder=21)


def _vcbar(pl, x, y, h, cmap, vmin, vmax, label):
    cb = pl.ax(x, y, 0.1, h)
    cb.imshow(np.linspace(1, 0, 200)[:, None], aspect="auto", cmap=cmap, extent=(0, 1, vmin, vmax))
    cb.set_xticks([]); cb.yaxis.tick_right(); cb.yaxis.set_label_position("right"); cb.tick_params(labelsize=7, length=2)
    cb.set_ylabel(label, fontsize=7.5, color=MUTED, labelpad=4)


def _text_rows(pl, x, y0, rows, width, fs=8.2):
    for k_, v_ in rows:
        pl.text(x, y0, k_, fontsize=9, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v_, width))
        pl.text(x, y0 - 0.24, w, fontsize=fs, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.27 + 0.16 * (w.count("\n") + 1) + 0.14
    return y0


def fig_sites(sts, imgs):
    bbox = (-95.8, 29.5, -95.0, 30.1)
    L = sitemap.regional_layers("houston_rooftops", bbox, "Texas", pop_min=40000)
    pl = Plate("Where the two blocks are", "A residential block (West University Place) and a high-rise block (downtown Houston), 400 m squares, in Harris County, Texas",
               _src(sts["westu"]), "Red: freeways. Orange: trunk roads. Blue: rivers and lakes. Dashed grey: county lines")
    a = pl.ax(0.6, 1.1, 7.3, 7.2)
    for k, st in sts.items():
        lon, lat = st["lonlat"]
        sitemap.draw_regional(a, L, bbox, (lon, lat), (0.0021, 0.0018), label_places=("Houston", "Pasadena", "Sugar Land", "Katy", "Bellaire", "Cypress", "Humble", "Baytown", "Pearland"))
        a.annotate(NAMES[k], xy=(lon, lat), xytext=(lon + (0.07 if k == "westu" else 0.1), lat + (-0.1 if k == "westu" else 0.12)), fontsize=9, fontweight="bold", color="#E74C3C", ha="center", arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2), path_effects=_halo())
    a.set_xlim(bbox[0], bbox[2]); a.set_ylim(bbox[1], bbox[3])
    pl.label(0.6, 8.35, "Region")
    for i, k in enumerate(("westu", "downtown")):
        b = pl.ax(8.35 + i * 3.55, 4.3, 3.35, 3.35)
        h = sts[k]["analysis_area_ha"] ** 0.5 * 50
        b.imshow(np.moveaxis(imgs[k][:3], 0, -1), extent=(-h, h, -h, h)); b.set_xticks([]); b.set_yticks([])
        _scalebar(b, -h + 0.06 * 2 * h, -h + 0.05 * 2 * h, 100.0, "white")
        pl.label(8.35 + i * 3.55, 7.8, NAMES[k])
    rows = [("West University Place", f"29.7175 N, 95.4335 W. Single-family houses under mature trees; lidar {sts['westu']['density_per_m2']:.0f} returns per m2 (flown {sts['westu']['collected']})."),
            ("Downtown Houston", f"29.757 N, 95.365 W. Office towers up to {sts['downtown']['buildings']['height_max_m']:.0f} m with flat roofs; lidar {sts['downtown']['density_per_m2']:.0f} returns per m2 (flown {sts['downtown']['collected']})."),
            ("Shading context", "Each block is analysed in a 400 m square; the surface model for shading extends 300 m further on every side, so towers and trees outside the square still cast shadows into it.")]
    _text_rows(pl, 8.35, 4.0, rows, 74)
    return pl.save("39_rooftop_site_map.png")


def _roof_map(ax, F, cmapname, norm, size):
    cells = F["cells"]
    n = int(2 * F["ahalf"])
    img = np.full((n, n), np.nan)
    ix = np.floor(cells.x.to_numpy() + F["ahalf"]).astype(int)
    iy = np.floor(cells.y.to_numpy() + F["ahalf"]).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    img[iy[ok], ix[ok]] = cells.total.to_numpy()[ok]
    return img


def _base(F):
    h, a = F["half"], F["ahalf"]
    off = int(h - a)
    n = int(2 * a)
    dsm = F["dsm"][off:off + n, off:off + n]
    dtm = F["dtm"][off:off + n, off:off + n]
    return hillshade(np.nan_to_num(dsm, nan=float(np.nanmedian(dsm))), 1.0, 315, 40), dsm - dtm


def fig_extraction(key, st, F):
    pl = Plate(f"Finding the buildings and their roofs: {NAMES[key]}", "Roof returns are chosen by shape (locally planar, single-return, above 2.5 m), grouped into footprints, then split into planes by RANSAC",
               _src(st), "Footprints from lidar only; Overture footprints shown for comparison")
    a_ = F["ahalf"]
    ext = (-a_, a_, -a_, a_)
    hs, ndsm = _base(F)
    W = 3.35
    pl.label(0.6, 4.75 + W + 0.13, "Surface model and footprints")
    a = pl.ax(0.6, 4.75, W, W)
    a.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95)
    for p in F["polys"].values():
        a.plot(p[:, 0], p[:, 1], color="#E74C3C", lw=0.7)
    for q in F["ovp"]:
        x, y = q.exterior.xy
        a.plot(x, y, color="#2E86C1", lw=0.5, ls="--")
    a.set_xlim(-a_, a_); a.set_ylim(-a_, a_); a.set_xticks([]); a.set_yticks([])
    _scalebar(a, -a_ + 15, -a_ + 12, 100.0, "white")
    leg = [Line2D([], [], color="#E74C3C", lw=1.5, label="from lidar"), Line2D([], [], color="#2E86C1", lw=1.2, ls="--", label="Overture")]
    la = pl.ax(0.6, 4.35, W, 0.3); la.axis("off"); la.legend(handles=leg, loc="center left", ncol=2, fontsize=7.5, frameon=False)
    pl.label(4.7, 4.75 + W + 0.13, "Roof planes (colour = plane)")
    b = pl.ax(4.7, 4.75, W, W)
    b.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95, alpha=0.6)
    cm_ = plt_cm()
    for bd in F["B"]:
        for j in range(len(bd.planes)):
            s = bd.cell_plane == j
            b.scatter(bd.cells[s, 0], bd.cells[s, 1], s=1.4, color=cm_(j % 10), marker="s", lw=0)
    b.set_xlim(-a_, a_); b.set_ylim(-a_, a_); b.set_xticks([]); b.set_yticks([])
    pl.label(0.6, 0.95 + W + 0.13, "Roof slope (degrees)")
    c = pl.ax(0.6, 0.95, W, W)
    c.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95, alpha=0.5)
    cells = F["cells"]
    sc = c.scatter(cells.x, cells.y, c=cells.slope, s=1.4, cmap="viridis", vmin=0, vmax=45, marker="s", lw=0)
    c.set_xlim(-a_, a_); c.set_ylim(-a_, a_); c.set_xticks([]); c.set_yticks([])
    _vcbar(pl, 4.02, 0.95, W, "viridis", 0, 45, "slope, deg")
    del sc
    pl.label(4.7, 0.95 + W + 0.13, "Roof direction (facing)")
    d = pl.ax(4.7, 0.95, W, W)
    d.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95, alpha=0.5)
    ps = F["ps"].set_index(["building", "plane"])
    asp = np.array([ps.aspect_deg.get((int(bb), int(pp)), np.nan) for bb, pp in zip(cells.bid, cells.plane)])
    flat = cells.slope.to_numpy() < 10
    d.scatter(cells.x[flat], cells.y[flat], s=1.4, color="#BBBBBB", marker="s", lw=0)
    pit = ~flat
    d.scatter(cells.x[pit], cells.y[pit], c=asp[pit], s=1.4, cmap="twilight_shifted", vmin=0, vmax=360, marker="s", lw=0)
    d.set_xlim(-a_, a_); d.set_ylim(-a_, a_); d.set_xticks([]); d.set_yticks([])
    cb = pl.ax(8.07, 0.95, 0.1, W)
    cb.imshow(np.linspace(360, 0, 200)[:, None], aspect="auto", cmap="twilight_shifted", extent=(0, 1, 0, 360))
    cb.set_xticks([]); cb.yaxis.tick_right(); cb.set_yticks([0, 90, 180, 270, 360]); cb.set_yticklabels(["N", "E", "S", "W", "N"]); cb.tick_params(labelsize=7, length=2)
    ck, bl = st["checks"], st["buildings"]
    fo = ck["footprints_vs_overture"]
    rows = [("Buildings", f"{bl['n']} structures of 30 m2 or more; median footprint {bl['footprint_median_m2']:.0f} m2, median height {bl['height_median_m']:.1f} m (maximum {bl['height_max_m']:.0f} m). {bl['planes_total']} roof planes ({bl['planes_per_building_median']:.0f} per building, median); {bl['flat_roof_area_share'] * 100:.0f}% of roof area is flat (under 10 degrees)."),
            ("Against Overture", f"Median overlap (IoU) with the best Overture footprint {fo['lidar_iou_median']:.2f}; {fo['lidar_with_iou_ge_0p5'] * 100:.0f}% of lidar footprints reach 0.5 and {fo['lidar_with_no_overlap'] * 100:.0f}% overlap nothing. {fo['overture_covered_ge_half'] * 100:.0f}% of Overture buildings are at least half covered. Footprint area: lidar {fo['lidar_area_ha']:.1f} ha, Overture {fo['overture_area_ha']:.1f} ha."),
            ("What it misses", "Only returns that look planar are used, so roofs hidden by tree crowns, small outbuildings and dormers are partly or wholly missing; the footprint then covers only the visible part of a roof. Trees over a roof also remain in the shading surface.")]
    _text_rows(pl, 8.6, 8.0, rows, 74)
    return pl.save({"westu": "40_westu_roof_extraction.png", "downtown": "42_downtown_roof_extraction.png"}[key])


def plt_cm():
    import matplotlib.pyplot as plt
    return plt.get_cmap("tab10")


def fig_solar(key, st, F, num):
    pl = Plate(f"Rooftop solar: {NAMES[key]}", "Annual energy arriving on each roof (direct sun plus sky light), with the shadows of nearby buildings and trees",
               _src(st), "kWh per m2 of roof surface per year; typical weather, not a particular year")
    a_ = F["ahalf"]
    ext = (-a_, a_, -a_, a_)
    hs, _ = _base(F)
    norm = Normalize(*KW)
    W = 3.7
    a = pl.ax(0.6, 4.55, W, W)
    a.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95, alpha=0.55)
    cells = F["cells"]
    a.scatter(cells.x, cells.y, c=cells.total, s=1.5, cmap=CMAP, norm=norm, marker="s", lw=0)
    a.set_xlim(-a_, a_); a.set_ylim(-a_, a_); a.set_xticks([]); a.set_yticks([])
    _scalebar(a, -a_ + 15, -a_ + 12, 100.0, "white")
    _vcbar(pl, 4.4, 4.55, W, CMAP, KW[0], KW[1], "kWh per m2 per year")
    pl.label(0.6, 4.55 + W + 0.13, "Annual irradiance on roofs")
    b = pl.ax(5.1, 4.55, W, W)
    b.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0.2, vmax=0.95, alpha=0.55)
    loss = 1 - cells.beam / np.maximum(cells.beam_open, 1e-9)
    b.scatter(cells.x, cells.y, c=loss.clip(0, 1) * 100, s=1.5, cmap="Blues", vmin=0, vmax=100, marker="s", lw=0)
    b.set_xlim(-a_, a_); b.set_ylim(-a_, a_); b.set_xticks([]); b.set_yticks([])
    _vcbar(pl, 8.9, 4.55, W, "Blues", 0, 100, "direct sun lost to shade, %")
    pl.label(5.1, 4.55 + W + 0.13, "Share of direct sun blocked")
    T = F["T"]
    c = pl.ax(0.9, 1.05, 3.5, 2.7)
    c.hist(T.total_kwh_yr / 1000, bins=np.logspace(np.log10(max(T.total_kwh_yr.min() / 1000, 1)), np.log10(T.total_kwh_yr.max() / 1000), 24), color=COL[key])
    c.set_xscale("log"); c.set_xlabel("energy on the roof, MWh per year", fontsize=8, color=MUTED); c.set_ylabel("buildings", fontsize=8, color=MUTED)
    c.xaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["x"]).FuncFormatter(lambda v, _: f"{v:g}"))
    c.tick_params(labelsize=8, length=2); c.spines[["top", "right"]].set_visible(False)
    pl.label(0.9, 3.95, "Energy per rooftop")
    d = pl.ax(5.2, 1.05, 3.5, 2.7)
    so = st["solar"]["by_orientation"]
    names = ["flat", "south", "east", "west", "north"]
    vals = [so[n]["mean_total_kwh_m2"] or 0 for n in names]
    d.bar(range(5), vals, color=[cm.inferno(norm(v)) for v in vals])
    for i, v in enumerate(vals):
        d.text(i, v + 15, f"{v:.0f}", ha="center", fontsize=8)
    d.set_xticks(range(5)); d.set_xticklabels(["flat", "faces S", "faces E", "faces W", "faces N"], fontsize=8)
    d.set_ylabel("mean kWh per m2 per year", fontsize=8, color=MUTED); d.set_ylim(0, 1700); d.tick_params(labelsize=8, length=2); d.spines[["top", "right"]].set_visible(False)
    pl.label(5.2, 3.95, "By roof direction (pitched roofs 10 degrees or more)")
    s = st["solar"]
    rows = [("Totals", f"{st['buildings']['n']} roofs, {st['buildings']['roof_area_ha']:.1f} ha of roof. Energy on the roofs {s['total_energy_on_roofs_gwh']:.1f} GWh per year: {s['roof_area_weighted_beam_kwh_m2']:.0f} kWh/m2 direct and {s['diffuse_kwh_m2']:.0f} diffuse on average (open-sky direct would be {s['beam_open_sky_kwh_m2']:.0f}: shade removes {s['beam_shading_loss'] * 100:.0f}% of the direct sun)."),
            ("Suitable roof", f"{s['suitable_roof_area_share'] * 100:.0f}% of roof area receives at least {s['suitable_threshold_kwh_m2']:.0f} kWh/m2 per year, a screening level. At {s['pv_assumptions']['efficiency'] * 100:.0f}% module efficiency and {s['pv_assumptions']['performance_ratio'] * 100:.0f}% performance ratio (assumed, not measured) that suitable area would yield about {s['pv_estimate_gwh']:.1f} GWh of electricity a year."),
            ("Weather", f"Hourly DNI and DHI of the NSRDB typical year at {st['weather']['site_lat']:.2f} N, {st['weather']['site_lon']:.2f} W ({st['weather']['annual_ghi_kwh_m2']:.0f} kWh/m2 global horizontal a year). A typical year is not a forecast for any one year."),
            ("Not included", "Light reflected from the ground or neighbouring walls, the bright sky near the sun, soiling, snow, roof obstructions the lidar missed, and roofs hidden under trees.")]
    _text_rows(pl, 9.1, 8.2, rows, 70)
    return pl.save(f"{num}_{key}_rooftop_solar.png")


def fig_tests(sts):
    pl = Plate("How far can the rooftop numbers be trusted?", "Planted roofs, repeat fits, footprints against Overture, geometry against pvlib, and what changes the totals",
               _src(sts["westu"]), "Planted roofs are clean synthetic gables and flats, so real error is larger")
    cols = COL
    a = pl.ax(0.9, 5.15, 3.4, 2.7)
    ks = list(sts["westu"]["checks"]["planted_roofs"])
    labs = ["real density\nand noise", "40% of the\ndensity", "10 cm noise"]
    for i, k in enumerate(("westu", "downtown")):
        pr = sts[k]["checks"]["planted_roofs"]
        v = [pr[kk]["slope_err_p95_deg"] for kk in ks]
        a.bar(np.arange(3) + i * 0.38 - 0.19, v, 0.36, color=cols[k], label=NAMES[k])
    a.set_xticks(range(3)); a.set_xticklabels(labs, fontsize=7.5); a.set_ylabel("slope error, 95th percentile, deg", fontsize=8, color=MUTED); a.tick_params(labelsize=8, length=2); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=7.5, frameon=False)
    pl.label(0.9, 8.35, "Planted roofs: slope error")
    b = pl.ax(5.4, 5.15, 3.4, 2.7)
    for i, k in enumerate(("westu", "downtown")):
        pr = sts[k]["checks"]["planted_roofs"]
        v = [pr[kk]["irradiance_err_p95_kwh_m2"] for kk in ks]
        b.bar(np.arange(3) + i * 0.38 - 0.19, v, 0.36, color=cols[k])
    b.set_xticks(range(3)); b.set_xticklabels(labs, fontsize=7.5); b.set_ylabel("irradiance error, 95th pct, kWh/m2/yr", fontsize=8, color=MUTED); b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False)
    pl.label(5.4, 8.35, "Planted roofs: energy error")
    c = pl.ax(9.9, 5.15, 5.0, 2.7)
    x = np.arange(6)
    cc = sts["westu"]["checks"]["beam_vs_pvlib"]
    c.bar(x - 0.19, [r["mine_kwh_m2"] for r in cc], 0.36, color="#7F8C8D", label="this code")
    c.bar(x + 0.19, [r["pvlib_kwh_m2"] for r in cc], 0.36, color="#27AE60", label="pvlib")
    c.set_xticks(x); c.set_xticklabels([f"{r['tilt']} deg\nfaces {r['aspect']}" for r in cc], fontsize=7.5); c.set_ylabel("annual beam, kWh/m2", fontsize=8, color=MUTED); c.tick_params(labelsize=8, length=2); c.spines[["top", "right"]].set_visible(False); c.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center")
    c.set_ylim(0, 1500)
    pl.label(9.9, 8.35, "Sun geometry against pvlib")
    d = pl.ax(0.9, 1.3, 4.3, 2.5)
    allnames = ["baseline", "no_trees_in_shading_surface", "coarser_2m_shading_surface", "half_the_returns"]
    lab_ = ["baseline", "trees removed\nfrom shading", "2 m shading\nsurface", "half the\nreturns"]
    for i, (k, st) in enumerate(sts.items()):
        sv = st["sensitivity"]
        vals = [sv[n]["mean_total"] if n in sv else np.nan for n in allnames]
        d.bar(np.arange(4) + i * 0.38 - 0.19, vals, 0.36, color=cols[k])
        for j, v in enumerate(vals):
            if np.isfinite(v):
                d.text(j + i * 0.38 - 0.19, v + 12, f"{v:.0f}", ha="center", fontsize=7)
    d.set_xticks(range(4)); d.set_xticklabels(lab_, fontsize=7.5); d.set_ylabel("mean kWh/m2/yr on roofs", fontsize=8, color=MUTED); d.set_ylim(0, 1600); d.tick_params(labelsize=8, length=2); d.spines[["top", "right"]].set_visible(False)
    pl.label(0.9, 4.05, "What moves the average")
    e = pl.ax(6.0, 1.3, 3.0, 2.5)
    for k, st in sts.items():
        r = st["checks"]["split_half_largest_plane"]
        e.bar([0 if k == "westu" else 1], [r["normal_angle_p95_deg"]], color=cols[k])
        e.text(0 if k == "westu" else 1, r["normal_angle_p95_deg"] + 0.2, f"median {r['normal_angle_median_deg']:.1f}", ha="center", fontsize=7.5)
    e.set_xticks([0, 1]); e.set_xticklabels(["West U", "Downtown"], fontsize=8); e.set_ylabel("plane tilt difference between\ntwo halves of the returns, 95th pct, deg", fontsize=7.5, color=MUTED); e.tick_params(labelsize=8, length=2); e.spines[["top", "right"]].set_visible(False)
    pl.label(6.0, 4.05, "Same roof, two random halves")
    rows = []
    for k, st in sts.items():
        ck = st["checks"]
        pr = ck["planted_roofs"][ks[0]]
        sa = ck["shadow_vs_analytic"]
        rows.append((NAMES[k], f"Planted roofs at this block's density: slope error median {pr['slope_err_median_deg']:.1f} deg, 95th percentile {pr['slope_err_p95_deg']:.1f}; energy error median {pr['irradiance_err_median_kwh_m2']:.0f}, 95th pct {pr['irradiance_err_p95_kwh_m2']:.0f} kWh/m2/yr. "
                              f"Shadow engine against a 20 m block: horizon within {max(abs(r_['horizon_deg'] - r_['expected_deg']) for r_ in sa):.1f} deg of the exact angle. Footprints: {ck['footprints_vs_overture']['lidar_with_iou_ge_0p5'] * 100:.0f}% overlap Overture by IoU 0.5 or more."))
    _text_rows(pl, 9.6, 3.95, rows, 56, 7.9)
    return pl.save("45_rooftop_accuracy_tests.png")


def fig_3d(sts, scenes):
    pl = Plate("The blocks in 3D", "Roof planes fitted from lidar and coloured by the energy they receive in a typical year; walls grey; terrain pale",
               _src(sts["westu"]), "3D files: outputs/rooftops/<block>/block.glb and block.obj")
    for i, k in enumerate(("westu", "downtown")):
        im = scenes[k]
        c = im
        m = (c < 250).any(2)
        r, cc = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        c = c[max(r[0] - 8, 0):r[-1] + 8, max(cc[0] - 8, 0):cc[-1] + 8]
        h, w = c.shape[:2]
        W = 7.3
        H = min(W * h / w, 6.0)
        W = H * w / h
        a = pl.ax(0.6 + i * 7.6 + (7.3 - W) / 2, 1.5, W, H)
        a.imshow(c); a.axis("off")
        pl.label(0.6 + i * 7.6, 1.5 + 6.0 + 0.4, NAMES[k])
    _vcbar(pl, 15.15, 2.2, 3.0, CMAP, KW[0], KW[1], "kWh per m2 per year")
    return pl.save("44_rooftop_3d.png")
