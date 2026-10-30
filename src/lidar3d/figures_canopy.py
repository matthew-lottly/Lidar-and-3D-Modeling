"""Plates for the canopy-structure study: site maps, results per site, accuracy tests and comparison."""
from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy import ndimage

from . import sitemap
from .plates import MUTED, Plate

GREEN = "#2E7D32"
SITE_VIEW = dict(
    houston=dict(bbox=(-96.0, 29.5, -94.9, 30.35), region="Texas", places=("Houston", "Cypress", "Tomball", "The Woodlands", "Katy", "Spring", "Conroe", "Sugar Land", "Pasadena", "Humble"), pop_min=20000, land=()),
    tahoe=dict(bbox=(-120.9, 38.85, -119.7, 39.75), region="California", places=("Truckee", "South Lake Tahoe", "Reno", "Auburn", "Grass Valley", "Nevada City", "Incline Village", "Sparks", "Carson City"), pop_min=3000, land=("Tahoe National Forest",)),
)


def _src(st):
    return (f"Lidar: USGS 3DEP {st['project']}, flown {st['collected']}. Imagery: NAIP {st['naip_agreement']['naip_date']}. Orientation layers: Overture Maps (OpenStreetMap-derived); not used in the analysis.")


def _halo(color="white", w=2.2):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=w, foreground=color)]


def _scalebar(ax, extent, length, y_frac=0.05, color="k"):
    """Scale bar with a white outline on the bar and the label so it reads over any background."""
    import matplotlib.patheffects as pe
    x0 = extent[0] + 0.05 * (extent[1] - extent[0])
    y0 = extent[2] + y_frac * (extent[3] - extent[2])
    halo = "black" if color == "white" else "white"
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.5, solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=4.5, foreground=halo)], zorder=20)
    ax.text(x0 + length / 2, y0 + 0.02 * (extent[3] - extent[2]), f"{length:g} m", ha="center", va="bottom", fontsize=7.5, color=color, path_effects=_halo(halo), zorder=21)


def fig_site(key, st, rgb):
    """Regional map, state locator and the study area in imagery."""
    v = SITE_VIEW[key]
    L = sitemap.regional_layers(key, v["bbox"], v["region"], pop_min=v["pop_min"], land_names=v["land"])
    lon, lat = st["lonlat"]
    half = st["area_ha"] ** 0.5 * 50.0
    dlat = half / 111000.0
    dlon = half / (111000.0 * np.cos(np.radians(lat)))
    county = sitemap.county_of(L, lon, lat)
    land = sitemap.land_of(L, lon, lat)
    nl = sitemap.nearest_land(L, lon, lat)
    land_txt = (f"; inside {land[0]}" if land else (f"; {nl[1]:.1f} km outside the mapped boundary of {nl[0]}" if nl and nl[1] < 5 else ""))
    pl = Plate(f"Where the study area is: {st['site'].split(',')[0]}", f"The {st['area_ha'] / 100:.2f} km2 square outlined in red, in its region and in the state",
               _src(st), "Dashed grey: county lines. Red: freeways. Orange: trunk roads. Blue: rivers and lakes" + (". Green: national forest" if land or v["land"] else ""))
    a = pl.ax(0.6, 1.1, 8.2, 7.2)
    sitemap.draw_regional(a, L, v["bbox"], (lon, lat), (dlon, dlat), label_places=v["places"])
    from shapely.geometry import box as _box
    place_names = {n for n, _, _ in L["places"]}
    W_, S_, E_, N_ = v["bbox"]
    inner = _box(W_ + 0.08 * (E_ - W_), S_ + 0.06 * (N_ - S_), E_ - 0.08 * (E_ - W_), N_ - 0.06 * (N_ - S_))
    for nm, g in L["counties"]:
        gg = g.intersection(_box(*v["bbox"]))
        if gg.is_empty or nm in place_names:
            continue
        c = gg.representative_point()
        near_place = any(abs(pg.x - c.x) < 0.12 * (E_ - W_) and abs(pg.y - c.y) < 0.05 * (N_ - S_) for pn, _, pg in L["places"] if pn in v["places"])
        if inner.contains(c) and not near_place:
            a.text(c.x, c.y, nm, fontsize=7.5, color="#7A7A7A", ha="center", style="italic", zorder=2, path_effects=_halo())
    a.annotate("study area", xy=(lon, lat), xytext=(lon - 0.2 * (E_ - W_), lat + 0.13 * (N_ - S_)), fontsize=9, fontweight="bold", color="#E74C3C", zorder=9, ha="center",
               arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.3), path_effects=_halo())
    pl.label(0.6, 8.35, "Region")
    b = pl.ax(9.2, 6.05, 3.0, 2.3)
    for g in L["region"]:
        for p in (g.geoms if hasattr(g, "geoms") else [g]):
            if p.geom_type == "Polygon" and p.area > 0.02:
                b.fill(*p.exterior.xy, fc="#ECE9DD", ec="#8A8A8A", lw=0.6)
    b.plot(lon, lat, "o", color="#E74C3C", ms=6); b.set_aspect(1 / np.cos(np.radians(lat))); b.axis("off")
    pl.label(9.2, 8.35, f"In {v['region']}")
    pl.text(12.5, 8.2, "\n".join(textwrap.wrap(f"{county or 'County not found'}{land_txt}. Centre {lat:.4f} N, {abs(lon):.4f} W. {st['area_ha']:.0f} ha. Lidar {st['density_per_m2']:.1f} returns per m2.", 34)), fontsize=8.6, color=MUTED, linespacing=1.4, va="top")
    c = pl.ax(9.2, 1.15, 4.8, 4.7)
    ext = (-st["area_ha"] ** 0.5 * 50, st["area_ha"] ** 0.5 * 50) * 2
    c.imshow(np.moveaxis(rgb[:3], 0, -1), extent=(ext[0], ext[1], ext[0], ext[1]))
    c.set_xticks([]); c.set_yticks([])
    _scalebar(c, (ext[0], ext[1], ext[0], ext[1]), 200.0 if half > 600 else 100.0, color="white")
    pl.label(9.2, 6.05, "The study area in aerial imagery")
    return pl.save(f"{'32' if key == 'houston' else '34'}_{key}_site_map.png"), county, (land or ([f"{nl[0]} ({nl[1]:.1f} km outside the mapped boundary)"] if nl and nl[1] < 5 else []))


def _crop_best(trees_rc, shape, win):
    h, _, _ = np.histogram2d(trees_rc[0], trees_rc[1], bins=[shape[0] // 10, shape[1] // 10], range=[[0, shape[0]], [0, shape[1]]])
    k = win // 10
    ker = np.ones((k, k))
    s = ndimage.convolve(h, ker, mode="constant")
    i, j = np.unravel_index(np.argmax(s[k // 2: -k // 2, k // 2: -k // 2]), (s.shape[0] - k, s.shape[1] - k))
    return (i + 0) * 10, (j + 0) * 10


def _vcbar(pl, x, y, h, cmap, vmin, vmax, label, norm=None):
    """Vertical colour bar in its own axes beside a panel (ticks and label to its right, so nothing sits on the map)."""
    cb = pl.ax(x, y, 0.1, h)
    grad = np.linspace(1, 0, 200)[:, None]
    cb.imshow(grad, aspect="auto", cmap=cmap, extent=(0, 1, vmin, vmax))
    cb.set_xticks([]); cb.yaxis.tick_right(); cb.yaxis.set_label_position("right")
    cb.tick_params(labelsize=7, length=2)
    cb.set_ylabel(label, fontsize=7.5, color=MUTED, labelpad=4)
    return cb


def fig_results(key, st, F, num):
    chm, lab, trees, gl, dens = F["chm"], F["lab"], F["trees"], F["gl"], F["dens"]
    n = chm.shape[0]
    ext = (-n / 2, n / 2, -n / 2, n / 2)
    pl = Plate(f"Canopy structure: {st['site'].split(',')[0].split('(')[0].strip()}", f"Canopy height from lidar, {len(trees['height']):,} individual trees, tree density and canopy openings ({st['area_ha']:.0f} ha)",
               _src(st), "Heights above the ground; canopy = 2 m or higher; model cell 1 m; trees = local maxima of a smoothed height model")
    vmax = float(np.percentile(chm[chm >= 2], 99.5)) if (chm >= 2).any() else 30.0
    cm = ListedColormap(["#F2EFE4"] + [c_ for c_ in __import__("matplotlib.cm", fromlist=["cm"]).YlGn(np.linspace(0.25, 1.0, 255))])
    W = 3.35

    def box(x, y, title):
        ax = pl.ax(x, y, W, W)
        ax.set_xticks([]); ax.set_yticks([])
        pl.label(x, y + W + 0.13, title)
        return ax
    a = box(0.6, 4.75, "Canopy height model")
    a.imshow(np.where(chm < 0.5, 0, chm), origin="lower", extent=ext, cmap=cm, vmin=0, vmax=vmax)
    _scalebar(a, ext, 200.0 if n > 1000 else 100.0)
    _vcbar(pl, 4.02, 4.75, W, cm, 0, vmax, "height, m")
    b = box(4.95, 4.75, "Individual trees (colour = height)")
    b.imshow(np.where(chm < 0.5, 0, chm), origin="lower", extent=ext, cmap=cm, vmin=0, vmax=vmax, alpha=0.5)
    b.scatter(trees["col"] + 0.5 - n / 2, trees["row"] + 0.5 - n / 2, c=trees["height"], s=1.6, cmap="plasma", vmin=0, vmax=vmax, lw=0)
    b.set_xlim(ext[0], ext[1]); b.set_ylim(ext[2], ext[3])
    _vcbar(pl, 8.37, 4.75, W, "plasma", 0, vmax, "tree height, m")
    c = box(0.6, 0.95, "Trees per hectare (50 m cells)")
    dmax = max(float(np.percentile(dens, 99)), 1.0)
    c.imshow(dens, origin="lower", extent=(ext[0], ext[0] + dens.shape[1] * 50, ext[2], ext[2] + dens.shape[0] * 50), cmap="magma_r", vmin=0, vmax=dmax)
    c.set_xlim(ext[0], ext[1]); c.set_ylim(ext[2], ext[3])
    _vcbar(pl, 4.02, 0.95, W, "magma_r", 0, dmax, "trees per ha")
    d = box(4.95, 0.95, "Canopy openings by size")
    img = np.zeros(chm.shape + (3,))
    img[:] = (0.95, 0.94, 0.90)
    img[chm >= 2] = (0.18, 0.49, 0.20)
    area = ndimage.sum(np.ones_like(gl), gl, np.arange(1, gl.max() + 1)) if gl.max() else np.zeros(0)
    colors = [((10, 100), (0.99, 0.85, 0.35)), ((100, 1000), (0.95, 0.55, 0.15)), ((1000, 1e4), (0.80, 0.25, 0.15)), ((1e4, 1e12), (0.45, 0.10, 0.25))]
    asz = np.concatenate([[0], area])[gl]
    for (lo, hi), col in colors:
        img[(gl > 0) & (asz >= lo) & (asz < hi)] = col
    d.imshow(img, origin="lower", extent=ext)
    leg = [Line2D([], [], marker="s", ls="none", color=(0.18, 0.49, 0.20), label="canopy"), Line2D([], [], marker="s", ls="none", color=(0.95, 0.94, 0.90), mec="#BBBBBB", label="open, under 10 m2")] + \
          [Line2D([], [], marker="s", ls="none", color=col, label=(f"opening {lo:g} to {hi:g} m2" if hi < 1e11 else f"opening over {lo:g} m2")) for (lo, hi), col in colors]
    lax = pl.ax(8.95, 0.95, 6.4, 0.55)
    lax.axis("off")
    lax.legend(handles=leg, loc="center left", ncol=3, fontsize=7.6, frameon=False, columnspacing=1.4, handletextpad=0.5)
    win = 150
    r0, c0 = _crop_best((trees["row"], trees["col"]), chm.shape, win)
    z = pl.ax(8.95, 4.75, W, W)
    z.imshow(np.where(chm < 0.5, 0, chm)[r0:r0 + win, c0:c0 + win], origin="lower", extent=(0, win, 0, win), cmap=cm, vmin=0, vmax=vmax)
    sub = lab[r0:r0 + win, c0:c0 + win]
    for lv in np.unique(sub)[1:]:
        z.contour(np.arange(win) + 0.5, np.arange(win) + 0.5, (sub == lv).astype(float), levels=[0.5], colors="#1B1B1B", linewidths=0.35)
    m = (trees["row"] >= r0) & (trees["row"] < r0 + win) & (trees["col"] >= c0) & (trees["col"] < c0 + win)
    z.plot(trees["col"][m] - c0 + 0.5, trees["row"][m] - r0 + 0.5, "r.", ms=3)
    z.set_xticks([]); z.set_yticks([]); z.set_xlim(0, win); z.set_ylim(0, win)
    _scalebar(z, (0, win, 0, win), 25.0)
    pl.label(8.95, 4.75 + W + 0.13, "Detail: crowns (outlined), tree tops (red)")
    s_, t_, f_, g_ = st["structure"], st["trees"], st["fragmentation"], st["gaps"]
    rows = [("Canopy", f"cover {s_['canopy_cover_ge2m'] * 100:.0f}% (2 m or higher), {s_['cover_ge5m'] * 100:.0f}% (5 m or higher); mean canopy height {s_['mean_canopy_height_m']:.1f} m, 95th percentile {s_['p95_m']:.0f} m."),
            ("Trees", f"{t_['n']:,} detected, {t_['per_ha']:.0f} per ha; mean height {t_['mean_height_m']:.1f} m; median crown {t_['median_crown_diameter_m']:.1f} m across. Counts of crowded trees are low (see the tests plate)."),
            ("Fragmentation", f"{f_['patches']:,} canopy patches; the largest holds {f_['largest_patch_index'] * 100:.0f}% of the area; effective mesh size {f_['effective_mesh_ha']:.1f} ha; {f_['edge_density_m_per_ha']:.0f} m of canopy edge per ha."),
            ("Openings", f"{g_['open_area_share'] * 100:.0f}% of the area is open; {g_['by_size']['10000-1e+09 m2']['area_share_of_openings'] * 100:.0f}% of that is in openings over 1 ha; farthest point from canopy {g_['max_distance_to_canopy_m']:.0f} m.")]
    y0 = 4.45
    for k_, v_ in rows:
        pl.text(8.95, y0, k_, fontsize=9, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v_, 86))
        pl.text(8.95, y0 - 0.24, w, fontsize=8.2, color=MUTED, linespacing=1.35, va="top")
        y0 -= 0.27 + 0.165 * (w.count("\n") + 1) + 0.14
    return pl.save(f"{num}_{key}_canopy.png")


def fig_tests(sts):
    """Accuracy and sensitivity tests for both sites."""
    names = {"houston": "NW Houston", "tahoe": "Tahoe forest"}
    cols = {"houston": "#B9770E", "tahoe": "#1F6B3A"}
    pl = Plate("How far can the canopy numbers be trusted?", "Terrain error, trees of known size planted and recovered, thinning of the returns, and model resolution",
               "Lidar: USGS 3DEP (Houston TX_Coastal_B1_2018, Tahoe CA_SierraNevada_5_2022). Imagery: NAIP. Synthetic stands: half-ellipsoid crowns scanned at each site's measured return density.",
               "Planted trees are idealised crowns, so real recall will differ")
    a = pl.ax(0.9, 5.15, 3.4, 2.7)
    for i, (k, st) in enumerate(sts.items()):
        h = st["terrain_holdout"]
        for j, (nm, key) in enumerate((("open ground", "open_ground"), ("under canopy 5 m+", "under_canopy_ge5m"))):
            a.bar(j * 3 + i, h[key]["rmse_m"] * 100, color=cols[k], label=names[k] if j == 0 else None)
            a.text(j * 3 + i, h[key]["rmse_m"] * 100 + 0.3, f"{h[key]['rmse_m'] * 100:.1f}", ha="center", fontsize=8)
    a.set_xticks([0.5, 3.5]); a.set_xticklabels(["open ground", "under canopy 5 m+"], fontsize=8)
    a.set_ylabel("terrain error, cm rms", fontsize=8, color=MUTED); a.tick_params(labelsize=8, length=2); a.spines[["top", "right"]].set_visible(False); a.set_ylim(0, 8.5); a.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)
    pl.label(0.9, 8.35, "Terrain error (10% held out)")
    b = pl.ax(5.15, 5.15, 4.9, 2.7)
    for k, st in sts.items():
        for kind, ls in (("regular", "-"), ("clumped", "--")):
            ds_ = [100, 200, 400]
            rec = [st["synthetic_stands"][f"{kind}_{d}"]["recall"] for d in ds_]
            td = [st["synthetic_stands"][f"{kind}_{d}"]["actual_trees_per_ha"] for d in ds_]
            b.plot(td, rec, ls, marker="o", ms=4, color=cols[k], label=f"{names[k]}, {kind} spacing")
    b.set_ylim(0, 1.02); b.set_xlabel("true trees per hectare", fontsize=8, color=MUTED); b.set_ylabel("share of true trees found", fontsize=8, color=MUTED); b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False); b.legend(fontsize=7, frameon=False, loc="lower left")
    pl.label(5.15, 8.35, "Planted trees found, by density")
    c = pl.ax(10.9, 5.15, 4.4, 2.7)
    for i, (k, st) in enumerate(sts.items()):
        t = st["synthetic_stands"]["regular_200"]["recall_by_height_tercile"]
        c.bar(np.arange(3) + i * 0.38 - 0.19, t, 0.36, color=cols[k], label=names[k])
    c.set_xticks(range(3)); c.set_xticklabels(["shortest third", "middle third", "tallest third"], fontsize=8); c.set_ylim(0, 1.02); c.set_ylabel("share found (regular, 200 per ha)", fontsize=8, color=MUTED); c.tick_params(labelsize=8, length=2); c.spines[["top", "right"]].set_visible(False)
    pl.label(10.9, 8.35, "Which planted trees are missed")
    d = pl.ax(0.9, 1.35, 4.4, 2.4)
    for i, (k, st) in enumerate(sts.items()):
        th = st["thinning"]
        d.bar(np.arange(2) + i * 0.38 - 0.19, [th[f]["share_of_full_trees_found"] for f in ("0.5", "0.25")], 0.36, color=cols[k], label=names[k])
        for j, f in enumerate(("0.5", "0.25")):
            d.text(j + i * 0.38 - 0.19, th[f]["share_of_full_trees_found"] + 0.02, f"{th[f]['height_rmse_m']:.1f} m", ha="center", fontsize=7)
    d.set_xticks(range(2)); d.set_xticklabels(["half the returns", "quarter of the returns"], fontsize=8); d.set_ylim(0, 1.25); d.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); d.set_ylabel("share of full-data trees found", fontsize=8, color=MUTED); d.tick_params(labelsize=8, length=2); d.spines[["top", "right"]].set_visible(False); d.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)
    pl.label(0.9, 4.1, "Thinning the returns (labels: height change)")
    e = pl.ax(6.2, 1.35, 4.4, 2.4)
    for k, st in sts.items():
        rs = st["resolution_sensitivity"]
        xs = [0.5, 1.0, 2.0]
        e.plot(xs, [rs[str(x)]["cover"] * 100 for x in xs], "-o", color=cols[k], label=f"{names[k]}: cover %")
        e.plot(xs, [rs[str(x)]["n"] / rs["1.0"]["n"] * 50 for x in xs], ":s", color=cols[k], label=f"{names[k]}: trees (50 = 1 m count)")
    e.set_xscale("log", base=2); e.set_xticks([0.5, 1, 2]); e.set_xticklabels(["0.5", "1", "2"]); e.set_xlabel("canopy model cell size, m", fontsize=8, color=MUTED); e.tick_params(labelsize=8, length=2); e.spines[["top", "right"]].set_visible(False); e.legend(fontsize=6.6, frameon=False)
    pl.label(6.2, 4.1, "Model resolution changes the answer")
    t = []
    for k, st in sts.items():
        n_ = st["naip_agreement"]
        t.append(f"{names[k]}: {n_['lidar_canopy_that_naip_calls_vegetation'] * 100:.0f}% of lidar canopy pixels are vegetation in NAIP {n_['naip_date'][:4]} (NAIP also counts lawn and grass, so the reverse cannot be tested).")
    pl.text(11.0, 3.95, "\n".join(textwrap.wrap(" ".join(t), 62)), fontsize=8.2, color=MUTED, linespacing=1.4, va="top")
    notes = ("Houston lidar was flown in January: deciduous trees were leafless, so their crowns are traced by branches and canopy cover is probably lower than in leaf. "
             "The Tahoe lidar has no vegetation classes: vegetation is any return at least 0.5 m above the terrain, so shrubs and snags count. "
             "Both sites: the tree counts are lower bounds in crowded stands, and cover and counts depend on the model cell size.")
    pl.text(11.0, 2.55, "\n".join(textwrap.wrap(notes, 62)), fontsize=8.2, color=MUTED, linespacing=1.4, va="top")
    return pl.save("36_canopy_accuracy_tests.png")


NAMES = {"houston": "NW Houston", "tahoe": "Tahoe forest"}


def _cluster_image(g, cl, ca, min_h_mask):
    """RGB image of canopy coloured by cluster: largest cluster dark green, other clusters of 2 or more patches in a cycling palette, single isolated patches orange, canopy in patches under the node size grey."""
    import matplotlib.cm as cm
    lab = g["labels"]
    n = lab.shape[0]
    img = np.ones((n, n, 3)) * np.array([0.97, 0.96, 0.93])
    img[min_h_mask] = (0.80, 0.80, 0.80)
    sizes = np.bincount(cl)
    order = np.argsort(-ca)
    pal = cm.tab20(np.linspace(0, 1, 20))[:, :3]
    col = np.zeros((len(ca), 3))
    for rank, c in enumerate(order):
        if rank == 0:
            col[c] = (0.09, 0.37, 0.13)
        elif sizes[c] == 1:
            col[c] = (0.96, 0.60, 0.15)
        else:
            col[c] = pal[(rank * 7) % 20]
    full = np.concatenate([[[0, 0, 0]], col[cl]])
    m = lab > 0
    img[m] = full[lab[m]]
    return img


def _plain_log(ax, axis="both"):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def fig_conn_maps(sts, Fz):
    pl = Plate("Canopy connectivity maps", "Which canopy is linked to which: clusters of patches whose crowns lie within 3 m of each other, and how much canopy each patch can reach through stepping stones",
               "Lidar: USGS 3DEP (Houston TX_Coastal_B1_2018, Tahoe CA_SierraNevada_5_2022). Nodes: canopy patches (2 m or higher) of at least 20 m2; gaps: closest distance between patch edges.",
               "A 3 m gap and a 5 m reach scale are scenarios, not a species' ability")
    W = 3.35
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm
    for r, key in enumerate(("houston", "tahoe")):
        st, F = sts[key], Fz[key]
        g, chm = F["graph"], F["chm"]
        n = chm.shape[0]
        ext = (-n / 2, n / 2, -n / 2, n / 2)
        y = 4.75 if r == 0 else 0.95
        cl, ca = F["cl_map"], F["ca_map"]
        pl.label(0.6, y + W + 0.13, f"{NAMES[key]}: clusters across 3 m gaps")
        a = pl.ax(0.6, y, W, W)
        a.imshow(_cluster_image(g, cl, ca, (chm >= 2) & (g["labels"] == 0)), origin="lower", extent=ext)
        a.set_xticks([]); a.set_yticks([])
        _scalebar(a, ext, 200.0 if n > 1000 else 100.0)
        pl.label(4.95, y + W + 0.13, f"{NAMES[key]}: canopy each patch reaches")
        b = pl.ax(4.95, y, W, W)
        reach = F["reach_map"] / 1e4
        vmin, vmax = max(reach.min(), 1e-3), reach.max()
        norm = LogNorm(vmin, vmax)
        img = np.ones((n, n, 3)) * np.array([0.97, 0.96, 0.93])
        img[(chm >= 2) & (g["labels"] == 0)] = (0.80, 0.80, 0.80)
        rgb = cm.viridis(norm(np.clip(reach, vmin, vmax)))[:, :3]
        m = g["labels"] > 0
        img[m] = np.concatenate([[[0, 0, 0]], rgb])[g["labels"][m]]
        b.imshow(img, origin="lower", extent=ext)
        st_idx = np.flatnonzero(F["stones"])
        if len(st_idx):
            cen = g["centroid"][st_idx]
            b.plot(cen[:, 1] + 0.5 - n / 2, cen[:, 0] + 0.5 - n / 2, "o", ms=3.2, mfc="#E74C3C", mec="white", mew=0.5)
        b.set_xticks([]); b.set_yticks([]); b.set_xlim(ext[0], ext[1]); b.set_ylim(ext[2], ext[3])
        cb = pl.ax(8.37, y, 0.1, W)
        cb.imshow(np.linspace(1, 0, 200)[:, None], aspect="auto", cmap="viridis", extent=(0, 1, np.log10(vmin), np.log10(vmax)))
        cb.set_xticks([]); cb.yaxis.tick_right(); cb.yaxis.set_label_position("right"); cb.tick_params(labelsize=7, length=2)
        ticks = np.arange(np.ceil(np.log10(vmin)), np.floor(np.log10(vmax)) + 1)
        cb.set_yticks(ticks); cb.set_yticklabels([f"{10 ** t:g}" for t in ticks])
        cb.set_ylabel("reachable canopy, ha (log scale)", fontsize=7.5, color=MUTED, labelpad=4)
        cn = st["connectivity"]
        g3 = cn["by_gap"]["3"]
        pc5 = cn["pc"]["5.0"]
        rows = [("Clusters", f"{cn['nodes']:,} patches; at a 3 m gap they form {g3['clusters']:,} clusters. The largest (dark green) holds {g3['largest_cluster_share'] * 100:.0f}% of the patch canopy ({g3['largest_cluster_ha']:.1f} ha); {g3['isolated_nodes']:,} patches (orange) have no neighbour within 3 m. Grey: canopy in patches under 20 m2."),
                ("Reach", f"Each patch's reach is the canopy it can get to, counting a patch at weight exp(-gap / 5 m) along the best chain of stepping stones. Reachable canopy ('equivalent connected area') is {pc5['eca_ha']:.1f} ha of {pc5['canopy_ha']:.1f} ha of patch canopy ({pc5['eca_share_of_canopy'] * 100:.0f}%)."),
                ("Stepping stones", f"{cn['stepping_stones']:,} patches (red dots) are cut points: removing one splits its cluster at 3 m. Patches cut by the window edge continue outside it, so clusters and reach near the edge are understated.")]
        y0 = y + W
        for k_, v_ in rows:
            pl.text(9.0, y0, k_, fontsize=9, fontweight="bold", va="top")
            w = "\n".join(textwrap.wrap(v_, 80))
            pl.text(9.0, y0 - 0.24, w, fontsize=8.1, color=MUTED, linespacing=1.35, va="top")
            y0 -= 0.27 + 0.16 * (w.count("\n") + 1) + 0.12
    return pl.save("37_canopy_connectivity_maps.png")


def _curves(g, gaps):
    from . import connectivity as CN
    lcc, ncl = [], []
    for T in gaps:
        lab, ca = CN.clusters(g, T)
        lcc.append(ca.max() / g["area"].sum())
        ncl.append(len(ca))
    return np.array(lcc), np.array(ncl)


def _abstract_graph(ax, g, cl, ca, gap=5.0, min_area=200.0, seed=3):
    """Network drawing: nodes = patches of at least `min_area` m2, links = gaps up to `gap` m, positions from a spring layout (not geography), size by area, colour by cluster."""
    import networkx as nx
    keep = np.flatnonzero(g["area"] >= min_area)
    idx = {int(v): i for i, v in enumerate(keep)}
    G = nx.Graph()
    G.add_nodes_from(range(len(keep)))
    for a, b, d in g["edges"]:
        if d <= gap and int(a) in idx and int(b) in idx:
            G.add_edge(idx[int(a)], idx[int(b)], weight=1.0 / (1.0 + d))
    pos = nx.spring_layout(G, seed=seed, weight="weight", k=0.35, iterations=120)
    lcc = int(np.argmax(ca))
    col = ["#1B5E20" if cl[v] == lcc else "#E59866" for v in keep]
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.4, edge_color="#9AA79A", alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=np.sqrt(g["area"][keep]) * 0.7, node_color=col, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#999999")
    return len(keep), G.number_of_edges()


def fig_conn_graph(sts, Fz):
    from matplotlib.collections import LineCollection
    cols = {"houston": "#B9770E", "tahoe": "#1F6B3A"}
    pl = Plate("Canopy connectivity graph", "The patches and the gaps between them as a network, how connectivity falls as the gap that can be crossed shrinks, and how stable the numbers are",
               "Lidar: USGS 3DEP (Houston TX_Coastal_B1_2018, Tahoe CA_SierraNevada_5_2022). Node = canopy patch of 20 m2 or more (CHM 2 m or higher); link = gap between patch edges.", "Network layouts are not geography")
    W = 3.1
    for r, key in enumerate(("houston", "tahoe")):
        F = Fz[key]
        g, chm = F["graph"], F["chm"]
        n = chm.shape[0]
        y = 4.75 if r == 0 else 0.95
        pl.label(0.6, y + W + 0.13, f"{NAMES[key]}: links across gaps")
        a = pl.ax(0.6, y, W, W)
        a.imshow(np.where(g["labels"] > 0, 0.88, 1.0), origin="lower", extent=(-n / 2, n / 2, -n / 2, n / 2), cmap="gray", vmin=0.5, vmax=1.0)
        e, exy = g["edges"], g["edge_xy"]
        for lo, hi, col, lw in ((10, 25, "#BDBDBD", 0.5), (5, 10, "#E6A23C", 0.7), (3, 5, "#D35400", 0.8), (-1, 3, "#7B241C", 0.9)):
            sel = (e[:, 2] > lo) & (e[:, 2] <= hi)
            seg = np.stack([np.c_[exy[sel, 1], exy[sel, 0]], np.c_[exy[sel, 3], exy[sel, 2]]], 1) + 0.5 - n / 2
            a.add_collection(LineCollection(seg, colors=col, linewidths=lw, zorder=3))
        a.set_xlim(-n / 2, n / 2); a.set_ylim(-n / 2, n / 2); a.set_xticks([]); a.set_yticks([])
        b = pl.ax(4.0, y, W, W)
        nn, ne = _abstract_graph(b, g, F["cl_map"], F["ca_map"])
        pl.label(4.0, y + W + 0.13, f"{NAMES[key]}: network, {nn} patches, {ne} links")
    lg = [Line2D([], [], color="#7B241C", lw=2, label="gap under 3 m"), Line2D([], [], color="#D35400", lw=2, label="3 to 5 m"), Line2D([], [], color="#E6A23C", lw=2, label="5 to 10 m"), Line2D([], [], color="#BDBDBD", lw=2, label="10 to 25 m"),
          Line2D([], [], marker="o", ls="none", color="#1B5E20", label="in the largest cluster (3 m)"), Line2D([], [], marker="o", ls="none", color="#E59866", label="in another cluster")]
    lax = pl.ax(0.6, 4.36, 6.5, 0.28)
    lax.axis("off")
    lax.legend(handles=lg, loc="center left", ncol=3, fontsize=7.4, frameon=False, columnspacing=1.2, handletextpad=0.5)
    gaps = np.arange(0, 41, 1.0)
    a = pl.ax(8.0, 5.1, 3.3, 2.9)
    b = pl.ax(8.0, 1.3, 3.3, 2.9)
    for key in ("houston", "tahoe"):
        lcc, ncl = _curves(Fz[key]["graph"], gaps)
        a.plot(gaps, lcc * 100, color=cols[key], label=NAMES[key], lw=1.8)
        b.plot(gaps, ncl, color=cols[key], label=NAMES[key], lw=1.8)
    for ax_ in (a, b):
        ax_.axvline(3, color="#999999", lw=0.8, ls=":")
        ax_.set_xlabel("gap that can be crossed, m", fontsize=8, color=MUTED)
        ax_.tick_params(labelsize=8, length=2); ax_.spines[["top", "right"]].set_visible(False)
        ax_.legend(fontsize=7.8, frameon=False, loc="lower right" if ax_ is a else "upper right")
    a.set_ylim(0, 100); a.set_ylabel("canopy in the largest cluster, %", fontsize=8, color=MUTED)
    b.set_yscale("log"); _plain_log(b, "y"); b.set_ylabel("number of clusters (log scale)", fontsize=8, color=MUTED)
    pl.label(8.0, 8.35, "Largest cluster by gap")
    pl.label(8.0, 4.55, "Number of clusters by gap")
    c = pl.ax(11.9, 5.1, 3.4, 2.9)
    for key in ("houston", "tahoe"):
        g = Fz[key]["graph"]
        c.scatter(g["area"], Fz[key]["reach_map"], s=3, color=cols[key], alpha=0.5, lw=0, label=NAMES[key])
    c.set_xscale("log"); c.set_yscale("log"); _plain_log(c)
    c.set_xlabel("patch area, m2", fontsize=8, color=MUTED); c.set_ylabel("canopy it reaches, m2 (5 m scale)", fontsize=8, color=MUTED)
    c.tick_params(labelsize=8, length=2); c.spines[["top", "right"]].set_visible(False); c.legend(fontsize=7.8, frameon=False, markerscale=3, loc="lower right")
    pl.label(11.9, 8.35, "Small patches reach little")
    d = pl.ax(12.4, 1.3, 2.9, 2.9)
    keys = ["baseline_1m_h2_a20", "cell_0.5m", "cell_2m", "canopy_5m_or_higher", "min_patch_10m2", "min_patch_50m2", "half_returns", "quarter_returns"]
    lab_ = ["baseline", "0.5 m cells", "2 m cells", "canopy 5 m+", "patches 10 m2+", "patches 50 m2+", "half the returns", "quarter returns"]
    yy = np.arange(len(keys))
    for i, key in enumerate(("houston", "tahoe")):
        sv = [sts[key]["connectivity"]["sensitivity"][k]["eca_share_of_canopy"] * 100 for k in keys]
        d.barh(yy + (i - 0.5) * 0.38, sv, 0.36, color=cols[key], label=NAMES[key])
    d.set_yticks(yy); d.set_yticklabels(lab_, fontsize=7.5); d.invert_yaxis(); d.set_xlim(0, 100)
    d.set_xlabel("reachable share of patch canopy, %", fontsize=8, color=MUTED); d.tick_params(labelsize=8, length=2); d.spines[["top", "right"]].set_visible(False)
    d.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.01), ncol=2)
    pl.label(11.9, 4.55, "How much the answer moves")
    return pl.save("38_canopy_connectivity_graph.png")
