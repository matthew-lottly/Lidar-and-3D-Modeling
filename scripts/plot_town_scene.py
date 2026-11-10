"""Plates 75 to 82 for the White River scene: classified and realistic 3D maps, facilities, classification, trees, buildings, rendering methods and wire clean-up.
Usage: xvfb-run -a python scripts/plot_town_scene.py [75|76|77|78|79|80|81|82|all]"""
import json
import pickle
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import build_town_scene as BS
from lidar3d import render_pv as R, scene3d as SC, unified as U
from lidar3d.plates import MUTED, Plate

PROC = ROOT / "data" / "processed" / "town"
OUTD = ROOT / "outputs"


def rows(pl, x, y0, items, width, fs=8.2, head=9.5):
    for k_, v_ in items:
        pl.text(x, y0, k_, fontsize=head, fontweight="bold", va="top")
        w = "\n".join(textwrap.wrap(v_, width))
        pl.text(x, y0 - 0.26, w, fontsize=fs, color=MUTED, linespacing=1.4, va="top")
        y0 -= 0.3 + 0.17 * (w.count("\n") + 1) + 0.28
    return y0


def panel(pl, im, l, b, w, h, title=None, tfs=None):
    a = pl.ax(l, b, w, h)
    a.imshow(im); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color("#999999")
    if title:
        pl.label(l, b + h + 0.12, title)
    return a


class Scene:
    def __init__(self):
        self.S = BS.prepare(); self.D = BS.detect(self.S); self.N = BS.network(self.S, self.D)
        self.gz = float(np.nanmean(self.S["G"]["z"]))
        self.yc = self.N["Y"]["centre"]
        p = BS.CACHE_D / "scene_layers.pkl"
        self.L = {}
        self.info = {}
        for mode in ("realistic", "classified"):
            L, info = BS.assemble(self.S, self.D, self.N, mode)
            self.L[mode] = [dict(mesh=m, kind=("points" if (k == "frames" or (hasattr(m, "n_faces") and m.n_faces == 0 and m.n_points > 0)) else "mesh"), point_size=3.0) for k, m in L.items() if m is not None]
            self.info[mode] = info
        # first transformer template block
        boxy = [b for b in self.N["Y"]["blocks"] if b.get("fill", 1.0) >= 0.22]
        _, kinds = SC.equipment_templates(boxy)
        self.tf_block = next((b for b, k in zip(boxy, kinds) if k == "transformer"), max(boxy, key=lambda b: b["n"]))
        self.views = dict(
            overview=((-125, -135, self.gz + 5), 800, 200, 36),
            yard=((self.yc[0], self.yc[1], self.gz + 6), 230, 215, 28),
            corridor=((-250, 10, self.gz + 12), 210, 330, 20),
            houses=((30, -260, self.gz + 2), 200, 160, 28),
            transformer=((self.tf_block["centre"][0], self.tf_block["centre"][1], self.tf_block["centre"][2] + 1.5), 40, 200, 18),
            trees=((-300, 90, self.gz + 14), 110, 160, 18),
        )

    def render(self, mode, view, size, **kw):
        f, dist, az, el = self.views[view]
        kw.setdefault("ssao", True); kw.setdefault("shadows", False)
        if mode == "classified":
            kw.setdefault("bg", "white")
        return R.render(self.L[mode], f, dist, az, el, size, **kw)


def src(S):
    return (f"Lidar: USGS 3DEP WA_PierceCounty_1_2020, collected 2020-04-13 (about 25 returns per m2); no building, vegetation or wire classes are present, so every class here is derived from the points. Imagery: NAIP {S['naip_date']}. "
            "Trees, supports, wires and equipment are fitted or templated geometry: sizes and positions are measured, shapes are assumed.")


def plate_big(sc, mode, name, title, sub, items, note):
    pl = Plate(title, sub, src(sc.S), note)
    big = sc.render(mode, "overview", (1920, 1350))
    panel(pl, big, 0.6, 1.0, 9.6, 6.75, "White River substation and its surroundings, looking north-north-east" if mode == "realistic" else "Classified 3D map: no imagery, colour by class")
    ys = [6.15, 4.15, 2.15]
    for (view, ttl), y in zip((("yard", "The yard"), ("corridor", "Lines and towers"), ("houses", "Houses")), ys):
        panel(pl, sc.render(mode, view, (1000, 380)), 10.6, y, 4.9, 1.85, ttl)
    rows(pl, 10.6, 0.98 - 0.0, [], 40)
    return pl, items


def run_75(sc):
    I = sc.info["realistic"]
    pl, _ = plate_big(sc, "realistic", "75_whiteriver_scene_realistic_3d.png", "The utility model in its lidar and imagery scene", "Ground draped with the aerial photo; trees, buildings, towers, poles, wires and substation equipment as 3D models fitted to the lidar",
                      [], "Trees and roofs take their colour from the photo after correcting its relief displacement")
    pl.text(10.6, 1.9, f"{I['trees']:,} tree crowns, {I['buildings']} buildings, {sum(I['equipment_kinds'].values())} equipment blocks as templates, {I['clean']['after']} cleaned wire polylines.", fontsize=8.2, color=MUTED, va="top")
    pl.save("75_whiteriver_scene_realistic_3d.png")


def run_76(sc):
    pl, _ = plate_big(sc, "classified", "76_whiteriver_scene_classified_3d.png", "The utility model as a classified 3D map", "The same scene with no imagery: every class in its own colour so the features read at a glance",
                      [], "Ground: green-tinted where vegetated, grey where bare or paved (from NAIP NDVI); yard gravel sand")
    # legend swatches
    leg = [("trees (darker = taller)", "#4C9A60"), ("buildings (roofs)", "#DEA862"), ("wood poles", "#8C5C30"), ("steel towers, yard structures", "#46607F"), ("substation equipment", "#3C6EA0"), ("wires", "#141414"), ("ground: vegetated", "#C4D2B4"), ("ground: bare or paved", "#D6D6D2"), ("substation yard", "#C8C4BA")]
    a = pl.ax(10.6, 0.95, 4.9, 1.0); a.axis("off")
    for i, (t, c) in enumerate(leg):
        col, row = divmod(i, 3)
        x, y = col * 0.34, 0.92 - row * 0.32
        a.add_patch(plt.Rectangle((x, y - 0.14), 0.04, 0.2, fc=c, ec="#888", lw=0.4)); a.text(x + 0.055, y - 0.04, t, fontsize=6.8, va="center")
    a.set_xlim(0, 1); a.set_ylim(0, 1.05)
    pl.save("76_whiteriver_scene_classified_3d.png")


def _sample_ndvi(sc, x, y, h=0.0):
    fit = sc.info["realistic"]["displacement"]
    _, nd = BS.sample_naip(sc.S, np.asarray(x) + h * (fit["k"] * np.asarray(x) + fit["bx"]), np.asarray(y) + h * (fit["k"] * np.asarray(y) + fit["by"]))
    return nd


def run_77(sc):
    I = sc.info["realistic"]
    pl = Plate("Facilities: the substation yard and a transformer", "Equipment templates shaped to measured envelopes, open frames as lidar returns, gantries fitted to their returns, bus wires and fence",
               src(sc.S), "Left: on the aerial photo. Right: classified colours")
    panel(pl, sc.render("realistic", "yard", (1800, 900)), 0.6, 4.65, 7.2, 3.55, "Yard, on the aerial photo")
    panel(pl, sc.render("classified", "yard", (1800, 900)), 8.2, 4.65, 7.2, 3.55, "Yard, classified colours")
    panel(pl, sc.render("realistic", "transformer", (1800, 880)), 0.6, 1.0, 7.2, 3.4, "A transformer bay, on the aerial photo")
    k = I["equipment_kinds"]
    rows(pl, 8.3, 4.0, [
        ("What is drawn", f"{I['boxy_blocks']} box-like equipment blocks as templates ({k.get('transformer', 0)} transformer, {k.get('column apparatus', 0)} column apparatus, {k.get('cabinet or small building', 0)} cabinets or small buildings) and {I['open_frames']} open-frame blocks (under 22% of their envelope filled) shown as the lidar returns themselves, because a box would claim a solid that is not there. Gantries and towers are lattices fitted to their own returns; the yard fence is drawn along the mapped yard outline."),
        ("Measured and assumed", "Envelope size and position come from the lidar. The template follows from size alone (low and large: transformer; tall and narrow: porcelain column; else cabinet), so it can be wrong. Tank, radiators, conservator and bushings are assumed shapes."),
        ("Not claimed", "No equipment type or rating is read from the lidar, and no drawing was used. Where the lidar saw no wire between two gantries, none is drawn.")], 92, fs=8.0)
    pl.save("77_whiteriver_scene_facilities_3d.png")


def run_78(sc):
    S, D, N = sc.S, sc.D, sc.N
    d = S["d"]
    win = BS.WIN
    x0, x1, y0, y1 = win
    pl = Plate("Classifying the scene", "Ground cover, tree crowns, buildings, the substation yard, wires and supports found from the points and the photo, with the checks each class can be tested by", src(S), "Local grid, metres from the substation centre (UTM 10N)")
    a = pl.ax(0.6, 0.95, 8.4, 6.85)
    ny, nx = S["G"]["z"].shape
    X = np.arange(nx) * 0.5 + x0 + 0.25; Y = np.arange(ny) * 0.5 + y0 + 0.25
    rgb, nd = BS.sample_naip(S, *np.meshgrid(X[::2], Y[::2]))
    ground = np.where((nd >= 0.25)[..., None], np.array([0.80, 0.86, 0.72]), np.array([0.86, 0.86, 0.84]))
    a.imshow(ground, origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest", zorder=1)
    tl = D["tree_lab"]
    T = D["trees"]
    hmap = np.full(tl.shape, np.nan)
    hmap[tl > 0] = np.interp(tl[tl > 0], T["id"], T["height"], left=np.nan, right=np.nan) if len(T["id"]) else np.nan
    a.imshow(np.ma.masked_invalid(hmap), origin="lower", extent=(x0, x1, y0, y1), cmap="Greens", vmin=0, vmax=35, alpha=0.9, interpolation="nearest", zorder=2)
    from matplotlib.patches import Polygon as MP
    ccx, ccy = D["bld_offset"]
    for b in D["buildings"]:
        q = np.array(b.footprint.exterior.coords) + [ccx, ccy]
        a.add_patch(MP(q, closed=True, fc="#DEA862", ec="#8a5a20", lw=0.5, zorder=4))
    yx, yy = N["Y"]["yard"].exterior.xy
    a.fill(yx, yy, fc="#C8C4BA", ec="#7a7568", lw=0.8, zorder=3)
    for w in N["K"]["span_w"] + [x["xyz"] for x in N["K"]["adds"]] + [x["xyz"] for x in N["br"]]:
        a.plot(w[:, 0], w[:, 1], color="#141414", lw=0.6, zorder=5)
    for w in N["Y"]["wires"]:
        a.plot(w["xyz"][:, 0], w["xyz"][:, 1], color="#141414", lw=0.5, zorder=5)
    sup = N["K"]["sup"][N["K"]["sup"].tier == "support"]
    a.scatter(sup.x, sup.y, s=14, marker="s", color="#46607F", zorder=6, edgecolors="white", linewidths=0.3)
    ts = N["Y"]["towers"]
    a.scatter([t["x"] for t in ts], [t["y"] for t in ts], s=10, marker="^", color="#46607F", zorder=6)
    for b in N["Y"]["blocks"]:
        a.plot(*b["points"][:, :2].T, ".", ms=0.5, color="#3C6EA0", zorder=4)
    a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_aspect("equal"); a.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(fc="#80B070", label="tree crowns (darker = taller)"), Patch(fc="#DEA862", label="buildings"), Patch(fc="#C8C4BA", label="substation yard"), Patch(fc="#CCDBB8", label="vegetated ground"), Patch(fc="#DCDCD6", label="bare or paved ground"),
                       Line2D([], [], color="#141414", lw=1, label="wires"), Line2D([], [], marker="s", ls="", color="#46607F", label="supports"), Line2D([], [], marker=".", ls="", color="#3C6EA0", label="yard equipment returns")], loc="lower right", fontsize=7, framealpha=0.92)
    pl.label(0.6, 8.0, "Classes in plan")
    # numbers
    ntree = len(T["x"])
    crown_cells = tl > 0
    area_tree = crown_cells.sum() * 0.25
    area_b = sum(b.footprint.area for b in D["buildings"])
    # tree check: NDVI under crown cells (cell centres, shifted for the crown's height)
    jj, ii = np.nonzero(crown_cells[::3, ::3])
    px, py = X[::3][ii], Y[::3][jj]
    hh = np.nan_to_num(hmap[::3, ::3][jj, ii], nan=10.0) * 0.66
    ndc = _sample_ndvi(sc, px, py, hh)
    ndc0 = _sample_ndvi(sc, px, py, 0.0)
    chm = D["chm"]
    tall = np.isfinite(chm) & (chm >= 5.0)
    complete = float((tl[tall] > 0).mean())
    bc = D["building_check"]
    fit = sc.info["realistic"]["displacement"]
    rows(pl, 9.4, 7.95, [
        ("Trees", f"{ntree:,} crowns ({ntree / ((x1 - x0) * (y1 - y0) / 1e4):.0f} per ha), median {np.median(T['height']):.0f} m tall, {area_tree / 1e4:.1f} ha of crown. Check against the photo: {np.mean(ndc >= 0.3) * 100:.0f}% of crown cells are vegetated (NDVI 0.3 or more) at the position the relief displacement predicts and {np.mean(ndc0 >= 0.3) * 100:.0f}% without it, so the correction changes nothing measurable here. Completeness: {complete * 100:.0f}% of cells with a return 5 m or more above ground (not building, wire or yard) lie inside a crown."),
        ("Buildings", f"{bc['kept']['buildings']} buildings kept out of {sum([1]) * D['buildings_all']} planar footprints; the other {D['buildings_all'] - bc['kept']['buildings']} were crowns or towers (roof NDVI of 0.25 or more, or too tall for their size, or cut by the window edge). Against Overture's {bc['overture_buildings']} footprints: {bc['kept']['share_of_lidar_area_inside_overture'] * 100:.0f}% of the kept footprint area lies inside a mapped building (42% before filtering), and the kept set covers {bc['kept']['share_of_overture_area_covered'] * 100:.0f}% of the mapped area."),
        ("Image registration", f"The aerial photo is displaced by height. Fitting it from roof edges (stable across four sub-windows) gives about {fit['shift_at_10m_height_m']:.1f} m of shift per 10 m of height, and roof outlines sit on stronger image edges after the shift ({fit['edge_before']:.1f} to {fit['edge_after']:.1f}). A fit from canopy colour was tried first and rejected: its answer changed a great deal between window halves (plate 81)."),
        ("Not classified", "Roads, cars, water and low shrubs are not separate classes; ground cover is only vegetated or not (NDVI 0.25). Class areas are from 0.5 m cells.")], 66, fs=7.9)
    pl.save("78_whiteriver_scene_classification.png")


def run_79(sc):
    D, S = sc.D, sc.S
    T = D["trees"]
    pl = Plate("Trees: individual crowns from the canopy height model", "Watershed segmentation of the canopy height model into crowns, each with a top, height, radius and crown base, drawn as a crown and trunk", src(S), "Crown shape (cone or ellipsoid) is decided by crown length against width")
    win = (-420, -150, 0, 270)
    a = pl.ax(0.6, 4.2, 4.2, 3.9)
    ch = D["chm"]
    x0, y0 = BS.WIN[0], BS.WIN[2]
    sl = (slice(int((win[2] - y0) / 0.5), int((win[3] - y0) / 0.5)), slice(int((win[0] - x0) / 0.5), int((win[1] - x0) / 0.5)))
    a.imshow(ch[sl], origin="lower", extent=(win[0], win[1], win[2], win[3]), cmap="Greens", vmin=0, vmax=40, interpolation="nearest")
    m = (T["x"] >= win[0]) & (T["x"] <= win[1]) & (T["y"] >= win[2]) & (T["y"] <= win[3])
    a.scatter(T["x"][m], T["y"][m], s=4, color="#B03A2E", zorder=3)
    from matplotlib.patches import Circle
    for xx, yy, rr in zip(T["x"][m], T["y"][m], T["radius"][m]):
        a.add_patch(Circle((xx, yy), rr, fill=False, ec="#222", lw=0.3))
    a.set_aspect("equal"); a.tick_params(labelsize=7)
    pl.label(0.6, 8.35, "Canopy height model, tree tops and crowns")
    conifer = ((T["height"] - np.minimum(T["base"], 0.6 * T["height"])) / np.maximum(2 * T["radius"] * 1.25, 0.5)) > 1.6
    b = pl.ax(5.7, 4.55, 4.2, 3.55)
    bins = np.arange(3, 50, 2)
    b.hist([T["height"][~conifer], T["height"][conifer]], bins=bins, stacked=True, color=["#8FBF6A", "#2F6B3F"], label=["broadleaf-shaped", "conifer-shaped"])
    b.set_xlabel("tree height, m", fontsize=8); b.set_ylabel("trees", fontsize=8); b.legend(fontsize=7.5, frameon=False); b.tick_params(labelsize=8); b.spines[["top", "right"]].set_visible(False)
    pl.label(5.7, 8.35, "Height")
    c = pl.ax(10.9, 4.55, 4.3, 3.55)
    c.scatter(T["height"], T["radius"], s=3, alpha=0.35, color="#2F6B3F", linewidths=0)
    c.set_xlabel("tree height, m", fontsize=8); c.set_ylabel("crown radius, m", fontsize=8); c.tick_params(labelsize=8); c.spines[["top", "right"]].set_visible(False)
    pl.label(10.9, 8.35, "Crown radius against height")
    panel(pl, sc.render("realistic", "trees", (1500, 700)), 0.6, 0.95, 7.2, 2.3, "Crowns on the photo colours")
    panel(pl, sc.render("classified", "trees", (1500, 700)), 8.2, 0.95, 7.2, 2.3, "Crowns, classified colours (darker = taller)")
    pl.text(5.7, 4.0, "\n".join(textwrap.wrap(f"{len(T['x']):,} crowns; {conifer.mean() * 100:.0f}% conifer-shaped; median height {np.median(T['height']):.0f} m, tallest {T['height'].max():.0f} m; median radius {np.median(T['radius']):.1f} m. Crown base from the 12th percentile of returns inside the crown. No field data: heights and counts are not checked against plots; the photo check and completeness are on plate 78.", 95)), fontsize=8, color=MUTED, va="top")
    pl.save("79_whiteriver_scene_trees.png")


def run_80(sc):
    D, S, N = sc.D, sc.S, sc.N
    bc = D["building_check"]
    pl = Plate("Buildings: roofs from the lidar, checked against the map and the photo", "Planar returns become footprints and roof planes (RANSAC); crowns that look planar are removed with the photo's NDVI; the photo is registered to the roofs", src(S), "Local grid, metres from the substation centre (UTM 10N)")
    a = pl.ax(0.6, 3.5, 6.2, 4.7)
    from matplotlib.patches import Polygon as MP
    ccx, ccy = D["bld_offset"]
    cm = plt.get_cmap("YlOrBr")
    for b in D["buildings"]:
        q = np.array(b.footprint.exterior.coords) + [ccx, ccy]
        a.add_patch(MP(q, closed=True, fc=cm(min(b.height / 16, 1)), ec="#555", lw=0.4))
    pl.label(0.6, 8.35, "Kept buildings, coloured by height (0 to 16 m)")
    import pyarrow.dataset as pads
    from lidar3d import sitemap
    from pyproj import Transformer
    from shapely import wkb as W_
    d = S["d"]
    trf = Transformer.from_crs(4326, BS.BT.CRS, always_xy=True); bk = Transformer.from_crs(BS.BT.CRS, 4326, always_xy=True)
    Wl, Sl = bk.transform(d["cx"] + BS.WIN[0], d["cy"] + BS.WIN[2]); El, Nl = bk.transform(d["cx"] + BS.WIN[1], d["cy"] + BS.WIN[3])
    tb = sitemap._table("buildings", "building", ["geometry"], (Wl, Sl, El, Nl))
    for g in tb.geometry:
        q = np.array([(trf.transform(x, y)[0] - d["cx"], trf.transform(x, y)[1] - d["cy"]) for x, y in W_.loads(g).exterior.coords])
        a.plot(q[:, 0], q[:, 1], color="#2E6F9E", lw=0.5)
    a.set_xlim(BS.WIN[0], BS.WIN[1]); a.set_ylim(BS.WIN[2], BS.WIN[3]); a.set_aspect("equal"); a.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], color="#2E6F9E", lw=1, label="Overture building footprint"), Line2D([], [], color="#555", lw=1, label="building found in the lidar")], loc="lower right", fontsize=7.5)
    b = pl.ax(7.5, 4.45, 3.4, 3.5)
    allv = [bc["all_planar_footprints"]["share_of_lidar_area_inside_overture"] * 100, bc["kept"]["share_of_lidar_area_inside_overture"] * 100]
    b.bar([0, 1], allv, color=["#BDC3C7", "#1F4E79"])
    for i, v in enumerate(allv):
        b.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    b.set_xticks([0, 1]); b.set_xticklabels([f"all planar\n({D['buildings_all']})", f"kept\n({bc['kept']['buildings']})"], fontsize=8); b.set_ylim(0, 100); b.set_ylabel("lidar footprint area inside a mapped building, %", fontsize=7.5); b.tick_params(labelsize=8); b.spines[["top", "right"]].set_visible(False)
    pl.label(7.5, 8.35, "Precision of the filter")
    c = pl.ax(11.7, 4.45, 3.6, 3.5)
    sp = sc.info["realistic"]["displacement"]["split"]
    names = list(sp)
    xs = np.arange(len(names))
    c.bar(xs - 0.2, [sp[n]["edge_before"] for n in names], 0.38, color="#BDC3C7", label="no shift"); c.bar(xs + 0.2, [sp[n]["edge_after"] for n in names], 0.38, color="#1F4E79", label="roof-fitted shift")
    c.set_xticks(xs); c.set_xticklabels(names, fontsize=8); c.set_ylabel("mean image edge strength under roof outlines", fontsize=7.5); c.legend(fontsize=7.5, frameon=False, loc="lower right"); c.tick_params(labelsize=8); c.spines[["top", "right"]].set_visible(False)
    pl.label(11.7, 8.35, "Registering the photo to the roofs")
    panel(pl, sc.render("realistic", "houses", (1800, 600)), 0.6, 0.95, 7.2, 2.0, "Houses on the photo")
    panel(pl, sc.render("classified", "houses", (1800, 600)), 8.2, 0.95, 7.2, 2.0, "Houses, classified colours")
    heights = [b.height for b in D["buildings"]]
    pl.text(7.5, 4.1, "\n".join(textwrap.wrap(f"{bc['kept']['buildings']} buildings, {bc['kept']['footprint_area_m2'] / 1e4:.1f} ha of roof, median height {np.median(heights):.1f} m. They cover {bc['kept']['share_of_overture_area_covered'] * 100:.0f}% of the area Overture maps as buildings ({bc['overture_buildings']} footprints; rows of houses merge into one lidar footprint, so counts differ). Walls are drawn from the ground to the roof outline, so they are plain surfaces: no windows, doors or eaves.", 125)), fontsize=8, color=MUTED, va="top")
    pl.save("80_whiteriver_scene_buildings.png")


def run_81(sc):
    from lidar3d import classify as CL
    S, D = sc.S, sc.D
    pl = Plate("Better ways to make the maps: what was tried and what helped", "The same view of the yard with each rendering step added, and the two ways of registering the photo to the lidar", src(S), "Views share one camera")
    import pyvista as pv
    d = S["d"]
    # a: lidar returns as points, coloured by height (the earlier method)
    m = (np.abs(d["x"] - sc.yc[0]) <= 115) & (np.abs(d["y"] - sc.yc[1]) <= 115) & (S["hag"] > 0.4)
    P = np.c_[d["x"][m], d["y"][m], d["z"][m]].astype(float)
    pc = pv.PolyData(P); pc["rgb"] = (plt.get_cmap("turbo")(np.clip(S["hag"][m] / 24, 0, 1))[:, :3] * 255).astype(np.uint8)
    gl = [x for x in sc.L["realistic"] if x.get("kind") == "mesh"][:1]
    f, dist, az, el = sc.views["yard"]
    a1 = R.render(gl + [dict(mesh=pc, kind="points", point_size=3.0)], f, dist, az, el, (1500, 760), ssao=False)
    a2 = sc.render("realistic", "yard", (1500, 760), ssao=False)
    a3 = sc.render("realistic", "yard", (1500, 760), ssao=True)
    a4 = sc.render("classified", "yard", (1500, 760), ssao=True)
    for im, ttl, (l, bb) in ((a1, "1  Points coloured by height (the earlier method)", (0.6, 4.65)), (a2, "2  Meshes, templates, photo ground", (8.2, 4.65)), (a3, "3  Plus ambient occlusion (depth cues)", (0.6, 0.95)), (a4, "4  Classified colours, no photo", (8.2, 0.95))):
        panel(pl, im, l, bb, 7.2, 3.1 if bb < 2 else 3.4, ttl)
    # registration comparison
    img, tf = S["img"], S["tf"]
    ndvi = (img[3] - img[0]) / np.maximum(img[3] + img[0], 1.0)
    x, y, h = d["x"].astype(float), d["y"].astype(float), S["hag"]
    canopy = {}
    for nm, mk in (("west", x < -125), ("east", x >= -125), ("north", y >= -135), ("south", y < -135)):
        r = CL.fit_relief_displacement(ndvi, tf, d["cx"], d["cy"], x[mk], y[mk], h[mk], n=30000, seed=2)
        canopy[nm] = np.hypot(10 * r["bx"], 10 * r["by"])
    roofs = sc.info["realistic"]["displacement"]["split"]
    roofs_sh = {n: np.hypot(10 * v["bx"], 10 * v["by"]) for n, v in roofs.items()}
    # the middle strip: registration chart and the notes
    ax = pl.ax(8.9, 1.15, 2.9, 2.8)
    names = list(roofs_sh)
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [canopy[n] for n in names], 0.38, color="#C0392B", label="canopy colour fit")
    ax.bar(xs + 0.2, [roofs_sh[n] for n in names], 0.38, color="#1F4E79", label="roof edge fit")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylabel("fitted image shift per 10 m of height, m", fontsize=7); ax.tick_params(labelsize=7.5); ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    pl.label(8.9, 4.05, "Registering the photo")
    rows(pl, 12.0, 3.95, [("What helped", "Meshes lit and drawn as solids instead of points; ambient occlusion for depth; templates shaped to measured envelopes; a ground with no holes; photo colours for crowns and roofs registered with the roof-edge fit."),
                          ("What did not", "The canopy-colour fit looked better on its own score but gave different answers in each sub-window, so it was dropped. Hard shadows made the scene too dark to read and were left out."),
                          ("Still open", "True orthophotos from the lidar surface, textured walls, window and eave detail, and a higher-resolution photo: the ground texture is limited to NAIP's 0.6 m.")], 38, fs=7.6, head=9)
    pl.save("81_whiteriver_scene_methods.png")
    return canopy, roofs_sh


def run_82(sc):
    I = sc.info["realistic"]
    cs = I["clean"]
    S, D, N = sc.S, sc.D, sc.N
    d = S["d"]
    wm = D["wmask"]
    W = np.c_[d["x"][wm], d["y"][wm], d["z"][wm]].astype(float)
    cov_b = U.coverage(W, I["wires_raw"]); cov_a = U.coverage(W, I["wires_final"])
    pl = Plate("Cleaning up the wire model", "Duplicates, stubs and jitter removed from the network of fitted spans, observed pieces and bridges, with the coverage of detected wire points checked before and after", src(S), "Local grid, metres from the substation centre (UTM 10N)")
    win = BS.WIN
    for k, (wires, ttl) in enumerate(((I["wires_raw"], "Before"), (I["wires_final"], "After"))):
        a = pl.ax(0.6 + k * 4.45, 3.5, 4.35, 4.7)
        for w in wires:
            a.plot(w[:, 0], w[:, 1], color="#1F4E79", lw=0.5)
        a.set_xlim(win[0], win[1]); a.set_ylim(win[2] + 270, win[3]); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        pl.label(0.6 + k * 4.45, 8.35, f"{ttl}: {len(wires):,} polylines")
    b = pl.ax(10.0, 4.2, 2.4, 3.7)
    b.bar([0, 1], [cs["roughness_before_m"] * 100, cs["roughness_after_m"] * 100], color=["#BDC3C7", "#1F4E79"])
    b.set_xticks([0, 1]); b.set_xticklabels(["before", "after"], fontsize=8); b.set_ylabel("height jitter, cm per m (rms 2nd difference)", fontsize=7); b.tick_params(labelsize=8); b.spines[["top", "right"]].set_visible(False)
    pl.label(10.0, 8.35, "Smoothness")
    c = pl.ax(13.0, 4.2, 2.4, 3.7)
    v = [cov_b["share_wire_points_within_1m"] * 100, cov_a["share_wire_points_within_1m"] * 100]
    c.bar([0, 1], v, color=["#BDC3C7", "#1F4E79"])
    for i, x_ in enumerate(v):
        c.text(i, x_ + 1.5, f"{x_:.0f}%", ha="center", fontsize=9)
    c.set_xticks([0, 1]); c.set_xticklabels(["before", "after"], fontsize=8); c.set_ylim(0, 105); c.set_ylabel("detected wire points within 1 m of a model wire, %", fontsize=7); c.tick_params(labelsize=8); c.spines[["top", "right"]].set_visible(False)
    pl.label(13.0, 8.35, "Coverage kept")
    rows(pl, 0.7, 3.2, [("What the clean-up does", f"Pieces shorter than 3 m are dropped ({cs['dropped_short']}); a piece is dropped when 70% of it lies within 0.45 m of a longer piece, i.e. the same wire drawn twice ({cs['dropped_duplicate']}); heights are smoothed with a 5 m running mean, keeping the end points. {cs['before']:,} polylines become {cs['after']:,}, {cs['length_before_m'] / 1000:.1f} km becomes {cs['length_after_m'] / 1000:.1f} km."),
                        ("What it costs", f"Coverage of detected wire points within 1 m moves from {v[0]:.0f}% to {v[1]:.0f}%; height jitter falls from {cs['roughness_before_m'] * 100:.1f} to {cs['roughness_after_m'] * 100:.1f} cm per m. Wires drawn twice were overlapping tubes, which show as thick or flickering lines in the 3D views.")], 120, fs=8.2)
    pl.save("82_whiteriver_scene_wire_cleanup.png")


if __name__ == "__main__":
    sc = Scene()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, fn in (("75", run_75), ("76", run_76), ("77", run_77), ("78", run_78), ("79", run_79), ("80", run_80), ("81", run_81), ("82", run_82)):
        if which in (k, "all"):
            fn(sc)
