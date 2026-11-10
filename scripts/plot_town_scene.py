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
from lidar3d.plates_pro import ProPlate, text_width, wrap

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
            self.L[mode] = {k: m for k, m in L.items() if m is not None}
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
        return R.render(self.layers(mode, dist, view), f, dist, az, el, size, **kw)

    def layers(self, mode, dist, view=None):
        """Layers for one view. Far views draw wires as screen-space lines (a 7 cm tube is under a pixel at 800 m) and, in the classified map, mark every support with a bead; near views draw wire tubes."""
        out = []
        for k, m in self.L[mode].items():
            if k == "frames":
                out.append(dict(mesh=m, kind="points", point_size=3.0))
            elif k == "wires_lines":
                out.append(dict(mesh=m, kind="lines", color="#E8E8E8" if False else ("#141414" if mode == "realistic" else "#111111"), width=2.8 if dist > 500 else (1.6 if dist > 150 else 1.0)))
            elif k == "wires":
                if dist <= 500:
                    out.append(dict(mesh=m))
            elif k == "support_markers":
                if mode == "classified" and dist > 500:
                    out.append(dict(mesh=m))
            else:
                out.append(dict(mesh=m))
        return out


def src(S):
    return (f"Lidar: USGS 3DEP project WA_PierceCounty_1_2020, collected 2020-04-13, about 25 returns per m2; the survey carries no building, vegetation or wire classes, so every class is derived from the points. Imagery: NAIP, {S['naip_date']}. "
            "Trees, supports, wires and equipment are fitted or templated geometry: positions and sizes are measured, shapes are assumed. Reference: Overture Maps (used only to check).")


KICK = "White River scene  ·  3D models"
SBX, SBW = 9.85, 5.55


def thumbs(pl, sc, mode, y, h=1.3, views=(("yard", "Yard"), ("corridor", "Lines"), ("houses", "Houses"))):
    w = (SBW - 0.3) / 3
    for k, (v, t) in enumerate(views):
        x = SBX + k * (w + 0.15)
        pl.image(sc.render(mode, v, (int(w * 330), int(h * 330))), x, y, w, h)
        pl.caption(x, y - 0.03, w, t, fs=7.4)


def run_75(sc):
    I = sc.info["realistic"]
    pl = ProPlate(75, KICK, "The utility network in its lidar and imagery scene", "Aerial photo on the ground; trees, buildings, towers, poles, wires and substation equipment as 3D models fitted to the lidar", src(sc.S), "Crown and roof colours use the photo registered to the roofs")
    pl.panel_title(0.6, 8.14, "A", "White River substation and surroundings, looking north-north-east", w=9.0)
    pl.image(sc.render("realistic", "overview", (1920, 1390)), 0.6, 1.0, 9.0, 6.95)
    pl.panel_title(SBX, 8.14, "B", "What the model contains", w=SBW)
    y0 = pl.tiles(SBX, 7.85, SBW, [(f"{I['trees']:,}", "individual tree crowns"), (f"{I['buildings']}", "buildings with roof planes"), (f"{I['clean']['after']}", "cleaned wire polylines")])
    y0 = pl.notes(SBX, y0 - 0.05, SBW, [
        ("Built from the points", "Ground is the lidar surface with every hole interpolated. Crowns are cut from the canopy height model, roofs from planar returns, and towers and gantries are lattices fitted to their own returns."),
        ("Coloured from the photo", "Ground, crowns and roofs take their colour from the 0.6 m aerial photo, registered to the roof outlines because a photo shifts tall objects sideways."),
        ("Surface and templates", "Tree crowns are the lidar canopy surface itself. Equipment, poles and cross-arms are templates scaled to measured sizes. Nothing in the model is a photograph of the object.")])
    pl.panel_title(SBX, y0 - 0.02, "C", "Close views", w=SBW)
    thumbs(pl, sc, "realistic", max(1.5, y0 - 0.45 - 1.3))
    pl.save("75_whiteriver_scene_realistic_3d.png")


def run_76(sc):
    I = sc.info["realistic"]
    pl = ProPlate(76, KICK, "The same scene as a classified 3D map", "No imagery: every class in its own colour so the features read at a glance", src(sc.S), "Ground tint from the photo's NDVI")
    pl.panel_title(0.6, 8.14, "A", "Classified 3D map, looking north-north-east", w=9.0)
    pl.image(sc.render("classified", "overview", (1920, 1390)), 0.6, 1.0, 9.0, 6.95)
    pl.panel_title(SBX, 8.14, "B", "Classes", w=SBW)
    y0 = pl.legend(SBX, 7.9, SBW, [("patch", "#4C9A60", "trees (darker = taller)"), ("patch", "#DEA862", "buildings"), ("patch", "#8C5C30", "wood poles"), ("patch", "#46607F", "towers and yard structures"), ("patch", "#3C6EA0", "substation equipment"),
                                   ("patch", "#141414", "wires"), ("patch", "#C4D2B4", "ground, vegetated"), ("patch", "#D6D6D2", "ground, bare or paved"), ("patch", "#C8C4BA", "substation yard")], row_h=0.23)
    y0 = pl.notes(SBX, y0 - 0.1, SBW, [
        ("Why a classified view", "Removing the photo shows what was found rather than what was photographed. Every colour is a class the lidar supports, so gaps in the model are visible instead of being hidden by texture."),
        ("What the ground tint means", "Ground is tinted green where the aerial photo's vegetation index (NDVI) is 0.25 or more and grey elsewhere; the yard is sand. This is the only use of the photo in this view."),
        ("Reading the heights", "Tree colour darkens with height from 3 m to 35 m. Roofs are drawn with every roof plane; walls run from the ground to the roof outline.")])
    pl.panel_title(SBX, y0 - 0.02, "C", "Close views", w=SBW)
    thumbs(pl, sc, "classified", max(1.5, y0 - 0.45 - 1.3))
    pl.save("76_whiteriver_scene_classified_3d.png")


def run_77(sc):
    I = sc.info["realistic"]
    k = I["equipment_kinds"]
    pl = ProPlate(77, KICK, "Facilities: the substation yard and a transformer bay", "Equipment templates shaped to measured envelopes, open frames shown as lidar returns, gantries fitted to their returns, bus wires and fence", src(sc.S), "Left: photo ground. Right: classified colours")
    pl.panel_title(0.6, 8.14, "A", "Yard on the aerial photo", w=7.3)
    pl.image(sc.render("realistic", "yard", (1825, 825)), 0.6, 4.65, 7.3, 3.3)
    pl.panel_title(8.1, 8.14, "B", "Yard in classified colours", w=7.3)
    pl.image(sc.render("classified", "yard", (1825, 825)), 8.1, 4.65, 7.3, 3.3)
    pl.panel_title(0.6, 4.35, "C", "A transformer bay on the aerial photo", w=7.3)
    pl.image(sc.render("realistic", "transformer", (1825, 800)), 0.6, 1.0, 7.3, 3.2)
    pl.panel_title(8.1, 4.35, "D", "What is drawn, and how far to trust it", w=7.3)
    pl.notes(8.1, 4.05, 7.3, [
        ("Drawn from the lidar", f"{I['boxy_blocks']} box-like blocks become templates: {k.get('transformer', 0)} transformer, {k.get('column apparatus', 0)} porcelain-column apparatus and {k.get('cabinet or small building', 0)} cabinets or small buildings. {I['open_frames']} open-frame blocks, filled under 22% of their envelope, stay as the lidar returns themselves, because a box would claim a solid that is not there. Gantries and towers are lattices fitted to their own returns, and the fence follows the mapped yard outline."),
        ("Assumed", "The template follows from size alone (large and low: transformer; tall and narrow: porcelain column; otherwise cabinet), so it can be wrong. Tank, radiators, conservator and bushings are assumed shapes. No equipment type or rating is read from the lidar and no drawing was used."),
        ("Gaps", "Where the lidar saw no wire between two gantries, none is drawn.")], fs=8.4)
    pl.save("77_whiteriver_scene_facilities_3d.png")


def _sample_ndvi(sc, x, y, h=0.0):
    fit = sc.info["realistic"]["displacement"]
    _, nd = BS.sample_naip(sc.S, np.asarray(x) + h * (fit["k"] * np.asarray(x) + fit["bx"]), np.asarray(y) + h * (fit["k"] * np.asarray(y) + fit["by"]))
    return nd


def run_78(sc):
    S, D, N = sc.S, sc.D, sc.N
    d = S["d"]
    win = BS.WIN
    x0, x1, y0, y1 = win
    pl = ProPlate(78, KICK, "Classifying the scene", "Ground cover, tree crowns, buildings, the substation yard, wires and supports, each with the check it can be tested by", src(S))
    pl.panel_title(0.6, 8.14, "A", "Classes in plan", w=8.9)
    a = pl.ax(0.6, 1.0, 8.9, 6.95)
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
    for w in N["K"]["span_w"] + [x_["xyz"] for x_ in N["K"]["adds"]] + [x_["xyz"] for x_ in N["br"]]:
        a.plot(w[:, 0], w[:, 1], color="#141414", lw=0.6, zorder=5)
    for w in N["Y"]["wires"]:
        a.plot(w["xyz"][:, 0], w["xyz"][:, 1], color="#141414", lw=0.5, zorder=5)
    sup = N["K"]["sup"][N["K"]["sup"].tier == "support"]
    a.scatter(sup.x, sup.y, s=14, marker="s", color="#46607F", zorder=6, edgecolors="white", linewidths=0.3)
    ts = N["Y"]["towers"]
    a.scatter([t["x"] for t in ts], [t["y"] for t in ts], s=10, marker="^", color="#46607F", zorder=6)
    for b in N["Y"]["blocks"]:
        a.plot(*b["points"][:, :2].T, ".", ms=0.5, color="#3C6EA0", zorder=4)
    a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_aspect("equal"); a.tick_params(labelsize=7, colors="#5B6470")
    from lidar3d.plates import north_arrow, scalebar
    scalebar(a, win, 100.0); north_arrow(a, win)
    ntree = len(T["x"])
    crown_cells = tl > 0
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
    pl.panel_title(SBX, 8.14, "B", "Key numbers", w=SBW)
    y0_ = pl.tiles(SBX, 7.85, SBW, [(f"{ntree:,}", f"tree crowns, {ntree / ((x1 - x0) * (y1 - y0) / 1e4):.0f} per ha"), (f"{bc['kept']['buildings']}", f"buildings, {bc['kept']['footprint_area_m2'] / 1e4:.1f} ha of roof"), (f"{complete * 100:.0f}%", "of tall returns (5 m up) fall inside a crown")])
    y0_ = pl.notes(SBX, y0_ - 0.05, SBW, [
        ("Trees", f"Median height {np.median(T['height']):.0f} m, {crown_cells.sum() * 0.25 / 1e4:.1f} ha of crown. Photo check: {np.mean(ndc >= 0.3) * 100:.0f}% of crown cells are vegetated (NDVI 0.3 or more) at the photo position the displacement model predicts, {np.mean(ndc0 >= 0.3) * 100:.0f}% without it, so the correction changes nothing measurable here."),
        ("Buildings", f"Of {D['buildings_all']} planar footprints, {bc['kept']['buildings']} are real roofs; the rest are crowns or towers (roof NDVI 0.25 or more, or too tall for their size, or cut by the window edge). {bc['kept']['share_of_lidar_area_inside_overture'] * 100:.0f}% of the kept area lies inside a mapped building (42% before filtering), and the kept set covers {bc['kept']['share_of_overture_area_covered'] * 100:.0f}% of the mapped area."),
        ("Image registration", f"Roof outlines sit on stronger image edges after a shift of about {fit['shift_at_10m_height_m']:.1f} m per 10 m of height ({fit['edge_before']:.1f} to {fit['edge_after']:.1f}), stable across four sub-windows. A fit from canopy colour was rejected: its answer changed too much between halves (plate 81)."),
        ("Not classified", "Roads, cars, water and low shrubs are not separate classes. Ground cover is only vegetated or not.")], fs=8.3)
    from matplotlib.lines import Line2D
    pl.legend(SBX, y0_ - 0.0, SBW, [("patch", "#80B070", "tree crowns (darker = taller)"), ("patch", "#DEA862", "buildings"), ("patch", "#C8C4BA", "substation yard"), ("patch", "#CCDBB8", "vegetated ground"), ("patch", "#DCDCD6", "bare or paved ground"),
                                      ("line", "#141414", "wires"), ("square", "#46607F", "supports"), ("dot", "#3C6EA0", "yard equipment returns")], fs=7.2, row_h=0.19)
    pl.save("78_whiteriver_scene_classification.png")


def run_79(sc):
    D, S = sc.D, sc.S
    T = D["trees"]
    conifer = ((T["height"] - np.minimum(T["base"], 0.6 * T["height"])) / np.maximum(2 * T["radius"] * 1.25, 0.5)) > 1.6
    pl = ProPlate(79, KICK, "Trees: individual crowns from the canopy height model", "Watershed segmentation cuts the canopy into crowns, each with a top, height, radius and crown base, drawn as a crown and a trunk", src(S), "Crown shape (cone or ellipsoid) follows crown length against width")
    win = (-420, -150, 0, 270)
    pl.panel_title(0.6, 8.14, "A", "Canopy height model, tree tops and crowns", w=4.4)
    a = pl.ax(0.75, 4.35, 4.15, 3.6)
    ch = D["chm"]
    x0, y0 = BS.WIN[0], BS.WIN[2]
    sl = (slice(int((win[2] - y0) / 0.5), int((win[3] - y0) / 0.5)), slice(int((win[0] - x0) / 0.5), int((win[1] - x0) / 0.5)))
    a.imshow(ch[sl], origin="lower", extent=(win[0], win[1], win[2], win[3]), cmap="Greens", vmin=0, vmax=40, interpolation="nearest")
    m = (T["x"] >= win[0]) & (T["x"] <= win[1]) & (T["y"] >= win[2]) & (T["y"] <= win[3])
    a.scatter(T["x"][m], T["y"][m], s=4, color="#B03A2E", zorder=3)
    from matplotlib.patches import Circle
    for xx, yy, rr in zip(T["x"][m], T["y"][m], T["radius"][m]):
        a.add_patch(Circle((xx, yy), rr, fill=False, ec="#222", lw=0.3))
    a.set_aspect("equal"); a.tick_params(labelsize=7, colors="#5B6470")
    pl.panel_title(5.2, 8.14, "B", "Tree height by crown shape", w=4.3)
    b = pl.ax(5.45, 4.35, 3.95, 3.6)
    bins = np.arange(3, 50, 2)
    b.hist([T["height"][~conifer], T["height"][conifer]], bins=bins, stacked=True, color=["#8FBF6A", "#2F6B3F"], label=["broadleaf-shaped", "conifer-shaped"])
    b.set_xlabel("tree height, m", fontsize=8, color="#5B6470"); b.set_ylabel("trees", fontsize=8, color="#5B6470"); b.legend(fontsize=7.5, frameon=False); b.tick_params(labelsize=8, colors="#5B6470"); b.spines[["top", "right"]].set_visible(False)
    pl.panel_title(0.6, 3.95, "C", "Crowns on the photo colours", w=4.4)
    pl.image(sc.render("realistic", "trees", (1100, 800)), 0.6, 1.0, 4.4, 2.75)
    pl.panel_title(5.2, 3.95, "D", "Crowns in classified colours", w=4.3)
    pl.image(sc.render("classified", "trees", (1100, 800)), 5.2, 1.0, 4.3, 2.75)
    pl.panel_title(SBX, 8.14, "E", "Key numbers", w=SBW)
    y0_ = pl.tiles(SBX, 7.85, SBW, [(f"{len(T['x']):,}", "tree crowns found"), (f"{np.median(T['height']):.0f} m", f"median height, tallest {T['height'].max():.0f} m"), (f"{conifer.mean() * 100:.0f}%", "of crowns are conifer-shaped")])
    y0_ = pl.notes(SBX, y0_ - 0.05, SBW, [
        ("Method", "The canopy height model (highest return per 0.5 m cell above 1.5 m) is smoothed and split at local maxima at least 1.8 m apart; each crown is its basin down to 40% of its own height. The crown base is the 12th percentile of returns inside the crown."),
        ("Drawing", "A crown is an ellipsoid, or three stacked cones when it is long and narrow. The radius is widened by a quarter because the 40% cut trims the edges. Trunks are cylinders up to the crown base.")])
    pl.panel_title(SBX, y0_ - 0.02, "F", "Crown radius against height", w=SBW)
    c = pl.ax(SBX + 0.35, 1.2, SBW - 0.5, 1.85)
    c.scatter(T["height"], T["radius"], s=3, alpha=0.3, color="#2F6B3F", linewidths=0)
    c.set_xlabel("tree height, m", fontsize=7.5, color="#5B6470"); c.set_ylabel("crown radius, m", fontsize=7.5, color="#5B6470"); c.tick_params(labelsize=7.5, colors="#5B6470"); c.spines[["top", "right"]].set_visible(False)
    pl.save("79_whiteriver_scene_trees.png")


def run_80(sc):
    D, S, N = sc.D, sc.S, sc.N
    bc = D["building_check"]
    heights = [b.height for b in D["buildings"]]
    pl = ProPlate(80, KICK, "Buildings: roofs from the lidar, checked against the map and the photo", "Planar returns become footprints and roof planes; crowns that look planar are removed using the photo's vegetation index; the photo is registered to the roofs", src(S))
    pl.panel_title(0.6, 8.14, "A", "Buildings found, coloured by height (0 to 16 m)", w=5.6)
    a = pl.ax(0.75, 3.85, 5.4, 4.1)
    from matplotlib.patches import Polygon as MP
    ccx, ccy = D["bld_offset"]
    cm = plt.get_cmap("YlOrBr")
    for b in D["buildings"]:
        q = np.array(b.footprint.exterior.coords) + [ccx, ccy]
        a.add_patch(MP(q, closed=True, fc=cm(min(b.height / 16, 1)), ec="#555", lw=0.4))
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
    a.set_xlim(BS.WIN[0], BS.WIN[1]); a.set_ylim(BS.WIN[2], BS.WIN[3]); a.set_aspect("equal"); a.tick_params(labelsize=7, colors="#5B6470")
    pl.legend(6.45, 7.9, 3.1, [("line", "#2E6F9E", "mapped building footprint"), ("patch", "#E8A04A", "building found in the lidar")], fs=7.2)
    pl.panel_title(6.45, 7.2, "B", "Precision of the roof filter", w=3.05)
    b = pl.ax(6.65, 4.35, 2.8, 2.4)
    allv = [bc["all_planar_footprints"]["share_of_lidar_area_inside_overture"] * 100, bc["kept"]["share_of_lidar_area_inside_overture"] * 100]
    b.bar([0, 1], allv, color=["#BDC3C7", "#1F4E79"])
    for i, v in enumerate(allv):
        b.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9, color="#15191D")
    b.set_xticks([0, 1]); b.set_xticklabels([f"all planar\n({D['buildings_all']})", f"kept\n({bc['kept']['buildings']})"], fontsize=8); b.set_ylim(0, 100); b.set_ylabel("area inside a mapped building, %", fontsize=7.5, color="#5B6470"); b.tick_params(labelsize=8, colors="#5B6470"); b.spines[["top", "right"]].set_visible(False)
    pl.panel_title(0.6, 3.5, "C", "Houses on the photo", w=4.4)
    pl.image(sc.render("realistic", "houses", (1100, 760)), 0.6, 1.0, 4.4, 2.3)
    pl.panel_title(5.2, 3.5, "D", "Houses in classified colours", w=4.3)
    pl.image(sc.render("classified", "houses", (1100, 760)), 5.2, 1.0, 4.3, 2.3)
    pl.panel_title(SBX, 8.14, "E", "Key numbers", w=SBW)
    y0_ = pl.tiles(SBX, 7.85, SBW, [(f"{bc['kept']['buildings']}", f"buildings, {bc['kept']['footprint_area_m2'] / 1e4:.1f} ha of roof, median {np.median(heights):.1f} m"), (f"{bc['kept']['share_of_lidar_area_inside_overture'] * 100:.0f}%", "of found roof area lies inside a mapped building"), (f"{bc['kept']['share_of_overture_area_covered'] * 100:.0f}%", "of the mapped building area is covered")])
    y0_ = pl.notes(SBX, y0_ - 0.05, SBW, [
        ("Why a filter is needed", f"Dense crowns also form planar returns, so {D['buildings_all']} footprints were found but only {bc['kept']['buildings']} are roofs. A footprint is kept when its roof is not vegetated in the photo (NDVI under 0.25 at the displaced roof position), is at least 40 m2, is 2.5 to 25 m tall (and over 300 m2 when over 12 m), and is not cut by the window edge."),
        ("Reading the check", f"Overture maps {bc['overture_buildings']} separate footprints where the lidar merges each row of houses into one, so counts differ and area overlap is the fair measure. Walls are plain surfaces from ground to roof outline: no windows, doors or eaves.")], fs=8.3)
    pl.panel_title(SBX, y0_ - 0.02, "F", "Registering the photo to the roofs", w=SBW)
    c = pl.ax(SBX + 0.35, 1.2, SBW - 0.5, 1.85)
    sp = sc.info["realistic"]["displacement"]["split"]
    names = list(sp)
    xs = np.arange(len(names))
    c.bar(xs - 0.2, [sp[n]["edge_before"] for n in names], 0.38, color="#BDC3C7", label="no shift"); c.bar(xs + 0.2, [sp[n]["edge_after"] for n in names], 0.38, color="#1F4E79", label="roof-fitted shift")
    c.set_xticks(xs); c.set_xticklabels(names, fontsize=8); c.set_ylabel("edge strength under roofs", fontsize=7.5, color="#5B6470"); c.legend(fontsize=7.2, frameon=False, loc="lower right"); c.tick_params(labelsize=8, colors="#5B6470"); c.spines[["top", "right"]].set_visible(False)
    pl.save("80_whiteriver_scene_buildings.png")


def run_81(sc):
    from lidar3d import classify as CL
    import pyvista as pv
    S, D = sc.S, sc.D
    d = S["d"]
    pl = ProPlate(81, KICK, "Better ways to make the maps: what was tried and what helped", "The same view of the yard with each rendering step added, and two ways of registering the photo to the lidar", src(S), "All four views share one camera")
    m = (np.abs(d["x"] - sc.yc[0]) <= 115) & (np.abs(d["y"] - sc.yc[1]) <= 115) & (S["hag"] > 0.4)
    P = np.c_[d["x"][m], d["y"][m], d["z"][m]].astype(float)
    pc = pv.PolyData(P); pc["rgb"] = (plt.get_cmap("turbo")(np.clip(S["hag"][m] / 24, 0, 1))[:, :3] * 255).astype(np.uint8)
    gl = [dict(mesh=sc.L["realistic"]["ground"])]
    f, dist, az, el = sc.views["yard"]
    a1 = R.render(gl + [dict(mesh=pc, kind="points", point_size=3.0)], f, dist, az, el, (1070, 620), ssao=False)
    a2 = sc.render("realistic", "yard", (1070, 620), ssao=False)
    a3 = sc.render("realistic", "yard", (1070, 620), ssao=True)
    a4 = sc.render("classified", "yard", (1070, 620), ssao=True)
    cells = ((a1, "A", "Points coloured by height (the earlier method)", 0.6, 4.7), (a2, "B", "Lit meshes and templates on the photo", 5.1, 4.7), (a3, "C", "Plus ambient occlusion for depth", 0.6, 1.0), (a4, "D", "Classified colours, no photo", 5.1, 1.0))
    for im, L_, ttl, x, y in cells:
        pl.panel_title(x, y + 3.15, L_, ttl, w=4.3)
        pl.image(im, x, y, 4.3, 2.95)
    img, tf = S["img"], S["tf"]
    ndvi = (img[3] - img[0]) / np.maximum(img[3] + img[0], 1.0)
    x, y, h = d["x"].astype(float), d["y"].astype(float), S["hag"]
    canopy = {}
    for nm, mk in (("west", x < -125), ("east", x >= -125), ("north", y >= -135), ("south", y < -135)):
        r = CL.fit_relief_displacement(ndvi, tf, d["cx"], d["cy"], x[mk], y[mk], h[mk], n=30000, seed=2)
        canopy[nm] = float(np.hypot(10 * r["bx"], 10 * r["by"]))
    roofs = sc.info["realistic"]["displacement"]["split"]
    roofs_sh = {n: float(np.hypot(10 * v["bx"], 10 * v["by"])) for n, v in roofs.items()}
    pl.panel_title(SBX, 8.14, "E", "Registering the photo: two methods", w=SBW)
    ax = pl.ax(SBX + 0.35, 5.2, SBW - 0.5, 2.45)
    names = list(roofs_sh)
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [canopy[n] for n in names], 0.38, color="#C0392B", label="canopy colour fit")
    ax.bar(xs + 0.2, [roofs_sh[n] for n in names], 0.38, color="#1F4E79", label="roof edge fit")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8); ax.set_ylabel("shift per 10 m of height, m", fontsize=7.5, color="#5B6470"); ax.tick_params(labelsize=8, colors="#5B6470"); ax.legend(fontsize=7.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    pl.notes(SBX, 4.75, SBW, [
        ("What helped", "Lit meshes instead of points, ambient occlusion for depth cues, templates scaled to measured envelopes, a ground with no holes, and photo colours registered with the roof-edge fit."),
        ("What did not", "Fitting the photo shift from canopy colour scored well on its own but gave a different answer in every sub-window, so it was dropped. Hard shadows made the scene too dark to read."),
        ("Still open", "True orthophotos made from the lidar surface, textured walls, window and eave detail, and a photo sharper than NAIP's 0.6 m, which limits ground texture.")], fs=8.3)
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
    v = [cov_b["share_wire_points_within_1m"] * 100, cov_a["share_wire_points_within_1m"] * 100]
    pl = ProPlate(82, KICK, "Cleaning up the wire model", "Duplicates, stubs and jitter removed from the network of fitted spans, observed pieces and bridges, with the coverage of detected wire points checked before and after", src(S))
    win = BS.WIN
    for k, (wires, ttl, L_) in enumerate(((I["wires_raw"], "Before", "A"), (I["wires_final"], "After", "B"))):
        x = 0.6 + k * 4.6
        pl.panel_title(x, 8.14, L_, f"{ttl}: {len(wires):,} polylines", w=4.4)
        a = pl.ax(x + 0.05, 3.35, 4.3, 4.6)
        for w in wires:
            a.plot(w[:, 0], w[:, 1], color="#1F4E79", lw=0.5)
        a.set_xlim(win[0], win[1]); a.set_ylim(win[2] + 270, win[3]); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#B7BCC3")
    pl.panel_title(0.6, 3.0, "C", "Smoothness of the wires", w=4.4)
    b = pl.ax(1.0, 1.15, 3.8, 1.5)
    b.barh([0, 1], [cs["roughness_before_m"] * 100, cs["roughness_after_m"] * 100], color=["#BDC3C7", "#1F4E79"])
    b.set_yticks([0, 1]); b.set_yticklabels(["before", "after"], fontsize=8); b.set_xlabel("height jitter, cm per m (rms second difference)", fontsize=7.5, color="#5B6470"); b.tick_params(labelsize=8, colors="#5B6470"); b.spines[["top", "right"]].set_visible(False)
    pl.panel_title(5.2, 3.0, "D", "Detected wire kept in the model", w=4.3)
    c = pl.ax(5.6, 1.15, 3.6, 1.5)
    c.barh([0, 1], v, color=["#BDC3C7", "#1F4E79"])
    for i, x_ in enumerate(v):
        c.text(x_ + 1, i, f"{x_:.0f}%", va="center", fontsize=9)
    c.set_yticks([0, 1]); c.set_yticklabels(["before", "after"], fontsize=8); c.set_xlim(0, 110); c.set_xlabel("wire points within 1 m of a model wire, %", fontsize=7.5, color="#5B6470"); c.tick_params(labelsize=8, colors="#5B6470"); c.spines[["top", "right"]].set_visible(False)
    pl.panel_title(SBX, 8.14, "E", "Key numbers", w=SBW)
    y0_ = pl.tiles(SBX, 7.85, SBW, [(f"{cs['before']:,}", f"polylines before, {cs['after']:,} after"), (f"{cs['dropped_duplicate']}", "duplicate pieces removed"), (f"{cs['dropped_short']}", "stubs under 3 m removed")])
    pl.notes(SBX, y0_ - 0.05, SBW, [
        ("What the clean-up does", f"A piece is dropped when 70% of it lies within 0.45 m of a longer piece, which means the same wire was drawn twice, once by a span fit and once as an observed piece. Pieces under 3 m are dropped. Heights are smoothed with a 5 m running mean and the end points are kept. Length goes from {cs['length_before_m'] / 1000:.1f} km to {cs['length_after_m'] / 1000:.1f} km."),
        ("What it costs", f"Coverage of detected wire points within 1 m moves from {v[0]:.0f}% to {v[1]:.0f}%, so almost nothing real is removed, while height jitter falls from {cs['roughness_before_m'] * 100:.1f} to {cs['roughness_after_m'] * 100:.1f} cm per m."),
        ("Why it matters", "Wires drawn twice are overlapping tubes, which show as thick or flickering lines in the 3D views. Smoothing removes the stair-step left by fitting each span separately.")], fs=8.4)
    pl.save("82_whiteriver_scene_wire_cleanup.png")


if __name__ == "__main__":
    sc = Scene()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, fn in (("75", run_75), ("76", run_76), ("77", run_77), ("78", run_78), ("79", run_79), ("80", run_80), ("81", run_81), ("82", run_82)):
        if which in (k, "all"):
            fn(sc)
