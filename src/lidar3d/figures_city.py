"""Plates 09 to 11: Seattle Center buildings from lidar."""
from __future__ import annotations

import numpy as np
import trimesh
from matplotlib import colormaps
from matplotlib.colors import Normalize
from shapely.affinity import translate
from shapely.geometry import MultiPolygon

from . import render3d
from .plates import FAINT, INK, MUTED, Plate, colorbar, map_frame, north_arrow, scalebar

SRC = ("Lidar: USGS 3D Elevation Program (3DEP) project WA_KingCo_1_2021, public on AWS (usgs-lidar-public), collected {d0}. Classes present: 1 unclassified, 2 ground; no building class. "
       "Footprints: Overture Maps buildings, release 2026-09-23.1 (aggregated from open sources; the footprints come from a 2026 release, five years after the lidar).")
VMAX = 45.0


def _label(r):
    return r["name"] if isinstance(r.get("name"), str) and r["name"] else "unnamed"


def _fmt(v, d=0):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def fig_buildings_map(P, B, d0):
    ok = B[B.status == "ok"].copy()
    ex = (P["x0"], P["x0"] + P["w"], P["y1"] - P["h"], P["y1"])
    pl = Plate("Seattle Center buildings from lidar", f"{len(ok)} building footprints given a roof height from {P['npts'] / 1e6:.1f} million lidar returns ({P['ppm2']:.0f} per m²); height is the main roof level above the ground around the building",
               SRC.format(d0=d0), "Grid: UTM zone 10N, metres")
    a = pl.ax(0.6, 1.05, 7.0, 7.0)
    a.imshow(P["hs"], extent=ex, cmap="gray", vmin=0.2, vmax=1.0)
    cm = colormaps["viridis"]; nm = Normalize(0, VMAX)
    for _, r in ok.iterrows():
        gs = list(r.geometry.geoms) if isinstance(r.geometry, MultiPolygon) else [r.geometry]
        for g in gs:
            xs, ys = g.exterior.xy
            a.fill(xs, ys, fc=cm(nm(r.height_m)), ec="white", lw=0.35, alpha=0.92)
    top = ok[~ok.clipped].sort_values("height_max_m", ascending=False).head(10).reset_index(drop=True)
    for i, r in top.iterrows():
        p = r.geometry.representative_point()
        a.text(p.x, p.y, str(i + 1), fontsize=8, fontweight="bold", color="white", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.18", fc="#111111", ec="white", lw=0.5))
    map_frame(a, ex, 100, grid=False); scalebar(a, ex, 100); north_arrow(a, ex)
    pl.label(0.6, 8.25, "Footprints coloured by lidar height, over lidar surface hillshade")
    colorbar(pl, "viridis", 0, VMAX, 8.3, 7.65, 3.6, 0.14, "Height above ground, m (values above 45 m share the top colour)")
    n_flat = int((ok.roof_type == "flat").sum()); n_p = int((ok.roof_type == "pitched").sum()); n_c = int((ok.roof_type == "complex").sum())
    pl.label(8.3, 7.15, "The ten highest structures (numbers on the map)")
    cols = [8.3, 8.75, 11.3, 12.5, 13.6, 14.6]
    for x, t in zip(cols, ["", "Name", "Main roof, m", "Highest, m", "Overture, m", ""]):
        if t:
            pl.text(x, 6.9, t, fontsize=8, color=MUTED, fontweight="bold")
    for i, r in top.iterrows():
        y = 6.6 - i * 0.25
        pl.text(cols[0], y, str(i + 1), fontsize=8.5, fontweight="bold")
        pl.text(cols[1], y, _label(r)[:30], fontsize=8.5)
        pl.text(cols[2], y, _fmt(r.height_m, 1), fontsize=8.5)
        pl.text(cols[3], y, _fmt(r.height_max_m, 1), fontsize=8.5)
        pl.text(cols[4], y, _fmt(r.get("height"), 1), fontsize=8.5)
    pl.text(8.3, 4.05, "\n".join(__import__("textwrap").wrap(f"The main roof is the 70th percentile of the roof surface. For the Space Needle that is the saucer, and the highest return is the spire. {int(ok.clipped.sum())} buildings cut by the edge of the area are measured on the part inside and are not ranked.", 100)),
            fontsize=8, color=MUTED, va="top", linespacing=1.35, wrap=False)
    # distribution
    h = pl.ax(8.5, 1.25, 3.4, 1.9)
    h.hist(ok.height_m.clip(0, 45), bins=np.arange(0, 48, 3), color="#3B6E8F", ec="white", lw=0.5)
    h.set_xlabel("main roof height above ground, m", fontsize=8, color=MUTED); h.set_ylabel("buildings", fontsize=8, color=MUTED)
    h.tick_params(labelsize=7.5, length=2); h.spines[["top", "right"]].set_visible(False)
    pl.text(8.5, 3.2, f"Median {ok.height_m.median():.1f} m; {int((ok.height_m < 5).sum())} of {len(ok)} are under 5 m", fontsize=8.3, color=MUTED, va="bottom")
    pl.text(12.3, 3.2, "Roof shape from a fitted plane", fontsize=8.3, color=MUTED, va="bottom")
    b = pl.ax(12.3, 1.25, 3.1, 1.9)
    b.barh(["flat (< 5°)", "pitched", "complex"], [n_flat, n_p, n_c], color=["#3B6E8F", "#9A3B12", "#8A8A8A"])
    b.invert_yaxis(); b.tick_params(labelsize=8, length=2); b.spines[["top", "right"]].set_visible(False); b.set_xlabel("buildings", fontsize=8, color=MUTED)
    for i, v in enumerate([n_flat, n_p, n_c]):
        b.text(v + 0.8, i, str(v), va="center", fontsize=8)
    return pl.save("09_seattle_buildings.png"), top


def _scene_points(B, terrain, P, n_total=9_000_000, seed=5):
    """Sample points on the LOD1 model (terrain plus blocks) with simple lambert shading, for the software renderer."""
    rng = np.random.default_rng(seed)
    cm = colormaps["viridis"]
    meshes = []
    t = terrain
    meshes.append((t, np.array([200, 196, 186], float), None))
    for _, r in B[B.status == "ok"].iterrows():
        if r.roof_z <= r.base_z + 0.5:
            continue
        gs = list(r.geometry.geoms) if isinstance(r.geometry, MultiPolygon) else [r.geometry]
        if r.get("name") in P.get("solids", {}):
            m = P["solids"][r.get("name")].copy()
            m.apply_translation([-P["cx"], -P["cy"], -P["cz"]])
            meshes.append((m, np.array(cm(min(r.height_m / VMAX, 1.0))[:3]) * 255, None))
            continue
        for g in gs:
            m = trimesh.creation.extrude_polygon(g, float(r.roof_z - r.base_z))
            m.apply_translation([-P["cx"], -P["cy"], r.base_z - P["cz"]])
            meshes.append((m, np.array(cm(min(r.height_m / VMAX, 1.0))[:3]) * 255, None))
    import shapely
    from shapely.affinity import translate
    foot_union = translate(B[B.status == 'ok'].geometry.buffer(0.6).union_all(), -P['cx'], -P['cy'])
    areas = np.array([m.area for m, _, _ in meshes]); areas[0] *= 0.35; cnt = np.maximum((areas / areas.sum() * n_total).astype(int), 50)
    L = np.array([0.4, -0.5, 0.75]); L /= np.linalg.norm(L)
    pts, cols = [], []
    for (m, c, _), k in zip(meshes, cnt):
        p, f = trimesh.sample.sample_surface(m, int(k), seed=int(rng.integers(1 << 30)))
        if m is terrain:                                    # ground under a footprint is interpolated, not measured: leave it out so it cannot poke through the walls
            keep = ~shapely.contains_xy(foot_union, p[:, 0], p[:, 1])
            p, f = p[keep], f[keep]
        sh = 0.5 + 0.5 * np.clip(m.face_normals[f] @ L, 0, 1)
        pts.append(p); cols.append(np.clip(c[None, :] * sh[:, None], 0, 255).astype(np.uint8))
    return np.vstack(pts), np.vstack(cols)


def fig_buildings_3d(P, B, cloud_pts, cloud_hag, terrain, d0):
    pl = Plate("Seattle Center in 3D: lidar points and the building model", "Left: the point cloud coloured by height above ground. Right: the block (LOD1) model built from the footprints and the lidar roof heights",
               SRC.format(d0=d0), "Vertical exaggeration 1.0; software renderer, no GPU")
    ctr = np.array([0.0, 0.0, 0.0])
    cm = colormaps["viridis"]
    col = (cm(np.clip(cloud_hag / VMAX, 0, 1))[:, :3] * 255).astype(np.uint8)
    col[cloud_hag < 0.5] = (205, 200, 190)
    img1, info = render3d.render(cloud_pts, col, az=215, el=42, width=1500, height=1000, zexag=1.0, center=ctr, splat=2)
    a = pl.ax(0.6, 2.6, 7.4, 4.93 * 1.0); a.imshow(img1); a.axis("off")
    pl.label(0.6, 8.25, "Lidar returns, 2.4 million of the 5.9 million, by height above ground")
    pts, colm = _scene_points(B, terrain, P)
    img2, _ = render3d.render(pts, colm, az=215, el=42, width=1500, height=1000, zexag=1.0, center=ctr, scale=info["scale"], off=info["off"], splat=2)
    # same projection so both views match
    b = pl.ax(8.2, 2.6, 7.4, 4.93); b.imshow(img2); b.axis("off")
    pl.label(8.2, 8.25, "LOD1 model: footprint extruded from the ground to the main roof")
    pl.text(0.6, 2.2, "Both views use the same viewing direction and scale. The block at the left edge is the Verve tower; the Space Needle is the thin tower near the centre-left, modelled as a shape fitted to its own lidar returns (plate 13).\n"
                      "LOD1 means flat roofs at one height per footprint: pitched roofs, setbacks and rooftop equipment are not modelled. Overture footprints include buildings finished after the 2021 lidar,\n"
                      "which appear as low blocks or are skipped.", fontsize=8.5, color=MUTED, linespacing=1.4)
    return pl.save("10_seattle_3d.png")


def fig_buildings_check(B, d0, n_all):
    ok = B[B.status == "ok"]
    v = ok.dropna(subset=["height"])
    pl = Plate("How the lidar heights compare with Overture's", f"Overture gives a height for {len(v)} of these buildings; its source is not stated, so this is a comparison between two datasets, not an accuracy test",
               SRC.format(d0=d0), "No field survey was available")
    res = {}
    for k, (col, title, xo) in enumerate([("height_m", "Main roof level (used by the model)", 0.9), ("height_max_m", "Highest return in the footprint", 5.6)]):
        d = v[col] - v.height
        res[col] = dict(n=len(v), r=float(np.corrcoef(v[col], v.height)[0, 1]), bias=float(d.median()), mae=float(d.abs().mean()), within3=float((d.abs() <= 3).mean()))
        a = pl.ax(xo, 4.2, 4.0, 3.7)
        a.scatter(v.height, v[col].clip(0, 60), s=14, c="#3B6E8F", alpha=0.75, lw=0)
        a.plot([0, 60], [0, 60], color=INK, lw=0.8); a.plot([0, 60], [3, 63], color=FAINT, lw=0.6, ls="--"); a.plot([0, 60], [-3, 57], color=FAINT, lw=0.6, ls="--")
        a.set_xlim(0, 60); a.set_ylim(0, 60); a.set_xlabel("Overture height, m", fontsize=8, color=MUTED); a.set_ylabel("lidar height, m", fontsize=8, color=MUTED)
        a.tick_params(labelsize=7.5, length=2); a.spines[["top", "right"]].set_visible(False)
        pl.label(xo, 8.25, title)
        r_ = res[col]
        pl.text(xo, 3.7, f"r = {r_['r']:.2f}   median difference {r_['bias']:+.1f} m   mean absolute {r_['mae']:.1f} m   within 3 m: {r_['within3'] * 100:.0f}%   (n = {r_['n']})", fontsize=8.2, color=INK, va="top")
    # biggest disagreements by the highest-return measure
    w = v.assign(d=v.height_max_m - v.height)
    w = w.reindex(w.d.abs().sort_values(ascending=False).index).head(8)
    pl.label(10.6, 8.25, "Largest differences (highest return minus Overture)")
    pl.text(10.6, 7.95, "Building", fontsize=8, color=MUTED, fontweight="bold"); pl.text(13.2, 7.95, "Overture", fontsize=8, color=MUTED, fontweight="bold")
    pl.text(14.1, 7.95, "Lidar max", fontsize=8, color=MUTED, fontweight="bold")
    for i, (_, r) in enumerate(w.iterrows()):
        y = 7.6 - i * 0.3
        pl.text(10.6, y, _label(r)[:34], fontsize=8.5); pl.text(13.2, y, _fmt(r.height, 1), fontsize=8.5); pl.text(14.1, y, _fmt(r.height_max_m, 1), fontsize=8.5)
    notes = ["Possible reasons, none checked on the ground: the footprints come from a 2026 release and the lidar is from April 2021, so a building finished or demolished since would disagree; Overture's height source may differ from a lidar surface; a footprint can cover a podium and a tower; and trees touching a roof raise the highest return.",
             "The model uses the 70th percentile of the roof surface, which sits below the highest return on buildings with rooftop equipment, so it reads lower than Overture on average.",
             f"Of {n_all} footprints in the area, {int((B.status == 'ok').sum())} were measured. {int((ok.height_m < 3).sum())} of those stand under 3 m in the 2021 lidar (a few are low structures, others may not exist in 2021). Footprints under 15 m² and ones without lidar returns or ground nearby were skipped."]
    y = 3.2
    for t in notes:
        import textwrap
        w_ = "\n".join(textwrap.wrap(t, 150))
        pl.text(0.9, y, w_, fontsize=8.5, color=MUTED, linespacing=1.35)
        y -= 0.2 * (w_.count("\n") + 1) + 0.22
    pl.save("11_seattle_height_check.png")
    return res


# --------------------------------------------------------------------------------------------- 12 imagery draped on the models
def _sample_textured(surf, walls, region, n, seed=7):
    """Points on the textured surface and walls inside region (xmin, xmax, ymin, ymax in local metres), colours shaded by a fixed light.
    Triangles are chosen by overlap with the region (big roof triangles reach far outside it) and the sampled points are then clipped to it."""
    rng = np.random.default_rng(seed)
    L = np.array([0.4, -0.5, 0.75]); L /= np.linalg.norm(L)
    sel = []
    for m, textured in ((surf, True), (walls, False)):
        tri = m.triangles
        lo, hi = tri.min(1), tri.max(1)
        ov = (hi[:, 0] > region[0]) & (lo[:, 0] < region[1]) & (hi[:, 1] > region[2]) & (lo[:, 1] < region[3])
        sel.append(ov * m.area_faces)
    density = n / max(sum(w.sum() for w in sel), 1e-9)
    pts, cols = [], []
    for (m, textured), w in zip(((surf, True), (walls, False)), sel):
        k = int(density * w.sum())
        if k < 10:
            continue
        if textured:
            p, f, col = trimesh.sample.sample_surface(m, k, face_weight=w, sample_color=True, seed=int(rng.integers(1 << 30)))
            col = np.asarray(col)[:, :3].astype(float)
        else:
            p, f = trimesh.sample.sample_surface(m, k, face_weight=w, seed=int(rng.integers(1 << 30)))
            col = m.visual.face_colors[f][:, :3].astype(float)
        sh = 0.62 + 0.38 * np.clip(m.face_normals[f] @ L, 0, 1)
        inside = (p[:, 0] > region[0]) & (p[:, 0] < region[1]) & (p[:, 1] > region[2]) & (p[:, 1] < region[3])
        pts.append(p[inside]); cols.append(np.clip(col * sh[:, None], 0, 255).astype(np.uint8)[inside])
    return np.vstack(pts), np.vstack(cols)


def fig_naip(P, B, raw, fixed, tf, model, evidence, surf, walls, year, dates, parts_fn=None):
    """raw / fixed: (4, h, w) NAIP before / after the lidar-based correction."""
    pl = Plate("High-resolution aerial imagery on the lidar buildings", f"NAIP aerial photography ({dates}, 0.6 m pixels) corrected with the lidar surface and draped on the building model; roofs are photographic, walls are plain because no photo shows them",
               SRC.format(d0=P["d0"]) + " Imagery: USDA National Agriculture Imagery Program (NAIP), via the Microsoft Planetary Computer.", "Photogrammetry was not possible: single downward photos only")
    ex = (tf.c, tf.c + tf.a * raw.shape[2], tf.f - tf.a * raw.shape[1], tf.f)
    mrow = B[B["name"] == "MoPOP: Museum of Pop Culture"].iloc[0]                 # shown on plate 14; kept out of this plate's windows
    mx0, my0 = mrow.geometry.bounds[0] - P["cx"], mrow.geometry.bounds[1] - P["cy"]
    nx, ny = 129.0, -109.0                                                         # the Space Needle, metres from the area centre
    half = 100.0
    cx, cy = P["cx"] + nx - 40.0, P["cy"] + ny - 50.0
    assert cy + half < P["cy"] + my0 - 5, "photo crop would include MoPOP"
    for k, (img, ttl) in enumerate(((raw, "As photographed"), (fixed, "Corrected with lidar heights"))):
        a = pl.ax(0.6 + k * 4.2, 4.35, 3.9, 3.9)
        a.imshow(np.moveaxis(img[:3], 0, -1), extent=ex)
        for _, r in B[B.status == "ok"].iterrows():
            gs = list(r.geometry.geoms) if isinstance(r.geometry, MultiPolygon) else [r.geometry]
            for g in gs:
                xs, ys = g.exterior.xy
                a.plot(xs, ys, color="#FFD400", lw=0.9)
        a.set_xlim(cx - half, cx + half); a.set_ylim(cy - half, cy + half); a.set_xticks([]); a.set_yticks([])
        pl.label(0.6 + k * 4.2, 8.35, ttl)
    ev = evidence
    notes = ["Yellow lines are the lidar footprints. A downward aerial photo is flattened onto the bare ground, so the top of a tall building appears displaced away from the point under the camera: the Space Needle's saucer sits about 65 m west of its own footprint on the left.",
             f"The correction shifts each roof, as one piece so the photo is not warped, by an amount proportional to its lidar height above ground; ground and trees use per-cell lidar heights, and strongly curved roofs (see plate 14) use smoothed per-cell heights. The shift is fitted to {model['n']} tall buildings. After it, the roofs move onto their footprints.",
             f"Check: across {ev['n']} tall footprints the median distance between each outline and its best image match fell from {ev['offset_before_m']:.1f} m to {ev['offset_after_m']:.1f} m, and the match score rose for {ev['improved_share'] * 100:.0f}% of them. It is not perfect: the fit leaves about {model['rms_after']:.0f} m typical error on some buildings, and surfaces hidden behind tall buildings in the photo cannot be recovered.",
             "In the two 3D views, buildings cut by the edge of the view are left undrawn and show as flat photo on the ground, and the museum is kept out of both (plate 14)."]
    import textwrap
    y = 4.05
    for t in notes:
        w_ = "\n".join(textwrap.wrap(t, 90))
        pl.text(0.6, y, w_, fontsize=8.5, color=MUTED, linespacing=1.35)
        y -= 0.19 * (w_.count("\n") + 1) + 0.2
    size = P["w"]
    west = (-size / 2, mx0 - 10.0, -size / 2, size / 2)                            # everything west of the museum
    zoom = (nx - 110, nx + 110, ny - 95, my0 - 10.0)                               # around the Needle, stopping before the museum
    for k, (reg, ttl, n_pts, b0) in enumerate(((west, "Western two thirds of the area (the museum is on plate 14)", 5_000_000, 4.75), (zoom, "Close up around the Space Needle (shape fitted to its lidar returns)", 4_000_000, 0.95))):
        if parts_fn is not None:                                                   # buildings straddling this view's edge are left out so no sliver of wall shows
            from shapely.geometry import box as _box
            win = _box(reg[0], reg[2], reg[1], reg[3])
            cut = [i for i, g in zip(B.index, [translate(q, -P["cx"], -P["cy"]) for q in B.geometry]) if g.intersects(win.boundary) and B.loc[i, "status"] == "ok" and B.loc[i, "name"] != "Space Needle"]
            surf_w, walls_w = parts_fn(cut)
        else:
            surf_w, walls_w = surf, walls
        p_, c_ = _sample_textured(surf_w, walls_w, reg, n_pts)
        img, _ = render3d.render(p_, c_, az=215, el=42, width=1500, height=800, zexag=1.0, center=p_.mean(0), splat=2)
        a = pl.ax(9.0, b0, 6.4, 3.41); a.imshow(img); a.axis("off")
        pl.label(9.0, b0 + 3.55, ttl)
    return pl.save("12_seattle_imagery.png")


# --------------------------------------------------------------------------------------------- 13 the Space Needle
def fig_needle(P, pts, prof, quality, sym_saucer, sym_legs, parts_old, parts_new, axis_local, vol_old, vol_new, d0):
    import textwrap
    pl = Plate("The Space Needle: fitted to its lidar returns", "Lidar returns and the fitted outline (left); the tower as a fitted shape of revolution and as the extruded footprint it replaces (right)",
               SRC.format(d0=d0) + " Imagery: USDA NAIP 2019, via the Microsoft Planetary Computer.", "Camera and scale identical in both 3D views")
    r = np.hypot(pts[:, 0], pts[:, 1])
    keep = r < 32
    a = pl.ax(0.9, 2.45, 1.9, 5.5)
    a.scatter(pts[keep, 0], pts[keep, 2], s=0.3, c="#555555", alpha=0.35, lw=0, rasterized=True)
    z, rr, it = prof["z"], prof["r"], prof["interpolated"]
    for sgn in (-1, 1):
        for k in range(len(z) - 1):
            a.plot([sgn * rr[k], sgn * rr[k + 1]], [z[k], z[k + 1]], color="#C0392B", lw=1.3, ls="--" if (it[k] and it[k + 1]) else "-")
    a.set_xlim(-32, 32); a.set_ylim(0, 190); a.set_aspect("equal"); a.tick_params(labelsize=7.5, length=2); a.spines[["top", "right"]].set_visible(False)
    a.set_xlabel("m from the axis", fontsize=8, color=MUTED); a.set_ylabel("height above ground, m", fontsize=8, color=MUTED)
    pl.label(0.9, 8.3, "Lidar returns and fitted outline")
    region = (axis_local[0] - 48, axis_local[0] + 48, axis_local[1] - 48, axis_local[1] + 48)
    info = None
    for k, ((surf, walls), ttl, x0) in enumerate(((parts_new, "After: fitted to the lidar", 4.1), (parts_old, "Before: footprint extruded to the main roof", 9.4))):
        p_, c_ = _sample_textured(surf, walls, region, 5_000_000)
        if info is None:
            cen = p_.mean(0)
        img, inf = render3d.render(p_, c_, az=215, el=14, width=1000, height=1200, zexag=1.0, center=cen, scale=info["scale"] if info else None, off=info["off"] if info else None, splat=2)
        info = info or inf
        ax_ = pl.ax(x0, 2.3, 4.5, 5.4); ax_.imshow(img); ax_.axis("off")
        pl.label(x0, 8.3, ttl)
    inter = quality["interpolated_height_m"]
    notes = [f"The tower is nearly symmetric about its axis (saucer radius {sym_saucer[0]:.1f} to {sym_saucer[1]:.1f} m by compass sector, legs {sym_legs[0]:.1f} to {sym_legs[1]:.1f} m), so the outline, the 95th-percentile radius of the returns in each 2 m of height, is revolved around the axis. {quality['share_inside'] * 100:.0f}% of the {quality['points']:,} returns lie inside it (+1.5 m).",
             f"About {inter:.0f} m of the height has almost no returns (the open steel between the shaft and the saucer) and is interpolated (dashed); the spire above the top house is a thin cylinder set by a few returns. Highest return {quality['top_m']:.1f} m (Overture: 184 m). Volume {vol_new / 1000:.0f} thousand m³ against {vol_old / 1000:.0f} thousand for the extruded footprint.",
             "This is the lidar silhouette, so the shaft is solid and the legs, glass and spire detail are not modelled; side faces are plain because no photograph shows them."]
    y = 2.05
    for t in notes:
        w_ = "\n".join(textwrap.wrap(t, 185))
        pl.text(0.9, y, w_, fontsize=8, color=MUTED, linespacing=1.3)
        y -= 0.17 * (w_.count("\n") + 1) + 0.08
    return pl.save("13_space_needle.png")


# --------------------------------------------------------------------------------------------- 14 a curved roof: MoPOP
def fig_curved_roof(P, B, raw, fixed, tf, parts_flat, parts_real, hole_share, n_curved, name="MoPOP: Museum of Pop Culture"):
    import textwrap
    g = B[B["name"] == name].iloc[0]
    pl = Plate("A curved roof: the Museum of Pop Culture", "The red, blue, silver and gold roof is real; the first drape of it on a flat block looked distorted, so curved roofs now follow their lidar surface",
               SRC.format(d0=P["d0"]) + " Imagery: USDA NAIP 2019, via the Microsoft Planetary Computer.", "Same camera in both 3D views")
    x0, y0, x1, y1 = g.geometry.bounds
    m = 25.0
    ex = (tf.c, tf.c + tf.a * raw.shape[2], tf.f - tf.a * raw.shape[1], tf.f)
    for k, (img, ttl) in enumerate(((raw, "Photograph as received"), (fixed, "Corrected with lidar heights"))):
        a = pl.ax(0.9 + k * 3.3, 4.55, 3.0, 3.6)
        a.imshow(np.moveaxis(img[:3], 0, -1), extent=ex)
        gs = list(g.geometry.geoms) if isinstance(g.geometry, MultiPolygon) else [g.geometry]
        for pg in gs:
            xs, ys = pg.exterior.xy
            a.plot(xs, ys, color="#FFD400", lw=0.9)
        a.set_xlim(x0 - m, x1 + m); a.set_ylim(y0 - m, y1 + m); a.set_xticks([]); a.set_yticks([])
        pl.label(0.9 + k * 3.3, 8.35, ttl)
    cx, cy = g.geometry.centroid.x - P["cx"], g.geometry.centroid.y - P["cy"]
    region = (cx - 50, cx + 50, cy - 70, cy + 70)
    info = None
    for k, ((surf, walls), ttl) in enumerate(((parts_flat, "Photo on a flat roof (first version)"), (parts_real, "Photo on the lidar roof surface"))):
        p_, c_ = _sample_textured(surf, walls, region, 4_000_000)
        if info is None:
            cen = p_.mean(0)
        img, inf = render3d.render(p_, c_, az=215, el=38, width=1000, height=720, zexag=1.0, center=cen, scale=info["scale"] if info else None, off=info["off"] if info else None, splat=2)
        info = info or inf
        a = pl.ax(0.9 + k * 4.7, 0.95, 4.5, 3.24); a.imshow(img); a.axis("off")
        pl.label(0.9 + k * 4.7, 4.35, ttl)
    notes = ["The red is not an error: it is in the unprocessed photograph, top left. The museum's roof is curved metal in several colours.",
             f"Two things made the first version look distorted. A photo of a curved roof was stretched over a flat block, and the lidar has no returns for {hole_share * 100:.0f}% of this roof's cells (glossy metal reflects the laser away), so the height-based shift of the image was wrong and tore it.",
             f"Now, for the {n_curved} buildings over 150 m² whose roofs vary by more than 6 m, dropouts take the nearest real roof return, the heights are smoothed, and the roof is built as the lidar surface (about 1.2 m triangles) so the photo lies on the shape it was taken from. Other roofs are shifted as one rigid piece.",
             "Limits: the roof is a smoothed approximation, not a survey of the panels; the sides of curved roofs take the roof edge's colour from the photo, darkened, because no photo shows them; parts of the roof hidden from the camera in the photo are filled from neighbouring pixels."]
    y = 7.9
    for t in notes:
        w_ = "\n".join(textwrap.wrap(t, 62))
        pl.text(10.3, y, w_, fontsize=8.8, color=MUTED, linespacing=1.4)
        y -= 0.2 * (w_.count("\n") + 1) + 0.3
    return pl.save("14_mopop_curved_roof.png")
