"""Offscreen 3D rendering with PyVista/VTK (run under xvfb-run on a machine with no display): a textured ground mesh, lit meshes for supports and equipment, wires as tubes and lidar returns as coloured points."""
from __future__ import annotations

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True


def ground_mesh(G, rgb, step=1):
    """Textured terrain: G from `unified.ground_surface`, rgb (ny, nx, 3) uint8 on the same grid. Returns a PyVista grid with an 'rgb' array."""
    z = G["z"][::step, ::step]
    ny, nx = z.shape
    X = (np.arange(nx) * G["res"] * step + G["x0"] + G["res"] * step / 2)
    Y = (np.arange(ny) * G["res"] * step + G["y0"] + G["res"] * step / 2)
    XX, YY = np.meshgrid(X, Y)
    grid = pv.StructuredGrid(XX, YY, z)
    sub = rgb[::step, ::step]
    grid["rgb"] = np.stack([sub[..., k].ravel(order="F") for k in range(3)], 1).astype(np.uint8)
    return grid


def tmesh(m):
    """A trimesh with face colours as PyVista polydata."""
    f = np.c_[np.full(len(m.faces), 3), m.faces].ravel()
    p = pv.PolyData(np.asarray(m.vertices), f)
    p["rgb"] = np.asarray(m.visual.face_colors)[:, :3].astype(np.uint8)
    return p


def wire_tube(xyz, radius=0.07, sides=5):
    xyz = np.asarray(xyz, float)
    if len(xyz) < 2:
        return None
    return pv.lines_from_points(xyz).tube(radius=radius, n_sides=sides)


def lines_mesh(polys):
    """One polydata holding many polylines ((n, 3) arrays), for screen-space line widths."""
    polys = [np.asarray(p, float) for p in polys if len(p) >= 2]
    if not polys:
        return None
    pts = np.vstack(polys)
    cells, off = [], 0
    for q in polys:
        cells.append(np.r_[len(q), np.arange(off, off + len(q))])
        off += len(q)
    m = pv.PolyData(pts)
    m.lines = np.concatenate(cells)
    return m


def canopy_mesh(z_top, z_ground, mask, rgb, x0, y0, res, step=1):
    """The tree canopy as a surface built from the canopy height model: cells with canopy follow the top of the returns, every other vertex is pushed just under the ground so the crowns stand as solids. rgb (ny, nx, 3) uint8."""
    z = np.where(mask, z_top, z_ground - 0.15)[::step, ::step]
    ny, nx = z.shape
    X = x0 + (np.arange(nx) * step + 0.5) * res
    Y = y0 + (np.arange(ny) * step + 0.5) * res
    XX, YY = np.meshgrid(X, Y)
    g = pv.StructuredGrid(XX, YY, z)
    sub = rgb[::step, ::step]
    g["rgb"] = np.stack([sub[..., k].ravel(order="F") for k in range(3)], 1).astype(np.uint8)
    return g


def render(layers, focus, dist, az, el, size=(1700, 1000), sky=("#dbe9f5", "#8fb4d9"), zoom=1.0, light_az=300.0, light_el=45.0, ssao=False, shadows=False, bg=None):
    """layers: list of dicts(mesh=..., kind='mesh'|'points'|'tube', color/rgb, opacity, point_size). Camera looks at `focus` from compass bearing `az` (0 = camera north of the target, 90 = east) at elevation `el` degrees and distance `dist` m."""
    pl = pv.Plotter(off_screen=True, window_size=size)
    if bg is not None:
        pl.set_background(bg)
    else:
        pl.set_background(sky[0], top=sky[1])
    for L in layers:
        kind = L.get("kind", "mesh")
        if kind == "lines":
            pl.add_mesh(L["mesh"], color=L.get("color", "black"), line_width=L.get("width", 2.0), render_lines_as_tubes=True, lighting=False)
            continue
        if kind == "points":
            pl.add_mesh(L["mesh"], scalars="rgb", rgb=True, point_size=L.get("point_size", 3.0), render_points_as_spheres=False, lighting=False)
        elif L.get("rgb", True) and "rgb" in L["mesh"].array_names:
            pl.add_mesh(L["mesh"], scalars="rgb", rgb=True, opacity=L.get("opacity", 1.0), smooth_shading=L.get("smooth", False), ambient=0.35, diffuse=0.75, specular=0.05)
        else:
            pl.add_mesh(L["mesh"], color=L.get("color", "gray"), opacity=L.get("opacity", 1.0), smooth_shading=True, ambient=0.3, diffuse=0.8, specular=0.1)
    a, e = np.radians(az), np.radians(el)
    fp = np.asarray(focus, float)
    pos = fp + dist * np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    pl.camera_position = [tuple(pos), tuple(fp), (0, 0, 1)]
    pl.camera.zoom(zoom)
    pl.remove_all_lights()
    la, le = np.radians(light_az), np.radians(light_el)
    pl.add_light(pv.Light(position=tuple(fp + 500 * np.array([np.sin(la) * np.cos(le), np.cos(la) * np.cos(le), np.sin(le)])), focal_point=tuple(fp), intensity=0.9, light_type="scene light"))
    pl.add_light(pv.Light(position=tuple(pos), focal_point=tuple(fp), intensity=0.35, light_type="scene light"))
    pl.enable_anti_aliasing("ssaa")
    if ssao:
        try:
            pl.enable_ssao(radius=max(dist * 0.01, 1.0), bias=max(dist * 0.0004, 0.02), kernel_size=128, blur=True)
        except Exception:
            pass
    if shadows:
        try:
            pl.enable_shadows()
        except Exception:
            pass
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def crowns_mesh(chm, lab, ids, tops_xy, bases, ground, x0, y0, res, base_rgb, n_az=20, n_lev=10, seed=0):
    """One 3D crown per segmented tree, shaped by the lidar: for each tree the canopy-height cells of its watershed crown give, at each of `n_lev` heights between
    the crown base and the top and in each of `n_az` directions from the tree top, the furthest cell still at least that tall. That radius envelope is the crown
    (so a cone-shaped conifer, a broad round crown and a crown pressed against its neighbour each keep their own shape). A trunk joins the crown base to the ground.
    chm, lab, ground: grids on the same 0.5 m lattice; base_rgb: (n, 3) colour per tree. Returns (crowns polydata, trunks polydata), each with an 'rgb' array."""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    sm = ndi.gaussian_filter(np.where(np.isfinite(chm), chm, 0.0), 0.6)
    objs = ndi.find_objects(lab)
    V, F, C, off = [], [], [], 0
    TV, TF, off_t = [], [], 0
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    ca, sa = np.cos(az), np.sin(az)
    for t, tid in enumerate(ids):
        if tid - 1 >= len(objs) or objs[tid - 1] is None:
            continue
        sl = objs[tid - 1]
        m = lab[sl] == tid
        if m.sum() < 3:
            continue
        rr, cc = np.nonzero(m)
        px = x0 + (cc + sl[1].start + 0.5) * res
        py = y0 + (rr + sl[0].start + 0.5) * res
        h = sm[sl][m]
        tx, ty = tops_xy[t]
        ti, tj = int((ty - y0) / res), int((tx - x0) / res)
        g = float(ground[np.clip(ti, 0, ground.shape[0] - 1), np.clip(tj, 0, ground.shape[1] - 1)])
        top = float(h.max())
        base = float(np.clip(bases[t], 0.5, top * 0.75))
        dx, dy = px - tx, py - ty
        dist = np.hypot(dx, dy) + res * 0.5
        sec = (np.round(np.arctan2(dy, dx) / (2 * np.pi / n_az)).astype(int)) % n_az
        levs = base + (top - base) * np.linspace(0, 1, n_lev + 1)[:-1]
        R = np.zeros((n_lev, n_az))
        for li, L in enumerate(levs):
            ok = h >= L
            if ok.any():
                np.maximum.at(R[li], sec[ok], dist[ok])
        # empty sectors take the circular mean of their neighbours; smooth across direction and height; a crown never grows wider upward than below by much
        for li in range(n_lev):
            r = R[li]
            if (r > 0).any():
                fill = np.where(r > 0, r, np.nan)
                idx = np.arange(n_az)
                good = ~np.isnan(fill)
                fill = np.interp(idx, np.r_[idx[good] - n_az, idx[good], idx[good] + n_az], np.r_[fill[good], fill[good], fill[good]])
                R[li] = fill
            else:
                R[li] = R[li - 1] * 0.6 if li else res
        R = (R + np.roll(R, 1, 1) + np.roll(R, -1, 1)) / 3
        R = ndi.uniform_filter1d(R, 3, axis=0, mode="nearest")
        # foliage hangs: below the widest level of the upper crown the radius does not shrink by more than 15% per level (so crowns taper upward, as they do)
        for li in range(n_lev - 2, -1, -1):
            R[li] = np.maximum(R[li], R[li + 1] * 0.92)
        R = np.maximum(R, res * 0.6)
        R *= 1.0 + 0.05 * rng.standard_normal(R.shape).clip(-2, 2)
        # rings: a small inward-curving underside, the levels, and an apex
        zz = g + levs
        rings = [(g + base - 0.25, R[0] * 0.55)] + [(zz[i], R[i]) for i in range(n_lev)]
        vs = [np.c_[tx + r * ca, ty + r * sa, np.full(n_az, z)] for z, r in rings]
        vs.append(np.array([[tx, ty, g + top]]))
        vs.insert(0, np.array([[tx, ty, g + base - 0.35]]))
        P = np.vstack(vs)
        nr = len(rings)
        f = []
        # bottom fan
        for k in range(n_az):
            f.append((0, 1 + k, 1 + (k + 1) % n_az))
        for ri in range(nr - 1):
            a0, b0 = 1 + ri * n_az, 1 + (ri + 1) * n_az
            for k in range(n_az):
                k1 = (k + 1) % n_az
                f += [(a0 + k, b0 + k, b0 + k1), (a0 + k, b0 + k1, a0 + k1)]
        apex = len(P) - 1
        a0 = 1 + (nr - 1) * n_az
        for k in range(n_az):
            f.append((a0 + k, apex, a0 + (k + 1) % n_az))
        f = np.array(f) + off
        V.append(P); F.append(f); off += len(P)
        # colour: the tree's own colour, darker low and inside, lit at the top, with foliage noise
        frac = np.clip((P[:, 2] - g - base) / max(top - base, 0.5), 0, 1)
        shade = 0.62 + 0.5 * frac + 0.07 * rng.standard_normal(len(P))
        C.append(np.clip(base_rgb[t][None, :] * shade[:, None], 0, 255))
        # trunk
        rt = 0.08 + 0.012 * top
        k6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ring_b = np.c_[tx + rt * np.cos(k6), ty + rt * np.sin(k6), np.full(6, g - 0.2)]
        ring_t = np.c_[tx + rt * 0.7 * np.cos(k6), ty + rt * 0.7 * np.sin(k6), np.full(6, g + base + 0.3)]
        TV.append(np.vstack([ring_b, ring_t]))
        tf = []
        for k in range(6):
            k1 = (k + 1) % 6
            tf += [(k, 6 + k, 6 + k1), (k, 6 + k1, k1)]
        TF.append(np.array(tf) + off_t); off_t += 12
    def poly(V, F):
        F = np.vstack(F)
        return pv.PolyData(np.vstack(V), np.c_[np.full(len(F), 3), F].ravel())
    cr = poly(V, F)
    cr["rgb"] = np.vstack(C).astype(np.uint8)
    cr = cr.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=False, split_vertices=False)
    tr = poly(TV, TF)
    tr["rgb"] = np.tile(np.array([92, 72, 54], np.uint8), (tr.n_points, 1))
    return cr, tr
