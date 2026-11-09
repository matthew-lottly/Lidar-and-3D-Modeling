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


def render(layers, focus, dist, az, el, size=(1700, 1000), sky=("#dbe9f5", "#8fb4d9"), zoom=1.0, light_az=300.0, light_el=45.0):
    """layers: list of dicts(mesh=..., kind='mesh'|'points'|'tube', color/rgb, opacity, point_size). Camera looks at `focus` from compass bearing `az` (0 = camera north of the target, 90 = east) at elevation `el` degrees and distance `dist` m."""
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background(sky[0], top=sky[1])
    for L in layers:
        kind = L.get("kind", "mesh")
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
    img = pl.screenshot(return_img=True)
    pl.close()
    return img
