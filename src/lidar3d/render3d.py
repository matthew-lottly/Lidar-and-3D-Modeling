"""Small software renderer for point clouds and terrain grids: orthographic projection with a nearest-point z-buffer. No GPU or OpenGL needed."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def rotation(az_deg: float, el_deg: float) -> np.ndarray:
    """World (x east, y north, z up) -> view. az = compass direction the camera looks FROM (0 = looking north from the south)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.sin(el), np.cos(el)], [0, -np.cos(el), np.sin(el)]])
    return Rx @ Rz


def render(pts: np.ndarray, colors: np.ndarray, az=200.0, el=35.0, width=1600, height=900, zexag=1.0, center=None, scale=None, bg=(255, 255, 255), splat=2) -> tuple[np.ndarray, dict]:
    """pts: (n,3) in metres, colors: (n,3) uint8. Returns an RGB image and the projection info (for placing annotations)."""
    c = pts.mean(0) if center is None else np.asarray(center, float)
    p = (pts - c) * np.array([1, 1, zexag])
    v = p @ rotation(az, el).T                         # v[:,0] right, v[:,1] up (view), v[:,2] depth (toward viewer negative)
    off = np.zeros(2)
    if scale is None:                                   # fit the whole cloud in the frame and centre it
        span_x, span_y = np.ptp(v[:, 0]), np.ptp(v[:, 1])
        scale = 0.94 * min(width / span_x, height / span_y)
        off = np.array([(v[:, 0].max() + v[:, 0].min()) / 2, (v[:, 1].max() + v[:, 1].min()) / 2])
    px = ((v[:, 0] - off[0]) * scale + width / 2).astype(int)
    py = (height / 2 - (v[:, 1] - off[1]) * scale).astype(int)
    ok = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py, d, col = px[ok], py[ok], v[ok, 2], colors[ok]
    key = py * width + px
    order = np.lexsort((d, key))                        # per pixel, smallest depth value first = nearest to the viewer (depth axis points away from viewer)
    key_s = key[order]
    first = np.r_[True, key_s[1:] != key_s[:-1]]
    sel = order[first]
    img = np.empty((height * width, 3), np.uint8)
    img[:] = bg
    img[key[sel]] = col[sel]
    img = img.reshape(height, width, 3)
    if splat > 1:                                       # fill pixel gaps between sparse points with the nearest rendered colour
        filled = np.zeros((height, width), bool)
        filled.flat[key[sel]] = True
        dist = ndimage.distance_transform_edt(~filled, return_distances=True, return_indices=True)
        idx = dist[1]
        near = img[idx[0], idx[1]]
        use = (~filled) & (dist[0] <= splat)
        img = np.where(use[..., None], near, img)
    return img, dict(center=c, scale=scale, width=width, height=height, az=az, el=el, zexag=zexag, off=off)


def project(xyz, info) -> np.ndarray:
    p = (np.atleast_2d(xyz) - info["center"]) * np.array([1, 1, info["zexag"]])
    v = p @ rotation(info["az"], info["el"]).T
    o = info.get("off", np.zeros(2))
    return np.c_[(v[:, 0] - o[0]) * info["scale"] + info["width"] / 2, info["height"] / 2 - (v[:, 1] - o[1]) * info["scale"]]


def terrain_points(z: np.ndarray, x0: float, y1: float, res: float, rgb: np.ndarray, step: int = 1):
    """Turn a DEM grid + RGB texture (rows, cols, 3) into (n,3) points and colours."""
    r, c = np.mgrid[0:z.shape[0]:step, 0:z.shape[1]:step]
    pts = np.c_[x0 + (c.ravel() + 0.5) * res, y1 - (r.ravel() + 0.5) * res, z[::step, ::step].ravel()]
    return pts, rgb[::step, ::step].reshape(-1, 3)


def shade_texture(z: np.ndarray, res: float, base_rgb: np.ndarray, azimuth=315.0, altitude=40.0, strength=0.7) -> np.ndarray:
    """Multiply a base colour image by a hillshade so relief reads in 3D."""
    from .cloud import hillshade
    hs = hillshade(ndimage.gaussian_filter(z, 1.0), res, azimuth, altitude)
    k = (1 - strength) + strength * hs[..., None] * 1.25
    return np.clip(base_rgb.astype(float) * k, 0, 255).astype(np.uint8)
