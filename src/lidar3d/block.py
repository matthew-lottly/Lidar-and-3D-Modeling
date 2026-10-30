"""Run the rooftop solar pipeline for any bounding box (used by the notebook): choose a lidar project, build the buildings, roof planes and irradiance, and write a 3D mesh (.glb and .obj) coloured by energy."""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer

from . import forestry as FO
from . import rooftop as RT
from . import roofsolar as RS

BASE = "https://usgs-lidar-public.s3.amazonaws.com"


def find_projects(lon, lat, log=print):
    """3DEP lidar projects whose bounds contain the point, newest first by the year in the name (reads the bounds of every project's ept.json: about 2,300 small requests)."""
    names, tok = [], None
    while True:
        p = {"list-type": 2, "delimiter": "/"}
        if tok:
            p["continuation-token"] = tok
        t = requests.get(BASE + "/", params=p, timeout=60).text
        names += re.findall(r"<Prefix>([^<]*)/</Prefix>", t)
        m = re.search(r"<NextContinuationToken>([^<]*)", t)
        if not m:
            break
        tok = m.group(1)
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)

    def get(n):
        try:
            b = requests.get(f"{BASE}/{n}/ept.json", timeout=30).json()["boundsConforming"]
            return n if b[0] <= x <= b[3] and b[1] <= y <= b[4] else None
        except Exception:
            return None
    with ThreadPoolExecutor(32) as ex:
        hit = [n for n in ex.map(get, names) if n]
    yr = lambda n: max([int(v) for v in re.findall(r"(?:19|20)\d\d", n)] or [0])
    return sorted(hit, key=yr, reverse=True)


def site_from_bbox(west, south, east, north, project, context=300.0, max_half=300.0):
    """Square analysis area around the centre of a lon/lat box (half-width = half the longer side, at most `max_half` m), in the UTM zone of the centre."""
    lon, lat = (west + east) / 2, (south + north) / 2
    zone = int((lon + 180) // 6) + 1
    crs = f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    (x0, y0), (x1, y1) = tr.transform(west, south), tr.transform(east, north)
    half = float(min(max(abs(x1 - x0), abs(y1 - y0)) / 2, max_half))
    slug = "bbox_" + hashlib.md5(f"{west:.5f},{south:.5f},{east:.5f},{north:.5f},{project}".encode()).hexdigest()[:8]
    return dict(slug=slug, name=f"Block at {lat:.4f} N, {abs(lon):.4f} {'W' if lon < 0 else 'E'}", lonlat=(lon, lat), half=round(half), context=context, project=project, crs=crs)


def write_mesh(R, outdir, kwh_range=(600.0, 1500.0)):
    """.glb and .obj of the block: terrain plus every building (roof planes coloured by annual kWh/m2, grey walls). Returns the paths."""
    import matplotlib.cm as cm
    import trimesh
    from matplotlib.colors import Normalize
    cells = R["cells"]
    pm = cells.groupby(["bid", "plane"]).apply(lambda d: float((d.total * d.area).sum() / d.area.sum()), include_groups=False)
    norm = Normalize(*kwh_range)
    z_ref = float(np.nanmedian(R["dtm"]))
    verts, faces, cols, off = [], [], [], 0
    for b in R["B"]:
        def col(j, b=b):
            return tuple(int(t) for t in (np.array(cm.inferno(norm(pm[(b.bid, j)]))[:3]) * 255)) if (b.bid, j) in pm.index else (150, 150, 150)
        v, f, c = RT.building_mesh(b, col)
        if len(v):
            verts.append(v); faces.append(f + off); cols.append(c); off += len(v)
    V = np.vstack(verts)
    V[:, 2] -= z_ref
    bm = trimesh.Trimesh(V, np.vstack(faces), vertex_colors=np.c_[np.vstack(cols), np.full(len(V), 255)], process=False)
    h, half = R["site"]["half"], R["C"].half
    g = np.arange(-h, h + 1e-6, 2.0)
    GX, GY = np.meshgrid(g, g)
    gz = FO.sample(R["dtm"], half, 1.0, GX.ravel(), GY.ravel()).reshape(GX.shape) - z_ref
    idx = np.arange(GX.size).reshape(GX.shape)
    a, b_, c_, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    gm = trimesh.Trimesh(np.c_[GX.ravel(), GY.ravel(), np.nan_to_num(gz.ravel())], np.r_[np.c_[a, c_, b_], np.c_[b_, c_, d]], vertex_colors=np.tile([196, 202, 190, 255], (GX.size, 1)), process=False)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    trimesh.Scene([gm, bm]).export(outdir / "block.glb")
    trimesh.util.concatenate([gm, bm]).export(outdir / "block.obj", include_color=True)
    return outdir / "block.glb", outdir / "block.obj", gm, bm


def run_block(west, south, east, north, outdir, project=None, log=print):
    """Whole workflow for a lon/lat box. Returns (R, T, paths): the analysis dict, the per-building table, and the written files."""
    lon, lat = (west + east) / 2, (south + north) / 2
    if project is None:
        found = find_projects(lon, lat, log)
        if not found:
            raise RuntimeError("no 3DEP lidar project covers this box")
        log("lidar projects covering the box, newest first: " + ", ".join(found[:6]))
        project = found[0]
    site = site_from_bbox(west, south, east, north, project)
    R = RS.analyse(site, log=log)
    T = RS.building_table(R)
    glb, obj, _, _ = write_mesh(R, outdir)
    T.round(3).to_csv(Path(outdir) / "buildings.csv", index=False)
    (Path(outdir) / "run.json").write_text(json.dumps(dict(site=site, buildings=len(T), roof_area_m2=float(T.roof_area_m2.sum()), total_gwh=float(T.total_kwh_yr.sum() / 1e6)), indent=1))
    return R, T, dict(glb=glb, obj=obj, csv=Path(outdir) / "buildings.csv")
