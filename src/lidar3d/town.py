"""Tiled utility extraction for an area too big to hold in memory. The area is cut into square tiles; each tile is read with a buffer so shapes and heights near its edges see their neighbours, and only the points inside the
tile itself are kept, so merging the tiles never duplicates a detection. Detectors are the ones of `utility`; chains are rebuilt across tile edges afterwards."""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import numpy as np
from pyproj import Transformer

from . import spans, utility
from .ept import CACHE, EPT

BUFFER = 40.0


def tile_grid(half: float, tile: float):
    """Tiles as (i, j, x0, y0, x1, y1) in metres relative to the centre, covering [-half, half]."""
    n = int(np.ceil(2 * half / tile))
    return [(i, j, -half + i * tile, -half + j * tile, min(-half + (i + 1) * tile, half), min(-half + (j + 1) * tile, half)) for j in range(n) for i in range(n)]


def process_tile(project, lonlat, crs, tile, out_dir: Path, log=lambda s: None, max_cache_gb=50.0):
    """Detect wires, poles and elevated non-mass returns in one tile and save the part that lies inside it. Returns the output path."""
    i, j, x0, y0, x1, y1 = tile
    out = out_dir / f"tile_{i:02d}_{j:02d}.npz"
    if out.exists():
        return out
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    cx, cy = tr.transform(*lonlat)
    back = Transformer.from_crs(crs, 4326, always_xy=True)
    W, S = back.transform(cx + x0 - BUFFER, cy + y0 - BUFFER)
    E, N = back.transform(cx + x1 + BUFFER, cy + y1 + BUFFER)
    c = EPT(project).fetch((W, S, E, N), max_depth=20, log=log)
    keep = (c.x >= cx + x0 - BUFFER) & (c.x < cx + x1 + BUFFER) & (c.y >= cy + y0 - BUFFER) & (c.y < cy + y1 + BUFFER) & (c.cls != 7)
    g = c.gps[c.gps > 0]
    v = float(np.median(g)) if len(g) else 0.0
    v += 1e9 if 0 < v < 1e9 else 0.0
    date = str((dt.datetime(1980, 1, 6) + dt.timedelta(seconds=v)).date()) if v else ""
    x, y, z, cls = c.x[keep] - cx, c.y[keep] - cy, c.z[keep], c.cls[keep]
    n_all = int(len(x))
    del c
    xyz = np.c_[x, y, z]
    ext = (x0 - BUFFER, y0 - BUFFER, x1 + BUFFER, y1 + BUFFER)
    hag, gg = utility.height_above_ground(x, y, z, cls, ext, 1.0)
    wmask, wlab, winfo = utility.detect_wires(xyz, hag)
    poles = utility.detect_poles(xyz, hag, exclude=wmask)
    P, pidx = spans.candidate_points(xyz, hag)
    inside = lambda a, b: (a >= x0) & (a < x1) & (b >= y0) & (b < y1)
    wi = np.flatnonzero(wmask)
    wi = wi[inside(x[wi], y[wi])]
    pm = inside(P[:, 0], P[:, 1])
    pol = [p for p in poles if x0 <= p["x"] < x1 and y0 <= p["y"] < y1]
    gx0, gy0 = int(np.floor(x0 - ext[0])), int(np.floor(y0 - ext[1]))
    gcore = gg[gy0:gy0 + int(round(y1 - y0)), gx0:gx0 + int(round(x1 - x0))]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, wire_xyz=xyz[wi].astype("float32"), wire_hag=hag[wi].astype("float32"), wire_chain=wlab[wi], P=P[pm].astype("float32"),
                        poles=np.array([[p["x"], p["y"], p["top_hag"], p["height"], p["points"], p["radius"], p["linearity"], p.get("ring_ratio", np.nan)] for p in pol]).reshape(-1, 8),
                        ground=gcore.astype("float32"), ground_origin=np.array([x0, y0]), points_read=n_all, date=date, cx=cx, cy=cy)
    d = CACHE / "ept" / project / "data"
    if d.exists() and sum(f.stat().st_size for f in d.glob("*.laz")) > max_cache_gb * 1e9:
        shutil.rmtree(d, ignore_errors=True)
    return out
