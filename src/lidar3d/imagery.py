"""Sentinel-2 L2A true-colour (TCI) window for draping on the terrain, from the public sentinel-cogs bucket (AWS)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import rasterio
import requests
from rasterio.windows import from_bounds

from .aoi import AOI
from .ept import CACHE
from .radar import _env, mgrs_square

BASE = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"


def _list(prefix):
    out, tok = [], None
    while True:
        p = {"list-type": 2, "prefix": prefix, "delimiter": "/"}
        if tok:
            p["continuation-token"] = tok
        t = requests.get(BASE + "/", params=p, timeout=60).text
        out += [x for x in re.findall(r"<Prefix>([^<]*)</Prefix>", t) if x != prefix]
        m = re.search(r"<NextContinuationToken>([^<]*)", t)
        if not m:
            return out
        tok = m.group(1)


@dataclass
class S2Scene:
    prefix: str
    date: str
    cloud: float


def scenes(lon, lat, year, months):
    z, band, sq = mgrs_square(lon, lat)
    out = []
    for m in months:
        for p in _list(f"sentinel-s2-l2a-cogs/{z}/{band}/{sq}/{year}/{m}/"):
            mm = re.search(r"_(\d{8})_\d_L2A/$", p)
            if not mm:
                continue
            name = p.strip("/").split("/")[-1]
            jp = CACHE / "s2" / f"{name}.json"
            jp.parent.mkdir(parents=True, exist_ok=True)
            if not jp.exists():
                r = requests.get(f"{BASE}/{p}{name}.json", timeout=60)
                if r.status_code != 200:
                    continue
                jp.write_text(r.text)
            j = json.loads(jp.read_text())
            d = mm.group(1)
            out.append(S2Scene(p.strip("/"), f"{d[:4]}-{d[4:6]}-{d[6:]}", float(j["properties"].get("eo:cloud_cover", 100))))
    return sorted(out, key=lambda s: s.date)


def read(scene: S2Scene, asset: str, aoi: AOI, pad=40.0):
    with _env(), rasterio.open(f"/vsicurl/{BASE}/{scene.prefix}/{asset}.tif") as src:
        x0, y0, x1, y1 = aoi.bounds
        w = from_bounds(x0 - pad, y0 - pad, x1 + pad, y1 + pad, src.transform).round_offsets().round_lengths()
        return src.read(window=w, masked=True), src.window_transform(w)


def best_scene(lon, lat, aoi, year=2022, months=(8, 9)):
    """Clearest scene by cloud/shadow share of the study-area window (from the scene classification layer), preferring dates close to the lidar (September 2022)."""
    best = None
    for s in scenes(lon, lat, year, months):
        if s.cloud > 40:
            continue
        scl, _ = read(s, "SCL", aoi)
        a = np.ma.filled(scl[0], 0)
        if (a == 0).mean() > 0.01:
            continue
        bad = float(np.isin(a, (3, 8, 9, 10, 11)).mean())
        key = (round(bad, 2), s.date)
        if best is None or key < best[0]:
            best = (key, s, bad)
    return best
