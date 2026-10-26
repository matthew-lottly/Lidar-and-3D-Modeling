"""Read points from a USGS 3DEP Entwine Point Tile (EPT) dataset on AWS, restricted to a bounding box.

EPT stores points in an octree of LAZ files. The octree's coordinates are in the dataset's own CRS (Web Mercator, EPSG:3857, for 3DEP);
we select nodes that intersect the box, download them in parallel, crop points to the box, and reproject to the local UTM zone so that
horizontal units are true metres. Vertical values are already metres.
"""
from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import requests
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
BASE = "https://usgs-lidar-public.s3.amazonaws.com"


@dataclass
class Cloud:
    x: np.ndarray            # UTM easting, m
    y: np.ndarray            # UTM northing, m
    z: np.ndarray            # elevation, m
    cls: np.ndarray          # ASPRS classification
    intensity: np.ndarray
    ret: np.ndarray          # return number
    nret: np.ndarray         # number of returns
    gps: np.ndarray          # GPS time (adjusted standard GPS seconds)
    crs: str

    def __len__(self):
        return len(self.x)


class EPT:
    def __init__(self, name: str):
        self.name = name
        self.root = f"{BASE}/{name}"
        self.meta = self._json(f"{self.root}/ept.json", f"{name}/ept.json")
        self.cube = np.array(self.meta["bounds"], float)
        self.span = self.meta["span"]
        self._hier: dict[str, int] = {}

    @staticmethod
    def _json(url, cache_name):
        p = CACHE / "ept" / cache_name
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            return json.loads(p.read_text())
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        p.write_text(r.text)
        return r.json()

    def node_bounds(self, key: str) -> np.ndarray:
        d, x, y, z = (int(k) for k in key.split("-"))
        w = (self.cube[3:] - self.cube[:3]) / (2 ** d)
        lo = self.cube[:3] + w * np.array([x, y, z])
        return np.r_[lo, lo + w]

    def _load_hierarchy(self, key: str, box, max_depth: int):
        page = self._json(f"{self.root}/ept-hierarchy/{key}.json", f"{self.name}/h/{key}.json")
        for k, n in page.items():
            d = int(k.split("-")[0])
            b = self.node_bounds(k)
            if b[0] > box[2] or b[3] < box[0] or b[1] > box[3] or b[4] < box[1] or d > max_depth:
                continue
            if n == -1:
                self._load_hierarchy(k, box, max_depth)
            elif n > 0:
                self._hier[k] = n

    def nodes(self, box_3857, max_depth: int = 20) -> dict[str, int]:
        self._hier = {}
        self._load_hierarchy("0-0-0-0", box_3857, max_depth)
        return dict(self._hier)

    def spacing(self, depth: int) -> float:
        """Approximate point spacing at an octree depth, in true metres at the dataset's latitude is not applied here (units of the dataset CRS)."""
        return (self.cube[3] - self.cube[0]) / self.span / 2 ** depth

    def fetch(self, bbox_ll: tuple[float, float, float, float], max_depth: int = 20, workers: int = 6, log=print) -> Cloud:
        """Points inside (W, S, E, N) WGS84, as a Cloud in the UTM zone of the box centre."""
        W, S, E, N = bbox_ll
        t3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x0, y0 = t3857.transform(W, S)
        x1, y1 = t3857.transform(E, N)
        box = (x0, y0, x1, y1)
        nodes = self.nodes(box, max_depth)
        log(f"{self.name}: {len(nodes)} octree nodes, {sum(nodes.values()):,} points before cropping")
        zone = int((W + E) / 2 + 180) // 6 + 1
        utm = f"EPSG:{32600 + zone}"
        to_utm = Transformer.from_crs("EPSG:3857", utm, always_xy=True)
        e0, n0 = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform(W, S)
        e1, n1 = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform(E, N)

        def one(key):
            p = CACHE / "ept" / self.name / "data" / f"{key}.laz"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                for attempt in range(6):
                    try:
                        r = requests.get(f"{self.root}/ept-data/{key}.laz", timeout=120)
                        r.raise_for_status()
                        break
                    except requests.RequestException:
                        if attempt == 5:
                            raise
                        time.sleep(2 * (attempt + 1))
                tmp = p.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.rename(p)
            las = laspy.read(str(p))
            X, Y = np.asarray(las.x), np.asarray(las.y)
            m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
            if not m.any():
                return None
            E_, N_ = to_utm.transform(X[m], Y[m])
            keep = (E_ >= e0) & (E_ <= e1) & (N_ >= n0) & (N_ <= n1)
            gps = np.asarray(las.gps_time)[m][keep][::500] if "gps_time" in las.point_format.dimension_names else np.zeros(0)     # a sample is enough to date the survey
            g = lambda n: np.asarray(las[n])[m][keep]
            return E_[keep], N_[keep], np.asarray(las.z)[m][keep].astype(np.float32), g("classification"), g("intensity"), g("return_number"), g("number_of_returns"), gps

        parts = []
        with ThreadPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(one, list(nodes)), 1):
                if r is not None:
                    parts.append(r)
                if i % 200 == 0:
                    log(f"  {i}/{len(nodes)} nodes")
        cat = [np.concatenate([p[i] for p in parts]) for i in range(8)]
        return Cloud(*cat, crs=utm)
