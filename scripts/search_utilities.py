"""Search for 3DEP lidar sites that also have mapped utility features (Overture Maps power infrastructure). Usage: python scripts/search_utilities.py

1. Take substations and plants from Overture inside each project's bounding box.
2. Keep sampled ones where the lidar octree is deep (a cheap test for dense data, from the cached hierarchy only).
3. Fetch a 300 m window at the best few and record point density, class histogram, and how many Overture power features fall in the window.
4. For each probed site, count how many Overture towers and poles have a column of above-ground returns on top of them.
Writes data/processed/utilities/candidates.csv and tower_check.csv. The result depends on the Overture release and on which substations the random sample draws (seed 0).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import requests
from pyarrow import fs
from pyproj import Transformer
from scipy import ndimage
from shapely import wkb

from lidar3d.ept import EPT
from lidar3d.names import RELEASE

PROJECTS = ["WA_KingCo_1_2021", "WA_PierceCounty_1_2020", "CT_Statewide_B1_2016", "RI_Statewide_1_D22", "MA_CentralEastern_2_2021", "NJ_SouthernNJ_2018",
            "CA_SanFrancisco_1_B23", "NY_NewYorkCity", "OR_OLCMetro_2019"]
HALF = 150.0
OUT = ROOT / "data" / "processed" / "utilities"


def overture():
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=os.environ.get("HTTPS_PROXY"))
    return ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/theme=base/type=infrastructure", filesystem=s3, format="parquet")


def power_in(inf, W, S, E, N, columns, classes=None):
    flt = (ds.field("bbox", "xmin") < E) & (ds.field("bbox", "xmax") > W) & (ds.field("bbox", "ymin") < N) & (ds.field("bbox", "ymax") > S) & (ds.field("subtype") == "power")
    if classes:
        flt &= ds.field("class").isin(classes)
    return inf.to_table(filter=flt, columns=columns).to_pandas()


def window(e, lon, lat, half):
    crs = f"EPSG:{32600 + int((lon + 180) // 6) + 1}"
    tr, tb = Transformer.from_crs(4326, crs, always_xy=True), Transformer.from_crs(crs, 4326, always_xy=True)
    cx, cy = tr.transform(lon, lat)
    W, S = tb.transform(cx - half, cy - half)
    E, N = tb.transform(cx + half, cy + half)
    c = e.fetch((W, S, E, N), max_depth=20, log=lambda s: None)
    m = (c.x >= cx - half) & (c.x < cx + half) & (c.y >= cy - half) & (c.y < cy + half)
    return c, m, cx, cy, tr, (W, S, E, N)


def tower_check(c, m, cx, cy, tr, bbox, inf, half):
    """Share of Overture towers/poles with a column of >= 20 returns more than 3 m above the ground within 2.5 m, and their median top height."""
    X, Y, Z, K = c.x[m] - cx, c.y[m] - cy, c.z[m], c.cls[m]
    res = 2.0
    n = int(2 * half / res)
    ix, iy = ((X + half) / res).astype(int).clip(0, n - 1), ((Y + half) / res).astype(int).clip(0, n - 1)
    g = np.full((n, n), np.inf)
    gm = K == 2
    np.minimum.at(g, (iy[gm], ix[gm]), Z[gm])
    g[~np.isfinite(g)] = np.nan
    if np.isnan(g).all():
        return {}
    idx = ndimage.distance_transform_edt(np.isnan(g), return_distances=False, return_indices=True)
    hag = Z - g[tuple(idx)][iy, ix]
    t = power_in(inf, *bbox, ["class", "geometry"])
    out = {}
    for kind in ("power_tower", "power_pole"):
        tops, ok, n_ = [], 0, 0
        for b in t[t["class"] == kind].geometry:
            g_ = wkb.loads(b)
            if g_.geom_type != "Point":
                continue
            x, y = tr.transform(g_.x, g_.y)
            x, y = x - cx, y - cy
            if abs(x) > half - 5 or abs(y) > half - 5:
                continue
            near = (np.hypot(X - x, Y - y) < 2.5) & (hag > 3)
            n_ += 1
            if near.sum() >= 20:
                ok += 1
                tops.append(float(np.percentile(hag[near], 99)))
        out[kind] = dict(n=n_, with_column=ok, median_top_m=float(np.median(tops)) if tops else None)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inf = overture()
    t3 = Transformer.from_crs(3857, 4326, always_xy=True)
    t3857 = Transformer.from_crs(4326, 3857, always_xy=True)
    rng = np.random.default_rng(0)
    rows, checks = [], []
    for p in PROJECTS:
        m_ = requests.get(f"https://usgs-lidar-public.s3.amazonaws.com/{p}/ept.json", timeout=60).json()
        b = m_["bounds"]
        W, S = t3.transform(b[0], b[1])
        E, N = t3.transform(b[3], b[4])
        t = power_in(inf, W, S, E, N, ["id", "class", "names", "geometry"], ["substation"])
        pts = [wkb.loads(g).centroid for g in t.geometry]
        t["lon"], t["lat"] = [q.x for q in pts], [q.y for q in pts]
        t = t[(t.lon > W) & (t.lon < E) & (t.lat > S) & (t.lat < N)].reset_index(drop=True)
        t["name"] = [(n or {}).get("primary") if isinstance(n, dict) else None for n in t.names]
        e = EPT(p)
        cand = []
        for i in rng.permutation(len(t))[:60]:
            r = t.loc[i]
            x, y = t3857.transform(r.lon, r.lat)
            try:
                nd = e.nodes((x - 100, y - 100, x + 100, y + 100), 20)
            except Exception:
                continue
            cand.append((max([int(k.split("-")[0]) for k in nd], default=0), bool(r["name"]), i))
        cand.sort(reverse=True)
        dmax = cand[0][0] if cand else 0
        for d, _, i in [c for c in cand if c[0] >= dmax - 1][:4]:
            r = t.loc[i]
            try:
                c, m, cx, cy, tr, bbox = window(e, r.lon, r.lat, HALF)
            except Exception as ex:
                print(p, r["name"], "no points", str(ex)[:40])
                continue
            cls = dict(zip(*[a.tolist() for a in np.unique(c.cls[m], return_counts=True)]))
            ov = power_in(inf, *bbox, ["class"])["class"].value_counts().to_dict()
            rows.append(dict(project=p, name=r["name"], lon=round(r.lon, 5), lat=round(r.lat, 5), points=int(m.sum()), pts_per_m2=round(m.sum() / (2 * HALF) ** 2, 1),
                             wire_or_tower_class=int(cls.get(14, 0) + cls.get(15, 0)), classes=json.dumps(cls), **{f"ov_{k}": int(v) for k, v in ov.items()}))
            print(p, r["name"], rows[-1]["pts_per_m2"], ov, flush=True)
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c.startswith("ov_")]
    df[num] = df[num].fillna(0).astype(int)
    df["name"] = df["name"].fillna("")
    df.to_csv(OUT / "candidates.csv", index=False)
    # tower/pole check at the richest sites (most Overture towers + poles in a larger window)
    df["tp"] = df.get("ov_power_tower", 0) + df.get("ov_power_pole", 0)
    for _, r in df.sort_values("tp", ascending=False).head(6).iterrows():
        e = EPT(r.project)
        c, m, cx, cy, tr, bbox = window(e, r.lon, r.lat, 350.0)
        res = tower_check(c, m, cx, cy, tr, bbox, inf, 350.0)
        for kind, v in res.items():
            checks.append(dict(project=r.project, name=r["name"], lon=r.lon, lat=r.lat, density=round(m.sum() / 700.0 ** 2, 1), kind=kind, **v))
        print(r.project, r["name"], res, flush=True)
    pd.DataFrame(checks).to_csv(OUT / "tower_check.csv", index=False)


if __name__ == "__main__":
    main()
