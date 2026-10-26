"""Urban study area: a 3DEP point cloud (any EPT project) plus Overture building footprints, cached under data/cache/<slug>."""
from __future__ import annotations

import os

import numpy as np
from pyproj import Transformer
from shapely import wkb

from .aoi import AOI
from .ept import CACHE, Cloud, EPT
from .names import RELEASE

SEATTLE_CENTER = dict(slug="seattle_center", project="WA_KingCo_1_2021", lonlat=(-122.3510, 47.6215), half=350.0, crs="EPSG:32610")


def make_aoi(spec=SEATTLE_CENTER) -> AOI:
    cx, cy = Transformer.from_crs("EPSG:4326", spec["crs"], always_xy=True).transform(*spec["lonlat"])
    return AOI(round(cx), round(cy), spec["half"], spec["crs"])


def load_cloud(aoi: AOI, spec=SEATTLE_CENTER, log=print) -> Cloud:
    npz = CACHE / f"{spec['slug']}_cloud.npz"
    if npz.exists():
        d = np.load(npz)
        return Cloud(**{k: d[k] for k in ("x", "y", "z", "cls", "intensity", "ret", "nret", "gps")}, crs=str(d["crs"]))
    c = EPT(spec["project"]).fetch(aoi.bbox_ll, max_depth=20, log=log)
    x0, y0, x1, y1 = aoi.bounds
    m = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1) & ~np.isin(c.cls, [7, 18])
    c = Cloud(c.x[m], c.y[m], c.z[m], c.cls[m], c.intensity[m], c.ret[m], c.nret[m], c.gps, c.crs)
    np.savez(npz, x=c.x, y=c.y, z=c.z, cls=c.cls, intensity=c.intensity, ret=c.ret, nret=c.nret, gps=c.gps, crs=c.crs)
    return c


def overture_buildings(aoi: AOI):
    """Overture building footprints intersecting the AOI, in the AOI's UTM CRS. Returns a GeoDataFrame with Overture's own height / floors / class attributes (often empty)."""
    import geopandas as gpd
    import pyarrow.dataset as ds
    from pyarrow import fs
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=os.environ.get("HTTPS_PROXY"))
    W, S, E, N = aoi.bbox_ll
    flt = (ds.field("bbox", "xmin") < E) & (ds.field("bbox", "xmax") > W) & (ds.field("bbox", "ymin") < N) & (ds.field("bbox", "ymax") > S)
    cols = ["id", "names", "height", "num_floors", "class", "subtype", "geometry"]
    t = ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building", filesystem=s3, format="parquet").to_table(filter=flt, columns=cols).to_pandas()
    g = gpd.GeoDataFrame(t.drop(columns="geometry"), geometry=[wkb.loads(b) for b in t["geometry"]], crs="EPSG:4326").to_crs(aoi.crs)
    g["name"] = [n.get("primary") if isinstance(n, dict) else None for n in g["names"]]
    x0, y0, x1, y1 = aoi.bounds
    from shapely.geometry import box
    g = g[g.intersects(box(x0, y0, x1, y1))].drop(columns="names").reset_index(drop=True)
    return g
