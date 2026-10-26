"""Named natural features near the study area from Overture Maps (peaks, lakes), so map labels come from data."""
from __future__ import annotations

import os

import numpy as np
import pyarrow.dataset as ds
from pyarrow import fs
from pyproj import Transformer

RELEASE = "2026-09-23.1"


def features(bbox_ll, kinds=(("land", "theme=base/type=land"), ("water", "theme=base/type=water"))):
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=os.environ.get("HTTPS_PROXY"))
    W, S, E, N = bbox_ll
    flt = (ds.field("bbox", "xmin") < E) & (ds.field("bbox", "xmax") > W) & (ds.field("bbox", "ymin") < N) & (ds.field("bbox", "ymax") > S)
    out = []
    from shapely import wkb
    for k, path in kinds:
        t = ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/{path}", filesystem=s3, format="parquet").to_table(filter=flt, columns=["names", "subtype", "class", "geometry"] + (["elevation"] if k == "land" else [])).to_pandas()
        for _, r in t.iterrows():
            nm = r["names"].get("primary") if isinstance(r["names"], dict) else None
            if nm:
                g = wkb.loads(r["geometry"])
                out.append(dict(kind=k, name=nm, cls=r["class"], elevation=r.get("elevation"), x=g.centroid.x, y=g.centroid.y, geom=g))
    return out
