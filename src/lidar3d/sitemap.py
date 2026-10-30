"""Site maps: regional context from Overture Maps (county boundaries, roads, water, places, public land) and a state locator, for a study area given by its centre and size.
Overture's geometry layers carry OpenStreetMap-derived data; they are used for orientation only, never to build the analysis."""
from __future__ import annotations

import os
import pickle

import numpy as np
from shapely import wkb

from .ept import CACHE
from .names import RELEASE


def _table(theme, typ, cols, bbox, extra=None):
    import pyarrow.dataset as ds
    from pyarrow import fs
    W, S, E, N = bbox
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=os.environ.get("HTTPS_PROXY"))
    flt = (ds.field("bbox", "xmin") < E) & (ds.field("bbox", "xmax") > W) & (ds.field("bbox", "ymin") < N) & (ds.field("bbox", "ymax") > S)
    if extra is not None:
        flt = flt & extra
    return ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/theme={theme}/type={typ}", filesystem=s3, format="parquet").to_table(filter=flt, columns=cols).to_pandas()


def regional_layers(key, bbox, region_name, pop_min=20000, roads=("motorway", "trunk"), land_names=()):
    """Layers for the regional map, cached. Returns dict of lists of shapely geometries (lon/lat) and labelled places."""
    import pyarrow.dataset as ds
    p = CACHE / "forestry" / f"overture_{key}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    nm = lambda r: (r.names or {}).get("primary")
    cty = _table("divisions", "division_area", ["subtype", "names", "geometry"], bbox, ds.field("subtype") == "county")
    reg = _table("divisions", "division_area", ["subtype", "names", "geometry"], (-125, 24, -66, 50), (ds.field("subtype") == "region") & (ds.field("names", "primary") == region_name))
    rd = _table("transportation", "segment", ["class", "geometry", "names"], bbox, ds.field("class").isin(list(roads)))
    wt = _table("base", "water", ["class", "geometry", "names"], bbox, ds.field("class").isin(["river", "lake", "reservoir"]))
    pl = _table("divisions", "division", ["subtype", "names", "geometry", "population"], bbox, ds.field("subtype").isin(["locality"]))
    pl = pl[pl.population.fillna(0) >= pop_min]
    land = _table("base", "land_use", ["subtype", "class", "names", "geometry"], bbox, ds.field("subtype") == "protected")
    out = dict(
        counties=[(nm(r), wkb.loads(r.geometry)) for _, r in cty.iterrows()],
        region=[wkb.loads(r.geometry) for _, r in reg.iterrows()],
        roads=[(r["class"], wkb.loads(r.geometry)) for _, r in rd.iterrows()],
        water=[(r["class"], wkb.loads(r.geometry)) for _, r in wt.iterrows()],
        places=[(nm(r), float(r.population), wkb.loads(r.geometry)) for _, r in pl.iterrows()],
        land=[(nm(r), r["class"], wkb.loads(r.geometry)) for _, r in land.iterrows() if (nm(r) or "") in land_names or "National Forest" in (nm(r) or "")],
    )
    p.write_bytes(pickle.dumps(out))
    return out


def _lines(ax, g, **kw):
    for p in (g.geoms if hasattr(g, "geoms") else [g]):
        if p.geom_type == "LineString":
            ax.plot(*p.xy, **kw)
        elif p.geom_type == "Polygon":
            ax.plot(*p.exterior.xy, **kw)


def _fill(ax, g, **kw):
    for p in (g.geoms if hasattr(g, "geoms") else [g]):
        if p.geom_type == "Polygon":
            ax.fill(*p.exterior.xy, **kw)


def draw_regional(ax, L, bbox, aoi_lonlat, aoi_half_deg, label_places=(), water_min_deg2=2e-5):
    W, S, E, N = bbox
    from shapely.geometry import box
    clip = box(W, S, E, N)
    for nm, cl, g in L["land"]:
        _fill(ax, g.intersection(clip), fc="#DDEBD3", ec="none", zorder=1)
    for cl, g in L["water"]:
        gg = g.intersection(clip)
        if gg.is_empty:
            continue
        if cl == "river":
            _lines(ax, gg, color="#A9C8E6", lw=0.5, zorder=2)
        elif gg.area > water_min_deg2:
            _fill(ax, gg, fc="#B9D6EE", ec="#8FB8DE", lw=0.3, zorder=2)
    for nm, g in L["counties"]:
        _lines(ax, g.intersection(clip), color="#9A9A9A", lw=0.9, ls="--", zorder=3)
    for cl, g in L["roads"]:
        _lines(ax, g.intersection(clip), color="#C0392B" if cl == "motorway" else "#E59866", lw=0.7 if cl == "motorway" else 0.5, alpha=0.85, zorder=4)
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    seen, placed = set(), []
    minx, miny = 0.085 * (E - W), 0.035 * (N - S)           # labels closer than this to an earlier label are dropped, so names never stack
    for nm, pop, g in sorted(L["places"], key=lambda t: -t[1]):
        if nm in label_places and clip.contains(g) and nm not in seen and all(abs(g.x - px) > minx or abs(g.y - py) > miny for px, py in placed):
            seen.add(nm); placed.append((g.x, g.y))
            ax.plot(g.x, g.y, "o", ms=3.5, color="#333333", zorder=6, mec="white", mew=0.6)
            ax.text(g.x + (E - W) * 0.008, g.y + (N - S) * 0.008, nm, fontsize=8.5, color="#222222", zorder=7, path_effects=halo)
    lon, lat = aoi_lonlat
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((lon - aoi_half_deg[0], lat - aoi_half_deg[1]), 2 * aoi_half_deg[0], 2 * aoi_half_deg[1], fill=False, ec="#E74C3C", lw=2.0, zorder=8))
    ax.set_xlim(W, E); ax.set_ylim(S, N); ax.set_aspect(1 / np.cos(np.radians((S + N) / 2)))
    ax.tick_params(labelsize=7, length=2)


def county_of(L, lon, lat):
    from shapely.geometry import Point
    pt = Point(lon, lat)
    for nm, g in L["counties"]:
        if g.contains(pt):
            return nm
    return None


def land_of(L, lon, lat):
    from shapely.geometry import Point
    pt = Point(lon, lat)
    return [nm for nm, cl, g in L["land"] if g.contains(pt) and nm]


def nearest_land(L, lon, lat):
    """(name, distance in km) of the nearest mapped public-land polygon, 0 when inside."""
    from shapely.geometry import Point
    pt = Point(lon, lat)
    best = None
    for nm, cl, g in L["land"]:
        d = 0.0 if g.contains(pt) else g.distance(pt) * 111.0
        if nm and (best is None or d < best[1]):
            best = (nm, d)
    return best
