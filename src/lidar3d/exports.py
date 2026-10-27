"""Small, openable exports: terrain mesh (GLB/OBJ), land-cover GeoTIFF, decimated colour point cloud (PLY), standalone HTML point-cloud viewer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import trimesh
from rasterio.transform import from_origin

from .landcover import CLASSES

OUT = Path(__file__).resolve().parents[2] / "data" / "processed"


def landcover_tif(P, LC, path=None):
    path = Path(path or OUT / "land_cover_1m.tif")
    t = from_origin(P.aoi.bounds[0], P.aoi.bounds[3], 1.0, 1.0)
    with rasterio.open(path, "w", driver="GTiff", height=LC.cls.shape[0], width=LC.cls.shape[1], count=1, dtype="uint8", crs=P.aoi.crs, transform=t, nodata=0, compress="deflate") as d:
        d.write(LC.cls.astype("uint8"), 1)
        d.update_tags(classes=json.dumps({int(k): v[0] for k, v in CLASSES.items()}))
    return path


def terrain_mesh(z, x0, y1, step, rgb, path_glb=None, path_obj=None):
    """Regular-grid triangle mesh (two triangles per cell) with vertex colours, in local metres (origin at the AOI centre)."""
    zs = z[::step, ::step]
    r, c = zs.shape
    jj, ii = np.meshgrid(np.arange(c), np.arange(r))
    v = np.c_[(jj.ravel() * step + 0.5) - z.shape[1] / 2, z.shape[0] / 2 - (ii.ravel() * step + 0.5), zs.ravel()]
    idx = np.arange(r * c).reshape(r, c)
    a, b, d, e = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    f = np.r_[np.c_[a, d, b], np.c_[b, d, e]]
    m = trimesh.Trimesh(v, f, vertex_colors=np.c_[rgb[::step, ::step].reshape(-1, 3), np.full(r * c, 255)], process=False)
    if path_glb:
        m.export(path_glb)
    if path_obj:
        m.export(path_obj)
    return m


def point_ply(pts, rgb, path):
    """Binary little-endian PLY, local coordinates (centre subtracted) in float32 so viewers do not lose precision."""
    n = len(pts)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    a = np.empty(n, dt)
    a["x"], a["y"], a["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    a["r"], a["g"], a["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    with open(path, "wb") as fh:
        fh.write(hdr.encode()); fh.write(a.tobytes())


_VIEWER = """<!doctype html><meta charset=utf-8><title>%(title)s</title>
<style>html,body{margin:0;height:100%%;background:#f5f4ef;font:14px system-ui,sans-serif;color:#222}canvas{display:block;width:100%%;height:100%%;cursor:grab}
#i{position:fixed;left:14px;top:12px;max-width:420px;line-height:1.35}#i b{font-size:16px}</style>
<div id=i><b>%(title)s</b><br>%(n)s points sampled from %(tot)s (USGS 3DEP, collected %(dates)s). Drag to rotate, wheel to zoom, shift-drag to pan. Colour: %(mode)s.</div>
<canvas id=c></canvas><script>
const D=%(data)s;const N=D.n,cx=D.cx,cy=D.cy,cz=D.cz;
const P=new Float32Array(N*3),C=new Uint8Array(N*3);
(function(){const b=atob(D.p),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);
const q=new Int16Array(a.buffer);for(let i=0;i<N*3;i++)P[i]=q[i]*D.s;
const b2=atob(D.c);for(let i=0;i<b2.length;i++)C[i]=b2.charCodeAt(i);})();
const cv=document.getElementById('c'),g=cv.getContext('2d');let az=3.5,el=0.6,zm=1,px=0,py=0;
function draw(){const W=cv.width=cv.clientWidth*devicePixelRatio,H=cv.height=cv.clientHeight*devicePixelRatio;g.fillStyle='#f5f4ef';g.fillRect(0,0,W,H);
const ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el),sc=Math.min(W,H)/(2.3*D.r)*zm;
const im=g.getImageData(0,0,W,H),d=im.data,zb=new Float32Array(W*H).fill(1e9);
for(let i=0;i<N;i++){const x=P[i*3],y=P[i*3+1],z=P[i*3+2]*1.3;const u=x*ca-y*sa,v0=x*sa+y*ca,dep=-(v0*ce+z*se),v=z*ce-v0*se;
const X=(W/2+px*devicePixelRatio+u*sc)|0,Y=(H/2+py*devicePixelRatio-v*sc)|0;
for(let a=0;a<2;a++)for(let b=0;b<2;b++){const xx=X+a,yy=Y+b;if(xx<0||yy<0||xx>=W||yy>=H)continue;const k=yy*W+xx;if(dep<zb[k]){zb[k]=dep;d[k*4]=C[i*3];d[k*4+1]=C[i*3+1];d[k*4+2]=C[i*3+2];d[k*4+3]=255}}}
g.putImageData(im,0,0)}
let dn=0,sh=0,lx=0,ly=0,t=0;cv.onmousedown=e=>{dn=1;sh=e.shiftKey;lx=e.clientX;ly=e.clientY};onmouseup=()=>dn=0;
onmousemove=e=>{if(!dn)return;if(sh){px+=e.clientX-lx;py+=e.clientY-ly}else{az+=(e.clientX-lx)*.008;el=Math.max(-.2,Math.min(1.5,el+(e.clientY-ly)*.008))}lx=e.clientX;ly=e.clientY;req()};
cv.onwheel=e=>{e.preventDefault();zm*=e.deltaY<0?1.1:.9;req()};function req(){if(!t){t=1;requestAnimationFrame(()=>{t=0;draw()})}}onresize=req;req();
</script>"""


def html_viewer(pts, rgb, path, total, d0, d1, mode, title="Lidar point cloud, Tatoosh Range, Washington"):
    """Self-contained viewer: positions as int16 (cm-level quantisation over the AOI), colours as bytes, a z-buffered software splat on a 2D canvas."""
    c = pts.mean(0)
    loc = pts - c
    s = float(np.abs(loc).max() / 32000)
    q = np.round(loc / s).astype("<i2")
    import base64
    data = dict(n=int(len(pts)), cx=float(c[0]), cy=float(c[1]), cz=float(c[2]), s=s, r=float(np.abs(loc).max()), p=base64.b64encode(q.tobytes()).decode(), c=base64.b64encode(rgb.astype(np.uint8).tobytes()).decode())
    Path(path).write_text(_VIEWER % dict(n=f"{len(pts):,}", tot=f"{total:,}", dates=(str(d0) if str(d0) == str(d1) else f'{d0} to {d1}'), mode=mode, title=title, data=json.dumps(data)))


# ------------------------------------------------------------------------------------------------ buildings
def cityjson_lod1(gdf, path, crs_epsg=32610, source="", solids=None):
    """CityJSON 1.1, one Building per row with an LOD1 Solid (ground, walls, flat roof) between base_z and roof_z, coordinates in the projected CRS."""
    from shapely.geometry import MultiPolygon
    from shapely.geometry.polygon import orient
    verts, index = [], {}

    def vid(x, y, z):
        k = (round(x, 3), round(y, 3), round(z, 3))
        if k not in index:
            index[k] = len(verts)
            verts.append(k)
        return index[k]

    objs = {}
    solids = solids or {}
    for i, r in gdf[gdf.status == "ok"].iterrows():
        if r.get("name") in solids:                                       # a modelled solid (triangle mesh, LOD2-style) instead of the extruded footprint
            m = solids[r.get("name")]
            surfaces = [[[vid(*m.vertices[v]) for v in f]] for f in m.faces]
            nz = m.face_normals[:, 2]
            sem = [0 if q > 0.3 else 1 if q < -0.9 else 2 for q in nz]
            a = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in dict(
                name=r.get("name"), overture_id=r.get("id"), overture_height=r.get("height"), height_m=round(float(r.height_m), 2), height_max_m=round(float(r.height_max_m), 2),
                ground_z=round(float(r.ground_z), 2), roof_type="modelled from lidar (surface of revolution)", footprint_area_m2=round(float(r.area_m2), 1)).items()}
            objs[f"bldg_{i}"] = dict(type="Building", attributes=a, geometry=[dict(type="Solid", lod="2", boundaries=[surfaces], semantics=dict(
                surfaces=[{"type": "RoofSurface"}, {"type": "GroundSurface"}, {"type": "WallSurface"}], values=[sem]))])
            continue
        polys = list(r.geometry.geoms) if isinstance(r.geometry, MultiPolygon) else [r.geometry]
        surfaces, sem = [], []
        b, t = float(r.base_z), float(r.roof_z)
        if t <= b + 0.5:
            continue
        for pg in polys:
            pg = orient(pg, 1.0)
            rings = [list(pg.exterior.coords)[:-1]] + [list(h.coords)[:-1] for h in pg.interiors]
            surfaces.append([[vid(x, y, t) for x, y in rg] for rg in rings]); sem.append(0)                     # roof, normal up
            surfaces.append([[vid(x, y, b) for x, y in rg[::-1]] for rg in rings]); sem.append(1)               # ground, normal down
            for rg in rings:
                for (x0, y0), (x1, y1) in zip(rg, rg[1:] + rg[:1]):
                    surfaces.append([[vid(x0, y0, b), vid(x1, y1, b), vid(x1, y1, t), vid(x0, y0, t)]]); sem.append(2)   # outward wall
        a = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in dict(
            name=r.get("name"), overture_id=r.get("id"), overture_height=r.get("height"), height_m=round(float(r.height_m), 2), height_max_m=round(float(r.height_max_m), 2),
            ground_z=round(float(r.ground_z), 2), roof_type=r.roof_type, footprint_area_m2=round(float(r.area_m2), 1)).items()}
        objs[f"bldg_{i}"] = dict(type="Building", attributes=a, geometry=[dict(type="Solid", lod="1", boundaries=[surfaces], semantics=dict(
            surfaces=[{"type": "RoofSurface"}, {"type": "GroundSurface"}, {"type": "WallSurface"}], values=[sem]))])
    v = np.array(verts)
    tr = dict(scale=[0.001] * 3, translate=[float(v[:, 0].min()), float(v[:, 1].min()), float(v[:, 2].min())])
    iv = np.round((v - np.array(tr["translate"])) / 0.001).astype(int).tolist()
    doc = {"type": "CityJSON", "version": "1.1", "transform": tr, "CityObjects": objs, "vertices": iv,
           "metadata": {"referenceSystem": f"https://www.opengis.net/def/crs/EPSG/0/{crs_epsg}", "title": "Seattle Center buildings, LOD1 from 3DEP lidar", "description": source}}
    Path(path).write_text(json.dumps(doc, separators=(",", ":")))
    return len(objs)


def buildings_glb(gdf, terrain_z, x0, y1, res, centre, path, cmap_name="viridis", vmax=45.0, solids=None):
    """Scene with a terrain mesh and the LOD1 blocks coloured by height above ground. Local coordinates: metres from `centre` (x, y, z)."""
    from shapely.geometry import MultiPolygon
    sc = trimesh.Scene()
    step = 4
    zs = terrain_z[::step, ::step]
    h, w = zs.shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    v = np.c_[x0 + (jj.ravel() * step + 0.5) * res - centre[0], y1 - (ii.ravel() * step + 0.5) * res - centre[1], zs.ravel() - centre[2]]
    idx = np.arange(h * w).reshape(h, w)
    a, b, d, e = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    terr = trimesh.Trimesh(v, np.r_[np.c_[a, d, b], np.c_[b, d, e]], process=False)
    terr.visual.vertex_colors = np.tile([205, 200, 190, 255], (len(v), 1)).astype(np.uint8)
    sc.add_geometry(terr, node_name="terrain")
    cmap = __import__("matplotlib").colormaps[cmap_name]
    for i, r in gdf[gdf.status == "ok"].iterrows():
        if r.roof_z <= r.base_z + 0.5:
            continue
        polys = list(r.geometry.geoms) if isinstance(r.geometry, MultiPolygon) else [r.geometry]
        col = (np.array(cmap(min(r.height_m / vmax, 1.0)))[:4] * 255).astype(np.uint8)
        if solids and r.get("name") in solids:
            m = solids[r.get("name")].copy()
            m.apply_translation([-centre[0], -centre[1], -centre[2]])
            m.visual.face_colors = col
            sc.add_geometry(m, node_name=f"bldg_{i}_{r.get('name')}")
            continue
        for pg in polys:
            m = trimesh.creation.extrude_polygon(pg, float(r.roof_z - r.base_z))
            m.apply_translation([-centre[0], -centre[1], r.base_z - centre[2]])
            m.visual.face_colors = col
            sc.add_geometry(m, node_name=f"bldg_{i}_{r.get('name') or ''}".strip())
    sc.export(path)


def cityjson_open_edges(doc: dict) -> dict[str, int]:
    """For every Building solid, count directed edges without an opposite partner. A closed, consistently oriented shell has none."""
    bad = {}
    for k, o in doc["CityObjects"].items():
        edges = {}
        for srf in o["geometry"][0]["boundaries"][0]:
            for ring in srf:
                for a, b in zip(ring, ring[1:] + ring[:1]):
                    edges[(a, b)] = edges.get((a, b), 0) + 1
        n = sum(1 for (a, b), c in edges.items() if c != edges.get((b, a), 0))
        if n:
            bad[k] = n
    return bad


def textured_parts(gdf, terrain_z, x0, y1, res, size, ortho_rgb, centre, step=4, wall_rgb=(186, 180, 170), solids=None, roof_rows=(), roof_surface=None):
    """Terrain and flat roofs share one image texture (the lidar-corrected NAIP ortho); walls get a plain colour because no image shows them. Local coordinates, metres from `centre`.
    Footprints listed in `roof_rows` (index labels) get their real roof shape instead of a flat cap: roof_surface = (absolute z array, transform) is sampled on a ~1.2 m triangulation that includes the footprint outline.
    Returns (surface mesh with UVs and texture, wall mesh)."""
    from PIL import Image
    from shapely.geometry import MultiPolygon
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import SimpleMaterial
    y0 = y1 - size

    def uv_of(xy):
        return np.c_[(xy[:, 0] - x0) / size, (xy[:, 1] - y0) / size]

    zs = terrain_z[::step, ::step]
    h, w = zs.shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    gx, gy = x0 + (jj.ravel() * step + 0.5) * res, y1 - (ii.ravel() * step + 0.5) * res
    idx = np.arange(h * w).reshape(h, w)
    a, b, d, e = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    V = [np.c_[gx, gy, zs.ravel()]]
    F = [np.r_[np.c_[a, d, b], np.c_[b, d, e]]]
    off = len(V[0])
    wallV, wallF, wallC, woff = [], [], [], 0
    otile = size / ortho_rgb.shape[2]

    def photo_at(xy):
        c = np.clip(((xy[:, 0] - x0) / otile).astype(int), 0, ortho_rgb.shape[2] - 1)
        r = np.clip(((y1 - xy[:, 1]) / otile).astype(int), 0, ortho_rgb.shape[1] - 1)
        return ortho_rgb[:3, r, c].T
    solids = solids or {}
    for gi, r in gdf[gdf.status == "ok"].iterrows():
        if r.roof_z <= r.base_z + 0.5:
            continue
        parts = list(r.geometry.geoms) if isinstance(r.geometry, MultiPolygon) else [r.geometry]
        if r.get("name") in solids:
            parts = [None]
        elif gi in roof_rows and roof_surface is not None:
            for pg in parts:
                rv, rf, wv, wf = roof_heightfield(pg, roof_surface, float(r.base_z))
                V.append(rv); F.append(rf + off); off += len(rv)
                wallV.append(wv); wallF.append(wf + woff); woff += len(wv)
                top_xy = wv[wf].mean(1)[:, :2]                      # sides of a curved metal roof continue the roof: colour each wall triangle from the photo at its top edge, darkened
                wallC.append(np.c_[(photo_at(top_xy) * 0.7).astype(np.uint8), np.full(len(wf), 255, np.uint8)])
            continue
        for pg in parts:
            if pg is None:
                m = solids[r.get("name")]
                v = m.vertices.copy()
                up = m.face_normals[:, 2] > 0.5                    # upward faces (saucer deck, top house) get the aerial photo; the rest is plain
            else:
                m = trimesh.creation.extrude_polygon(pg, float(r.roof_z - r.base_z))
                v = m.vertices.copy(); v[:, 2] += r.base_z
            if pg is not None:
                up = m.face_normals[:, 2] > 0.9
            cap = m.faces[up]
            used, inv = np.unique(cap, return_inverse=True)
            V.append(v[used]); F.append(inv.reshape(-1, 3) + off); off += len(used)
            wf = m.faces[~up]
            wallV.append(v); wallF.append(wf + woff); woff += len(v)
            wallC.append(np.tile(np.r_[wall_rgb, 255].astype(np.uint8), (len(wf), 1)))
    Vs = np.vstack(V)
    img = Image.fromarray(np.moveaxis(ortho_rgb[:3], 0, -1))
    surf = trimesh.Trimesh(Vs - np.array(centre), np.vstack(F), process=False)
    surf.visual = TextureVisuals(uv=uv_of(Vs[:, :2]), material=SimpleMaterial(image=img))
    walls = trimesh.Trimesh(np.vstack(wallV) - np.array(centre), np.vstack(wallF), process=False)
    walls.visual.face_colors = np.vstack(wallC)
    return surf, walls


def ortho_tif(rgbn, tf, crs, path, dates=""):
    with rasterio.open(path, "w", driver="GTiff", height=rgbn.shape[1], width=rgbn.shape[2], count=rgbn.shape[0], dtype="uint8", crs=crs, transform=tf, compress="deflate", photometric="RGB") as d:
        d.write(rgbn)
        d.update_tags(dates=dates, note="NAIP re-sampled with lidar surface heights; bands R,G,B,NIR")


def roof_heightfield(poly, roof_surface, base_z, step=1.2):
    """Roof triangles following a surface raster, clipped to a footprint, plus the walls from the footprint outline down to base_z.
    Points: the outline (every <= step m) and a grid inside the outline pulled back 0.6 m; triangulated by Delaunay and kept where the triangle centre is inside the footprint.
    Returns (roof vertices, roof faces facing up, wall vertices, wall faces facing out); roof face indices are local to the returned vertices."""
    import shapely
    from scipy.spatial import Delaunay
    from shapely.geometry.polygon import orient
    z_abs, tf = roof_surface
    poly = orient(poly, 1.0)
    seg = shapely.segmentize(poly, step)
    rings = [np.asarray(seg.exterior.coords)[:-1]] + [np.asarray(i.coords)[:-1] for i in seg.interiors]
    inner = poly.buffer(-0.6)
    x0, y0, x1, y1 = poly.bounds
    gx, gy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    gp = np.c_[gx.ravel(), gy.ravel()]
    gp = gp[shapely.contains_xy(inner, gp[:, 0], gp[:, 1])] if not inner.is_empty else gp[:0]
    xy = np.vstack(rings + [gp])

    def zat(p):
        c = np.clip(((p[:, 0] - tf.c) / tf.a).astype(int), 0, z_abs.shape[1] - 1)
        r = np.clip(((tf.f - p[:, 1]) / tf.a).astype(int), 0, z_abs.shape[0] - 1)
        return z_abs[r, c]

    roof_v = np.c_[xy, zat(xy)]
    tri = Delaunay(xy).simplices
    cen = xy[tri].mean(1)
    keep = shapely.contains_xy(poly.buffer(0.05), cen[:, 0], cen[:, 1])
    tri = tri[keep]
    n = np.cross(roof_v[tri[:, 1]] - roof_v[tri[:, 0]], roof_v[tri[:, 2]] - roof_v[tri[:, 0]])
    tri = np.where((n[:, 2] < 0)[:, None], tri[:, ::-1], tri)
    wv, wf, o = [], [], 0
    for rg in rings:
        k = len(rg)
        top = np.c_[rg, zat(rg)]
        bot = np.c_[rg, np.full(k, base_z)]
        wv.append(np.vstack([bot, top]))
        for i in range(k):
            j = (i + 1) % k
            wf += [[o + i, o + j, o + k + j], [o + i, o + k + j, o + k + i]]
        o += 2 * k
    return roof_v, tri, np.vstack(wv), np.array(wf)
