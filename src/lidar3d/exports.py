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


_VIEWER = """<!doctype html><meta charset=utf-8><title>Tatoosh Range point cloud</title>
<style>html,body{margin:0;height:100%%;background:#f5f4ef;font:14px system-ui,sans-serif;color:#222}canvas{display:block;width:100%%;height:100%%;cursor:grab}
#i{position:fixed;left:14px;top:12px;max-width:420px;line-height:1.35}#i b{font-size:16px}</style>
<div id=i><b>Lidar point cloud, Tatoosh Range, Washington</b><br>%(n)s points sampled from %(tot)s (USGS 3DEP, collected %(d0)s to %(d1)s). Drag to rotate, wheel to zoom, shift-drag to pan. Colour: %(mode)s.</div>
<canvas id=c></canvas><script>
const D=%(data)s;const N=D.n,cx=D.cx,cy=D.cy,cz=D.cz;
const P=new Float32Array(N*3),C=new Uint8Array(N*3);
(function(){const b=atob(D.p),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);
const q=new Int16Array(a.buffer);for(let i=0;i<N*3;i++)P[i]=q[i]*D.s;
const b2=atob(D.c);for(let i=0;i<b2.length;i++)C[i]=b2.charCodeAt(i);})();
const cv=document.getElementById('c'),g=cv.getContext('2d');let az=3.5,el=0.6,zm=1,px=0,py=0;
function draw(){const W=cv.width=cv.clientWidth*devicePixelRatio,H=cv.height=cv.clientHeight*devicePixelRatio;g.fillStyle='#f5f4ef';g.fillRect(0,0,W,H);
const ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el),sc=Math.min(W,H)/1500*zm;
const im=g.getImageData(0,0,W,H),d=im.data,zb=new Float32Array(W*H).fill(1e9);
for(let i=0;i<N;i++){const x=P[i*3],y=P[i*3+1],z=P[i*3+2]*1.3;const u=x*ca-y*sa,v0=x*sa+y*ca,dep=v0*ce+z*se*-1+0,v=z*ce-v0*se;
const X=(W/2+px*devicePixelRatio+u*sc)|0,Y=(H/2+py*devicePixelRatio-v*sc)|0;
for(let a=0;a<2;a++)for(let b=0;b<2;b++){const xx=X+a,yy=Y+b;if(xx<0||yy<0||xx>=W||yy>=H)continue;const k=yy*W+xx;if(dep<zb[k]){zb[k]=dep;d[k*4]=C[i*3];d[k*4+1]=C[i*3+1];d[k*4+2]=C[i*3+2];d[k*4+3]=255}}}
g.putImageData(im,0,0)}
let dn=0,sh=0,lx=0,ly=0,t=0;cv.onmousedown=e=>{dn=1;sh=e.shiftKey;lx=e.clientX;ly=e.clientY};onmouseup=()=>dn=0;
onmousemove=e=>{if(!dn)return;if(sh){px+=e.clientX-lx;py+=e.clientY-ly}else{az+=(e.clientX-lx)*.008;el=Math.max(-.2,Math.min(1.5,el+(e.clientY-ly)*.008))}lx=e.clientX;ly=e.clientY;req()};
cv.onwheel=e=>{e.preventDefault();zm*=e.deltaY<0?1.1:.9;req()};function req(){if(!t){t=1;requestAnimationFrame(()=>{t=0;draw()})}}onresize=req;req();
</script>"""


def html_viewer(pts, rgb, path, total, d0, d1, mode):
    """Self-contained viewer: positions as int16 (cm-level quantisation over the AOI), colours as bytes, a z-buffered software splat on a 2D canvas."""
    c = pts.mean(0)
    loc = pts - c
    s = float(np.abs(loc).max() / 32000)
    q = np.round(loc / s).astype("<i2")
    import base64
    data = dict(n=int(len(pts)), cx=float(c[0]), cy=float(c[1]), cz=float(c[2]), s=s, p=base64.b64encode(q.tobytes()).decode(), c=base64.b64encode(rgb.astype(np.uint8).tobytes()).decode())
    Path(path).write_text(_VIEWER % dict(n=f"{len(pts):,}", tot=f"{total:,}", d0=d0, d1=d1, mode=mode, data=json.dumps(data)))
