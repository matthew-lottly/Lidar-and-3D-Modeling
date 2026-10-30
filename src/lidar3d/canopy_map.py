"""A self-contained interactive map of a canopy analysis: toggleable raster layers (as embedded PNGs), the detected trees as clickable points, pan and zoom, no network and no libraries."""
from __future__ import annotations

import json
from pathlib import Path

_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>%(title)s</title>
<style>html,body{margin:0;height:100%%;font:13px system-ui,sans-serif;background:#f4f3ee;color:#222}#c{width:100%%;height:100%%;display:block;cursor:grab}
#p{position:fixed;left:10px;top:10px;background:rgba(255,255,255,.93);padding:10px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.25);max-width:290px}
#p h1{font-size:14px;margin:0 0 6px}#p label{display:block;margin:3px 0}#p small{color:#555;display:block;margin-top:6px;line-height:1.35}
#t{position:fixed;pointer-events:none;background:rgba(20,20,20,.9);color:#fff;padding:5px 8px;border-radius:4px;font-size:12px;display:none}
#sb{position:fixed;left:12px;bottom:12px;background:rgba(255,255,255,.9);padding:4px 8px;border-radius:4px}</style></head><body>
<canvas id="c"></canvas><div id="p"><h1>%(title)s</h1><div id="ly"></div><label><input type="checkbox" id="tr" checked> trees (%(ntrees)s)</label><small>%(note)s<br>Drag to pan, wheel to zoom, hover a tree for its height and crown.</small></div>
<div id="t"></div><div id="sb"></div>
<script>
const D=%(data)s;const cv=document.getElementById('c'),g=cv.getContext('2d'),tip=document.getElementById('t'),sb=document.getElementById('sb');
let imgs={},on={},z=1,px=0,py=0,showTrees=true;const S=D.size;
D.layers.forEach((l,i)=>{const im=new Image();im.src='data:image/png;base64,'+l.png;im.onload=()=>draw();imgs[l.name]=im;on[l.name]=i==0;
 const lab=document.createElement('label');const cb=document.createElement('input');cb.type='radio';cb.name='base';cb.checked=i==0;cb.onchange=()=>{D.layers.forEach(m=>on[m.name]=(m.name==l.name));draw()};
 lab.appendChild(cb);lab.append(' '+l.name);document.getElementById('ly').appendChild(lab)});
document.getElementById('tr').onchange=e=>{showTrees=e.target.checked;draw()};
function fit(){const W=cv.width=cv.clientWidth*devicePixelRatio,H=cv.height=cv.clientHeight*devicePixelRatio;return [W,H,Math.min(W,H)*0.92/S*z]}
function toS(x,y,W,H,k){return [W/2+px*devicePixelRatio+(x-S/2)*k,H/2+py*devicePixelRatio-(y-S/2)*k]}
function ramp(h){const t=Math.max(0,Math.min(1,h/D.hmax));return 'hsl('+(60+60*(1-t))+',70%%,'+(65-35*t)+'%%)'}
function draw(){const [W,H,k]=fit();g.fillStyle='#f4f3ee';g.fillRect(0,0,W,H);g.imageSmoothingEnabled=false;
 D.layers.forEach(l=>{if(on[l.name]&&imgs[l.name].complete){const [x0,y0]=toS(0,S,W,H,k);g.drawImage(imgs[l.name],x0,y0,S*k,S*k)}});
 if(showTrees){const r=Math.max(1.2,k*1.2);D.trees.forEach(t=>{const [x,y]=toS(t[0],t[1],W,H,k);if(x<0||y<0||x>W||y>H)return;g.beginPath();g.arc(x,y,r,0,6.283);g.fillStyle=ramp(t[2]);g.fill();g.lineWidth=.6;g.strokeStyle='rgba(0,0,0,.55)';g.stroke()})}
 const m=100*k/devicePixelRatio;sb.textContent='100 m = '+Math.round(m)+' px'}
let dn=false,lx=0,ly=0;cv.onmousedown=e=>{dn=true;lx=e.clientX;ly=e.clientY;cv.style.cursor='grabbing'};onmouseup=()=>{dn=false;cv.style.cursor='grab'};
onmousemove=e=>{if(dn){px+=e.clientX-lx;py+=e.clientY-ly;lx=e.clientX;ly=e.clientY;draw()}
 const [W,H,k]=fit();const mx=e.clientX*devicePixelRatio,my=e.clientY*devicePixelRatio;let best=null,bd=(9*devicePixelRatio)**2;
 if(showTrees)D.trees.forEach(t=>{const [x,y]=toS(t[0],t[1],W,H,k);const d=(x-mx)**2+(y-my)**2;if(d<bd){bd=d;best=t}});
 if(best){tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent='height '+best[2].toFixed(1)+' m, crown '+best[3].toFixed(1)+' m wide'}else tip.style.display='none'};
cv.onwheel=e=>{e.preventDefault();const f=e.deltaY<0?1.15:1/1.15;z=Math.max(.5,Math.min(40,z*f));draw()};onresize=draw;draw();
</script></body></html>"""


def write_map(path, title, size_m, layers, trees, hmax, note=""):
    """layers: list of (name, base64 PNG) covering the same square (north up); trees: list of (x, y, height, crown diameter) in metres from the south-west corner."""
    data = dict(size=float(size_m), hmax=float(hmax), layers=[dict(name=n, png=p) for n, p in layers], trees=[[round(float(a), 1), round(float(b), 1), round(float(c), 1), round(float(d), 1)] for a, b, c, d in trees])
    Path(path).write_text(_HTML % dict(title=title, ntrees=f"{len(trees):,}", note=note, data=json.dumps(data, separators=(",", ":"))))
