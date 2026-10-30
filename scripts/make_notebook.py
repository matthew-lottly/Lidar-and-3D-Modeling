"""Writes (and optionally executes) notebooks/rooftop_solar.ipynb. Usage: python scripts/make_notebook.py [--run]"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
    M("# Rooftop solar potential for any city block\n\nGive a longitude/latitude box; the notebook finds a 3DEP lidar project covering it, extracts the buildings and their roof planes from the point cloud, shades each roof with the surrounding buildings and trees, "
      "adds up a typical year of sun, writes a 3D model of the block (`.glb` and `.obj`) and plots the energy arriving on every rooftop.\n\n"
      "**What the numbers are.** Energy *arriving* on each roof surface in a typical meteorological year (NREL NSRDB, hourly direct and diffuse sunlight at the nearest 4 km site), after shading. "
      "It is not electricity: the optional PV estimate multiplies by a module efficiency and performance ratio that are assumptions. Ground and wall reflections, the bright sky around the sun, soiling and snow are not included. "
      "Roofs hidden under tree crowns, small outbuildings and dormers are partly or wholly missing from the lidar-derived roofs. Roof planes come from a numpy RANSAC (Open3D was not available in this environment).\n\n"
      "The box is limited to about 600 m on a side (runtime and memory); shading from a further 300 m around it is included."),
    C("# Parameters: edit these. The default is a 300 m block of West University Place, Houston.\n"
      "WEST, SOUTH, EAST, NORTH = -95.4352, 29.7162, -95.4318, 29.7188\n"
      "PROJECT = None          # a 3DEP project name such as 'TX_Coastal_B1_2018'; None picks the newest one covering the box\n"
      "OUTDIR = 'notebook_output/block'"),
    C("import sys, warnings\nfrom pathlib import Path\nwarnings.filterwarnings('ignore')\nROOT = Path.cwd().resolve()\nif not (ROOT / 'src').exists():\n    ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT / 'src'))\n"
      "import numpy as np\nimport matplotlib.pyplot as plt\nfrom matplotlib.colors import Normalize\nfrom lidar3d import block, rooftop"),
    M("## 1. Lidar, buildings, roof planes, shading and sunlight"),
    C("R, T, files = block.run_block(WEST, SOUTH, EAST, NORTH, ROOT / OUTDIR, project=PROJECT, log=lambda s: None)\n"
      "site, C = R['site'], R['C']\n"
      "print(f\"Lidar project {site['project']}, flown {C.date}; analysis square {2 * site['half']:.0f} m on a side (plus {site['context']:.0f} m of shading context)\")\n"
      "print(f\"{len(T)} buildings, {T.roof_area_m2.sum() / 1e4:.2f} ha of roof, {T.total_kwh_yr.sum() / 1e6:.2f} GWh/year arriving on the roofs\")\n"
      "df = R['weather']['df']\nprint(f\"Typical-year sunlight at the NSRDB site {R['weather']['site_lat']:.2f} N, {R['weather']['site_lon']:.2f} E: GHI {df.ghi.sum() / 1000:.0f}, DNI {df.dni.sum() / 1000:.0f}, DHI {df.dhi.sum() / 1000:.0f} kWh/m2/yr\")"),
    M("## 2. Energy on every rooftop"),
    C("cells = R['cells']\nfig, ax = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw=dict(width_ratios=[1.1, 1]))\nh = site['half']\n"
      "ax[0].set_facecolor('#eeeeea')\nsc = ax[0].scatter(cells.x, cells.y, c=cells.total, s=6, cmap='inferno', norm=Normalize(600, 1500), marker='s', lw=0)\n"
      "for b in R['B']:\n    ax[0].plot(*b.footprint.exterior.xy, color='#444444', lw=0.5)\nax[0].set_xlim(-h, h); ax[0].set_ylim(-h, h); ax[0].set_aspect('equal'); ax[0].set_title('annual energy on the roof, kWh per m2 of roof'); plt.colorbar(sc, ax=ax[0], shrink=0.8)\n"
      "top = T.sort_values('total_kwh_yr', ascending=False).head(20)\nax[1].barh(range(len(top)), top.total_kwh_yr / 1000, color=plt.cm.inferno(Normalize(600, 1500)(top.mean_total_kwh_m2)))\n"
      "ax[1].set_yticks(range(len(top))); ax[1].set_yticklabels([f'building {int(b)}' for b in top.building], fontsize=8); ax[1].invert_yaxis(); ax[1].set_xlabel('MWh per year arriving on the roof')\n"
      "ax[1].set_title('the 20 rooftops that receive most (colour = kWh per m2)')\nplt.tight_layout(); plt.show()"),
    C("cols = ['building', 'footprint_m2', 'height_m', 'planes', 'main_slope_deg', 'main_aspect_deg', 'roof_area_m2', 'mean_total_kwh_m2', 'total_kwh_yr', 'shading_loss_beam', 'suitable_area_m2', 'pv_kwh_yr_est']\n"
      "T[cols].round(1).sort_values('total_kwh_yr', ascending=False).head(15)"),
    M("## 3. The block in 3D\n\nThe files are written to the output folder: `block.glb` (open in any glTF viewer or Blender) and `block.obj` (with vertex colours). Below is a software-rendered preview."),
    C("import trimesh\nfrom lidar3d import render3d\nsc = trimesh.load(files['glb'])\npts, cols_ = [], []\nfor g in sc.geometry.values():\n    P, fi = trimesh.sample.sample_surface(g, int(max(g.area * 40, 100)))\n"
      "    pts.append(P); cols_.append(np.asarray(g.visual.vertex_colors)[g.faces[fi, 0]][:, :3])\nim, _ = render3d.render(np.vstack(pts), np.vstack(cols_).astype(np.uint8), az=205, el=38, width=1500, height=1000, splat=2)\n"
      "plt.figure(figsize=(11, 7.5)); plt.imshow(im); plt.axis('off'); plt.show()\nprint({k: str(v) for k, v in files.items()})"),
    M("## 4. Optional: electricity estimate (assumptions, not measurements)"),
    C("EFFICIENCY, PERFORMANCE_RATIO, MIN_KWH_M2 = 0.20, 0.80, 1000.0   # module efficiency, system losses, screening level for a usable roof cell\n"
      "ok = cells.total >= MIN_KWH_M2\nkwh = float((cells.total * cells.area)[ok].sum() * EFFICIENCY * PERFORMANCE_RATIO)\n"
      "print(f\"{(cells.area[ok].sum() / cells.area.sum()) * 100:.0f}% of the roof area receives at least {MIN_KWH_M2:.0f} kWh/m2/yr; panels there would make about {kwh / 1e6:.2f} GWh of electricity a year under these assumptions\")"),
]
out = ROOT / "notebooks" / "rooftop_solar.ipynb"
nbf.write(nb, out)
if "--run" in sys.argv:
    from nbconvert.preprocessors import ExecutePreprocessor
    ExecutePreprocessor(timeout=3000, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out.parent)}})
    nbf.write(nb, out)
print("wrote", out)
