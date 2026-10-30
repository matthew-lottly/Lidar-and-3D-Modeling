"""End-to-end rooftop solar pipeline for a block: lidar -> buildings -> roof planes -> shading -> annual irradiance per roof cell, plane and building (see rooftop.py and solar.py for the steps)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import rooftop as RT
from . import solar as SO
from . import solarwx

PV_EFFICIENCY, PV_PERFORMANCE_RATIO = 0.20, 0.80       # assumptions for the electricity estimate, not measurements
SUITABLE_KWH_M2 = 1000.0                                # a roof cell is "suitable" when it receives at least this much (beam + diffuse) per year, a common screening level


def analyse(site, weather=None, log=print, use_trees=True, thin=1.0, dsm_res=1.0, seed=0, reuse=None):
    """Run the pipeline for `site` (see rooftop.SITES; a custom dict with the same keys works). `weather` is the dict from solarwx.fetch_tmy (fetched if None).
    thin < 1 keeps that share of the returns, dsm_res = 2 uses a coarser surface model for shading (sensitivity tests)."""
    if reuse is not None:                      # same buildings and planes, different surface model for the shading (sensitivity tests)
        C, hag, dtm, pm, lab_k, polys_k, B = (reuse[k] for k in ("C", "hag", "dtm", "planar", "lab", "polys", "B"))
        _, _, dsm = RT.surfaces(C, 1.0, trees=use_trees)
    else:
        C = RT.load_cloud(site, log=log)
        if thin < 1.0:
            k = np.random.default_rng(seed).random(len(C.x)) < thin
            C = RT.Cloud(C.x[k], C.y[k], C.z[k], C.cls[k], C.nret[k], C.date, C.half, C.cx, C.cy)
        dtm, hag, dsm = RT.surfaces(C, 1.0, trees=use_trees)
        pm = RT.planar_points(C, hag) & (C.nret == 1)
        lab, polys = RT.footprints(C, hag, use=pm)
        A = site["half"]
        keep = [i for i, p in enumerate(polys) if abs(p.centroid.x) <= A and abs(p.centroid.y) <= A]
        log(f"{len(polys)} structures in the loaded square, {len(keep)} in the analysis area")
        polys_k = [polys[i] for i in keep]
        lab_k = np.zeros_like(lab)
        for new, old in enumerate(keep, 1):
            lab_k[lab == old + 1] = new
        B = RT.build_buildings(C, hag, dtm, lab_k, polys_k, use=pm)
    if weather is None:
        lon, lat = site["lonlat"]
        weather = solarwx.fetch_tmy(lon, lat, site["slug"])
    df = weather["df"]
    lon, lat = site["lonlat"]
    el, az = SO.sun_positions(df.index, lat, lon)
    # all roof cells
    xy, z0, nrm, bid, pid, cslope = [], [], [], [], [], []
    for b in B:
        for j, p in enumerate(b.planes):
            sel = b.cell_plane == j
            if not sel.any():
                continue
            c = b.cells[sel]
            xy.append(c)
            z0.append(p.z(c[:, 0], c[:, 1]))
            nrm.append(np.tile(SO.plane_normals([p.a], [p.b]), (len(c), 1)))
            bid.append(np.full(len(c), b.bid))
            pid.append(np.full(len(c), j))
            cslope.append(np.full(len(c), p.slope_deg))
    xy, z0, nrm = np.vstack(xy), np.concatenate(z0), np.vstack(nrm)
    bid, pid, cslope = np.concatenate(bid), np.concatenate(pid), np.concatenate(cslope)
    log(f"{len(B)} buildings, {len(xy):,} roof cells; shading ...")
    d_in = dsm
    if dsm_res != 1.0:
        k = int(dsm_res)
        n = dsm.shape[0] // k * k
        d_in = dsm[:n, :n].reshape(n // k, k, n // k, k).max(axis=(1, 3))
    H = SO.horizon_angles(xy, z0, d_in, C.half, res=dsm_res)
    beam, diff = SO.annual_irradiance(H, nrm, el, az, df.dni.values, df.dhi.values)
    Hflat = np.zeros_like(H)
    beam_open, diff_open = SO.annual_irradiance(Hflat, nrm, el, az, df.dni.values, df.dhi.values)
    cells = pd.DataFrame(dict(x=xy[:, 0], y=xy[:, 1], z=z0, bid=bid, plane=pid, slope=cslope, beam=beam, diffuse=diff, beam_open=beam_open, diffuse_open=diff_open, horizon_max=H.max(1)))
    cells["total"] = cells.beam + cells.diffuse
    cells["area"] = 1.0 / np.cos(np.radians(cells.slope))
    return dict(C=C, dtm=dtm, hag=hag, dsm=dsm, planar=pm, lab=lab_k, polys=polys_k, B=B, cells=cells, H=H, weather=weather, sun=(el, az), site=site)


def annual_totals(R):
    """Roof-area-weighted annual totals of a run: (beam kWh/m2, diffuse kWh/m2, open-sky beam kWh/m2, total energy GWh)."""
    c = R["cells"]
    a = c.area
    return float((c.beam * a).sum() / a.sum()), float((c.diffuse * a).sum() / a.sum()), float((c.beam_open * a).sum() / a.sum()), float((c.total * a).sum() / 1e6)


def building_table(R):
    """One row per building: footprint, height, roof area, planes, annual beam and diffuse energy on the roof (kWh/yr), shading loss, suitable area and an electricity estimate under stated assumptions."""
    cells, B = R["cells"], R["B"]
    rows = []
    g = cells.groupby("bid")
    for b in B:
        if b.bid not in g.groups:
            continue
        c = g.get_group(b.bid)
        a = c.area.to_numpy()
        roof = a.sum()
        e_beam, e_diff, e_open = (c.beam * a).sum(), (c.diffuse * a).sum(), (c.beam_open * a).sum()
        suit = c.total >= SUITABLE_KWH_M2
        ps = RT.plane_stats(b)
        main = max(ps, key=lambda r: r["area_m2"])
        rows.append(dict(building=b.bid, x=b.footprint.centroid.x, y=b.footprint.centroid.y, footprint_m2=b.footprint.area, height_m=b.height, roof_area_m2=roof,
                         roof_coverage=roof / (b.footprint.area / np.cos(np.radians(np.average(c.slope, weights=a)))) if len(c) else np.nan, planes=len(ps), main_slope_deg=main["slope_deg"], main_aspect_deg=main["aspect_deg"],
                         beam_kwh_yr=e_beam, diffuse_kwh_yr=e_diff, total_kwh_yr=e_beam + e_diff, mean_total_kwh_m2=(e_beam + e_diff) / roof, mean_beam_kwh_m2=e_beam / roof,
                         shading_loss_beam=1 - e_beam / e_open if e_open > 0 else np.nan, suitable_area_m2=a[suit].sum(), suitable_share=a[suit].sum() / roof,
                         pv_kwh_yr_est=float((c.total * a)[suit].sum() * PV_EFFICIENCY * PV_PERFORMANCE_RATIO)))
    return pd.DataFrame(rows)
