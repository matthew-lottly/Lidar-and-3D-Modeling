"""Plates 15 and 16: coastal bluff change at Cape Cod from two 3DEP epochs with M3C2."""
from __future__ import annotations

import textwrap

import numpy as np

from . import change
from .cloud import hillshade
from .plates import INK, MUTED, Plate, colorbar, map_frame, north_arrow, scalebar

SRC = ("Lidar: USGS 3D Elevation Program (3DEP) on AWS (usgs-lidar-public). Epoch 1: {p1}, collected {d1}. Epoch 2: {p2}, collected {d2}. Change: M3C2 (Lague et al. 2013) as implemented in py4dgeo; "
       "surface chosen geometrically, because the two projects classify the bluff face differently.")
VM = 3.0


def _src(S):
    e1, e2 = S["epochs"]
    return SRC.format(p1=e1["project"], d1=e1["date"], p2=e2["project"], d2=e2["date"])


def _arrays(npz, P2):
    c, d, sig, sm, ok = npz["core"], npz["d"], npz["sig"], npz["smooth"], npz["ok"]
    hs = hillshade(np.nan_to_num(change.grid_dem(P2, 2.0), nan=0.0), 2.0, 315, 40, 1.5)
    return c, d, sig, sm, ok, hs


def _profile(p, v0, v1, ang, width=6.0, du=1.0):
    cs, sn = np.cos(ang), np.sin(ang)
    u, v = p[:, 0] * cs - p[:, 1] * sn, p[:, 0] * sn + p[:, 1] * cs
    m = np.abs(v - (v0 + v1) / 2) < width / 2
    return u[m], p[m, 2]


def fig_change(S, npz, P1, P2, ang_deg):
    c, d, sig, sm, ok, hs = _arrays(npz, P2)
    ang = np.radians(ang_deg)
    z = S["zones"]
    pl = Plate("Bluff erosion at Cape Cod, 2014 to 2021", f"Surface change between two USGS lidar surveys {S['years_between']:.0f} years apart, measured with M3C2: distance along the surface normal, no meshes, only changes larger than the level of detection",
               _src(S), "Local grid, metres east and north of 41.9128 N, 69.9715 W (UTM 19N)")
    ex = (-300, 300, -300, 300)
    a = pl.ax(0.6, 1.25, 6.9, 6.9)
    a.imshow(hs, extent=ex, cmap="gray", vmin=0.15, vmax=1.0, origin="lower")
    mask = ok & sm & sig
    k = ok & sm & ~sig
    a.scatter(c[k, 0], c[k, 1], s=2.2, c="#E8E6DE", marker="s", lw=0, alpha=0.55)
    k = ok & ~sm
    a.scatter(c[k, 0], c[k, 1], s=2.2, c="#B8B2A0", marker="s", lw=0, alpha=0.5)
    a.scatter(c[mask, 0], c[mask, 1], s=3.6, c=d[mask], cmap="RdBu", vmin=-VM, vmax=VM, marker="s", lw=0)
    a.set_xlim(-130, 300); a.set_ylim(-300, 300)
    a.set_aspect("equal")
    map_frame(a, (-130, 300, -300, 300), 100, grid=False); scalebar(a, (-130, 300, -300, 300), 100); north_arrow(a, (-130, 300, -300, 300))
    for y0 in (210, 30, -200):
        u0 = np.array([-30, 120]); x = u0 * np.cos(ang) + (y0 * np.sin(ang)); yy = -u0 * np.sin(ang) + y0 * np.cos(ang)
        a.plot(x, yy, color="#111111", lw=1.0, ls="--"); a.text(x[0] - 4, yy[0], "ABC"[(210, 30, -200).index(y0)], fontsize=11, fontweight="bold", ha="right", va="center")
    pl.label(0.6, 8.3, "Distance along the surface normal, 2021 minus 2014 (m)")
    colorbar(pl, "RdBu", -VM, VM, 8.0, 7.45, 3.2, 0.14, "Red: surface lowered. Blue: raised. Values beyond 3 m share the end colour.", ticks=[-3, -1.5, 0, 1.5, 3])
    pl.bg.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((11.6, 7.43), 0.2, 0.18, fc="#E8E6DE", ec=MUTED, lw=0.4)); pl.text(11.9, 7.5, "no change above the level of detection", fontsize=8, va="center")
    pl.bg.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((11.6, 7.15), 0.2, 0.18, fc="#B8B2A0", ec=MUTED, lw=0.4)); pl.text(11.9, 7.22, "not interpreted: vegetation or rough surface", fontsize=8, va="center")
    # profiles
    for i, (v, lab) in enumerate(((210, "A"), (30, "B"), (-200, "C"))):
        b = 5.5 - i * 1.3
        f = pl.ax(8.1, b, 7.2, 1.0)
        for P, col, nm in ((P1, "#8A8A8A", "2014"), (P2, "#9A3B12", "2021")):
            u, zz = _profile(P, v, v, ang, width=6.0)
            m = (u > -40) & (u < 180)
            f.scatter(u[m], zz[m], s=0.25, c=col, lw=0, alpha=0.6, rasterized=True)
            f.plot([], [], color=col, label=nm, lw=2)
        f.set_xlim(-40, 180); f.set_ylim(-3, 36)
        f.tick_params(labelsize=7, length=2); f.spines[["top", "right"]].set_visible(False)
        if i == 0:
            f.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
        if i == 2:
            f.set_xlabel("distance across the shore, m (seaward to the right)", fontsize=8, color=MUTED)
        pl.text(7.8, b + 1.0, lab, fontsize=11, fontweight="bold", va="top", ha="right")
        f.set_ylabel("m", fontsize=7.5, color=MUTED)
    pl.label(8.1, 8.3, "Profiles through 6 m wide slabs (dashed lines on the map)")
    # numbers
    nb, mb, sb, bh = z["north bluff"], z["middle bluff"], z["south bank"], z["beach"]
    L = S["along_length_m"]
    rows = [("North bluff face", nb, L["north bluff"]), ("Middle bluff face", mb, L["middle bluff"]), ("South bank", sb, L["south bank"])]
    y = 2.0
    pl.text(8.1, y + 0.3, "Significant volume change, 6 to 26 m above sea level (smooth cells only)", fontsize=8.5, color=INK, fontweight="bold", va="top")
    rows.append(("Beach, below 3 m (not interpreted: tide, waves)", bh, None))
    for k_, (nm, st, ln) in enumerate(rows):
        pl.text(8.1, y - 0.05 - k_ * 0.25, nm, fontsize=8.3, va="top")
        pl.text(11.2, y - 0.05 - k_ * 0.25, f"net {st['net_m3']:+,.0f} m³ (loss {st['loss_m3']:,.0f}, gain {st['gain_m3']:,.0f}), ±{st['uncertainty_m3']:,.0f}", fontsize=8.3, va="top")
    return pl.save("15_capecod_bluff_change.png")


def fig_validation(S, npz):
    c, d, sm, ok = npz["core"], npz["d"], npz["smooth"], npz["ok"]
    st = S["stable"]
    pl = Plate("How far to trust the change numbers", "Checks on ground that should not have moved, an independent volume calculation, and what the two surveys can and cannot see",
               _src(S), "Stable area: inland ground west of x = -120 m, chosen by eye")
    inl = (c[:, 0] < st["x_max"]) & ok
    a = pl.ax(0.9, 5.2, 4.6, 2.75)
    bins = np.linspace(-1.0, 0.6, 90)
    a.hist(d[inl], bins=bins, color="#B8B2A0", label=f"all surface ({inl.sum():,})")
    a.hist(d[inl & sm], bins=bins, color="#3B6E8F", label=f"smooth only ({(inl & sm).sum():,})")
    a.axvline(0, color=INK, lw=0.8); a.set_xlabel("M3C2 distance on stable ground, m", fontsize=8, color=MUTED); a.set_ylabel("core points", fontsize=8, color=MUTED)
    a.legend(fontsize=8, frameon=False, loc="upper left"); a.tick_params(labelsize=7.5, length=2); a.spines[["top", "right"]].set_visible(False)
    pl.label(0.9, 8.3, "Stable ground should read zero")
    t1 = f"Smooth stable ground: median {st['median_m'] * 100:+.1f} cm, robust scatter {st['robust_sd_m'] * 100:.1f} cm, {st['share_over_015'] * 100:.1f}% of points beyond 15 cm. Rough, vegetated patches read lower because the sparse 2014 survey kept low shrubs in its surface, so cells with a spread above 0.10 m ({S['core']['rough_excluded_share'] * 100:.0f}% of the area) are not interpreted."
    pl.text(0.9, 4.65, "\n".join(textwrap.wrap(t1, 68)), fontsize=8.2, color=MUTED, linespacing=1.35)
    # DoD check
    b = pl.ax(6.6, 5.2, 4.2, 2.75)
    ck = S["dod_check"]; names = list(ck)
    xs = np.arange(len(names)); w = 0.38
    b.bar(xs - w / 2, [ck[n]["dod_net_m3"] / 1000 for n in names], w, color="#8A8A8A", label="DEM difference")
    b.bar(xs + w / 2, [ck[n]["m3c2_net_m3"] / 1000 for n in names], w, color="#9A3B12", label="M3C2")
    b.set_xticks(xs); b.set_xticklabels([n.replace(" bluff", "\nbluff").replace(" bank", "\nbank") for n in names], fontsize=8)
    b.axhline(0, color=INK, lw=0.8); b.set_ylabel("net volume change, thousand m³", fontsize=8, color=MUTED); b.legend(fontsize=8, frameon=False, loc="lower right")
    b.tick_params(labelsize=7.5, length=2); b.spines[["top", "right"]].set_visible(False)
    pl.label(6.6, 8.3, "Second method on the same cells")
    t2 = "A DEM of difference (vertical difference times plan area) over exactly the cells M3C2 used: " + "; ".join(f"{n} {v['dod_net_m3']:+,.0f} against {v['m3c2_net_m3']:+,.0f} m³" for n, v in ck.items()) + ". They agree to within 5% where the change is large; the south bank is within noise."
    pl.text(6.6, 4.65, "\n".join(textwrap.wrap(t2, 62)), fontsize=8.2, color=MUTED, linespacing=1.35)
    # surveys table
    pl.label(11.6, 8.3, "The two surveys")
    y = 7.9
    for e in S["epochs"]:
        for line in (f"{e['tag']} survey: {e['project']}", f"collected {e['date']}", f"{e['pts_per_m2']:.1f} points per m² ({e['surface_pts_per_m2']:.1f} on the surface)"):
            pl.text(11.6, y, line, fontsize=8.6, fontweight="bold" if line.startswith(e["tag"]) else "normal", color=INK if line.startswith(e["tag"]) else MUTED); y -= 0.26
        y -= 0.15
    notes = [f"Registration error {S['registration_error_m'] * 100:.1f} cm (the stable-ground scatter) is added to M3C2's level of detection, which is {st['lod_median_m'] * 100:.0f} cm on stable ground; {st['share_over_lod'] * 100:.1f}% of stable points exceed it.",
             "Normals use three scales (2, 4, 6 m) and the most planar is kept; distances are searched in a 1.5 m radius cylinder out to 15 m.",
             "The 2014 survey is 4 times sparser than 2021. M3C2 averages inside the cylinder, so the density difference adds noise but no bias.",
             "Not tested: absolute accuracy (no ground control or survey), tide and wave state on the beach, and whether the two projects share a vertical datum beyond the 1.7 cm agreement on stable ground."]
    y = 3.3
    pl.label(0.9, y, "Limits")
    y -= 0.35
    for t in notes:
        w_ = "\n".join(textwrap.wrap(t, 190))
        pl.text(0.9, y, w_, fontsize=8.5, color=MUTED, linespacing=1.35); y -= 0.19 * (w_.count("\n") + 1) + 0.15
    return pl.save("16_capecod_validation.png")

