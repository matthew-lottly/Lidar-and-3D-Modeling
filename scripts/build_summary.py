"""Writes index.html, the one-page summary of every analysis in this repository. Numbers for the later analyses are read from their stats.json files so the page cannot drift from the results; numbers for the
first four analyses are the ones stated in the README. Usage: python scripts/build_summary.py"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = lambda p: json.loads((ROOT / "data" / "processed" / p).read_text())


def f0(v):
    return f"{v:,.0f}"


def cards():
    cn_h, cn_t = J("canopy/houston.json"), J("canopy/tahoe.json")
    rs_w, rs_d = J("rooftops/westu.json"), J("rooftops/downtown.json")
    td = J("topodiff/stats.json")
    fl = J("flood/stats.json")
    md = J("meander/stats.json")
    sp = J("spans/stats.json")
    fu = J("fusion/stats.json")
    c = []
    c.append(dict(n=1, group="Terrain and land cover", title="Tatoosh Range cirque, Washington", img="outputs/05_land_cover.png", doc="README.md#what-is-here",
                  what="Terrain, canopy and eight land-cover classes from a 60.3 million point lidar cloud, checked against Sentinel-2 and WorldCover; Sentinel-1 radar by season.",
                  stats=[("2.0 km2", "area"), ("6,268", "tree tops"), ("8", "land-cover classes")], limit="Land cover is checked against satellite maps, not field plots."))
    c.append(dict(n=2, group="Buildings and 3D", title="Seattle Center buildings and the Space Needle", img="outputs/09_seattle_buildings.png", doc="docs/seattle-buildings.md",
                  what="Block models at lidar roof height, the Space Needle fitted to its own returns, curved roofs, and aerial imagery corrected with lidar heights and draped in 3D.",
                  stats=[("132 of 139", "footprints measured"), ("12 per m2", "lidar density"), ("0", "photogrammetry (no overlapping photos)")], limit="Roofs are flat blocks except the Needle and the curved roofs; walls are plain."))
    c.append(dict(n=3, group="Change detection", title="Cape Cod bluff erosion, 2014 to 2021", img="outputs/15_capecod_bluff_change.png", doc="docs/coastal-erosion.md",
                  what="M3C2 change along the surface normal between two lidar surveys, with an independent DEM-of-difference volume check.",
                  stats=[("73 m3 per m", "lost from the north face in 7 years"), ("+1.7 cm", "stable-ground bias"), ("within 5%", "DoD volume check")], limit="Vegetated and rough surface (41%) is excluded."))
    c.append(dict(n=4, group="Terrain and land cover", title="Hydro-flattened terrain and micro-drainage", img="outputs/19_hydro_drainage.png", doc="docs/hydrology.md",
                  what="A lake flattened to one level, pits filled and a D8 flow network on the 1 m surface of the Tatoosh cirque.",
                  stats=[("1,286", "pits filled"), ("29.0 km", "defined channels"), ("69%", "of shrub cells a simple ground filter keeps wrongly")], limit="Flood-prone ground is a height-above-drainage proxy, not a flood model."))
    c.append(dict(n=5, group="Utilities", title="Poles, wires and a substation, White River, Washington", img="outputs/20_utility_overview.png", doc="docs/utility-extraction.md",
                  what="Wires and poles found from lidar where the file has no wire or tower classes, compared with mapped features.",
                  stats=[("155", "wire chains (5.7 km)"), ("93%", "on mapped lines"), ("12%", "blind pole recall")], limit="Blind pole detection is weak."))
    c.append(dict(n=6, group="Utilities", title="Fusing lidar, imagery and radar for lines and poles", img="outputs/23_fused_vs_osm.png", doc="docs/fusion-model.md",
                  what="Conductors joined and gaps filled with sag curves, supports scored from three sensors, then checked against OpenStreetMap (via Overture).",
                  stats=[(f"{fu['supports']['ablation']['lidar score']['ranking']['auc']:.2f}", "support AUC, lidar"), ("no gain", "from imagery or radar"), (f"{fu['osm_lines']['coverage_with_interpolation'] * 100:.0f}%", "of mapped line length covered")], limit="Radar at 20 m cannot see towers; the final model was chosen after seeing the results."))
    c.append(dict(n=7, group="Utilities", title="Wires fitted between supports, and a 3D scene", img="outputs/26_scene_overview.png", doc="docs/span-model.md",
                  what="Every pair of nearby anchors tested for a wire (straight in plan, parabolic in height); towers, poles, cross-arms and wires drawn over lidar coloured from imagery.",
                  stats=[(f0(sp["wires"]), "wires"), (f"{sp['wire_length_m'] / 1000:.0f} km", "of wire"), (f"{sp['null']['null_to_real_ratio'] * 100:.0f}%", "false length on shifted data")], limit="Tower proportions and wire thickness in the 3D scene are assumed."))
    c.append(dict(n=8, group="Rivers", title="Scroll bars of a meandering river, Wabash, Illinois", img="outputs/29_meander_scroll_bars.png", doc="docs/meander-scroll-bars.md",
                  what="Crest and swale lines traced from one 2011 lidar flight, with planted-ridge, repeat-fit and imagery tests.",
                  stats=[(f0(md["ridges"]["crest_chains"]), "crest lines"), (f"{md['ridges']['median_relief_m'] * 100:.0f} cm", "median crest relief"), (f"{md['dem']['holdout']['rmse_m'] * 100:.1f} cm", "terrain hold-out error")], limit="Imagery could not give a reliable migration rate, so no ridge ages are claimed."))
    c.append(dict(n=9, group="Forests", title="Canopy structure and connectivity, NW Houston and a Sierra conifer forest", img="outputs/37_canopy_connectivity_maps.png", doc="docs/canopy-structure.md",
                  what="Canopy height, individual trees, gaps, fragmentation and a patch-and-gap connectivity graph at two contrasting sites, with planted-tree tests.",
                  stats=[(f"{cn_h['structure']['canopy_cover_ge2m'] * 100:.0f}% / {cn_t['structure']['canopy_cover_ge2m'] * 100:.0f}%", "canopy cover, Houston / Tahoe"), (f"{cn_h['connectivity']['by_gap']['3']['largest_cluster_share'] * 100:.0f}% / {cn_t['connectivity']['by_gap']['3']['largest_cluster_share'] * 100:.0f}%", "in the largest cluster (3 m gap)"),
                         (f"{cn_h['trees']['n'] + cn_t['trees']['n']:,}", "trees detected")], limit="Tree counts in crowded stands are lower bounds; Houston's connectivity swings with mapping choices."))
    c.append(dict(n=10, group="Buildings and 3D", title="Rooftop solar potential, West University Place and downtown Houston", img="outputs/41_westu_rooftop_solar.png", doc="docs/rooftop-solar.md",
                  what="Roofs found by shape, split into planes by RANSAC, shaded by neighbours and trees, summed over an NREL typical year; a notebook does it for any box and writes .glb and .obj.",
                  stats=[(f"{rs_w['buildings']['n']} / {rs_d['buildings']['n']}", "buildings, West U / downtown"), (f"{rs_w['solar']['beam_shading_loss'] * 100:.0f}% / {rs_d['solar']['beam_shading_loss'] * 100:.0f}%", "direct sun lost to shade"), (f"{rs_w['solar']['roof_area_weighted_beam_kwh_m2'] + rs_w['solar']['diffuse_kwh_m2']:,.0f} kWh/m2", "mean on West U roofs per year")], limit="Roofs under trees are missing; totals move about 10% with the shading surface."))
    c.append(dict(n=11, group="Change detection", title="Topographic differencing, Pacifica bluffs, 2010 to 2023", img="outputs/47_pacifica_dod_maps.png", doc="docs/topographic-differencing.md",
                  what="Three surveys registered on stable ground (ICP), DEMs of difference with a slope-dependent level of detection, M3C2 on the face, cut-and-fill volumes and contour retreat.",
                  stats=[(f"{td['dod']['2016-2023']['coastal_band_60m']['lod95']['net_m3']:,.0f} m3", "net, 2016 to 2023, coastal band"), (f"{td['dod']['2010-2016']['coastal_band_60m']['lod95']['net_m3']:,.0f} m3", "net, 2010 to 2016"), (f"{td['contour_retreat']['12.0']['2016-2023']['median_m']:.1f} m", "median retreat of the 12 m contour, 2016 to 2023")], limit="No field validation; the 2010 survey is sparse and undated."))
    c.append(dict(n=12, group="Floods", title="Drainage and flood-stage road vulnerability, Norfolk, Virginia", img="outputs/53_norfolk_flood_stages.png", doc="docs/flood-drainage.md",
                  what="A 1 m bare-earth terrain conditioned with pysheds, culvert-obstruction candidates at road embankments, and bathtub flooding at stages from 1.5 to 4 m against the drivable road network.",
                  stats=[(f"{fl['dtm_holdout']['rmse_m'] * 100:.1f} cm", "terrain hold-out error"), (f"{fl['road_vulnerability']['3']['any_low_ground']['km_impassable']:.0f} of {fl['roads']['drivable_km']:.0f} km", "roads under 30 cm of water at a 3 m stage"), (f"{fl['stages']['2.5']['raw_stage_minus_0.1']:.0f} to {fl['stages']['2.5']['raw_stage_plus_0.1']:.0f} ha", "flooded at 2.5 m if the stage is off by 10 cm")], limit="Static bathtub: no rain, drains, pumps, tide or storm."))
    return c


def main():
    cs = cards()
    plates = len(list((ROOT / "outputs").glob("[0-9][0-9]_*.png")))
    groups = []
    for c in cs:
        if c["group"] not in groups:
            groups.append(c["group"])
    parts = []
    for g in groups:
        parts.append(f'<h2>{html.escape(g)}</h2><div class="grid">')
        for c in [x for x in cs if x["group"] == g]:
            st = "".join(f'<li><b>{html.escape(a)}</b><span>{html.escape(b)}</span></li>' for a, b in c["stats"])
            parts.append(f'<article><a class="im" href="{c["doc"]}"><img loading="lazy" src="{c["img"]}" alt="{html.escape(c["title"])}"></a><div class="tx"><h3>{html.escape(c["title"])}</h3><p>{html.escape(c["what"])}</p>'
                         f'<ul class="stats">{st}</ul><p class="lim"><b>Limit:</b> {html.escape(c["limit"])}</p><p class="lk"><a href="{c["doc"]}">Method, numbers and limits</a></p></div></article>')
        parts.append("</div>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lidar and 3D Modeling: summary</title>
<style>
:root{{--bg:#fbfaf7;--card:#fff;--ink:#1d1d1b;--mut:#5d5d58;--rule:#dcdad2;--acc:#1f4e79}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151614;--card:#1d1f1c;--ink:#ecebe6;--mut:#a5a59d;--rule:#34362f;--acc:#8fb8de}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:32px 16px 64px}}h1{{font-size:2rem;margin:0 0 8px}}h2{{margin:40px 0 12px;font-size:1.15rem;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);border-bottom:1px solid var(--rule);padding-bottom:6px}}
.lede{{color:var(--mut);max-width:70ch}}.strip{{display:flex;flex-wrap:wrap;gap:12px 28px;margin:20px 0 4px}}.strip div{{min-width:90px}}.strip b{{display:block;font-size:1.6rem;color:var(--acc)}}.strip span{{color:var(--mut);font-size:.85rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:20px}}@media (max-width:600px){{.grid{{grid-template-columns:1fr}}}}
article{{background:var(--card);border:1px solid var(--rule);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}}article .im{{display:block;background:#fff}}article img{{width:100%;display:block;aspect-ratio:8/5;object-fit:cover;object-position:top}}
.tx{{padding:14px 16px 16px}}h3{{margin:0 0 6px;font-size:1.05rem}}.tx p{{margin:6px 0;font-size:.93rem}}.stats{{list-style:none;display:flex;flex-wrap:wrap;gap:8px 20px;padding:0;margin:10px 0}}.stats li b{{display:block;font-size:1.05rem}}.stats li span{{color:var(--mut);font-size:.78rem}}
.lim{{color:var(--mut)}}a{{color:var(--acc)}}.note{{margin-top:44px;padding:14px 16px;border:1px solid var(--rule);border-radius:10px;background:var(--card);font-size:.92rem}}
</style></head><body><main>
<h1>Lidar and 3D modeling</h1>
<p class="lede">{len(cs)} analyses built from public USGS 3DEP lidar, with Sentinel radar, NAIP imagery and Overture maps where they help. Every number on a plate is computed by code in this repository; every analysis has error tests, and the limits are stated next to the results.</p>
<div class="strip"><div><b>{len(cs)}</b><span>analyses</span></div><div><b>{plates}</b><span>plates</span></div><div><b>6</b><span>US states (WA, CA, TX, IL, VA, MA)</span></div></div>
{''.join(parts)}
<div class="note"><b>How to read this.</b> The plates are the results; each card links to the write-up with the method, the error tests and what was not done. Where something could not be verified (no field data, no survey checkpoints, no repeat lidar) the write-up says so. Interactive files: <a href="outputs/point_cloud_viewer.html">Tatoosh point cloud</a>, <a href="outputs/seattle_point_cloud_viewer.html">Seattle point cloud</a>, <a href="outputs/utility3d/white_river_utility_viewer.html">White River utility scene</a>, <a href="outputs/canopy/houston/canopy_map.html">Houston canopy map</a>, <a href="outputs/canopy/tahoe/canopy_map.html">Tahoe canopy map</a>, and the <a href="notebooks/rooftop_solar.ipynb">rooftop solar notebook</a>. Full detail: <a href="README.md">README</a>.</div>
</main></body></html>"""
    (ROOT / "index.html").write_text(page)
    print("wrote index.html", len(cs), "cards,", plates, "plates")


if __name__ == "__main__":
    main()
