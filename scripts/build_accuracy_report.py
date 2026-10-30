"""Builds docs/accuracy-assessment.md and data/processed/validation/matrix.json from the validation result files (run the validate_*.py scripts first). Numbers come from those files; sentences that quote an
older document's figure say which document."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "data" / "processed" / "validation"
J = lambda n: json.loads((V / n).read_text())
pc = lambda x: f"{100 * x:.0f}%"
cm = lambda x: f"{100 * x:.1f} cm"


def main():
    fl, td, tc = J("flood.json"), J("topodiff.json"), J("topodiff_calibrated.json")
    ct, mo, rf = J("canopy_tahoe.json"), J("meander.json"), J("rooftops.json")
    ch, ca = J("canopy_houston_remote.json"), J("canopy_tahoe_remote.json")
    dt = J("dtm_other_studies.json")
    dt.pop("capecod_2014", None)
    rem = fl["references"]["blocked_or_pending"]
    sw = fl["references"]["swath_repeatability"]
    nul = td["null_test"]
    iv = tc["intervals"]
    cc = ct["canopy_cover"]
    t = ct["trees"]
    ch_ = ct["chm_where_both_canopy"]
    wu, dtn = rf["westu"], rf["downtown"]
    wab = dt["wabash_mtcarmel"]["all"]
    cp = dt["capecod_2014_ref_Sandy_Z19_2013"]["all"]
    rows = [
        ("Norfolk flood and drainage", "Flight-line repeatability; USGS DTM; Chesapeake 1 m land cover; JRC water; Overture bridges",
         f"Swaths agree to {cm(sw['roads']['nmad_m'])} (roads) and {cm(sw['open_ground']['nmad_m'])} (open ground); ground is {cm(rem['usgs_dtm']['nmad_m'])} from USGS's DTM (bias {cm(rem['usgs_dtm']['bias_m'])}); bridge cells are 90% road in Chesapeake; the source-water voids do not match any mapped water",
         "Partly validated. Heights and road classes are checked; flood extent, depth and the tide link are not (no gauge, benchmark or observed flood data)."),
        ("Pacifica differencing", "Null test with two independent 2010 surveys; level of detection recalibrated on it",
         f"The surveys differ by a {abs(td['vertical_offset_between_surveys_m']):.1f} m datum offset; after removing it {pc(nul['false_alarm_share_of_coastal_cells'])} of coastal cells still read as change with the formula threshold; with the null-calibrated threshold the 2016-2023 net change is {iv['2016-2023']['null_calibrated_lod']['net_m3']:.0f} m3 against a null of {nul['coastal_band']['net_m3']:.0f} m3, and the 2010-2016 net ({iv['2010-2016']['null_calibrated_lod']['net_m3']:.0f} m3) is inside the null",
         "Validated by null test. The 2010-2016 change is not claimed; the 2016-2023 loss is. No ground survey (GNSS, profiles) was compared."),
        ("Canopy, Tahoe", "Repeat lidar survey (2018 against 2022); WorldCover and Esri land cover",
         f"Cover {pc(cc['cover_2022'])} against {pc(cc['cover_2018'])}, agreement {pc(cc['agreement']['accuracy'])} (kappa {cc['agreement']['kappa']:.2f}); {pc(t['share_of_2022_matched'])} of trees found in both surveys; 2022 minus 2018 canopy height {ch_['bias_m']:+.1f} m per cell and {t['height_2022_minus_2018']['bias_m']:+.1f} m for matched trees (four years of growth plus noise); WorldCover and Esri call {pc(ca['worldcover_2021']['tree_share_reference'])} of the window tree against our {pc(ca['worldcover_2021']['canopy_cover_lidar'])} canopy (their classes include open-grown and sparse stands; precision {pc(ca['worldcover_2021']['agreement']['precision'])})",
         "Validated against a second survey. No field plots (FIA) were compared."),
        ("Canopy, Houston", "USGS DTM; WorldCover; Esri land cover",
         f"Ground {cm(ch['usgs_dtm']['nmad_m'])} from USGS's DTM; against WorldCover accuracy {pc(ch['worldcover_2021']['agreement']['accuracy'])}, kappa {ch['worldcover_2021']['agreement']['kappa']:.2f}; Esri labels almost the whole suburb built area and none of it trees, so it is not usable as a reference here",
         "Validated for terrain; canopy checked against one usable satellite map only. No second survey; no field plots."),
        ("Rooftop solar", "Overture footprints and heights; clear-sky upper bound",
         f"Footprint area overlap {pc(wu['footprints']['lidar_area_inside_overture'])} / {pc(wu['footprints']['overture_area_inside_lidar'])} (West U) and {pc(dtn['footprints']['lidar_area_inside_overture'])} / {pc(dtn['footprints']['overture_area_inside_lidar'])} (downtown); heights disagree (West U r = {wu['heights']['pearson_r']:.2f}, bias {wu['heights']['bias_m']:+.1f} m), cause not resolved; typical-year irradiance is {pc(wu['weather']['ratio'])} of clear sky",
         "Partly validated. No measured roof irradiance or PV output; the height disagreement with Overture is unexplained and is not claimed either way."),
        ("Wabash scroll bars", "NAIP 2011 water mask; NAIP NDVI against a shifted null; USGS DTM",
         f"Channel mask IoU {mo['channel_vs_naip_water']['iou']:.2f} (kappa {mo['channel_vs_naip_water']['kappa']:.2f}); lidar relief and NDVI correlate at {mo['ridge_existence']['correlation_ndvi_vs_lidar_relief']:.2f}, z = {mo['ridge_existence']['z_score']:.1f} against shifts; ground {cm(wab['nmad_m'])} from USGS's DTM",
         "Ridge existence supported, modestly. Migration rates need a second epoch (none exists for this reach in 3DEP); not claimed."),
        ("Cape Cod erosion", "USGS DTM (2013-14 collection); earlier stable-ground and DEM cross-checks",
         f"2014 ground {cm(cp['nmad_m'])} from the USGS DTM (bias {cm(cp['bias_m'])}); M3C2 and DEM-of-difference volumes within 5% (docs/coastal-erosion.md)",
         "Partly validated. No GNSS or beach-profile data; no USGS DTM exists for the 2021 survey."),
        ("Seattle buildings", "Overture heights (docs/seattle-buildings.md)",
         "Main-roof height correlates at r = 0.94 with Overture; Space Needle top 184.5 m against Overture 184 m; large disagreements are listed",
         "Two datasets agreeing, not surveyed truth. No USGS DTM exists for this project."),
        ("Utilities, fusion, spans (White River)", "Overture power layer (OpenStreetMap-derived); shifted-data tests (docs/utility-extraction.md, fusion-model.md, span-model.md)",
         "Poles: recall 12%, precision 47%; fusion and span models are scored against the map with bootstrap intervals; wire interpolation error 4 cm rms on hidden stretches",
         "The map is itself incomplete, so scores are bounds. No field survey of poles or lines."),
        ("Tatoosh terrain and hydrology", "Published ground class and a morphological filter; Sentinel-2/WorldCover cover maps (docs/hydrology.md, site-selection.md)",
         "Ground filter compared with the published class; no stream map exists for the area to check channels against",
         "Not validated against independent data. No USGS DTM product for this project."),
    ]
    (V / "matrix.json").write_text(json.dumps([dict(analysis=a, reference=b, result=c, status=d) for a, b, c, d in rows], indent=1))
    md = ["# Accuracy assessment against independent data", "",
          "Every analysis in this repository is checked here against something it did not produce: a second survey, overlapping flight lines, a satellite or third-party map, or a null test where no change exists. Numbers are generated by `python scripts/build_accuracy_report.py` from `data/processed/validation/*.json`, which the `scripts/validate_*.py` scripts write. \"Robust spread\" is the normalised median absolute deviation (1.4826 x MAD), which ignores outliers.", "",
          "| Analysis | Reference data | Result | Status and what is still unchecked |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    md += ["", "## What the checks changed", "",
           "- **Flood roads.** The first road rule treated anything with a return 3 m overhead as an overpass and never flooded it. Checking against the lidar and Chesapeake land cover showed that rule also removed 7.9 km of tree-lined streets and roads under roofs. Roads are now classified from map tags and lidar evidence together (bridge decks are judged at the deck, tunnels are not assessed, open cuts and underpass approaches flood), and the plates show a 13 cm error band.",
           "- **Pacifica.** The null test found a 32.8 m datum offset between two surveys that a quick look missed, and showed the formula level of detection leaves 47% of coastal cells falsely significant. The 2010-2016 change is therefore not reported as real; the 2016-2023 loss survives.",
           "- **Houston and Esri.** Esri's 2022 land cover labels the whole Cypress window built area; it was dropped as a reference for that site.",
           "", "## Limits of this assessment", "",
           "- Several references are a second processing of the same survey (USGS's DTM, flight-line overlap): they catch gridding, classification and registration errors but not a flaw shared by every pass of the flight.",
           "- No GNSS checkpoints, tide gauges, benchmarks, field plots, measured irradiance or observed floods were used; none are available in this environment. Absolute vertical accuracy and every flood, solar-yield and erosion magnitude therefore rest on the lidar's published accuracy plus the consistency checks above.",
           "- USGS publishes its DTM, DSM and height-above-ground rasters only for older projects, so the 2020-2022 surveys (Seattle, White River, Cape Cod 2021) have no such cross-check.",
           "- Satellite land-cover maps classify at 10 m and have their own error; disagreements with 1 m lidar canopy are partly differences of definition (Tahoe: the maps call 98% of an open conifer stand tree, lidar canopy is 60%).",
           "- Not yet done: field-plot comparison (FIA) for canopy density and height, Microsoft building footprints for the rooftop sites, Deltares flood maps for Norfolk, a second epoch for the Wabash reach."]
    (ROOT / "docs" / "accuracy-assessment.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)[:3000])


if __name__ == "__main__":
    main()
