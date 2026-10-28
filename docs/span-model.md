# Span-anchored wires, supports and a 3D scene (White River, WA), lidar only

Same site and data as `fusion-model.md`. Plates 25 to 27; outputs in `outputs/utility3d/`; tables in `data/processed/spans/`.
Run `python scripts/build_spans.py` then `python scripts/build_spans_3d.py` (about 10 minutes the first time, cached after).
The model uses lidar returns only. NAIP imagery only colours the 3D scene. OpenStreetMap (via Overture) is read at the end to compare.

## Why the earlier model missed wire
The earlier model grew wires from local fragments. Where returns were sparse, a fragment was dropped and the wire stopped, so a whole span could be missing.
The new model reverses the question: for each pair of nearby anchors it asks whether the elevated returns in the corridor between them contain a wire (straight in plan, parabolic in height), and accepts the span if at least half of it has returns.
The part with no returns is then filled from the same curve. Anchors come from the earlier sag-peak supports and the ends of detected wire chains (191 anchors, 619 pairs tested).

## Results (all measured by `scripts/build_spans.py`; `data/processed/spans/stats.json`)
- **620 wires, 32.3 km, 80% of that length with lidar returns**, the rest filled. Wire returns found by the earlier detector (outside the yard): 65% lie within 1 m of a fitted wire.
- **False wires.** The same search on lidar shifted 120 to 250 m (4 shifts) accepts 10% as much wire length. This is an upper bound, since shifted returns still land on real wires elsewhere.
  The acceptance rule (at least half the span has returns, at least 25 returns) was chosen from this shifted-data test, not from the map. The search first ran at looser settings (30% coverage), where the shifted data gave 17%.
- **Interpolation test.** A stretch of a well-observed wire (84 tests per size) is hidden and refitted: error inside the hidden stretch is 4 cm rms for the fitted curve against 8 cm (20 m) and 24 cm (40 m) for a straight line.
  The hidden returns were inliers of the original fit, so this is optimistic for noisy wires. Tying the curve to the sag of neighbouring wires in the same span did not improve it (5 cm), so it is not used.
- **Against the map.** 44% of mapped line length (outside the yard) lies within 6 m of a fitted wire, against 43% for the earlier model. Nearly all fitted wire (100%) lies within 10 m of a mapped line.
  So the main gain is more complete wire along the lines both models find, not more mapped length reached. Of the mapped length with no wire within 6 m, 31% has a fitted wire within 15 m, which suggests a position offset in the map; for the rest the median distance to the nearest fitted wire is 30 m. Whether those lines are missing, de-energised, mapped in the wrong place or simply not visible is not established here and was not checked by eye.

## Supports
A support is a place where fitted wires end and no unbroken wire passes over (a suspension tower would pass them; a termination or anchor would not). 59 supports (45 towers, 14 poles); the type is inferred from height and arm width, not measured.
Outside the yard, 44 supports; 17 lie within 5 m of one of the 55 mapped towers or poles (precision 39%, recall 29%). Counting every wire end as a support (120) gives precision 18% and recall 36%: the unbroken-wire test removes many false supports at the cost of some mapped ones.
A further 89 wire ends with a wire passing over are drawn as thin posts so no wire ends in mid-air.
Positions come from where the wires attach. Cross-arm heights and widths come from the attachment heights and sideways offsets.

## 3D scene
`outputs/utility3d/`: `white_river_lines_and_supports.glb` (supports, cross-arms, insulators, wire tubes), `white_river_ground_naip.glb` (ground mesh coloured from NAIP), and `white_river_utility_viewer.html` (rotatable point-cloud scene with imagery colours; drag to rotate, shift-drag to pan, wheel to zoom).
Ground is the lidar ground grid; vegetation, buildings and other returns are lidar points coloured from the 0.6 m NAIP image (the 2019 image against 2020 lidar, with leaf-on shadows, so colours on vertical surfaces and tree edges are approximate).
**Assumed, not measured:** pole radius, lattice-tower taper and bracing, insulator length, wire radius (drawn 4.5 cm for visibility). Only positions, heights, cross-arm tiers and wire curves come from the data.

## Not done
- Wire radius, conductor type, towers' true cross-sections and any structure detail need ground or oblique imagery or closer lidar.
- The mapped lines with no wire nearby were not inspected visually.
- Anchors still come in part from the earlier detector; a fully independent anchor search (for example from tower point clouds) is the obvious next step.
