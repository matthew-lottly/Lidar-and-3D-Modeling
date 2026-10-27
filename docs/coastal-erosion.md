# Coastal bluff erosion from repeat 3DEP lidar (M3C2)

**Problem.** Measure how much a coastal bluff lost between two lidar surveys, without gridding the clouds into meshes or DEMs first.

**Site.** A 600 m square of the outer Cape Cod shore in Massachusetts, centred on 41.9128 N, 69.9715 W (UTM 19N). It has a roughly 25 m high sandy bluff, a beach below it, and flat inland ground that serves as the stable reference.

**Data (USGS 3DEP on AWS, `usgs-lidar-public`).**

| Epoch | Project | Collected | Points per m² |
|---|---|---|---|
| 1 | `MA_NE_CMGP_Sandy_Z19_A1_2015` | 2014-04-04 | 3.3 |
| 2 | `MA_CentralEastern_2_2021` | 2021-04-09 | 13.6 |

The 2014 project carries a 2015 label but was flown in April 2014 (dates come from the point GPS times). Both surveys are April, which limits seasonal differences, but beach and wave conditions on the two days are unknown.

## Method (`src/lidar3d/change.py`, `scripts/build_change.py`)

1. **Surface selection without class codes.** The two projects classify differently (2014 labels the bluff face 17 and 18, 2021 labels it ground), so the surface is chosen geometrically: the lowest points per 1 m cell within a slope-dependent tolerance, minus cells far above their neighbourhood (canopy).
2. **M3C2** (Lague et al. 2013) via `py4dgeo`: normals from epoch 1 at three scales (2, 4, 6 m, most planar kept), distances epoch 2 minus epoch 1 along the normal in a 1.5 m radius cylinder out to 15 m. Core points are on a 2 m plan grid.
3. **Registration and level of detection.** Stable inland ground (west of x = -120 m) should read zero: its median is +1.7 cm with robust scatter 4.5 cm. That scatter (4.5 cm) is added to M3C2's 95 % level of detection (13 cm median on stable ground); 1.2 % of stable points exceed it. No ICP was applied because the stable ground already agrees.
4. **Rough surface is not interpreted.** On stable ground, patches where either epoch's local spread exceeds 0.10 m read much lower (down to -1 m). The cause is vegetation: the sparse 2014 survey kept low shrubs in its surface while 2021 reached the true ground. Those cells (41 % of the area) are masked.
5. **Volumes.** Normal distance times surface area (plan area divided by |n_z|), significant cells only, with the registration error times the surface area as the uncertainty.

## Results

| Zone (6 to 26 m elevation) | Net change | Loss | Gain | ± registration |
|---|---|---|---|---|
| North bluff face (180 m of shore) | -13,108 m³ | 13,366 | 258 | 295 |
| Middle bluff face (180 m) | -7,356 m³ | 7,568 | 211 | 293 |
| South bank (200 m) | +190 m³ | 322 | 512 | 1,122 |

* The north bluff lost about 73 m³ per metre of shore in 7 years (about 10 m³ per metre per year), the middle about 41 (6 per year). The largest local lowering is 4.6 m. The south bank shows no change beyond the uncertainty.
* **Independent check.** A DEM of difference over exactly the same cells gives north -12,605 m³ against M3C2's -13,077, and middle -7,020 against -7,346: within 5 %. The two methods share the surface selection, so this checks the distance and volume calculation, not the selection.
* The **beach** (below 3 m) shows a net gain of +2,449 m³ with large gains and losses; it is reported on plate 15 but not interpreted, because tide, waves and swash differ between the surveys.

## Files

* `outputs/15_capecod_bluff_change.png`: change map, three profiles, volumes.
* `outputs/16_capecod_validation.png`: stable-ground check, DEM comparison, limits.
* `data/processed/capecod/m3c2_core_points.csv`, `.ply`: core points with distances (UTM 19N, EPSG:32619) for CloudCompare or any GIS.
* `data/processed/capecod/m3c2_significant_change_2m.tif`: 2 m raster of significant change (NaN elsewhere).
* `data/processed/capecod/stats.json`: every number quoted here.

## Limits

* **CloudCompare was not used.** It could not be installed here. `py4dgeo` implements the same M3C2 algorithm, but results are not guaranteed identical to CloudCompare's plugin; the CSV and PLY open there for a side-by-side run with the parameters above.
* **No ground truth.** Absolute accuracy was not tested against GNSS or survey data; the only reference is the stable ground, whose choice (inland, west of x = -120 m) was made by looking at the data.
* **Vegetated and rough areas are not measured** (41 % of the area), including much of the beach and the vegetated crest. Erosion there is missing from the totals.
* The 2014 survey is about 4 times sparser; this raises noise but not bias in M3C2.
* A 3D view of the change was tried and dropped: the steep face is sampled unevenly in the 2021 surface points and the render was speckled.
