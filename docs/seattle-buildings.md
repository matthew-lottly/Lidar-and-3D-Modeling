# Seattle Center buildings from 3DEP lidar

## Inputs

* **Area:** a 700 m square centred on 47.6215 N, 122.3510 W (UTM 10N), covering Seattle Center: the Space Needle, Climate Pledge Arena, MoPOP, McCaw Hall, the Armory and the blocks around them.
* **Lidar:** USGS 3DEP `WA_KingCo_1_2021`, collected 2021-04-14, 5.95 million points in 0.49 km² (12 per m²). Only classes 1 (unclassified) and 2 (ground) are present, so there is no building class: buildings come from footprints plus height above ground.
* **Footprints:** Overture Maps buildings, release 2026-09-23.1, clipped to the area (139 footprints, 26 of them cut by the edge). They come from other open datasets and are five years newer than the lidar.
* **Aerial imagery:** USDA NAIP 2019-10-11, 0.6 m, RGB + near-infrared, read as a window from the Microsoft Planetary Computer mirror. There is no 2021 NAIP flight for Seattle, so the photo is 18 months older than the lidar; buildings built or demolished in between (the Climate Pledge Arena site is a construction yard in the photo) will not match.

## Method

**Measuring each footprint** (`src/lidar3d/buildings.py`)

1. Per 0.5 m cell: highest return (surface) and lowest ground-class return.
2. Ground level = median of ground returns in a 4 m ring around the footprint, outside every footprint (12 m if the ring is empty); base = 10th percentile of the same returns.
3. Roof = surface cells inside the footprint shrunk by 0.75 m (to avoid walls). Main roof = 70th percentile; the highest return is reported as well.
4. Roof shape = a RANSAC plane (25 cm tolerance) on those cells. Flat: the plane covers over half the cells and tilts under 5 degrees; pitched: over half, steeper; complex: no plane covers half the roof (rooftop equipment, terraces and curved roofs all land here).
5. LOD1 block = footprint extruded from the base to the main roof. Footprints under 15 m², with no lidar cells or no ground nearby, or standing under 0.5 m, are skipped.

**The Space Needle** (`src/lidar3d/needle.py`). The footprint is the saucer outline, so extruding it gives a 163 m cylinder. The 22,089 returns on the tower (above 18 m, within 30 m of the axis) are close to axisymmetric (saucer radius 20.3 to 21.7 m, leg radius 12.6 to 15.4 m across eight compass sectors). The 95th-percentile radius of the returns in each 2 m of height is revolved around the axis into a closed mesh: volume 68 thousand m³ against 182 thousand for the cylinder, highest point 184.5 m (Overture: 184 m).

* Recovered from the data: the hourglass shaft (14 m radius at 20 m height narrowing to about 4.5 m near 105 m), the wider ring near 34 m, the saucer (widest, 21 m radius, near 157 m), the top house and the spire.
* Interpolated, not measured: 54 m of height (about 107 to 153 m) where the open steel returns almost nothing; the spire above the top house is a thin cylinder set by four returns; below 18 m the radius is held constant because surrounding pavilions contaminate the returns.
* Not modelled: the three legs, glass and halo detail. The shaft is solid because the lidar gives a silhouette.

**Aerial imagery on the models** (`src/lidar3d/naip.py`)

* **Why it needs correcting.** NAIP is flattened onto the bare ground, so the top of a tall building appears displaced away from the point under the camera. The Space Needle's saucer appears 65 m west of its footprint.
* **Fitting the shift.** For 24 well-matched tall buildings the displacement of the roof outline in the image was measured (cross-correlating the lidar footprint outline with the image edges), and a shift proportional to height above ground, `h * (k*x + b)`, was fitted. The camera position is not recoverable from a 700 m area, so only the fitted shifts are used.
* **Applying it.** Each roof is shifted as one rigid piece by its lidar roof height, which cannot distort the photo; ground and trees use per-cell lidar heights.
* **Curved and stepped roofs** (the 25 footprints over 150 m² whose roof height varies by more than 6 m). A rigid shift is wrong across a roof like MoPOP's, and per-cell shifts from the raw surface tear the image because glossy metal returns no lidar points (16 % of MoPOP's roof cells are empty) and a blended fill gives wrong heights. For these roofs dropouts take the nearest real return inside the footprint, the heights are Gaussian-smoothed, and the roof is built in 3D as that surface (about 1.2 m triangles) so the photo lies on the shape it was taken from. Their sides take the roof edge's colour from the photo, darkened.
* **Evidence.** Over 39 tall footprints the median offset between each outline and its best image match fell from 2.7 m to 1.3 m, and the match score rose for 77 % of them (median score 0.13 to 0.21). A rigid shift for every building scored higher (0.23) but distorts the curved roofs, so those use the smoothed per-cell version. The fit still leaves about 3 m typical error on some buildings. About 8 % of the area is hidden behind tall buildings in the photo and is filled from neighbouring pixels, so it is not real imagery.

## Results

* 132 footprints measured: 76 flat, 10 pitched, 46 complex roofs. Median height 8.7 m.
* Comparison with Overture's own `height` (114 buildings): the main roof level correlates at r = 0.94 and reads 3.0 m lower at the median (mean absolute difference 5.3 m, 41 % within 3 m); the highest return correlates at r = 0.95, median +0.7 m, mean absolute 3.7 m, 64 % within 3 m. This is two datasets agreeing or not, not an accuracy test: there is no surveyed height.
* The Space Needle's saucer (main roof) is at 162.9 m, its spire (highest return) at 184.5 m; Overture says 184 m.
* Large disagreements are listed on `outputs/11_seattle_height_check.png`. Examples: 222 Fifth (Overture 38.1 m, lidar highest 15.9 m) and MoPOP (Overture 9.7 m, lidar 30.8 m). Why was not checked; a building completed after April 2021 would explain the first. The red on MoPOP is the real roof colour: it is in the unprocessed photograph.

## Files

* `data/processed/seattle/seattle_center_lod1.city.json`: CityJSON 1.1. LOD1 Building solids with roof, wall and ground surfaces and attributes; the Space Needle is an LOD2 solid. Every solid is a closed, consistently oriented shell (`exports.cityjson_open_edges` found none open). It has **not** been checked against the CityJSON schema validator, which could not be installed here.
* `.../seattle_center_textured.glb`: terrain and roofs textured with the corrected image, walls plain; curved roofs follow the lidar surface, the Space Needle is the fitted shape.
* `.../seattle_center_lod1.glb`: terrain and blocks coloured by height; the Space Needle is the fitted shape.
* `.../buildings_lod1.gpkg`: footprints with all measured attributes.
* `.../naip_2019_lidar_corrected.tif`: the corrected 4-band image.
* `.../stats.json`: every number quoted here and on plates 09 to 14, written by `scripts/build_seattle.py`.
* `outputs/seattle_point_cloud_viewer.html`: 600,000 points, any browser.

## Limits

* **Flat roofs.** Roofs are flat blocks except the Space Needle and the curved roofs; the CityJSON and the block GLB keep the curved roofs as blocks. The fitted planes are only used to label roof shape.
* **Trees and non-building structures** inside a footprint are not separated from the roof.
* **No photogrammetry and no facade textures.** NAIP is a single downward image; reconstructing building faces needs overlapping oblique photos. See [photogrammetry.md](photogrammetry.md).
