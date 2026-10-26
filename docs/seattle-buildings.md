# Seattle Center buildings from 3DEP lidar

**Area:** a 700 m square centred on 47.6215 N, 122.3510 W (UTM 10N), covering Seattle Center: the Space Needle, Climate Pledge Arena, MoPOP, McCaw Hall, the Armory and the blocks around them.

**Lidar:** USGS 3DEP `WA_KingCo_1_2021`, collected 2021-04-14, 5.95 million points in 0.49 km² (12 per m²). Only classes 1 (unclassified) and 2 (ground) are present, so there is no building class: buildings come from footprints plus height above ground.

**Footprints:** Overture Maps buildings, release 2026-09-23.1, clipped to the area (139 footprints, 26 of them cut by the edge). Overture's footprints come from other open datasets and are newer than the lidar.

## Method (`src/lidar3d/buildings.py`)

1. Per 0.5 m cell: highest return (surface) and lowest ground-class return.
2. Per footprint: ground level = median of ground returns in a 4 m ring around it, outside every footprint (12 m if the ring is empty); base = 10th percentile of the same returns.
3. Roof: the surface cells inside the footprint shrunk by 0.75 m (to avoid walls). Main roof = 70th percentile; the highest return is reported as well.
4. Roof shape: a RANSAC plane (25 cm tolerance) on those cells. Flat: plane covers over half the cells and tilts under 5 degrees; pitched: over half, steeper; complex: no plane covers half of the roof (rooftop equipment, terraces and curved roofs all land here).
5. LOD1 solid: footprint extruded from the base to the main roof. Footprints under 15 m², with no lidar cells or no ground nearby are skipped, as are ones that stand under 0.5 m.

## Results

* 132 footprints measured; 76 flat, 10 pitched, 46 complex. Median height 8.7 m.
* Comparison with Overture's own `height` (114 buildings): the main roof level correlates at r = 0.94 and reads 3.0 m lower at the median (mean absolute difference 5.3 m, 41 % within 3 m); the highest return correlates at r = 0.95, median +0.7 m, mean absolute 3.7 m, 64 % within 3 m. This is two datasets agreeing or not, not an accuracy test: there is no surveyed height.
* The Space Needle's footprint is the saucer outline: main roof 162.9 m, highest return 184.5 m (spire), Overture 184 m. The fitted tower model is described below.
* Large disagreements are listed on `outputs/11_seattle_height_check.png`. Examples: 222 Fifth (Overture 38.1 m, lidar highest 15.9 m) and MoPOP (Overture 9.7 m, lidar 30.8 m). Why was not checked; a building completed after April 2021 would explain the first.

## Files

* `data/processed/seattle/seattle_center_lod1.city.json`: CityJSON 1.1, LOD1 Building solids with roof, wall and ground surfaces and attributes. Every solid is a closed, consistently oriented shell (checked by `exports.cityjson_open_edges`, which found none open). It has **not** been checked against the CityJSON schema validator, which could not be installed here.
* `.../seattle_center_lod1.glb`: terrain mesh plus building blocks coloured by height.
* `.../buildings_lod1.gpkg`: footprints with all measured attributes.
* `.../seattle_center_textured.glb` and `.../naip_2019_lidar_corrected.tif`: imagery draped on the models, and the corrected 4-band image.
* `outputs/seattle_point_cloud_viewer.html`: 600,000 points, any browser.

## The Space Needle (`src/lidar3d/needle.py`, plate 13)

The footprint extrusion made the Needle a 163 m cylinder, so it is modelled separately. The lidar has 22,089 returns on the tower (above 18 m, within 30 m of the axis). They are close to axisymmetric (saucer radius 20.3 to 21.7 m and leg radius 12.6 to 15.4 m across eight compass sectors), so the 95th-percentile radius of the returns in each 2 m of height is revolved around the axis to give a closed mesh (volume 68,000 m³ against 182,000 m³ for the cylinder; highest point 184.5 m, Overture 184 m).

* **Recovered from the data:** the hourglass shaft (14 m radius at 20 m height narrowing to about 4.5 m near 105 m), the wider ring near 34 m, the saucer (widest, 21 m radius, near 157 m), the top house and the spire.
* **Interpolated, not measured:** 54 m of height (about 107 to 153 m) where the open steel returns almost nothing, drawn dashed on plate 13; the spire above the top house is a thin cylinder set by four returns; below 18 m the radius is held constant because surrounding pavilions contaminate the returns.
* **Not modelled:** the three legs, glass, the halo's detail. The shaft is a solid, because the lidar gives a silhouette. Upward faces take the aerial photo; side faces are plain.
* The CityJSON stores it as an LOD2 solid (closed shell, no open edges) for this one building; all others stay LOD1.

## Aerial imagery on the models (`src/lidar3d/naip.py`)

* **Imagery:** USDA NAIP 2019-10-11, 0.6 m, RGB + near-infrared, read as a window from the Microsoft Planetary Computer mirror. No NAIP flight exists for 2021, so the photo is 18 months older than the lidar and 7 years older than the footprints; buildings built or demolished in between (the Climate Pledge Arena site is visibly a construction yard in the photo) will not match.
* **Why it needs correcting:** NAIP is flattened onto the bare ground, so the top of a tall building appears displaced away from the point under the camera. The Space Needle's saucer appears 65 m west of its footprint. Draping the raw photo on the models would put roofs in the wrong place.
* **Correction:** for 24 well-matched tall buildings, the displacement of each roof outline in the image was measured (cross-correlating the lidar footprint outline with the image edges), and a shift proportional to height above ground (`h * (k*x + b)`) was fitted. Every surface cell then takes its colour from the shifted image position. The camera position is not recoverable from a 700 m area, so only the fitted shifts are used.
* **Evidence:** over 39 tall footprints the median offset between each outline and its best image match fell from 2.7 m to 1.3 m; the match score rose for 74 %. Typical fit error is still about 3 m on some buildings. About 9 % of the area is hidden behind tall buildings in the photo and is filled from neighbouring pixels, so it is not real imagery.
* **Model:** `seattle_center_textured.glb` has the terrain and roofs textured with the corrected image and plain grey walls: no photo shows a wall. It looks photographic from above and like a block model from the side.

## Not done

* **No LOD2 roofs** (except the fitted Space Needle). Roofs are flat blocks. The fitted planes are used only to label roof shape.
* **No photogrammetry and no facade textures.** NAIP is a single downward image; reconstructing building faces needs overlapping oblique photos (Mapillary and Wikimedia were not reachable). See [photogrammetry.md](photogrammetry.md).
* **Trees and non-building structures** inside a footprint are not separated from the roof.
