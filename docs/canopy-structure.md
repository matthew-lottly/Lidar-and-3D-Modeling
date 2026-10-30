# Canopy structure and fragmentation from lidar: northwest Houston and a Sierra conifer forest

Two contrasting areas, one automated pipeline (`python scripts/build_canopy.py all`, then `python scripts/plot_canopy.py`):

| | NW Houston (Cypress), Texas | Forest north of Truckee, California |
|---|---|---|
| Where | Harris County, 30.00 N 95.69 W, 225 ha | Nevada County, 39.40 N 120.32 W, 100 ha; 0.4 km outside the mapped boundary of Tahoe National Forest (Overture) |
| Lidar | 3DEP `TX_Coastal_B1_2018`, flown 2018-01-19, 8.0 returns per m2 | 3DEP `CA_SierraNevada_5_2022`, flown 2022-08-19, 32.5 returns per m2 |
| Classes in the file | ground (2), low/medium/high vegetation (3 to 5), buildings (6) | only unclassified (1) and ground (2) |
| Canopy cover (CHM 2 m or higher) | 44% | 60% |
| Mean canopy height / 95th percentile | 10.3 m / 21 m | 16.4 m / 33 m |
| Trees detected (per ha), mean height | 11,240 (50), 12.6 m | 9,156 (92), 21.7 m |
| Canopy patches, largest patch share | 3,206, 16% | 2,145, 51% |
| Effective mesh size, canopy edge per ha | 6.6 ha, 1,382 m | 25.9 ha, 2,257 m |
| Area that is open / share of that in openings over 1 ha | 56% / 88% | 40% / 65% |

Site maps: plates 32 (Houston) and 34 (Tahoe): the study square in its region (counties, freeways, rivers, places), in the state, and in aerial imagery. Results: plates 33 and 35. Tests: plate 36.

## Pipeline
1. Stream the area from the 3DEP Entwine Point Tile dataset (Python, `laspy`; not PDAL or lidR, which were not used), drop noise classes 7 and 18.
2. Ground returns (class 2) give the terrain model (1 m, mean per cell, smooth fill of empty cells).
3. Height above ground for every return. Vegetation: classes 3 to 5 where the file has them (Houston; unclassified returns 2 m or more above ground are added). Where it has none (Tahoe), every non-ground return at least 0.5 m above the terrain. Buildings (class 6) are excluded. Returns above 45 m (Houston) or 70 m (Tahoe) are not trees (towers, birds).
4. Canopy height model (CHM = highest return per 1 m cell above the terrain, single-pixel pits filled); surface model = terrain + CHM.
5. Trees: smoothed CHM (1 m), local maxima in a window that grows with height (radius 1 + 0.08 h m, 1.5 to 5 m), watershed from those tops, each crown cut at 40% of its tree's height, crowns under 3 m2 dropped.
6. Cover, patch and edge metrics, effective mesh size, openings (10 m2 or larger), trees per hectare in 50 m cells.

Outputs per site in `outputs/canopy/<site>/`: `dtm.tif`, `dsm.tif`, `chm.tif`, `crowns.tif`, `tree_density_per_ha.tif`, `canopy_gaps.tif`, `trees.csv` (position, height, crown size) and `canopy_map.html`, a self-contained interactive map (layers, pan and zoom, hover a tree). Statistics: `data/processed/canopy/<site>.json`.

## How far to trust it (plate 36; all numbers in the JSON)
- **Terrain:** 10% of ground returns held out, rebuilt without them: 2.9 cm rms in Houston, 6.1 cm in Tahoe (steeper ground), and no worse under canopy of 5 m or more. It is a check of noise and interpolation, not of absolute accuracy (no surveyed points).
- **Planted trees:** trees of known position and height (half-ellipsoid crowns) were scanned at each site's return density and put through the same steps. With regular spacing at 100 trees per ha, 89% (Houston density) and 72% (Tahoe density) are found, and 78% and 41% at 200 per ha; precision is about 100% and heights are within 0.1 m. The missing ones are the short ones under taller neighbours: at 200 per ha the shortest third is found 57% and 17% of the time, the tallest third 94% and 71%. Clumped stands do worse. **Tree counts in crowded stands, and Tahoe's in particular, are lower bounds; canopy height and cover are far better determined than counts.**
- **Thinning the returns:** keeping half or a quarter of the returns finds 88% and 80% of the full-data trees in Houston, 97% and 95% in Tahoe, with height changes of 0.3 to 0.9 m rms.
- **Model resolution:** at 0.5, 1 and 2 m cells canopy cover is 36%, 44% and 56% (Houston) and 54%, 60% and 71% (Tahoe), and tree counts differ (Houston at 2 m finds half as many). Numbers must be quoted with the cell size.
- **Imagery:** 89% (Houston) and 67% (Tahoe) of lidar canopy pixels are vegetation in NAIP (NDVI above an Otsu threshold). NAIP also counts lawn and grass, so the reverse (missed canopy) cannot be tested; the Tahoe figure is lower, plausibly because of relief displacement of tall conifers in the aerial image and shadow between crowns (not tested).
- **Season:** the Houston lidar was flown in January, when deciduous trees are leafless, so their crowns are traced by branches and cover is probably lower than in leaf. The Tahoe vegetation is anything at least 0.5 m above the terrain, so shrubs and snags are included.

## Not done
No field tree inventory was available, so detection accuracy rests on the planted-tree test, not on real stems. Biomass and fuel loads were not estimated. The pipeline handles one area per run; it is not tiled for county-scale use.

## Connectivity (plates 37 and 38; `data/processed/canopy/<site>.json`, key `connectivity`)
Canopy (CHM 2 m or higher) is split into 8-connected patches; patches of at least 20 m2 are nodes, and two nodes are linked when their nearest-patch regions touch, with the gap = closest distance between their edges. Metrics: clusters at a gap-crossing distance, share of canopy in the largest cluster, probability of connectivity and equivalent connected area (Saura and Pascual-Vazquez 2007, p = exp(-gap / alpha) along the best path of gaps), reachable canopy per patch, and stepping stones (cut points). Gap and alpha are scenarios, not a species' ability. Maps use a 3 m gap and a 5 m alpha, because at 10 m and wider almost everything joins at both sites.

| | NW Houston | Tahoe forest |
|---|---|---|
| Patches (20 m2+) / clusters at 3 m | 1,351 / 661 | 624 / 144 |
| Largest cluster at 3 m | 86% of patch canopy (83 ha) | 96% (57 ha) |
| Patches with no neighbour within 3 m | 529 | 117 |
| Reachable canopy (alpha 5 m), share of patch canopy | 75% (72.8 of 96.4 ha) | 92% (54.6 of 59.4 ha) |
| Stepping stones at 3 m | 216 | 111 |

Houston's canopy is held together by a few large stands joined by many small patches; the south-west (a subdivision) is a scatter of isolated patches that reach little. Tahoe is one near-continuous canopy with isolated patches in its openings.
**Sensitivity (plate 38, right):** the reachable share moves a lot with how the canopy is mapped. Houston: 75% baseline, 42% with 0.5 m cells, 93% with 2 m cells, 48% for canopy of 5 m or higher, 30% with a quarter of the returns. Tahoe: 92%, 88%, 97%, 81%, 92%. So Houston's connectivity number is only meaningful with its mapping choices stated; the ranking (Tahoe more connected than Houston) holds in every variant. Patches cut by the window edge continue outside it, so connectivity is understated at the edges.
Extra outputs: `canopy_clusters_3m_gap.tif`, `canopy_reachable_area_ha.tif`, `canopy_stepping_stones.tif`, `canopy_patches.csv`, `canopy_links.csv` (links up to 50 m), plus two connectivity layers in each interactive map.

## Independent-data checks (`scripts/validate_canopy.py`, `scripts/validate_remote.py`; `data/processed/validation/canopy_*.json`)
- **Tahoe, second survey.** The same pipeline on the 2018 survey of the same trees: canopy cover 60% (2022) against 63% (2018), cell agreement 90% (kappa 0.80); 90% of 2022 trees have a 2018 tree within 2 m (96% of trees 30 m or taller, 66% of trees under 10 m); matched trees are 1.3 m taller in 2022 with a robust spread of 0.8 m, which mixes four years of growth with noise and cannot separate them.
- **Ground against USGS's DTM of the same survey (Houston):** bias -0.5 cm, robust spread 2.6 cm, also under canopy (2.7 cm). USGS publishes no DTM for the 2022 Tahoe survey.
- **Satellite land cover.** Houston against ESA WorldCover 2021: accuracy 71%, kappa 0.44, our canopy 44% against WorldCover's 62% tree (lidar canopy of at least 2 m, called tree where half a 10 m cell is covered). Esri's 2022 map labels almost the whole Houston window built area and none of it trees, so it is not a usable reference there. In the Tahoe stand WorldCover and Esri call 98% of cells tree against 60% lidar canopy: every cell we call canopy is tree in their maps (precision 100%), the difference is the open ground between the crowns, which a 10 m map absorbs into "tree".
- **Not done:** FIA field plots for density and height; no second survey for Houston.
