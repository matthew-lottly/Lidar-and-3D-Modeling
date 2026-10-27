# Hydro-flattened terrain and micro-drainage from lidar

**Problem.** Derive a terrain model that water can run across (lakes flat, pits filled) and find the small drainage channels that map and satellite products miss.

**Site.** The 1.4 km square Tatoosh Range cirque used for the first analysis (USGS 3DEP `WA_CentralWildfire_2_D22`, collected 2022-09-04 to 2022-09-08, 31 points per m²). It has one lake (7,618 m²), steep slopes, and no tree cover to speak of, so the terrain is well sampled. It is an alpine basin, not a floodplain: the flood-risk part is a terrain proxy (below), not an inundation model.

## Method (`src/lidar3d/hydro.py`, `scripts/build_hydro.py`)

1. **Ground classification.** The terrain model uses the published ground class (2): the lowest ground return per 1 m cell, gaps filled by interpolation. As a check, a progressive morphological filter (Zhang et al. 2003, windows 3 to 33 cells, slope-aware thresholds) was run on the lowest return per cell. It keeps 98% of cells and misses 1.2% of cells that sit on the terrain, but also keeps 69% of the cells whose lowest return is more than 0.5 m above the published terrain (47% inside canopy). Those cells have no ground return, and a grid of lowest returns cannot tell shrubs and boulders from ground, so the published class is used. The filter is shown on plate 18, not used downstream.
2. **Hydro-flattening.** Water-classified returns (74,828) are gridded, closed, filled and cleaned into a water mask (one body). Water has no ground returns, so the raw model interpolates across the lake from its shore: the surface varies by 10 cm (5th to 95th percentile 1815.85 to 1816.19 m). The mask is set to one level, the median of its water returns, 1815.92 m.
3. **Depression filling.** Morphological reconstruction fills every closed pit so water can leave through the grid edge: 1,286 pits, 237 m³ in all, deepest 1.25 m. The lake needed no filling (it already sits at its spill level).
4. **Flats.** Flat areas get a gradient of 0.1 mm per cell toward their outlets, far below the lidar's precision, so D8 can route across them. 6 interior sink cells remain, from flats with no outlet.
5. **D8 flow direction and accumulation.** Each cell drains to its steepest lower neighbour (diagonals divided by √2); accumulation sums the cells draining through each cell, processed from highest to lowest.
6. **Channels.** A cell is a defined channel when its upstream area is at least 1,000 m² **and** it lies at least 0.5 m below the mean of its 15 m surroundings. The upstream-area threshold is a choice, so the plate shows five.
7. **Height above nearest drainage (HAND)** from the flow directions: each cell's elevation minus that of the channel cell it drains to.

## Results

* 29.0 km of defined channel, 15 km per km². Without the hollow test, flow paths above 1,000 m² total 56.4 km, and 49% of the extra ones sit less than 0.5 m below their surroundings, so they are hillslope flow lines, not channels.
* **Mapped streams.** Overture Maps has no stream or river line anywhere in this area, so all 29.0 km of defined channel are missing from that map. This shows the map is sparse; it does not show that satellite imagery cannot see the channels. No imagery was compared, and a 10 m pixel is wider than most of them.
* **Flood-prone ground (terrain proxy).** 3.8% of the area is within 1 m of the drainage in height, 7.2% within 2 m (against 3.4% and 5.7% for streams of 5,000 m² or more upstream). The lake's catchment is 5.9 ha.

## Files (`data/processed/hydro/`)

* `dtm_hydroflattened_1m.tif`, `dtm_filled_1m.tif`, `flow_accumulation_m2.tif`, `d8_flow_direction.tif` (codes 0 = E, 1 = SE, 2 = S, 3 = SW, 4 = W, 5 = NW, 6 = N, 7 = NE; -1 = leaves the grid), all EPSG:32610 at 1 m.
* `stats.json`: every number quoted here. `outputs/18_hydro_flattening.png`, `outputs/19_hydro_drainage.png`.

## Limits

* **Not a flood model.** No rainfall, snowmelt, infiltration, roughness, culverts or channel hydraulics. HAND marks ground that is low relative to the drainage, nothing more. A lake-rise inundation scenario was tried and dropped: this lake sits at its spill point on a steep slope, so a rise only connects it to all lower ground.
* **D8 on a 1 m lidar surface is noisy** (single-direction flow, boulders, micro-relief); channel length depends strongly on the threshold (plate 19). The depression filling removes real closed hollows as well as noise.
* **Alpine, snow-free September surface.** Channels under snow or vegetation, and under the lake, are not seen. Breaklines for bridges and culverts were not needed here and are not handled.
* The hydro-flattening has one body and no rivers; river flattening (monotone downstream levels along a channel) is not implemented.
