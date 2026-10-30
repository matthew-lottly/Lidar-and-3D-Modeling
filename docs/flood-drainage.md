# Drainage, culvert obstructions and flood-stage road vulnerability: Ghent and the Hague, Norfolk, Virginia

Run `python scripts/build_flood.py` then `python scripts/plot_flood.py`. Plates 51 to 55; GeoTIFFs in `outputs/flood/`; tables in `data/processed/flood/` (`stats.json`, `culvert_candidates.csv`, `road_segments.csv`).
Lidar: 3DEP `USGS_LPC_VA_Norfolk_2013_LAS_2015`, flown 2013-04-01, 6.6 returns per m2 (2.0 reach the ground or road surface), NAVD88. A 1.4 km window at 36.868 N, 76.301 W.

## Method
1. **Bare earth at 1 m** from ground returns (class 2) plus road-surface returns (class 11) within 1.5 m above and 1 m below a ground-only terrain (taller class-11 returns are bridge decks and ramps; 57% of road returns are kept). Gaps (36% of cells, under buildings and trees) are filled by smooth interpolation.
2. **Hydrological conditioning with pysheds:** fill pits and depressions, resolve flats, D8 flow directions, flow accumulation. (WhiteboxTools could not be downloaded here.) Closed depressions come from the difference between the filled and original terrain.
3. **Culvert candidates:** a depression whose spill point is at a road and which the embankment holds back by at least 10 cm; a least-cost path to lower ground beyond the road gives the cut needed, and the terrain is re-conditioned with the cut to see the storage released.
4. **Bathtub flooding** at stages 1.5 to 4.0 m (NAVD88): terrain below the stage connected to the open water seen in the lidar (voids with no returns), or any terrain below the stage (ponding). Roads: Overture drivable segments; bridges, tunnels and overpasses excluded; a road is impassable where 30 cm or more of water stands on it (15 cm difficult), a scenario threshold.

## Results
- **Terrain:** held-out ground returns matched to 4.5 cm rms, road-surface returns 13 cm, 95% of all within 10 cm.
- **Drainage:** 75 closed depressions (60 m2 or more, 15 cm or deeper) hold 67,000 m3 over 11 ha; the deepest (7 m) are underpass and tunnel-approach trenches, which drain by pumps in reality.
- **Culvert candidates:** 10 depressions are held back by at least 10 cm of road embankment (23,000 m3); the strongest barrier is 0.74 m at Hampton Boulevard. Several have only 10 to 20 cm of barrier, which is within a few times the terrain error and is weak evidence. Opening them releases almost no stored water in the model because most spill into the same depressions.
- **Flooding (connected to open water, ha):** 2.8 at 1.5 m; 6.6 at 2.0; 27 at 2.5; 122 at 3.0; 181 at 3.5; 192 at 4.0 (of 196 ha). Between 2.5 and 3 m the whole low plain fills.
- **Roads (51 km drivable):** impassable with 30 cm of water, ponding scenario / connected scenario: 2.5 m: 22 / 11 km; 3 m: 36 / 34 km; 4 m: 46 / 46 km. The largest connected piece of the network holds 51% of its length at 2.5 m (connected scenario), 12% at 3 m and 2% at 3.5 m.

## Tests
- **Planted embankments** (V valley, 10 m road embankment, a culvert in half the trials, noise at the measured terrain level): found 67% of the time at 0.5 m, 75% at 1 m, 88% at 2 m, 13% at 0.3 m, 0% at 0.2 m or less; never flagged when the culvert was open.
- **Vertical error:** a 10 cm error in the stage changes the connected flooded area at 3 m from 108 to 134 ha, and at 2.5 m from 24 to 53 ha; 20 cm is larger still. On a plain this flat the datum matters more than the model.
- **Cell size:** 1 m and 2 m cells give the same flooded area within 1%.
- **Buildings and trees:** about half the connected flooded area at 3 m and above lies under building and tree surfaces, which are filled terrain; the plates show terrain depth everywhere.

## Limits
- **Not a flood forecast.** Static bathtub: no rain, drains, pumps or flow; the stage is a water-surface height in NAVD88, with no tide or storm assigned. The source water is the set of no-return voids in the south-west of the window (ponds and basins), not a mapped tidal creek, and water arriving from outside the window is not modelled.
- Culverts and storm drains are not in the lidar or the map data, so "candidates" are places where terrain suggests an obstruction, not confirmed culverts; real culverts open below the surface.
- Lidar vertical accuracy against benchmarks or tide gauges was not checked (only internal consistency), and the 2013 terrain predates later construction.
- Overture road geometry is OpenStreetMap-derived; bridge and tunnel flags come from it (15 flagged segments plus elevated ones found from the lidar).
