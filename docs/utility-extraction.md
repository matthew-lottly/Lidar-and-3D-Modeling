# Poles, wires and a substation from lidar: White River Substation, Washington

**Question.** Starting from the best site in the [utility data search](utilities-data-search.md), what can be found in a 3DEP point cloud without any wire or tower classes, and how well does it agree with the mapped power features?

**Site.** White River Substation, Pierce County, Washington (47.2390 N, 122.2104 W): an 800 m window of `WA_PierceCounty_1_2020`, collected 2020-04-13, 16.3 million points (25 per m²). Classes present: 1 (unclassified), 2 (ground) and 18 (high noise). The surroundings are houses, forest, a substation yard and several transmission and distribution lines.
Mapped features (Overture Maps, release 2026-09-23.1) in the window: 50 power lines, 24 towers, 48 poles, 87 switches, 3 transformers, 7 portals and the substation outline.

## Method (`src/lidar3d/utility.py`, `scripts/build_utility.py`)

1. **Height above ground** from the lowest class-2 return in each 1 m cell, gaps filled from the nearest cell.
2. **Local shape.** For every point more than 4 m above ground, the eigenvalues of the covariance of its 12 nearest neighbours give *linearity* (a wire is a line), and the first eigenvector gives its direction.
3. **Wires** are points with linearity of at least 0.9, a near-horizontal direction (|z component| below 0.6) and **open space around them** (at most 80 other points within 3 m). The last test matters: without it, roof edges and ridges also look like straight horizontal lines: in the first run only 27% of the detected wire points were within 5 m of a mapped line, and the rest sat mostly on house roofs (seen on the map, not separately counted). Surviving points are joined into chains of at least 15 m.
4. **Poles and towers** are vertical lines (|z component| above 0.85, linearity of at least 0.7) of at least 4 m rise within 1 m of their axis and at most twice as many elevated points in a 1.5 to 3.5 m ring as in the column (a trunk inside a crown fails this).
5. **Yard structures** are connected groups of points more than 1.5 m above ground, inside the mapped substation outline, that are not wires.
6. **Comparison.** Detections against Overture within the window: nearest-neighbour matching at 3 m for poles, and for wires the share of detected points within 5 m of a mapped line and the share of mapped line length with a detected wire within 6 m.

## Results

* **Wires.** 155 chains, 5.7 km, median 13 m above ground. 93% of the wire points lie within 5 m of a mapped line. But only 43% of the 9.4 km of mapped line has a detected wire within 6 m. Many mapped lines are duplicates (several mapped lines per corridor), underground or under trees, and thin or low conductors are missed; I did not separate those causes.
* **Poles and towers.** 19 columns found, 9 of them within 3 m of one of the 72 mapped poles and towers: recall 12%, precision 47%, median offset 1.7 m. This is weak. At 25 points per m² a 0.2 to 0.4 m pole is only a few returns thick, so blind detection fails in most places. The earlier [search](utilities-data-search.md) found that nearly every mapped pole and tower has returns above it, so the mapped positions are the better guide and lidar is best used to measure heights and check them.
* **Substation yard.** 432 elevated structures in the 34 thousand m² yard, median top 7 m above ground. Of 97 mapped switches, transformers and portals, 72% have a lidar structure within 3 m, against 28% for random points in the same yard; within 1.5 m, 46% against 9%. So the map's equipment positions are real, but the structures are not identified: a group can be one piece of equipment or several that touch.

## Files

* `outputs/20_utility_overview.png`, `outputs/21_utility_profile.png`.
* `data/processed/utility/`: `wire_chains.csv` (length, mean height, position), `detected_poles.csv` (position, height, whether it matched a mapped feature), `yard_structures.csv`, `stats.json` with every number here.
* Run `python scripts/build_utility.py` (fetches about 16 million points on the first run).

## Limits

* One site, one survey, one map source. The thresholds were chosen by looking at this site (the isolation test after seeing roofs flagged as wires), so these numbers are not an out-of-sample accuracy.
* **Sag, clearance and vegetation encroachment were not computed.** The wire points support them (each chain is a catenary), but conductors are not separated and no span model is fitted.
* The window is small and mostly open ground around the yard; results under dense forest will be worse. Overture positions have their own errors, so a "miss" can be a map error.
* No wire, tower or pole class exists in this survey; a project that does classify them would make the task much easier.
