# Utility data search: 3DEP lidar with power facilities, lines and poles

**Question.** Where can USGS 3DEP point clouds be combined with mapped utility features (facilities, powerlines, poles)?

**Method** (`scripts/search_utilities.py`). Substations from Overture Maps (release 2026-09-23.1, `theme=base/type=infrastructure`, `subtype=power`) inside nine dense, recent 3DEP projects. For a random sample of 60 substations per project, the octree depth of the lidar at the site was read from the hierarchy (a cheap proxy for density); the deepest ones were then fetched as 300 m windows and measured. The six sites with the most Overture towers and poles were fetched at 700 m and each Overture tower and pole was tested for a column of at least 20 returns more than 3 m above the ground within 2.5 m. 36 sites were probed. The sample is random (seed 0), so these are examples of what exists, not a ranking of all sites.

## What Overture provides

Power classes seen near the sites: `substation`, `plant`, `power_line`, `power_tower`, `power_pole`, `minor_line`, `transformer`, `switch`, `portal`, `cable`, `generator`, `catenary_mast`. They come from OpenStreetMap and other sources, so coverage varies by place. No tower or pole in the windows had a height tag.

## Lidar classes: wires and towers are not classified

Across all 36 windows, the number of points in ASPRS class 14 (wire conductor) or 15 (transmission tower) is **0**. Towers and poles are in other classes: at Dighton, Massachusetts the points on all 20 Overture towers are labelled high vegetation (5) or medium vegetation (4). So wires, poles and towers must be found in the unclassified or vegetation points; they cannot be read from a class.

## Best candidate sites

Counts are Overture features inside a 300 m window around a substation.

| Project | Substation | lon, lat | pts/m² | towers | poles | lines | substations | switches, transformers, portals |
|---|---|---|---|---|---|---|---|---|
| WA_PierceCounty_1_2020 | White River Substation | -122.2104, 47.2390 | 28.3 | 18 | 20 | 51 | 1 | 97 |
| WA_KingCo_1_2021 | Sammamish Substation | -122.1498, 47.6853 | 11.8 | 1 | 27 | 55 | 1 | 24 |
| NJ_SouthernNJ_2018 | Lewis Substation | -74.5293, 39.4170 | 16.9 | 25 | 20 | 34 | 1 | 50 |
| NJ_SouthernNJ_2018 | Mill Substation | -74.6166, 39.3972 | 16.2 | 2 | 14 | 22 | 1 | 26 |
| CT_Statewide_B1_2016 | Mill River Substation | -72.9086, 41.3082 | 4.8 | 14 | 13 | 11 | 2 | 12 |
| CA_SanFrancisco_1_B23 | (unnamed substation) | -122.3684, 37.8272 | 88.7 | 0 | 33 | 1 | 2 | 0 |
| WA_KingCo_1_2021 | (unnamed substation) | -122.1577, 47.6740 | 16.1 | 0 | 20 | 9 | 2 | 0 |
| CT_Statewide_B1_2016 | East Devon Substation | -73.0985, 41.2278 | 9.6 | 9 | 4 | 15 | 1 | 18 |

Full table: `data/processed/utilities/candidates.csv` (class histograms included).

## Are poles and towers visible in the points?

Yes, where tested. Share of Overture towers and poles with a column of returns more than 3 m above ground (`tower_check.csv`):

| Project | Site | Feature | pts/m² | Overture features | with a column | median top |
|---|---|---|---|---|---|---|
| NJ_SouthernNJ_2018 | Lewis Substation | tower | 18.6 | 45 | 42 | 18 m |
| NJ_SouthernNJ_2018 | Lewis Substation | pole | 18.6 | 35 | 35 | 16 m |
| WA_PierceCounty_1_2020 | White River Substation | tower | 26.5 | 24 | 24 | 11 m |
| WA_PierceCounty_1_2020 | White River Substation | pole | 26.5 | 39 | 39 | 17 m |
| CA_SanFrancisco_1_B23 | 0 | tower | 70.8 | 0 | 0 |  |
| CA_SanFrancisco_1_B23 | 0 | pole | 70.8 | 93 | 91 | 14 m |
| WA_KingCo_1_2021 | Sammamish Substation | tower | 14.8 | 2 | 2 | 27 m |
| WA_KingCo_1_2021 | Sammamish Substation | pole | 14.8 | 61 | 50 | 16 m |
| CT_Statewide_B1_2016 | Mill River Substation | tower | 4 | 19 | 19 | 25 m |
| CT_Statewide_B1_2016 | Mill River Substation | pole | 4 | 93 | 33 | 12 m |
| CA_SanFrancisco_1_B23 | 0 | tower | 75.3 | 0 | 0 |  |
| CA_SanFrancisco_1_B23 | 0 | pole | 75.3 | 102 | 100 | 13 m |

* The match is a detection of *something tall at that position*, not proof it is a pole: trees and buildings also give columns. At sites in forest or dense urban areas this overstates detection.
* **San Francisco** has no mapped towers but 93 to 102 poles per window at 71 to 75 points per m²: the densest pole data found.
* A separate manual look (not in the script) at **Dighton, MA** (`MA_CentralEastern_2_2021`, 36 pts/m², 20 towers, 8 lines): all 20 towers have returns up to 17 to 27 m above them, in a cleared right-of-way through forest, with a second line of elevated returns crossing it that looks like a distribution line. That makes it a good site for vegetation encroachment as well as tower and wire extraction.
* A manual look at **Franklin Square, Providence, RI** (`RI_Statewide_1_D22`, 9 pts/m²): 54 of 54 mapped "towers" have columns, but median heights are only 5 to 10 m: they are substation structures, not transmission towers.

## Follow-up: White River extraction

The White River site was analysed in [utility-extraction.md](utility-extraction.md): wires are found well, blind pole detection is weak, and the yard equipment positions check out against a random baseline.

## Recommended next steps

1. **Facilities + lines + poles together:** White River Substation (Pierce County, WA) and Lewis Substation (southern New Jersey) have the largest mix of features at 19 to 28 points per m².
2. **Transmission towers and corridor clearance:** Dighton, MA (forest corridor, 36 pts/m²).
3. **Poles in a city:** San Francisco, Treasure Island (71 to 75 points per m²).

## Limits and what was not done

* Coverage of Overture power features is uneven; a missing pole is not evidence there is none.
* Only nine projects were searched; other 3DEP projects (and the HIFLD and OpenStreetMap sources directly) were not.
* Substations were the entry point, so sites without a mapped substation were not found. The Comanche power plant (Pueblo, Colorado) turned up in `CO_Eastern_B5_2018`'s bounding box, but that project has no points there.
* Wires and poles were not extracted from the lidar, and the match to mapped features was not checked for position error. Next step: detect them from the points and compare.
