# Fusing lidar, imagery and radar for lines and poles, checked against OpenStreetMap (White River, WA)

Site: White River Substation, Pierce County, WA (800 m window, UTM 10N). Lidar: USGS 3DEP WA_PierceCounty_1_2020 (collected 2020-04-13). Imagery: NAIP 2019-10-11 (0.6 m).
Radar: Sentinel-1 RTC, 105 ascending scenes 2017-01-02 to 2021-04-23 (20 m). Run: `python scripts/build_fusion.py`. Plates 22 to 24 in `outputs/`; tables in `data/processed/fusion/`.

## What was built
1. **Conductors** (`conductors.py`): the wire fragments from the lidar (see `utility-extraction.md`) are joined when collinear within 4 degrees and 1.5 m, at most 60 m apart and within 6 m in height.
   155 fragments became 88 conductors.
2. **Supports from the wires**: the peaks of each conductor's height profile (where the wire is held up) are merged across conductors (6 m) into 157 candidate supports, 110 outside the yard.
3. **Gap filling**: each span between supports is fitted with a parabola and filled only if over a third of the span was observed, the curve is concave-up and rms is within 0.8 m. 69 of 97 spans were fitted;
   468 m of conductor was filled, against 5,720 m observed.
4. **Fusion** (`fusion.py`): each candidate support gets lidar features (distance to a compact tall object, extent, spread, number of conductors), imagery features (NAIP vegetation index, thin-line filter at the base) and radar features.
   The score is the mean of signed z-scores, with the signs fixed from physics before looking at any map. OpenStreetMap was never used to build or tune the model.
5. **OSM check**: the only reachable OSM source was Overture Maps' infrastructure layer (all 228 power features at the site are OSM-sourced, planet snapshot 2026-09-06; direct OSM hosts are blocked here).
   Mapped lines are compared by length covered within 6 m; supports by matching within 3, 5 and 8 m. The yard is excluded (15 m buffer).

## Results
Ranking the 110 candidate supports against the 55 mapped supports outside the yard (26 candidates within 5 m of one). AUC with 95% bootstrap intervals over candidates:

| Evidence | AUC | Detected / matched at 5 m | Precision | Recall |
|---|---|---|---|---|
| wire sag peaks only | 0.50 | 110 / 22 | 0.20 | 0.40 |
| lidar score | 0.81 [0.73, 0.89] | 56 / 20 | 0.36 | 0.36 |
| lidar + imagery (final model) | 0.81 [0.71, 0.90] | 48 / 18 | 0.38 | 0.33 |
| lidar + imagery + radar | 0.77 [0.67, 0.86] | 55 / 17 | 0.31 | 0.31 |
| unsupervised two-component mixture, same features | 0.78 | | | |

- Lidar carries the signal. Adding imagery changes AUC by -0.07 to +0.06 and radar by -0.13 to +0.06: neither is distinguishable from no effect at this sample size.
- Radar at 20 m cannot see towers (AUC 0.41 for the 10 mapped towers outside the yard) and corridors are only weakly different (VV AUC 0.57). Imagery thin-line contrast along mapped lines is +0.023 and does not separate lines the lidar confirms from those it misses.
- **Choice of final model is post hoc.** "Lidar + imagery" was picked after seeing these numbers; its interval overlaps the lidar-only one entirely. Treat it as the same model as lidar alone.
- The unsupervised mixture ranked at chance (0.44) without the object-distance cue and reaches 0.78 with it; the cue, not the method, does the work.
- Precision and recall are against a map that is itself incomplete and of unknown accuracy, so they are lower bounds on quality only in one direction; an unmatched support is not necessarily wrong. Median offset of matched pairs is 2.4 m.

### Lines against the map (6,542 m of mapped line outside the yard)
| Status | Lines | Length |
|---|---|---|
| confirmed (network within 6 m of at least 60% of the line) | 6 | 2.5 km (39%) |
| partial (20 to 60%) | 4 | 1.7 km (25%) |
| elevated returns, no wire found | 5 | 1.6 km (24%) |
| no lidar evidence | 4 | 0.7 km (11%) |

- Length coverage is 41.7% from observed wire and 42.3% with gap filling: interpolation adds 0.6 points. The missing length is mostly whole spans with no observed wire, which no curve can recover.
  The filled parts are accurate where they can be checked: 97% of interpolated points lie near mapped lines.
- Median offset of network from mapped line where they agree: 1.2 m. None of the sampled observed conductor points (every third) is more than 10 m from a mapped line, so no wire was found that the map lacks.
- Coverage by voltage tag: 230 kV 22% (2.0 km), 115 kV 47% (3.4 km), 55 kV 63% (1.1 km). All four "no lidar evidence" lines are 230 kV. That points at the detector (high, sparse conductors) rather than a map error; **not inspected visually, so no claim is made that the map is wrong**.
- Only 2 of the 19 line parts carry a source tag (imagery tracing).

## Not done / next
- Cable-count check against the OSM `cables` tag (the lateral-conductor helper exists, not run).
- Visual spot check of model-only supports on NAIP.
- Direct OSM (needs overpass-api.de allowed) to confirm the Overture snapshot matches current OSM.
- Other pointers that would likely help more than more sensors: higher-density lidar (quality level 1 flights record wires as returns far more reliably), classified wire/tower classes (14 and 15) where a project has them, using return number and intensity (stored in the cache, only partly used), oblique or street-level imagery for poles, and Sentinel-2 or higher-resolution SAR for towers.
