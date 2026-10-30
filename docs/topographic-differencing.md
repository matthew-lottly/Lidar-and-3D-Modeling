# Topographic differencing: bluff erosion at Pacifica, California (2010, 2016, 2023 lidar)

Run: `python scripts/build_topodiff.py` then `python scripts/plot_topodiff.py` (about 10 minutes the first time). Plates 46 to 50; rasters in `outputs/topodiff/`; numbers in `data/processed/topodiff/stats.json`, `segments.csv`, `m3c2_*.csv`.
This complements the Cape Cod M3C2 study (plates 15 and 16, `docs/coastal-erosion.md`); here three epochs, a registration step, DEMs of difference with a slope-dependent level of detection, cut-and-fill volumes, and tests with planted changes.

**Site.** A 900 m window of the San Mateo County coast (37.6175 N, 122.4985 W): Esplanade bluffs to the north, Mori Point headland to the south. Surveys: 3DEP `ARRA-CA_SanFranCoast_2010` (0.5 ground returns per m2; the file carries no collection date), `USGS_LPC_CA_WestCoastElNinoUTM10_2016_LAS_2017` (flown 2016-04-16, 3.8 per m2) and `CA_CaliforniaGaps_3_B23` (2023-11-27, 7.2 per m2).

## Method
1. **Ground model:** the mean of ground-class returns in 2 m cells (class codes differ between surveys, but class 2 is present and consistent in all three).
2. **Stable reference ground:** inland (at least 150 m from the sea), under 8 degrees in the 2023 DEM, chosen from geometry only. It is small: 0.44 ha (1,098 cells), because the surroundings are built up.
3. **Registration (ICP):** point-to-plane ICP of the 2010 and 2016 ground returns onto 2023 over the stable ground (own numpy implementation; Open3D was not available). Held out: fit on the southern half of the stable ground, test on the northern half.
4. **DEM of difference** for 2010 to 2016, 2016 to 2023 and 2010 to 2023, with a per-cell 95% level of detection: 1.96 x sqrt(sigma_z^2 + (tan(slope) x sigma_xy)^2), sigma_z from the spread of the difference on stable ground (the larger of the robust spread and the spread that flags 5% of flat stable cells), sigma_xy from a bootstrap of the ICP translation. Volumes count only cells above the level of detection.
5. **M3C2** (py4dgeo) on the bluff face (20 degrees or more, 2 m cores), with normals from the older epoch; **contour retreat** on east-west transects every 10 m (where the 6, 12 and 20 m contours cross the steep face).

## Results (coastal band = within 60 m of the sea, volumes above the 95% level of detection)
| Interval | Lowered (cut) | Raised (fill) | Net | Area changed |
|---|---|---|---|---|
| 2010 to 2016 | 17,653 m3 | 14,983 m3 | -2,670 m3 | 5.8 ha of 11.9 ha |
| 2016 to 2023 | 33,078 m3 | 7,669 m3 | -25,409 m3 | 6.0 ha of 13.0 ha |
| 2010 to 2023 | 34,864 m3 | 8,493 m3 | -26,371 m3 | 6.2 ha of 13.9 ha |

- Most of the net loss happened between 2016 and 2023 (about 3,300 m3 a year against about 450 before); the earlier interval has cut and fill in near-equal measure, and 2010 is the sparse survey.
- **The 12 m contour on the bluff face moved inland by a median 0.8 m between 2016 and 2023** (about 0.1 m a year; 18% of 67 profiles by 3 m or more; the 10th to 90th percentile range is -0.1 to +4.5 m). Between 2010 and 2016 the median is zero, within the +-1 m that the sparse 2010 data allow. Retreat is concentrated in a few stretches, not uniform.
- The largest lowering in a 50 m block between 2010 and 2023 is 2,931 m3 (plate 48).
- Inland of the coastal band the 2010 to 2023 net difference is -10,751 m3; that is construction, grading and classification, not erosion, and is not counted above.

## Accuracy and error tests (plate 50)
- **Registration:** ICP moved 2010 by 0.12 m horizontally and 0.16 m vertically, and 2016 by 0.29 m and 0.09 m: the surveys were already close. On held-out ground the 2016 fit improved the spread (2.8 to 1.7 cm) but the 2010 fit made no improvement when fitted on half the ground (4.2 to 5.7 cm; 4.0 cm when fitted on all): the sparse 2010 survey is only weakly constrained. Skipping registration changes the 2010 to 2023 coastal net volume from -26.4k to -21.9k m3 (17%).
- **Level of detection:** stable-ground spread 3.5 cm (2016 to 2023) and 6.9 cm (2010 to 2023); the 95% level is 7 and 14 cm on flat ground and 30 to 32 cm on slopes over 45 degrees. It flags 4.9 to 5.6% of the stable cells. The stable ground is small, so these spreads are uncertain.
- **Planted changes:** steps of 10 cm or more added to the measured 2016 to 2023 difference over 6 m and 12 m patches of flat stable ground are found in 97% of cells (100% from 20 cm) and their volume is recovered within 4%; 5 cm steps are not detected (8%). The steep, vegetated face is not tested this way.
- **Two methods on the face:** per-cell agreement between M3C2 and the DEM of difference is r = 0.73 (2010 to 2016) and 0.77 (2016 to 2023); net volumes on the face differ by 1.4k and 8.1k m3 (M3C2 -14.7k against DEM-of-difference -22.8k m3 for 2016 to 2023). M3C2 counts only normal-direction change at 2 m cores with its own level of detection, the DEM of difference counts every cell above its level; both are given, and the face total should be read as 15,000 to 23,000 m3 for that interval.
- **Closure:** 2010 to 2016 plus 2016 to 2023 gives -28,079 m3; 2010 to 2023 directly gives -26,371 m3 (6% apart).
- **Sensitivity (2010 to 2023 net, coastal band):** 1 m cells -18.3k m3 (only 71% of cells have both surveys), 2 m -26.4k, 4 m -25.8k; with no detection threshold -27.4k, 68% level -26.9k.

## Limits
- No independent field measurements (GPS profiles, repeat photographs) were available, so the volumes are not validated against ground truth, only against each other and planted changes.
- The 2010 survey has no collection date in the file and 0.5 ground returns per m2, so the 2010 to 2016 interval is the weakest; the 2010 vertical datum and horizontal accuracy are not documented here.
- The stable ground is only 0.44 ha, so the error model rests on a small sample; the uncertainty of net volumes is not given as a single number because blocks of the size of the changed area do not fit on the stable ground (the block bootstrap found too few blocks). The sensitivity ranges above are the honest uncertainty.
- Class-2 ground returns on a vegetated bluff face are sparse; cells without ground returns in either survey are excluded, so volumes are for the cells with data in both.
- Volumes in the coastal band exclude the beach below the cliff where the sea (no returns) begins; changes below the tide line are not seen.
