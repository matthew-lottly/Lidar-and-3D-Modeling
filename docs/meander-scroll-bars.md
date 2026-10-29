# Meander scroll bars from a single lidar flight (Wabash River, Illinois)

**Short answer to "do we need repeat lidar?": no, for the geometry.** A migrating bend leaves curved ridges and swales (scroll bars) in the floodplain, and one bare-earth surface records them. What one flight cannot give is dates, so a migration *rate* needs something else (repeat lidar, historical maps, or imagery). This study maps the ridges from one flight, tests how well that works, and tests whether NAIP imagery can supply the rate (it cannot, here).

Site: Wabash River south of Mount Carmel, IL (3.5 km window, 12.2 km2). Lidar: USGS 3DEP `IL_WabashRiver_2010`, flown 2011-09-17, 19.9 million returns, 6.8 million ground (0.56 per m2), 5 flight lines. Run: `python scripts/build_meander.py` (about 15 minutes the first time). Plates 28 to 31; numbers in `data/processed/meander/stats.json`.

## Method
1. 1 m bare-earth DEM from class 2 returns (cell mean, Gaussian fill of empty cells).
2. River = the cells with no returns of any class (water absorbs the pulse), largest connected body. A smooth bank line along its northern edge lays out transects every 10 m, 2 m apart along each, from 1.4 km south to 2.6 km north of the line (the DEM "straightened" along the bank, plate 29).
3. Band-pass of the straightened surface (3 m smoothing, minus a 30 m surface), crest and swale picked in each row at a prominence of at least 5 cm, picks linked between neighbouring rows into lines at least 150 m long, lines running steeply across the arcs dropped.

## What it found
144 crest lines and 128 swale lines, median crest relief 33 cm. Two ridge sets: older arcs north of the river (median spacing between neighbouring crests 72 m, middle half 46 to 104 m, about 5 crests per transect) and the point bar south of it (58 m, 42 to 76 m, about 8.5 crests per transect). Both run roughly parallel to the present bank.
These are lines in the surface. Field edges, levees and ditches also make lines; steep ones across the arcs were removed automatically but the parallel ones were not screened by hand, and no interpretation of the older northern arcs (an earlier bend position or something else) is claimed here.

## Error and accuracy tests
- **Surface:** 10% of ground returns held out, DEM rebuilt without them: 5.3 cm rms overall (3.0 cm on flat ground, 12 cm on 5 to 15 degree slopes, 27 cm on steeper), 95% within 8 cm. This is interpolation plus point noise, not absolute accuracy (no surveyed checkpoints were available).
- **Between flight lines:** adjacent lines agree on flat overlaps to a bias of 2 to 6 cm and a robust spread of 5 to 8 cm (a few hundred 8 m cells per pair; overlaps are small).
- **Detection limit:** ridges of known height were planted in the real sampling pattern and the whole procedure run. With 3 cm per-return noise only, 5 cm ridges (30 m wavelength) are found 99% of the time. With a 4 cm spatially correlated surface error added (an assumed stand-in for classification and interpolation mistakes), recall is 73% at 5 cm and 96% at 8 cm; 3 cm ridges are not reliably separable. The 5 cm threshold is therefore at the edge for gentle ridges.
- **False picks:** with no ridges the procedure still returns 802 picks (white noise) to 2,006 (correlated error) against 5,107 on the real surface, so up to about 39% of real picks could be noise-like.
- **Repeatability:** two independent random halves of the ground returns: 92% and 90% of one half's picks lie within 4 m of a pick in the other. 12 bootstrap resamples: a pick persists in 92% of runs on average, 86% persist in at least 80%. Resampling the same flight shares its systematic errors, so this is a floor on error.

## Can imagery supply the migration rate? Not reliably
NAIP 2011 to 2023 (8 full-coverage years) gave the edge of the active channel (open water and bare bars, NDVI below 0.20) along the same transects. Registration between years was checked on stable ground away from the river (median shift at most 0.05 pixel). But the median edge moves 32 m (south) and 18 m (north) between years, with river stage, bar exposure and crop cover changing with the date of each flight, and the fitted rate is not robust: the south edge is -1.6 m/yr with all years (95% interval -2.3 to -0.6) and +1.1 m/yr without 2019 (0.5 to 1.8). So no migration rate, and therefore no ridge ages, is claimed.
The lidar water edge and the 2011 imagery edge (23 days earlier) differ by a fixed 20 m (middle half 12 to 28 m) because they are different things (a water line against the edge of bare bank), not a positioning error that was measured separately.

## Next steps that would give a rate
Repeat lidar over the same reach (3DEP has later flights in other areas; this reach was not found with two usable epochs), stage-matched imagery using a gauge record, or historical maps and photographs. With a rate, ridge spacing gives the time per ridge.
