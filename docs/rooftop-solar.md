# Rooftop solar potential from 3DEP lidar: West University Place and downtown Houston

Two 400 m blocks of Houston, one pipeline (`python scripts/build_rooftops.py all`, then `python scripts/plot_rooftops.py`; the notebook `notebooks/rooftop_solar.ipynb` runs the same code for any longitude/latitude box and writes a `.glb` and `.obj`).
Lidar: 3DEP `TX_Coastal_B1_2018` (flown March 2018; 18.5 returns per m2 at West U, 50 at downtown). Sunlight: NREL NSRDB typical meteorological year (hourly direct and diffuse, 4 km grid), read directly from the public AWS copy.

## Pipeline
1. **Roof returns by shape, not by class code.** The project's class 6 ("building") also covers tree returns over houses, so a return counts as roof when it is at least 2.5 m above the terrain, is a single return, and the 10 returns around it (within 2 m) fit a plane to 10 cm rms with tilt under 70 degrees (`planar_points`). This is the Euclidean-clustering alternative to the class filter.
2. **Footprints** from those returns on a 1 m grid (at least 2 returns per cell, opened and closed, 8-connected components of 30 m2 or more). No mapped footprints are used.
3. **Roof planes** per building by sequential RANSAC (numpy implementation; Open3D could not be installed here): planes within 12 cm of at least 40 returns, spatially connected, at least 10 m2, near-coplanar ones merged. Each 1 m roof cell takes the plane of the nearest returns within 2 m.
4. **Shading**: horizon angle in 72 directions for every roof cell, marched over the surface model (buildings and trees, up to 250 m), with a 15 cm tolerance so roof noise does not shade the roof itself.
5. **Irradiance**: for every hour of the typical year, beam = DNI x cos(angle between sun and plane normal) when the sun clears the horizon; diffuse = isotropic sky, weighted by the part of the sky each plane sees. No ground reflection, circumsolar light, soiling or snow.
6. **LOD2-style mesh**: roof planes lifted from their cell outlines, walls from the footprint to the roof, coloured by energy; terrain at 2 m.

## Results (`data/processed/rooftops/*.json`, `*_buildings.csv`, `*_roof_planes.csv`)
| | West University Place | Downtown Houston |
|---|---|---|
| Buildings found | 104 (median footprint 279 m2, median height 9.4 m) | 30 (heights to 203 m) |
| Roof area; share flat (under 10 degrees) | 3.8 ha; 22% | 6.8 ha; 98% |
| Energy on roofs per year (beam + diffuse) | 45 GWh (1,198 kWh/m2 mean) | 51 GWh (750 kWh/m2 mean) |
| Direct sun blocked by shade | 23% | 55% |
| Roof area at 1,000 kWh/m2/yr or more | 73% | 27% |
| Electricity if that area were covered at 20% efficiency, 80% performance ratio (assumed) | 6.2 GWh | 3.6 GWh |
| Pitched roofs, mean kWh/m2/yr by facing | S 1,308, E 1,178, W 1,151, N 948 (flat 1,351) | too few pitched roofs to compare |

Typical-year horizontal sunlight at the NSRDB site is 1,744 kWh/m2 (direct 1,766 on a tracking-normal basis, diffuse 668). The beam and horizontal-irradiance totals recomputed from the hourly data agree with the file's GHI to 0.1%.

## Tests (plate 45)
- **Sun geometry** against pvlib on six plane orientations: identical to 4 digits. **Shadow engine** against a 20 m block: horizon angles 0.2 to 2.3 degrees below the exact angle (the 1 m raster softens the edge).
- **Planted roofs** (60 gables and flats at each block's measured return density and plane noise, through the same RANSAC): slope error median 0.1 degrees, 95th percentile 0.25 (West U) and 0.4 (downtown); irradiance error median 0.5 to 0.7 and 95th percentile 2.6 to 2.7 kWh/m2/yr. These are clean synthetic roofs, so real error is larger.
- **Same roof, two random halves of the returns:** the largest plane of one half and the closest-oriented plane of the other differ in tilt by median 0.2 degrees; 95th percentile 2.2 (West U) and 15.7 degrees (downtown, where roof clutter dominates).
- **Footprints against Overture:** West U: 82% of lidar footprint area is inside Overture buildings, 58% of Overture's building area is inside lidar footprints (median per-building IoU 0.56). Downtown: 84% and 51%; touching towers and podiums merge, so per-building IoU is not meaningful there.
- **What moves the totals** (West U mean kWh/m2/yr): removing trees from the shading surface 1,198 to 1,211; a 2 m instead of 1 m shading surface 1,198 to 1,045 (a coarser surface exaggerates shade); half the returns 1,198 to 1,289 (a different set of buildings is found, 135 against 104). So the absolute level is uncertain by about 10%, more than the geometry errors suggest.

## Limits
- **Roof planes are over-segmented** (median 7 per building at West U where 3 to 5 would be typical for a house): dormers, chimneys and returns through tree gaps create small planes. Energy totals are insensitive to this, per-plane statistics are not.
- Roofs under tree crowns are partly or wholly missing (lidar footprints cover 58% of Overture's building area), so building counts and total roof area are low; the analysed roofs are the visible ones.
- Downtown towers: only returns that look planar are used, and glass curtain walls return poorly, so only 30 structures are found; the top-of-tower mechanical clutter is part of the roofs.
- No metered PV output was available, so energy is not validated against any real system; 'electricity' uses assumed efficiency and losses. NSRDB is a typical year at 4 km, not the weather of any one year.
- Shading by trees uses the leaf-off March 2018 lidar for deciduous species, so summer shade from them is underestimated.

## Independent-data checks (`scripts/validate_rooftops.py`; `data/processed/validation/rooftops.json`)
- **Footprints against Overture** (compiled from several sources): West University Place 80% of our footprint area lies inside Overture footprints and 73% of Overture's inside ours (median IoU 0.56; 104 lidar buildings against 201 mapped, only about half of the mapped buildings were found as lidar buildings; why was not examined). Downtown 79% and 80% by area, but a per-building median IoU of 0.00 (the reason for the per-building mismatch was not examined).
- **Heights against Overture:** they disagree (West U r = 0.05, our highest roof point +4.0 m above Overture's height over 85 matched buildings; downtown bias is larger). Overture heights are mapper-entered or modelled, not surveyed, and the cause here is not established, so neither is claimed correct.
- **Weather bound.** The typical-year irradiance is 80% of a clear sky's, as it must be below 100%; 504 hours exceed clear sky by more than 5%, the usual mismatch between a typical-year file and a clear-sky model.
- **Not done:** measured roof irradiance or PV production; Microsoft building footprints.
