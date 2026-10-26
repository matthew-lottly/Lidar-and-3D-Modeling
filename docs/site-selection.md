# Site selection

**Request:** a small area near Mount Rainier, preferably low tree cover, using USGS 3DEP lidar.

**What the data allows.** 3DEP point clouds are published as Entwine Point Tiles on AWS (`usgs-lidar-public`). Within the projects checked, none gave dense coverage of the interior of Mount Rainier National Park. The area south of the mountain is covered by `WA_CentralWildfire_2_D22`, collected 2022-09-04 to 2022-09-08.

**How the spot was chosen.** Canopy was estimated from the lidar itself and 1.4 km squares were scanned across the project for the lowest tree cover at a 1.2 km window. The chosen square is centred at 46.7065 N, 121.6615 W (UTM 10N, EPSG:32610) in the Tatoosh Range. It lies about 17.9 km south-southeast of the summit and includes Tatoosh Peak (lidar terrain maximum 1925 m) and a small lake in a cirque basin.

**Measured on the full-density data:** 60.3 million points over 1.96 km² (30.8 per m²); vegetation above 3 m covers 11.4 % of the area and above 8 m 3.9 %; relief 500 m (1425 to 1925 m).

**Not examined:** other candidate squares were scored only by the quick canopy estimate, not mapped. Other 3DEP projects not checked may cover areas closer to the summit.
