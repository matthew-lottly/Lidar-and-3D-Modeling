# Photogrammetry: not done here, and why

Photogrammetry reconstructs 3D from many overlapping photographs taken from different positions. This repository has no such photographs of the study area, and the public archives used for it do not contain any. Making a "photogrammetry" model from the lidar, a satellite image or a single photo would be mislabelled, so none is included. The 3D model in `outputs/08_terrain_model.png` and `terrain_s2_draped.glb` is a **lidar terrain surface with a Sentinel-2 colour drape**.

## To build a real one

If you have photos (drone, aerial or ground) with 60 to 80 % overlap:

* **OpenDroneMap:** put the photos in `datasets/project/images`, then
  `docker run -ti --rm -v $PWD/datasets:/datasets opendronemap/odm --project-path /datasets project`
  The outputs are an orthophoto, a point cloud and a textured mesh.
* **COLMAP:** `colmap automatic_reconstructor --workspace_path ws --image_path photos`

To check a photogrammetric surface against this lidar, georeference it (ground control points or image GPS), export a DSM, and difference it with `data/processed/dsm_1m.tif` (EPSG:32610). The lidar was collected in September 2022, so photos from another date will also show real change.

## Seattle

The same applies to the Seattle buildings. Street-level and aerial photo sources (Mapillary, Wikimedia Commons, Flickr, OpenAerialMap, USGS EarthExplorer) were tried from the build environment and none could be reached, so there are no photographs to reconstruct from. A photogrammetric model of a building needs your own photos of it, or access to one of those sources.

Photogrammetry and lidar would complement each other on a building: lidar gives the roof and overall form (this repository), photos give facade detail and texture. A real comparison would be a photo-based surface differenced against the lidar surface in the same coordinates.
