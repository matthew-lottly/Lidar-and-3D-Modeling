"""Hydro-flattened DEM, depression filling and D8 drainage for the Tatoosh cirque. Usage: python scripts/build_hydro.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import rasterio
import rasterio.features
from scipy import ndimage

from lidar3d import hydro
from lidar3d.aoi import AOI
from lidar3d.cloud import Rasters, load_cloud

THRESH_M2 = 1000.0          # contributing area that makes a channel (0.1 ha); a choice, shown against others on the plate
EXC_M = 15.0                # match distance to a mapped stream
INCISION_MIN_M = 0.5        # a flow path counts as a defined channel only if it sits at least this far below its 15 m surroundings


def overture_streams(aoi):
    import os
    import pyarrow.dataset as ds
    from pyarrow import fs
    from shapely import wkb
    from shapely.geometry import box
    from lidar3d.names import RELEASE
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=os.environ.get("HTTPS_PROXY"))
    W, S, E, N = aoi.bbox_ll
    flt = (ds.field("bbox", "xmin") < E) & (ds.field("bbox", "xmax") > W) & (ds.field("bbox", "ymin") < N) & (ds.field("bbox", "ymax") > S)
    t = ds.dataset(f"overturemaps-us-west-2/release/{RELEASE}/theme=base/type=water", filesystem=s3, format="parquet").to_table(filter=flt, columns=["names", "subtype", "geometry"]).to_pandas()
    import geopandas as gpd
    g = gpd.GeoDataFrame(t.drop(columns="geometry"), geometry=[wkb.loads(b) for b in t.geometry], crs="EPSG:4326").to_crs(aoi.crs)
    g = g[g.intersects(box(*aoi.bounds)) & g.subtype.isin(["stream", "river"]) & (g.geom_type.isin(["LineString", "MultiLineString"]))]
    return g


def main():
    out = ROOT / "data" / "processed" / "hydro"
    out.mkdir(parents=True, exist_ok=True)
    aoi = AOI.default()
    c = load_cloud(aoi, log=lambda s: None)
    R = Rasters(aoi, c, res=1.0)
    zmin = hydro.lowest_grid(R.row, R.col, c.z, R.shape)
    dtm = R.dtm()
    chm = np.clip(zmin - dtm, 0, None)
    ours = hydro.pmf_ground(zmin)
    ground = hydro.agreement(ours, zmin, dtm, chm > 2)
    wc = R.count(c.cls == 9)
    wmask, wlab = hydro.water_mask(wc)
    wi = np.flatnonzero(c.cls == 9)
    flat, levels = hydro.hydro_flatten(dtm, wmask, wlab, c.z[wi], wlab[R.row[wi], R.col[wi]])
    filled = hydro.fill_depressions(flat)
    dep = filled - flat
    dep[dep < 1e-6] = 0
    routed = hydro.resolve_flats(filled)
    d = hydro.d8_direction(routed)
    acc = hydro.flow_accumulation(routed, d)
    chan = hydro.channels(acc, THRESH_M2)
    streams = overture_streams(aoi)
    smask = np.zeros(R.shape, bool)
    if len(streams):
        smask = rasterio.features.rasterize([(g, 1) for g in streams.geometry], out_shape=R.shape, transform=R.transform, fill=0, dtype="uint8", all_touched=True).astype(bool)
    matched = hydro.near(chan, smask, EXC_M)
    unmapped = chan & ~matched
    inc = hydro.incision(dtm)
    lake = wmask
    catch = hydro.catchment_of(d, lake)
    defined = chan & (inc >= INCISION_MIN_M)
    hand_main = hydro.hand(routed, d, hydro.channels(acc, 5000.0))
    hand_all = hydro.hand(routed, d, defined)
    flood = {f"{t:g}": dict(area_m2_main=float((hand_main <= t).sum()), share_main=float((hand_main <= t).mean()), area_m2_defined=float((hand_all <= t).sum()), share_defined=float((hand_all <= t).mean())) for t in (0.5, 1.0, 2.0, 5.0)}
    mapped_recall = float(hydro.near(smask, defined | chan, EXC_M).sum() / max(smask.sum(), 1)) if smask.any() else None
    area_km2 = R.shape[0] * R.shape[1] / 1e6
    stats = dict(
        ground_filter=ground,
        lake=dict(cells=int(lake.sum()), level_m=list(levels.values())[0], dtm_p5=float(np.percentile(dtm[lake], 5)), dtm_p95=float(np.percentile(dtm[lake], 95)), dtm_std_before_m=float(dtm[lake].std()),
                  catchment_km2=float(catch.sum() / 1e6), water_points=int(len(wi))),
        depressions=dict(cells=int((dep > 0).sum()), volume_m3=float(dep.sum()), max_depth_m=float(dep.max()), count=int(ndimage.label(dep > 0)[1]), in_lake=float(dep[lake].max())),
        d8=dict(interior_sinks=int(((d < 0)[1:-1, 1:-1]).sum()), max_accumulation_km2=float(acc.max() / 1e6)),
        channels=dict(threshold_m2=THRESH_M2, length_m=hydro.channel_length(chan, 1.0, d), density_km_per_km2=hydro.channel_length(chan, 1.0, d) / 1000 / area_km2,
                      by_threshold={str(int(t)): hydro.channel_length(acc >= t, 1.0, d) for t in (250, 500, 1000, 2000, 5000)},
                      mapped_streams_in_area=int(len(streams)), mapped_length_m=float(sum(g.length for g in streams.geometry)) if len(streams) else 0.0,
                      matched_length_m=hydro.channel_length(matched, 1.0, d), unmapped_length_m=hydro.channel_length(unmapped, 1.0, d), unmapped_share=float(unmapped.sum() / max(chan.sum(), 1)),
                      incision_median_unmapped_m=float(np.median(inc[unmapped])) if unmapped.any() else None, incision_p90_unmapped_m=float(np.percentile(inc[unmapped], 90)) if unmapped.any() else None,
                      share_unmapped_under_0p5m=float((inc[unmapped] < 0.5).mean()) if unmapped.any() else None),
        defined=dict(length_m=hydro.channel_length(defined, 1.0, d), density_km_per_km2=hydro.channel_length(defined, 1.0, d) / 1000 / area_km2, unmapped_length_m=hydro.channel_length(defined & ~matched, 1.0, d), mapped_stream_recall=mapped_recall,
                     incision_min_m=INCISION_MIN_M), hand=flood, area_km2=area_km2,
    )
    np.savez_compressed(out / "arrays.npz", dtm=dtm.astype("float32"), flat=flat.astype("float32"), filled=filled.astype("float32"), acc=acc.astype("float32"), d=d, lake=lake, chan=chan, matched=matched, unmapped=unmapped,
                        ours=ours, defined=defined, hand_main=hand_main.astype('float32'), hand_all=hand_all.astype('float32'), zmin=zmin.astype("float32"), inc=inc.astype("float32"), catch=catch, smask=smask)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    R.write_tif(out / "dtm_hydroflattened_1m.tif", flat)
    R.write_tif(out / "dtm_filled_1m.tif", filled)
    R.write_tif(out / "flow_accumulation_m2.tif", acc)
    with rasterio.open(out / "d8_flow_direction.tif", "w", driver="GTiff", height=d.shape[0], width=d.shape[1], count=1, dtype="int8", crs=aoi.crs, transform=R.transform, compress="deflate", nodata=-1) as dst:
        dst.write(d.astype("int8"), 1)
        dst.update_tags(codes="0=E 1=SE 2=S 3=SW 4=W 5=NW 6=N 7=NE; -1=no lower neighbour (grid edge)")
    dates = json.loads((ROOT / "data" / "processed" / "stats.json").read_text())["collected"]
    from lidar3d import figures_hydro
    A = np.load(out / "arrays.npz")
    figures_hydro.fig_flatten(stats, A, aoi, dates)
    figures_hydro.fig_drainage(stats, A, aoi, dates)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
