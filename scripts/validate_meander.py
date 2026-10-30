"""Accuracy assessment of the Wabash scroll-bar study against real imagery (run after scripts/build_meander.py):
- the lidar channel (no-return void) against a water mask from NAIP 2011 (near-infrared) taken 23 days before the flight: overlap, and the horizontal offset of the two banks;
- do the traced ridges exist in the real landscape? Along the same transects, the vegetation index of NAIP (an independent sensor, summer 2011) is band-passed like the lidar relief; if the ridges are real, NDVI
  anomalies should line up with lidar crests (floodplain ridges and swales hold different moisture and vegetation) more often than with shifted copies of the transects.
Usage: python scripts/validate_meander.py"""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy import ndimage
from skimage.filters import gaussian, threshold_otsu

from lidar3d import meander as M
from lidar3d import naip
from lidar3d import validation as V
from lidar3d.ept import CACHE

DATA = ROOT / "data" / "processed" / "validation"
HALF = M.SITE["half"]


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    F = pickle.loads((CACHE / "meander" / "final.pkl").read_bytes())
    P = M.load_points(log=lambda s: None)
    ch, _ = M.channel_mask(P, HALF, 2.0)
    aoi = M.make_aoi()
    img, tf, dts = naip.fetch(aoi, 2011, res=1.0, slug=M.SITE["slug"])
    chn = np.kron(ch, np.ones((2, 2), bool))[::-1]
    near = ndimage.binary_dilation(chn, iterations=300)
    nir = gaussian(img[3].astype("float32"), 1.5)
    thr = threshold_otsu(nir[near])
    w = (nir < thr) & near
    w = ndimage.binary_opening(w, iterations=3)
    lab, n = ndimage.label(w)
    sizes = ndimage.sum(w, lab, range(1, n + 1))
    w = ndimage.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
    out = dict(analysis="Wabash scroll bars", naip_date=dts, lidar_flown=P.date)
    out["channel_vs_naip_water"] = dict(V.binary_agreement(chn, w, near), lidar_channel_km2=float(chn.sum() / 1e6), naip_water_km2=float(w.sum() / 1e6),
                                         note="different dates (23 days) and different physics (no lidar return versus low near-infrared); stage differences move the banks")
    # bank offsets along transects: north edge of each mask in each column
    cols = np.arange(chn.shape[1])[200:-200:10]
    off = []
    for c in cols:
        a, b = np.flatnonzero(chn[:, c]), np.flatnonzero(w[:, c])
        if len(a) and len(b):
            off.append(float(a.min() - b.min()))                  # image rows run north to south, so a positive value means the lidar north bank is south of the NAIP one
    out["channel_vs_naip_water"]["north_bank_offset_m"] = V.error_summary(off)
    # ridge existence: NDVI along the transects
    nd = M.ndvi_image(img)
    X, Y, d, V_ = F["X"], F["Y"], F["d"], F["V"]
    ns = M.sample_image(nd, HALF, X, Y).astype("float32")
    step = abs(d[1] - d[0])
    from scipy.ndimage import gaussian_filter1d
    def band(a):
        a = np.where(np.isfinite(a), a, np.nanmedian(a))
        return gaussian_filter1d(a, 3.0 / step, axis=1) - gaussian_filter1d(a, 30.0 / step, axis=1)
    nb = band(ns)
    vb = np.where(np.isfinite(V_), V_, 0.0)
    sel = (d >= 60) & (d <= 900) | (d <= -300)
    rng = np.random.default_rng(0)
    def corr(a, b):
        a, b = a[:, sel].ravel(), b[:, sel].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    r0 = corr(nb, vb)
    shifts = rng.integers(40, vb.shape[0] - 40, 200)
    rn = [corr(nb, np.roll(vb, int(s), axis=0)) for s in shifts]
    out["ridge_existence"] = dict(correlation_ndvi_vs_lidar_relief=r0, shifted_null_mean=float(np.mean(rn)), shifted_null_sd=float(np.std(rn)), z_score=float((r0 - np.mean(rn)) / np.std(rn)), n_shifts=200,
                                  note="correlation between the band-passed NAIP NDVI and the band-passed lidar relief on the same transects, against the same correlation after shifting the lidar strip along the bank")
    # share of tracked crest picks that coincide with a local NDVI maximum or minimum within 4 m
    C = F["C"]
    hit, tot = 0, 0
    for c in C:
        for si, dd in zip(c["si"], c["d"]):
            j = int(round((dd - d[0]) / step))
            lo, hi = max(j - int(4 / step), 0), min(j + int(4 / step) + 1, nb.shape[1])
            if hi - lo > 2:
                seg = nb[si, lo:hi]
                tot += 1
                hit += int(abs(seg).max() >= 0.03)
    out["ridge_existence"]["crest_picks_with_ndvi_anomaly_0p03_within_4m"] = float(hit / max(tot, 1))
    # the same rate for random positions on the same transects
    rr = 0
    nr = 0
    for _ in range(4000):
        si = int(rng.integers(0, nb.shape[0])); j = int(rng.integers(0, nb.shape[1]))
        if not sel[j]:
            continue
        lo, hi = max(j - int(4 / step), 0), min(j + int(4 / step) + 1, nb.shape[1])
        nr += 1
        rr += int(abs(nb[si, lo:hi]).max() >= 0.03)
    out["ridge_existence"]["random_positions_with_ndvi_anomaly"] = float(rr / max(nr, 1))
    (DATA / "meander.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
