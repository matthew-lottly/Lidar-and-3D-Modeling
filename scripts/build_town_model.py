"""Stage 2 of the White River town-scale utility model: from the tile detections to wire chains, support anchors, fitted wires between supports (interpolating where returns are missing), supports, and a check against
Overture's power features over the whole 3 km square. Thresholds (half the span observed, 25 returns) are the ones the 800 m study chose from a shifted-data test; the shifted-data test is run again here.
Usage: python scripts/build_town_model.py (after scripts/build_town.py)"""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely import wkb
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform

import build_spans as BS
import build_town as BT
from lidar3d import conductors as C
from lidar3d import infra3d, sitemap, spans, town, utility
from lidar3d.ept import CACHE

OUT = ROOT / "data" / "processed" / "town"
HALF = BT.HALF
MIN_COV, MIN_N = 0.5, 25
NULL_SHIFTS = 4


def load_tiles():
    tiles = town.tile_grid(HALF, BT.TILE)
    W, H, P, poles, chain = [], [], [], [], []
    G = np.full((int(2 * HALF), int(2 * HALF)), np.nan, "float32")
    meta = {}
    for i, j, x0, y0, x1, y1 in tiles:
        d = np.load(BT.TILES / f"tile_{i:02d}_{j:02d}.npz")
        W.append(d["wire_xyz"]); H.append(d["wire_hag"]); P.append(d["P"]); poles.append(d["poles"])
        gx, gy = int(x0 + HALF), int(y0 + HALF)
        g = d["ground"]
        G[gy:gy + g.shape[0], gx:gx + g.shape[1]] = g
        meta[(i, j)] = (int(d["points_read"]), str(d["date"]), float(d["cx"]), float(d["cy"]))
    return np.vstack(W), np.concatenate(H), np.vstack(P), np.vstack(poles), G, meta


def overture(cx, cy):
    tb = Transformer.from_crs(BT.CRS, 4326, always_xy=True)
    tr = Transformer.from_crs(4326, BT.CRS, always_xy=True)
    W_, S_ = tb.transform(cx - HALF - 30, cy - HALF - 30)
    E_, N_ = tb.transform(cx + HALF + 30, cy + HALF + 30)
    t = sitemap._table("base", "infrastructure", ["class", "subtype", "geometry"], (W_, S_, E_, N_), ds.field("subtype") == "power")
    feats = {}
    for _, r in t.iterrows():
        g = shp_transform(lambda x, y, z=None: tuple(np.array(tr.transform(np.asarray(x), np.asarray(y))) - np.array([[cx], [cy]])), wkb.loads(r.geometry))
        feats.setdefault(r["class"], []).append(g)
    return feats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    W, Hg, P, poles, G, meta = load_tiles()
    cx, cy = next(iter(meta.values()))[2:]
    dates = sorted({m[1] for m in meta.values() if m[1]})
    # ---- chains across tile edges, conductors, anchors
    n, lab = utility.components(W, 2.0)
    chains, hags, ends = [], [], []
    for c in range(n):
        s = lab == c
        if s.sum() >= 8 and np.hypot(*np.ptp(W[s, :2], axis=0)) >= 15.0:
            chains.append(W[s]); hags.append(Hg[s])
    for q in (c[:, :2] for c in chains):
        pr = (q - q.mean(0)) @ np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]
        ends += [q[pr.argmin()], q[pr.argmax()]]
    cond = C.build_conductors(chains, hags)
    sag = C.merge_supports([c["support_xy"] for c in cond], 6.0)
    pxy = poles[:, :2]
    keep = np.ones(len(pxy), bool)
    t = cKDTree(pxy)
    for a, b in t.query_pairs(1.5):
        keep[max(a, b)] = False
    pxy = pxy[keep]
    A = spans.anchor_candidates(sag[:, :2] if len(sag) else np.zeros((0, 2)), pxy, np.array(ends))
    prs = spans.pairs(A, max_neighbours=7)
    print(f"{len(chains)} chains, {len(poles)} poles, {len(A)} anchors, {len(prs)} pairs", flush=True)
    tree = cKDTree(P[:, :2])

    def run(Pq_tree, Pq, prs):
        out = []
        for i, j in prs:
            a, b = A[i], A[j]
            idx = Pq_tree.query_ball_point((a + b) / 2, np.hypot(*(b - a)) / 2 + 12.0)
            if len(idx) < 12:
                continue
            for w in spans.fit_pair(Pq[idx], a, b, None, corridor=8.0, max_wires=14, rng=np.random.default_rng(1), min_cov=0.2, min_n=12):
                out.append((i, j, w))
        return out

    allres = run(tree, P, prs)
    res = [(i, j, w) for i, j, w in allres if w["coverage"] >= MIN_COV and w["n"] >= MIN_N]
    real_len = sum(w["L"] for _, _, w in res)
    rng = np.random.default_rng(11)
    null_len = []
    for _ in range(NULL_SHIFTS):
        ang = rng.uniform(0, 2 * np.pi); sh = rng.uniform(120, 250) * np.array([np.cos(ang), np.sin(ang)])
        Ps = P.copy(); Ps[:, :2] += sh
        r_ = run(cKDTree(Ps[:, :2]), Ps, prs)
        null_len.append(sum(w["L"] for _, _, w in r_ if w["coverage"] >= MIN_COV and w["n"] >= MIN_N))
        print("null", null_len[-1], flush=True)
    gx = np.arange(G.shape[1]) + 0.5 - HALF; gy = np.arange(G.shape[0]) + 0.5 - HALF
    gi = RegularGridInterpolator((gy, gx), np.where(np.isfinite(G), G, np.nanmedian(G)), bounds_error=False, fill_value=None)
    ground = lambda a, b: gi(np.c_[np.atleast_1d(b), np.atleast_1d(a)])[0] if np.ndim(a) == 0 else gi(np.c_[b, a])
    sup, tiers = infra3d.supports_from_spans(A, res, ground)
    sup["through_wires"] = infra3d.through_wire_counts(res, sup)
    sup["tier"] = np.where(sup.through_wires == 0, "support", "attachment")
    # ---- compare with Overture (read only now)
    feats = overture(cx, cy)
    lines = [g for k in ("power_line", "minor_line") for g in feats.get(k, [])]
    segs = []
    for g in lines:
        for p_ in (g.geoms if hasattr(g, "geoms") else [g]):
            if p_.geom_type == "LineString":
                segs.append(p_)
    LP = np.array([s.interpolate(k * 2.0).coords[0] for s in segs for k in range(int(s.length // 2) + 1)])
    ref = np.array([(g.x, g.y) for k in ("power_pole", "power_tower") for g in feats.get(k, []) if g.geom_type == "Point"]).reshape(-1, 2)
    inb = lambda xy: (np.abs(xy[:, 0]) <= HALF) & (np.abs(xy[:, 1]) <= HALF)
    LP, ref = LP[inb(LP)], ref[inb(ref)]
    Wf = np.vstack([spans.wire_xyz(w, 2.0)[0] for _, _, w in res]); Fl = np.concatenate([spans.wire_xyz(w, 2.0)[1] for _, _, w in res])
    tw = cKDTree(Wf[:, :2])
    dLP = tw.query(LP)[0]
    so = sup[sup.tier == "support"]
    stats = dict(
        project=BT.PROJECT, collected=dates, centre_lonlat=BT.LONLAT, area_km2=(2 * HALF / 1000) ** 2, tiles=len(meta), points_read_incl_buffers=int(sum(m[0] for m in meta.values())),
        detections=dict(wire_points=int(len(W)), wire_chains=len(chains), poles=int(keep.sum()), elevated_candidate_points=int(len(P))),
        anchors=int(len(A)), pairs=int(len(prs)), wires=int(len(res)), wire_length_m=float(real_len), observed_share=float((Fl == 1).mean()) if len(Fl) else None,
        supports=int((sup.tier == "support").sum()), attachment_only=int((sup.tier != "support").sum()),
        null=dict(shifts=NULL_SHIFTS, null_length_m=[float(v) for v in null_len], null_to_real_ratio=float(np.mean(null_len) / real_len) if real_len else None),
        overture=dict(power_poles_and_towers=int(len(ref)), mapped_line_points=int(len(LP)), mapped_line_covered_within_6m=float((dLP <= 6).mean()) if len(LP) else None,
                      wire_within_10m_of_a_mapped_line=float((cKDTree(LP).query(Wf[:, :2])[0] <= 10).mean()) if len(LP) else None,
                      supports_vs_poles={r_: BS._eval_supports(so[["x", "y"]].to_numpy(), ref, r_) for r_ in (3.0, 5.0, 8.0)} if len(ref) and len(so) else None,
                      detected_poles_vs_poles={r_: BS._eval_supports(pxy, ref, r_) for r_ in (3.0, 5.0)} if len(ref) else None),
        interpolation_holdout=BS.holdout(res),
    )
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=float))
    sup.to_csv(OUT / "supports.csv", index=False)
    pd.DataFrame([dict(easting=round(p[0] + cx, 2), northing=round(p[1] + cy, 2), top_hag_m=round(p[2], 1), height_m=round(p[3], 1), points=int(p[4])) for p in poles[keep]]).to_csv(OUT / "poles_detected.csv", index=False)
    pd.DataFrame([dict(pair=f"{i}-{j}", length_m=w["L"], coverage=w["coverage"], n=w["n"], sag_m=w["sag"], z_end_a=w["z_ends"][0], z_end_b=w["z_ends"][1]) for i, j, w in res]).to_csv(OUT / "wires.csv", index=False)
    pickle.dump(dict(A=A, res=res, sup=sup, tiers=tiers, W=W, poles=poles[keep], G=G, LP=LP, ref=ref, cx=cx, cy=cy), open(CACHE / "town" / "model.pkl", "wb"))
    print(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()
