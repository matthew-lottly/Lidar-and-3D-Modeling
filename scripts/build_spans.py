"""Support-anchored wire model, support and tower models and a 3D scene for White River Substation, from lidar alone (NAIP imagery only colours the 3D scene).
Usage: python scripts/build_spans.py   (needs data fetched by scripts/build_utility.py and scripts/build_fusion.py; about 10 minutes the first time, cached after)

OpenStreetMap (via Overture) is read only at the end, to compare."""
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
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import LineString
from shapely.ops import unary_union

import build_fusion as BF
import build_utility as BU
from lidar3d import infra3d, spans, utility
from lidar3d.ept import CACHE

OUT = ROOT / "data" / "processed" / "spans"
HALF = BU.HALF
MIN_COV, MIN_N = 0.5, 25          # chosen from the shifted-data test (about 10% false length), not from the map
NULL_SHIFTS = 4


def cached(name, fn):
    p = CACHE / "utility" / name
    if p.exists():
        return pickle.loads(p.read_bytes())
    v = fn()
    p.write_bytes(pickle.dumps(v))
    return v


def run_pairs(P, A, prs, **kw):
    out = []
    for i, j in prs:
        for w in spans.fit_pair(P, A[i], A[j], None, corridor=8.0, max_wires=14, rng=np.random.default_rng(1), **kw):
            out.append((i, j, w))
    return out


def holdout(res, gaps=(20, 40), trials=3, seed=0):
    """Hide a stretch of a well-observed wire, refit on the rest and measure the error inside the hidden stretch. Hidden stretches are inliers of the original fit, so errors are optimistic for noisy wires."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, _, w in res:
        if w["coverage"] < 0.85 or w["L"] < 90:
            continue
        pts, L = w["pts"], w["L"]
        for gap in gaps:
            if L < gap * 2.2:
                continue
            for _ in range(trials):
                c0 = rng.uniform(0.2 * L + gap / 2, 0.8 * L - gap / 2)
                m = np.abs(pts[:, 0] - c0) <= gap / 2
                if m.sum() < 5 or (~m).sum() < 20:
                    continue
                tr_, te = pts[~m], pts[m]
                l, r = tr_[tr_[:, 0] < c0 - gap / 2], tr_[tr_[:, 0] > c0 + gap / 2]
                if len(l) < 3 or len(r) < 3:
                    continue
                pl, pr = l[np.argsort(-l[:, 0])[:3]].mean(0), r[np.argsort(r[:, 0])[:3]].mean(0)
                zl = pl[1] + (te[:, 0] - pl[0]) * (pr[1] - pl[1]) / (pr[0] - pl[0])
                zp = np.polyval(np.polyfit(tr_[:, 0], tr_[:, 1], 2), te[:, 0])
                rows.append((gap, np.sqrt(np.mean((zl - te[:, 1]) ** 2)), np.sqrt(np.mean((zp - te[:, 1]) ** 2))))
    R = np.array(rows)
    return {int(g): dict(n=int((R[:, 0] == g).sum()), rms_linear_m=float(np.sqrt(np.mean(R[R[:, 0] == g, 1] ** 2))), rms_parabola_m=float(np.sqrt(np.mean(R[R[:, 0] == g, 2] ** 2)))) for g in gaps}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, z, cls, date, cx, cy, tr = BU.load_points()
    fr = pickle.loads((CACHE / "utility" / "fusion_result.pkl").read_bytes())
    xyz = np.c_[x, y, z]
    hag, gg = utility.height_above_ground(x, y, z, cls, (-HALF, -HALF, HALF, HALF), 1.0)
    gx = np.arange(gg.shape[1]) + 0.5 - HALF; gy = np.arange(gg.shape[0]) + 0.5 - HALF
    gi = RegularGridInterpolator((gy, gx), gg, bounds_error=False, fill_value=None)
    ground = lambda a, b: gi(np.c_[np.atleast_1d(b), np.atleast_1d(a)])[0] if np.ndim(a) == 0 else gi(np.c_[b, a])
    # ---- elevated returns that are not dense masses, anchors, spans
    P = cached("elevated_points.pkl", lambda: spans.candidate_points(xyz, hag)[0])
    wl = fr["wlab"]; ends = []
    for c in np.unique(wl[wl >= 0]):
        q = np.c_[x[wl == c], y[wl == c]]
        pr = (q - q.mean(0)) @ np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]
        ends += [q[pr.argmin()], q[pr.argmax()]]
    A = spans.anchor_candidates(fr["S"][["x", "y"]].to_numpy(), np.zeros((0, 2)), np.array(ends))
    prs = spans.pairs(A, max_neighbours=7)
    res = cached("spans_result.pkl", lambda: run_pairs(P, A, prs, min_cov=0.2, min_n=12))
    nulls = cached("spans_nulls.pkl", lambda: _nulls(P, A, prs))
    # thresholds are applied after the fact so the same search serves the real and the shifted data
    keep = lambda W: [w for w in W if w["coverage"] >= MIN_COV and w["n"] >= MIN_N]
    res = [(i, j, w) for i, j, w in res if w["coverage"] >= MIN_COV and w["n"] >= MIN_N]
    real_len = sum(w["L"] for _, _, w in res)
    null_len = [sum(w["L"] for w in keep(W)) for W in nulls]
    # ---- supports from where wires attach
    sup, tiers = infra3d.supports_from_spans(A, res, ground)
    sup["through_wires"] = infra3d.through_wire_counts(res, sup)
    sup["tier"] = np.where(sup.through_wires == 0, "support", "attachment")
    # ---- compare with the earlier fragment-chain model, and the map
    feats = BU.overture(cx, cy, tr)
    yard = [g for g, _ in feats["substation"] if g.geom_type == "Polygon"][0].buffer(BF.YARD_BUFFER)
    recs = BF.osm_lines_tagged(cx, cy, tr)
    U = unary_union([LineString(r["coords"]) for r in recs]).difference(yard)
    LP = np.array([g.interpolate(i * 2.0).coords[0] for g in (U.geoms if hasattr(U, "geoms") else [U]) for i in range(int(g.length // 2) + 1)])
    W = np.vstack([spans.wire_xyz(w, 2.0)[0] for _, _, w in res]); Fl = np.concatenate([spans.wire_xyz(w, 2.0)[1] for _, _, w in res])
    out = ~contains_xy(yard, W[:, 0], W[:, 1]); Wo, Fo = W[out], Fl[out]
    old = np.vstack([c_["xy"][c_["flag"] >= 1] for c_ in fr["cond"]])
    old = old[~contains_xy(yard, old[:, 0], old[:, 1])]
    cov = lambda pts, r=6.0: float((cKDTree(pts[:, :2]).query(LP)[0] <= r).mean())
    wm = np.c_[x, y, z][fr["wmask"]]; wm = wm[~contains_xy(yard, wm[:, 0], wm[:, 1])]
    rec = lambda pts, r=1.0: float((cKDTree(pts).query(wm)[0] <= r).mean())
    Wf = np.vstack([spans.wire_xyz(w, 0.5)[0] for _, _, w in res]); Wf = Wf[~contains_xy(yard, Wf[:, 0], Wf[:, 1])]
    ref = fr["ref"]
    outy = ~contains_xy(yard, sup.x.to_numpy(), sup.y.to_numpy())
    so = sup[outy & (sup.tier == "support")]
    ev = {r_: _eval_supports(so[["x", "y"]].to_numpy(), ref, r_) for r_ in (3.0, 5.0, 8.0)}
    ev_all = {r_: _eval_supports(sup[outy][["x", "y"]].to_numpy(), ref, r_) for r_ in (3.0, 5.0, 8.0)}
    # offset of mapped lines from the nearest fitted wire (lines the model reports uncovered)
    tw = cKDTree(Wo[:, :2]); dLP = tw.query(LP)[0]
    stats = dict(
        anchors=int(len(A)), pairs=int(len(prs)), wires=int(len(res)), wire_length_m=float(real_len), observed_share=float((Fl == 1).mean()), supports=int((sup.tier == 'support').sum()), attachment_only=int((sup.tier != 'support').sum()), supports_outside_yard=int(len(so)),
        support_kinds=sup[sup.tier == 'support'].kind.value_counts().to_dict(),
        null=dict(shifts=NULL_SHIFTS, null_length_m=[float(v) for v in null_len], null_to_real_ratio=float(np.mean(null_len) / real_len)),
        osm=dict(coverage_new=cov(Wo), coverage_new_observed_only=cov(Wo[Fo == 1]), coverage_old_chain_model=cov(old), mapped_points=int(len(LP)),
                 uncovered_median_offset_to_nearest_wire_m=float(np.median(dLP[dLP > 6])), uncovered_share_within_15m=float(((dLP > 6) & (dLP <= 15)).sum() / max((dLP > 6).sum(), 1)),
                 wire_near_map_10m=float((cKDTree(LP).query(Wo[:, :2])[0] <= 10).mean())),
        lidar_recall=dict(new=rec(Wf)),
        interpolation_holdout=holdout(res), supports_vs_osm=ev, all_attachments_vs_osm=ev_all, osm_supports_outside_yard=int(len(ref)),
    )
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=float))
    sup.to_csv(OUT / "supports.csv", index=False)
    pd.DataFrame([dict(pair=f"{i}-{j}", length_m=w["L"], coverage=w["coverage"], n=w["n"], sag_m=w["sag"], z_end_a=w["z_ends"][0], z_end_b=w["z_ends"][1], side0=w["side0"], side1=w["side1"]) for i, j, w in res]).to_csv(OUT / "wires.csv", index=False)
    pickle.dump(dict(A=A, res=res, sup=sup, tiers=tiers, P=P, gg=gg, nulls_len=null_len, Wo=Wo, LP=LP, hag=hag), open(CACHE / "utility" / "spans_final.pkl", "wb"))
    print(json.dumps(stats, indent=1, default=float))


def _nulls(P, A, prs):
    rng = np.random.default_rng(11)
    out = []
    for _ in range(NULL_SHIFTS):
        ang = rng.uniform(0, 2 * np.pi); sh = rng.uniform(120, 250) * np.array([np.cos(ang), np.sin(ang)])
        Ps = P.copy(); Ps[:, :2] += sh
        out.append([w for _, _, w in run_pairs(Ps, A, prs, min_cov=0.2, min_n=12)])
    return out


def _eval_supports(det, ref, r):
    t = cKDTree(ref); d, _ = t.query(det)
    td = cKDTree(det); d2, _ = td.query(ref)
    return dict(detected=int(len(det)), matched=int((d <= r).sum()), precision=float((d <= r).mean()), recall=float((d2 <= r).mean()), median_offset_m=float(np.median(d[d <= r])) if (d <= r).any() else None)


if __name__ == "__main__":
    main()
