"""Learned layers for the White River town model (run after scripts/build_town_model.py). Usage: python scripts/build_town_extras.py [diagnose|support|equipment|spans|all]
Output: data/processed/town/extras_*.json, CSVs, and caches under data/cache/town for the plates."""
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
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import build_town as BT
import build_town_model as BM
from lidar3d import fusion, naip, spans, town_learn as TL
from lidar3d.aoi import AOI
from lidar3d.ept import CACHE

OUT = ROOT / "data" / "processed" / "town"
HALF = BT.HALF


def load():
    d = pickle.load(open(CACHE / "town" / "model.pkl", "rb"))
    W_, Hg, P, poles, G, meta = BM.load_tiles()
    gi = TL.Grid(np.where(np.isfinite(G), G, np.nanmedian(G)), HALF, 1.0, True)
    img, tf, nd = naip.fetch(AOI(round(d["cx"]), round(d["cy"]), HALF, BT.CRS), 2019, res=1.0, slug="whiteriver_town")
    ndvi = TL.Grid(fusion.ndvi(img), HALF, 1.0, False)
    hagP = P[:, 2] - gi.at(P[:, 0], P[:, 1])
    feats = BM.overture(d["cx"], d["cy"])
    return d, W_, Hg, P, hagP, gi, ndvi, feats, meta


def sample_lines(feats, kinds=("power_line", "minor_line")):
    rows = []
    for k in kinds:
        for g in feats.get(k, []):
            for p_ in (g.geoms if hasattr(g, "geoms") else [g]):
                if p_.geom_type != "LineString":
                    continue
                for i in range(int(p_.length // 2) + 1):
                    x, y = p_.interpolate(i * 2.0).coords[0]
                    if abs(x) <= HALF and abs(y) <= HALF:
                        rows.append((x, y, k))
    return pd.DataFrame(rows, columns=["x", "y", "kind"])


def diagnose(d, P, hagP, ndvi, feats):
    from lidar3d import spans as SP
    Wf = np.vstack([SP.wire_xyz(w, 2.0)[0] for _, _, w in d["res"]])
    LP = sample_lines(feats)
    xy = LP[["x", "y"]].to_numpy()
    tw = cKDTree(Wf[:, :2])
    dw = tw.query(xy)[0]
    mask = hagP >= 5.0
    tP = cKDTree(P[mask][:, :2])
    nP = np.array(tP.query_ball_point(xy, 3.0, return_length=True))
    sup = d["sup"][d["sup"].tier == "support"][["x", "y"]].to_numpy()
    dsup = cKDTree(sup).query(xy)[0] if len(sup) else np.full(len(xy), np.inf)
    nd = ndvi.window_mean(xy[:, 0], xy[:, 1], 4.0)
    cat = np.where(dw <= 6, "fitted wire within 6 m",
          np.where(dw <= 15, "fitted wire 6 to 15 m away (map offset or parallel line)",
          np.where(nP >= 5, "elevated returns on the line, no wire fitted",
          np.where(nd >= 0.3, "no elevated returns, under vegetation", "no elevated returns, open ground"))))
    LP["category"], LP["dist_fitted_wire_m"], LP["elevated_returns_3m"], LP["ndvi_4m"], LP["dist_support_m"] = cat, dw, nP, nd, dsup
    km = (LP.groupby(["kind", "category"]).size() * 2 / 1000).round(2)
    res = dict(sample_spacing_m=2.0, km=json.loads(km.unstack(0).fillna(0).to_json()), total_km=float(len(LP) * 2 / 1000),
               share={c: float((cat == c).mean()) for c in np.unique(cat)},
               median_ndvi_by_category={c: float(np.nanmedian(nd[cat == c])) for c in np.unique(cat)},
               median_distance_to_nearest_detected_support_m=float(np.median(dsup[np.isfinite(dsup)])),
               note="ndvi_4m is the mean NAIP (2019) NDVI within 4 m of the mapped line; 0.3 and above is taken as vegetation. The map itself is not checked: a mapped line may be absent, relocated or mis-placed.")
    LP.to_csv(OUT / "mapped_line_diagnosis.csv", index=False)
    (OUT / "extras_diagnosis.json").write_text(json.dumps(res, indent=1))
    pickle.dump(LP, open(CACHE / "town" / "diag.pkl", "wb"))
    print(json.dumps(res, indent=1))


def learn_support(d, W_, Hg, P, hagP, gi, ndvi, tiles_meta):
    ref = d["ref"]
    Pm = hagP >= 5.0
    Pv = P[Pm]
    Hv = hagP[Pm]
    tree = cKDTree(Pv[:, :2])
    C = TL.candidates(Pv, Hv, half=HALF)
    n_all = len(C)
    C = C[TL.column_filter(C, Pv, Hv, tree)]
    wtree = cKDTree(W_[:, :2])
    print(len(C), "candidates", flush=True)
    X, names = TL.support_features(C, Pv, Hv, tree, W_, Hg, wtree, ndvi)
    dref = cKDTree(ref).query(C)[0]
    y = (dref <= 4.0).astype(int)
    grp = (np.floor((C[:, 0] + HALF) / BT.TILE).astype(int) * 10 + np.floor((C[:, 1] + HALF) / BT.TILE).astype(int))
    oof = np.zeros(len(C))
    gkf = GroupKFold(5)
    aucs = []
    for tr, te in gkf.split(X, y, grp):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        if y[te].min() != y[te].max():
            aucs.append(roc_auc_score(y[te], oof[te]))
    thr = np.round(np.arange(0.3, 0.99, 0.03), 2)
    geo = d["sup"][d["sup"].tier == "support"][["x", "y"]].to_numpy()
    rawp = d["poles"][:, :2]
    out = dict(candidates_before_column_filter=int(n_all), candidates=int(len(C)), positives=int(y.sum()), positive_share=float(y.mean()), features=names, folds=5, auc_per_fold=[float(a) for a in aucs], auc_mean=float(np.mean(aucs)), auc_oof=float(roc_auc_score(y, oof)),
               mapped_poles_and_towers=int(len(ref)), pr_3m={}, pr_5m={}, geometric={})
    for r_, key in ((3.0, "pr_3m"), (5.0, "pr_5m")):
        out[key] = [dict(threshold=float(t), detections=n, precision=None if p != p else float(p), recall=float(rc)) for t, n, p, rc in TL.pr_curve(C, oof, ref, r_, thr)]
        for nm, pts in (("geometric_supports", geo), ("raw_pole_detector", rawp)):
            mi, _, _ = TL.match(pts, ref, r_)
            out["geometric"][f"{nm}_{int(r_)}m"] = dict(detections=int(len(pts)), precision=float(len(mi) / len(pts)), recall=float(len(mi) / len(ref)))
    # permutation-free importance: the drop in held-out AUC is not computed; report the fitted model's split gains is not available, so only feature names are listed
    final = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X, y)
    pickle.dump(dict(C=C, X=X, y=y, oof=oof, names=names, grp=grp, model=final), open(CACHE / "town" / "support_model.pkl", "wb"))
    (OUT / "extras_support.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("features",)}, indent=1)[:3500])


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    d, W_, Hg, P, hagP, gi, ndvi, feats, meta = load()
    if what in ("diagnose", "all"):
        diagnose(d, P, hagP, ndvi, feats)
    if what in ("support", "all"):
        learn_support(d, W_, Hg, P, hagP, gi, ndvi, meta)


if __name__ == "__main__":
    main()
