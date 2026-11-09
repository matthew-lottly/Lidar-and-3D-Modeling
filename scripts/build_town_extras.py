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


def learned_supports(thr=0.12):
    m = pickle.load(open(CACHE / "town" / "support_model.pkl", "rb"))
    sel = m["oof"] >= thr
    k = TL.nms(m["C"][sel], m["oof"][sel], 6.0)
    return m["C"][sel][k], m["oof"][sel][k]


def span_stage(d, P, hagP, feats):
    """Candidate spans between anchors (the geometric ones plus the learned supports), scored by a model trained on spatially held-out tiles; dangling line ends are then bridged where two ends face each other."""
    LS, LSs = learned_supports()
    A0 = d["A"]
    extra = [p for p in LS if np.hypot(*(A0 - p).T).min() > 8.0]
    A = np.vstack([A0, np.array(extra).reshape(-1, 2)]) if extra else A0
    prs = spans.pairs(A, max_neighbours=7)
    tree = cKDTree(P[:, :2])
    allres = []
    for i, j in prs:
        a, b = A[i], A[j]
        idx = tree.query_ball_point((a + b) / 2, np.hypot(*(b - a)) / 2 + 12.0)
        if len(idx) < 12:
            continue
        for w in spans.fit_pair(P[idx], a, b, None, corridor=8.0, max_wires=14, rng=np.random.default_rng(1), min_cov=0.2, min_n=12):
            allres.append((i, j, w))
    print(len(A), "anchors", len(prs), "pairs", len(allres), "weak-or-better wires", flush=True)
    # pair level table
    LP = sample_lines(feats)
    tl = cKDTree(LP[["x", "y"]].to_numpy())
    best = {}
    for i, j, w in allres:
        sc = w["coverage"] * w["n"]
        if (i, j) not in best or sc > best[(i, j)][1]["coverage"] * best[(i, j)][1]["n"]:
            best[(i, j)] = (w, w)
    rows = []
    sup_t = cKDTree(LS)
    for (i, j), (w, _) in best.items():
        xyz, flag = spans.wire_xyz(w, 2.0)
        dm = tl.query(xyz[:, :2])[0]
        mid = (A[i] + A[j]) / 2
        e_i, e_j = sup_t.query(A[i])[0], sup_t.query(A[j])[0]
        rows.append(dict(i=i, j=j, L=w["L"], coverage=w["coverage"], n=w["n"], sag_ratio=w["sag"] / w["L"] ** 2 * 1000, dz_ratio=abs(w["z_ends"][0] - w["z_ends"][1]) / w["L"], z_a=w["z_ends"][0], z_b=w["z_ends"][1],
                         near_learned_a=float(e_i <= 8), near_learned_b=float(e_j <= 8), label=float((dm <= 6).mean() >= 0.7), frac_near_mapped=float((dm <= 6).mean()),
                         tile=int(np.floor((mid[0] + HALF) / BT.TILE) * 10 + np.floor((mid[1] + HALF) / BT.TILE))))
    T = pd.DataFrame(rows)
    # network context features: degree at each end and the best collinear confirmed neighbour
    conf = T[(T.coverage >= BM.MIN_COV) & (T.n >= BM.MIN_N)]
    def direction(r):
        v = A[int(r.j)] - A[int(r.i)]
        return v / np.hypot(*v)
    deg = {}
    for r in T.itertuples():
        for k in (r.i, r.j):
            deg[k] = deg.get(k, 0) + 1
    cdeg = {}
    cdirs = {}
    for r in conf.itertuples():
        u = direction(r)
        cdirs.setdefault(r.i, []).append(u); cdirs.setdefault(r.j, []).append(-u)
        for k in (r.i, r.j):
            cdeg[k] = cdeg.get(k, 0) + 1
    def cont_angle(r, end):
        u = direction(r) if end == "j" else -direction(r)
        k = r.j if end == "j" else r.i
        nb = [-v for v in cdirs.get(k, []) if abs(v @ u) < 0.999999 or True]
        a = []
        for v in cdirs.get(k, []):
            # confirmed span leaves anchor k along v; our span leaves k along -u; a straight continuation means v ~ u (v points away from k, -u points away too)
            a.append(np.degrees(np.arccos(np.clip(v @ (u * -1) * -1, -1, 1))))
        a = [x for x in a if x > 1.0]      # skip the span itself
        return min(a) if a else 90.0
    T["deg_a"] = [deg[i] for i in T.i]; T["deg_b"] = [deg[j] for j in T.j]
    T["cdeg_a"] = [cdeg.get(i, 0) for i in T.i]; T["cdeg_b"] = [cdeg.get(j, 0) for j in T.j]
    T["cont_a"] = [cont_angle(r, "i") for r in T.itertuples()]; T["cont_b"] = [cont_angle(r, "j") for r in T.itertuples()]
    names = ["L", "coverage", "n", "sag_ratio", "dz_ratio", "z_a", "z_b", "near_learned_a", "near_learned_b", "deg_a", "deg_b", "cdeg_a", "cdeg_b", "cont_a", "cont_b"]
    X, y, g = T[names].to_numpy(float), T.label.to_numpy(int), T.tile.to_numpy()
    oof = np.zeros(len(T)); aucs = []
    for tr, te in GroupKFold(5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=8, l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        if y[te].min() != y[te].max():
            aucs.append(roc_auc_score(y[te], oof[te]))
    # baseline: the coverage rule of the geometric model
    base = ((T.coverage >= BM.MIN_COV) & (T.n >= BM.MIN_N)).to_numpy()
    T["p"] = oof
    # operating point chosen on the held-out scores: the lowest threshold whose accepted spans keep precision against the map at least as high as the geometric rule's
    def stats(sel):
        return dict(spans=int(sel.sum()), length_m=float(T.L[sel].sum()), precision_by_length=float((T.frac_near_mapped[sel] * T.L[sel]).sum() / max(T.L[sel].sum(), 1)))
    gb = stats(base)
    pick = None
    for t_ in np.arange(0.9, 0.05, -0.025):
        s_ = stats(oof >= t_)
        if s_["precision_by_length"] >= gb["precision_by_length"] - 0.02 and s_["spans"] > 0:
            pick = float(t_)
        elif pick is not None:
            break
    pick = pick if pick is not None else 0.5
    acc = oof >= pick
    T["accepted"] = acc; T["baseline"] = base
    # bridging: dangling confirmed-or-accepted ends facing another dangling end
    use = T[acc | base]
    dd = {}
    for r in use.itertuples():
        for k in (r.i, r.j):
            dd[k] = dd.get(k, 0) + 1
    dirs = {}
    for r in use.itertuples():
        u = direction(r); dirs.setdefault(r.i, []).append(u); dirs.setdefault(r.j, []).append(-u)
    dang = [k for k, v in dd.items() if v == 1]
    have = {(min(r.i, r.j), max(r.i, r.j)) for r in use.itertuples()}
    bridges = []
    for a in dang:
        ua = -dirs[a][0]   # the span arrives at a travelling along -dirs? dirs[a][0] points away from a into the span; the continuation points the opposite way
        for b in dang:
            if b <= a or (a, b) in have:
                continue
            v = A[b] - A[a]; L = np.hypot(*v)
            if not (25 <= L <= 160):
                continue
            v /= L
            ub = -dirs[b][0]
            ang_a = np.degrees(np.arccos(np.clip(v @ ua, -1, 1))); ang_b = np.degrees(np.arccos(np.clip(-v @ ub, -1, 1)))
            if ang_a <= 15 and ang_b <= 15:
                bridges.append((a, b, L, max(ang_a, ang_b)))
    # sag model for bridged spans from the confirmed ones: parabola sag = k L^2
    k = float(np.median(T.sag_ratio[base]) / 1000) if base.any() else 0.0
    zend = {}
    for r in use.itertuples():
        zend.setdefault(r.i, []).append(r.z_a); zend.setdefault(r.j, []).append(r.z_b)
    brs = []
    for a, b, L, ang in bridges:
        za, zb = float(np.median(zend[a])), float(np.median(zend[b]))
        s_ = np.arange(0, L + 1e-6, 2.0)
        u = (A[b] - A[a]) / L
        xy = A[a] + s_[:, None] * u
        z = za + (zb - za) * s_ / L - k * s_ * (L - s_)
        brs.append(dict(a=int(a), b=int(b), L=float(L), angle_deg=float(ang), xyz=np.c_[xy, z]))
    # coverage of the mapped lines
    def cover(parts):
        Wp = np.vstack(parts)
        dl = cKDTree(Wp[:, :2]).query(LP[["x", "y"]].to_numpy())[0]
        return float((dl <= 6).mean())
    wires_of = lambda sel: [spans.wire_xyz(best[(int(r.i), int(r.j))][0], 2.0)[0] for r in T[sel].itertuples()]
    c_base = cover(wires_of(base))
    c_acc = cover(wires_of(base | acc))
    c_br = cover(wires_of(base | acc) + [b["xyz"] for b in brs]) if brs else c_acc
    brl = np.array([cKDTree(LP[["x", "y"]].to_numpy()).query(b["xyz"][:, :2])[0] for b in brs], dtype=object)
    br_prec = float(np.mean([np.mean(x <= 6) for x in brl])) if len(brl) else None
    # network connectivity: nodes = anchors, edges = spans
    import scipy.sparse as sp_
    from scipy.sparse.csgraph import connected_components as cc
    def comps(edges):
        e = np.array(edges).reshape(-1, 2)
        nodes = np.unique(e)
        mp = {v: n for n, v in enumerate(nodes)}
        g = sp_.coo_matrix((np.ones(len(e)), ([mp[a] for a in e[:, 0]], [mp[b] for b in e[:, 1]])), shape=(len(nodes),) * 2)
        n, lab = cc(g, directed=False)
        return int(n), float(np.bincount(lab).max() / len(nodes)), int(len(nodes))
    e0 = [(r.i, r.j) for r in T[base].itertuples()]
    e1 = e0 + [(r.i, r.j) for r in T[acc & ~base].itertuples()]
    e2 = e1 + [(b["a"], b["b"]) for b in brs]
    conn = {nm: dict(zip(("components", "largest_component_share_of_nodes", "nodes"), comps(e))) for nm, e in (("confirmed", e0), ("plus_scored_spans", e1), ("plus_bridged", e2))}
    out = dict(anchors=int(len(A)), anchors_added_from_learned_supports=int(len(extra)), candidate_pairs=int(len(prs)), pair_table=int(len(T)), labelled_on_map=int(y.sum()), auc_per_fold=[float(a) for a in aucs], auc_mean=float(np.mean(aucs)),
               features=names, geometric_rule=gb, threshold=pick, scored_spans=stats(acc), added_by_model=stats(acc & ~base), bridged=dict(spans=len(brs), length_m=float(sum(b["L"] for b in brs)), fraction_of_length_within_6m_of_mapped_line=br_prec, max_angle_deg=15, gap_range_m=[25, 160],
                                                                                                                    sag_coefficient=k), mapped_line_covered_within_6m=dict(confirmed=c_base, plus_scored_spans=c_acc, plus_bridged=c_br), connectivity=conn,
               note="labels are 'within 6 m of a mapped line for 70% of the span', so precision against the map is a lower bound; bridged spans have no returns at all and are inferred from the line continuing straight")
    (OUT / "extras_spans.json").write_text(json.dumps(out, indent=1, default=float))
    pickle.dump(dict(A=A, T=T, best=best, brs=brs, LS=LS, LSs=LSs, pick=pick), open(CACHE / "town" / "spans_extras.pkl", "wb"))
    print(json.dumps(out, indent=1, default=float))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    d, W_, Hg, P, hagP, gi, ndvi, feats, meta = load()
    if what in ("diagnose", "all"):
        diagnose(d, P, hagP, ndvi, feats)
    if what in ("support", "all"):
        learn_support(d, W_, Hg, P, hagP, gi, ndvi, meta)
    if what in ("spans", "all"):
        span_stage(d, P, hagP, feats)


if __name__ == "__main__":
    main()
