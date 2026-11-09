"""Whole-area wire completion for the White River town model (run after scripts/build_town_extras.py spans): every detected wire piece in the 3 km square joins the model, not only the spans fitted between supports.
Same method as the close-up models (src/lidar3d/unified.py): re-linked wire pieces drawn where observed, the pieces a fitted span already explains dropped, sag-curve bridges between pieces that line up, and posts at detected poles.
Output: data/processed/town/extras_unified.json and a cache for plate 57."""
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
from scipy.spatial import cKDTree

import build_town as BT
import build_town_model as BM
from lidar3d import spans, unified as U, utility
from lidar3d.ept import CACHE

OUT = ROOT / "data" / "processed" / "town"
HALF = BT.HALF


def main():
    d = pickle.load(open(CACHE / "town" / "model.pkl", "rb"))
    sp = pickle.load(open(CACHE / "town" / "spans_extras.pkl", "rb"))
    W, Hg, P, poles, G, meta = BM.load_tiles()
    Gd = dict(z=np.where(np.isfinite(G), G, np.nanmedian(G)), x0=-HALF, y0=-HALF, res=1.0)
    gz = U.ground_fn(Gd)
    n, lab = utility.components(W, 2.0)
    print(len(W), "wire points,", n, "chains", flush=True)
    chain_l, _ = U.chain_wires(W, Hg, lab, gz, min_pts=8)
    T, best = sp["T"], sp["best"]
    sel = T[T.baseline | T.accepted]
    span_w = [spans.wire_xyz(best[(int(r.i), int(r.j))][0], 1.0)[0] for r in sel.itertuples()] + [b["xyz"] for b in sp["brs"]]
    adds, kept = U.merge_wires(span_w, chain_l)
    before = U.coverage(W, span_w, None)
    after = U.coverage(W, [dict(xyz=w, flag=np.ones(len(w))) for w in span_w] + adds, None)
    # supports: span supports, learned supports and every detected pole
    sup_xy = np.vstack([d["sup"][d["sup"].tier == "support"][["x", "y"]].to_numpy(), sp["LS"], poles[:, :2]])
    k = float(sp["brs"] and 0) or float(np.median([w["sag"] / w["L"] ** 2 for _, _, w in d["res"]]))
    allw = span_w + [a["xyz"] for a in adds]
    br = U.bridge_ends(allw, sup_xy, k, max_gap=90.0)
    ends = np.array([e[:2] for w in allw if len(w) > 1 and np.hypot(*(w[-1, :2] - w[0, :2])) >= 10.0 for e in (w[0], w[-1])]).reshape(-1, 2)
    st = cKDTree(sup_xy)
    bt = cKDTree(np.array([e[:2] for b in br for e in (b["xyz"][0], b["xyz"][-1])]).reshape(-1, 2)) if br else None
    free_sup = float((st.query(ends)[0] > 3.0).mean())
    free_both = float(np.mean([(st.query(e)[0] > 3.0) and not (bt is not None and bt.query(e)[0] <= 1.0) for e in ends]))
    after_b = U.coverage(W, [dict(xyz=w, flag=np.ones(len(w))) for w in allw] + br, None)
    out = dict(wire_points=int(len(W)), fine_chains=len(chain_l), span_wires=len(span_w), chain_pieces_added=len(adds), chain_length_share_kept=kept, bridges=len(br), bridged_length_m=float(sum(np.hypot(*np.diff(b["xyz"][:, :2], axis=0).T).sum() for b in br)),
               sag_k=k, coverage_before=before, coverage_after_chain_pieces=after, coverage_after_with_bridges=after_b, wire_ends_ge10m=int(len(ends)), free_ends_before_bridges=free_sup, free_ends_after_bridges=free_both,
               note="coverage = share of detected wire points within 1 m of a model wire; 'before' is the fitted spans (first rule plus model-added) and the bridged line ends of plate 64")
    (OUT / "extras_unified.json").write_text(json.dumps(out, indent=1, default=float))
    pickle.dump(dict(adds=adds, br=br, chain_l=chain_l), open(CACHE / "town" / "unified.pkl", "wb"))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
