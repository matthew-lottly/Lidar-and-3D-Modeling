"""Canopy connectivity: a graph whose nodes are canopy patches and whose links are the gaps between them.

Nodes are 8-connected patches of canopy (CHM at or above a height) of at least `min_area` m2. Two patches are linked, with the distance of the closest approach of their edges, when their nearest-patch
(Voronoi) regions touch; a gap-crossing distance T keeps the links of T metres or less. From that graph:
  * clusters at T (components) and the share of canopy area in the largest one;
  * the probability of connectivity (PC, Saura and Pascual-Vazquez 2007) and its equivalent connected area (ECA), with p_ij = exp(-d_ij / alpha) where d_ij is the shortest path of summed gaps;
  * reachable canopy area of each patch, sum_j a_j p_ij (how much canopy a patch can get to);
  * stepping stones: patches whose removal splits a cluster (articulation points).
T and alpha are scenario parameters, not properties of any species. The window cuts patches at its edge, so connectivity through canopy outside the window is not counted.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def patch_graph(chm, res, min_h=2.0, min_area=20.0):
    """Returns dict: labels (grid, 0 = none, node ids start at 1), area (m2 per node), centroid (row, col per node), touches_edge (bool per node), edges (i, j, gap in m; zero-based node index)."""
    mask = chm >= min_h
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    area = ndimage.sum(mask, lab, np.arange(1, n + 1)) * res * res
    keep = area >= min_area
    remap = np.zeros(n + 1, int)
    remap[1:][keep] = np.arange(1, keep.sum() + 1)
    lab = remap[lab]
    m = lab > 0
    k = int(keep.sum())
    idx = np.arange(1, k + 1)
    area = ndimage.sum(m, lab, idx) * res * res
    cen = np.array(ndimage.center_of_mass(m, lab, idx)) if k else np.zeros((0, 2))
    edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    touches = np.array([i in edge_ids for i in idx])
    # nearest patch and distance for every cell
    dist, (ir, ic) = ndimage.distance_transform_edt(~m, return_indices=True)
    nl = lab[ir, ic]
    pairs = {}
    for a, b, da, db in ((nl[:, :-1], nl[:, 1:], dist[:, :-1], dist[:, 1:]), (nl[:-1], nl[1:], dist[:-1], dist[1:])):
        sel = (a != b)
        ia, ib, d = a[sel], b[sel], (da[sel] + db[sel] + 1.0)
        lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
        key = lo.astype(np.int64) * (k + 1) + hi
        o = np.argsort(key, kind="stable")
        ks, ds = key[o], d[o]
        first = np.r_[True, ks[1:] != ks[:-1]]
        starts = np.flatnonzero(first)
        mins = np.minimum.reduceat(ds, starts)
        for kk, dd in zip(ks[starts], mins):
            if kk in pairs:
                pairs[kk] = min(pairs[kk], dd)
            else:
                pairs[kk] = dd
    ei = np.array([(int(kk // (k + 1)) - 1, int(kk % (k + 1)) - 1, (dd - 1.0) * res) for kk, dd in pairs.items()]) if pairs else np.zeros((0, 3))
    return dict(labels=lab, area=area, centroid=cen, touches_edge=touches, edges=ei, res=res, n=k)


def clusters(g, gap):
    """Component label per node when links up to `gap` m are kept, and the area of every component."""
    e = g["edges"]
    e = e[e[:, 2] <= gap] if len(e) else e
    n = g["n"]
    A = coo_matrix((np.ones(len(e)), (e[:, 0].astype(int), e[:, 1].astype(int))), shape=(n, n)) if len(e) else coo_matrix((n, n))
    nc, lab = connected_components(A, directed=False)
    carea = np.bincount(lab, weights=g["area"], minlength=nc)
    return lab, carea


def pc_matrix(g, alpha, cutoff_mult=6.0):
    """Probability matrix p_ij = exp(-shortest-path-gap / alpha) between all nodes (0 beyond `cutoff_mult` x alpha)."""
    e = g["edges"]
    n = g["n"]
    if not len(e):
        return np.eye(n)
    W = coo_matrix((np.r_[e[:, 2], e[:, 2]] + 1e-6, (np.r_[e[:, 0], e[:, 1]].astype(int), np.r_[e[:, 1], e[:, 0]].astype(int))), shape=(n, n)).tocsr()
    D = dijkstra(W, directed=False, limit=cutoff_mult * alpha)
    P = np.exp(-np.where(np.isfinite(D), D, np.inf) / alpha)
    np.fill_diagonal(P, 1.0)
    return P


def pc_metrics(g, alpha, landscape_m2):
    """PC (0 to 1) and ECA (m2) for dispersal scale `alpha` m, normalised by the whole window area `landscape_m2`, plus the reachable canopy area of every node."""
    P = pc_matrix(g, alpha)
    a = g["area"]
    reach = P @ a
    eca2 = float(a @ reach)
    return dict(pc=eca2 / landscape_m2 ** 2, eca_m2=float(np.sqrt(eca2)), reach_m2=reach)


def stepping_stones(g, gap):
    """Nodes whose removal splits their cluster at gap-crossing distance `gap` (articulation points of the thresholded graph) as a boolean array."""
    import networkx as nx
    e = g["edges"]
    e = e[e[:, 2] <= gap] if len(e) else e
    G = nx.Graph()
    G.add_nodes_from(range(g["n"]))
    G.add_edges_from((int(a), int(b)) for a, b, _ in e)
    out = np.zeros(g["n"], bool)
    for v in nx.articulation_points(G):
        out[v] = True
    return out


def summary(g, landscape_m2, gaps=(2, 5, 10, 25, 50), alphas=(10.0, 30.0)):
    tot = float(g["area"].sum())
    st = dict(nodes=int(g["n"]), links=int(len(g["edges"])), canopy_in_nodes_m2=tot, edge_touching_nodes=int(g["touches_edge"].sum()), by_gap={}, pc={})
    for T in gaps:
        lab, ca = clusters(g, T)
        st["by_gap"][str(T)] = dict(clusters=int(len(ca)), largest_cluster_share=float(ca.max() / tot) if tot else 0.0, largest_cluster_ha=float(ca.max() / 1e4), isolated_nodes=int((np.bincount(lab) == 1).sum()))
    for al in alphas:
        r = pc_metrics(g, al, landscape_m2)
        st["pc"][str(al)] = dict(pc=r["pc"], eca_ha=r["eca_m2"] / 1e4, canopy_ha=tot / 1e4)
    return st
