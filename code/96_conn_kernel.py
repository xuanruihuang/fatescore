"""CellRank's connectivity kernel on a lineage data set without a splicing layer.

The same predictor as for LARRY (02c_cellrank.py, 66_unsupervised_reference.py):
absorption probabilities of a random walk on the expression neighbour graph into
the annotated mature cells, read off for the early cells. Graph settings as for
LARRY: 15, 30 or 60 neighbours x 15, 30 or 50 principal components x density
normalisation off or on (18 settings). A second mode builds the graph on the
early cells alone, with the mature-annotated early cells as terminal states, to
show what the later cells contribute. Mode "pseudotime" adds CellRank's pseudotime kernel
(written as cr_pseudo.npy), built as on LARRY.

Writes <featdir>/cr_conn_<mode>_n<k>_p<p>_d<0|1>.npy (one row per early cell, one
column per terminal population in the order of cellrank_log.json) and a
matching _log.json with the populations.

Usage: 96_conn_kernel.py <h5ad> <featdir> [--modes all,early]
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

# the same compatibility shim as 85_solver_check.py: this scanpy calls getnnz,
# which newer scipy sparse arrays expose only as _getnnz
for _cls in (sp.csr_array, sp.csc_array):
    if not hasattr(_cls, "getnnz"):
        _cls.getnnz = _cls._getnnz
import anndata as ad  # noqa: E402
import scanpy as sc  # noqa: E402
import cellrank as cr  # noqa: E402
from sklearn.neighbors import KNeighborsTransformer  # noqa: E402


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("--modes", default="all,early")
a = ap.parse_args()

hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
early = json.load(open(os.path.join(a.featdir, "d2_names.json")))
pops = list(json.load(open(os.path.join(a.featdir, "cellrank_log.json")))["terminal_states"].keys())
A = ad.read_h5ad(a.h5ad)
sc.pp.normalize_total(A)
sc.pp.log1p(A)
A = A[:, [g for g in hvg if g in A.var_names]].copy()


def absorb(T, masks, trans, tol=1e-8, max_iter=200000):
    T = sp.csr_matrix(T)
    Q = T[trans][:, trans].tocsr()
    R = np.column_stack([np.asarray(T[trans][:, m].sum(1)).ravel() for m in masks])
    X = np.zeros_like(R)
    it = 0
    while it < max_iter:
        X = Q @ X + R
        it += 1
        if it % 50 == 0 and np.abs(1.0 - X.sum(1)).max() < tol:
            break
    rs = X.sum(1, keepdims=True)
    return X / np.where(rs == 0, 1, rs), it


def pseudotime_kernel(B, masks, trans):
    """CellRank's pseudotime kernel as on LARRY (02c_cellrank.py): diffusion pseudotime
    on the n30 p30 graph, rooted at the earliest-day non-terminal cell furthest in
    diffusion space from the terminal cells; reversed if the terminal cells come out
    earlier than the rest."""
    sc.pp.neighbors(B, n_neighbors=30, n_pcs=30, use_rep="X_pca",
                    transformer=KNeighborsTransformer(n_neighbors=30, metric="euclidean",
                                                      algorithm="brute", n_jobs=8))
    sc.tl.diffmap(B)
    dm = B.obsm["X_diffmap"][:, 1:11]
    t = B.obs["day"].astype(float).values
    is_term = ~trans
    cand = trans & (t == t.min())
    dist = np.linalg.norm(dm - dm[is_term].mean(0), axis=1)
    B.uns["iroot"] = int(np.where(cand)[0][np.argmax(dist[cand])])
    sc.tl.dpt(B)
    pt = B.obs["dpt_pseudotime"].values
    if np.nanmean(pt[is_term]) < np.nanmean(pt[trans]):
        B.obs["dpt_pseudotime"] = np.nanmax(pt) - pt
    pt = B.obs["dpt_pseudotime"].values
    say("pseudotime: terminal minus other %.3f; r with day %.3f"
        % (np.nanmean(pt[is_term]) - np.nanmean(pt[trans]), np.corrcoef(pt, t)[0, 1]))
    pk = cr.kernels.PseudotimeKernel(B, time_key="dpt_pseudotime")
    pk.compute_transition_matrix()
    X, it = absorb(pk.transition_matrix, masks, trans)
    return X, it


for mode in a.modes.split(","):
    B = A if mode in ("all", "pseudotime") else A[A.obs_names.isin(early)].copy()
    sc.tl.pca(B, n_comps=50, svd_solver="arpack")
    state = B.obs["state_info"].astype(str).values
    masks = [state == p for p in pops]
    keep_pops = [p for p, m in zip(pops, masks) if m.sum() > 0]
    masks = [m for m in masks if m.sum() > 0]
    if not masks:
        say("%s: no cell carries a terminal annotation; mode skipped" % mode)
        continue
    is_term = np.any(masks, axis=0)
    trans = ~is_term
    idx = np.array([B.obs_names.get_loc(n) for n in early])
    say("%s: %d cells, terminal %s" % (mode, B.n_obs, {p: int(m.sum()) for p, m in zip(keep_pops, masks)}))
    if mode == "pseudotime":
        X, it = pseudotime_kernel(B, masks, trans)
        F = np.zeros((B.n_obs, len(pops)), dtype=np.float32)
        full = np.zeros((B.n_obs, len(keep_pops)))
        full[trans] = X
        for j, m in enumerate(masks):
            full[m, j] = 1.0
        for j, pname in enumerate(keep_pops):
            F[:, pops.index(pname)] = full[:, j]
        np.save(os.path.join(a.featdir, "cr_pseudo.npy"), F[idx])
        json.dump(dict(populations=pops), open(os.path.join(a.featdir, "cr_pseudo_log.json"), "w"))
        say("  cr_pseudo: %d iterations" % it)
        continue
    for k in (15, 30, 60):
        for p in (15, 30, 50):
            # exact neighbours, as in 85/92 (scanpy's default approximate search fails in
            # numba with these package versions)
            sc.pp.neighbors(B, n_neighbors=k, n_pcs=p, use_rep="X_pca",
                            transformer=KNeighborsTransformer(n_neighbors=k, metric="euclidean",
                                                              algorithm="brute", n_jobs=8))
            for d in (0, 1):
                ck = cr.kernels.ConnectivityKernel(B)
                ck.compute_transition_matrix(density_normalize=bool(d))
                X, it = absorb(ck.transition_matrix, masks, trans)
                F = np.zeros((B.n_obs, len(pops)), dtype=np.float32)
                full = np.zeros((B.n_obs, len(keep_pops)))
                full[trans] = X
                for j, m in enumerate(masks):
                    full[m, j] = 1.0
                for j, pname in enumerate(keep_pops):
                    F[:, pops.index(pname)] = full[:, j]
                name = "cr_conn_%s_n%d_p%d_d%d" % (mode, k, p, d)
                np.save(os.path.join(a.featdir, name + ".npy"), F[idx])
                json.dump(dict(populations=pops), open(os.path.join(a.featdir, name + "_log.json"), "w"))
                say("  %s: %d iterations" % (name, it))
say("done")
