"""Checks on the absorption probabilities of CellRank's velocity kernel (referee).

1. Solver. 02c_cellrank.py and 29_expression_kernel.py accumulate the
   absorption probabilities X = (I - Q)^-1 R as a Neumann series because
   CellRank's GPCCA.compute_fate_probabilities refuses solutions with small
   negative values. Here the same system is also solved directly (sparse LU,
   scipy.sparse.linalg.splu), for the velocity kernel and the connectivity
   kernel, and the two solutions are compared: maximum absolute difference, the
   number and size of negative entries in the direct solution, and the change in
   the day-2 cells' neutrophil fate bias.
2. Mass. The fate bias is Neu / (Neu + Mono). Over all day-2 cells, the
   distribution of Neu + Mono, so that a ratio of two near-zero numbers would be
   visible, and the share of cells with bias in [0.93, 1.00] among cells with
   Neu + Mono >= 0.5.
The pipeline object of graph mode "all" was not kept, so it is rebuilt exactly
as in 70_combined_kernel.py (same genes, exact neighbours, dynamical velocity).

Usage: 85_solver_check.py <raw h5ad> <featdir> <out.csv>
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy.testing as npt
import anndata as ad

_orig = npt.assert_array_equal


def _compat(*x, **k):
    if "x" in k:
        k["actual"] = k.pop("x")
    if "y" in k:
        k["desired"] = k.pop("y")
    return _orig(*x, **k)


npt.assert_array_equal = _compat
for cls in (sp.csr_array, sp.csc_array):
    if not hasattr(cls, "getnnz"):
        cls.getnnz = cls._getnnz
import scvelo as scv  # noqa: E402
import scanpy as sc  # noqa: E402
import cellrank as cr  # noqa: E402
from sklearn.neighbors import KNeighborsTransformer  # noqa: E402

RAW, FD, OUT = sys.argv[1:4]


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


B = ad.read_h5ad(RAW)
hvg = json.load(open(os.path.join(FD, "hvg.json")))
scv.pp.filter_genes(B, min_shared_counts=20)
scv.pp.normalize_per_cell(B)
sc.pp.log1p(B)
B = B[:, hvg].copy()
sc.tl.pca(B, n_comps=30, svd_solver="arpack")
sc.pp.neighbors(B, n_neighbors=30, n_pcs=30, use_rep="X_pca",
                transformer=KNeighborsTransformer(n_neighbors=30, metric="euclidean",
                                                  algorithm="brute", n_jobs=8))
scv.pp.moments(B, n_pcs=30, n_neighbors=30)
scv.tl.recover_dynamics(B, n_jobs=16, var_names="all")
scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
scv.tl.velocity_graph(B, vkey="vel_dyn", n_jobs=8, show_progress_bar=False)
say("pipeline rebuilt %s" % (B.shape,))

d2 = np.array([B.obs_names.get_loc(n) for n in json.load(open(os.path.join(FD, "d2_names.json")))])
present = list(json.load(open(os.path.join(FD, "cellrank_log.json")))["terminal_states"].keys())
state = B.obs["state_info"].values
masks = [np.asarray(state == s) for s in present]
is_term = np.any(masks, axis=0)
trans = ~is_term
pos = np.full(B.n_obs, -1)
pos[trans] = np.arange(trans.sum())


def solve(T):
    T = sp.csr_matrix(T)
    Q = T[trans][:, trans].tocsc()
    R = np.column_stack([np.asarray(T[trans][:, m].sum(1)).ravel() for m in masks])
    X = np.zeros_like(R)
    it = 0
    while it < 200000:
        X = Q @ X + R
        it += 1
        if it % 50 == 0 and np.abs(1.0 - X.sum(1)).max() < 1e-8:
            break
    t0 = time.time()
    Xd = spla.splu((sp.identity(Q.shape[0], format="csc") - Q).tocsc()).solve(R)
    say("  neumann %d iterations; direct LU %.0f s" % (it, time.time() - t0))
    return X, Xd


rows = []


def report(tag, X, Xd):
    ii = pos[d2]
    ok = ii >= 0
    nm, nmd = X[ii[ok]][:, :2], Xd[ii[ok]][:, :2]
    mass = nm.sum(1)
    fb = nm[:, 0] / np.where(mass > 0, mass, np.nan)
    sd = nmd.sum(1)
    fbd = nmd[:, 0] / np.where(sd != 0, sd, np.nan)
    big = mass >= 0.5
    r = dict(kernel=tag, max_abs_diff=float(np.abs(X - Xd).max()),
             n_negative_direct=int((Xd < 0).sum()), min_direct=float(Xd.min()),
             n_values=int(Xd.size), max_abs_diff_fate_bias=float(np.nanmax(np.abs(fb - fbd))),
             n_d2_transient=int(ok.sum()), n_d2_terminal=int((~ok).sum()),
             mass_nm_median=float(np.median(mass)), mass_nm_p05=float(np.percentile(mass, 5)),
             mass_nm_min=float(mass.min()), share_mass_ge_05=float(big.mean()),
             share_bias_093_100=float(np.mean(fb >= 0.93)),
             share_bias_093_100_mass_ge_05=float(np.mean(fb[big] >= 0.93)) if big.any() else np.nan,
             mean_mass_neu=float(nm[:, 0].mean()), mean_mass_mono=float(nm[:, 1].mean()))
    rows.append(r)
    say("  " + json.dumps(r))


vk = cr.kernels.VelocityKernel(B, vkey="vel_dyn")
vk.compute_transition_matrix()
report("velocity", *solve(vk.transition_matrix))
ck = cr.kernels.ConnectivityKernel(B)
ck.compute_transition_matrix()
report("connectivity", *solve(ck.transition_matrix))
pd.DataFrame(rows).to_csv(OUT, index=False)
say("done")
