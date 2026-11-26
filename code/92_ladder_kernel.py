"""The deconstruction ladder on the clonal-fate validation.

CellRank's velocity kernel is rebuilt exactly as in 85_solver_check.py (the
pipeline of 70_combined_kernel.py, graph mode "all", same terminal states), and
its velocity field is replaced, one component at a time, by the fields of the
direction analysis: shuffled unspliced counts (each gene's gamma kept, the cell's
own smoothed unspliced counts replaced by another cell's), degradation only
(-gamma * s), and finally no field at all (connectivity kernel). Each kernel's
absorption probabilities are turned into the day-2 cells' neutrophil share as in
02c_cellrank.py / 51_fate_share.py and written as nm_cr_lad_<field>.npy, to be
scored by 66_unsupervised_reference.py like every other predictor.

Usage: 92_ladder_kernel.py <raw h5ad> <featdir> <outdir>
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
scv.tl.velocity(B, mode="deterministic", vkey="vel_det")
say("pipeline rebuilt %s" % (B.shape,))

# the fields of the ladder, as in 71_direction_embedding.py: each gene's gamma from
# the steady-state fit; the cell's own unspliced counts replaced by another cell's;
# the unspliced counts dropped
Ms, Mu = np.asarray(B.layers["Ms"]), np.asarray(B.layers["Mu"])
gam = np.nan_to_num(np.asarray(B.var["vel_det_gamma"], dtype=np.float64))
vg = np.asarray(B.var["vel_det_genes"]).astype(bool)
perm = np.random.default_rng(31).permutation(B.n_obs)
B.layers["lad_shuffled"] = np.where(vg, Mu[perm] - gam * Ms, 0.0)
B.layers["lad_degradation"] = np.where(vg, -gam * Ms, 0.0)
for k in ("lad_shuffled", "lad_degradation"):
    B.var[k + "_genes"] = vg

d2 = np.array([B.obs_names.get_loc(n) for n in json.load(open(os.path.join(FD, "d2_names.json")))])
present = list(json.load(open(os.path.join(FD, "cellrank_log.json")))["terminal_states"].keys())
state = B.obs["state_info"].values
masks = [np.asarray(state == s) for s in present]
is_term = np.any(masks, axis=0)
trans = ~is_term
ineu, imono = present.index("Neutrophil"), present.index("Monocyte")


def share(T, tag, tol=1e-8, max_iter=200000):
    """Absorption probabilities as in 02c_cellrank.py, then the neutrophil share."""
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
    X = X / np.where(rs == 0, 1, rs)
    F = np.zeros((B.n_obs, len(present)))
    F[trans] = X
    for j, m in enumerate(masks):
        F[m, j] = 1.0
    F = F[d2]
    s = F[:, ineu] + F[:, imono]
    p = np.where(s > 0, F[:, ineu] / np.where(s > 0, s, 1.0), 0.5)
    np.save(os.path.join(OUT, "nm_cr_lad_%s.npy" % tag), p[:, None].astype(np.float32))
    say("  %s: %d iterations, mean share %.3f, share in [0.93,1] %.3f"
        % (tag, it, p.mean(), np.mean(p >= 0.93)))


for tag, vkey in (("velocity_dyn", "vel_dyn"), ("velocity_ss", "vel_det"),
                  ("shuffled", "lad_shuffled"), ("degradation", "lad_degradation")):
    vk = cr.kernels.VelocityKernel(B, vkey=vkey)
    vk.compute_transition_matrix()
    share(vk.transition_matrix, tag)
ck = cr.kernels.ConnectivityKernel(B)
ck.compute_transition_matrix()
share(ck.transition_matrix, "connectivity")
say("done")
