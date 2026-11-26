"""The velocity kernel combined with the connectivity kernel, as CellRank recommends.

02c_cellrank.py scores CellRank 2's velocity kernel alone. The CellRank
tutorials combine it with the connectivity kernel, 0.8 x velocity + 0.2 x
connectivity, to make the chain robust to noise in the velocity vectors. This
computes that combination on the pipeline object of graph mode "all" (rebuilt
by 02_features.py into a separate directory, since the original object was not
kept), with the same terminal states and absorption solver as 02c, and checks
that the rebuilt dynamical velocities of the day-2 cells agree with the ones
the analysis used. Writes cr_vkck.npy (ten columns) and nm_cr_vkck.npy (the
neutrophil share) into the analysis feature directory, and cr_velo_rebuilt.npy
(the velocity kernel alone on the rebuilt object) for comparison.

Usage: 70_combined_kernel.py <h5ad> <featdir> [--vk-weight 0.8]
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import numpy.testing as npt
import anndata as ad
import scvelo as scv

_orig = npt.assert_array_equal


def _compat(*x, **k):
    if "x" in k:
        k["actual"] = k.pop("x")
    if "y" in k:
        k["desired"] = k.pop("y")
    return _orig(*x, **k)


npt.assert_array_equal = _compat
import cellrank as cr  # noqa: E402


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("rebuilt")
ap.add_argument("featdir")
ap.add_argument("--vk-weight", type=float, default=0.8)
a = ap.parse_args()

if not hasattr(sp.csr_array, "getnnz"):
    sp.csr_array.getnnz = sp.csr_array._getnnz
if not hasattr(sp.csc_array, "getnnz"):
    sp.csc_array.getnnz = sp.csc_array._getnnz
import scanpy as sc  # noqa: E402
from sklearn.neighbors import KNeighborsTransformer  # noqa: E402

# The pipeline of 02_features.py, mode "all", up to the dynamical velocities.
# The original object was not kept; 02_features.py no longer runs unchanged in
# the rebuilt environment (scanpy's approximate neighbour backend), so the
# graph is built with the exact search used by 29_expression_kernel.py, and the
# rebuilt velocities are checked against the originals below.
B = ad.read_h5ad(a.rebuilt)
hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
scv.pp.filter_genes(B, min_shared_counts=20)
scv.pp.normalize_per_cell(B)
sc.pp.log1p(B)
B = B[:, hvg].copy()
sc.tl.pca(B, n_comps=30, svd_solver="arpack")
sc.pp.neighbors(B, n_neighbors=30, n_pcs=30, use_rep="X_pca",
                transformer=KNeighborsTransformer(
                    n_neighbors=30, metric="euclidean", algorithm="brute",
                    n_jobs=8))
scv.pp.moments(B, n_pcs=30, n_neighbors=30)
say("moments done %s" % (B.shape,))
scv.tl.recover_dynamics(B, n_jobs=16, var_names="all")
scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
say("dynamical velocity done")
d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
d2 = np.array([B.obs_names.get_loc(n) for n in d2_names])
present = list(json.load(open(os.path.join(a.featdir, "cellrank_log.json")))
               ["terminal_states"].keys())
log = {"terminal_states": present, "vk_weight": a.vk_weight}

# the rebuilt velocities against the ones the analysis used
v_old = np.load(os.path.join(a.featdir, "vel_dyn.npy"))
v_new = np.asarray(B.layers["vel_dyn"])[d2]
ok = np.isfinite(v_old) & np.isfinite(v_new)
log["vel_dyn_rebuilt_corr"] = float(np.corrcoef(v_old[ok], v_new[ok])[0, 1])
log["vel_dyn_rebuilt_maxabsdiff"] = float(np.abs(v_old[ok] - v_new[ok]).max())
say("rebuilt vs original dynamical velocity (day-2 cells): r = %.6f, max |diff| %.3g"
    % (log["vel_dyn_rebuilt_corr"], log["vel_dyn_rebuilt_maxabsdiff"]))

scv.tl.velocity_graph(B, vkey="vel_dyn", n_jobs=8)
state = B.obs["state_info"].values
masks = [np.asarray(state == s) for s in present]
is_term = np.any(masks, axis=0)
trans = ~is_term


def fate(kernel, tag):
    T = sp.csr_matrix(kernel.transition_matrix)
    Q = T[trans][:, trans].tocsr()
    R = np.column_stack([np.asarray(T[trans][:, m].sum(1)).ravel() for m in masks])
    X = np.zeros_like(R)
    it = 0
    while it < 200000:
        X = Q @ X + R
        it += 1
        if it % 50 == 0 and float(np.abs(1.0 - X.sum(1)).max()) < 1e-8:
            break
    gap = float(np.abs(1.0 - X.sum(1)).max())
    X = X / X.sum(1, keepdims=True)
    F = np.zeros((B.n_obs, len(present)), dtype=np.float32)
    F[trans] = X
    for j, m in enumerate(masks):
        F[m, j] = 1.0
    P = F[d2]
    np.save(os.path.join(a.featdir, "cr_%s.npy" % tag), P)
    s = P[:, 0] + P[:, 1]
    sh = np.where(s > 0, P[:, 0] / np.where(s > 0, s, 1), 0.5)
    np.save(os.path.join(a.featdir, "nm_cr_%s.npy" % tag), sh.astype(np.float32))
    log[tag] = dict(iterations=it, max_rowsum_gap=gap,
                    frac_share_093_100=float(np.mean((sh >= 0.93) & (sh <= 1.0))))
    say("%s: %d iterations, gap %.3g, share in [0.93,1]: %.3f"
        % (tag, it, gap, log[tag]["frac_share_093_100"]))


vk = cr.kernels.VelocityKernel(B, vkey="vel_dyn")
vk.compute_transition_matrix()
fate(vk, "velo_rebuilt")
ck = cr.kernels.ConnectivityKernel(B)
ck.compute_transition_matrix()
comb = a.vk_weight * vk + (1 - a.vk_weight) * ck
comb.compute_transition_matrix()
fate(comb, "vkck")
json.dump(log, open(os.path.join(a.featdir, "combined_kernel_log.json"), "w"),
          indent=1, default=str)
say("done")
