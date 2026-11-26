"""Start from the walk and add a little direction (the paper's own advice, tried on LARRY).

The pipeline of graph mode "all" is rebuilt exactly as in 92_ladder_kernel.py (same genes, exact
30-neighbour graph on 30 PCs, moments, dynamical velocity on all genes), with the same annotated
terminal states and the same Neumann-series absorption. On it:

  CK   connectivity kernel (no direction)
  VK   velocity kernel, dynamical velocity
  PK   pseudotime kernel, diffusion pseudotime rooted as in 02c_cellrank.py
  VKd  velocity kernel after removing, in each cell's velocity, its component along the
       neutrophil-minus-monocyte axis (difference of the two terminal populations' mean smoothed
       spliced expression over the velocity genes): a direction with the between-fate bias taken out
  VKp  velocity kernel restricted to progression: transitions to neighbours with higher diffusion
       pseudotime only, rows renormalised (a row left empty falls back to the CK row)

Mixtures K = (1 - w) CK + w D for D in {VK, PK, VKd, VKp} and w in {0.02, 0.05, 0.1, 0.2, 0.3, 0.5},
plus each D alone (w = 1) and CK (w = 0). Each kernel's absorption probabilities give the day-2
cells' neutrophil share as in 92; written as <outdir>/nm_mix_<D>_w<w>.npy (d2_names order).
The grid and both debiased variants were fixed before any of these was scored.

Usage: 111_mix_velocity.py <raw h5ad> <featdir> <outdir>
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
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
os.makedirs(OUT, exist_ok=True)
W = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


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
say("pipeline rebuilt %s" % (B.shape,))

d2 = np.array([B.obs_names.get_loc(n) for n in json.load(open(os.path.join(FD, "d2_names.json")))])
present = list(json.load(open(os.path.join(FD, "cellrank_log.json")))["terminal_states"].keys())
state = B.obs["state_info"].values
masks = [np.asarray(state == s) for s in present]
is_term = np.any(masks, axis=0)
trans = ~is_term
ineu, imono = present.index("Neutrophil"), present.index("Monocyte")
log = dict(populations=present, w=W)


def share(T, tag, tol=1e-8, max_iter=200000):
    """Absorption probabilities as in 92_ladder_kernel.py, then the neutrophil share."""
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
    gap = float(np.abs(1.0 - X.sum(1)).max())
    rs = X.sum(1, keepdims=True)
    X = X / np.where(rs == 0, 1, rs)
    F = np.zeros((B.n_obs, len(present)))
    F[trans] = X
    for j, m in enumerate(masks):
        F[m, j] = 1.0
    F = F[d2]
    s = F[:, ineu] + F[:, imono]
    p = np.where(s > 0, F[:, ineu] / np.where(s > 0, s, 1.0), 0.5)
    np.save(os.path.join(OUT, "nm_mix_%s.npy" % tag), p[:, None].astype(np.float32))
    log[tag] = dict(iterations=it, rowsum_gap=gap, mean_share=float(p.mean()))
    say("  %s: %d iterations (gap %.2g), mean share %.3f" % (tag, it, gap, p.mean()))


def rownorm(T):
    T = sp.csr_matrix(T)
    rs = np.asarray(T.sum(1)).ravel()
    return sp.diags(1.0 / np.where(rs > 0, rs, 1.0)) @ T


ck = cr.kernels.ConnectivityKernel(B)
ck.compute_transition_matrix()
CK = sp.csr_matrix(ck.transition_matrix)
share(CK, "CK_w0")

vk = cr.kernels.VelocityKernel(B, vkey="vel_dyn")
vk.compute_transition_matrix()
VK = sp.csr_matrix(vk.transition_matrix)

# diffusion pseudotime, rooted as in 02c_cellrank.py
tvals = B.obs["time_info"].astype(float).values
sc.tl.diffmap(B)
dm = B.obsm["X_diffmap"][:, 1:11]
cand = (B.obs["state_info"] == "Undifferentiated").values
earliest = cand & (tvals == tvals.min())
if earliest.sum() > 0:
    cand = earliest
centre = dm[is_term].mean(0)
dist = np.linalg.norm(dm - centre, axis=1)
B.uns["iroot"] = int(np.where(cand)[0][np.argmax(dist[cand])])
sc.tl.dpt(B)
pt = B.obs["dpt_pseudotime"].values
if np.nanmean(pt[is_term]) < np.nanmean(pt[~is_term]):
    pt = np.nanmax(pt) - pt
    B.obs["dpt_pseudotime"] = pt
log["dpt_time_corr"] = float(np.corrcoef(np.nan_to_num(pt), tvals)[0, 1])
pk = cr.kernels.PseudotimeKernel(B, time_key="dpt_pseudotime")
pk.compute_transition_matrix()
PK = sp.csr_matrix(pk.transition_matrix)

# VKd: the between-fate component of each cell's velocity removed
vg = np.asarray(B.var["vel_dyn_genes"]).astype(bool)
Ms = np.asarray(B.layers["Ms"])
V = np.nan_to_num(np.asarray(B.layers["vel_dyn"], dtype=np.float64))
a = Ms[masks[ineu]][:, vg].mean(0) - Ms[masks[imono]][:, vg].mean(0)
a = a / np.linalg.norm(a)
Vd = V.copy()
proj = V[:, vg] @ a
Vd[:, vg] = V[:, vg] - proj[:, None] * a[None, :]
log["vkd_removed_share_of_norm"] = float(np.mean(np.abs(proj) / (np.linalg.norm(V[:, vg], axis=1) + 1e-12)))
B.layers["vel_deb"] = np.where(vg, Vd, np.nan)
B.var["vel_deb_genes"] = vg
vkd = cr.kernels.VelocityKernel(B, vkey="vel_deb")
vkd.compute_transition_matrix()
VKd = sp.csr_matrix(vkd.transition_matrix)

# VKp: progression only (higher pseudotime), rows renormalised; empty rows fall back to CK
Vc = VK.tocoo()
keep = np.nan_to_num(pt)[Vc.col] > np.nan_to_num(pt)[Vc.row]
P = sp.csr_matrix((Vc.data[keep], (Vc.row[keep], Vc.col[keep])), shape=VK.shape)
rs = np.asarray(P.sum(1)).ravel()
empty = rs == 0
log["vkp_empty_rows"] = int(empty.sum())
P = rownorm(P)
VKp = (P + sp.diags(empty.astype(float)) @ CK).tocsr()
say("kernels built; VKp empty rows %d; mean |between-fate component| / |v| %.3f"
    % (empty.sum(), log["vkd_removed_share_of_norm"]))

for name, D in (("VK", VK), ("PK", PK), ("VKd", VKd), ("VKp", VKp)):
    share(D, "%s_w1" % name)
    for w in W:
        share((1.0 - w) * CK + w * D, "%s_w%s" % (name, w))
json.dump(log, open(os.path.join(OUT, "mix_velocity_log.json"), "w"), indent=1)
say("done")
