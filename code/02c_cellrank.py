"""Fate probabilities for the day-2 cells from CellRank 2's two kernels.

Usage: 02c_cellrank.py <featdir>

Both kernels are given the same terminal states, the annotated mature
populations present in the object, so the only difference between them is the
transition matrix: cosine similarities of the dynamical velocities for the
velocity kernel, a diffusion pseudotime for the pseudotime kernel. Under graph
mode "all" the terminal states are day-4/6 cells; under mode "d2" they are the
mature cells already present at day 2.

The absorption probabilities are solved here rather than through
GPCCA.compute_fate_probabilities, which refuses a solution when the linear
solver leaves a handful of values slightly below zero; it did so on this graph
with the iterative solver (12 of 237,700 values) and with the direct one (66).
The quantity is the standard one for an absorbing Markov chain,
X = (I - Q)^-1 R, with Q the transient-to-transient block of the kernel's own
transition matrix and R the transient-to-terminal block aggregated by terminal
state. It is accumulated as the Neumann series, X <- Q X + R from X = 0, which
is monotone and non-negative term by term, so no clipping is needed; the
iteration runs until the row sums are within 1e-8 of one, and the achieved
accuracy is recorded.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy.testing as npt
import scanpy as sc
import scvelo as scv
import anndata as ad

# cellrank 2.0.7 calls np.testing.assert_array_equal(x=..., y=...); numpy 2
# renamed those parameters to actual/desired.
_orig = npt.assert_array_equal


def _compat(*a, **k):
    if "x" in k:
        k["actual"] = k.pop("x")
    if "y" in k:
        k["desired"] = k.pop("y")
    return _orig(*a, **k)


npt.assert_array_equal = _compat
import cellrank as cr  # noqa: E402

FD = sys.argv[1]
# optional, for a second dataset: the candidate terminal populations, in the
# order their columns should appear. The default is the LARRY annotation.
MATURE = (sys.argv[2].split(",") if len(sys.argv) > 2 else
          ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
           "Lymphoid", "Eos", "Ccr7_DC", "pDC"])
# the label of the undifferentiated compartment, used only to root the
# pseudotime
ROOT_STATE = sys.argv[3] if len(sys.argv) > 3 else "Undifferentiated"
log = {}


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


B = ad.read_h5ad(os.path.join(FD, "pipeline.h5ad"))
d2_names = json.load(open(os.path.join(FD, "d2_names.json")))
d2_idx = np.array([B.obs_names.get_loc(n) for n in d2_names])
# the sampling day survives the h5ad round trip as an unordered categorical
tvals = B.obs["time"].astype(int).values
say("loaded %s" % (B.shape,))

present = [s for s in MATURE if (B.obs["state_info"] == s).sum() >= 30]
log["terminal_states"] = {s: int((B.obs["state_info"] == s).sum())
                          for s in present}
say("terminal states: %s" % log["terminal_states"])

scv.tl.velocity_graph(B, vkey="vel_dyn", n_jobs=16)
say("velocity graph done")

state = B.obs["state_info"].values
term_masks = [np.asarray(state == s) for s in present]
is_term = np.zeros(B.n_obs, bool)
for m in term_masks:
    is_term |= m
trans = ~is_term
log["n_transient"] = int(trans.sum())
log["n_terminal"] = int(is_term.sum())
say("transient %d, terminal %d" % (trans.sum(), is_term.sum()))


def fate(kernel, tag, tol=1e-8, max_iter=200000):
    T = sp.csr_matrix(kernel.transition_matrix)
    Q = T[trans][:, trans].tocsr()
    R = np.column_stack([np.asarray(T[trans][:, m].sum(1)).ravel()
                         for m in term_masks])
    X = np.zeros_like(R)
    it = 0
    while it < max_iter:
        X = Q @ X + R
        it += 1
        if it % 50 == 0:
            gap = float(np.abs(1.0 - X.sum(1)).max())
            if gap < tol:
                break
    log["%s_iterations" % tag] = it
    log["%s_max_rowsum_gap" % tag] = float(np.abs(1.0 - X.sum(1)).max())
    log["%s_min_value" % tag] = float(X.min())
    say("  %s: %d iterations, max row-sum gap %.3g, min value %.3g"
        % (tag, it, log["%s_max_rowsum_gap" % tag], log["%s_min_value" % tag]))
    rs = X.sum(1, keepdims=True)
    X = X / np.where(rs == 0, 1, rs)
    F = np.zeros((B.n_obs, len(present)), dtype=np.float32)
    F[trans] = X
    for j, m in enumerate(term_masks):
        F[m, j] = 1.0
    np.save(os.path.join(FD, "cr_%s.npy" % tag), F[d2_idx])
    log["cr_%s_states" % tag] = present
    say("saved cr_%s %s" % (tag, F[d2_idx].shape))


vk = cr.kernels.VelocityKernel(B, vkey="vel_dyn")
vk.compute_transition_matrix()
fate(vk, "velo")

sc.pp.neighbors(B, n_neighbors=30, n_pcs=30)
sc.tl.diffmap(B)
# Root the diffusion pseudotime at the undifferentiated cell furthest, in
# diffusion space, from the terminal populations. Under graph mode "all" the
# candidates are restricted to the earliest sampling day. This uses the same
# state annotation the terminal states come from and no clonal label.
dm = B.obsm["X_diffmap"][:, 1:11]
cand = (B.obs["state_info"] == ROOT_STATE).values
if cand.sum() == 0:
    cand = ~is_term
if len(np.unique(tvals)) > 1:
    earliest = cand & (tvals == tvals.min())
    if earliest.sum() > 0:
        cand = earliest
if cand.sum() == 0:
    cand = ~is_term
log["root_candidates"] = int(cand.sum())
centre = dm[is_term].mean(0)
dist = np.linalg.norm(dm - centre, axis=1)
B.uns["iroot"] = int(np.where(cand)[0][np.argmax(dist[cand])])
sc.tl.dpt(B)
pt = B.obs["dpt_pseudotime"].values
if np.nanmean(pt[is_term]) < np.nanmean(pt[~is_term]):
    pt = np.nanmax(pt) - pt
    B.obs["dpt_pseudotime"] = pt
log["dpt_mature_minus_other"] = float(
    np.nanmean(pt[is_term]) - np.nanmean(pt[~is_term]))
log["dpt_time_corr"] = (
    float(np.corrcoef(pt, tvals.astype(float))[0, 1])
    if len(np.unique(tvals)) > 1 else None)
say("pseudotime: mature minus other = %.3f, vs sampling day r = %s"
    % (log["dpt_mature_minus_other"], log["dpt_time_corr"]))
pk = cr.kernels.PseudotimeKernel(B, time_key="dpt_pseudotime")
pk.compute_transition_matrix()
fate(pk, "pseudo")

json.dump(log, open(os.path.join(FD, "cellrank_log.json"), "w"), indent=1,
          default=str)
say("done")
