"""Leakage check for the connectivity kernel, the reference without dynamics.

The connectivity kernel is computed on the neighbour graph of all cells, whose
terminal states are the annotated mature cells of days 4 and 6. Those include
the held-out clones' own descendants, the cells that define the clones' labels.
The kernel never sees barcodes, but a day-2 cell's absorption mass could reach
its own sisters, which resemble it. StationaryOT was checked for this in
59_leakage_cells.py / 41_ot_fate.py --drop-cells; this applies the same check to
the reference.

Pipeline as 29_expression_kernel.py (same gene filter, normalisation, the
feature directory's highly variable genes, 50 principal components on all
cells, exact neighbour search, the same terminal states and absorption solver).
For each fold of split 0, the barcoded day-4 and day-6 cells of that fold's test
clones (the lists written by 59_leakage_cells.py) are removed after the PCA, so
the representation is unchanged; the graph, kernel and absorption probabilities
are recomputed; and the test clones of that fold are scored with that fold's
run. The same pipeline without removal is run alongside, so the comparison is
within one implementation.

Scores (shares used as they are, no classifier; paired bootstrap over clones,
2,000 resamples): clone-level AUROC on the main task, and cell-level Pearson r
with the observed neutrophil share on the continuous-task cells whose clone is
in the main task (fold membership is defined on the main task's clones).

Usage: 68_kernel_leakage.py <h5ad> <featdir> <labels.tsv> <drop_prefix> <out_prefix>
         [--settings 30,30,1;30,15,1] [--jobs 8]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import numpy.testing as npt
import anndata as ad
import scanpy as sc
import scvelo as scv
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsTransformer

_orig = npt.assert_array_equal


def _compat(*x, **k):
    if "x" in k:
        k["actual"] = k.pop("x")
    if "y" in k:
        k["desired"] = k.pop("y")
    return _orig(*x, **k)


npt.assert_array_equal = _compat
if not hasattr(sp.csr_array, "getnnz"):
    sp.csr_array.getnnz = sp.csr_array._getnnz
if not hasattr(sp.csc_array, "getnnz"):
    sp.csc_array.getnnz = sp.csc_array._getnnz
import cellrank as cr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvlib import folds_of  # noqa: E402


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("drop_prefix")
ap.add_argument("out")
ap.add_argument("--settings", default="30,30,1;30,15,1")
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
present = list(json.load(open(os.path.join(a.featdir, "cellrank_log.json")))
               ["terminal_states"].keys())
B0 = ad.read_h5ad(a.h5ad)
scv.pp.filter_genes(B0, min_shared_counts=20)
scv.pp.normalize_per_cell(B0)
sc.pp.log1p(B0)
B0 = B0[:, hvg].copy()
sc.tl.pca(B0, n_comps=50, svd_solver="arpack")
B0 = ad.AnnData(obs=B0.obs[["state_info"]].copy(),
                obsm={"X_pca": B0.obsm["X_pca"]})
say("object %s; terminal states %s" % (B0.shape, present))


def kernel_share(keep, nn, npc, dn):
    B = B0[keep].copy()
    sc.pp.neighbors(B, n_neighbors=nn, n_pcs=npc, use_rep="X_pca",
                    transformer=KNeighborsTransformer(
                        n_neighbors=nn, metric="euclidean", algorithm="brute",
                        n_jobs=a.jobs))
    ck = cr.kernels.ConnectivityKernel(B)
    ck.compute_transition_matrix(density_normalize=dn)
    state = B.obs["state_info"].values
    masks = [np.asarray(state == s) for s in present]
    is_term = np.any(masks, axis=0)
    trans = ~is_term
    T = sp.csr_matrix(ck.transition_matrix)
    Q = T[trans][:, trans].tocsr()
    R = np.column_stack([np.asarray(T[trans][:, m].sum(1)).ravel() for m in masks])
    X = np.zeros_like(R)
    it = 0
    while it < 200000:
        X = Q @ X + R
        it += 1
        if it % 50 == 0 and float(np.abs(1.0 - X.sum(1)).max()) < 1e-8:
            break
    X = X / X.sum(1, keepdims=True)
    F = np.zeros((B.n_obs, len(present)))
    F[trans] = X
    pos = pd.Series(np.arange(B.n_obs), index=B.obs_names)
    P = F[pos[d2_names].to_numpy()]
    s = P[:, 0] + P[:, 1]
    return np.where(s > 0, P[:, 0] / np.where(s > 0, s, 1), 0.5), it


T = pd.read_csv(a.labels, sep="\t")
assert list(T["cell"]) == list(d2_names), "label rows must follow d2_names"
m = (T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])
groups = T.loc[m, "clone"].values
folds = list(folds_of(groups, 0, a.folds))
fold_of_clone = {}
for f, (tr, te) in enumerate(folds):
    for g in np.unique(groups[te]):
        fold_of_clone[g] = f
names = np.asarray(B0.obs_names)
rows = []
for st in a.settings.split(";"):
    nn, npc, dn = [int(x) for x in st.split(",")]
    tag = "n%d_p%d_d%d" % (nn, npc, dn)
    full, it = kernel_share(np.ones(B0.n_obs, bool), nn, npc, bool(dn))
    say("%s full: %d iterations" % (tag, it))
    lo = np.full(len(d2_names), np.nan)
    for f in range(a.folds):
        drop = set(open("%s_s0_f%d.txt" % (a.drop_prefix, f)).read().split())
        keep = ~np.isin(names, list(drop))
        sh, it = kernel_share(keep, nn, npc, bool(dn))
        sel = T["clone"].map(fold_of_clone).values == f
        lo[sel] = sh[sel]
        say("%s fold %d: dropped %d cells, %d iterations" % (tag, f, (~keep).sum(), it))
    np.save("%s_%s_full.npy" % (a.out, tag), full)
    np.save("%s_%s_lo.npy" % (a.out, tag), lo)
    # main task, clone-level AUROC
    D = pd.DataFrame(dict(full=full[m.values], lo=lo[m.values], g=groups,
                          y=(T.loc[m, "dom"].values == "Neutrophil").astype(int)))
    C = D.groupby("g").mean()
    y = C["y"].values.astype(int)
    rng = np.random.default_rng(68)
    Bm = rng.integers(0, len(C), size=(a.boot, len(C)))
    af, al = roc_auc_score(y, C["full"]), roc_auc_score(y, C["lo"])
    d = np.array([roc_auc_score(y[b], C["lo"].values[b]) -
                  roc_auc_score(y[b], C["full"].values[b]) for b in Bm])
    rec = dict(setting=tag, auroc_full=af, auroc_without=al, auroc_diff=al - af,
               auroc_lo=np.percentile(d, 2.5), auroc_hi=np.percentile(d, 97.5))
    # continuous task cells whose clone is in the main task
    c = (T["n_neu_mono"] >= 3) & T["f_neu"].notna() & T["clone"].isin(fold_of_clone)
    ic = np.where(c.values)[0]
    yc = T.loc[c, "f_neu"].values.astype(float)
    gc = T.loc[c, "clone"].values
    ug, inv = np.unique(gc, return_inverse=True)
    mem = [np.where(inv == k)[0] for k in range(len(ug))]
    Bc = rng.integers(0, len(ug), size=(a.boot, len(ug)))
    xf, xl = full[ic], lo[ic]
    rf, rl = np.corrcoef(xf, yc)[0, 1], np.corrcoef(xl, yc)[0, 1]
    d = []
    for b in Bc:
        i = np.concatenate([mem[k] for k in b])
        d.append(np.corrcoef(xl[i], yc[i])[0, 1] - np.corrcoef(xf[i], yc[i])[0, 1])
    rec.update(n_cells_r=len(ic), r_full=rf, r_without=rl, r_diff=rl - rf,
               r_lo=np.percentile(d, 2.5), r_hi=np.percentile(d, 97.5),
               max_abs_share_change=float(np.nanmax(np.abs(lo - full))))
    rows.append(rec)
    say(str({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
R = pd.DataFrame(rows).round(4)
R.to_csv(a.out + "_summary.csv", index=False)
print(R.to_string(index=False))
