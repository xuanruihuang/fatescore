"""Positive control: do the proposed checks detect dynamics when they are there?

A simulated bifurcation with known velocities and known fates. Cells move along
a latent stem and then split into two branches; every cell carries its fate
from the start. The progenitors' positions do not depend on their fate (no
lineage priming in the state), but their true velocity has a component towards
their future branch of relative size delta. With delta > 0 the dynamics carry
fate information the present state does not; with delta = 0 they carry none.
The latent coordinates are embedded in 50 genes with noise, and the velocity is
embedded the same way with its own noise.

Checks, each against its reference without dynamics:
  1. fate concordance: CellRank's velocity kernel (built with scVelo's
     velocity_graph from the simulated velocities) against the connectivity
     kernel on the same graph; AUROC of P(branch A) for the true fate of
     progenitor cells, with a paired bootstrap over cells;
  2. incremental value: the velocity kernel's probability added to expression
     (logistic regression on 20 principal components, five-fold
     cross-validation), against expression alone;
  3. direction: CBDir forward and reverse at the boundaries between
     progenitors and each branch, for the true velocity, the kNN-mean
     displacement and random vectors, in 30 principal components.

Usage: 74_positive_control.py <out_prefix> [--deltas 0,0.25,0.5,1] [--seeds 3]
"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import numpy.testing as npt
import anndata as ad
import scanpy as sc
import scvelo as scv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
import cellrank as cr  # noqa: E402


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def simulate(rng, delta, n_prog=3000, n_branch=3000, n_genes=50,
             pos_sd=0.25, vel_sd=0.3, gene_sd=0.3):
    fate = rng.integers(0, 2, size=n_prog + n_branch)          # 1 = branch A
    s = np.where(fate == 1, 1.0, -1.0)
    tau = np.concatenate([rng.uniform(0, 0.5, n_prog), rng.uniform(0.5, 1, n_branch)])
    prog = tau < 0.5
    L = np.zeros((tau.size, 2))
    L[prog, 0] = 2 * tau[prog]
    L[~prog, 0] = 1 + 2 * (tau[~prog] - 0.5)
    L[~prog, 1] = s[~prog] * 2 * (tau[~prog] - 0.5)
    L += pos_sd * rng.standard_normal(L.shape)
    V = np.zeros_like(L)
    V[prog, 0] = 1.0
    V[prog, 1] = delta * s[prog]
    V[~prog, 0] = 1.0
    V[~prog, 1] = s[~prog]
    V += vel_sd * rng.standard_normal(V.shape)
    G = rng.standard_normal((2, n_genes)) / np.sqrt(2)
    X = L @ G + gene_sd * rng.standard_normal((tau.size, n_genes)) + 5.0
    Vg = V @ G + gene_sd * 0.5 * rng.standard_normal((tau.size, n_genes))
    state = np.where(prog, "Prog", np.where(fate == 1, "A", "B"))
    term = (~prog) & (tau > 0.85)
    return X, Vg, state, fate, tau, term


def absorb(T, masks):
    is_term = np.any(masks, axis=0)
    trans = ~is_term
    Q = T[trans][:, trans].tocsr()
    R = np.column_stack([np.asarray(T[trans][:, m].sum(1)).ravel() for m in masks])
    X = np.zeros_like(R)
    for it in range(200000):
        X = Q @ X + R
        if it % 50 == 0 and float(np.abs(1 - X.sum(1)).max()) < 1e-8:
            break
    F = np.zeros((T.shape[0], len(masks)))
    F[trans] = X / X.sum(1, keepdims=True)
    for j, m in enumerate(masks):
        F[m, j] = 1
    return F


def cos_rows(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def cbdir(Vp, P, nbr, src, tgts):
    vals = []
    for tgt in tgts:
        for i in np.where(src)[0]:
            js = nbr[i][tgt[nbr[i]]]
            if js.size:
                c = cos_rows(np.repeat(Vp[i][None], js.size, 0), P[js] - P[i])
                vals.append(np.nanmean(c))
    return float(np.mean(vals)), len(vals)


ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--deltas", default="0,0.25,0.5,1")
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()

rows = []
for delta in (float(x) for x in a.deltas.split(",")):
    for seed in range(a.seeds):
        rng = np.random.default_rng(7400 + seed)
        X, Vg, state, fate, tau, term = simulate(rng, delta)
        B = ad.AnnData(X=X.astype(np.float32))
        B.layers["Ms"] = X.astype(np.float32)
        B.layers["velocity"] = Vg.astype(np.float32)
        B.var["velocity_genes"] = True
        sc.tl.pca(B, n_comps=30, svd_solver="arpack")
        sc.pp.neighbors(B, n_neighbors=30, use_rep="X_pca",
                        transformer=KNeighborsTransformer(
                            n_neighbors=30, metric="euclidean", algorithm="brute"))
        scv.tl.velocity_graph(B, vkey="velocity", xkey="Ms", show_progress_bar=False)
        masks = [term & (fate == 1), term & (fate == 0)]
        vk = cr.kernels.VelocityKernel(B, vkey="velocity", xkey="Ms")
        vk.compute_transition_matrix()
        ck = cr.kernels.ConnectivityKernel(B)
        ck.compute_transition_matrix()
        Fv = absorb(sp.csr_matrix(vk.transition_matrix), masks)[:, 0]
        Fc = absorb(sp.csr_matrix(ck.transition_matrix), masks)[:, 0]
        ev = (state == "Prog") & (tau < 0.4)        # progenitors well before the split
        y = fate[ev]
        av, ac = roc_auc_score(y, Fv[ev]), roc_auc_score(y, Fc[ev])
        bi = rng.integers(0, ev.sum(), size=(2000, ev.sum()))
        d = np.array([roc_auc_score(y[b], Fv[ev][b]) - roc_auc_score(y[b], Fc[ev][b])
                      for b in bi])
        # incremental: expression PCs alone vs plus the velocity kernel's probability
        Z = B.obsm["X_pca"][ev, :20]
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        pe, pv = np.zeros(y.size), np.zeros(y.size)
        for tr, te in skf.split(Z, y):
            m1 = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
            pe[te] = m1.decision_function(Z[te])
            Z2 = np.column_stack([Z, Fv[ev]])
            m2 = LogisticRegression(max_iter=2000).fit(Z2[tr], y[tr])
            pv[te] = m2.decision_function(Z2[te])
        di = np.array([roc_auc_score(y[b], pv[b]) - roc_auc_score(y[b], pe[b]) for b in bi])
        # direction
        P = B.obsm["X_pca"]
        Dm = sp.csr_matrix(B.obsp["distances"])
        nbr = [Dm.indices[Dm.indptr[i]:Dm.indptr[i + 1]] for i in range(B.n_obs)]
        pcs = B.varm["PCs"]
        Vp = (Vg - 0) @ pcs
        knn = np.array([P[nbr[i]].mean(0) - P[i] for i in range(B.n_obs)])
        rnd = rng.standard_normal(P.shape)
        src = state == "Prog"
        tg = [state == "A", state == "B"]
        rec = dict(delta=delta, seed=seed, auroc_velocity_kernel=av,
                   auroc_connectivity_kernel=ac, vk_minus_ck=av - ac,
                   vk_minus_ck_lo=np.percentile(d, 2.5), vk_minus_ck_hi=np.percentile(d, 97.5),
                   auroc_expression=roc_auc_score(y, pe),
                   increment=roc_auc_score(y, pv) - roc_auc_score(y, pe),
                   increment_lo=np.percentile(di, 2.5), increment_hi=np.percentile(di, 97.5))
        for name, F in (("true_velocity", Vp), ("knn_mean", knn), ("random", rnd)):
            fw, nf = cbdir(F, P, nbr, src, tg)
            rv = np.mean([cbdir(F, P, nbr, t, [src])[0] for t in tg])
            rec["cbdir_fwd_" + name] = fw
            rec["cbdir_rev_" + name] = rv
        rows.append(rec)
        say({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})
R = pd.DataFrame(rows).round(4)
R.to_csv(a.out + ".csv", index=False)
print(R.groupby("delta").mean(numeric_only=True).round(3).T.to_string())
