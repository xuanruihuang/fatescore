"""Count-level positive control: would the tests detect a kinetic fate signal at
LARRY's depth, through the actual velocity pipeline?

74_positive_control.py gives CellRank the true velocity. Here the velocity has
to be estimated from simulated spliced and unspliced counts, as on LARRY.

Simulation. Cells lie on a pseudotime tau in [0, 1]. Every cell belongs to one
of two fates (A, B), drawn with probability one half. Genes (300):
  - 100 maturation genes, switched on at gene-specific times along tau for all
    cells;
  - 50 A genes and 50 B genes, switched on at tau_on (fate-specific programme)
    only in cells of that fate, with tau_on ~ N(t_switch, 0.03) per cell;
  - 100 constant genes.
In progenitors the fate genes may carry a small basal difference (priming,
relative size `priming`), so the present state can be partly fate-informative.
Each gene follows the standard transcription-splicing-degradation model
(alpha on/off, beta = 1, gamma ~ U(0.2, 0.6), time scale 10 per unit tau), so a
gene that has just switched on has its unspliced molecules up before its
spliced molecules follow: the classic premise of RNA velocity. Counts are
Poisson with per-cell capture scaled so that the median cell has about 650
spliced and 120 unspliced molecules (LARRY's day-2 depth on the selected
genes; `depth` multiplies both).

Scored cells: progenitors with tau in [t_switch - 0.06, t_switch + 0.02], some
of which have just switched on their fate programme. Terminal states: cells of
each fate with tau > 0.85. Pipeline: per-cell normalization, log, 30 PCs, 30
exact neighbours, scVelo moments, steady-state (or, with --mode dynamical,
dynamical) velocity, CellRank velocity
kernel and connectivity kernel, absorption probabilities. Tests:
  - stand-alone: AUROC of the velocity kernel against the connectivity kernel
    (paired bootstrap over cells);
  - incremental: logistic regression on 20 PCs of spliced counts, with and
    without the velocity kernel's probability (five-fold cross-validation);
  - CBDir forward and reverse (progenitors to the two branches) for the
    estimated velocity and the kNN-mean displacement, in 30 PCs.

Usage: 78_kinetic_control.py <out_prefix> [--seeds 3] [--mode deterministic|dynamical]
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

scv.settings.verbosity = 0


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def kinetics(t, alpha, gamma, beta=1.0):
    """u, s at time t >= 0 after switching on from zero (alpha constant)."""
    t = np.maximum(t, 0.0)
    u = alpha / beta * (1 - np.exp(-beta * t))
    s = (alpha / gamma * (1 - np.exp(-gamma * t))
         + alpha / (gamma - beta) * (np.exp(-gamma * t) - np.exp(-beta * t)))
    return u, s


def simulate(rng, priming, t_switch, depth, n_cells=6000, T=10.0, G_mat=100, G_fate=50):
    fate = rng.integers(0, 2, n_cells)
    tau = rng.uniform(0, 1, n_cells)
    G_const = 100
    gamma = rng.uniform(0.2, 0.6, G_mat + 2 * G_fate + G_const)
    alpha = rng.lognormal(0, 0.5, gamma.size)
    U = np.zeros((n_cells, gamma.size))
    S = np.zeros_like(U)
    on_mat = rng.uniform(0, 0.9, G_mat)
    for g in range(G_mat):
        U[:, g], S[:, g] = kinetics((tau - on_mat[g]) * T, alpha[g], gamma[g])
    t_on = t_switch + 0.03 * rng.standard_normal(n_cells)
    for k, f in ((0, 1), (1, 0)):            # A genes for fate 1, B genes for fate 0
        cols = G_mat + k * G_fate + np.arange(G_fate)
        for g in cols:
            act = (fate == f)
            u, s = kinetics((tau - t_on) * T, alpha[g], gamma[g])
            U[:, g] = np.where(act, u, 0.0)
            S[:, g] = np.where(act, s, 0.0)
            # basal level with lineage priming before the switch
            base = 0.05 * alpha[g] / gamma[g] * (1 + priming * np.where(act, 1, -1))
            S[:, g] += np.clip(base, 0, None)
            U[:, g] += np.clip(base, 0, None) * gamma[g]
    cg = G_mat + 2 * G_fate + np.arange(G_const)
    U[:, cg] = alpha[cg] / 1.0
    S[:, cg] = alpha[cg] / gamma[cg]
    cap = rng.lognormal(0, 0.3, n_cells)[:, None]
    S *= cap
    U *= cap
    s_scale = 650 / np.median(S.sum(1))
    u_scale = 120 / np.median(U.sum(1))
    Sc = rng.poisson(S * s_scale * depth)
    Uc = rng.poisson(U * u_scale * depth)
    return Sc, Uc, fate, tau


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


def cbdir(V, P, nbr, src, tgts):
    vals = []
    for tgt in tgts:
        for i in np.where(src)[0]:
            js = nbr[i][tgt[nbr[i]]]
            if js.size:
                vals.append(np.nanmean(cos_rows(np.repeat(V[i][None], js.size, 0), P[js] - P[i])))
    return float(np.mean(vals))


ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--primings", default="0,0.5")
ap.add_argument("--depths", default="1,5,20")
ap.add_argument("--t-switch", type=float, default=0.33)
# "deterministic" (steady-state, the original runs) or "dynamical", the model
# whose velocities CellRank's velocity kernel uses on LARRY (referee)
ap.add_argument("--mode", default="deterministic")
# scoring window relative to the switch: [t_switch - 0.06 + shift, t_switch + 0.02 + shift];
# shift 0.06 scores cells most of which have switched on (referee)
ap.add_argument("--window-shift", type=float, default=0.0)
# genes per fate programme and maturation genes (default 50 and 100)
ap.add_argument("--n-fate", type=int, default=50)
ap.add_argument("--n-mat", type=int, default=100)
a = ap.parse_args()

rows = []
for priming in (float(x) for x in a.primings.split(",")):
    for depth in (float(x) for x in a.depths.split(",")):
        for seed in range(a.seeds):
            rng = np.random.default_rng(7800 + seed)
            Sc, Uc, fate, tau = simulate(rng, priming, a.t_switch, depth, G_mat=a.n_mat, G_fate=a.n_fate)
            B = ad.AnnData(X=sp.csr_matrix(Sc.astype(np.float32)))
            B.layers["spliced"] = sp.csr_matrix(Sc.astype(np.float32))
            B.layers["unspliced"] = sp.csr_matrix(Uc.astype(np.float32))
            scv.pp.normalize_per_cell(B)
            sc.pp.log1p(B)
            sc.tl.pca(B, n_comps=30, svd_solver="arpack")
            sc.pp.neighbors(B, n_neighbors=30, use_rep="X_pca",
                            transformer=KNeighborsTransformer(
                                n_neighbors=30, metric="euclidean", algorithm="brute"))
            scv.pp.moments(B, n_pcs=30, n_neighbors=30)
            if a.mode == "dynamical":
                scv.tl.recover_dynamics(B, n_jobs=8, var_names="all")
            scv.tl.velocity(B, mode=a.mode)
            scv.tl.velocity_graph(B, show_progress_bar=False)
            term = tau > 0.85
            masks = [term & (fate == 1), term & (fate == 0)]
            vk = cr.kernels.VelocityKernel(B)
            vk.compute_transition_matrix()
            ck = cr.kernels.ConnectivityKernel(B)
            ck.compute_transition_matrix()
            Fv = absorb(sp.csr_matrix(vk.transition_matrix), masks)[:, 0]
            Fc = absorb(sp.csr_matrix(ck.transition_matrix), masks)[:, 0]
            ev = (tau >= a.t_switch - 0.06 + a.window_shift) & (tau <= a.t_switch + 0.02 + a.window_shift)
            y = fate[ev]
            av, ac = roc_auc_score(y, Fv[ev]), roc_auc_score(y, Fc[ev])
            bi = rng.integers(0, ev.sum(), size=(2000, ev.sum()))
            d = np.array([roc_auc_score(y[b], Fv[ev][b]) - roc_auc_score(y[b], Fc[ev][b])
                          for b in bi])
            Z = B.obsm["X_pca"][ev, :20]
            skf = StratifiedKFold(5, shuffle=True, random_state=seed)
            pe, pv = np.zeros(y.size), np.zeros(y.size)
            for tr, te in skf.split(Z, y):
                m1 = LogisticRegression(max_iter=3000).fit(Z[tr], y[tr])
                pe[te] = m1.decision_function(Z[te])
                Z2 = np.column_stack([Z, Fv[ev]])
                m2 = LogisticRegression(max_iter=3000).fit(Z2[tr], y[tr])
                pv[te] = m2.decision_function(Z2[te])
            di = np.array([roc_auc_score(y[b], pv[b]) - roc_auc_score(y[b], pe[b]) for b in bi])
            # the unspliced layer itself added to expression (20 PCs of log counts)
            Ul = np.log1p(Uc[ev] / np.maximum(Uc[ev].sum(1, keepdims=True), 1) * 1e4)
            from sklearn.decomposition import PCA as _PCA
            pu = np.zeros(y.size)
            for tr, te in skf.split(Z, y):
                pc = _PCA(20, random_state=0).fit(Ul[tr])
                Z3 = np.column_stack([Z, pc.transform(Ul)])
                m3 = LogisticRegression(max_iter=3000).fit(Z3[tr], y[tr])
                pu[te] = m3.decision_function(Z3[te])
            du = np.array([roc_auc_score(y[b], pu[b]) - roc_auc_score(y[b], pe[b]) for b in bi])
            P = B.obsm["X_pca"]
            Dm = sp.csr_matrix(B.obsp["distances"])
            nbr = [Dm.indices[Dm.indptr[i]:Dm.indptr[i + 1]] for i in range(B.n_obs)]
            Vp = np.nan_to_num(np.asarray(B.layers["velocity"])) @ B.varm["PCs"]
            knn = np.array([P[nbr[i]].mean(0) - P[i] for i in range(B.n_obs)])
            src = tau < 0.4
            tg = [(tau >= 0.4) & (fate == 1), (tau >= 0.4) & (fate == 0)]
            rec = dict(mode=a.mode, window_shift=a.window_shift, n_fate=a.n_fate, n_mat=a.n_mat,
                       frac_switched=float(np.mean(tau[(tau >= a.t_switch - 0.06 + a.window_shift) & (tau <= a.t_switch + 0.02 + a.window_shift)] > a.t_switch)),
                       priming=priming, depth=depth, seed=seed, n_scored=int(ev.sum()),
                       median_spliced=float(np.median(Sc.sum(1))),
                       median_unspliced=float(np.median(Uc.sum(1))),
                       auroc_velocity_kernel=av, auroc_connectivity_kernel=ac,
                       vk_minus_ck=av - ac, vk_minus_ck_lo=np.percentile(d, 2.5),
                       vk_minus_ck_hi=np.percentile(d, 97.5),
                       auroc_expression=roc_auc_score(y, pe),
                       increment=roc_auc_score(y, pv) - roc_auc_score(y, pe),
                       increment_lo=np.percentile(di, 2.5), increment_hi=np.percentile(di, 97.5),
                       increment_unspliced=roc_auc_score(y, pu) - roc_auc_score(y, pe),
                       increment_unspliced_lo=np.percentile(du, 2.5),
                       increment_unspliced_hi=np.percentile(du, 97.5))
            for name, F in (("velocity", Vp), ("knn_mean", knn)):
                rec["cbdir_fwd_" + name] = cbdir(F, P, nbr, src, tg)
                rec["cbdir_rev_" + name] = np.mean([cbdir(F, P, nbr, t, [src]) for t in tg])
            rows.append(rec)
            say({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})
R = pd.DataFrame(rows).round(4)
R.to_csv(a.out + ".csv", index=False)
print(R.groupby(["priming", "depth"]).mean(numeric_only=True).drop(columns="seed").round(3).T.to_string())
