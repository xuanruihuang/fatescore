"""The positive control of 98_mechanism_sim.py on a trifurcation, with and without
fate-dependent growth.

Progenitors move along a stem (latent x) and split into three branches whose directions lie
120 degrees apart in the (y, z) plane. Every cell carries its fate from the start.
  priming  progenitors are shifted towards their future branch direction by `priming`
           (in units of the branch separation at the end): the present state reveals fate;
  delta    progenitors' true velocity has a component towards their future branch of
           relative size `delta`: the dynamics reveal fate;
  noise    the velocity given to the kernel is (1 - noise) x true + noise x another cell's;
  growth   fate-dependent growth: branch 0 proliferates faster, so among branch cells its
           share rises with time as exp(growth x (tau - 0.5)) relative to the others
           (progenitors are sampled with equal fates; growth 0 = none).
Both kernels run on the same neighbour graph to the same terminal cells (tau > 0.85 of each
branch): CellRank's velocity kernel from the true velocity and the connectivity kernel.
Score: one-vs-rest AUROC of P(fate k) for the true fate of progenitors with tau < 0.4,
averaged over the three fates (macro), and the paired bootstrap over cells of the difference.

Usage: 108_mechanism_multi.py <out.csv> [--primings 0.1,0.2,0.3 --deltas 0,0.5,1
       --noises 0 --growths 0,2 --seeds 3]
"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import numpy.testing as npt
import anndata as ad
import scanpy as sc
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
for _cls in (sp.csr_array, sp.csc_array):
    if not hasattr(_cls, "getnnz"):
        _cls.getnnz = _cls._getnnz
import scvelo as scv  # noqa: E402
import cellrank as cr  # noqa: E402

DIRS = np.array([[np.cos(a), np.sin(a)] for a in (0, 2 * np.pi / 3, 4 * np.pi / 3)])


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def simulate(rng, priming, delta, growth, n_prog=3000, n_branch=4500, n_genes=50,
             pos_sd=0.25, vel_sd=0.3, gene_sd=0.3):
    tau = np.concatenate([rng.uniform(0, 0.5, n_prog), rng.uniform(0.5, 1, n_branch)])
    prog = tau < 0.5
    fate = np.empty(tau.size, int)
    fate[prog] = rng.integers(0, 3, prog.sum())
    w = np.exp(growth * (tau[~prog] - 0.5))                       # branch 0 grows faster
    p0 = w / (w + 2.0)
    u = rng.uniform(size=(~prog).sum())
    fb = np.where(u < p0, 0, np.where(u < p0 + (1 - p0) / 2, 1, 2))
    fate[~prog] = fb
    D = DIRS[fate]
    L = np.zeros((tau.size, 3))
    L[prog, 0] = 2 * tau[prog]
    L[prog, 1:] = priming * D[prog]
    L[~prog, 0] = 1 + 2 * (tau[~prog] - 0.5)
    L[~prog, 1:] = D[~prog] * (priming + (1 - priming) * 2 * (tau[~prog] - 0.5))[:, None]
    L += pos_sd * rng.standard_normal(L.shape)
    V = np.zeros_like(L)
    V[:, 0] = 1.0
    V[prog, 1:] = delta * D[prog]
    V[~prog, 1:] = D[~prog]
    V += vel_sd * rng.standard_normal(V.shape)
    G = rng.standard_normal((3, n_genes)) / np.sqrt(3)
    X = L @ G + gene_sd * rng.standard_normal((tau.size, n_genes)) + 5.0
    Vg = V @ G + gene_sd * 0.5 * rng.standard_normal((tau.size, n_genes))
    term = (~prog) & (tau > 0.85)
    return X, Vg, fate, tau, prog, term


def absorb(T, masks):
    is_term = np.any(masks, axis=0)
    trans = ~is_term
    Q = T[trans][:, trans].tocsr()
    R = np.column_stack([np.asarray(T[trans][:, m].sum(1)).ravel() for m in masks])
    X = np.zeros_like(R)
    for it in range(200000):
        X = Q @ X + R
        if it % 50 == 0 and np.abs(1.0 - X.sum(1)).max() < 1e-8:
            break
    F = np.zeros((T.shape[0], len(masks)))
    F[trans] = X / np.where(X.sum(1, keepdims=True) == 0, 1, X.sum(1, keepdims=True))
    for j, m in enumerate(masks):
        F[m, j] = 1.0
    return F


def macro_auc(y, F):
    return np.mean([roc_auc_score((y == k).astype(int), F[:, k]) for k in range(3)])


ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--primings", default="0.1,0.2,0.3")
ap.add_argument("--deltas", default="0,0.5,1")
ap.add_argument("--noises", default="0")
ap.add_argument("--growths", default="0,2")
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()

rows = []
for growth in (float(x) for x in a.growths.split(",")):
    for priming in (float(x) for x in a.primings.split(",")):
        for delta in (float(x) for x in a.deltas.split(",")):
            for noise in (float(x) for x in a.noises.split(",")):
                for seed in range(a.seeds):
                    rng = np.random.default_rng(10800 + seed)
                    X, Vg, fate, tau, prog, term = simulate(rng, priming, delta, growth)
                    if noise > 0:
                        Vg = (1 - noise) * Vg + noise * Vg[rng.permutation(Vg.shape[0])]
                    B = ad.AnnData(X=X.astype(np.float32))
                    B.layers["Ms"] = X.astype(np.float32)
                    B.layers["velocity"] = Vg.astype(np.float32)
                    B.var["velocity_genes"] = True
                    sc.tl.pca(B, n_comps=30, svd_solver="arpack")
                    sc.pp.neighbors(B, n_neighbors=30, use_rep="X_pca",
                                    transformer=KNeighborsTransformer(n_neighbors=30, metric="euclidean",
                                                                      algorithm="brute"))
                    scv.tl.velocity_graph(B, vkey="velocity", xkey="Ms", show_progress_bar=False)
                    masks = [term & (fate == k) for k in range(3)]
                    vk = cr.kernels.VelocityKernel(B, vkey="velocity", xkey="Ms")
                    vk.compute_transition_matrix()
                    ck = cr.kernels.ConnectivityKernel(B)
                    ck.compute_transition_matrix()
                    Fv = absorb(sp.csr_matrix(vk.transition_matrix), masks)
                    Fc = absorb(sp.csr_matrix(ck.transition_matrix), masks)
                    ev = prog & (tau < 0.4)
                    y, Fve, Fce = fate[ev], Fv[ev], Fc[ev]
                    av, ac = macro_auc(y, Fve), macro_auc(y, Fce)
                    bi = rng.integers(0, ev.sum(), size=(500, ev.sum()))
                    d = np.array([macro_auc(y[b], Fve[b]) - macro_auc(y[b], Fce[b]) for b in bi])
                    rec = dict(growth=growth, priming=priming, delta=delta, noise=noise, seed=seed,
                               auroc_velocity_kernel=av, auroc_connectivity_kernel=ac, vk_minus_ck=av - ac,
                               lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                               branch0_share_terminal=float(np.mean(fate[term] == 0)))
                    rows.append(rec)
                    say({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})
                    pd.DataFrame(rows).round(4).to_csv(a.out, index=False)
R = pd.DataFrame(rows)
print(R.groupby(["growth", "priming", "delta", "noise"])[["auroc_connectivity_kernel", "vk_minus_ck", "lo", "hi"]]
      .agg({"auroc_connectivity_kernel": "mean", "vk_minus_ck": "mean", "lo": "min", "hi": "max"}).round(3).to_string())
