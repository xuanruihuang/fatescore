"""When does a predictor without dynamics match one with them? A two-factor simulation.

The bifurcation of 74_positive_control.py, with a second factor. Progenitors move
along a stem and split into two branches; every cell carries its fate from the start.
  priming  the progenitors' position is shifted towards their future branch by
           `priming` (in units of the branch separation at the end): the present
           state reveals fate. 0 = no priming.
  delta    the progenitors' true velocity has a component towards their future
           branch of relative size `delta`: the dynamics reveal fate. 0 = none.
Both kernels run on the same neighbour graph to the same terminal cells: CellRank's
velocity kernel from the true velocity (the method with dynamics, at its best: the
velocity is known, not estimated) and the connectivity kernel (no dynamics). Score:
AUROC of P(branch A) for the true fate of progenitors well before the split, and the
paired bootstrap (over cells) of velocity kernel minus connectivity kernel.
A third factor stands in for estimation error:
  noise    the velocity given to the kernel is (1 - noise) x the true velocity plus
           noise x the velocity of a randomly chosen other cell (1 = velocities
           shuffled across cells, the ladder's lowest step with a direction).
And a systematic error like the one splicing velocity carries:
  decay    a degradation-like term -s is added to the velocity (s = expression
           shifted to be non-negative, like counts), scaled so that at decay 1 its
           mean norm equals that of the velocity.

Usage: 98_mechanism_sim.py <out.csv> [--primings 0,0.1,0.2,0.4 --deltas 0,0.25,0.5,1 --noises 0 --decays 0 --seeds 3]; negative values as --deltas=-0.5
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


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def simulate(rng, priming, delta, n_prog=3000, n_branch=3000, n_genes=50,
             pos_sd=0.25, vel_sd=0.3, gene_sd=0.3):
    fate = rng.integers(0, 2, size=n_prog + n_branch)          # 1 = branch A
    s = np.where(fate == 1, 1.0, -1.0)
    tau = np.concatenate([rng.uniform(0, 0.5, n_prog), rng.uniform(0.5, 1, n_branch)])
    prog = tau < 0.5
    L = np.zeros((tau.size, 2))
    L[prog, 0] = 2 * tau[prog]
    L[prog, 1] = priming * s[prog]
    L[~prog, 0] = 1 + 2 * (tau[~prog] - 0.5)
    L[~prog, 1] = s[~prog] * (priming + (1 - priming) * 2 * (tau[~prog] - 0.5))
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


ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--primings", default="0,0.1,0.2,0.4")
ap.add_argument("--deltas", default="0,0.25,0.5,1")
ap.add_argument("--noises", default="0")
ap.add_argument("--decays", default="0")
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()

rows = []
for priming in (float(x) for x in a.primings.split(",")):
    for delta in (float(x) for x in a.deltas.split(",")):
        for noise, decay in ((float(x), float(y)) for x in a.noises.split(",")
                             for y in a.decays.split(",")):
            for seed in range(a.seeds):
                rng = np.random.default_rng(9800 + seed)
                X, Vg, fate, tau, prog, term = simulate(rng, priming, delta)
                if noise > 0:
                    Vg = (1 - noise) * Vg + noise * Vg[rng.permutation(Vg.shape[0])]
                if decay > 0:
                    Dg = -(X - X.min(0))
                    Vg = Vg + decay * Dg * (np.linalg.norm(Vg, axis=1).mean()
                                            / np.linalg.norm(Dg, axis=1).mean())
                B = ad.AnnData(X=X.astype(np.float32))
                B.layers["Ms"] = X.astype(np.float32)
                B.layers["velocity"] = Vg.astype(np.float32)
                B.var["velocity_genes"] = True
                sc.tl.pca(B, n_comps=30, svd_solver="arpack")
                sc.pp.neighbors(B, n_neighbors=30, use_rep="X_pca",
                                transformer=KNeighborsTransformer(n_neighbors=30, metric="euclidean",
                                                                  algorithm="brute"))
                scv.tl.velocity_graph(B, vkey="velocity", xkey="Ms", show_progress_bar=False)
                masks = [term & (fate == 1), term & (fate == 0)]
                vk = cr.kernels.VelocityKernel(B, vkey="velocity", xkey="Ms")
                vk.compute_transition_matrix()
                ck = cr.kernels.ConnectivityKernel(B)
                ck.compute_transition_matrix()
                Fv = absorb(sp.csr_matrix(vk.transition_matrix), masks)[:, 0]
                Fc = absorb(sp.csr_matrix(ck.transition_matrix), masks)[:, 0]
                ev = prog & (tau < 0.4)
                y = fate[ev]
                av, ac = roc_auc_score(y, Fv[ev]), roc_auc_score(y, Fc[ev])
                bi = rng.integers(0, ev.sum(), size=(1000, ev.sum()))
                d = np.array([roc_auc_score(y[b], Fv[ev][b]) - roc_auc_score(y[b], Fc[ev][b]) for b in bi])
                rec = dict(priming=priming, delta=delta, noise=noise, decay=decay, seed=seed, auroc_velocity_kernel=av,
                           auroc_connectivity_kernel=ac, vk_minus_ck=av - ac,
                           lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5))
                rows.append(rec)
                say({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})
                pd.DataFrame(rows).round(4).to_csv(a.out, index=False)
R = pd.DataFrame(rows)
print(R.groupby(["priming", "delta", "noise", "decay"])[["auroc_velocity_kernel", "auroc_connectivity_kernel", "vk_minus_ck"]]
      .mean().round(3).to_string())
