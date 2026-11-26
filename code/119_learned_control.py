"""Positive control for the learned-model pipeline (a reader asked whether the learned models'
shortfall could come from the pipeline itself: training on snapshots, carrying each early cell
forward, classifying its endpoint by the nearest cells of the last day).

Simulated snapshots on three days, then everything as on the real data:
  walk        CellRank's connectivity kernel on a 30-neighbour graph of all sampled cells (30 PCs),
              absorbed into the last day's cells of each branch;
  velocity    CellRank's velocity kernel from the true velocity on the same graph (dynamics given,
              at their best, as in 98_mechanism_sim.py);
  state       a cross-fitted logistic regression on the early cells' expression (5 folds): the most
              any model that sees only the present state can read from it;
  OT-CFM      trained across the three days in the 30-component PCA exactly as
              94_flow_matching_fate.py (exact minibatch OT, sigma 0.1, 3-layer MLP, Euler with 100
              steps per day), each early cell carried forward and its endpoint classified by its 20
              nearest last-day cells.
Score: AUROC of P(branch A) for the early cells' true fate (each early cell its own clone), and
the paired bootstrap over cells (1,000) of each predictor minus the walk.

Days: day 0 = progenitors early on the stem (tau < 0.2, the task cells), day 1 = tau 0.45-0.65
(late progenitors and the start of both branches), day 2 = tau 0.85-1 (the terminal cells).
Geometries (latent 2-D positions, mapped to 50 genes as in 98):
  bifurcation  98's stem and two straight branches; priming shifts the progenitors towards their
               branch, delta adds a component towards it to their true velocity.
  loop         as the bifurcation, but branch B turns back: its terminal cells lie behind the stem's
               start, level with the progenitors primed for A. The graph links the two, so the
               walk pulls A-primed cells towards B; the days do not, so a model trained across them
               need not.
A snapshot does not contain the true velocity, so delta reaches only the velocity kernel; the state
and OT-CFM predictors are the same for every delta.

Usage: 119_learned_control.py <out.csv> [--seeds 3 --steps 20000 --device cuda:0]
"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import numpy.testing as npt
import anndata as ad
import scanpy as sc
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsTransformer, NearestNeighbors

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
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--sigma", type=float, default=0.1)
ap.add_argument("--n-int", type=int, default=100)
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--n-day", type=int, default=1500, help="cells per day")
ap.add_argument("--grid", default="bifurcation:0.1,bifurcation:0.2,bifurcation:0.4,loop:0.2,loop:0.4")
ap.add_argument("--deltas", default="0,1")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
dev = torch.device(a.device)


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def branch(geom, s, p, u):
    """Position and direction of a branch cell at progress u in [0, 1] (A: s = +1, B: s = -1).
    Only u <= 0.3 (day 1) and u >= 0.7 (day 2) are sampled."""
    S = np.array([1.0, s * p])
    if geom == "bifurcation" or s > 0:
        D = np.array([1.0, s * (1 - p)])
        return S + u[:, None] * D, np.tile(D, (len(u), 1))
    # loop, branch B: heads down from the stem's end, then (unsampled) turns back; its terminal
    # cells lie at E, behind the stem's start and level with the A-primed progenitors, so that the
    # true direction of those progenitors (rightwards) points away from them
    D1 = np.array([0.3, -1.0])
    E, D2 = np.array([-0.6, p]), np.array([0.0, 1.0])
    early = u <= 0.5
    P = np.where(early[:, None], S + u[:, None] * D1, E + (u - 1)[:, None] * D2)
    V = np.where(early[:, None], D1, D2)
    return P, V


def simulate(rng, geom, p, n=1500, n_genes=50, pos_sd=0.25, vel_sd=0.3, gene_sd=0.3):
    tau = np.concatenate([rng.uniform(0, 0.2, n), rng.uniform(0.45, 0.65, n), rng.uniform(0.85, 1, n)])
    day = np.repeat([0, 1, 2], n)
    fate = rng.integers(0, 2, size=tau.size)                     # 1 = branch A
    s = np.where(fate == 1, 1.0, -1.0)
    prog = tau < 0.5
    L = np.zeros((tau.size, 2))
    V = np.zeros_like(L)
    L[prog, 0] = 2 * tau[prog]
    L[prog, 1] = p * s[prog]
    V[prog, 0] = 1.0                                               # delta added later
    for sg in (1.0, -1.0):
        m = (~prog) & (s == sg)
        L[m], V[m] = branch(geom, sg, p, 2 * (tau[m] - 0.5))
    L += pos_sd * rng.standard_normal(L.shape)
    noiseV = vel_sd * rng.standard_normal(V.shape)
    G = rng.standard_normal((2, n_genes)) / np.sqrt(2)
    X = L @ G + gene_sd * rng.standard_normal((tau.size, n_genes)) + 5.0
    gene_noise = gene_sd * 0.5 * rng.standard_normal((tau.size, n_genes))

    def velocity(delta):
        W = V.copy()
        W[prog, 1] = delta * s[prog]
        return (W + noiseV) @ G + gene_noise
    return X, velocity, fate, day, prog


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


class MLP(torch.nn.Module):
    def __init__(self, dim, w=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, w), torch.nn.SELU(), torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(), torch.nn.Linear(w, dim))

    def forward(self, x, t):
        return self.net(torch.cat([x, t[:, None]], 1))


def otcfm_fate(Z, day, fate, seed):
    by_day = [torch.tensor(Z[day == d], dtype=torch.float32) for d in (0, 1, 2)]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    v = MLP(Z.shape[1]).to(dev)
    opt = torch.optim.Adam(v.parameters(), lr=1e-3)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=a.sigma)
    for it in range(a.steps):
        k = int(rng.integers(0, 2))
        x0 = by_day[k][rng.integers(0, by_day[k].shape[0], a.batch)].to(dev)
        x1 = by_day[k + 1][rng.integers(0, by_day[k + 1].shape[0], a.batch)].to(dev)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        loss = ((v(xt, t + k) - ut) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = by_day[0].to(dev)
        dt = 1.0 / a.n_int
        for i in range(2 * a.n_int):
            z = z + v(z, torch.full((z.shape[0],), i * dt, device=dev)) * dt
        ends = z.cpu().numpy()
    last = np.where(day == 2)[0]
    _, nb = NearestNeighbors(n_neighbors=a.k).fit(Z[last]).kneighbors(ends)
    return (fate[last][nb] == 1).mean(1)


def state_fate(X0, y, seed):
    P = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X0, y):
        P[te] = LogisticRegression(max_iter=2000).fit(X0[tr], y[tr]).predict_proba(X0[te])[:, 1]
    return P


def boot(y, p, q, rng, n=1000):
    b = rng.integers(0, len(y), size=(n, len(y)))
    d = np.array([roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], q[i]) for i in b])
    return np.percentile(d, [2.5, 97.5])


rows = []
for cell in a.grid.split(","):
    geom, p = cell.split(":")
    p = float(p)
    for seed in range(a.seeds):
        t0 = time.time()
        rng = np.random.default_rng(11900 + seed)
        X, velocity, fate, day, prog = simulate(rng, geom, p, n=a.n_day)
        B = ad.AnnData(X=X.astype(np.float32))
        B.layers["Ms"] = X.astype(np.float32)
        B.var["velocity_genes"] = True
        sc.tl.pca(B, n_comps=30, svd_solver="arpack")
        sc.pp.neighbors(B, n_neighbors=30, use_rep="X_pca",
                        transformer=KNeighborsTransformer(n_neighbors=30, metric="euclidean", algorithm="brute"))
        masks = [(day == 2) & (fate == 1), (day == 2) & (fate == 0)]
        ck = cr.kernels.ConnectivityKernel(B)
        ck.compute_transition_matrix()
        Fc = absorb(sp.csr_matrix(ck.transition_matrix), masks)[:, 0]
        e0 = day == 0
        y = fate[e0]
        Zp = np.asarray(B.obsm["X_pca"], dtype=np.float32)
        Z = (Zp - Zp.mean(0)) / Zp.std()
        Pot = otcfm_fate(Z, day, fate, 11900 + seed)
        Pst = state_fate(X[e0], y, seed)
        brng = np.random.default_rng(119 + seed)
        base = dict(geometry=geom, priming=p, seed=seed, auroc_walk=roc_auc_score(y, Fc[e0]),
                    auroc_state=roc_auc_score(y, Pst), auroc_otcfm=roc_auc_score(y, Pot))
        lo_o, hi_o = boot(y, Pot, Fc[e0], brng)
        lo_s, hi_s = boot(y, Pst, Fc[e0], brng)
        for delta in (float(x) for x in a.deltas.split(",")):
            B.layers["velocity"] = velocity(delta).astype(np.float32)
            scv.tl.velocity_graph(B, vkey="velocity", xkey="Ms", show_progress_bar=False)
            vk = cr.kernels.VelocityKernel(B, vkey="velocity", xkey="Ms")
            vk.compute_transition_matrix()
            Fv = absorb(sp.csr_matrix(vk.transition_matrix), masks)[:, 0]
            lo_v, hi_v = boot(y, Fv[e0], Fc[e0], brng)
            r = dict(base, delta=delta, auroc_velocity=roc_auc_score(y, Fv[e0]))
            r.update(gain_velocity=r["auroc_velocity"] - r["auroc_walk"], lo_velocity=lo_v, hi_velocity=hi_v,
                     gain_state=r["auroc_state"] - r["auroc_walk"], lo_state=lo_s, hi_state=hi_s,
                     gain_otcfm=r["auroc_otcfm"] - r["auroc_walk"], lo_otcfm=lo_o, hi_otcfm=hi_o)
            rows.append(r)
            say({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()})
        pd.DataFrame(rows).round(4).to_csv(a.out, index=False)
        say("%s priming %.1f seed %d: %.0f s" % (geom, p, seed, time.time() - t0))
R = pd.DataFrame(rows)
print(R.groupby(["geometry", "priming", "delta"])[["auroc_walk", "auroc_state", "auroc_otcfm", "auroc_velocity",
                                                   "gain_otcfm", "gain_velocity", "gain_state"]].mean().round(3).to_string())
