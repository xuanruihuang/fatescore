"""Fate probabilities from flow-matching models, on the same task as every other arm.

Flow matching is how much of the recent machine-learning work on single-cell
dynamics is built (TrajectoryNet, OT-CFM, [SF]2M, metric flow matching), and these
models are evaluated on LARRY as the others are. Two models from torchcfm:

  otcfm  conditional flow matching with exact minibatch optimal-transport couplings
         (OT-CFM, Tong et al. 2024); an ODE, one trajectory per cell
  sbcfm  Schrodinger-bridge conditional flow matching with a score network
         ([SF]2M, Tong et al. 2024); an SDE, many trajectories per cell

Both are trained across the sampling days in the 30-component PCA of the same object
every other method uses (41_ot_fate.build), consecutive days as the marginals of one
time axis (day 2 -> 4 on t in [0, 1], day 4 -> 6 on [1, 2]). Fate as in PRESCIENT
(43_prescient_fate.py): every day-2 cell is simulated forward to day 6 and each
endpoint is classified by its nearest observed cells (k = 10); the fate probability
is the fraction of endpoints (SDE) or of the endpoint's neighbours (ODE) in each
terminal population. Three training seeds, probabilities averaged.

Writes snap_otcfm.npy and snap_sbcfm.npy (one row per day-2 cell, one column per
terminal population in the order of the directory's cellrank log) and a json log.

Usage: 94_flow_matching_fate.py <h5ad> <featdir> [--steps 20000 --seeds 3 --device cuda:0]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("otfate", os.path.join(HERE, "41_ot_fate.py"))
otfate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(otfate)


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--sigma", type=float, default=0.1)
ap.add_argument("--n-sims", type=int, default=50)
ap.add_argument("--n-int", type=int, default=100, help="integration steps per day interval")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--scale", choices=["global", "pc"], default="global",
                help="global: one scale for all components (keeps the PCA geometry); pc: per component")
ap.add_argument("--clf", choices=["day6", "all"], default="day6",
                help="cells the endpoint classifier uses (PRESCIENT/Chreode protocol: day-6 cells)")
ap.add_argument("--tag", default="", help="suffix of the output names, for variants")
ap.add_argument("--fit-n", type=int, default=2000)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--models", default="otcfm,sbcfm")
a = ap.parse_args()

from torchcfm.conditional_flow_matching import (  # noqa: E402
    ExactOptimalTransportConditionalFlowMatcher, SchrodingerBridgeConditionalFlowMatcher)

B, d2 = otfate.build(a.h5ad, a.featdir, "time_info", "state_info")
pops = otfate.populations(a.featdir, B.obs["state"].values)
X = np.asarray(B.obsm["X_pca"], dtype=np.float32)
mu = X.mean(0)
sd = X.std(0) if a.scale == "pc" else np.full(X.shape[1], X.std(), dtype=np.float32)
Z = (X - mu) / sd
day = B.obs["day"].values
days = sorted(set(day[~np.isnan(day)].tolist()))
assert len(days) >= 2, days
by_day = [torch.tensor(Z[day == d]) for d in days]
state = B.obs["state"].values
say("cells per day %s; terminal populations %s" % ([int(t.shape[0]) for t in by_day], pops))
dev = torch.device(a.device)
dim = Z.shape[1]


class MLP(torch.nn.Module):
    def __init__(self, dim, w=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, w), torch.nn.SELU(), torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(), torch.nn.Linear(w, dim))

    def forward(self, x, t):
        return self.net(torch.cat([x, t[:, None]], 1))


def batch_pair(rng):
    k = int(rng.integers(0, len(days) - 1))
    i0 = rng.integers(0, by_day[k].shape[0], a.batch)
    i1 = rng.integers(0, by_day[k + 1].shape[0], a.batch)
    return k, by_day[k][i0].to(dev), by_day[k + 1][i1].to(dev)


def train(kind, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    v = MLP(dim).to(dev)
    s = MLP(dim).to(dev) if kind == "sbcfm" else None
    params = list(v.parameters()) + (list(s.parameters()) if s is not None else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    if kind == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=a.sigma)
    else:
        FM = SchrodingerBridgeConditionalFlowMatcher(sigma=a.sigma, ot_method="exact")
    for it in range(a.steps):
        k, x0, x1 = batch_pair(rng)
        if kind == "otcfm":
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            loss = ((v(xt, t + k) - ut) ** 2).mean()
        else:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
            lam = FM.compute_lambda(t)
            loss = ((v(xt, t + k) - ut) ** 2).mean() + ((lam[:, None] * s(xt, t + k) + eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 5000 == 0:
            say("  %s seed %d step %d loss %.4f" % (kind, seed, it + 1, loss.item()))
    return v, s


@torch.no_grad()
def simulate(kind, v, s, x, n_sims):
    """Integrate from t = 0 (day 2) to t = 2 (day 6); Euler for the ODE,
    Euler-Maruyama for the SDE (drift v + sigma^2/2 * score, noise sigma)."""
    n_steps = a.n_int * (len(days) - 1)
    dt = (len(days) - 1) / n_steps
    out = []
    for _ in range(n_sims):
        z = x.clone()
        for i in range(n_steps):
            t = torch.full((z.shape[0],), i * dt, device=dev)
            if kind == "otcfm":
                z = z + v(z, t) * dt
            else:
                drift = v(z, t) + 0.5 * a.sigma ** 2 * s(z, t)
                z = z + drift * dt + a.sigma * np.sqrt(dt) * torch.randn_like(z)
        out.append(z.cpu().numpy())
    return np.stack(out, 1)  # cells x sims x dim


pop_idx = {p: j for j, p in enumerate(pops)}
clf_cells = np.where(day == days[-1])[0] if a.clf == "day6" else np.arange(len(day))
nn_index = NearestNeighbors(n_neighbors=a.k).fit(Z[clf_cells])
lab = np.array([pop_idx.get(x, -1) for x in state[clf_cells]])


@torch.no_grad()
def marginals(kind, v, s, x):
    """One trajectory per start cell; the states at t = 1 (day 4) and t = 2 (day 6)."""
    dt = 1.0 / a.n_int
    z = x.clone()
    snaps = {}
    for i in range(a.n_int * (len(days) - 1)):
        t = torch.full((z.shape[0],), i * dt, device=dev)
        if kind == "otcfm":
            z = z + v(z, t) * dt
        else:
            z = z + (v(z, t) + 0.5 * a.sigma ** 2 * s(z, t)) * dt + a.sigma * np.sqrt(dt) * torch.randn_like(z)
        if (i + 1) % a.n_int == 0:
            snaps[(i + 1) // a.n_int] = z.cpu().numpy()
    return snaps


def w2(A, Bm):
    """Exact earth mover's distance (squared Euclidean cost, square root) in the
    original PCA units."""
    import ot
    A, Bm = A * sd + mu, Bm * sd + mu
    M = ot.dist(A, Bm)
    return float(np.sqrt(ot.emd2(np.full(len(A), 1 / len(A)), np.full(len(Bm), 1 / len(Bm)), M,
                                 numItermax=2_000_000)))


fit_rng = np.random.default_rng(940)
fit_idx = {}
for d in days:
    pool = np.where(day == d)[0]
    fit_idx[d] = fit_rng.choice(pool, size=min(2 * a.fit_n, len(pool)), replace=False)
half = {d: len(fit_idx[d]) // 2 for d in days}
fit_ref = {}
for d0, d1 in zip(days[:-1], days[1:]):
    # moving nothing: the earlier day against the later one
    fit_ref["day%g_vs_day%g" % (d0, d1)] = w2(Z[fit_idx[d0][:half[d0]]], Z[fit_idx[d1][:half[d1]]])
for d in days[1:]:
    # the floor: two disjoint batches of the same day
    fit_ref["day%g_vs_day%g" % (d, d)] = w2(Z[fit_idx[d][:half[d]]], Z[fit_idx[d][half[d]:2 * half[d]]])
say("fit references (W2, PCA units): %s" % {k: round(v_, 3) for k, v_ in fit_ref.items()})


def classify(ends):
    """Fraction of each endpoint's k nearest observed cells in each terminal
    population, averaged over endpoints (cells x pops)."""
    n, m, _ = ends.shape
    _, nb = nn_index.kneighbors(ends.reshape(n * m, dim))
    L = lab[nb]  # (n*m, k)
    P = np.zeros((n * m, len(pops)))
    for j in range(len(pops)):
        P[:, j] = (L == j).mean(1)
    return P.reshape(n, m, len(pops)).mean(1)


x2 = torch.tensor(Z[d2]).to(dev)
log = dict(pops=pops, steps=a.steps, batch=a.batch, sigma=a.sigma, seeds=a.seeds, k=a.k,
           n_sims=a.n_sims, n_int=a.n_int, n_d2=int(len(d2)), scale=a.scale, clf=a.clf,
           fit_reference=fit_ref)
for kind in a.models.split(","):
    Ps, fits = [], []
    for seed in range(a.seeds):
        t0 = time.time()
        v, s = train(kind, 9400 + seed)
        snaps = marginals(kind, v, s, torch.tensor(Z[fit_idx[days[0]][:half[days[0]]]]).to(dev))
        f = {"sim_day%g_vs_day%g" % (d, d): w2(snaps[i], Z[fit_idx[d][:half[d]]])
             for i, d in enumerate(days[1:], 1)}
        fits.append(f)
        say("%s seed %d fit (W2): %s" % (kind, seed, {k: round(v_, 3) for k, v_ in f.items()}))
        ends = simulate(kind, v, s, x2, 1 if kind == "otcfm" else a.n_sims)
        P = classify(ends)
        Ps.append(P)
        say("%s seed %d: %.0f s; mean probability per population %s; cells with no terminal mass %d"
            % (kind, seed, time.time() - t0, dict(zip(pops, P.mean(0).round(3))), int((P.sum(1) == 0).sum())))
    P = np.mean(Ps, 0)
    name = "snap_%s%s" % (kind, a.tag)
    np.save(os.path.join(a.featdir, name + ".npy"), P.astype(np.float32))
    # 51_fate_share.py reads the column order from <name>_log.json
    json.dump(dict(populations=pops, method=kind), open(os.path.join(a.featdir, name + "_log.json"), "w"))
    log[kind] = dict(seed_corr=[float(np.corrcoef(Ps[0][:, 0], q[:, 0])[0, 1]) for q in Ps[1:]],
                     fit=fits)
    say("wrote %s.npy %s" % (name, P.shape))
json.dump(log, open(os.path.join(a.featdir, "flow_matching%s_log.json" % a.tag), "w"), indent=1)
say("done")
