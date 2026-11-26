"""Held-out time point for flow matching: OT-CFM and SB-CFM ([SF]2M) from torchcfm, as in
94_flow_matching_fate.py (same network, sigma 0.1, exact minibatch couplings, one global scale,
the 50,000-step setting chosen there by fit), but trained on the first and last days only (one
time interval, t in [0, 1]) in the shared space of 106_heldout_common, and integrated from the
seed's first-day start cells to the middle day's position in time (one trajectory per cell).

Usage: 106_heldout_cfm.py <h5ad> <featdir> <dataset> <out.csv> [--seeds 3 --n 2000 --steps 50000]
"""
import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("hc", os.path.join(HERE, "106_heldout_common.py"))
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)

ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("dataset")
ap.add_argument("out")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--sigma", type=float, default=0.1)
ap.add_argument("--n-int", type=int, default=100)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--models", default="otcfm,sbcfm")
a = ap.parse_args()


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


from torchcfm.conditional_flow_matching import (  # noqa: E402
    ExactOptimalTransportConditionalFlowMatcher, SchrodingerBridgeConditionalFlowMatcher)

B, X, day, days = hc.build(a.h5ad, a.featdir)
pca, Zraw = hc.space(X, day, days)
f = hc.frac(days)
train_m = np.isin(day, [days[0], days[2]])
mu = Zraw[train_m].mean(0)
sd = float(Zraw[train_m].std())
Z = (Zraw - mu) / sd
dev = torch.device(a.device)
T0 = torch.tensor(Z[day == days[0]])
T1 = torch.tensor(Z[day == days[2]])
dim = Z.shape[1]
say("%s: days %s, middle at %.3f; training cells %d + %d" % (a.dataset, days, f, len(T0), len(T1)))


class MLP(torch.nn.Module):
    def __init__(self, dim, w=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, w), torch.nn.SELU(), torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(), torch.nn.Linear(w, dim))

    def forward(self, x, t):
        return self.net(torch.cat([x, t[:, None]], 1))


def train(kind, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    v = MLP(dim).to(dev)
    s = MLP(dim).to(dev) if kind == "sbcfm" else None
    params = list(v.parameters()) + (list(s.parameters()) if s is not None else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    FM = (ExactOptimalTransportConditionalFlowMatcher(sigma=a.sigma) if kind == "otcfm"
          else SchrodingerBridgeConditionalFlowMatcher(sigma=a.sigma, ot_method="exact"))
    for it in range(a.steps):
        x0 = T0[rng.integers(0, len(T0), a.batch)].to(dev)
        x1 = T1[rng.integers(0, len(T1), a.batch)].to(dev)
        if kind == "otcfm":
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            loss = ((v(xt, t) - ut) ** 2).mean()
        else:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
            lam = FM.compute_lambda(t)
            loss = ((v(xt, t) - ut) ** 2).mean() + ((lam[:, None] * s(xt, t) + eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 10000 == 0:
            say("  %s seed %d step %d loss %.4f" % (kind, seed, it + 1, loss.item()))
    return v, s


@torch.no_grad()
def to_middle(kind, v, s, x):
    n_steps = int(round(a.n_int * f))
    dt = f / n_steps
    z = x.clone()
    for i in range(n_steps):
        t = torch.full((z.shape[0],), i * dt, device=dev)
        if kind == "otcfm":
            z = z + v(z, t) * dt
        else:
            z = z + (v(z, t) + 0.5 * a.sigma ** 2 * s(z, t)) * dt + a.sigma * np.sqrt(dt) * torch.randn_like(z)
    return z.cpu().numpy()


rows = []
for seed in range(a.seeds):
    S = hc.draw(day, days, seed, a.n)
    E = Zraw[S["eval"]]
    x = torch.tensor(Z[S["first"]]).to(dev)
    for kind in a.models.split(","):
        t0 = time.time()
        v, s = train(kind, 9400 + seed)
        torch.manual_seed(9500 + seed)
        P = to_middle(kind, v, s, x) * sd + mu
        d = hc.w2(P, E)
        rows.append(dict(dataset=a.dataset, method=kind, seed=seed, w2=d, n_eval=len(E),
                         seconds=time.time() - t0))
        say("%s seed %d: W2 to the held-out day %.3f (%.0f s)" % (kind, seed, d, time.time() - t0))
        pd.DataFrame(rows).to_csv(a.out, index=False)
say("done")
