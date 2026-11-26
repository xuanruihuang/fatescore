"""USB (unbalanced Schrodinger bridge; Ying et al., ICML 2026; github.com/Sirin6/USB, commit 0cad29b)
under this analysis' protocol. USB's own module (utils.py: USB network, UOT plans, plan sampler)
is imported from its clone; the repository has no licence, so none of its code is copied here.

Training follows its Weinreb script (scripts/experiments/fast_mouse.py): 5 x 256 networks for
velocity, score and birth-death rate; unbalanced OT plans between consecutive days with mini-batch
UOT (chunks of 2,000); as many plan samples per interval as cells; batch 2,048; Adam 1e-3 with
cosine decay; 9,000 steps; time = index of the sampling day (0, 1, 2). nu = 1e-4 for the full fit
and 1e-3 for the held-out fit, as in its two Weinreb scripts. delta (the growth scale) is a
distance scale; their 15 was set for their 50-component space, whose median pairwise distance is
1.75 times ours on LARRY, so delta is chosen per data set from their documented values {10, 15}
by distribution fit only (mean W2 over the later days), before any clonal score is computed.

Simulation: Euler-Maruyama with their step (1/100 per day index), drift v + s, noise nu, and
birth-death jumps from the rate network (a particle splits with probability k dt if k > 0, dies
with probability |k| dt if k < 0), every particle keeping the index of the early cell it descends
from. Fate rule, fixed before scoring: at the last day each surviving particle is classified by its
20 nearest annotated cells; in each simulation an early cell takes the majority label over its
surviving descendants (one-hot); the probabilities average this over simulations; an early cell
with no surviving descendant in any simulation keeps zero mass (scored 0.5, as in 100). Variant
"_nojump": the same drift and noise with jumps switched off, endpoint fractions averaged as in 100.

Modes:
  fate    <h5ad> <featdir>                      -> <featdir>/snap_usb.npy, snap_usb_nojump.npy (+ logs)
  heldout <h5ad> <featdir> <dataset> <out.csv>  -> W2 of the predicted middle day, 3 seeds (as 106)

Usage: 110_usb.py fate ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all [--delta 15]
       110_usb.py heldout ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all larry ../out/heldout/larry_usb.csv --delta 10
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
USB_SRC = os.path.abspath(os.path.join(HERE, "..", "ext", "usb_repo", "src"))
sys.path.insert(0, USB_SRC)
import utils as usb  # noqa: E402  (USB's own module)

spec = importlib.util.spec_from_file_location("c110", os.path.join(HERE, "110_common.py"))
c110 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c110)
say = c110.say

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["fate", "heldout"])
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("dataset", nargs="?", default="")
ap.add_argument("out", nargs="?", default="")
ap.add_argument("--delta", type=float, nargs="+", default=[10.0, 15.0])
ap.add_argument("--steps", type=int, default=9000)
ap.add_argument("--batch", type=int, default=2048)
ap.add_argument("--n-sims", type=int, default=10)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max-particles", type=int, default=3_000_000)
a = ap.parse_args()
dev = torch.device(a.device)


def train(Xs, t_train, delta, nu, seed):
    """USB training as in its fast_mouse.py, on frames Xs at times t_train."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, g0, g1, _ = usb.compute_uot_plans(Xs, t_train, delta=delta, use_mini_batch_uot=True,
                                         chunk_size=2000, cuda=True)
    n_per = int(sum(len(x) for x in Xs))
    parts = []
    for k in range(len(t_train) - 1):
        x0, x1, i0, i1 = usb.sample_from_ot_plan(g0[k], Xs[k], Xs[k + 1], n_per)
        ratio = (g1[k][i0, i1] / g0[k][i0, i1]).reshape(-1, 1)
        m1 = np.log(1e-8 + ratio)
        n_ = len(x0)
        parts.append([torch.tensor(np.asarray(x0), dtype=torch.float32), torch.tensor(np.asarray(x1), dtype=torch.float32),
                      torch.zeros((n_, 1)), torch.tensor(m1, dtype=torch.float32),
                      torch.full((n_, 1), float(t_train[k])), torch.full((n_, 1), float(t_train[k + 1] - t_train[k]))])
    del g0, g1
    tensors = [torch.cat([p[j] for p in parts]).to(dev) for j in range(6)]
    loader = DataLoader(TensorDataset(*tensors), batch_size=a.batch, shuffle=True, drop_last=True)
    dim = Xs[0].shape[1]
    model = usb.USB([dim + 1, 256, 256, 256, 256, 256], nu=nu).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=1e-5)
    step = 0
    model.train()
    while step < a.steps:
        for bx0, bx1, bm0, bm1, bts, bdt in loader:
            opt.zero_grad()
            rt = torch.rand((bx0.shape[0], 1), device=dev)
            real_t = bdt * rt + bts
            xt, vt, kt, eps, mt = model.sample_comditional_flow(bx0, bx1, bm0, bm1, rt)
            kt, vt = kt / bdt, vt / bdt
            v, s, k_ = model(xt, real_t)
            w = torch.exp(mt)
            bridge = 2 * torch.sqrt(rt * (1 - rt)) / (nu + 1e-8)
            loss = (torch.mean((v - vt) ** 2 * w) + torch.mean((bridge * s + eps) ** 2 * w)
                    + torch.mean((k_ - kt) ** 2 * w))
            loss.backward()
            opt.step()
            sch.step()
            step += 1
            if step % 1500 == 0:
                say("  step %d loss %.4f" % (step, loss.item()))
            if step >= a.steps:
                break
    model.eval()
    return model


@torch.no_grad()
def simulate(model, x0, t0, t1, nu, jumps=True, seed=0):
    """Particles from x0 (cells x dim) between times t0 and t1; returns (positions, root index)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    n_steps = int(round((t1 - t0) * 100))
    dt = (t1 - t0) / n_steps
    x = torch.as_tensor(x0, dtype=torch.float32, device=dev).clone()
    root = torch.arange(len(x0), device=dev)
    for i in range(n_steps):
        t = torch.full((x.shape[0], 1), t0 + i * dt, device=dev)
        inp = torch.cat([x, t], -1)
        v, s, k = model.v_net(inp), model.s_net(inp), model.k_net(inp)
        x = x + (v + s) * dt + nu * np.sqrt(dt) * torch.randn(x.shape, device=dev, generator=g)
        if jumps:
            p = torch.clamp(k.abs() * dt, max=1.0).squeeze(-1)
            ev = torch.rand(p.shape, device=dev, generator=g) < p
            kk = k.squeeze(-1)
            die = ev & (kk < 0)
            split = ev & (kk > 0) & ~die
            keep = ~die
            x_new, r_new = x[split], root[split]
            x, root = torch.cat([x[keep], x_new]), torch.cat([root[keep], r_new])
            if x.shape[0] > a.max_particles:
                raise RuntimeError("particle count exceeded %d" % a.max_particles)
            if x.shape[0] == 0:
                break
    return x.cpu().numpy(), root.cpu().numpy()


def fate_mode():
    F = c110.FateData(a.h5ad, a.featdir)
    frames, days = F.frames(), F.days
    T = list(range(len(days)))  # day index as time, as in USB's scripts
    nu = 1e-4
    fit_ref = F.fit_reference(c110.w2)
    say("fit references: %s" % {k: round(v, 3) for k, v in fit_ref.items()})
    fits, models = {}, {}
    for delta in a.delta:
        t0 = time.time()
        m = train(frames, T, delta, nu, seed=113)
        f = {}
        for j, d in enumerate(days[1:], start=1):
            P, _ = simulate(m, F.fit_start(), 0.0, float(j), nu, jumps=True, seed=940)
            f["sim_day%g_vs_day%g" % (d, d)] = c110.w2(P, F.fit_target(d)) if len(P) else float("nan")
            f["n_particles_day%g" % d] = int(len(P))
        fits[delta] = f
        models[delta] = m
        say("delta %g fit %s (%.0f s)" % (delta, {k: round(v, 3) for k, v in f.items()}, time.time() - t0))
    score = {d_: np.nanmean([v for k, v in f.items() if k.startswith("sim_")]) for d_, f in fits.items()}
    best = min(score, key=score.get)
    say("chosen delta %g by fit (mean W2 %s)" % (best, {k: round(v, 3) for k, v in score.items()}))
    m = models[best]
    X0 = F.Z[F.d2i]
    K = len(F.pops)
    P_jump = np.zeros((len(X0), K))
    P_flat = np.zeros((len(X0), K))
    n_surv = np.zeros(len(X0))
    T_end = float(T[-1])
    for s in range(a.n_sims):
        E, root = simulate(m, X0, 0.0, T_end, nu, jumps=True, seed=1000 + s)
        if len(E):
            lab = F.classify(E).argmax(1)
            for j in range(K):
                cnt = np.bincount(root[lab == j], minlength=len(X0))
                if j == 0:
                    counts = np.zeros((len(X0), K))
                counts[:, j] = cnt
            tot = counts.sum(1)
            win = counts.argmax(1)
            has = tot > 0
            P_jump[np.where(has)[0], win[has]] += 1
            n_surv += has
        E2, _ = simulate(m, X0, 0.0, T_end, nu, jumps=False, seed=2000 + s)
        P_flat += F.classify(E2)
        say("simulation %d: %d particles at the last day, %d early cells with survivors"
            % (s, len(E), int((counts.sum(1) > 0).sum()) if len(E) else 0))
    P_jump /= a.n_sims
    P_flat /= a.n_sims
    log = dict(method="USB", commit="0cad29b72e2b6c5aa0f81bc77f15d426991b316a", nu=nu, delta=best,
               delta_fits={str(k): v for k, v in fits.items()}, fit_reference=fit_ref, steps=a.steps,
               n_sims=a.n_sims, n_early=int(len(X0)), early_cells_never_surviving=int((n_surv == 0).sum()),
               rule="majority label over surviving descendants per simulation, averaged over simulations")
    F.save(a.featdir, "snap_usb", P_jump, log)
    F.save(a.featdir, "snap_usb_nojump", P_flat, dict(log, rule="jumps off; endpoint fractions averaged"))
    say("early cells never surviving %d of %d; mean probability %s"
        % (int((n_surv == 0).sum()), len(X0), dict(zip(F.pops, P_jump.mean(0).round(3)))))


def heldout_mode():
    import pandas as pd
    hc = c110.heldout_common()
    B, X, day, days = hc.build(a.h5ad, a.featdir)
    pca, Zraw = hc.space(X, day, days)
    frames = [Zraw[day == days[0]], Zraw[day == days[2]]]
    nu = 1e-3
    delta = a.delta[0]
    rows = []
    for seed in range(a.seeds):
        S = hc.draw(day, days, seed, a.n)
        t0 = time.time()
        m = train(frames, [0, 2], delta, nu, seed=9400 + seed)
        P, _ = simulate(m, Zraw[S["first"]], 0.0, 1.0, nu, jumps=True, seed=9500 + seed)
        Pn, _ = simulate(m, Zraw[S["first"]], 0.0, 1.0, nu, jumps=False, seed=9600 + seed)
        E = Zraw[S["eval"]]
        for meth, PP in (("usb", P), ("usb_nojump", Pn)):
            d = hc.w2(PP[np.random.default_rng(seed).choice(len(PP), min(len(PP), 4000), replace=False)], E) if len(PP) else float("nan")
            rows.append(dict(dataset=a.dataset, method=meth, seed=seed, w2=d, n_eval=len(E), n_pred=len(PP),
                             delta=delta, nu=nu, seconds=time.time() - t0))
            say("%s seed %d: W2 to the held-out day %.3f (%d predicted cells)" % (meth, seed, d, len(PP)))
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    say("USB %s %s; delta %s" % (a.mode, a.featdir, a.delta))
    fate_mode() if a.mode == "fate" else heldout_mode()
    say("done")
