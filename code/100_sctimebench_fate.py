"""Fate probabilities from the machine-learning methods wrapped by scTimeBench, on the
same task and with the same endpoint classifier as every other method.

scTimeBench (Li lab, 2026; github.com/li-lab-mcgill/scTimeBench) wraps scNODE, scIMF,
PI-SDE and MIOFlow behind one training interface, with its own default settings
(configs/<method>.yaml). Each method is trained here through that interface, on the
same object as every other method (41_ot_fate.build: this directory's highly variable
genes, log-normalised), with all sampling days.

The benchmark's generators do not keep cell identity: they simulate a population from
the first day and hand out simulated cells at the next day without regard to which
cell they started from, which suffices for distribution metrics but not for the fate
of a given cell. So each early cell is carried forward here by the trained model
itself, from that cell's own state, in the model's own space (scNODE: its VAE latent
and ODE, decoded to genes; scIMF: its SDE in its 50-component PCA; PI-SDE: its SDE in
gene space; MIOFlow: its ODE/SDE in its 50-component PCA), n_sims times for the
stochastic ones. Endpoints at the last day are mapped into the 30-component PCA of the
shared object and classified as in 94_flow_matching_fate.py: the fraction of each
endpoint's k = 20 nearest last-day cells in each terminal population, averaged over
simulations.

Training is checked without the clonal labels, as for flow matching: first-day cells
are carried to every later day and the simulated population is compared with the real
one (earth mover's distance in PCA units), against two references: the same-day floor
(two disjoint batches of real cells) and moving nothing (the earlier day itself).

Writes <featdir>/snap_<method>.npy (+ _log.json) and <featdir>/stb_<method>_log.json.

Usage: 100_sctimebench_fate.py <h5ad> <featdir> <method> [--n-sims 10 --workdir DIR]
       method: scNODE | scIMF | PISDE | MIOFlow
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
import anndata as ad
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
STB = os.path.abspath(os.path.join(HERE, "..", "ext", "scTimeBench"))
spec = importlib.util.spec_from_file_location("otfate", os.path.join(HERE, "41_ot_fate.py"))
otfate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(otfate)


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("method", choices=["scNODE", "scIMF", "PISDE", "MIOFlow"])
ap.add_argument("--n-sims", type=int, default=10)
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--fit-n", type=int, default=2000)
ap.add_argument("--workdir", default="")
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--set", default="", help="metadata overrides, key=value,key=value (a variant, chosen by fit only)")
ap.add_argument("--tag", default="", help="suffix of the output names, for variants")
a = ap.parse_args()

# ---- the shared object, its PCA, the endpoint classifier
B, d2 = otfate.build(a.h5ad, a.featdir, "time_info", "state_info")
pops = otfate.populations(a.featdir, B.obs["state"].values)
X = B.X.toarray() if hasattr(B.X, "toarray") else np.asarray(B.X)
X = X.astype(np.float32)
gmu = X.mean(0)
PCs = np.asarray(B.varm["PCs"], dtype=np.float32)
Z = np.asarray(B.obsm["X_pca"], dtype=np.float32)
proj = lambda E: (E - gmu) @ PCs  # noqa: E731
err = np.abs(proj(X[:500]) - Z[:500]).max() / np.abs(Z[:500]).max()
assert err < 1e-3, "PCA projection does not reproduce X_pca (%.2e)" % err
day = B.obs["day"].values.astype(float)
days = sorted(set(day[~np.isnan(day)].tolist()))
state = B.obs["state"].values
say("%s: %s, days %s, %d early cells, populations %s" % (a.method, X.shape, days, len(d2), pops))
pop_idx = {p: j for j, p in enumerate(pops)}
last = np.where(day == days[-1])[0]
nn_index = NearestNeighbors(n_neighbors=a.k).fit(Z[last])
lab = np.array([pop_idx.get(x, -1) for x in state[last]])


def classify(E):
    """E: cells x sims x genes at the last day -> cells x populations."""
    n, m, g = E.shape
    _, nb = nn_index.kneighbors(proj(E.reshape(n * m, g)))
    L = lab[nb]
    P = np.stack([(L == j).mean(1) for j in range(len(pops))], 1)
    return P.reshape(n, m, len(pops)).mean(1)


# ---- train through the scTimeBench wrapper
from scTimeBench.shared.constants import ObservationColumns  # noqa: E402

if a.method == "MIOFlow":
    # MIOFlow's training loops use the notebook progress bar, which needs Jupyter widgets
    import tqdm as _tqdm
    import tqdm.notebook as _tqdm_nb
    _tqdm_nb.tqdm = _tqdm.tqdm
mdir = os.path.join(STB, "methods", a.method)
sys.path.insert(0, mdir)
mspec = importlib.util.spec_from_file_location("stb_" + a.method, os.path.join(mdir, "run.py"))
mod = importlib.util.module_from_spec(mspec)
mspec.loader.exec_module(mod)
cfg_path = os.path.join(STB, "configs", a.method + ".yaml")
meta = yaml.safe_load(open(cfg_path))["method"].get("metadata", {}) if os.path.exists(cfg_path) else {}
if a.method == "scIMF":
    meta.update(cuda_device=0, auto_device=False, use_cuda=torch.cuda.is_available())
for kv in filter(None, a.set.split(",")):
    k_, v_ = kv.split("=")
    meta[k_] = yaml.safe_load(v_)
wd = a.workdir or os.path.join(a.featdir, "stb_" + a.method + a.tag)
os.makedirs(wd, exist_ok=True)
config = dict(output_path=wd, train_output_path=wd,
              required_outputs=["next_timepoint_gene_expression.npy"],
              method=dict(name=a.method, metadata=meta))
A = ad.AnnData(X=X.copy())
A.obs_names = B.obs_names
A.obs[ObservationColumns.TIMEPOINT.value] = day
A.obs[ObservationColumns.CELL_TYPE.value] = state
obj = getattr(mod, a.method)(config)
t0 = time.time()
torch.manual_seed(0)
np.random.seed(0)
obj.train(A, all_tps=days, train_output_path=wd)
say("trained in %.0f s with %s" % (time.time() - t0, meta))


# ---- carry given cells forward with the trained model; returns sims x later days x cells x genes
@torch.no_grad()
def forward_chunk(X0):
    if a.method == "scNODE":
        m = obj.latent_ode_model.eval()
        mu, sd = m.latent_encoder(torch.FloatTensor(X0))
        out = []
        for _ in range(a.n_sims):
            seq = m.diffeq_decoder(m._sampleGaussian(mu, sd), torch.FloatTensor(days))
            out.append(m.obs_decoder(seq).cpu().numpy()[:, 1:, :].transpose(1, 0, 2))
        return np.stack(out)
    if a.method == "scIMF":
        m = obj.model.eval()
        x0 = torch.FloatTensor(obj.pca_model.transform(X0)).to(obj.device)
        ts = torch.FloatTensor(list(range(len(days)))).to(obj.device)
        out = []
        for _ in range(a.n_sims):
            lat = m.forward(ts, x0, batch_size=None)[:, :, :-1].cpu().numpy()[1:]
            out.append(np.stack([obj.pca_model.inverse_transform(l_) for l_ in lat]))
        return np.stack(out)
    if a.method == "PISDE":
        # the drift is the gradient of a learned potential, taken with autograd inside the model
        m = PISDE_MODEL
        x0 = torch.FloatTensor(X0)
        xr0 = torch.cat([x0, torch.zeros(x0.shape[0], 1)], 1).to(PISDE_DEV)
        ts = [np.float64(t) for t in PISDE_TS]
        out = []
        for _ in range(a.n_sims):
            with torch.enable_grad():
                xs = m(ts, xr0.clone().requires_grad_(True))
            out.append(np.stack([xs[i][:, :-1].detach().cpu().numpy() for i in range(1, len(ts))]))
            del xs
        return np.stack(out)
    if a.method == "MIOFlow":
        # as MIOFlow.eval.generate_points, but from the given cells instead of a random sample
        m = obj.model
        m.eval()
        dev = next(m.parameters()).device
        x0 = torch.FloatTensor(obj.pca.transform(X0)).to(dev)
        gae = getattr(obj, "autoencoder", None) if getattr(obj, "use_gae", False) else None
        if gae is not None:
            x0 = gae.encoder(x0)
        ts = torch.FloatTensor([obj.tp_to_idx[t] for t in days]).to(dev)
        out = []
        for _ in range(a.n_sims):
            traj = m(x0, ts, return_whole_sequence=True)
            if gae is not None:
                traj = gae.decoder(traj)
            traj = traj.cpu().numpy()[1:]
            out.append(np.stack([obj.pca.inverse_transform(t_) for t_ in traj]))
        return np.stack(out)


if a.method == "PISDE":
    from types import SimpleNamespace
    cfg = SimpleNamespace(**torch.load(os.path.join(obj.config_dir, "config.pt"), weights_only=False))
    PISDE_DEV = torch.device("cuda:0" if cfg.use_cuda and torch.cuda.is_available() else "cpu")
    PISDE_MODEL = mod.ForwardSDE(cfg)
    ck = torch.load(mod._select_checkpoint(cfg), map_location="cpu", weights_only=False)
    PISDE_MODEL.load_state_dict(ck["model_state_dict"])
    PISDE_MODEL.to(PISDE_DEV).eval()
    _, PISDE_TS, _ = mod.load_data(cfg)


def forward(X0):
    parts = [forward_chunk(X0[i:i + a.chunk]) for i in range(0, len(X0), a.chunk)]
    return np.concatenate(parts, axis=2)


# ---- fit check (no clonal labels): first-day cells carried to each later day
def w2(P_, Q_):
    import ot
    M = ot.dist(P_, Q_)
    return float(np.sqrt(ot.emd2(np.full(len(P_), 1 / len(P_)), np.full(len(Q_), 1 / len(Q_)), M,
                                 numItermax=2_000_000)))


fit_rng = np.random.default_rng(940)
fit_idx = {}
for d in days:
    pool = np.where(day == d)[0]
    fit_idx[d] = fit_rng.choice(pool, size=min(2 * a.fit_n, len(pool)), replace=False)
half = {d: len(fit_idx[d]) // 2 for d in days}
fit_ref = {}
for d0, d1 in zip(days[:-1], days[1:]):
    fit_ref["day%g_vs_day%g" % (d0, d1)] = w2(Z[fit_idx[d0][:half[d0]]], Z[fit_idx[d1][:half[d1]]])
for d in days[1:]:
    fit_ref["day%g_vs_day%g" % (d, d)] = w2(Z[fit_idx[d][:half[d]]], Z[fit_idx[d][half[d]:2 * half[d]]])
sims = forward(X[fit_idx[days[0]][:half[days[0]]]])[0]  # one simulation: later days x cells x genes
fit = {"sim_day%g_vs_day%g" % (d, d): w2(proj(sims[i]), Z[fit_idx[d][:half[d]]])
       for i, d in enumerate(days[1:])}
say("fit references (W2, PCA units): %s" % {k: round(v, 3) for k, v in fit_ref.items()})
say("fit of %s: %s" % (a.method, {k: round(v, 3) for k, v in fit.items()}))

# ---- fate of the early cells
d2i = np.array([B.obs_names.get_loc(n) for n in d2]) if isinstance(d2[0], str) else np.asarray(d2)
E = forward(X[d2i])[:, -1]  # sims x cells x genes at the last day
P = classify(E.transpose(1, 0, 2))
say("mean probability per population %s; cells with no terminal mass %d"
    % (dict(zip(pops, P.mean(0).round(3))), int((P.sum(1) == 0).sum())))
name = "snap_%s%s" % (a.method.lower(), a.tag)
np.save(os.path.join(a.featdir, name + ".npy"), P.astype(np.float32))
json.dump(dict(populations=pops, method=a.method), open(os.path.join(a.featdir, name + "_log.json"), "w"))
json.dump(dict(method=a.method, metadata=meta, n_sims=a.n_sims, k=a.k, fit_reference=fit_ref, fit=fit,
               train_seconds=time.time() - t0, n_early=int(len(d2i))),
          open(os.path.join(a.featdir, "stb_%s%s_log.json" % (a.method, a.tag)), "w"), indent=1)
say("done")
