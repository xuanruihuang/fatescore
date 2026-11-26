"""Held-out time point for the methods wrapped by scTimeBench (scNODE, scIMF, PI-SDE, MIOFlow).

Each method is trained through the scTimeBench interface (its default settings, as in
100_sctimebench_fate.py) on the cells of the first and last days only, with all three days passed
as all_tps, the interface's way of keeping a time point that has no training cells
(scNODE uses physical time directly; scIMF, PI-SDE and MIOFlow map all_tps to indices). The
seed's first-day start cells are then carried by the trained model to the middle day (one
trajectory per cell), mapped into the shared space of 106_heldout_common and compared with the
real middle-day cells.

Usage: 106_heldout_stb.py <h5ad> <featdir> <dataset> <method> <out.csv> [--seeds 3 --n 2000]
       method: scNODE | scIMF | PISDE | MIOFlow
"""
import argparse
import importlib.util
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
STB = os.path.abspath(os.path.join(HERE, "..", "ext", "scTimeBench"))
spec = importlib.util.spec_from_file_location("hc", os.path.join(HERE, "106_heldout_common.py"))
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)

ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("dataset")
ap.add_argument("method", choices=["scNODE", "scIMF", "PISDE", "MIOFlow"])
ap.add_argument("out")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--set", default="")
ap.add_argument("--seed-list", default="", help="seeds to try in order (default: 0..seeds-1)")
ap.add_argument("--need", type=int, default=0, help="stop after this many successful seeds (0: all)")
ap.add_argument("--append", action="store_true", help="keep the rows already in out.csv")
a = ap.parse_args()


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


B, X, day, days = hc.build(a.h5ad, a.featdir)
pca, Z = hc.space(X, day, days)
state = B.obs["state"].values
train_m = np.isin(day, [days[0], days[2]])
say("%s %s: days %s held out %s; training cells %d" % (a.dataset, a.method, days, days[1], int(train_m.sum())))

from scTimeBench.shared.constants import ObservationColumns  # noqa: E402

if a.method == "MIOFlow":
    import tqdm as _tqdm
    import tqdm.notebook as _tqdm_nb
    _tqdm_nb.tqdm = _tqdm.tqdm
mdir = os.path.join(STB, "methods", a.method)
sys.path.insert(0, mdir)
mspec = importlib.util.spec_from_file_location("stb_" + a.method, os.path.join(mdir, "run.py"))
mod = importlib.util.module_from_spec(mspec)
mspec.loader.exec_module(mod)
cfg_path = os.path.join(STB, "configs", a.method + ".yaml")
meta0 = yaml.safe_load(open(cfg_path))["method"].get("metadata", {}) if os.path.exists(cfg_path) else {}
if a.method == "scIMF":
    meta0.update(cuda_device=0, auto_device=False, use_cuda=torch.cuda.is_available())
if a.method == "PISDE":
    meta0.update(use_cuda=torch.cuda.is_available())
for kv in filter(None, a.set.split(",")):
    k_, v_ = kv.split("=")
    meta0[k_] = yaml.safe_load(v_)

A = ad.AnnData(X=X[train_m].copy())
A.obs_names = B.obs_names[train_m]
A.obs[ObservationColumns.TIMEPOINT.value] = day[train_m]
A.obs[ObservationColumns.CELL_TYPE.value] = state[train_m]


@torch.no_grad()
def to_middle(obj, X0):
    """Carry X0 (cells x genes) to the middle day with the trained model; returns genes."""
    if a.method == "scNODE":
        m = obj.latent_ode_model.eval()
        mu, sd = m.latent_encoder(torch.FloatTensor(X0))
        seq = m.diffeq_decoder(m._sampleGaussian(mu, sd), torch.FloatTensor(days))
        return m.obs_decoder(seq).cpu().numpy()[:, 1, :]
    if a.method == "scIMF":
        m = obj.model.eval()
        x0 = torch.FloatTensor(obj.pca_model.transform(X0)).to(obj.device)
        ts = torch.FloatTensor([obj.tp_to_idx[t] for t in days]).to(obj.device)
        lat = m.forward(ts, x0, batch_size=None)[:, :, :-1].cpu().numpy()
        return obj.pca_model.inverse_transform(lat[1])
    if a.method == "PISDE":
        m, dev_, ts = PISDE
        x0 = torch.FloatTensor(X0)
        xr0 = torch.cat([x0, torch.zeros(x0.shape[0], 1)], 1).to(dev_)
        with torch.enable_grad():
            xs = m(ts, xr0.clone().requires_grad_(True))
        return xs[1][:, :-1].detach().cpu().numpy()
    if a.method == "MIOFlow":
        m = obj.model
        m.eval()
        dev_ = next(m.parameters()).device
        x0 = torch.FloatTensor(obj.pca.transform(X0)).to(dev_)
        gae = getattr(obj, "autoencoder", None) if getattr(obj, "use_gae", False) else None
        if gae is not None:
            x0 = gae.encoder(x0)
        ts = torch.FloatTensor([obj.tp_to_idx[t] for t in days]).to(dev_)
        traj = m(x0, ts, return_whole_sequence=True)
        if gae is not None:
            traj = gae.decoder(traj)
        return obj.pca.inverse_transform(traj.cpu().numpy()[1])


rows = pd.read_csv(a.out).to_dict("records") if a.append and os.path.exists(a.out) else []
seeds = [int(x) for x in a.seed_list.split(",")] if a.seed_list else list(range(a.seeds))
n_ok = 0
for seed in seeds:
    if a.need and n_ok >= a.need:
        break
    S = hc.draw(day, days, seed, a.n)
    meta = dict(meta0)
    if "seed" in meta:
        meta["seed"] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    wd = os.path.join(a.featdir, "heldout", "stb_%s_s%d" % (a.method, seed))
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)
    config = dict(output_path=wd, train_output_path=wd,
                  required_outputs=["next_timepoint_gene_expression.npy"],
                  method=dict(name=a.method, metadata=meta))
    obj = getattr(mod, a.method)(config)
    t0 = time.time()
    try:
        obj.train(A, all_tps=days, train_output_path=wd)
    except ValueError as e:
        say("seed %d diverged in training: %s" % (seed, str(e)[:160]))
        rows.append(dict(dataset=a.dataset, method=a.method, seed=seed, w2=float("nan"), n_eval=0,
                         seconds=time.time() - t0, note="diverged"))
        pd.DataFrame(rows).to_csv(a.out, index=False)
        continue
    say("seed %d trained in %.0f s" % (seed, time.time() - t0))
    if a.method == "PISDE":
        from types import SimpleNamespace
        cfg = SimpleNamespace(**torch.load(os.path.join(obj.config_dir, "config.pt"), weights_only=False))
        dev_ = torch.device("cuda:0" if cfg.use_cuda and torch.cuda.is_available() else "cpu")
        pm = mod.ForwardSDE(cfg)
        ck = torch.load(mod._select_checkpoint(cfg), map_location="cpu", weights_only=False)
        pm.load_state_dict(ck["model_state_dict"])
        pm.to(dev_).eval()
        _, ts_, _ = mod.load_data(cfg)
        ts_ = [np.float64(t) for t in ts_]
        say("PI-SDE time points %s" % ts_)
        assert len(ts_) == 3, ts_
        PISDE = (pm, dev_, ts_)
    torch.manual_seed(9600 + seed)
    X0 = X[S["first"]]
    G = np.concatenate([to_middle(obj, X0[i:i + a.chunk]) for i in range(0, len(X0), a.chunk)])
    d = hc.w2(pca.transform(G.astype(np.float32)), Z[S["eval"]])
    rows.append(dict(dataset=a.dataset, method=a.method, seed=seed, w2=d, n_eval=len(S["eval"]),
                     seconds=time.time() - t0))
    say("%s seed %d: W2 to the held-out day %.3f" % (a.method, seed, d))
    n_ok += 1
    pd.DataFrame(rows).to_csv(a.out, index=False)
say("done")
