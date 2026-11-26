"""Fate probabilities from PRESCIENT, on the same task as every other arm.

PRESCIENT is the fourth snapshot-dynamics method in this comparison and the only
one that does not produce a coupling: it fits a potential function whose gradient
drives a diffusion, and a cell's fate is read off by simulating it forward and
seeing where it lands. So the fate probability here is the fraction of a cell's
simulated trajectories whose endpoint is classified into each mature population,
with the package's own nearest-neighbour classifier doing the classification.

Three things about the package shape this script and are deviations worth
stating.

1. Its process_data command only reads csv and raises NotImplementedError for
   h5ad, and it needs a UMAP it then never uses for training. The torch file it
   writes is a plain dictionary, so it is built here directly, with the first
   two principal components in place of the UMAP.
2. Its train_init indexes the timepoint list by the first timepoint's *value*
   (y[config.start_t] with start_t = y[0]), so it only works when the timepoints
   are 0, 1, 2, ... The sampling days are relabelled accordingly, which also
   fixes the simulation length: train_dt 0.1 over a unit interval is ten steps
   per sampling interval.
3. Its growth-rate estimator needs birth and death gene-set files that are in
   the authors' repository but not in the installed package, so growth weights
   are uniform here. That is the package's own no-growth setting (the weights
   enter as exp((t - t0) * w) with w = 0), and it is recorded in the log.

The simulation loop is written here rather than taken from
simulate_trajectories, which samples a fixed number of random cells per
simulation; a per-cell fate probability needs every early cell started the same
number of times. The step itself is the package's model._step.

Usage:
  python 43_prescient_fate.py <h5ad> <featdir> --out-dir <dir> [--device cpu]
"""
import argparse
import copy
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import sklearn.decomposition
import sklearn.preprocessing
import torch

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


# The package's data file holds numpy arrays and an sklearn PCA, and torch 2.6
# made weights_only=True the default for torch.load, so the package cannot read
# the file it defines. The file in question is the one this script writes a few
# lines below, not a downloaded checkpoint, so loading it in full is loading our
# own data. The default is restored only for the duration of this script.
_torch_load = torch.load


def _load_full(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _load_full


def populations(featdir, state):
    p = os.path.join(featdir, "cellrank_log.json")
    if os.path.exists(p):
        st = json.load(open(p)).get("terminal_states")
        if st:
            return list(st.keys())
    return [s for s in MATURE if (state == s).sum() >= 30]


def build_pt(a):
    hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
    d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
    A = ad.read_h5ad(a.h5ad)
    keep = [g for g in hvg if g in A.var_names]
    assert len(keep) == len(hvg), (len(keep), len(hvg))
    # a duplicated cell name makes obs_names.get_loc return a mask instead of
    # a position, which would corrupt the early-cell index with no error
    assert A.obs_names.is_unique, "duplicated cell names"
    A = A[:, keep].copy()
    # The same two day filters as 41_ot_fate.py, and the same distinction:
    # --keep-days drops the other sampling days before the scaling and the
    # PCA, so the representation never sees them (the experiment that did not
    # sample that day), while --couple-days drops them afterwards, leaving the
    # full representation and only shortening the sequence the model is
    # trained on. Every early cell has to survive either way.
    kd = [float(x) for x in a.keep_days.split(",") if x.strip()]
    cd = [float(x) for x in a.couple_days.split(",") if x.strip()]
    assert not (kd and cd), "--keep-days and --couple-days answer different " \
        "questions; pass one at a time"
    dayv = pd.to_numeric(A.obs[a.time_col], errors="coerce").astype(float)
    if kd:
        m = dayv.isin(kd).to_numpy()
        say("keep-days %s before the PCA: %d of %d cells"
            % (kd, int(m.sum()), m.size))
        A = A[m].copy()
    if not a.no_renormalise:
        sc.pp.normalize_total(A, target_sum=None)
    sc.pp.log1p(A)
    X = A.X
    X = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float32)

    # The scaler and the PCA are fitted here, before --couple-days takes
    # anything out, so that reading keeps the full representation and removes
    # only the timepoints the model is trained on. --keep-days has already
    # removed its cells above, so it changes the representation too.
    scaler = sklearn.preprocessing.StandardScaler()
    pca = sklearn.decomposition.PCA(n_components=a.num_pcs)
    xs = scaler.fit_transform(X)
    xp = pca.fit_transform(xs).astype(np.float32)
    say("scaled %s, principal components %s, variance kept %.3f"
        % (xs.shape, xp.shape, float(pca.explained_variance_ratio_.sum())))

    if cd:
        m = pd.to_numeric(A.obs[a.time_col], errors="coerce").isin(cd).to_numpy()
        say("couple-days %s after the PCA: %d of %d cells"
            % (cd, int(m.sum()), m.size))
        A, X, xs, xp = A[m].copy(), X[m], xs[m], xp[m]
    gone = [n for n in d2_names if n not in set(A.obs_names)]
    assert not gone, "%d early cells were dropped by a day filter" % len(gone)

    days = pd.to_numeric(A.obs[a.time_col], errors="coerce").astype(float).values
    uniq = np.array(sorted(set(days[~np.isnan(days)].tolist())))
    tps = np.searchsorted(uniq, days)          # 0, 1, 2, ... as the package needs
    state = A.obs[a.state_col].astype(str).values
    say("sampling days %s relabelled %s" % (uniq.tolist(),
                                            sorted(set(tps.tolist()))))

    y = sorted(set(tps.tolist()))
    pt = dict(
        data=X, genes=np.asarray(keep, dtype=object), celltype=state, tps=tps,
        x=[torch.from_numpy(xs[tps == d]).float() for d in y],
        xp=[torch.from_numpy(xp[tps == d]).float() for d in y],
        xu=[torch.from_numpy(xp[tps == d][:, :2]).float() for d in y],
        y=y, pca=pca, um=None,
        # no growth: the weights enter the loss as exp((t - t0) * w)
        w=[np.zeros(int((tps == d).sum()), dtype=np.float32) for d in y])
    if a.growth_from_wot:
        # The package's own growth estimator needs KEGG birth and death gene
        # lists that are in the authors' repository but not in the installed
        # package. This second reading takes the growth rate Waddington-OT
        # estimated for the same cells on the same arm instead, so that
        # PRESCIENT is not the only method run without the growth model its
        # loss expects. train_init uses only the first timepoint's vector, and
        # uses it as exp((t - t0) * w), so w is the log of that rate per unit
        # relabelled time.
        g = ad.read_h5ad(a.growth_from_wot, backed="r").obs["g1"]
        first = pd.Index(np.asarray(A.obs_names)[tps == y[0]])
        gv = pd.Series(g).reindex(first)
        miss = int(gv.isna().sum())
        assert miss == 0, "%d of %d first-timepoint cells have no wot growth " \
            "rate" % (miss, len(first))
        w0 = np.log(np.clip(gv.values.astype(np.float64), 1e-3, None))
        pt["w"][0] = w0.astype(np.float32)
        say("growth from %s: log rate over %d cells, mean %.4f, sd %.4f, "
            "range %.3f to %.3f" % (os.path.basename(a.growth_from_wot),
                                    w0.size, float(w0.mean()), float(w0.std()),
                                    float(w0.min()), float(w0.max())))
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "data.pt")
    torch.save(pt, path)
    say("wrote %s" % path)
    d2 = np.array([A.obs_names.get_loc(n) for n in d2_names])
    return path, xp, tps, state, d2, A.obs_names


def train(a, data_path):
    from prescient.commands import train_model
    import prescient.train as ptrain
    ns = train_model.create_parser().parse_args([
        "-i", data_path, "--out_dir", os.path.join(a.out_dir, "run"),
        "--seed", str(a.seed), "--train_epochs", str(a.train_epochs),
        "--pretrain_epochs", str(a.pretrain_epochs),
        "--weight_name", "uniform"] + (["--no-cuda"] if a.device == "cpu"
                                       else ["--gpu", str(a.gpu)]))
    say("training: %d pretrain epochs, %d train epochs, k_dim %d, layers %d"
        % (ns.pretrain_epochs, ns.train_epochs, ns.k_dim, ns.layers))
    ptrain.run(ns, train_model.train_init)
    # the run directory is named from the hyperparameters; find the last epoch
    base = os.path.join(a.out_dir, "run")
    cands = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.startswith("train.epoch_") and f.endswith(".pt"):
                cands.append(os.path.join(root, f))
    assert cands, "no checkpoint under %s" % base
    ckpt = sorted(cands)[-1]
    say("trained; using checkpoint %s" % ckpt)
    return ckpt, ns


def load_model(ckpt, ns, data_path, device):
    from prescient.train.model import AutoGenerator
    from prescient.commands.train_model import train_init
    _, _, _, config = train_init(ns)
    sd = torch.load(ckpt, map_location=device)
    model = AutoGenerator(config)
    model.load_state_dict(sd["model_state_dict"])
    model.to(device)
    return model, config


def simulate(model, config, xp, d2, device, num_sims, num_steps, batch):
    """Every early cell started num_sims times and stepped forward with the
    package's own step, to the last sampling day."""
    out = np.zeros((d2.size, num_sims, xp.shape[1]), dtype=np.float32)
    starts = torch.from_numpy(xp[d2]).float()
    for s in range(num_sims):
        parts = []
        for i in range(0, d2.size, batch):
            x = starts[i:i + batch].to(device)
            for _ in range(num_steps):
                z = (torch.randn(x.shape[0], x.shape[1], device=device)
                     * config.train_sd)
                x = model._step(x.float(), dt=config.train_dt, z=z)
            parts.append(x.detach().cpu().numpy())
        out[:, s, :] = np.concatenate(parts, 0)
        say("  simulation %d/%d done" % (s + 1, num_sims))
    return out


def classify(xp, state, ends, pops, n_trees=10, n_neighbors=10):
    """The package's classifier: an annoy index over the observed cells in
    principal-component space, the majority annotation among a simulated
    endpoint's ten nearest observed cells."""
    from annoy import AnnoyIndex
    t = AnnoyIndex(xp.shape[1], "euclidean")
    for i in range(xp.shape[0]):
        t.add_item(i, xp[i])
    t.build(n_trees)
    code = {p: j for j, p in enumerate(pops)}
    n_cells, n_sims = ends.shape[0], ends.shape[1]
    counts = np.zeros((n_cells, len(pops)), dtype=np.float64)
    for i in range(n_cells):
        for s in range(n_sims):
            nn = t.get_nns_by_vector(ends[i, s], n_neighbors)
            lab = pd.Series(state[nn]).value_counts()
            top = str(lab.index[0])
            if top in code:
                counts[i, code[top]] += 1.0
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("featdir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="prescient")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-pcs", type=int, default=50)
    ap.add_argument("--num-sims", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--train-epochs", type=int, default=2500)
    ap.add_argument("--pretrain-epochs", type=int, default=500)
    ap.add_argument("--time-col", default="time_info")
    ap.add_argument("--state-col", default="state_info")
    ap.add_argument("--keep-days", default="",
                    help="sampling days to keep BEFORE the PCA, i.e. the "
                         "experiment that never sampled the others, e.g. 2,6")
    ap.add_argument("--couple-days", default="",
                    help="sampling days to keep AFTER the PCA, i.e. the full "
                         "representation with a shorter trained sequence")
    ap.add_argument("--growth-from-wot", default="",
                    help="a wot transport map for the first day pair; "
                         "its g1 column becomes the growth input")
    ap.add_argument("--no-renormalise", action="store_true",
                    help="arm deposited as normalised counts")
    a = ap.parse_args()

    data_path, xp, tps, state, d2, names = build_pt(a)
    pops = populations(a.featdir, state)
    say("%d early cells, populations %s" % (d2.size, pops))

    ckpt, ns = train(a, data_path)
    device = torch.device("cpu" if a.device == "cpu"
                          else "cuda:%d" % a.gpu)
    model, config = load_model(ckpt, ns, data_path, device)
    n_steps = int(round((max(tps) - min(tps)) / config.train_dt))
    say("simulating %d steps of dt %g with noise sd %g, %d simulations per cell"
        % (n_steps, config.train_dt, config.train_sd, a.num_sims))
    ends = simulate(model, config, xp, d2, device, a.num_sims, n_steps, a.batch)
    counts = classify(xp, state, ends, pops)
    land = counts.sum(1)
    say("simulations landing in a mature population: %.3f of all"
        % float(land.sum() / (d2.size * a.num_sims)))
    P = counts / np.where(land[:, None] == 0, 1.0, land[:, None])
    out = os.path.join(a.featdir, "snap_%s.npy" % a.tag)
    np.save(out, P.astype(np.float32))
    say("saved %s %s; mean of the largest probability %.3f"
        % (out, P.shape, float(P.max(1).mean())))
    json.dump(dict(method="prescient", checkpoint=ckpt, num_sims=a.num_sims,
                   num_steps=n_steps, train_dt=float(config.train_dt),
                   train_sd=float(config.train_sd), num_pcs=a.num_pcs,
                   train_epochs=a.train_epochs,
                   growth=(a.growth_from_wot or "uniform (w = 0)"),
                   populations=pops, n_cells_in_object=int(xp.shape[0]),
                   keep_days=[float(x) for x in a.keep_days.split(",")
                              if x.strip()],
                   couple_days=[float(x) for x in a.couple_days.split(",")
                                if x.strip()],
                   fraction_landing_mature=float(
                       land.sum() / (d2.size * a.num_sims)),
                   all_converged=True, saved=True),
              open(os.path.join(a.featdir, "snap_%s_log.json" % a.tag), "w"),
              indent=1)
    say("done")


if __name__ == "__main__":
    main()
