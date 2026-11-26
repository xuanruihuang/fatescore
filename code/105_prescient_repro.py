"""PRESCIENT on LARRY as close to the published fate-bias setting as our data allow
(Yeo et al. 2021; gifford-lab/prescient depr/scripts/weinreb-fate.sh and depr/weinreb.py, read 11-26).

Matched: physical days 2/4/6 (40 steps of dt 0.1 from day 2 to day 6), train_sd 0.1, lr 0.005,
gradient clip 0.1, batch 10%, tau 1e-6, 2 layers x 400 units, 500 pretraining + 2500 training epochs,
checkpoints every 100; 50 principal components of scaled log-normalised highly variable genes; all
cells used for training; growth enters the loss as exp((t - 2) * w) for day-2 cells. Read-out as their
evaluate_fate: 2,000 trajectories per starting cell, endpoint label = majority of its 20 nearest
observed cells (ties -> Other), bias = (Neu + 1) / (Neu + Mo + 2), mean over epochs 2100-2500.
Growth modes: none (w = 0), wot (log of Waddington-OT's g1 for the same cells, as 43), kegg (their
get_growth_weights / compute_growth: smoothed mean z-scores of birth and death gene sets; needs the
two gene-set csv files with a gene_symbol column).
Not matched (no data here): Weinreb's normalised matrix and cell-cycle-filtered genes on all
130,887 cells; we use our 49,302-cell object and its highly variable genes. Exact neighbours on the
GPU replace annoy.

Scores: Pearson r / AUROC on the 298 matched test cells (as 101); main task clone AUROC and its gain
over the connectivity kernel (nm_cr_conn_n30_p30_d1) with a paired bootstrap over clones (2,000,
seed 97), as 99.

Usage: 105_prescient_repro.py <h5ad> <featdir> <d2_labels.tsv> <fig5_dir> <out_dir> --seed 1
       --growth none|wot|kegg [--wot-tmap f] [--birth csv --death csv] [--gpu 0] [--num-sims 2000]
       [--stage all|train|eval]
"""
import argparse
import copy
import importlib.util
import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import sklearn.decomposition
import sklearn.preprocessing
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
for k in ("h5ad", "featdir", "labels", "fig5", "out_dir"):
    ap.add_argument(k)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--growth", default="none", choices=["none", "wot", "kegg"])
ap.add_argument("--wot-tmap", default="")
ap.add_argument("--birth", default="")
ap.add_argument("--death", default="")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--num-sims", type=int, default=2000)
ap.add_argument("--epochs-eval", default="2100,2200,2300,2400,2500")
ap.add_argument("--stage", default="all", choices=["all", "train", "eval"])
ap.add_argument("--batch", type=int, default=4096)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p43", os.path.join(here, "43_prescient_fate.py"))
p43 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p43)   # torch.load patch for our own data.pt
os.makedirs(a.out_dir, exist_ok=True)
say = p43.say
dev = torch.device("cuda:%d" % a.gpu)


def kegg_growth(xs_df, xp, birth_csv, death_csv, beta=0.1, L0=0.3, L=1.1, k=0.001):
    """Port of prescient.utils.get_growth_weights + compute_growth (repo commit 50971c7), including its
    in-place smoothing (prev_score and cur_score become one array after the first sweep)."""
    from annoy import AnnoyIndex
    genes = list(xs_df.columns)
    b = [g for g in pd.read_csv(birth_csv, index_col=0)["gene_symbol"].unique() if g in genes]
    d = [g for g in pd.read_csv(death_csv, index_col=0)["gene_symbol"].unique() if g in genes]
    b, d = [g for g in b if g not in d], [g for g in d if g not in b]
    say("kegg growth: %d birth and %d death genes among the variable genes" % (len(b), len(d)))
    ay = AnnoyIndex(xp.shape[1], "euclidean")
    for i in range(xp.shape[0]):
        ay.add_item(i, xp[i])
    ay.build(10)
    nbr = [ay.get_nns_by_item(i, 20) for i in range(xp.shape[0])]

    def smooth(raw):
        prev = raw.copy(); cur = np.zeros(prev.shape)
        for _ in range(5):
            for i in range(len(prev)):
                xn = prev[nbr[i]]
                cur[i] = beta * xn[0] + (1 - beta) * xn[1:].mean(axis=0)
            prev = cur
        return cur
    bs, ds = xs_df[b].mean(axis=1).values, xs_df[d].mean(axis=1).values
    kb, kd = np.log(k) / np.min(bs), np.log(k) / np.min(ds)
    bb = L0 + L / (1 + np.exp(-kb * smooth(bs)))
    dd = L0 + L / (1 + np.exp(-kd * smooth(ds)))
    return bb - dd


def build():
    hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
    A = ad.read_h5ad(a.h5ad)
    assert A.obs_names.is_unique
    A = A[:, [g for g in hvg if g in A.var_names]].copy()
    sc.pp.normalize_total(A, target_sum=None); sc.pp.log1p(A)
    X = np.asarray(A.X.todense() if sp.issparse(A.X) else A.X, dtype=np.float32)
    xs = sklearn.preprocessing.StandardScaler().fit_transform(X)
    xp = sklearn.decomposition.PCA(n_components=50, random_state=0).fit_transform(xs).astype(np.float32)
    days = pd.to_numeric(A.obs["time_info"], errors="coerce").astype(float).values
    y = sorted(set(days.tolist()))
    assert y == [2.0, 4.0, 6.0], y
    w0 = np.zeros(int((days == 2).sum()), dtype=np.float32)
    if a.growth == "wot":
        g = ad.read_h5ad(a.wot_tmap, backed="r").obs["g1"]
        gv = pd.Series(g).reindex(pd.Index(A.obs_names[days == 2]))
        assert gv.isna().sum() == 0
        # g1 is the wot growth over the 2-day interval of that map; per day as their w
        w0 = (np.log(np.clip(gv.values.astype(np.float64), 1e-3, None)) / 2.0).astype(np.float32)
    elif a.growth == "kegg":
        g = kegg_growth(pd.DataFrame(xs, columns=list(A.var_names)), xp, a.birth, a.death)
        w0 = g[days == 2].astype(np.float32)
    say("growth %s on day-2 cells: mean %.3f sd %.3f" % (a.growth, float(w0.mean()), float(w0.std())))
    pt = dict(xp=[torch.from_numpy(xp[days == d]) for d in y], y=y,
              w=[w0] + [np.zeros(int((days == d).sum()), dtype=np.float32) for d in y[1:]],
              celltype=A.obs["state_info"].astype(str).values, tps=days, names=np.asarray(A.obs_names))
    path = os.path.join(a.out_dir, "data_%s_seed%d.pt" % (a.growth, a.seed))
    torch.save(pt, path)
    return path, xp, days, pt["celltype"], np.asarray(A.obs_names)


def make_ns(data_path):
    from prescient.commands import train_model
    return train_model.create_parser().parse_args([
        "-i", data_path, "--out_dir", os.path.join(a.out_dir, "run_%s" % a.growth), "--seed", str(a.seed),
        "--weight_name", a.growth, "--k_dim", "400", "--layers", "2", "--train_sd", "0.1", "--train_lr", "0.005",
        "--train_clip", "0.1", "--train_batch", "0.1", "--train_tau", "1e-6", "--train_epochs", "2500",
        "--pretrain_epochs", "500", "--save", "100", "--gpu", str(a.gpu)])


def init_task(args):
    """As depr/weinreb.py train_fate/load_data: start_t index 0, train_t [1, 2], physical days."""
    from prescient.commands.train_model import init_config
    D = torch.load(args.data_path)
    x, y = D["xp"], D["y"]
    aa = copy.copy(args); aa.weight = args.weight_name
    name = "{weight}-{activation}_{layers}_{k_dim}-{train_tau}".format(**aa.__dict__)
    aa.out_dir = os.path.join(args.out_dir, name, "seed_%d" % args.seed)
    config = init_config(aa)
    config.x_dim = x[0].shape[-1]; config.t = y[-1] - y[0]
    config.start_t = 0; config.train_t = [1, 2]
    w0 = D["w"][0]
    w = {(y[0], yy): torch.from_numpy(np.exp((yy - y[0]) * w0)) for yy in y[1:]}
    return x, y, w, config


def ckdir(ns):
    name = "%s-softplus_2_400-1e-06" % a.growth
    return os.path.join(ns.out_dir, name, "seed_%d" % a.seed)


data_path, xp, days, state, names = build()
ns = make_ns(data_path)
if a.stage in ("all", "train"):
    import prescient.train as ptrain
    t0 = time.time()
    ptrain.run(ns, init_task)
    say("trained in %.0f s" % (time.time() - t0))
if a.stage == "train":
    raise SystemExit(0)

# ---- read-out (their evaluate_fate) on the test and main-task cells
row = {n: i for i, n in enumerate(names)}
ld = lambda n: np.load(os.path.join(a.fig5, n + ".npy"), allow_pickle=True)  # noqa: E731
tp, nmk, early, held = ld("timepoints"), ld("neutrophil_monocyte_trajectory_mask"), ld("early_cells"), ld("heldout_mask")
cfm, sx, sy = ld("clonal_fate_matrix"), ld("coordinates_x"), ld("coordinates_y")
nm2 = nmk[tp == 2]; e_nm = early[nm2]; keep = ~held
has_fate = (cfm[:, 5:7].sum(1) > 0) & nm2
hf = has_fate[nm2 & early][keep[e_nm]]
test_orig = np.where(tp == 2)[0][nm2 & early][keep[e_nm]][hf]
ytest = (cfm[has_fate, 5] / cfm[has_fate, 5:7].sum(1))[(e_nm & keep)[has_fate[nm2]]]
obs = ad.read_h5ad(a.h5ad, backed="r").obs
key = lambda x, y_: np.round(x, 2).astype(str).astype(object) + "_" + np.round(y_, 2).astype(str).astype(object)  # noqa: E731
okey = pd.Series(np.asarray(obs.index), index=key(obs["SPRING-x"].values.astype(float), obs["SPRING-y"].values.astype(float)))
okey = okey[~okey.index.duplicated(keep=False)]
tcell = np.array([okey.get(k, None) for k in key(sx[test_orig].astype(float), sy[test_orig].astype(float))], dtype=object)
d2 = json.load(open(os.path.join(a.featdir, "d2_names.json")))
pos = {n: i for i, n in enumerate(d2)}
ok = np.array([c is not None and c in pos for c in tcell])
test_names, yt = list(tcell[ok]), ytest[ok]
T = pd.read_csv(a.labels, sep="\t"); assert list(T["cell"]) == d2
m = ((T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])).values
task_names = list(T.loc[m, "cell"])
sim = sorted(set(test_names) | set(task_names), key=lambda n: row[n])
idx = np.array([row[n] for n in sim])

ref = torch.from_numpy(xp).to(dev)
codes, labs = np.unique(state, return_inverse=True)
labs_t = torch.from_numpy(labs).to(dev)
ineu, imo = list(codes).index("Neutrophil"), list(codes).index("Monocyte")


def classify(E):
    out = np.empty(E.shape[0], dtype=np.int64)
    for i in range(0, E.shape[0], a.batch):
        q = torch.from_numpy(E[i:i + a.batch]).to(dev)
        lab = labs_t[torch.cdist(q, ref).topk(20, largest=False).indices]
        cnt = torch.zeros(lab.shape[0], len(codes), device=dev).scatter_add_(1, lab, torch.ones_like(lab, dtype=torch.float))
        top2 = cnt.topk(2, dim=1)
        lab_top = top2.indices[:, 0].clone()
        lab_top[top2.values[:, 0] == top2.values[:, 1]] = -1      # tie -> Other, as theirs
        out[i:i + a.batch] = lab_top.cpu().numpy()
    return out


from prescient.train.model import AutoGenerator  # noqa: E402
_, _, _, config = init_task(ns)
torch.manual_seed(0)
bias = []
for ep in [int(e) for e in a.epochs_eval.split(",")]:
    sd = torch.load(os.path.join(ckdir(ns), "train.epoch_%06d.pt" % ep), map_location=dev)
    model = AutoGenerator(config); model.load_state_dict(sd["model_state_dict"]); model.to(dev)
    n_steps = int(round(config.t / config.train_dt))
    neu = np.zeros(len(idx)); mo = np.zeros(len(idx))
    starts = torch.from_numpy(xp[idx]).float(); done = 0
    while done < a.num_sims:
        kk = min(50, a.num_sims - done)
        x = starts.repeat(kk, 1).to(dev)
        for _ in range(n_steps):
            z = torch.randn_like(x) * config.train_sd
            x = model._step(x.requires_grad_(True), dt=config.train_dt, z=z).detach()
        c = classify(x.cpu().numpy()).reshape(kk, len(idx))
        neu += (c == ineu).sum(0); mo += (c == imo).sum(0); done += kk
    bias.append((neu + 1) / (neu + mo + 2))
    say("epoch %d: %d steps, sd %g; cells with no Neu/Mo endpoint %d of %d; mean Neu+Mo share %.3f"
        % (ep, n_steps, config.train_sd, int(((neu + mo) == 0).sum()), len(idx), float(((neu + mo) / a.num_sims).mean())))
B = np.mean(bias, 0); sb = dict(zip(sim, B))
st = np.array([sb[n] for n in test_names]); lab = (yt > 0.5).astype(int)
r_t, auc_t = pearsonr(st, yt)[0], roc_auc_score(lab, st)
ucl, inv = np.unique(T.loc[m, "clone"].values, return_inverse=True)
ydom = (T.loc[m].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)
cm = lambda s: np.bincount(inv, weights=s, minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))  # noqa: E731
c_new = cm(np.array([sb[n] for n in task_names]))
K = np.load(os.path.join(a.featdir, "nm_cr_conn_n30_p30_d1.npy")).reshape(len(d2), -1)
kshare = K[:, 0] if K.shape[1] == 1 else np.where(K[:, :2].sum(1) > 0, K[:, 0] / np.where(K[:, :2].sum(1) > 0, K[:, :2].sum(1), 1), 0.5)
c_k = cm(kshare[m])
rng = np.random.default_rng(97); Bt = rng.integers(0, len(ucl), size=(2000, len(ucl)))
dd = [roc_auc_score(ydom[b], c_new[b]) - roc_auc_score(ydom[b], c_k[b]) for b in Bt if ydom[b].min() != ydom[b].max()]
res = dict(growth=a.growth, seed=a.seed, num_sims=a.num_sims, r_test=r_t, auroc_test=auc_t, n_test=len(yt),
           auroc_task=roc_auc_score(ydom, c_new), auroc_kernel=roc_auc_score(ydom, c_k),
           gain=roc_auc_score(ydom, c_new) - roc_auc_score(ydom, c_k), gain_lo=float(np.percentile(dd, 2.5)),
           gain_hi=float(np.percentile(dd, 97.5)), n_clones=len(ucl))
json.dump(res, open(os.path.join(a.out_dir, "result_%s_seed%d.json" % (a.growth, a.seed)), "w"), indent=1)
np.savez(os.path.join(a.out_dir, "bias_%s_seed%d.npz" % (a.growth, a.seed)), names=np.array(sim), bias=B, per_epoch=np.array(bias))
say("RESULT " + json.dumps(res))
