"""PRESCIENT trained and read out as published on the full Weinreb et al. in vitro data
(input from 105_prescient_weinreb_prep.py). Training and read-out exactly as 105_prescient_repro.py
(their weinreb-fate.sh settings; their evaluate_fate read-out), but on all 130,887 cells.

Scores: Pearson r / AUROC on all 335 published test cells (Weinreb Fig. 5 construction, indices
in the full data's order) and on the 298 of them in our object; main task on our object (clones
with >= 3 sisters dominated by neutrophils or monocytes; cells mapped to the full data by SPRING
coordinates as in 101): clone AUROC and gain over our connectivity kernel with a paired bootstrap
over clones (2,000, seed 97).

Usage: 105_prescient_full.py <data_full_GROWTH.pt> <h5ad_ours> <featdir> <d2_labels.tsv> <fig5_dir> <out_dir>
       --growth none|kegg --seed 1 [--gpu 0] [--num-sims 2000] [--stage all|train|eval]
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
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
for k in ("data_pt", "h5ad", "featdir", "labels", "fig5", "out_dir"):
    ap.add_argument(k)
ap.add_argument("--growth", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--num-sims", type=int, default=2000)
ap.add_argument("--epochs-eval", default="2100,2200,2300,2400,2500")
ap.add_argument("--stage", default="all", choices=["all", "train", "eval"])
ap.add_argument("--batch", type=int, default=4096)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p43", os.path.join(here, "43_prescient_fate.py"))
p43 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p43)
say = p43.say
dev = torch.device("cuda:%d" % a.gpu)
os.makedirs(a.out_dir, exist_ok=True)

from prescient.commands import train_model  # noqa: E402
ns = train_model.create_parser().parse_args([
    "-i", a.data_pt, "--out_dir", os.path.join(a.out_dir, "run_full_%s" % a.growth), "--seed", str(a.seed),
    "--weight_name", a.growth, "--k_dim", "400", "--layers", "2", "--train_sd", "0.1", "--train_lr", "0.005",
    "--train_clip", "0.1", "--train_batch", "0.1", "--train_tau", "1e-6", "--train_epochs", "2500",
    "--pretrain_epochs", "500", "--save", "100", "--gpu", str(a.gpu)])


def init_task(args):
    from prescient.commands.train_model import init_config
    D = torch.load(args.data_path)
    x, y = D["xp"], D["y"]
    aa = copy.copy(args); aa.weight = args.weight_name
    aa.out_dir = os.path.join(args.out_dir, "{weight}-{activation}_{layers}_{k_dim}-{train_tau}".format(**aa.__dict__),
                              "seed_%d" % args.seed)
    config = init_config(aa)
    config.x_dim = x[0].shape[-1]; config.t = y[-1] - y[0]
    config.start_t = 0; config.train_t = [1, 2]
    w0 = D["w"][0]
    return x, y, {(y[0], yy): torch.from_numpy(np.exp((yy - y[0]) * w0)) for yy in y[1:]}, config


if a.stage in ("all", "train"):
    import prescient.train as ptrain
    t0 = time.time(); ptrain.run(ns, init_task); say("trained in %.0f s" % (time.time() - t0))
if a.stage == "train":
    raise SystemExit(0)

D = torch.load(a.data_pt)
tps = np.asarray(D["tps"]); state = np.asarray(D["celltype"]).astype(str)
xp = np.zeros((len(tps), D["xp"][0].shape[1]), dtype=np.float32)
for j, d in enumerate(D["y"]):
    xp[tps == d] = D["xp"][j].numpy()
key = lambda x, y_: np.round(np.asarray(x, float), 2).astype(str).astype(object) + "_" + np.round(np.asarray(y_, float), 2).astype(str).astype(object)  # noqa: E731
full_key = key(D["spring_x"], D["spring_y"])

ld = lambda n: np.load(os.path.join(a.fig5, n + ".npy"), allow_pickle=True)  # noqa: E731
tp, nmk, early, held = ld("timepoints"), ld("neutrophil_monocyte_trajectory_mask"), ld("early_cells"), ld("heldout_mask")
cfm = ld("clonal_fate_matrix")
assert len(tp) == len(tps) and np.all(tp.astype(float) == tps), "fig5 files and the full data differ in order"
nm2 = nmk[tp == 2]; e_nm = early[nm2]; keep = ~held
has_fate = (cfm[:, 5:7].sum(1) > 0) & nm2
hf = has_fate[nm2 & early][keep[e_nm]]
test_rows = np.where(tp == 2)[0][nm2 & early][keep[e_nm]][hf]
ytest = (cfm[has_fate, 5] / cfm[has_fate, 5:7].sum(1))[(e_nm & keep)[has_fate[nm2]]]

obs = ad.read_h5ad(a.h5ad, backed="r").obs
ours = pd.Series(np.asarray(obs.index), index=key(obs["SPRING-x"].values, obs["SPRING-y"].values))
ours = ours[~ours.index.duplicated(keep=False)]
fk = pd.Series(np.arange(len(full_key)), index=full_key); fk = fk[~fk.index.duplicated(keep=False)]
d2 = json.load(open(os.path.join(a.featdir, "d2_names.json")))
pos = {n: i for i, n in enumerate(d2)}
in_ours = np.array([(ours.get(full_key[r], None) in pos) for r in test_rows])
T = pd.read_csv(a.labels, sep="\t"); assert list(T["cell"]) == d2
m = ((T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])).values
okey = pd.Series(key(obs["SPRING-x"].values, obs["SPRING-y"].values), index=np.asarray(obs.index))
task_cells = list(T.loc[m, "cell"])
task_rows = np.array([fk.get(okey[c], -1) for c in task_cells])
say("test cells %d (%d in our object); main-task cells %d, mapped to the full data %d"
    % (len(test_rows), in_ours.sum(), len(task_cells), (task_rows >= 0).sum()))
sim_rows = np.unique(np.concatenate([test_rows, task_rows[task_rows >= 0]]))

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
        t2 = cnt.topk(2, dim=1); lt = t2.indices[:, 0].clone(); lt[t2.values[:, 0] == t2.values[:, 1]] = -1
        out[i:i + a.batch] = lt.cpu().numpy()
    return out


from prescient.train.model import AutoGenerator  # noqa: E402
_, _, _, config = init_task(ns)
ck = os.path.join(ns.out_dir, "%s-softplus_2_400-1e-06" % a.growth, "seed_%d" % a.seed)
torch.manual_seed(0)
bias = []
for ep in [int(e) for e in a.epochs_eval.split(",")]:
    sd = torch.load(os.path.join(ck, "train.epoch_%06d.pt" % ep), map_location=dev)
    model = AutoGenerator(config); model.load_state_dict(sd["model_state_dict"]); model.to(dev)
    n_steps = int(round(config.t / config.train_dt))
    neu = np.zeros(len(sim_rows)); mo = np.zeros(len(sim_rows)); done = 0
    starts = torch.from_numpy(xp[sim_rows]).float()
    while done < a.num_sims:
        kk = min(50, a.num_sims - done)
        x = starts.repeat(kk, 1).to(dev)
        for _ in range(n_steps):
            x = model._step(x.requires_grad_(True), dt=config.train_dt, z=torch.randn_like(x) * config.train_sd).detach()
        c = classify(x.cpu().numpy()).reshape(kk, len(sim_rows))
        neu += (c == ineu).sum(0); mo += (c == imo).sum(0); done += kk
    bias.append((neu + 1) / (neu + mo + 2))
    say("epoch %d: %d steps, sd %g; cells with no Neu/Mo endpoint %d of %d; mean Neu+Mo share %.3f"
        % (ep, n_steps, config.train_sd, int(((neu + mo) == 0).sum()), len(sim_rows), float(((neu + mo) / a.num_sims).mean())))
B = dict(zip(sim_rows.tolist(), np.mean(bias, 0)))
st = np.array([B[r] for r in test_rows]); lab = (ytest > 0.5).astype(int)
res = dict(growth=a.growth, seed=a.seed, r_335=pearsonr(st, ytest)[0], auroc_335=roc_auc_score(lab, st),
           r_298=pearsonr(st[in_ours], ytest[in_ours])[0], auroc_298=roc_auc_score(lab[in_ours], st[in_ours]))
mk = task_rows >= 0
if mk.all():
    ucl, inv = np.unique(T.loc[m, "clone"].values, return_inverse=True)
    ydom = (T.loc[m].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)
    cm = lambda s: np.bincount(inv, weights=s, minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))  # noqa: E731
    cn = cm(np.array([B[r] for r in task_rows]))
    K = np.load(os.path.join(a.featdir, "nm_cr_conn_n30_p30_d1.npy")).reshape(len(d2), -1)[:, 0]
    ckn = cm(K[m])
    rng = np.random.default_rng(97); Bt = rng.integers(0, len(ucl), size=(2000, len(ucl)))
    dd = [roc_auc_score(ydom[b], cn[b]) - roc_auc_score(ydom[b], ckn[b]) for b in Bt if ydom[b].min() != ydom[b].max()]
    res.update(auroc_task=roc_auc_score(ydom, cn), auroc_kernel=roc_auc_score(ydom, ckn),
               gain=roc_auc_score(ydom, cn) - roc_auc_score(ydom, ckn),
               gain_lo=float(np.percentile(dd, 2.5)), gain_hi=float(np.percentile(dd, 97.5)), n_clones=len(ucl))
else:
    res.update(task_unmapped=int((~mk).sum()))
json.dump(res, open(os.path.join(a.out_dir, "result_full_%s_seed%d.json" % (a.growth, a.seed)), "w"), indent=1)
say("RESULT " + json.dumps(res))
