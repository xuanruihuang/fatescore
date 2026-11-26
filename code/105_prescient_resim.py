"""PRESCIENT's own fate-bias protocol on our trained LARRY models (Yeo et al. 2021, Methods):
2,000 simulated trajectories per starting cell, endpoints classified by the majority label of their
20 nearest observed cells in the 50-component PCA space (exact neighbours on the GPU in place of
annoy's approximate ones), fate bias = (neutrophils + 1) / (neutrophils + monocytes + 2) (their
pseudocount of 1, 0.5 when no simulation lands in either), ensembled over epochs 2100-2500.
Our earlier arm (43_prescient_fate.py) used 20 simulations, 10 neighbours, one epoch and no
pseudocount. The trained checkpoints, the PCA and the cells are those saved by 43 (data.pt), so
only the read-out changes.

Scored on the 298 matched test cells of the published benchmark (Pearson r with the neutrophil share,
AUROC for share > 0.5; as 101) and on our main task (clone-level AUROC, clones with >= 3 sisters
dominated by neutrophils or monocytes; as 99). The existing nm_ file of the same arm is scored on the
same cells for comparison.

Usage: 105_prescient_resim.py <h5ad> <run_root> <featdir> <d2_labels.tsv> <fig5_dir> <arm_name> <out.npz>
       [--num-sims 2000] [--epochs 2100,2200,2300,2400,2500] [--neighbors 20] [--gpu 0]
"""
import argparse
import importlib.util
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
for k in ("h5ad", "run_root", "featdir", "labels", "fig5", "arm", "out"):
    ap.add_argument(k)
ap.add_argument("--num-sims", type=int, default=2000)
ap.add_argument("--epochs", default="2100,2200,2300,2400,2500")
ap.add_argument("--neighbors", type=int, default=20)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--batch", type=int, default=4096)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p43", os.path.join(here, "43_prescient_fate.py"))
p43 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p43)   # also lets torch.load read data.pt
dev = torch.device("cuda:%d" % a.gpu)
torch.manual_seed(a.seed)

data_path = os.path.join(a.run_root, "data.pt")
D = torch.load(data_path)
tps = np.asarray(D["tps"]); state = np.asarray(D["celltype"]).astype(str)
xp = np.zeros((len(tps), D["xp"][0].shape[1]), dtype=np.float32)
for j, d in enumerate(D["y"]):
    xp[tps == d] = D["xp"][j].numpy()
obs = ad.read_h5ad(a.h5ad, backed="r").obs
assert len(obs) == len(tps), "arm with a day filter; this script assumes none (%d vs %d)" % (len(obs), len(tps))
names = np.asarray(obs.index)
row = {n: i for i, n in enumerate(names)}

# cells: the 298 matched test cells and the main-task cells
ld = lambda n: np.load(os.path.join(a.fig5, n + ".npy"), allow_pickle=True)  # noqa: E731
tp, nmk, early, held = ld("timepoints"), ld("neutrophil_monocyte_trajectory_mask"), ld("early_cells"), ld("heldout_mask")
cfm, sx, sy = ld("clonal_fate_matrix"), ld("coordinates_x"), ld("coordinates_y")
nm2 = nmk[tp == 2]; e_nm = early[nm2]; keep = ~held
has_fate = (cfm[:, 5:7].sum(1) > 0) & nm2
hf = has_fate[nm2 & early][keep[e_nm]]
test_orig = np.where(tp == 2)[0][nm2 & early][keep[e_nm]][hf]
ytest = (cfm[has_fate, 5] / cfm[has_fate, 5:7].sum(1))[(e_nm & keep)[has_fate[nm2]]]
key = lambda x, y_: np.round(x, 2).astype(str).astype(object) + "_" + np.round(y_, 2).astype(str).astype(object)  # noqa: E731
okey = pd.Series(names, index=key(obs["SPRING-x"].values.astype(float), obs["SPRING-y"].values.astype(float)))
okey = okey[~okey.index.duplicated(keep=False)]
tcell = np.array([okey.get(k, None) for k in key(sx[test_orig].astype(float), sy[test_orig].astype(float))], dtype=object)
d2 = json.load(open(os.path.join(a.featdir, "d2_names.json")))
pos = {n: i for i, n in enumerate(d2)}
ok = np.array([c is not None and c in pos for c in tcell])
test_names, yt = list(tcell[ok]), ytest[ok]
T = pd.read_csv(a.labels, sep="\t")
assert list(T["cell"]) == d2
m = ((T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])).values
task_names = list(T.loc[m, "cell"])
sim_names = sorted(set(test_names) | set(task_names), key=lambda n: row[n])
idx = np.array([row[n] for n in sim_names])
print("test cells %d, task cells %d, simulated %d, %d sims each" % (len(test_names), len(task_names), len(idx), a.num_sims), flush=True)

# exact kNN classifier on the GPU
ref = torch.from_numpy(xp).to(dev)
codes, labs = np.unique(state, return_inverse=True)
labs_t = torch.from_numpy(labs).to(dev)
ineu, imo = list(codes).index("Neutrophil"), list(codes).index("Monocyte")


def classify(E):
    out = np.empty(E.shape[0], dtype=np.int64)
    for i in range(0, E.shape[0], a.batch):
        q = torch.from_numpy(E[i:i + a.batch]).to(dev)
        nn = torch.cdist(q, ref).topk(a.neighbors, largest=False).indices
        lab = labs_t[nn]
        cnt = torch.zeros(lab.shape[0], len(codes), device=dev).scatter_add_(1, lab, torch.ones_like(lab, dtype=torch.float))
        out[i:i + a.batch] = cnt.argmax(1).cpu().numpy()
    return out


from prescient.commands import train_model  # noqa: E402
ns = train_model.create_parser().parse_args([
    "-i", data_path, "--out_dir", os.path.join(a.run_root, "run"), "--seed", str(a.seed),
    "--train_epochs", "2500", "--pretrain_epochs", "500", "--weight_name", "uniform", "--gpu", str(a.gpu)])
ckdir = [r for r, _, f in os.walk(os.path.join(a.run_root, "run")) if any(x.startswith("train.epoch_") for x in f)][0]
bias = []
for ep in [int(e) for e in a.epochs.split(",")]:
    model, config = p43.load_model(os.path.join(ckdir, "train.epoch_%06d.pt" % ep), ns, data_path, dev)
    n_steps = int(round((max(tps) - min(tps)) / config.train_dt))
    neu = np.zeros(len(idx)); mo = np.zeros(len(idx))
    starts = torch.from_numpy(xp[idx]).float()
    blk = 50                                    # simulations run together, as repeated rows
    done = 0
    while done < a.num_sims:
        k = min(blk, a.num_sims - done)
        x = starts.repeat(k, 1).to(dev)
        for _ in range(n_steps):
            z = torch.randn_like(x) * config.train_sd
            # the drift is the gradient of the learned potential, so autograd stays on
            x = model._step(x.requires_grad_(True), dt=config.train_dt, z=z).detach()
        c = classify(x.cpu().numpy()).reshape(k, len(idx))
        neu += (c == ineu).sum(0); mo += (c == imo).sum(0)
        done += k
    b = (neu + 1) / (neu + mo + 2)
    bias.append(b)
    print("epoch %d: %d steps of dt %g, sd %g; cells with no Neu/Mo endpoint %d; mean Neu+Mo share of endpoints %.3f"
          % (ep, n_steps, config.train_dt, config.train_sd, int(((neu + mo) == 0).sum()), float(((neu + mo) / a.num_sims).mean())), flush=True)
B = np.mean(bias, 0)
sb = dict(zip(sim_names, B))
np.savez(a.out, names=np.array(sim_names), bias=B, bias_per_epoch=np.array(bias))


def old_share():
    M = np.load(os.path.join(a.featdir, "nm_%s.npy" % a.arm)).reshape(len(d2), -1)
    if M.shape[1] > 1:
        t = M[:, 0] + M[:, 1]
        return np.where(t > 0, M[:, 0] / np.where(t > 0, t, 1), 0.5)
    return M[:, 0]


old = old_share()
st_new = np.array([sb[n] for n in test_names]); st_old = old[[pos[n] for n in test_names]]
lab = (yt > 0.5).astype(int)
print("298 test cells: protocol r %.3f AUROC %.3f | earlier read-out r %.3f AUROC %.3f | single last epoch r %.3f"
      % (pearsonr(st_new, yt)[0], roc_auc_score(lab, st_new), pearsonr(st_old, yt)[0], roc_auc_score(lab, st_old),
         pearsonr(np.array([dict(zip(sim_names, bias[-1]))[n] for n in test_names]), yt)[0]))
ucl, inv = np.unique(T.loc[m, "clone"].values, return_inverse=True)
ydom = (T.loc[m].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)
cm = lambda s: np.bincount(inv, weights=s, minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))  # noqa: E731
new_task = np.array([sb[n] for n in task_names]); old_task = old[m]
print("main task (%d clones): protocol AUROC %.3f | earlier read-out %.3f"
      % (len(ucl), roc_auc_score(ydom, cm(new_task)), roc_auc_score(ydom, cm(old_task))))
