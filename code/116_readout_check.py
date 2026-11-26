"""Does the read-out of the learned models explain why they fall below the walk?

Learned generative models are scored by carrying each early cell forward and classifying its
simulated endpoints; a cell none of whose endpoints reaches a scored fate gets 0.5. Here every learned
model is rescored on the clones whose task cells all reach a scored fate (no 0.5 fill), against the walk
(reference connectivity kernel) on exactly the same clones, with a paired bootstrap over clones (2,000,
seed 97). LARRY: neutrophil vs monocyte share (as 99); CellTag-multi: positive population over total
mass (as 97).

Usage: 116_readout_check.py <out_dir> <out.csv>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUTD, CSV = sys.argv[1:3]
KEYS = {"snap_prescient": "PRESCIENT", "snap_otcfm_g50k": "OT-CFM", "snap_sbcfm_g50k": "SB-CFM",
        "snap_scnode": "scNODE", "snap_scimf": "scIMF", "snap_pisde": "PI-SDE", "snap_mioflow": "MIOFlow",
        "snap_usb": "USB", "snap_pace_fit": "PACE"}


def boot(y, a, b):
    rng = np.random.default_rng(97)
    B = rng.integers(0, len(y), size=(2000, len(y)))
    d = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in B if y[i].min() != y[i].max()]
    return np.percentile(d, [2.5, 97.5])


def clone_mean(v, iv, n):
    return np.bincount(iv, weights=v, minlength=n) / np.bincount(iv, minlength=n)


rows = []
# ---------- LARRY (as 99: shares of Neu among Neu+Mo; task = clones with >=3 sisters, Neu/Mo dominated)
FD = os.path.join(OUTD, "feat_all")
T = pd.read_csv(os.path.join(OUTD, "d2_labels.tsv"), sep="\t")
m = (T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])
im = np.where(m.values)[0]
ucl, iv = np.unique(T.loc[m, "clone"].values, return_inverse=True)
y = (T.loc[m].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)
W = np.load(os.path.join(FD, "nm_cr_conn_n30_p30_d1.npy"))
if W.ndim == 2 and W.shape[1] > 1:
    s_ = W[:, 0] + W[:, 1]
    W = np.where(s_ > 0, W[:, 0] / np.where(s_ > 0, s_, 1), 0.5)
W = W.ravel()[im]
for key, name in KEYS.items():
    p = os.path.join(FD, key + ".npy")
    if not os.path.exists(p):
        continue
    M = np.load(p)
    pops = json.load(open(os.path.join(FD, key + "_log.json")))["populations"]
    ni = [i for i, q in enumerate(pops) if q.lower().startswith("neu")][0]
    mi = [i for i, q in enumerate(pops) if q.lower().startswith("mono")][0]
    M = M[im]
    mass = M[:, ni] + M[:, mi]
    share = np.where(mass > 0, M[:, ni] / np.where(mass > 0, mass, 1), 0.5)
    ok_cell = mass > 0
    ok_clone = np.bincount(iv, weights=(~ok_cell).astype(float), minlength=len(ucl)) == 0
    cm, cw = clone_mean(share, iv, len(ucl)), clone_mean(W, iv, len(ucl))
    g_all = roc_auc_score(y, cm) - roc_auc_score(y, cw)
    k = ok_clone
    lo, hi = boot(y[k], cm[k], cw[k]) if k.sum() > 20 and y[k].min() != y[k].max() else (np.nan, np.nan)
    rows.append(dict(dataset="LARRY", method=name, cells_no_mass=int((~ok_cell).sum()), n_clones=len(ucl),
                     n_kept=int(k.sum()), gain_all=g_all,
                     auroc_kept=roc_auc_score(y[k], cm[k]), walk_kept=roc_auc_score(y[k], cw[k]),
                     gain_kept=roc_auc_score(y[k], cm[k]) - roc_auc_score(y[k], cw[k]), lo=lo, hi=hi))

# ---------- CellTag-multi (as 97)
for ds, fdn, pos in (("CellTag-multi haematopoiesis", "feat_ctm", "Mono"),
                     ("CellTag-multi reprogramming", "feat_ctr", "reprogramming")):
    FD = os.path.join(OUTD, fdn)
    L = pd.read_csv(os.path.join(FD, "labels.tsv"), sep="\t")
    task = (L["n_mature"] >= 3).values
    it = np.where(task)[0]
    uc, ivc = np.unique(L.loc[task, "clone"].values, return_inverse=True)
    yy = (L.loc[task].groupby("clone")["dom"].first().reindex(uc).values == pos).astype(int)

    def share_mass(key):
        M = np.load(os.path.join(FD, key + ".npy"))
        pops = json.load(open(os.path.join(FD, key + "_log.json")))["populations"]
        tot = M.sum(1)
        return np.where(tot > 0, M[:, pops.index(pos)] / np.where(tot > 0, tot, 1), 0.5)[it], (tot > 0)[it]

    ws, _ = share_mass("cr_conn_all_n30_p30_d1")
    cw = clone_mean(ws, ivc, len(uc))
    for key, name in KEYS.items():
        if not os.path.exists(os.path.join(FD, key + ".npy")):
            continue
        sh, ok_cell = share_mass(key)
        ok_clone = np.bincount(ivc, weights=(~ok_cell).astype(float), minlength=len(uc)) == 0
        cm = clone_mean(sh, ivc, len(uc))
        k = ok_clone
        lo, hi = boot(yy[k], cm[k], cw[k]) if k.sum() > 20 and yy[k].min() != yy[k].max() else (np.nan, np.nan)
        rows.append(dict(dataset=ds, method=name, cells_no_mass=int((~ok_cell).sum()), n_clones=len(uc),
                         n_kept=int(k.sum()), gain_all=roc_auc_score(yy, cm) - roc_auc_score(yy, cw),
                         auroc_kept=roc_auc_score(yy[k], cm[k]), walk_kept=roc_auc_score(yy[k], cw[k]),
                         gain_kept=roc_auc_score(yy[k], cm[k]) - roc_auc_score(yy[k], cw[k]), lo=lo, hi=hi))
R = pd.DataFrame(rows)
R.round(4).to_csv(CSV, index=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
print("pairs with an interval above zero on fully-reached clones:", int((R.lo > 0).sum()), "of", len(R))
