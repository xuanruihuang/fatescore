"""Score CoSpar's state-only fate predictions (107_cospar_state.py) exactly as the other methods.

LARRY: the 567-clone main task of 99_calibrated_table.py (clones with >= 3 sisters dominated by
neutrophils or monocytes; neutrophil share from nm_<name>.npy). CellTag-multi: the task of
97_fate_score.py (clones with >= 3 mature descendants; share of the positive population among the
terminal populations). Reference: the connectivity kernel at n30 p30 d1 (cr_conn_n30_p30_d1 on
LARRY, cr_conn_all_n30_p30_d1 on CellTag-multi), band over its 18 settings. Gain with a paired
bootstrap over clones (2,000 resamples, seed 97) and the minimum detectable difference, and the
increment over the kernel (cross-fitted logistic models), both from the fatescore tool, which
reproduces 99 and 104.

Usage: 107_cospar_score.py <larry_featdir> <larry_labels.tsv> <ctm_featdir> <ctr_featdir> <out.csv>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tool"))
from fatescore import gain_over_kernel, increment_over_kernel  # noqa: E402

FALL, LAB, CTM, CTR, OUT = sys.argv[1:6]
rows = []

# LARRY (as 99)
T = pd.read_csv(LAB, sep="\t")
m = (T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])
im = np.where(m.values)[0]
ucl, inv = np.unique(T.loc[m, "clone"].values, return_inverse=True)
y = (T.loc[m].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)


def larry(name):
    M = np.load(os.path.join(FALL, "nm_" + name + ".npy"))
    if M.ndim == 2 and M.shape[1] > 1:
        s = M[:, 0] + M[:, 1]
        M = np.where(s > 0, M[:, 0] / np.where(s > 0, s, 1), 0.5)
    s = M.ravel()
    return np.bincount(inv, weights=s[im], minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))


def score(ds, yy, ref, band, preds):
    for name, c in preds.items():
        g = gain_over_kernel(c, ref, yy)
        inc = increment_over_kernel(c, ref, yy)
        rows.append(dict(dataset=ds, method=name, auroc=roc_auc_score(yy, c), kernel_auroc=roc_auc_score(yy, ref),
                         kernel_min=min(band), kernel_max=max(band), gain=g["gain"], lo=g["lo"], hi=g["hi"],
                         mdd=g["mdd"], verdict=g["verdict"], increment=inc["increment"], inc_lo=inc["lo"],
                         inc_hi=inc["hi"], adds=inc["adds"], n_clones=len(yy), n_pos=int(yy.sum())))
        print(rows[-1], flush=True)


names = sorted(os.path.basename(p)[3:-4] for p in glob.glob(os.path.join(FALL, "nm_snap_cospar_state*.npy")))
band = [roc_auc_score(y, larry(os.path.basename(p)[3:-4]))
        for p in sorted(glob.glob(os.path.join(FALL, "nm_cr_conn_n*_p*_d*.npy")))]
score("LARRY", y, larry("cr_conn_n30_p30_d1"), band, {n: larry(n) for n in names})

# CellTag-multi (as 97)
for ds, fd, pos in (("CellTag-multi haematopoiesis", CTM, "Mono"), ("CellTag-multi reprogramming", CTR, "reprogramming")):
    L = pd.read_csv(os.path.join(fd, "labels.tsv"), sep="\t")
    early = json.load(open(os.path.join(fd, "d2_names.json")))
    assert list(L["cell"]) == early
    task = (L["n_mature"] >= 3).values
    it = np.where(task)[0]
    uc, iv = np.unique(L.loc[task, "clone"].values, return_inverse=True)
    yy = (L.loc[task].groupby("clone")["dom"].first().reindex(uc).values == pos).astype(int)

    def ct(name):
        M = np.load(os.path.join(fd, name + ".npy"))
        pops = json.load(open(os.path.join(fd, name + "_log.json")))["populations"]
        tot = M.sum(1)
        s = np.where(tot > 0, M[:, pops.index(pos)] / np.where(tot > 0, tot, 1), 0.5)
        return np.bincount(iv, weights=s[it], minlength=len(uc)) / np.bincount(iv, minlength=len(uc))

    nm = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(fd, "snap_cospar_state*.npy")))
    band = [roc_auc_score(yy, ct(os.path.basename(p)[:-4]))
            for p in sorted(glob.glob(os.path.join(fd, "cr_conn_all_n*_p*_d*.npy")))]
    score(ds, yy, ct("cr_conn_all_n30_p30_d1"), band, {n: ct(n) for n in nm})

R = pd.DataFrame(rows)
R.to_csv(OUT, index=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
