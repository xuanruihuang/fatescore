"""Score the USB and PACE arms (110_usb.py, 110_pace.py) exactly as the other methods: clone-level
AUROC, gain over the connectivity kernel with the paired clone bootstrap (2,000 resamples, seed 97;
LARRY as 99_calibrated_table.py, CellTag-multi as 97_fate_score.py), minimum detectable difference,
and increment over the kernel (104_incremental.py logic, via tool/fatescore). For LARRY it also
writes nm_<arm>.npy (neutrophil share, 0.5 without mass, as 51_fate_share.py) so that
101_weinreb_test_set.py scores the arms on the 335 published test cells.

Usage: 110_score.py <larry_featdir> <larry_labels.tsv> <ctm_featdir> <ctr_featdir> <out.csv> [arms...]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tool"))
from fatescore import gain_over_kernel, increment_over_kernel  # noqa: E402

FALL, LAB, CTM, CTR, OUT = sys.argv[1:6]
ARMS = sys.argv[6:] or ["snap_usb", "snap_usb_nojump", "snap_pace"]
rows = []


def share(fd, name, pos, neg=None):
    P = np.load(os.path.join(fd, name + ".npy"))
    pops = json.load(open(os.path.join(fd, name + "_log.json")))["populations"]
    if neg is None:
        tot = P.sum(1)
        num = P[:, pops.index(pos)]
    else:
        num = P[:, pops.index(pos)]
        tot = num + P[:, pops.index(neg)]
    return np.where(tot > 0, num / np.where(tot > 0, tot, 1), 0.5)


def add(ds, arm, c, ref, y):
    g = gain_over_kernel(c, ref, y)
    inc = increment_over_kernel(c, ref, y)
    rows.append(dict(dataset=ds, method=arm, auroc_clone=roc_auc_score(y, c), kernel_auroc=roc_auc_score(y, ref),
                     gain=g["gain"], lo=g["lo"], hi=g["hi"], mdd=g["mdd"], separated=g["verdict"],
                     increment=inc["increment"], inc_lo=inc["lo"], inc_hi=inc["hi"], n_clones=len(y),
                     n_pos=int(y.sum())))
    print(rows[-1], flush=True)


# LARRY (99's task: clones with >= 3 sisters dominated by neutrophils or monocytes)
T = pd.read_csv(LAB, sep="\t")
m = (T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])
im = np.where(m.values)[0]
ucl, inv = np.unique(T.loc[m, "clone"].values, return_inverse=True)
y = (T.loc[m].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)
cm = lambda s: np.bincount(inv, weights=s[im], minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))  # noqa: E731
K = np.load(os.path.join(FALL, "nm_cr_conn_n30_p30_d1.npy"))
K = K.ravel() if K.ndim == 1 or K.shape[1] == 1 else np.where(K[:, :2].sum(1) > 0, K[:, 0] / np.maximum(K[:, :2].sum(1), 1e-12), 0.5)
for arm in ARMS:
    if not os.path.exists(os.path.join(FALL, arm + ".npy")):
        continue
    s = share(FALL, arm, "Neutrophil", "Monocyte")
    np.save(os.path.join(FALL, "nm_" + arm + ".npy"), s.astype(np.float32))
    add("LARRY", arm, cm(s), cm(K), y)

# CellTag-multi (97's task: early cells whose clone has >= 3 mature descendants)
for ds, fd, pos in (("CellTag-multi haematopoiesis", CTM, "Mono"), ("CellTag-multi reprogramming", CTR, "reprogramming")):
    L = pd.read_csv(os.path.join(fd, "labels.tsv"), sep="\t")
    task = (L["n_mature"] >= 3).values
    it = np.where(task)[0]
    uc, iv = np.unique(L.loc[task, "clone"].values, return_inverse=True)
    yy = (L.loc[task].groupby("clone")["dom"].first().reindex(uc).values == pos).astype(int)
    cmc = lambda s: np.bincount(iv, weights=s[it], minlength=len(uc)) / np.bincount(iv, minlength=len(uc))  # noqa: E731
    ref = cmc(share(fd, "cr_conn_all_n30_p30_d1", pos))
    for arm in ARMS:
        if os.path.exists(os.path.join(fd, arm + ".npy")):
            add(ds, arm, cmc(share(fd, arm, pos)), ref, yy)

R = pd.DataFrame(rows)
R.round(4).to_csv(OUT, index=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
