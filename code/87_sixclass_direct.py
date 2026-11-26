"""Direct scoring of the methods on the six-class task, against the connectivity
kernel (referee: is the neutrophil-monocyte decision especially favourable to a
geometric baseline?).

Every method's ten fate probabilities (feat_all/<block>.npy, columns in the
terminal-state order of cellrank_log.json) are restricted to the six classes of
the six-class task (Neutrophil, Monocyte, Baso, Meg, Mast, Erythroid),
renormalized, averaged over a clone's day-2 cells and scored without any
classifier by macro-averaged one-vs-rest AUROC over the clones whose dominant
fate is one of the six (>= 3 mature descendants). Each method is compared with
each of the 18 connectivity-kernel settings by a paired bootstrap over clones
(2,000 resamples); counts of settings with the interval above or below zero.

Usage: 87_sixclass_direct.py <featdir> <labels.tsv> <out_prefix>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FD, LAB, OUT = sys.argv[1:4]
CLASSES = ["Neutrophil", "Monocyte", "Baso", "Meg", "Mast", "Erythroid"]
order = list(json.load(open(os.path.join(FD, "cellrank_log.json")))["terminal_states"].keys())
cols = [order.index(c) for c in CLASSES]
T = pd.read_csv(LAB, sep="\t")
assert list(T["cell"]) == json.load(open(os.path.join(FD, "d2_names.json")))
sel = ((T["n_sis"] >= 3) & T["dom"].isin(CLASSES)).values
clone = T["clone"].values[sel]
y = T["dom"].values[sel]
uc, inv = np.unique(clone, return_inverse=True)
yc = pd.Series(y).groupby(inv).first().values
Y = np.array([[c == k for k in CLASSES] for c in yc], dtype=int)

methods = {"StationaryOT": "snap_statot", "Waddington-OT": "snap_wot", "moscot": "snap_moscot",
           "PRESCIENT": "snap_prescient", "Real-time kernel": "snap_rtk_ub",
           "Velocity kernel": "cr_velo", "Pseudotime kernel": "cr_pseudo",
           "Velocity + connectivity kernel": "cr_vkck"}
conn = ["cr_conn_n%d_p%d_d%d" % (n, p, d) for n in (15, 30, 60) for p in (15, 30, 50) for d in (0, 1)]


def clone_probs(block):
    P = np.load(os.path.join(FD, block + ".npy"))[sel][:, cols].astype(float)
    s = P.sum(1, keepdims=True)
    P = np.where(s > 0, P / np.where(s > 0, s, 1), 1.0 / len(CLASSES))
    return pd.DataFrame(P).groupby(inv).mean().values


def macro(Yb, Pb):
    a = [roc_auc_score(Yb[:, j], Pb[:, j]) for j in range(len(CLASSES))
         if 0 < Yb[:, j].sum() < len(Yb)]
    return float(np.mean(a))


P = {k: clone_probs(b) for k, b in list(methods.items()) + [(c, c) for c in conn]}
rng = np.random.default_rng(87)
n = len(yc)
B = [rng.integers(0, n, n) for _ in range(2000)]
boot = {k: np.array([macro(Y[i], v[i]) for i in B]) for k, v in P.items()}
rows = []
for m in methods:
    above = below = 0
    for c in conn:
        d = boot[m] - boot[c]
        lo, hi = np.percentile(d, [2.5, 97.5])
        above += lo > 0
        below += hi < 0
    rows.append(dict(method=m, macro_auroc=macro(Y, P[m]), n_conn_above=above, n_conn_below=below))
cv = [macro(Y, P[c]) for c in conn]
rows.append(dict(method="Connectivity kernel (18 settings)", macro_auroc=float(np.mean(cv)),
                 n_conn_above=np.nan, n_conn_below=np.nan, range_lo=min(cv), range_hi=max(cv)))
R = pd.DataFrame(rows).round(4)
R.to_csv(OUT + ".csv", index=False)
print("clones %d; classes %s" % (n, dict(zip(CLASSES, Y.sum(0)))))
print(R.to_string(index=False))
