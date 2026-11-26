"""Score the real-time kernel family of 111 on a CellTag-multi data set, as 97_fate_score.py does
(task: early cells whose clone has >= 3 mature cells; clone-level AUROC for a clone dominated by the
positive population), against the data set's reference kernel cr_conn_all_n30_p30_d1.
(i) Grid on all clones (descriptive). (ii) Split validation as 111_mix_score.py: 20 stratified
halvings, w in {0} + grid chosen on one half (w = 0 is the reference kernel), gain on the other.

Usage: 111_mix_score_ct.py <mixdir> <featdir> <pos> <out.csv>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MIX, FD, POS, OUT = sys.argv[1:5]
W = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
T = pd.read_csv(os.path.join(FD, "labels.tsv"), sep="\t")
task = (T["n_mature"] >= 3).values
it = np.where(task)[0]
ucl, inv = np.unique(T.loc[task, "clone"].values, return_inverse=True)
y = (T.loc[task].groupby("clone")["dom"].first().reindex(ucl).values == POS).astype(int)


def clone(npy, logf):
    P = np.load(npy)
    pops = json.load(open(logf))["populations"]
    tot = P.sum(1)
    s = np.where(tot > 0, P[:, pops.index(POS)] / np.where(tot > 0, tot, 1), 0.5)
    return np.bincount(inv, weights=s[it], minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))


ref = clone(os.path.join(FD, "cr_conn_all_n30_p30_d1.npy"), os.path.join(FD, "cr_conn_all_n30_p30_d1_log.json"))
C = {w: clone(os.path.join(MIX, "mix_RTK_w%s.npy" % w), os.path.join(MIX, "mix_RTK_w%s_log.json" % w))
     for w in W if os.path.exists(os.path.join(MIX, "mix_RTK_w%s.npy" % w))}
print("reference kernel AUROC %.4f; %d clones, %d positive" % (roc_auc_score(y, ref), len(y), y.sum()))


def gain(a, b, idx, n_boot, seed=97):
    yy, aa, bb = y[idx], a[idx], b[idx]
    Bs = np.random.default_rng(seed).integers(0, len(idx), size=(n_boot, len(idx)))
    d = [roc_auc_score(yy[s], aa[s]) - roc_auc_score(yy[s], bb[s]) for s in Bs if yy[s].min() != yy[s].max()]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return roc_auc_score(yy, aa) - roc_auc_score(yy, bb), lo, hi


allidx = np.arange(len(y))
rows = []
for w, c in C.items():
    g, lo, hi = gain(c, ref, allidx, 2000)
    rows.append(dict(part="grid_all_clones", family="RTK", w=w, auroc=roc_auc_score(y, c), gain=g, lo=lo, hi=hi))
cands = {0.0: ref, **{w: c for w, c in C.items() if w <= 0.5}}
res = []
for r in range(20):
    tr, te = next(StratifiedKFold(2, shuffle=True, random_state=500 + r).split(allidx, y))
    best = max(cands, key=lambda w: roc_auc_score(y[tr], cands[w][tr]))
    g, lo, hi = gain(cands[best], ref, te, 1000, seed=700 + r)
    res.append((best, g, lo, hi))
R = np.array(res)
rows.append(dict(part="split_validated", family="RTK", w=float(R[:, 0].mean()), auroc=np.nan,
                 gain=float(R[:, 1].mean()), lo=float(np.percentile(R[:, 1], 5)), hi=float(np.percentile(R[:, 1], 95)),
                 share_above=float(np.mean(R[:, 2] > 0)), chose_w0=float(np.mean(R[:, 0] == 0))))
D = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(D.round(4).to_string(index=False))
D.round(5).to_csv(OUT, index=False)
