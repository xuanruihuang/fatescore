"""Score the 111 mixtures on the LARRY main task (567 clones, as 99_calibrated_table.py).

Reference: CK_w0, the connectivity kernel of the same rebuilt pipeline (w = 0); the paper's reference
kernel (cr_conn_n30_p30_d1 in the feature directory) is printed alongside.
(i) The full grid on all clones: AUROC and gain over CK_w0 with a paired bootstrap over clones
    (2,000 resamples, seed 97). Descriptive only: w is not chosen here.
(ii) Split validation: 20 stratified halvings of the clones; on one half choose w in {0} + grid per
    family by AUROC (w = 0 is the kernel itself); on the other half, the gain of the chosen mixture
    over CK_w0 with a paired bootstrap (1,000 resamples). Mean gain, mean chosen w, and the share of
    splits whose interval lies above zero.

Usage: 111_mix_score.py <mixdir> <larry_featdir_all> <larry_labels.tsv> <out.csv>
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MIX, FALL, LAB, OUT = sys.argv[1:5]
W = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
FAM = ["VK", "PK", "VKd", "VKp", "RTK"]

T = pd.read_csv(LAB, sep="\t")
m = (T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])
im = np.where(m.values)[0]
ucl, inv = np.unique(T.loc[m, "clone"].values, return_inverse=True)
y = (T.loc[m].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)


def clone(path):
    M = np.load(path)
    if M.ndim == 2 and M.shape[1] > 1:
        s = M[:, 0] + M[:, 1]
        M = np.where(s > 0, M[:, 0] / np.where(s > 0, s, 1), 0.5)
    s = M.ravel()[im]
    return np.bincount(inv, weights=s, minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))


def mix(tag):
    p = os.path.join(MIX, "nm_mix_%s.npy" % tag)
    return clone(p) if os.path.exists(p) else None


ck = mix("CK_w0")
ref = clone(os.path.join(FALL, "nm_cr_conn_n30_p30_d1.npy"))
print("CK_w0 AUROC %.4f; paper reference kernel %.4f; %d clones" % (roc_auc_score(y, ck), roc_auc_score(y, ref), len(y)))


def gain(a, b, idx, n_boot, seed=97):
    yy, aa, bb = y[idx], a[idx], b[idx]
    B = np.random.default_rng(seed).integers(0, len(idx), size=(n_boot, len(idx)))
    d = [roc_auc_score(yy[s], aa[s]) - roc_auc_score(yy[s], bb[s]) for s in B if yy[s].min() != yy[s].max()]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return roc_auc_score(yy, aa) - roc_auc_score(yy, bb), lo, hi


allidx = np.arange(len(y))
rows = []
for f in FAM:
    for w in W + [0.8, 1]:
        c = mix("%s_w%s" % (f, w))
        if c is None:
            continue
        g, lo, hi = gain(c, ck, allidx, 2000)
        rows.append(dict(part="grid_all_clones", family=f, w=w, auroc=roc_auc_score(y, c), gain=g, lo=lo, hi=hi))
G = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(G.round(4).to_string(index=False))

srows = []
for f in FAM:
    cands = {0.0: ck}
    for w in W:
        c = mix("%s_w%s" % (f, w))
        if c is not None:
            cands[w] = c
    if len(cands) == 1:
        continue
    res = []
    for r in range(20):
        tr, te = next(StratifiedKFold(2, shuffle=True, random_state=500 + r).split(allidx, y))
        best = max(cands, key=lambda w: roc_auc_score(y[tr], cands[w][tr]))
        g, lo, hi = gain(cands[best], ck, te, 1000, seed=700 + r)
        res.append((best, g, lo, hi))
    R = np.array(res)
    srows.append(dict(part="split_validated", family=f, w=float(R[:, 0].mean()), auroc=np.nan,
                      gain=float(R[:, 1].mean()), lo=float(np.percentile(R[:, 1], 5)),
                      hi=float(np.percentile(R[:, 1], 95)), share_above=float(np.mean(R[:, 2] > 0)),
                      chose_w0=float(np.mean(R[:, 0] == 0))))
S = pd.DataFrame(srows)
print(S.round(4).to_string(index=False))
pd.concat([G, S], ignore_index=True).round(5).to_csv(OUT, index=False)
print("wrote", OUT)
