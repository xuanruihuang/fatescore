"""Brier score of each method against the connectivity kernel (a reviewer asked whether a
calibration-sensitive score changes the conclusion).

Clone-level predictions as in 97/99/104. Two versions: the raw predicted share (clipped to [0, 1]),
and a recalibrated one (cross-fitted logistic regression of the clone label on the logit of the
prediction, 5 folds x 20 repetitions, out-of-fold probabilities), which removes differences in
calibration and keeps only the information in the ranking and spacing. Difference from the kernel
(negative = better) with a paired bootstrap over clones (2,000 resamples, seed 97).

Usage: 109_brier.py <larry_featdir_all> <larry_labels.tsv> <ctm_featdir> <ctr_featdir> <out.csv>
"""
import ast
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FALL, LAB, CTM, CTR, OUT = sys.argv[1:6]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "99_calibrated_table.py"),
           encoding="utf-8").read()
NAMES = ast.literal_eval(src[src.index("NAMES = ") + 8: src.index("}", src.index("NAMES = ")) + 1])
SANITY = {"cr_conn_n15_p50_d0": "kernel, other setting (n15 p50 d0)"}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def oof(X, y, reps=20):
    out = np.zeros(len(y))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1000 + r).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], y[tr])
            out[te] += m.predict_proba(X[te])[:, 1]
    return out / reps


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def recal(c, y, reps=20):
    return oof(logit(c)[:, None], y, reps)


def score(ds, y, ref, preds):
    rng = np.random.default_rng(97)
    B = rng.integers(0, len(y), size=(2000, len(y)))
    kr, kc = np.clip(ref, 0, 1), recal(ref, y)
    rows = []
    for lab, c in preds.items():
        cr, cc = np.clip(c, 0, 1), recal(c, y)
        d_raw = [brier(cr[b], y[b]) - brier(kr[b], y[b]) for b in B]
        d_cal = [brier(cc[b], y[b]) - brier(kc[b], y[b]) for b in B]
        lr, hr = np.percentile(d_raw, [2.5, 97.5]); lc, hc = np.percentile(d_cal, [2.5, 97.5])
        rows.append(dict(dataset=ds, method=lab, brier_raw=brier(cr, y), brier_raw_kernel=brier(kr, y),
                         d_raw=brier(cr, y) - brier(kr, y), d_raw_lo=lr, d_raw_hi=hr,
                         brier_cal=brier(cc, y), brier_cal_kernel=brier(kc, y),
                         d_cal=brier(cc, y) - brier(kc, y), d_cal_lo=lc, d_cal_hi=hc,
                         better_cal=("yes" if hc < 0 else ("no, worse" if lc > 0 else "not separated")),
                         n_clones=len(y)))
    return rows


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


P = {lab: larry(k) for k, lab in {**NAMES, **SANITY}.items()
     if os.path.exists(os.path.join(FALL, "nm_" + k + ".npy"))}
rows += score("LARRY", y, larry("cr_conn_n30_p30_d1"), P)
print("LARRY done", flush=True)

# CellTag-multi (as 97)
for ds, fd, pos in (("CellTag-multi haematopoiesis", CTM, "Mono"),
                    ("CellTag-multi reprogramming", CTR, "reprogramming")):
    L = pd.read_csv(os.path.join(fd, "labels.tsv"), sep="\t")
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

    sanity = {"cr_conn_all_n15_p50_d0": SANITY["cr_conn_n15_p50_d0"]}
    Q = {lab: ct(k) for k, lab in {**NAMES, **sanity}.items()
         if os.path.exists(os.path.join(fd, k + ".npy")) and os.path.exists(os.path.join(fd, k + "_log.json"))}
    rows += score(ds, yy, ct("cr_conn_all_n30_p30_d1"), Q)
    print(ds, "done", flush=True)

R = pd.DataFrame(rows)
R.round(4).to_csv(OUT, index=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
print("recalibrated Brier: better than the kernel %d, worse %d, not separated %d (of %d rows, sanity included)"
      % ((R.better_cal == "yes").sum(), (R.better_cal == "no, worse").sum(), (R.better_cal == "not separated").sum(), len(R)))
