"""Check of the one positive increment (scNODE, CellTag-multi haematopoiesis): does it come from the
early cells whose simulated endpoints carry no terminal mass (share set to 0.5)?

Prints: number of task cells and clones with no mass; the clone label rate among clones with any
no-mass cell versus the rest; and the increment over the kernel (as 104) recomputed (a) on clones
without any no-mass cell, (b) with the method replaced by a no-mass indicator alone.

Usage: 104b_nomass_check.py <featdir> <pos> <method>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FD, POS, MET = sys.argv[1:4]
L = pd.read_csv(os.path.join(FD, "labels.tsv"), sep="\t")
task = (L["n_mature"] >= 3).values
it = np.where(task)[0]
uc, iv = np.unique(L.loc[task, "clone"].values, return_inverse=True)
y = (L.loc[task].groupby("clone")["dom"].first().reindex(uc).values == POS).astype(int)


def raw(name):
    M = np.load(os.path.join(FD, name + ".npy"))
    pops = json.load(open(os.path.join(FD, name + "_log.json")))["populations"]
    tot = M.sum(1)
    return np.where(tot > 0, M[:, pops.index(POS)] / np.where(tot > 0, tot, 1), 0.5), tot > 0


def cm(s):
    return np.bincount(iv, weights=s[it], minlength=len(uc)) / np.bincount(iv, minlength=len(uc))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def oof(X, yy, reps=20):
    out = np.zeros(len(yy))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1000 + r).split(X, yy):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], yy[tr])
            out[te] += m.predict_proba(X[te])[:, 1]
    return out / reps


def inc(k, c, yy):
    rng = np.random.default_rng(97)
    B = rng.integers(0, len(yy), size=(2000, len(yy)))
    pa, pb = oof(logit(k)[:, None], yy), oof(np.c_[logit(k), c], yy)
    d = [roc_auc_score(yy[b], pb[b]) - roc_auc_score(yy[b], pa[b]) for b in B if yy[b].min() != yy[b].max()]
    return roc_auc_score(yy, pb) - roc_auc_score(yy, pa), *np.percentile(d, [2.5, 97.5])


s, has = raw(MET)
k, _ = raw("cr_conn_all_n30_p30_d1")
nomass_cl = np.bincount(iv, weights=(~has[it]).astype(float), minlength=len(uc)) > 0
print("task cells %d, without mass %d; clones %d, with any no-mass cell %d"
      % (len(it), (~has[it]).sum(), len(uc), nomass_cl.sum()))
print("positive-clone rate: clones with a no-mass cell %.3f, others %.3f" % (y[nomass_cl].mean(), y[~nomass_cl].mean()))
K, C = cm(k), cm(s)
print("full: increment %.3f (%.3f, %.3f)" % inc(K, logit(C), y))
keep = ~nomass_cl
print("clones without no-mass cells (%d): increment %.3f (%.3f, %.3f); method AUROC %.3f, kernel %.3f"
      % ((keep.sum(),) + inc(K[keep], logit(C[keep]), y[keep]) + (roc_auc_score(y[keep], C[keep]), roc_auc_score(y[keep], K[keep]))))
frac = np.bincount(iv, weights=(~has[it]).astype(float), minlength=len(uc)) / np.bincount(iv, minlength=len(uc))
print("no-mass share alone as the added predictor: increment %.3f (%.3f, %.3f)" % inc(K, frac, y))
