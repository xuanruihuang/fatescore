"""Score fate predictors on a lineage data set built by 95_ctmulti_build.py (or the
same layout): every <featdir>/<name>.npy with a <name>_log.json listing its columns.

A predictor's score for an early cell is its probability of the positive population
divided by its total probability of the terminal populations (0.5 where that total is
zero). Task cells: early cells whose clone has at least --min-mature mature
descendants. Scores: clone-level AUROC for a clone dominated by the positive
population (cell scores averaged within clones), and the cell-level Pearson r with the
clone's positive share. Paired bootstrap over clones (2,000 resamples) of the AUROC
difference from the reference predictor, and the minimum detectable difference of
that comparison (0.714 x interval width).

Usage: 97_fate_score.py <featdir> <out.csv> --pos Mono --ref cr_conn_all_n30_p30_d1
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("out")
ap.add_argument("--pos", default="Mono")
ap.add_argument("--ref", default="cr_conn_all_n30_p30_d1")
ap.add_argument("--min-mature", type=int, default=3)
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

T = pd.read_csv(os.path.join(a.featdir, "labels.tsv"), sep="\t")
early = json.load(open(os.path.join(a.featdir, "d2_names.json")))
assert list(T["cell"]) == early
task = (T["n_mature"] >= a.min_mature).values
it = np.where(task)[0]
clones = T.loc[task, "clone"].values
yc = T.loc[task, "f_" + a.pos.lower()].values if "f_" + a.pos.lower() in T else None
ucl, inv = np.unique(clones, return_inverse=True)
ydom = (T.loc[task].groupby("clone")["dom"].first().reindex(ucl).values == a.pos).astype(int)


def share(name):
    P = np.load(os.path.join(a.featdir, name + ".npy"))
    pops = json.load(open(os.path.join(a.featdir, name + "_log.json")))["populations"]
    tot = P.sum(1)
    return np.where(tot > 0, P[:, pops.index(a.pos)] / np.where(tot > 0, tot, 1), 0.5)


def clone_mean(s):
    return np.bincount(inv, weights=s[it], minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))


names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.featdir, "*.npy"))
               if os.path.exists(p[:-4] + "_log.json"))
S = {n: share(n) for n in names}
C = {n: clone_mean(S[n]) for n in names}
rng = np.random.default_rng(97)
B = rng.integers(0, len(ucl), size=(a.boot, len(ucl)))
ref = C[a.ref]
rows = []
for n in names:
    auc = roc_auc_score(ydom, C[n])
    r = pearsonr(S[n][it], yc)[0] if yc is not None else np.nan
    d = []
    for b in B:
        if ydom[b].min() == ydom[b].max():
            continue
        d.append(roc_auc_score(ydom[b], C[n][b]) - roc_auc_score(ydom[b], ref[b]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append(dict(predictor=n, auroc_clone=auc, r_cell=r, diff_vs_ref=auc - roc_auc_score(ydom, ref),
                     diff_lo=lo, diff_hi=hi, mdd=0.714 * (hi - lo)))
R = pd.DataFrame(rows).sort_values("auroc_clone", ascending=False)
R["n_cells"], R["n_clones"], R["n_pos_clones"] = int(task.sum()), len(ucl), int(ydom.sum())
R.to_csv(a.out, index=False)
print(R.round(3).to_string(index=False))
