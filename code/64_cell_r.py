"""The published LARRY metric with an expression-only baseline beside it.

PRESCIENT and Chreode rank methods by the Pearson r, over day-2 cells, between a
method's predicted neutrophil share p_neu / (p_neu + p_mono) and the clone's
observed neutrophil fraction among its neutrophil and monocyte descendants.
Their tables carry no expression-only predictor. This computes, on the day-2
cells whose clone has at least 3 neutrophil or monocyte descendants (the
continuous task of 06_tasks_bc.py):

  - each method's share as is, no fitting (the published way of scoring);
  - out-of-fold predictions of a regressor on the sorting gate alone, and on the
    gate plus day-2 expression (total counts; spliced counts), clone-disjoint
    folds, inner tuning as everywhere else (cvlib.run_arm_reg), 10 repeats;

and reports the cell-level Pearson r (the published unit) and the clone-level r
(cells averaged within a clone), with the 10-repeat mean and SD for the fitted
predictors.

Usage: 64_cell_r.py <featdir_d2> <featdir_all> <labels.tsv> <out.csv>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvlib import cp10k_log, run_arm_reg  # noqa: E402

FD2, FALL, LAB, OUT = sys.argv[1:5]
T = pd.read_csv(LAB, sep="\t")
names = json.load(open(os.path.join(FD2, "d2_names.json")))
assert list(T["cell"]) == names
assert json.load(open(os.path.join(FALL, "d2_names.json"))) == names
sel = (T["n_neu_mono"] >= 3) & T["f_neu"].notna()
i = np.where(sel.values)[0]
y = T.loc[sel, "f_neu"].values.astype(float)
g = T.loc[sel, "clone"].values
print("cells %d, clones %d" % (len(i), len(np.unique(g))), flush=True)


def rs(p):
    cell = pearsonr(p, y)[0]
    cl = (pd.DataFrame(dict(p=p, y=y, g=g)).groupby("g")
          .agg(p=("p", "mean"), y=("y", "first")))
    return cell, pearsonr(cl["p"], cl["y"])[0]


rows = []
for m in ("statot", "wot", "moscot", "prescient"):
    P = np.load(os.path.join(FALL, "nm_snap_%s.npy" % m)).ravel()[i]
    c, k = rs(P)
    rows.append(dict(predictor="share: " + m, kind="method share, no fitting",
                     r_cell=c, r_cell_sd=np.nan, r_clone=k, r_clone_sd=np.nan))
for m in ("velo", "pseudo"):
    p = os.path.join(FALL, "nm_cr_%s.npy" % m)
    if os.path.exists(p):
        P = np.load(p).ravel()[i]
        c, k = rs(P)
        rows.append(dict(predictor="share: CellRank " + m, kind="method share, no fitting",
                         r_cell=c, r_cell_sd=np.nan, r_clone=k, r_clone_sd=np.nan))

PCA = {"tot": cp10k_log(np.load(os.path.join(FD2, "raw_matrix.npy")))[i],
       "spl": cp10k_log(np.load(os.path.join(FD2, "raw_spliced.npy")))[i]}
RAW = {"pop": (T["pop"].values == "LSK").astype(float)[i, None]}
ARMS = {"gate only": ["pop"], "gate + expression (total)": ["pop", "tot"],
        "gate + expression (spliced)": ["pop", "spl"]}
res = Parallel(n_jobs=15)(delayed(run_arm_reg)(a, b, PCA, RAW, y, g, s, 5, 50)
                          for a, b in ARMS.items() for s in range(10))
per = {}
for arm, seed, oof, picked in res:
    per.setdefault(arm, []).append(rs(oof))
for arm, v in per.items():
    v = np.array(v)
    rows.append(dict(predictor=arm, kind="expression baseline, out-of-fold",
                     r_cell=v[:, 0].mean(), r_cell_sd=v[:, 0].std(ddof=1),
                     r_clone=v[:, 1].mean(), r_clone_sd=v[:, 1].std(ddof=1)))
R = pd.DataFrame(rows).round(4)
R.to_csv(OUT, index=False)
print(R.to_string(index=False))
