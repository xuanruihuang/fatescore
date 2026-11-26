"""Unsupervised dynamics-free references on the published metric and on AUROC.

A method's fate probabilities are unsupervised: they are computed without the
clonal labels. The fair dynamics-free reference for them is therefore also
unsupervised: a fate probability computed from expression alone, with no
dynamics and no sampling time. Two such references are available on the same
cells: CellRank's connectivity kernel (absorption probabilities on the
expression neighbour graph, 18 graph settings) and inverse-distance weighting
(each cell's fate probabilities proportional to a power of its inverse distance
to the centroid of each terminal population, exponent 1, 5, 10, 20 or 50).
For each, and for the four optimal-transport and potential-based methods and the
two CellRank kernels, this computes from the neutrophil share
p_neu / (p_neu + p_mono), with no fitting:
  - the cell-level Pearson r with the clone's observed neutrophil share on the
    cells of the continuous task (the metric published LARRY rankings use);
  - the Spearman correlation on the same cells, and the fraction of those
    cells whose predicted share lies in [0.93, 1.00];
  - the clone-level AUROC for the dominant fate on the main task (the share
    averaged within clones).

Usage: 66_unsupervised_reference.py <featdir_all> <labels.tsv> <out.csv> [extra,shares]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

FALL, LAB, OUT = sys.argv[1:4]
# optional: further shares to score the same way (method configuration variants)
EXTRA = sys.argv[4].split(",") if len(sys.argv) > 4 else []
T = pd.read_csv(LAB, sep="\t")
c = (T["n_neu_mono"] >= 3) & T["f_neu"].notna()
ic, yc = np.where(c.values)[0], T.loc[c, "f_neu"].values.astype(float)
m = (T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])
im = np.where(m.values)[0]
ym = (T.loc[m, "dom"].values == "Neutrophil").astype(int)
gm = T.loc[m, "clone"].values


def share(name):
    p = os.path.join(FALL, name + ".npy")
    M = np.load(p)
    if M.ndim == 2 and M.shape[1] > 1:
        s = M[:, 0] + M[:, 1]
        return np.where(s > 0, M[:, 0] / np.where(s > 0, s, 1), 0.5)
    return M.ravel()


def score(name, kind):
    p = share(name)
    r = pearsonr(p[ic], yc)[0]
    cl = (pd.DataFrame(dict(p=p[im], y=ym, g=gm)).groupby("g")
          .agg(p=("p", "mean"), y=("y", "first")))
    return dict(predictor=name, kind=kind, r_cell=r,
                spearman_cell=spearmanr(p[ic], yc)[0],
                frac_share_093_100=np.mean((p[ic] >= 0.93) & (p[ic] <= 1.0)),
                auroc_clone=roc_auc_score(cl["y"], cl["p"]))


rows = []
for k in ("snap_statot", "snap_wot", "snap_moscot", "snap_prescient"):
    rows.append(score("nm_" + k, "method"))
for k in EXTRA:
    if os.path.exists(os.path.join(FALL, k + ".npy")):
        rows.append(score(k, "method variant"))
for k in ("cr_velo", "cr_pseudo"):
    rows.append(score("nm_" + k, "CellRank velocity/pseudotime kernel"))
for f in sorted(glob.glob(os.path.join(FALL, "nm_cr_conn_*.npy"))):
    rows.append(score(os.path.basename(f)[:-4], "connectivity kernel (dynamics-free)"))
for f in sorted(glob.glob(os.path.join(FALL, "snap_idw_g*.npy"))):
    rows.append(score(os.path.basename(f)[:-4], "inverse-distance weighting (dynamics-free)"))
R = pd.DataFrame(rows).round(4)
R.to_csv(OUT, index=False)
print(R.to_string(index=False))
print(R.groupby("kind")[["r_cell", "auroc_clone"]].agg(["min", "median", "max"]).round(3))
