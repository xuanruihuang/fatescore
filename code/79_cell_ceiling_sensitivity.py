"""Sensitivity of the cell-level ceiling to the assumed dispersion of a single cell.

57_label_ceiling.py's lower ceiling (0.933) assumes that the sequenced day-2
cell's own propensity varies around the clone propensity with the same
concentration nu as a whole well. A single cell may deviate more or less than a
well of descendants. This recomputes that ceiling with the cell's concentration
set to 0.25, 0.5, 1, 2 and 4 times nu (labels from the fitted two-level model, as
in 57; 2,000 simulated label sets each).

Usage: 79_cell_ceiling_sensitivity.py <d2_labels.tsv> <out.csv>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

A2, B2, NU = 0.2828, 0.2920, 1.4436
T = pd.read_csv(sys.argv[1], sep="\t")
M = T[(T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])].groupby("clone").first()
W = [(M["n_neu_w%d" % w] + M["n_mono_w%d" % w]).to_numpy(int) for w in (1, 2)]
rng = np.random.default_rng(79)
rows = []
for f in (0.25, 0.5, 1.0, 2.0, 4.0):
    v = []
    for _ in range(2000):
        pc = rng.beta(A2, B2, size=W[0].size)
        k = 0
        for n in W:
            pw = rng.beta(np.maximum(NU * pc, 1e-9), np.maximum(NU * (1 - pc), 1e-9))
            k = k + rng.binomial(n, pw)
        y = (2 * k >= W[0] + W[1]).astype(int)
        if y.min() == y.max():
            continue
        nc = f * NU
        ps = rng.beta(np.maximum(nc * pc, 1e-9), np.maximum(nc * (1 - pc), 1e-9))
        v.append(roc_auc_score(y, ps))
    v = np.array(v)
    rows.append(dict(cell_concentration_over_nu=f, ceiling_mean=v.mean(),
                     lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).round(4).to_csv(sys.argv[2], index=False)
