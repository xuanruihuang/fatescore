"""Margin of error for a difference in Pearson r between predictors scored on
different cells.

53_r_floor.py treats Chreode's comparisons as paired: both predictors scored on
the same 183 clones, with correlated noise. Chreode's table scores the methods
on different numbers of cells (183 for Chreode, 296 for scDiffEq, 265 for
Waddington-OT, 335 for the static control), so unless the smaller sets are
nested in the larger ones and the noise is correlated, the comparison is
unpaired. This gives the other extreme: the two r values come from independent
samples of clones (one early cell each) under the two-level label model at the
fitted parameters, each predictor logit(p_c) plus independent noise calibrated
to the same AUROC (0.75, r about 0.47, the level of the published values), so
the true difference is zero. The margin of error is 1.96 times the standard
deviation of r1 - r2 over simulated data sets.

Usage: 73_unpaired_margin.py <d2_labels.tsv> <out.csv> [--datasets 4000]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--two-level", default="0.2828,0.2920,1.4436")
ap.add_argument("--auroc", type=float, default=0.75)
ap.add_argument("--datasets", type=int, default=4000)
a = ap.parse_args()
A2, B2, NU = (float(x) for x in a.two_level.split(","))

T = pd.read_csv(a.labels, sep="\t")
M = T[(T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])]
C = M.groupby("clone").agg(nw1=("n_neu_w1", "first"), mw1=("n_mono_w1", "first"),
                           nw2=("n_neu_w2", "first"), mw2=("n_mono_w2", "first"))
W1 = (C["nw1"] + C["mw1"]).to_numpy(int)
W2 = (C["nw2"] + C["mw2"]).to_numpy(int)
rng = np.random.default_rng(73)


def gen(sel):
    pc = rng.beta(A2, B2, size=sel.size)
    k = 0
    for w in (W1[sel], W2[sel]):
        pw = rng.beta(np.maximum(NU * pc, 1e-9), np.maximum(NU * (1 - pc), 1e-9))
        k = k + rng.binomial(w, pw)
    tot = W1[sel] + W2[sel]
    y = (2 * k >= tot).astype(int)
    lp = np.log(np.clip(pc, 1e-6, 1 - 1e-6) / np.clip(1 - pc, 1e-6, 1))
    return y, k / tot, lp


# calibrate the noise SD to the target AUROC on a large sample
idx = rng.integers(0, W1.size, size=200000)
y, f, lp = gen(idx)
z = rng.standard_normal(idx.size)
lo, hi = 1e-3, 1e3
for _ in range(50):
    mid = np.sqrt(lo * hi)
    if roc_auc_score(y, lp + mid * z) > a.auroc:
        lo = mid
    else:
        hi = mid
sig = np.sqrt(lo * hi)
r_pop = np.corrcoef(lp + sig * z, f)[0, 1]
print("noise SD %.4f; AUROC %.3f; population r %.3f" % (sig, a.auroc, r_pop))


def r_on(n):
    sel = rng.choice(W1.size, size=n, replace=False)
    y, f, lp = gen(sel)
    return np.corrcoef(lp + sig * rng.standard_normal(n), f)[0, 1]


rows = []
for n2 in (183, 265, 296, 335):
    d = np.array([r_on(183) - r_on(n2) for _ in range(a.datasets)])
    rows.append(dict(n1=183, n2=n2, r_population=round(r_pop, 3),
                     sd_diff=round(d.std(), 4),
                     margin_unpaired=round(1.96 * d.std(), 4),
                     empirical_halfwidth=round((np.percentile(d, 97.5)
                                                - np.percentile(d, 2.5)) / 2, 4)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(a.out, index=False)
