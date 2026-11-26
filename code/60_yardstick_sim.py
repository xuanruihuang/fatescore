"""How much the synthetic-feature yardstick overstates an independent day-2 signal.

The calibration features of 50_lowdim_calibration.py carry the clone's observed
Neutrophil fraction f (both wells pooled), which is what defines the label. A
day-2 predictor can at most know the clone's shared propensity p_c (two-level
label model, 57_label_ceiling.py); the well-level part of f is out of its reach,
and it is out of the expression predictor's reach too. So a feature built from
f has errors independent of expression's in a way no real day-2 feature can,
and at equal stand-alone AUROC it adds more. This simulation puts a number on
the difference under the fitted two-level model, on the clones of the main task
with their own counts of early cells and well sizes.

Per simulated data set: clone propensities and well draws as in 53_r_floor.py
--two-level; an expression score logit(p_c) + noise at AUROC 0.87; and two
candidate added features at a target stand-alone AUROC, one built as the
calibration features are (f plus per-cell noise, averaged over the clone's
early cells), one built from p_c the same way. The increment is the AUROC of
the population-optimal logistic combination with the expression score minus
the expression score's own, on a large sample (the clone table repeated), so
what is compared is the information, not a finite-sample learner.

Usage: 60_yardstick_sim.py <d2_labels.tsv> <out.csv> [--two-level a,b,nu]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--two-level", default="0.2828,0.2920,1.4436")
ap.add_argument("--rep", type=int, default=200)
ap.add_argument("--expr-auroc", type=float, default=0.87)
ap.add_argument("--targets", default="0.72,0.75,0.80,0.86")
a = ap.parse_args()
A2, B2, NU = (float(x) for x in a.two_level.split(","))

T = pd.read_csv(a.labels, sep="\t")
T = T[(T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])]
C = T.groupby("clone").agg(n_early=("cell", "size"),
                           nw1=("n_neu_w1", "first"), mw1=("n_mono_w1", "first"),
                           nw2=("n_neu_w2", "first"), mw2=("n_mono_w2", "first"))
ne0 = C["n_early"].to_numpy(float)
w1 = (C["nw1"] + C["mw1"]).to_numpy(int)
w2 = (C["nw2"] + C["mw2"]).to_numpy(int)
ne, w1, w2 = (np.tile(x, a.rep) for x in (ne0, w1, w2))
n = ne.size
rng = np.random.default_rng(60)
pc = rng.beta(A2, B2, size=n)
k = 0
for w in (w1, w2):
    pw = rng.beta(np.maximum(NU * pc, 1e-9), np.maximum(NU * (1 - pc), 1e-9))
    k = k + rng.binomial(w, pw)
f = k / (w1 + w2)
y = (2 * k >= (w1 + w2)).astype(int)  # ties to Neutrophil, the fixed order
lp = np.log(np.clip(pc, 1e-6, 1 - 1e-6) / np.clip(1 - pc, 1e-6, 1))
print("clones %d (x%d); prevalence %.3f; ceiling AUROC(p_c) %.4f"
      % (ne0.size, a.rep, y.mean(), roc_auc_score(y, pc)), flush=True)


def scaled(sig, sd):
    """sig + per-cell N(0, sd) noise averaged over the clone's early cells."""
    return sig + sd * rng.standard_normal(n) / np.sqrt(ne)


def at_auroc(make, target):
    lo, hi = 1e-3, 1e3
    z = rng.standard_normal(n) / np.sqrt(ne)
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        if roc_auc_score(y, make(mid, z)) > target:
            lo = mid
        else:
            hi = mid
    return make(np.sqrt(lo * hi), z)


E = at_auroc(lambda s, z: lp + s * z, a.expr_auroc)
aE = roc_auc_score(y, E)
rows = []
for t in (float(x) for x in a.targets.split(",")):
    for name, sig in (("from_f", (f - f.mean()) / f.std()),
                      ("from_pc", (pc - pc.mean()) / pc.std())):
        S = at_auroc(lambda s, z: sig + s * z, t)
        X = np.column_stack([E, S])
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
        inc = roc_auc_score(y, m.decision_function(X)) - aE
        rc = np.corrcoef(S - np.where(y == 1, S[y == 1].mean(), S[y == 0].mean()),
                         E - np.where(y == 1, E[y == 1].mean(), E[y == 0].mean()))[0, 1]
        rows.append(dict(target=t, built=name, alone=round(roc_auc_score(y, S), 4),
                         expr=round(aE, 4), increment=round(inc, 4),
                         resid_corr_with_expr=round(rc, 3)))
        print(rows[-1], flush=True)
R = pd.DataFrame(rows)
P = R.pivot(index="target", columns="built", values="increment")
P["ratio_f_over_pc"] = (P["from_f"] / P["from_pc"]).round(2)
print(P)
R.to_csv(a.out, index=False)
