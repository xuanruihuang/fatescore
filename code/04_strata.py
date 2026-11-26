"""Where the predictions succeed and fail, and how far they are from the ceiling.

Usage: 04_strata.py <featdir> <labels.tsv> <evalprefix> <outprefix>

Reads the out-of-fold predictions written by 03_evaluate.py and
  (a) splits them by starting population, sister count, sister time point,
      day-2 annotation, library and discriminant margin;
  (b) compares every arm with the sister-based ceiling on exactly the same
      clones and the same metric: one well's neutrophil fraction used as a
      prediction of the other well's dominant fate.
No model is refitted here.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FD, LAB, EV, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
# the two fate names, overridable for a second dataset
POS = sys.argv[5] if len(sys.argv) > 5 else "Neutrophil"
NEG = sys.argv[6] if len(sys.argv) > 6 else "Monocyte"

T = pd.read_csv(LAB, sep="\t")
task = json.load(open(EV + "_task.json"))
OOF = np.load(EV + "_oof.npy")          # (arm, repeat, cell)
arms, nrep = task["arms"], task["repeats"]
rows_idx = np.array(task["task_rows"])
y = np.array(task["y"])
clones = np.array(task["clones"])
D = T.iloc[rows_idx].reset_index(drop=True)
assert len(D) == len(y)


def clone_auroc(p, mask=None):
    """Average predictions within a clone, then score at the clone level."""
    m = np.ones(len(y), bool) if mask is None else mask
    if m.sum() == 0:
        return np.nan, np.nan, 0
    d = (pd.DataFrame(dict(p=p[m], y=y[m], g=clones[m]))
         .groupby("g").agg(p=("p", "mean"), y=("y", "first")))
    if d["y"].nunique() < 2:
        return np.nan, np.nan, len(d)
    return (roc_auc_score(d["y"], d["p"]),
            float(((d["p"] > 0.5).astype(int) == d["y"]).mean()), len(d))


# ------------------------------------------------------------------- strata
margin_ref = arms.index("A2_state_spliced")
margin = np.abs(OOF[margin_ref].mean(0) - 0.5)
q = np.quantile(margin, [1 / 3, 2 / 3])
D["margin_tercile"] = np.where(margin <= q[0], "1_near_boundary",
                               np.where(margin <= q[1], "2_middle",
                                        "3_far"))
D["sis_bin"] = pd.cut(D["n_sis"], [2, 4, 9, 10 ** 6],
                      labels=["3-4", "5-9", ">=10"])
D["sis_time"] = np.where((D.n_sis_d4 > 0) & (D.n_sis_d6 > 0), "both",
                         np.where(D.n_sis_d4 > 0, "d4_only", "d6_only"))
D["d2_undiff"] = np.where(D.state_d2 == "Undifferentiated", "undiff", "other")

AXES = ["pop", "sis_bin", "sis_time", "d2_undiff", "margin_tercile",
        "library"]
rows = []
for ai, arm in enumerate(arms):
    for rep in range(nrep):
        p = OOF[ai, rep]
        a, acc, n = clone_auroc(p)
        rows.append(dict(arm=arm, rep=rep, axis="all", level="all",
                         auroc=a, acc=acc, n_clones=n))
        for ax in AXES:
            for lv in D[ax].dropna().unique():
                m = (D[ax] == lv).values
                a, acc, n = clone_auroc(p, m)
                rows.append(dict(arm=arm, rep=rep, axis=ax, level=str(lv),
                                 auroc=a, acc=acc, n_clones=n))
R = pd.DataFrame(rows)
S = (R.groupby(["arm", "axis", "level"])
     .agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
          acc=("acc", "mean"), n_clones=("n_clones", "first"))
     .reset_index())
S.to_csv(OUT + "_strata.csv", index=False)

# increments relative to the two state baselines, per stratum
piv = S.pivot_table(index=["axis", "level"], columns="arm", values="auroc")
inc = []
for ref in ("A2_state_spliced", "A4_state_smoothed"):
    for arm in arms:
        if arm == ref or arm not in piv.columns:
            continue
        d = (piv[arm] - piv[ref]).rename("increment").reset_index()
        d["arm"], d["ref"] = arm, ref
        d["n_clones"] = S.groupby(["axis", "level"])["n_clones"].first().values
        inc.append(d)
pd.concat(inc).to_csv(OUT + "_increments.csv", index=False)

# --------------------------------------------------- sister-based ceiling
# Same clones, same metric: predict one well's dominant fate, scoring either
# the other well's neutrophil fraction (the sister oracle) or the model.
NM = [POS, NEG]
cl = (D.assign(_i=np.arange(len(D)))
      .groupby("clone")
      .agg(dom_w1=("dom_w1", "first"), dom_w2=("dom_w2", "first"),
           n_neu_w1=("n_neu_w1", "first"), n_mono_w1=("n_mono_w1", "first"),
           n_neu_w2=("n_neu_w2", "first"), n_mono_w2=("n_mono_w2", "first")))
cl["nm1"] = cl.n_neu_w1 + cl.n_mono_w1
cl["nm2"] = cl.n_neu_w2 + cl.n_mono_w2
cl["f1"] = cl.n_neu_w1 / cl.nm1.replace(0, np.nan)
cl["f2"] = cl.n_neu_w2 / cl.nm2.replace(0, np.nan)

model_clone = {}
for ai, arm in enumerate(arms):
    p = OOF[ai].mean(0)
    model_clone[arm] = (pd.DataFrame(dict(p=p, g=clones))
                        .groupby("g")["p"].mean())

rows = []
for src, tgt, fs, dt in [(1, 2, "f1", "dom_w2"), (2, 1, "f2", "dom_w1")]:
    sel = cl[(cl["nm1"] >= 1) & (cl["nm2"] >= 1) & cl[dt].isin(NM)]
    lab = (sel[dt] == POS).astype(int)
    if lab.nunique() < 2:
        continue
    rows.append(dict(direction="w%d_to_w%d" % (src, tgt), who="sister_oracle",
                     n_clones=len(sel),
                     auroc=roc_auc_score(lab, sel[fs]),
                     acc=float(((sel[fs] > 0.5).astype(int) == lab).mean())))
    for arm in arms:
        s = model_clone[arm].reindex(sel.index)
        rows.append(dict(direction="w%d_to_w%d" % (src, tgt), who=arm,
                         n_clones=len(sel),
                         auroc=roc_auc_score(lab, s),
                         acc=float(((s > 0.5).astype(int) == lab).mean())))
C = pd.DataFrame(rows)
C.to_csv(OUT + "_ceiling.csv", index=False)
avg = C.groupby("who").agg(auroc=("auroc", "mean"), acc=("acc", "mean"),
                           n=("n_clones", "mean")).sort_values(
    "auroc", ascending=False)
avg.to_csv(OUT + "_ceiling_mean.csv")
print(avg.to_string())
print()
print(S[S.axis == "all"].sort_values("auroc", ascending=False).to_string())
