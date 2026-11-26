"""The multiclass and the continuous version of the same question.

Usage: 06_tasks_bc.py <featdir> <labels.tsv> <outprefix>

Task B: the clone's dominant mature fate among the six populations with at
least ten clones. Task C: the clone's neutrophil fraction among its neutrophil
and monocyte sisters, which is the quantity Weinreb et al. correlated their
algorithms against, so the numbers here are directly comparable with the R they
report for PBA, WOT and FateID and with the R = 0.5 they give as those
algorithms' ceiling.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from cvlib import (cp10k_log, run_arm_multi, run_arm_reg, _macro_auc)

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("outprefix")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--n-jobs", type=int, default=24)
ap.add_argument("--pcs", type=int, default=50)
CFG = ap.parse_args()


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


T = pd.read_csv(CFG.labels, sep="\t")
assert list(T["cell"]) == json.load(
    open(os.path.join(CFG.featdir, "d2_names.json")))


def load(n):
    p = os.path.join(CFG.featdir, n + ".npy")
    return np.load(p) if os.path.exists(p) else None


raw_s, raw_u, raw_tot = load("raw_spliced"), load("raw_unspliced"), load("raw_matrix")
Ms, Mu = load("Ms"), load("Mu")
vel = {k: load("vel_" + k) for k in ("det", "dyn", "velovi")}
crf = {k: load("cr_" + k) for k in ("velo", "pseudo")}

PCA_ALL = {
    "state_s": cp10k_log(raw_s), "state_tot": cp10k_log(raw_tot),
    "unspl": cp10k_log(raw_u), "Ms": np.log1p(np.clip(Ms, 0, None)),
    "ratio": Mu / (Ms + Mu + 1e-8),
}
for k, v in vel.items():
    if v is not None:
        PCA_ALL["vel_" + k] = v
RAW_ALL = {"pop": (T["pop"].values == "LSK").astype(float)[:, None]}
for k, v in crf.items():
    if v is not None:
        RAW_ALL["cr_" + k] = v

ARMS = {
    "A0_prior": [], "A1_pop": ["pop"],
    "A2_state_spliced": ["pop", "state_s"],
    "A4_state_smoothed": ["pop", "Ms"],
    "A5_unspliced_only": ["pop", "unspl"],
    "A6_state_plus_unspliced": ["pop", "state_s", "unspl"],
    "A7_smoothed_plus_ratio": ["pop", "Ms", "ratio"],
    "V1_smoothed_plus_vel_det": ["pop", "Ms", "vel_det"],
    "V2_smoothed_plus_vel_dyn": ["pop", "Ms", "vel_dyn"],
    "V3_smoothed_plus_vel_velovi": ["pop", "Ms", "vel_velovi"],
    "V4_smoothed_plus_cr_velo": ["pop", "Ms", "cr_velo"],
    "V5_smoothed_plus_cr_pseudo": ["pop", "Ms", "cr_pseudo"],
}
ARMS = {k: v for k, v in ARMS.items()
        if all(b in PCA_ALL or b in RAW_ALL for b in v)}

# ------------------------------------------------------------------- task B
CLASSES = ["Neutrophil", "Monocyte", "Baso", "Meg", "Mast", "Erythroid"]
selB = (T["n_sis"] >= CFG.min_sis) & T["dom"].isin(CLASSES)
iB = np.where(selB.values)[0]
yB = T.loc[selB, "dom"].values
gB = T.loc[selB, "clone"].values
say("task B: %d cells, %d clones; %s"
    % (len(iB), len(np.unique(gB)), pd.Series(yB).value_counts().to_dict()))
pcaB = {k: v[iB] for k, v in PCA_ALL.items()}
rawB = {k: v[iB] for k, v in RAW_ALL.items()}

jobs = [(a, b, s) for a, b in ARMS.items() for s in range(CFG.repeats)]
res = Parallel(n_jobs=CFG.n_jobs, verbose=5)(
    delayed(run_arm_multi)(a, b, pcaB, rawB, yB, gB, s, CFG.folds, CFG.pcs,
                           CLASSES) for a, b, s in jobs)

rows = []
for arm, seed, P, picked in res:
    d = pd.DataFrame(P, columns=CLASSES).assign(g=gB, y=yB)
    cl = d.groupby("g").agg({**{c: "mean" for c in CLASSES}, "y": "first"})
    yy = cl["y"].values
    PP = cl[CLASSES].values
    pred = np.array(CLASSES)[PP.argmax(1)]
    bal = np.mean([(pred[yy == c] == c).mean() for c in CLASSES
                   if (yy == c).sum() > 0])
    # one-vs-rest AUROC per class, each against its own probability column
    per = {}
    for j, c in enumerate(CLASSES):
        lab = (yy == c).astype(int)
        per["auroc_" + c] = (np.nan if lab.min() == lab.max()
                             else roc_auc_score(lab, PP[:, j]))
    rows.append(dict(arm=arm, seed=seed, macro_auroc=_macro_auc(yy, PP, CLASSES),
                     balanced_acc=bal, n_clones=len(cl), **per))
RB = pd.DataFrame(rows)
RB.to_csv(CFG.outprefix + "_taskB_perrepeat.csv", index=False)
SB = RB.drop(columns=["seed"]).groupby("arm").mean(numeric_only=True)
SB = SB.sort_values("macro_auroc", ascending=False)
SB.to_csv(CFG.outprefix + "_taskB_summary.csv")
say("\ntask B\n" + SB.to_string())

# ------------------------------------------------------------------- task C
selC = (T["n_neu_mono"] >= 3) & T["f_neu"].notna()
iC = np.where(selC.values)[0]
yC = T.loc[selC, "f_neu"].values.astype(float)
gC = T.loc[selC, "clone"].values
say("task C: %d cells, %d clones, mean f_neu %.3f"
    % (len(iC), len(np.unique(gC)), yC.mean()))
pcaC = {k: v[iC] for k, v in PCA_ALL.items()}
rawC = {k: v[iC] for k, v in RAW_ALL.items()}

jobs = [(a, b, s) for a, b in ARMS.items() for s in range(CFG.repeats)]
res = Parallel(n_jobs=CFG.n_jobs, verbose=5)(
    delayed(run_arm_reg)(a, b, pcaC, rawC, yC, gC, s, CFG.folds, CFG.pcs)
    for a, b, s in jobs)

rows = []
for arm, seed, oof, picked in res:
    cl = (pd.DataFrame(dict(p=oof, y=yC, g=gC)).groupby("g")
          .agg(p=("p", "mean"), y=("y", "first")))
    rows.append(dict(arm=arm, seed=seed,
                     spearman=spearmanr(cl["y"], cl["p"]).statistic,
                     pearson=pearsonr(cl["y"], cl["p"])[0],
                     n_clones=len(cl)))
RC = pd.DataFrame(rows)
RC.to_csv(CFG.outprefix + "_taskC_perrepeat.csv", index=False)
SC = RC.drop(columns=["seed"]).groupby("arm").mean(numeric_only=True)
SC = SC.sort_values("pearson", ascending=False)
SC.to_csv(CFG.outprefix + "_taskC_summary.csv")
say("\ntask C\n" + SC.to_string())

# the sister ceiling in the same units: one well's neutrophil fraction against
# the other well's
cl = (T.loc[selC].groupby("clone")
      .agg(n_neu_w1=("n_neu_w1", "first"), n_mono_w1=("n_mono_w1", "first"),
           n_neu_w2=("n_neu_w2", "first"), n_mono_w2=("n_mono_w2", "first")))
cl["nm1"] = cl.n_neu_w1 + cl.n_mono_w1
cl["nm2"] = cl.n_neu_w2 + cl.n_mono_w2
ceil_rows = []
for k in (1, 2, 3):
    s = cl[(cl.nm1 >= k) & (cl.nm2 >= k)]
    if len(s) < 10:
        continue
    f1 = s.n_neu_w1 / s.nm1
    f2 = s.n_neu_w2 / s.nm2
    ceil_rows.append(dict(min_per_readout=k, n_clones=len(s),
                          pearson=float(pearsonr(f1, f2)[0]),
                          spearman=float(spearmanr(f1, f2).statistic)))
    say("ceiling for task C, >=%d per read-out: n=%d pearson %.3f "
        "spearman %.3f" % (k, len(s), ceil_rows[-1]["pearson"],
                           ceil_rows[-1]["spearman"]))
pd.DataFrame(ceil_rows).to_csv(CFG.outprefix + "_taskC_ceiling.csv",
                               index=False)
say("done")
