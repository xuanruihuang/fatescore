"""Clone-grouped evaluation of every feature arm, for one kNN-graph mode.

Usage: 03_evaluate.py <featdir> <labels.tsv> <outprefix> [--repeats N]

Task A: predict whether a day-2 cell's clone ends up mostly neutrophil or
mostly monocyte, from that cell alone. Folds are disjoint in clones. Every
preprocessing step that could see a label (PCA, scaling, classifier choice and
hyperparameters) is fitted inside the training folds. Predictions are averaged
within a clone before any metric is computed, because the clone is the
independent unit.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cvlib import cp10k_log, thin_to, run_arm, clone_level, auroc

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("outprefix")
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--n-jobs", type=int, default=24)
ap.add_argument("--pcs", type=int, default=50)
ap.add_argument("--pos", default="Neutrophil",
                help="dominant fate counted as the positive class")
ap.add_argument("--neg", default="Monocyte",
                help="dominant fate counted as the negative class")
ap.add_argument("--pop-ref", default="LSK",
                help="value of the pop column encoded as 1")
ap.add_argument("--restrict-pop", default="",
                help="keep only cells whose pop column equals this")
CFG = ap.parse_args()

T = pd.read_csv(CFG.labels, sep="\t")
d2_names = json.load(open(os.path.join(CFG.featdir, "d2_names.json")))
assert list(T["cell"]) == d2_names, "label table and feature rows disagree"


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def load(name):
    p = os.path.join(CFG.featdir, name + ".npy")
    return np.load(p) if os.path.exists(p) else None


raw_s, raw_u, raw_tot = load("raw_spliced"), load("raw_unspliced"), load("raw_matrix")
Ms, Mu = load("Ms"), load("Mu")
vel = {k: load("vel_" + k) for k in ("det", "dyn", "velovi")}
crf = {k: load("cr_" + k) for k in ("velo", "pseudo")}
say("loaded; missing arrays: %s"
    % [k for k, v in list(vel.items()) + list(crf.items()) if v is None])

# A second copy of the spliced layer, thinned to each cell's own unspliced
# depth. Added in the same position as the unspliced layer, it says what a
# low-depth repeat measurement of abundance alone is worth.
depth = thin_to(raw_s, raw_u.sum(1), seed=0)
say("depth-matched subsample built; median target depth %.0f"
    % np.median(raw_u.sum(1)))

PCA_BLOCKS = {
    "state_s": cp10k_log(raw_s),
    "state_tot": cp10k_log(raw_tot),
    "unspl": cp10k_log(raw_u),
    "Ms": np.log1p(np.clip(Ms, 0, None)),
    "ratio": Mu / (Ms + Mu + 1e-8),
    "depth": cp10k_log(depth),
}
for k, v in vel.items():
    if v is not None:
        PCA_BLOCKS["vel_" + k] = v
RAW_BLOCKS = {"pop": (T["pop"].values == CFG.pop_ref).astype(float)[:, None]}
for k, v in crf.items():
    if v is not None:
        RAW_BLOCKS["cr_" + k] = v

ARMS = {
    "A0_prior": [],
    "A1_pop": ["pop"],
    "A2_state_spliced": ["pop", "state_s"],
    "A3_state_total": ["pop", "state_tot"],
    "A4_state_smoothed": ["pop", "Ms"],
    "A5_unspliced_only": ["pop", "unspl"],
    "A6_state_plus_unspliced": ["pop", "state_s", "unspl"],
    "A7_smoothed_plus_ratio": ["pop", "Ms", "ratio"],
    "N2_state_plus_depthmatched": ["pop", "state_s", "depth"],
    "V1_smoothed_plus_vel_det": ["pop", "Ms", "vel_det"],
    "V2_smoothed_plus_vel_dyn": ["pop", "Ms", "vel_dyn"],
    "V3_smoothed_plus_vel_velovi": ["pop", "Ms", "vel_velovi"],
    "V4_smoothed_plus_cr_velo": ["pop", "Ms", "cr_velo"],
    "V5_smoothed_plus_cr_pseudo": ["pop", "Ms", "cr_pseudo"],
    "V4a_cr_velo_only": ["pop", "cr_velo"],
    "V5a_cr_pseudo_only": ["pop", "cr_pseudo"],
    # velocity on its own, so that a null increment cannot be blamed on the
    # extra fifty components diluting a model that was already at its ceiling
    "V1a_vel_det_only": ["pop", "vel_det"],
    "V2a_vel_dyn_only": ["pop", "vel_dyn"],
    "V3a_vel_velovi_only": ["pop", "vel_velovi"],
    # the same comparison with ten components per block instead of fifty
    "A4b_state_smoothed_10pc": ["pop", "Ms"],
    "V2b_smoothed_plus_vel_dyn_10pc": ["pop", "Ms", "vel_dyn"],
}
PCS_BY_ARM = {
    "A4b_state_smoothed_10pc": {"Ms": 10},
    "V2b_smoothed_plus_vel_dyn_10pc": {"Ms": 10, "vel_dyn": 10},
}
ARMS = {k: v for k, v in ARMS.items()
        if all(b in PCA_BLOCKS or b in RAW_BLOCKS for b in v)}
say("arms: %s" % list(ARMS))

sel = (T["n_sis"] >= CFG.min_sis) & T["dom"].isin([CFG.pos, CFG.neg])
if CFG.restrict_pop:
    sel = sel & (T["pop"] == CFG.restrict_pop)
idx = np.where(sel.values)[0]
y = (T.loc[sel, "dom"].values == CFG.pos).astype(int)
groups = T.loc[sel, "clone"].values
say("task A: %d cells, %d clones, prevalence %.3f"
    % (len(idx), len(np.unique(groups)), y.mean()))

PCA_BLOCKS = {k: v[idx] for k, v in PCA_BLOCKS.items()}
RAW_BLOCKS = {k: v[idx] for k, v in RAW_BLOCKS.items()}

jobs = [(a, b, s) for a, b in ARMS.items() for s in range(CFG.repeats)]
say("%d (arm, repeat) jobs on %d workers" % (len(jobs), CFG.n_jobs))
res = Parallel(n_jobs=CFG.n_jobs, verbose=5)(
    delayed(run_arm)(a, b, PCA_BLOCKS, RAW_BLOCKS, y, groups, s, CFG.folds,
                     CFG.pcs, PCS_BY_ARM.get(a))
    for a, b, s in jobs)

OOF, rows = {}, []
for arm, seed, oof, picked in res:
    OOF[(arm, seed)] = oof
    d = clone_level(oof, y, groups)
    rows.append(dict(arm=arm, seed=seed, auroc=auroc(d),
                     acc=((d["p"] > 0.5).astype(int) == d["y"]).mean(),
                     n_clones=len(d), picked=";".join(sorted(set(picked)))))
R = pd.DataFrame(rows)
R.to_csv(CFG.outprefix + "_perrepeat.csv", index=False)
S = (R.groupby("arm")
     .agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
          acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
          n_clones=("n_clones", "first"))
     .sort_values("auroc_mean", ascending=False))
S.to_csv(CFG.outprefix + "_summary.csv")
say("\n" + S.to_string())

np.save(CFG.outprefix + "_oof.npy",
        np.stack([[OOF[(a, s)] for s in range(CFG.repeats)] for a in ARMS]))
json.dump(dict(arms=list(ARMS), repeats=CFG.repeats, task_rows=idx.tolist(),
               y=y.tolist(), clones=[g.item() if hasattr(g, "item") else str(g)
                       for g in groups]),
          open(CFG.outprefix + "_task.json", "w"))

# paired clone bootstrap on the first repeat's out-of-fold predictions
clones = np.unique(groups)
cl_tab = {a: clone_level(OOF[(a, 0)], y, groups).reindex(clones) for a in ARMS}
NB = 2000
rb = np.random.default_rng(1)
boot = {a: np.empty(NB) for a in ARMS}
for b in range(NB):
    take = rb.integers(0, len(clones), len(clones))
    for a in ARMS:
        boot[a][b] = auroc(cl_tab[a].iloc[take])
pd.DataFrame([dict(arm=a, auroc_seed0=auroc(cl_tab[a]),
                   lo=np.nanpercentile(boot[a], 2.5),
                   hi=np.nanpercentile(boot[a], 97.5)) for a in ARMS]
             ).to_csv(CFG.outprefix + "_bootstrap.csv", index=False)

pairs = []
for ref in [a for a in ("A2_state_spliced", "A4_state_smoothed") if a in ARMS]:
    for a in ARMS:
        if a == ref:
            continue
        d = boot[a] - boot[ref]
        pairs.append(dict(arm=a, ref=ref,
                          diff=auroc(cl_tab[a]) - auroc(cl_tab[ref]),
                          lo=np.nanpercentile(d, 2.5),
                          hi=np.nanpercentile(d, 97.5)))
P = pd.DataFrame(pairs)
P.to_csv(CFG.outprefix + "_pairs.csv", index=False)
say("\n" + P.to_string())
say("done")
