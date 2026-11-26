"""The same comparison inside each sorted progenitor pool.

Usage: 08_within_pop.py <featdir> <labels.tsv> <outprefix>

The two starting populations, Lin-Kit+Sca1+ and Lin-Kit+Sca1-, differ in their
fate composition and were loaded as separate libraries, so a model trained
across both can score well by recognising the sorting gate rather than the
cell. Here the folds are built inside one population at a time: training and
test cells come from the same gate, so whatever is predicted has to come from
the transcriptome.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cvlib import cp10k_log, run_arm, clone_level, auroc

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("outprefix")
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--n-jobs", type=int, default=24)
ap.add_argument("--pcs", type=int, default=50)
ap.add_argument("--pos", default="Neutrophil")
ap.add_argument("--neg", default="Monocyte")
ap.add_argument("--pops", default="LSK,LK",
                help="comma-separated values of the pop column to split on")
CFG = ap.parse_args()


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


T = pd.read_csv(CFG.labels, sep="\t")
assert list(T["cell"]) == json.load(
    open(os.path.join(CFG.featdir, "d2_names.json")))


def load(n):
    p = os.path.join(CFG.featdir, n + ".npy")
    return np.load(p) if os.path.exists(p) else None


raw_s, raw_u = load("raw_spliced"), load("raw_unspliced")
Ms, Mu = load("Ms"), load("Mu")
vel = {k: load("vel_" + k) for k in ("det", "dyn", "velovi")}
crf = {k: load("cr_" + k) for k in ("velo", "pseudo")}

PCA_ALL = {
    "state_s": cp10k_log(raw_s), "unspl": cp10k_log(raw_u),
    "Ms": np.log1p(np.clip(Ms, 0, None)), "ratio": Mu / (Ms + Mu + 1e-8),
}
for k, v in vel.items():
    if v is not None:
        PCA_ALL["vel_" + k] = v
RAW_ALL = {}
for k, v in crf.items():
    if v is not None:
        RAW_ALL["cr_" + k] = v

ARMS = {
    "A2_state_spliced": ["state_s"],
    "A4_state_smoothed": ["Ms"],
    "A5_unspliced_only": ["unspl"],
    "A6_state_plus_unspliced": ["state_s", "unspl"],
    "A7_smoothed_plus_ratio": ["Ms", "ratio"],
    "V1_smoothed_plus_vel_det": ["Ms", "vel_det"],
    "V2_smoothed_plus_vel_dyn": ["Ms", "vel_dyn"],
    "V3_smoothed_plus_vel_velovi": ["Ms", "vel_velovi"],
    "V4_smoothed_plus_cr_velo": ["Ms", "cr_velo"],
    "V5_smoothed_plus_cr_pseudo": ["Ms", "cr_pseudo"],
}
ARMS = {k: v for k, v in ARMS.items()
        if all(b in PCA_ALL or b in RAW_ALL for b in v)}

rows, boot_rows = [], []
for popname in CFG.pops.split(","):
    sel = ((T["n_sis"] >= CFG.min_sis) & T["dom"].isin([CFG.pos, CFG.neg])
           & (T["pop"] == popname))
    idx = np.where(sel.values)[0]
    y = (T.loc[sel, "dom"].values == CFG.pos).astype(int)
    g = T.loc[sel, "clone"].values
    say("%s: %d cells, %d clones, prevalence %.3f"
        % (popname, len(idx), len(np.unique(g)), y.mean()))
    pca = {k: v[idx] for k, v in PCA_ALL.items()}
    rawb = {k: v[idx] for k, v in RAW_ALL.items()}
    jobs = [(a, b, s) for a, b in ARMS.items() for s in range(CFG.repeats)]
    res = Parallel(n_jobs=CFG.n_jobs, verbose=0)(
        delayed(run_arm)(a, b, pca, rawb, y, g, s, CFG.folds, CFG.pcs)
        for a, b, s in jobs)
    oof_by_arm = {}
    for arm, seed, oof, picked in res:
        d = clone_level(oof, y, g)
        oof_by_arm.setdefault(arm, {})[seed] = oof
        rows.append(dict(pop=popname, arm=arm, seed=seed, auroc=auroc(d),
                         acc=float(((d["p"] > 0.5).astype(int) == d["y"]).mean()),
                         n_clones=len(d), prevalence=float(y.mean())))
    # paired bootstrap over clones, against the spliced-count model, on the
    # first fold assignment
    clones = np.unique(g)
    tab = {a: clone_level(v[0], y, g).reindex(clones)
           for a, v in oof_by_arm.items()}
    NB = 2000
    rb = np.random.default_rng(5)
    boot = {a: np.empty(NB) for a in tab}
    for b in range(NB):
        take = rb.integers(0, len(clones), len(clones))
        for a, t in tab.items():
            boot[a][b] = auroc(t.iloc[take])
    ref = "A2_state_spliced"
    for a in tab:
        if a == ref:
            continue
        dd = boot[a] - boot[ref]
        boot_rows.append(dict(pop=popname, arm=a, ref=ref,
                              diff=auroc(tab[a]) - auroc(tab[ref]),
                              lo=float(np.nanpercentile(dd, 2.5)),
                              hi=float(np.nanpercentile(dd, 97.5)),
                              n_clones=len(clones)))

R = pd.DataFrame(rows)
R.to_csv(CFG.outprefix + "_perrepeat.csv", index=False)
S = (R.groupby(["pop", "arm"])
     .agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
          acc=("acc", "mean"), n_clones=("n_clones", "first"),
          prevalence=("prevalence", "first"))
     .reset_index().sort_values(["pop", "auroc"], ascending=[True, False]))
S.to_csv(CFG.outprefix + "_summary.csv", index=False)
B = pd.DataFrame(boot_rows)
B.to_csv(CFG.outprefix + "_pairs.csv", index=False)
print(B.to_string(index=False))
print(S.to_string(index=False))
say("done")
