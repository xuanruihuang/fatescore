"""Which layer carries the batch, at matched depth.

Usage: 23_batch_carriage.py <featdir> <labels.tsv> <out.csv> [--pos P --neg N]

The read-splitting result on the second dataset disappears inside one batch,
which suggests the unspliced layer carries the difference between batches more
strongly than an equal-depth piece of the spliced layer does. That is a claim
about the data, so it is measured here rather than asserted: the same folds and
the same blocks are used to predict the batch label itself instead of the fate.
Folds remain disjoint in clones, so no clone contributes to both sides.
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
ap.add_argument("out")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--min-sis", type=int, default=1)
ap.add_argument("--n-jobs", type=int, default=20)
ap.add_argument("--pcs", type=int, default=50)
ap.add_argument("--pos", default="iEP")
ap.add_argument("--neg", default="Fibroblast")
ap.add_argument("--batch-ref", default="tc1")
ap.add_argument("--splits", type=int, default=10)
CFG = ap.parse_args()


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


T = pd.read_csv(CFG.labels, sep="\t")
assert list(T["cell"]) == json.load(
    open(os.path.join(CFG.featdir, "d2_names.json")))
raw_s = np.load(os.path.join(CFG.featdir, "raw_spliced.npy"))
raw_u = np.load(os.path.join(CFG.featdir, "raw_unspliced.npy"))

sel = (T["n_sis"] >= CFG.min_sis) & T["dom"].isin([CFG.pos, CFG.neg])
idx = np.where(sel.values)[0]
groups = T.loc[sel, "clone"].values
y = (T.loc[sel, "pop"].values == CFG.batch_ref).astype(int)
S, U = raw_s[idx], raw_u[idx]
say("%d cells, %d clones, batch prevalence %.3f"
    % (len(idx), len(np.unique(groups)), y.mean()))

pca_blocks = {"unspl": cp10k_log(U)}
raw_blocks = {}
jobs = [("unspliced layer", ["unspl"])]
for s in range(CFG.splits):
    part2 = thin_to(S, U.sum(1), seed=2200 + s)
    pca_blocks["p2_s%d" % s] = cp10k_log(part2)
    pca_blocks["p1_s%d" % s] = cp10k_log(S - part2)
    jobs.append(("equal-depth spliced complement, split %d" % s,
                 ["p2_s%d" % s]))
    jobs.append(("the rest of the spliced reads, split %d" % s,
                 ["p1_s%d" % s]))
pca_blocks["state_s"] = cp10k_log(S)
jobs.append(("all spliced counts", ["state_s"]))

res = Parallel(n_jobs=CFG.n_jobs, verbose=0)(
    delayed(run_arm)(a, b, pca_blocks, raw_blocks, y, groups, 0, CFG.folds,
                     CFG.pcs) for a, b in jobs)
rows = []
for arm, seed, oof, picked in res:
    rows.append(dict(block=arm, auroc=auroc(clone_level(oof, y, groups))))
R = pd.DataFrame(rows)


def fam(prefix):
    v = R[R.block.str.startswith(prefix)]["auroc"].values
    return (float(np.mean(v)), float(np.std(v)), len(v)) if len(v) else (
        np.nan, np.nan, 0)


out = [dict(block="unspliced layer",
            auroc=float(R[R.block == "unspliced layer"]["auroc"].iloc[0]),
            sd=np.nan, n_runs=1),
       dict(block="all spliced counts",
            auroc=float(R[R.block == "all spliced counts"]["auroc"].iloc[0]),
            sd=np.nan, n_runs=1)]
for name, pref in (("equal-depth spliced complement",
                    "equal-depth spliced complement"),
                   ("the rest of the spliced reads",
                    "the rest of the spliced reads")):
    m, sd, n = fam(pref)
    out.append(dict(block=name, auroc=m, sd=sd, n_runs=n))
# the composition that makes the batch worth something about the fate
comp = (pd.DataFrame(dict(clone=groups,
                          batch=T.loc[sel, "pop"].values,
                          fate=T.loc[sel, "dom"].values))
        .groupby("clone").first())
for b in sorted(comp["batch"].unique()):
    sub = comp[comp["batch"] == b]
    out.append(dict(block="clones in %s" % b, auroc=float(len(sub)),
                    sd=np.nan, n_runs=np.nan))
    out.append(dict(block="of those with fate %s in %s" % (CFG.pos, b),
                    auroc=float((sub["fate"] == CFG.pos).sum()), sd=np.nan,
                    n_runs=np.nan))
O = pd.DataFrame(out)
O.to_csv(CFG.out, index=False)
print(O.to_string(index=False))
say("done")
