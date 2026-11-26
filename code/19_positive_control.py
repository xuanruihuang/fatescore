"""How large an increment this design can actually detect.

Usage: 19_positive_control.py <featdir> <labels.tsv> <outprefix>

Adding a block of components to a model that already predicts well is not free:
the classifier and its single regularisation strength are chosen over the
concatenated blocks, so a block that carries little costs something even when
it carries nothing harmful. Before any null can be read as an absence, the
assay has to be calibrated on blocks whose information content is known.

Two calibrations are run, both added in exactly the position and with exactly
the number of components the splicing and velocity blocks occupy.

1. A synthetic block built from the clone's own neutrophil fraction at a series
   of signal-to-noise levels, from zero (pure noise) upward. Its own predictive
   value is measured by scoring it alone, so the recovered increment can be
   plotted against the information actually present.

2. The complement of a read split at a series of depths: the spliced reads are
   split into two parts and the second part is given one, two, four or eight
   times the cell's unspliced depth. More reads means strictly more
   information about abundance, so the increment must rise with depth if the
   assay can see abundance information at all.
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
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--n-jobs", type=int, default=24)
ap.add_argument("--pcs", type=int, default=50)
ap.add_argument("--pos", default="Neutrophil")
ap.add_argument("--neg", default="Monocyte")
ap.add_argument("--pop-ref", default="LSK")
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
sel = (T["n_sis"] >= CFG.min_sis) & T["dom"].isin([CFG.pos, CFG.neg])
idx = np.where(sel.values)[0]
y = (T.loc[sel, "dom"].values == CFG.pos).astype(int)
groups = T.loc[sel, "clone"].values
f_pos = T.loc[sel, "f_neu"].values.astype(float)
f_pos = np.where(np.isfinite(f_pos), f_pos, y.astype(float))
S, U = raw_s[idx], raw_u[idx]
u_depth = U.sum(1)
say("task: %d cells, %d clones" % (len(idx), len(np.unique(groups))))

pca_blocks = {"state_s": cp10k_log(S), "unspl": cp10k_log(U)}
raw_blocks = {"pop": (T.loc[sel, "pop"].values == CFG.pop_ref)
              .astype(float)[:, None]}
jobs = []

# ---- 1. a synthetic block of known strength -------------------------------
LEVELS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
rng = np.random.default_rng(21)
W = rng.normal(size=(1, CFG.pcs))
sig = (f_pos - f_pos.mean())[:, None] * W
for a in LEVELS:
    for s in range(3):
        r = np.random.default_rng(900 + 17 * s)
        key = "synth_a%g_s%d" % (a, s)
        pca_blocks[key] = a * sig + r.normal(size=(len(idx), CFG.pcs))
        jobs.append(("P_synth_a%g_s%d" % (a, s), ["pop", "state_s", key]))
        jobs.append(("Palone_synth_a%g_s%d" % (a, s), ["pop", key]))

# ---- 2. the read-split complement at a series of depths -------------------
DEPTHS = [1, 2, 4, 8]
for k in DEPTHS:
    for s in range(3):
        part2 = thin_to(S, np.minimum(u_depth * k, S.sum(1)), seed=1100 + s)
        part1 = S - part2
        pca_blocks["p1_k%d_s%d" % (k, s)] = cp10k_log(part1)
        pca_blocks["p2_k%d_s%d" % (k, s)] = cp10k_log(part2)
        jobs.append(("Q_base_k%d_s%d" % (k, s), ["pop", "p1_k%d_s%d" % (k, s)]))
        jobs.append(("Q_plus_k%d_s%d" % (k, s),
                     ["pop", "p1_k%d_s%d" % (k, s), "p2_k%d_s%d" % (k, s)]))

jobs.append(("A2_state_spliced", ["pop", "state_s"]))
jobs.append(("A6_state_plus_unspliced", ["pop", "state_s", "unspl"]))
say("%d jobs on %d workers" % (len(jobs), CFG.n_jobs))

res = Parallel(n_jobs=CFG.n_jobs, verbose=5)(
    delayed(run_arm)(a, b, pca_blocks, raw_blocks, y, groups, 0, CFG.folds,
                     CFG.pcs) for a, b in jobs)
score = {}
for armname, seed, oof, picked in res:
    score[armname] = auroc(clone_level(oof, y, groups))

rows = []
base = score["A2_state_spliced"]
for a in LEVELS:
    inc = [score["P_synth_a%g_s%d" % (a, s)] - base for s in range(3)]
    alone = [score["Palone_synth_a%g_s%d" % (a, s)] for s in range(3)]
    rows.append(dict(kind="synthetic block", level=a,
                     block_alone_auroc=float(np.mean(alone)),
                     increment=float(np.mean(inc)),
                     increment_sd=float(np.std(inc))))
for k in DEPTHS:
    inc = [score["Q_plus_k%d_s%d" % (k, s)] - score["Q_base_k%d_s%d" % (k, s)]
           for s in range(3)]
    b = [score["Q_base_k%d_s%d" % (k, s)] for s in range(3)]
    rows.append(dict(kind="read-split complement", level=k,
                     block_alone_auroc=np.nan, base_auroc=float(np.mean(b)),
                     increment=float(np.mean(inc)),
                     increment_sd=float(np.std(inc))))
rows.append(dict(kind="reference", level=np.nan, block_alone_auroc=np.nan,
                 increment=score["A6_state_plus_unspliced"] - base,
                 increment_sd=np.nan))
R = pd.DataFrame(rows)
R.to_csv(CFG.outprefix + "_calibration.csv", index=False)
print(R.to_string(index=False))
pd.DataFrame([dict(arm=k, auroc=v) for k, v in score.items()]).to_csv(
    CFG.outprefix + "_arms.csv", index=False)
say("done")
