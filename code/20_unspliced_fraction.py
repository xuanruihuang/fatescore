"""Does the per-cell unspliced fraction separate the fates on its own?

Usage: 20_unspliced_fraction.py <featdir> <labels.tsv> <out.csv> [--pos P --neg N]

The whole-transcriptome check reports a difference in the per-cell unspliced
fraction between neutrophil-fated and monocyte-fated clones, but it tests it
over cells, which are not independent because a clone contributes several. Here
the quantity is averaged within a clone first, and the null is built by
permuting the fate labels over clones, which is the unit the design makes
independent.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--pos", default="Neutrophil")
ap.add_argument("--neg", default="Monocyte")
ap.add_argument("--perms", type=int, default=20000)
CFG = ap.parse_args()

T = pd.read_csv(CFG.labels, sep="\t")
assert list(T["cell"]) == json.load(
    open(os.path.join(CFG.featdir, "d2_names.json")))
tot_s = np.load(os.path.join(CFG.featdir, "total_spliced.npy"))
tot_u = np.load(os.path.join(CFG.featdir, "total_unspliced.npy"))
frac = tot_u / np.maximum(tot_u + tot_s, 1)

sel = (T["n_sis"] >= CFG.min_sis) & T["dom"].isin([CFG.pos, CFG.neg])
idx = np.where(sel.values)[0]
y = (T.loc[sel, "dom"].values == CFG.pos).astype(int)
g = T.loc[sel, "clone"].values
d = (pd.DataFrame(dict(f=frac[idx], y=y, g=g)).groupby("g")
     .agg(f=("f", "mean"), y=("y", "first")))

obs_auc = roc_auc_score(d["y"], d["f"])
obs_diff = float(d[d.y == 1]["f"].mean() - d[d.y == 0]["f"].mean())
rng = np.random.default_rng(31)
null_auc = np.empty(CFG.perms)
lab = d["y"].values.copy()
for b in range(CFG.perms):
    null_auc[b] = roc_auc_score(rng.permutation(lab), d["f"].values)
p_two = float((np.abs(null_auc - 0.5) >= abs(obs_auc - 0.5)).mean())

rows = [
    dict(quantity="clones", value=len(d)),
    dict(quantity="cells", value=len(idx)),
    dict(quantity="clone-mean unspliced fraction, %s" % CFG.pos,
         value=float(d[d.y == 1]["f"].mean())),
    dict(quantity="clone-mean unspliced fraction, %s" % CFG.neg,
         value=float(d[d.y == 0]["f"].mean())),
    dict(quantity="difference", value=obs_diff),
    dict(quantity="clone-level AUROC of the unspliced fraction alone",
         value=float(obs_auc)),
    dict(quantity="two-sided permutation p over clones", value=p_two),
    dict(quantity="permutations", value=CFG.perms),
]
R = pd.DataFrame(rows)
R.to_csv(CFG.out, index=False)
print(R.to_string(index=False))
