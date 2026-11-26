"""A resolution scale for the read-splitting comparison that holds the base fixed.

Usage: 54_readsplit_scale.py <featdir> <labels.tsv> <outprefix> [--splits 5]

Why. 19_positive_control.py calibrated the read-splitting comparison by giving
the second part of a split one, two, four or eight times the cell's unspliced
depth. But the two parts are complementary, so every read added to the second
part is taken from the first, which is the model's base: the base AUROC falls
from 0.849 (one depth) to 0.835, 0.761 and 0.646, while base plus complement is
always the full spliced counts. The "increment" at each depth is mostly the
base getting weaker, so it is not a scale for what an added block is worth.
Found by a reader of the draft on 2026-11-23.

What this does instead. Each cell's spliced counts on the selected genes are
split three ways by multivariate hypergeometric sampling: c1 with exactly the
cell's unspliced depth, c2 with the same depth again from what is left, and the
rest as the base. The base is the same in every arm, so differences between
arms are differences between added blocks only:

  R0  gate + base
  R1  gate + base + c1                 one equal-depth repeat of abundance
  R2  gate + base + (c1 + c2)          two equal-depth repeats, one block
  R3  gate + base + unspliced          the unspliced layer in the same place

Cells whose spliced depth is below twice their unspliced depth give what they
have to c1 and c2 and are left with a thinner (possibly empty) base; their
number is reported. Every block enters as 50 principal components fitted in
the training folds, with the same classifier search as every other arm.

The quantities are the paired differences R1-R0, R2-R0, R3-R0, R2-R1 (what a
second repeat adds on top of the first: the scale), R3-R1 (the unspliced layer
against one repeat) and R3-R2 (against two). Following 15_readsplit_ci.py, the
AUROC is computed for each split separately inside every bootstrap resample
and then averaged over splits, so runs are never pooled into an ensemble;
2,000 resamples over clones.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cvlib import cp10k_log, thin_to, run_arm, auroc

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("outprefix")
ap.add_argument("--splits", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--n-jobs", type=int, default=20)
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
raw_s = np.load(os.path.join(CFG.featdir, "raw_spliced.npy"))
raw_u = np.load(os.path.join(CFG.featdir, "raw_unspliced.npy"))
sel = (T["n_sis"] >= CFG.min_sis) & T["dom"].isin([CFG.pos, CFG.neg])
idx = np.where(sel.values)[0]
y = (T.loc[sel, "dom"].values == CFG.pos).astype(int)
groups = T.loc[sel, "clone"].values
S, U = raw_s[idx], raw_u[idx]
u_depth = U.sum(1)
s_depth = S.sum(1)
short = int((s_depth < 2 * u_depth).sum())
say("task: %d cells, %d clones; median spliced %d, unspliced %d; %d cells with "
    "spliced depth below twice unspliced" % (len(idx), len(np.unique(groups)),
                                              np.median(s_depth),
                                              np.median(u_depth), short))

pca_blocks = {"unspl": cp10k_log(U)}
raw_blocks = {"pop": (T.loc[sel, "pop"].values == CFG.pop_ref)
              .astype(float)[:, None]}
jobs = []
base_depth = []
for s in range(CFG.splits):
    c1 = thin_to(S, u_depth, seed=5400 + s)
    rest = S - c1
    c2 = thin_to(rest, u_depth, seed=5500 + s)
    base = rest - c2
    assert (base >= 0).all() and (c1 + c2 + base == S).all()
    base_depth.append(float(np.median(base.sum(1))))
    pca_blocks["base_s%d" % s] = cp10k_log(base)
    pca_blocks["c1_s%d" % s] = cp10k_log(c1)
    pca_blocks["c12_s%d" % s] = cp10k_log(c1 + c2)
    jobs += [("R0_s%d" % s, ["pop", "base_s%d" % s]),
             ("R1_s%d" % s, ["pop", "base_s%d" % s, "c1_s%d" % s]),
             ("R2_s%d" % s, ["pop", "base_s%d" % s, "c12_s%d" % s]),
             ("R3_s%d" % s, ["pop", "base_s%d" % s, "unspl"])]
say("median base depth over splits: %s" % base_depth)
say("%d jobs on %d workers" % (len(jobs), CFG.n_jobs))
res = Parallel(n_jobs=CFG.n_jobs, verbose=5)(
    delayed(run_arm)(a, b, pca_blocks, raw_blocks, y, groups, 0, CFG.folds,
                     CFG.pcs) for a, b in jobs)
oof = {a: o for a, _, o, _ in res}
np.save(CFG.outprefix + "_oof.npy", np.array([oof[a] for a, _ in jobs]))
json.dump([a for a, _ in jobs], open(CFG.outprefix + "_armnames.json", "w"))

clones = np.unique(groups)


def tab(p):
    return (pd.DataFrame(dict(p=p, y=y, g=groups)).groupby("g")
            .agg(p=("p", "mean"), y=("y", "first")).reindex(clones))


tabs = {r: [tab(oof["%s_s%d" % (r, s)]) for s in range(CFG.splits)]
        for r in ("R0", "R1", "R2", "R3")}
point = {r: float(np.mean([auroc(t) for t in ts])) for r, ts in tabs.items()}
NB = 2000
rng = np.random.default_rng(11)
boot = {r: np.empty(NB) for r in tabs}
for b in range(NB):
    take = rng.integers(0, len(clones), len(clones))
    for r, ts in tabs.items():
        boot[r][b] = np.mean([auroc(t.iloc[take]) for t in ts])
rows = [dict(quantity=r, value=point[r], lo=np.nanpercentile(boot[r], 2.5),
             hi=np.nanpercentile(boot[r], 97.5)) for r in tabs]
for lab, a, b in [("one repeat added (R1-R0)", "R1", "R0"),
                  ("two repeats added (R2-R0)", "R2", "R0"),
                  ("unspliced added (R3-R0)", "R3", "R0"),
                  ("second repeat on top of the first (R2-R1)", "R2", "R1"),
                  ("unspliced minus one repeat (R3-R1)", "R3", "R1"),
                  ("unspliced minus two repeats (R3-R2)", "R3", "R2")]:
    d = boot[a] - boot[b]
    rows.append(dict(quantity=lab, value=point[a] - point[b],
                     lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5)))
R = pd.DataFrame(rows)
R.to_csv(CFG.outprefix + ".csv", index=False)
print(R.to_string(index=False), flush=True)
