"""Model against the precision of the label, on the same clones, with an
interval on the difference.

Usage: 16_ceiling_ci.py <ev_prefix> <labels.tsv> <out.csv> [--pos P --neg N]

On the clones whose late cells appear in both independent read-outs, one
read-out's positive-class fraction is used to predict the other read-out's
majority call. That is the sister oracle: a predictor with access to the
clone's own differentiated descendants, scored with exactly the metric the
models are scored with, on exactly the same clones. The difference between a
model and the oracle is bootstrapped over clones as a paired quantity.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("ev_prefix")
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--pos", default="Neutrophil")
ap.add_argument("--neg", default="Monocyte")
ap.add_argument("--min-per-readout", type=int, default=1)
ap.add_argument("--min-clones", type=int, default=20,
                help="a direction is scored only if this many clones have "
                     "both read-outs; below it the comparison is reported "
                     "as unresolvable rather than estimated")
ap.add_argument("--repeat", type=int, default=0,
                help="which fold assignment to score; -1 averages all of "
                     "them, which is an ensemble and scores better")
CFG = ap.parse_args()

task = json.load(open(CFG.ev_prefix + "_task.json"))
O = np.load(CFG.ev_prefix + "_oof.npy")          # (arm, repeat, cell)
arms = task["arms"]
rows_idx = np.array(task["task_rows"])
groups = np.array(task["clones"])
T = pd.read_csv(CFG.labels, sep="\t").iloc[rows_idx].reset_index(drop=True)

def oof_of(i):
    return O[i].mean(0) if CFG.repeat < 0 else O[i, CFG.repeat]


model = {a: (pd.DataFrame(dict(p=oof_of(i), g=groups))
             .groupby("g")["p"].mean()) for i, a in enumerate(arms)}

cl = (T.groupby("clone")
      .agg(dom_w1=("dom_w1", "first"), dom_w2=("dom_w2", "first"),
           n_neu_w1=("n_neu_w1", "first"), n_mono_w1=("n_mono_w1", "first"),
           n_neu_w2=("n_neu_w2", "first"), n_mono_w2=("n_mono_w2", "first")))
cl["nm1"] = cl.n_neu_w1 + cl.n_mono_w1
cl["nm2"] = cl.n_neu_w2 + cl.n_mono_w2
k = CFG.min_per_readout
base = cl[(cl.nm1 >= k) & (cl.nm2 >= k)]

out = []
for src, tgt in ((1, 2), (2, 1)):
    dt = "dom_w%d" % tgt
    sel = base[base[dt].isin([CFG.pos, CFG.neg])]
    if len(sel) < CFG.min_clones:
        continue
    lab = (sel[dt] == CFG.pos).astype(int).values
    oracle = (sel["n_neu_w%d" % src] / sel["nm%d" % src]).values
    scores = {"sister_oracle": oracle}
    for a in arms:
        scores[a] = model[a].reindex(sel.index).values
    idx = np.arange(len(sel))
    NB = 2000
    rng = np.random.default_rng(13)
    boot = {kk: np.empty(NB) for kk in scores}
    for b in range(NB):
        take = rng.integers(0, len(idx), len(idx))
        if len(np.unique(lab[take])) < 2:
            for kk in scores:
                boot[kk][b] = np.nan
            continue
        for kk, v in scores.items():
            boot[kk][b] = roc_auc_score(lab[take], v[take])
    oracle_acc = float(((oracle > 0.5).astype(int) == lab).mean())
    ties = float(np.mean([
        (oracle[lab == 1][:, None] == oracle[lab == 0][None, :]).mean()
    ])) if (lab == 1).any() and (lab == 0).any() else np.nan
    for a in arms:
        d = boot[a] - boot["sister_oracle"]
        out.append(dict(direction="readout%d_predicts_readout%d" % (src, tgt),
                        n_clones=len(sel), arm=a, repeat=CFG.repeat,
                        auroc=roc_auc_score(lab, scores[a]),
                        oracle_auroc=roc_auc_score(lab, oracle),
                        acc=float(((scores[a] > 0.5).astype(int) == lab).mean()),
                        oracle_acc=oracle_acc,
                        oracle_tied_pair_fraction=ties,
                        diff=roc_auc_score(lab, scores[a])
                        - roc_auc_score(lab, oracle),
                        lo=np.nanpercentile(d, 2.5),
                        hi=np.nanpercentile(d, 97.5)))

R = pd.DataFrame(out)
R.to_csv(CFG.out, index=False)
if len(R):
    M = (R.groupby("arm")
         .agg(auroc=("auroc", "mean"), oracle=("oracle_auroc", "mean"),
              diff=("diff", "mean"), lo=("lo", "mean"), hi=("hi", "mean"),
              n_clones=("n_clones", "first"))
         .sort_values("diff", ascending=False))
    print(M.to_string())
