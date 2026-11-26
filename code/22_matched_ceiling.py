"""The model and the sister oracle given exactly the same information.

Usage: 22_matched_ceiling.py <featdir> <labels.tsv> <out.csv> [--pos P --neg N]

In the comparison of 16_ceiling_ci.py the model is trained on labels pooled
over both wells, while the oracle sees one well only. Both are scored against
the same target, so neither sees the answer, but the model's training target is
the cleaner of the two. Here that advantage is removed: for each target well,
the models are trained on labels built from the *other* well alone, so model
and oracle rest on the same observations of each clone, and both are scored
against the target well's majority call.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from cvlib import cp10k_log, run_arm, clone_level

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
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


def load(n):
    p = os.path.join(CFG.featdir, n + ".npy")
    return np.load(p) if os.path.exists(p) else None


PCA_ALL = {"state_s": cp10k_log(load("raw_spliced")),
           "state_tot": cp10k_log(load("raw_matrix")),
           "Ms": np.log1p(np.clip(load("Ms"), 0, None))}
RAW_ALL = {"pop": (T["pop"].values == CFG.pop_ref).astype(float)[:, None]}
ARMS = {"A2_state_spliced": ["pop", "state_s"],
        "A3_state_total": ["pop", "state_tot"],
        "A4_state_smoothed": ["pop", "Ms"]}

rows = []
for src, tgt in ((1, 2), (2, 1)):
    dom_src, dom_tgt = "dom_w%d" % src, "dom_w%d" % tgt
    nm_src = T["n_neu_w%d" % src] + T["n_mono_w%d" % src]
    nm_tgt = T["n_neu_w%d" % tgt] + T["n_mono_w%d" % tgt]
    # cells whose clone has a call in the source well: these train the model
    train_ok = T[dom_src].isin([CFG.pos, CFG.neg]) & (nm_src >= 1)
    idx = np.where(train_ok.values)[0]
    y = (T.loc[train_ok, dom_src].values == CFG.pos).astype(int)
    g = T.loc[train_ok, "clone"].values
    say("well %d as the source: %d cells, %d clones"
        % (src, len(idx), len(np.unique(g))))

    pca = {k: v[idx] for k, v in PCA_ALL.items()}
    rawb = {k: v[idx] for k, v in RAW_ALL.items()}
    jobs = [(a, b, s) for a, b in ARMS.items() for s in range(CFG.repeats)]
    res = Parallel(n_jobs=CFG.n_jobs, verbose=0)(
        delayed(run_arm)(a, b, pca, rawb, y, g, s, CFG.folds, CFG.pcs)
        for a, b, s in jobs)
    oof = {}
    for arm, seed, p, picked in res:
        oof.setdefault(arm, {})[seed] = p

    # score on the clones that both wells reach, against the target well
    sub = T.loc[train_ok].reset_index(drop=True)
    ok = (sub[dom_tgt].isin([CFG.pos, CFG.neg])
          & (nm_tgt[train_ok].values >= 1)).values
    lab_cell = (sub.loc[ok, dom_tgt].values == CFG.pos).astype(int)
    g_ok = sub.loc[ok, "clone"].values
    oracle_cell = (sub.loc[ok, "n_neu_w%d" % src].values
                   / np.maximum(nm_src[train_ok].values[ok], 1))
    d = (pd.DataFrame(dict(p=oracle_cell, y=lab_cell, g=g_ok)).groupby("g")
         .agg(p=("p", "mean"), y=("y", "first")))
    clones = d.index.values
    say("  scored on %d clones" % len(d))
    scores = {"sister_oracle": d["p"].values}
    lab = d["y"].values
    for arm in ARMS:
        s0 = oof[arm][0][ok]
        q = (pd.DataFrame(dict(p=s0, g=g_ok)).groupby("g")["p"].mean()
             .reindex(clones).values)
        scores[arm] = q
    NB = 2000
    rb = np.random.default_rng(17)
    boot = {k: np.empty(NB) for k in scores}
    for b in range(NB):
        take = rb.integers(0, len(lab), len(lab))
        if len(np.unique(lab[take])) < 2:
            for k in scores:
                boot[k][b] = np.nan
            continue
        for k, v in scores.items():
            boot[k][b] = roc_auc_score(lab[take], v[take])
    for k, v in scores.items():
        dd = boot[k] - boot["sister_oracle"]
        rows.append(dict(direction="well%d_predicts_well%d" % (src, tgt),
                         n_clones=len(lab), who=k,
                         auroc=float(roc_auc_score(lab, v)),
                         acc=float(((v > 0.5).astype(int) == lab).mean()),
                         diff_vs_oracle=float(roc_auc_score(lab, v)
                                              - roc_auc_score(lab,
                                                              scores["sister_oracle"])),
                         lo=float(np.nanpercentile(dd, 2.5)),
                         hi=float(np.nanpercentile(dd, 97.5))))

R = pd.DataFrame(rows)
R.to_csv(CFG.out, index=False)
print(R.to_string(index=False))
say("done")
