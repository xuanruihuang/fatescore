"""Learning curve of the expression baseline: is the room below the ceiling
estimation or information?

The expression baseline (total counts with the sorted population) lies
0.06-0.09 AUROC below the noise ceiling of the label. Part of that gap could be
finite-sample estimation rather than fate information the transcriptome lacks.
Here the training part of every fold is subsampled by clone to a fraction of
its size (the test part is unchanged), with the same classifier search, PCA
inside the fold and clone-level AUROC as every other arm. If AUROC is still
rising at the full training size, part of the gap is estimation.

Usage: 76_learning_curve.py <featdir_d2> <labels.tsv> <out.csv>
         [--fractions 0.2,0.4,0.6,0.8,1.0] [--seeds 5] [--workers 25]
         [--arms total,spliced]   (also total_nopop: expression without the sorted population)
"""
import argparse
import json
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvlib import GRIDS, build, cp10k_log, folds_of, make, clone_level, auroc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--fractions", default="0.2,0.4,0.6,0.8,1.0")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--workers", type=int, default=25)
ap.add_argument("--arms", default="total,spliced")
a = ap.parse_args()

T = pd.read_csv(a.labels, sep="\t")
assert list(T["cell"]) == json.load(open(os.path.join(a.featdir, "d2_names.json")))
m = ((T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])).values
y = (T.loc[m, "dom"].values == "Neutrophil").astype(int)
groups = T.loc[m, "clone"].values
PCA_BLOCKS = {"state_tot": cp10k_log(np.load(os.path.join(a.featdir, "raw_matrix.npy"))[m]),
              "state_s": cp10k_log(np.load(os.path.join(a.featdir, "raw_spliced.npy"))[m])}
RAW = {"pop": (T.loc[m, "pop"].values == "LSK").astype(float)[:, None]}
ARMS = {"total": ["pop", "state_tot"], "spliced": ["pop", "state_s"],
        # without the sorted population, like-for-like with the methods' direct
        # AUROC, which does not use it (label efficiency, referee)
        "total_nopop": ["state_tot"]}


def run(job):
    arm, frac, seed = job
    blocks = ARMS[arm]
    rng = np.random.default_rng(76000 + seed)
    oof = np.full(len(y), np.nan)
    for tr, te in folds_of(groups, seed, 5):
        ug = np.unique(groups[tr])
        keep = rng.choice(ug, size=max(10, int(round(frac * ug.size))), replace=False)
        tr = tr[np.isin(groups[tr], keep)]
        inner = folds_of(groups[tr], seed + 1000, 3)
        best, bestscore = None, -np.inf
        for kind, grid in GRIDS.items():
            for p in grid:
                sc = []
                for itr, ite in inner:
                    i1, i2 = tr[itr], tr[ite]
                    if len(np.unique(y[i1])) < 2 or len(np.unique(y[i2])) < 2:
                        continue
                    X1, X2 = build(blocks, PCA_BLOCKS, RAW, i1, i2, 50)
                    mm = make(kind, p, len(i1)).fit(X1, y[i1])
                    sc.append(roc_auc_score(y[i2], mm.predict_proba(X2)[:, 1]))
                if sc and np.mean(sc) > bestscore:
                    best, bestscore = (kind, p), float(np.mean(sc))
        Xtr, Xte = build(blocks, PCA_BLOCKS, RAW, tr, te, 50)
        mm = make(best[0], best[1], len(tr)).fit(Xtr, y[tr])
        oof[te] = mm.predict_proba(Xte)[:, 1]
    return dict(arm=arm, fraction=frac, seed=seed,
                n_train_clones=int(round(frac * len(np.unique(groups)) * 0.8)),
                auroc=auroc(clone_level(oof, y, groups)))


if __name__ == "__main__":
    jobs = [(arm, float(f), s) for arm in a.arms.split(",") for f in a.fractions.split(",")
            for s in range(a.seeds)]
    with Pool(a.workers) as p:
        rows = p.map(run, jobs)
    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    S = R.groupby(["arm", "fraction"]).agg(n_train_clones=("n_train_clones", "first"),
                                          auroc_mean=("auroc", "mean"),
                                          auroc_sd=("auroc", "std")).round(4)
    S.to_csv(a.out.replace(".csv", "_summary.csv"))
    print(S.to_string())
