"""The distance baseline with its exponent chosen inside the training folds.

33_idw_fate.py builds the inverse-distance baseline at several exponents, and
the first pass reported the best of them -- which is selected on the scored
metric and so repeats the mistake we criticise in the paper that tuned the same
exponent on its own test cells. Here the exponent is just another
hyperparameter: inside each outer training fold, a clone-grouped inner split
chooses the exponent and the classifier together, and only then is the model
fitted and applied to the held-out fold. The exponent picked per fold is
recorded, and the spread across a fixed exponent is reported separately so the
two readings can be told apart.

The outer folds, the inner split, the classifier grid and the clone-level
aggregation are cvlib's, the same ones every arm in this evaluation uses.

Usage:
  python 34_idw_nested.py <featdir> <labels.tsv> <out_prefix> [--repeats 3]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from cvlib import GRIDS, make, build, folds_of, clone_level, auroc

N_BOOT = 2000
SEED = 13
FLOOR_PER_WIDTH = 2.80 / (2 * 1.96)


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def auroc_rows(score, y):
    B, n = score.shape
    n1 = y.sum(1).astype(np.float64)
    n0 = n - n1
    order = np.argsort(score, axis=1, kind="stable")
    ranks = np.empty((B, n), dtype=np.float64)
    seq = np.broadcast_to(np.arange(1, n + 1, dtype=np.float64), (B, n))
    np.put_along_axis(ranks, order, seq, axis=1)
    out = ((ranks * y).sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = np.nan
    return out


def run_nested(arm, candidates, raw, y, groups, seed, k, pcs):
    """Out-of-fold predictions with the feature block chosen inside the folds."""
    oof = np.full(len(y), np.nan)
    picked = []
    for tr, te in folds_of(groups, seed, k):
        inner = folds_of(groups[tr], seed + 1000, 3)
        best, bestscore = None, -np.inf
        for cand in candidates:
            blocks = ["pop", cand]
            for kind, grid in GRIDS.items():
                for p in grid:
                    sc = []
                    for itr, ite in inner:
                        a, b = tr[itr], tr[ite]
                        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
                            continue
                        Xa, Xb = build(blocks, {}, raw, a, b, pcs)
                        m = make(kind, p, len(a)).fit(Xa, y[a])
                        sc.append(roc_auc_score(y[b], m.predict_proba(Xb)[:, 1]))
                    if sc and np.mean(sc) > bestscore:
                        best = (cand, kind, p)
                        bestscore = float(np.mean(sc))
        blocks = ["pop", best[0]]
        Xtr, Xte = build(blocks, {}, raw, tr, te, pcs)
        m = make(best[1], best[2], len(tr)).fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
        picked.append("%s|%s:%s" % (best[0], best[1], best[2]))
    return arm, seed, oof, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("featdir")
    ap.add_argument("labels")
    ap.add_argument("out_prefix")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-sis", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--pcs", type=int, default=50)
    ap.add_argument("--pos", default="Neutrophil")
    ap.add_argument("--neg", default="Monocyte")
    ap.add_argument("--pop-ref", default="LSK")
    ap.add_argument("--ref", default="cr_velo",
                    help="the arm the nested baseline is compared against")
    a = ap.parse_args()

    T = pd.read_csv(a.labels, sep="\t")
    d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
    assert list(T["cell"]) == d2_names

    raw = {"pop": (T["pop"].values == a.pop_ref).astype(float)[:, None]}
    cands = []
    for fn in sorted(os.listdir(a.featdir)):
        if fn.startswith("snap_idw_g") and fn.endswith(".npy"):
            nm = fn[:-4]
            raw[nm] = np.load(os.path.join(a.featdir, fn))
            cands.append(nm)
    raw[a.ref] = np.load(os.path.join(a.featdir, a.ref + ".npy"))
    say("candidate exponents: %s" % cands)

    sel = (T["n_sis"] >= a.min_sis) & T["dom"].isin([a.pos, a.neg])
    idx = np.where(sel.values)[0]
    y = (T.loc[sel, "dom"].values == a.pos).astype(int)
    groups = T.loc[sel, "clone"].values
    raw = {k: v[idx] for k, v in raw.items()}
    say("task A: %d cells, %d clones" % (len(idx), len(np.unique(groups))))

    jobs = [("IDW_nested_gamma", cands, s) for s in range(a.repeats)]
    jobs += [("IDW_fixed_" + c, [c], s) for c in cands
             for s in range(a.repeats)]
    jobs += [("REF_" + a.ref, [a.ref], s) for s in range(a.repeats)]
    say("%d jobs on %d workers" % (len(jobs), a.n_jobs))
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(run_nested)(nm, cs, raw, y, groups, s, a.folds, a.pcs)
        for nm, cs, s in jobs)

    OOF, rows = {}, []
    for arm, seed, oof, picked in res:
        OOF[(arm, seed)] = oof
        d = clone_level(oof, y, groups)
        rows.append(dict(arm=arm, seed=seed, auroc=auroc(d), n_clones=len(d),
                         picked=";".join(picked)))
    R = pd.DataFrame(rows)
    R.to_csv(a.out_prefix + "_perrepeat.csv", index=False)
    S = (R.groupby("arm").agg(auroc_mean=("auroc", "mean"),
                              auroc_sd=("auroc", "std"))
         .sort_values("auroc_mean", ascending=False))
    S.to_csv(a.out_prefix + "_summary.csv")
    say("\n" + S.to_string())
    say("\nexponents chosen per fold, first repeat:\n%s"
        % R[R.arm == "IDW_nested_gamma"].iloc[0]["picked"])

    clones = np.unique(groups)
    arms = sorted({a_ for a_, _ in OOF})
    tab = {n: clone_level(OOF[(n, 0)], y, groups).reindex(clones) for n in arms}
    yv = tab[arms[0]]["y"].to_numpy().astype(bool)
    rng = np.random.default_rng(SEED)
    bidx = rng.integers(0, clones.size, size=(N_BOOT, clones.size))
    yb = yv[bidx]
    boot = {n: auroc_rows(tab[n]["p"].to_numpy()[bidx], yb) for n in arms}
    point = {n: float(auroc(tab[n])) for n in arms}

    ref = "REF_" + a.ref
    out = []
    for n in arms:
        if n == ref:
            continue
        d = boot[n] - boot[ref]
        d = d[np.isfinite(d)]
        q = np.percentile(d, [2.5, 97.5])
        w = float(q[1] - q[0])
        pi, pr = tab[n]["p"].to_numpy(), tab[ref]["p"].to_numpy()
        res_i = pi - np.where(yv, pi[yv].mean(), pi[~yv].mean())
        res_r = pr - np.where(yv, pr[yv].mean(), pr[~yv].mean())
        out.append(dict(arm=n, ref=ref, auroc=round(point[n], 4),
                        auroc_ref=round(point[ref], 4),
                        diff=round(point[n] - point[ref], 4),
                        lo=round(float(q[0]), 4), hi=round(float(q[1]), 4),
                        width=round(w, 4),
                        floor_implied=round(w * FLOOR_PER_WIDTH, 4),
                        corr_residuals=round(
                            float(np.corrcoef(res_i, res_r)[0, 1]), 4),
                        excludes_zero=int(q[0] > 0 or q[1] < 0)))
    P = pd.DataFrame(out).sort_values("diff", ascending=False)
    P.to_csv(a.out_prefix + "_vs_ref.csv", index=False)
    say("\n" + P.to_string(index=False))
    say("wrote %s_{perrepeat,summary,vs_ref}.csv" % a.out_prefix)


if __name__ == "__main__":
    main()
