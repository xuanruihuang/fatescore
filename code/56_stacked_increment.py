"""The increment design without the cost of adding features: stacking.

In the increment design used so far (30_kernel_eval.py --add-to, 03_evaluate.py)
a method's output is concatenated to the expression block and the classifier and
its hyperparameters are re-selected on the concatenation. Adding columns then
has a price of its own -- ten columns of noise cost 0.007-0.015 AUROC, fifty
cost 0.022 -- so a measured increment near zero mixes "no information" with
"information that did not pay for its columns". A reviewer can fairly say this
biases the design towards a null.

Stacking removes most of that price. Two first-stage predictors are trained
separately, each with the full clone-grouped tuning of cvlib.run_arm:

    base    [pop, BASE]   the expression reference
    method  [pop, M]      the method's output on its own (the replacement arm)

A second stage then combines their scores with a logistic regression on the
two logits, three parameters in all. The second stage is trained on
out-of-fold first-stage scores of the training clones (an inner clone-grouped
5-fold split), and applied to first-stage scores of the test clones from
models fitted on all training clones. The reference is the same second stage
on the base logit alone, so both arms pay the same, small, price of a second
stage. The paired difference is what the method's predictor adds to the
expression predictor.

--base may name several blocks, comma-separated. Each is then its own first-
stage predictor, and the reference second stage combines all their logits;
every method arm adds one more logit. This is how the expression reference is
made as strong as the stacking itself can make it: two predictors on different
views of the same expression already gain from being combined, so a method has
to add beyond that gain, not beyond a single expression predictor.

Uncertainty. Every repeat re-partitions the clones, and the difference moves
from one partition to the next by as much as the clone bootstrap says it is
uncertain, so an interval from one partition alone is too narrow. Reported per
arm: the difference on repeat 0 with its paired clone bootstrap (as in the
other scripts); the mean difference over the repeats; its standard deviation
across repeats; and an interval for the mean difference from a bootstrap that
draws one set of clones per replicate and averages the paired difference over
all repeats on that same draw ("repeat-averaged"), widened by the remaining
partition term, 1.96 * sd / sqrt(repeats). All out-of-fold scores are saved
(<out_prefix>_oof.npz) so the intervals can be recomputed.

The first-stage test scores are exactly those of cvlib.run_arm for the same
repeat (same folds, same inner tuning seed), so the base arm alone reproduces
the reported reference.

Usage:
  python 56_stacked_increment.py <featdir> <labels.tsv> <out_prefix> \
      --base raw_matrix --arms nm_snap_statot,snap_statot,... [--repeats 10]

Block handling: names starting raw_ are counts, scaled to 10,000 per cell and
log1p'ed, then reduced by a fold-internal PCA (as 03_evaluate.py and
30_kernel_eval.py --cp10k-log do); Ms is log1p(clip(Ms, 0)) then PCA (as
03_evaluate.py); vel_* and expr_* go to the PCA as they are; anything with at
most 20 columns goes in raw.
"""
import argparse
import pickle
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from cvlib import (GRIDS, build, make, folds_of, run_arm, clone_level, auroc,
                   cp10k_log)

N_BOOT = 2000
SEED = 13
INNER_K = 5
EPS = 1e-3


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


def tune_fit_predict(blocks, PCAB, RAW, y, groups, tr, te, seed, pcs):
    """The body of one outer fold of cvlib.run_arm, verbatim in effect."""
    inner = folds_of(groups[tr], seed + 1000, 3)
    best, bestscore = None, -np.inf
    for kind, grid in GRIDS.items():
        for p in grid:
            sc = []
            for itr, ite in inner:
                a, b = tr[itr], tr[ite]
                if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
                    continue
                Xa, Xb = build(blocks, PCAB, RAW, a, b, pcs)
                m = make(kind, p, len(a)).fit(Xa, y[a])
                sc.append(roc_auc_score(y[b], m.predict_proba(Xb)[:, 1]))
            if sc and np.mean(sc) > bestscore:
                best, bestscore = (kind, p), float(np.mean(sc))
    Xtr, Xte = build(blocks, PCAB, RAW, tr, te, pcs)
    m = make(best[0], best[1], len(tr)).fit(Xtr, y[tr])
    return m.predict_proba(Xte)[:, 1], "%s:%s" % best


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_second(Z, y, kind):
    """Second stage on first-stage logits. 'free': logistic regression with a
    negligible penalty (C=1e6). 'nonneg': the same likelihood with the weights
    on the logits bounded below by zero, the usual constraint in stacking
    (Breiman 1996; the super learner's convex weights), so that a weak or
    badly scaled first-stage score cannot enter with a large negative weight.
    Returns (intercept, weights)."""
    if kind == "free":
        m = LogisticRegression(C=1e6, max_iter=4000).fit(Z, y)
        return float(m.intercept_[0]), m.coef_[0].copy()

    def loss(w):
        eta = w[0] + Z @ w[1:]
        g = 1 / (1 + np.exp(-eta)) - y
        return (np.logaddexp(0, eta) - y * eta).sum(), np.r_[g.sum(), Z.T @ g]
    k = Z.shape[1]
    r = minimize(loss, np.r_[0.0, np.full(k, 1.0 / k)], jac=True,
                 method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * k)
    return float(r.x[0]), r.x[1:].copy()


def one_fold(s, f, tr, te, arms, PCAB, RAW, y, groups, pcs):
    """First-stage scores for every arm on this outer fold: inner out-of-fold
    on the training rows, and the fitted model's scores on the test rows."""
    sub = lambda D: {k: v[tr] for k, v in D.items()}
    PCAB_tr, RAW_tr = sub(PCAB), sub(RAW)
    out = {}
    for name, blocks in arms.items():
        _, _, oof_tr, _ = run_arm(name, blocks, PCAB_tr, RAW_tr, y[tr],
                                  groups[tr], 50000 + 100 * s + f, INNER_K, pcs)
        p_te, picked = tune_fit_predict(blocks, PCAB, RAW, y, groups, tr, te,
                                        s, pcs)
        out[name] = (oof_tr, p_te, picked)
    return s, f, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("featdir")
    ap.add_argument("labels")
    ap.add_argument("out_prefix")
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-sis", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=25)
    ap.add_argument("--pcs", type=int, default=50)
    ap.add_argument("--pos", default="Neutrophil")
    ap.add_argument("--neg", default="Monocyte")
    ap.add_argument("--pop-ref", default="LSK")
    ap.add_argument("--second", choices=["free", "nonneg"], default="free",
                    help="second-stage logistic regression: unconstrained "
                         "(C=1e6), or with non-negative weights on the "
                         "first-stage logits (intercept free)")
    ap.add_argument("--per-fold", default="",
                    help="NAME=PREFIX[,NAME=PREFIX]: arm NAME is read from "
                         "PREFIX_f<fold>.npy for each fold (features computed "
                         "without that fold's held-out clones, 59/41 "
                         "--drop-cells); needs --repeats 1 and the folds of "
                         "repeat 0")
    ap.add_argument("--stage1-in", default=None,
                    help="read first-stage scores from a previous run's "
                         "_stage1.pkl instead of refitting them")
    a = ap.parse_args()

    T = pd.read_csv(a.labels, sep="\t")
    d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
    assert list(T["cell"]) == d2_names, "label table and feature rows disagree"
    sel = (T["n_sis"] >= a.min_sis) & T["dom"].isin([a.pos, a.neg])
    idx = np.where(sel.values)[0]
    y = (T.loc[sel, "dom"].values == a.pos).astype(int)
    groups = T.loc[sel, "clone"].values
    say("task: %d cells, %d clones, prevalence %.3f"
        % (len(idx), len(np.unique(groups)), y.mean()))

    bases = [b for b in a.base.split(",") if b]
    names = bases + [m for m in a.arms.split(",") if m and m not in bases]
    PCAB = {}
    RAW = {"pop": (T["pop"].values == a.pop_ref).astype(float)[idx, None]}
    perfold = dict(kv.split("=", 1) for kv in a.per_fold.split(",") if kv)
    if perfold:
        assert a.repeats == 1, "--per-fold is defined on repeat 0 only"
    for n in names:
        M = np.load(perfold[n] + "_f0.npy" if n in perfold
                    else os.path.join(a.featdir, n + ".npy"))
        if M.ndim == 1:
            M = M[:, None]
        M = M[idx]
        if n.startswith("raw_"):
            PCAB[n] = cp10k_log(M)
        elif n == "Ms":
            PCAB[n] = np.log1p(np.clip(M, 0, None))
        elif n.startswith(("vel_", "expr_")) or M.shape[1] > 20:
            PCAB[n] = M
        else:
            RAW[n] = M.astype(float)
    say("PCA blocks %s; raw blocks %s" % (list(PCAB), list(RAW)))

    arms = {"base" if len(bases) == 1 else "base_" + b: ["pop", b]
            for b in bases}
    base_keys = list(arms)
    arms.update({m: ["pop", m] for m in names[len(bases):]})
    jobs = []
    for s in range(a.repeats):
        for f, (tr, te) in enumerate(folds_of(groups, s, a.folds)):
            jobs.append((s, f, tr, te))
    say("%d (repeat, fold) jobs, %d arms each, %d workers"
        % (len(jobs), len(arms), a.n_jobs))
    if a.stage1_in:
        with open(a.stage1_in, "rb") as fh:
            byjob = pickle.load(fh)
        missing = [(s, f, n) for s, f, _, _ in jobs for n in arms
                   if n not in byjob[(s, f)]]
        assert not missing, "stage-1 file lacks %s" % missing[:3]
        say("stage 1 read from %s" % a.stage1_in)
    else:
        def raw_for(f):
            if not perfold:
                return RAW
            R = dict(RAW)
            for n, pre in perfold.items():
                M = np.load("%s_f%d.npy" % (pre, f))
                R[n] = (M[:, None] if M.ndim == 1 else M)[idx].astype(float)
            return R
        res = Parallel(n_jobs=a.n_jobs, verbose=5)(
            delayed(one_fold)(s, f, tr, te, arms, PCAB, raw_for(f), y,
                              groups, a.pcs)
            for s, f, tr, te in jobs)
        byjob = {(s, f): out for s, f, out in res}
        with open(a.out_prefix + "_stage1.pkl", "wb") as fh:
            pickle.dump(byjob, fh)
    say("second stage: %s" % a.second)

    # second stage, per outer fold; arms: first-stage alone, stacked
    OOF = {}
    coefs = []
    for s in range(a.repeats):
        oof = {k: np.full(len(y), np.nan) for k in
               ["S1_" + n for n in arms] + ["ST_base"]
               + ["ST_" + n for n in arms if n not in base_keys]}
        for f, (tr, te) in enumerate(folds_of(groups, s, a.folds)):
            out = byjob[(s, f)]
            zb_tr = np.column_stack([logit(out[b][0]) for b in base_keys])
            zb_te = np.column_stack([logit(out[b][1]) for b in base_keys])
            for n in ["base"] + [m for m in arms if m != "base"]:
                if n in out:
                    oof["S1_" + n][te] = out[n][1]
                if n in base_keys and n != "base":
                    continue
                if n == "base":
                    Ztr, Zte = zb_tr, zb_te
                else:
                    Ztr = np.c_[zb_tr, logit(out[n][0])]
                    Zte = np.c_[zb_te, logit(out[n][1])]
                b0, w = fit_second(Ztr, y[tr], a.second)
                oof["ST_" + n][te] = 1 / (1 + np.exp(-(b0 + Zte @ w)))
                coefs.append(dict(seed=s, fold=f, arm=n,
                                  w_base=";".join("%.4f" % c for c in
                                                  w[:len(base_keys)]),
                                  w_method=(float(w[-1])
                                            if n != "base" else np.nan)))
        for k, v in oof.items():
            OOF[(k, s)] = v
    pd.DataFrame(coefs).to_csv(a.out_prefix + "_coefs.csv", index=False)

    keys = sorted({k for k, _ in OOF})
    np.savez_compressed(a.out_prefix + "_oof.npz", y=y, groups=groups,
                        keys=np.array(keys), repeats=a.repeats,
                        oof=np.stack([[OOF[(k, s)] for s in range(a.repeats)]
                                      for k in keys]))
    rows = []
    for k in keys:
        for s in range(a.repeats):
            rows.append(dict(arm=k, seed=s,
                             auroc=auroc(clone_level(OOF[(k, s)], y, groups))))
    R = pd.DataFrame(rows)
    R.to_csv(a.out_prefix + "_perrepeat.csv", index=False)
    S = R.groupby("arm").agg(auroc_mean=("auroc", "mean"),
                             auroc_sd=("auroc", "std"))
    S.to_csv(a.out_prefix + "_summary.csv")
    say("\n" + S.to_string())

    clones = np.unique(groups)
    tabs = {(k, s): clone_level(OOF[(k, s)], y, groups).reindex(clones)
            for k in keys for s in range(a.repeats)}
    tab = {k: tabs[(k, 0)] for k in keys}
    yv = tab[keys[0]]["y"].to_numpy().astype(bool)
    rng = np.random.default_rng(SEED)
    bidx = rng.integers(0, clones.size, size=(N_BOOT, clones.size))
    yb = yv[bidx]
    bootr = {(k, s): auroc_rows(tabs[(k, s)]["p"].to_numpy()[bidx], yb)
             for k in keys for s in range(a.repeats)}
    boot = {k: bootr[(k, 0)] for k in keys}
    point = {k: float(auroc(tab[k])) for k in keys}
    out = []
    for n in arms:
        if n in base_keys:
            continue
        for kind, ref in (("stacked", "ST_base"),):
            k = "ST_" + n
            d = boot[k] - boot[ref]
            d = d[np.isfinite(d)]
            q = np.percentile(d, [2.5, 97.5])
            davg = np.mean([bootr[(k, s)] - bootr[(ref, s)]
                            for s in range(a.repeats)], axis=0)
            davg = davg[np.isfinite(davg)]
            qa = np.percentile(davg, [2.5, 97.5])
            per = R.pivot(index="seed", columns="arm", values="auroc")
            dr = per[k] - per[ref]
            sd = float(dr.std(ddof=1))
            part = 1.96 * sd / np.sqrt(a.repeats)
            m10 = float(dr.mean())
            out.append(dict(arm=n, design=kind,
                            mean10=round(m10, 4), sd10=round(sd, 4),
                            lo_avg=round(float(qa[0]), 4),
                            hi_avg=round(float(qa[1]), 4),
                            lo_avg_part=round(m10 - np.sqrt(
                                (m10 - qa[0]) ** 2 + part ** 2), 4),
                            hi_avg_part=round(m10 + np.sqrt(
                                (qa[1] - m10) ** 2 + part ** 2), 4),
                            auroc=round(point[k], 4),
                            auroc_ref=round(point[ref], 4),
                            diff=round(point[k] - point[ref], 4),
                            lo=round(float(q[0]), 4), hi=round(float(q[1]), 4),
                            width=round(float(q[1] - q[0]), 4),
                            excludes_zero=int(q[0] > 0 or q[1] < 0),
                            mean_diff_10=round(
                                float(S.loc[k, "auroc_mean"]
                                      - S.loc[ref, "auroc_mean"]), 4)))
    P = pd.DataFrame(out)
    P.to_csv(a.out_prefix + "_pairs.csv", index=False)
    say("\n" + P.to_string(index=False))
    say("wrote %s_{perrepeat,summary,pairs,coefs}.csv" % a.out_prefix)


if __name__ == "__main__":
    main()
