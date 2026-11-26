"""The velocity kernel against a tuned expression kernel, on clonal fate.

Published comparisons of CellRank's kernels on clone-held-out data leave the
expression kernel at its defaults. This scores every kernel the same way the
manuscript scores its arms -- the kernel's fate probabilities as the only
feature besides the starting population, the same clone-disjoint folds, the
classifier and its hyperparameters chosen inside the training folds -- and then
compares each against the velocity kernel with a paired bootstrap over clones.

Two readings are reported and both are stated in advance. The DEFAULT
expression kernel is the one built on the same graph the manuscript's pseudotime
kernel uses, 30 neighbours and 30 principal components. The BEST expression
kernel is the highest-scoring point of the grid, which is selected on the
scored metric and is therefore an optimistic bound, not an honest estimate:
if the velocity kernel still wins against it, the finding is robust; if the
default loses and the best does not, the published comparison was about
default parameters.

Each pair also gets what 28_arm_correlation.py measures: the correlation of the
two kernels' per-clone scores and residuals, and the detection floor that the
pair's own interval width implies.

Usage:
  python 30_kernel_eval.py <featdir> <labels.tsv> <out_prefix> [--repeats 3]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cvlib import run_arm, clone_level, auroc, cp10k_log

N_BOOT = 2000
SEED = 13
# A 95% interval of width W implies sigma = W / (2 * 1.96); 80% power under the
# "lower bound above zero" rule needs delta >= 2.80 sigma. Same constant as
# 28_arm_correlation.py.
FLOOR_PER_WIDTH = 2.80 / (2 * 1.96)
DEFAULT_KERNEL = "cr_conn_n30_p30_d1"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("featdir")
    ap.add_argument("labels")
    ap.add_argument("out_prefix")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-sis", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=20)
    ap.add_argument("--pcs", type=int, default=50)
    ap.add_argument("--pos", default="Neutrophil")
    ap.add_argument("--neg", default="Monocyte")
    ap.add_argument("--pop-ref", default="LSK")
    ap.add_argument("--where", default="",
                    help="extra pandas expression restricting the label "
                         "table, e.g. n_sis_b2 == 0")
    ap.add_argument("--dump-clones", action="store_true",
                    help="also write <out_prefix>_cloneval.csv: one row per "
                         "clone, its label, and every arm's out-of-fold "
                         "score from repeat 0 -- the same table the paired "
                         "bootstrap uses, so downstream analyses resample "
                         "exactly the units this script does")
    ap.add_argument("--add-to", default="",
                    help="turn every arm into an increment: each block k is "
                         "scored as [pop, ADD_TO, k] instead of [pop, k], and "
                         "ADD_TO alone as [pop, ADD_TO]; with --ref ADD_TO the "
                         "paired difference is what k adds on top of that "
                         "block, the design the manuscript uses for velocity")
    ap.add_argument("--cp10k-log", action="store_true",
                    help="scale each cell of a --pca-prefixes block to 10,000 "
                         "counts and take log1p before the fold-internal PCA, "
                         "exactly as 03_evaluate.py builds its state arms; "
                         "for raw count blocks only (the cytokine and in vivo "
                         "expr_hvg blocks are already normalised and logged)")
    ap.add_argument("--pca-prefixes", default="",
                    help="blocks whose columns are reduced by a PCA fit "
                         "inside the training folds, e.g. expr_")
    ap.add_argument("--prefixes", default="cr_",
                    help="comma-separated file prefixes in the feature "
                         "directory to score as arms; 'field_' picks up the "
                         "fields saved by 31_direction_metrics.py")
    ap.add_argument("--ref", default="cr_velo",
                    help="the arm every other one is compared against")
    a = ap.parse_args()

    T = pd.read_csv(a.labels, sep="\t")
    d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
    assert list(T["cell"]) == d2_names, "label table and feature rows disagree"

    prefixes = tuple(p for p in a.prefixes.split(",") if p)
    kernels = {}
    for fn in sorted(os.listdir(a.featdir)):
        if fn.startswith(prefixes) and fn.endswith(".npy"):
            kernels[fn[:-4]] = np.load(os.path.join(a.featdir, fn))
    assert a.ref in kernels, "the reference arm %s is missing from the feature "\
        "directory" % a.ref
    say("kernels: %s" % list(kernels))

    # A fate-probability block has one column per population and goes in
    # raw. An expression block has thousands and must be reduced inside
    # the training folds, the way every expression arm in the manuscript
    # is; --pca-prefixes names those blocks. Empty by default, so the
    # runs already reported are untouched.
    pca_pref = tuple(x for x in a.pca_prefixes.split(",") if x)
    PCAB = {k: v for k, v in kernels.items()
            if pca_pref and k.startswith(pca_pref)}
    RAW = {"pop": (T["pop"].values == a.pop_ref).astype(float)[:, None]}
    RAW.update({k: v for k, v in kernels.items() if k not in PCAB})
    if a.cp10k_log and PCAB:
        # the manuscript's state arms are counts-per-10k then log1p, then the
        # fold-internal PCA; feeding raw counts to the PCA gives a weaker arm
        # no dtype cast: 03_evaluate.py applies cp10k_log to the array as
        # loaded, and matching it digit for digit is the point
        PCAB = {k: cp10k_log(v) for k, v in PCAB.items()}
        say("counts-per-10k then log1p applied to: %s" % list(PCAB))
    if PCAB:
        say("reduced inside the folds: %s" % list(PCAB))

    sel = (T["n_sis"] >= a.min_sis) & T["dom"].isin([a.pos, a.neg])
    if a.where:
        # an extra restriction on the label table, written out in the
        # command line and echoed here, e.g. "n_sis_b2 == 0" to keep only
        # clones whose late cells sit in a single batch
        keep = T.eval(a.where).astype(bool)
        say("extra restriction %r keeps %d of %d selected cells"
            % (a.where, int((sel & keep).sum()), int(sel.sum())))
        sel = sel & keep
    idx = np.where(sel.values)[0]
    y = (T.loc[sel, "dom"].values == a.pos).astype(int)
    groups = T.loc[sel, "clone"].values
    say("task A: %d cells, %d clones, prevalence %.3f"
        % (len(idx), len(np.unique(groups)), y.mean()))
    RAW = {k: v[idx] for k, v in RAW.items()}
    PCAB = {k: v[idx] for k, v in PCAB.items()}

    arms = {("K_" + k): ["pop", k] for k in kernels}
    if a.add_to:
        # Replacement asks whether a method's output can stand in for the
        # expression; increment asks whether it adds anything to it. Only the
        # second supports "adds no information".
        assert a.add_to in kernels, "--add-to %s is not a loaded block" % a.add_to
        assert a.ref == a.add_to, "use --ref %s with --add-to" % a.add_to
        arms = {("K_" + k): (["pop", k] if k == a.add_to
                             else ["pop", a.add_to, k]) for k in kernels}
        say("increment design: every arm is [pop, %s, k]" % a.add_to)
    jobs = [(n, b, s) for n, b in arms.items() for s in range(a.repeats)]
    say("%d (arm, repeat) jobs on %d workers" % (len(jobs), a.n_jobs))
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(run_arm)(n, b, PCAB, RAW, y, groups, s, a.folds, a.pcs,
                         None)
        for n, b, s in jobs)

    OOF, rows = {}, []
    for arm, seed, oof, picked in res:
        OOF[(arm, seed)] = oof
        d = clone_level(oof, y, groups)
        rows.append(dict(arm=arm, seed=seed, auroc=auroc(d), n_clones=len(d),
                         picked=";".join(sorted(set(picked)))))
    R = pd.DataFrame(rows)
    R.to_csv(a.out_prefix + "_perrepeat.csv", index=False)
    S = (R.groupby("arm").agg(auroc_mean=("auroc", "mean"),
                              auroc_sd=("auroc", "std"),
                              n_clones=("n_clones", "first"))
         .sort_values("auroc_mean", ascending=False))
    S.to_csv(a.out_prefix + "_summary.csv")
    say("\n" + S.to_string())

    # paired bootstrap on the first repeat, one set of resamples for every arm
    clones = np.unique(groups)
    tab = {n: clone_level(OOF[(n, 0)], y, groups).reindex(clones) for n in arms}
    yv = tab[list(arms)[0]]["y"].to_numpy().astype(bool)
    if a.dump_clones:
        D = pd.DataFrame({"clone": clones, "y": yv.astype(int)})
        for n in arms:
            D[n] = tab[n]["p"].to_numpy()
        D.to_csv(a.out_prefix + "_cloneval.csv", index=False)
        say("wrote %s_cloneval.csv (%d clones, %d arms)"
            % (a.out_prefix, len(clones), len(arms)))
    rng = np.random.default_rng(SEED)
    bidx = rng.integers(0, clones.size, size=(N_BOOT, clones.size))
    yb = yv[bidx]
    boot = {n: auroc_rows(tab[n]["p"].to_numpy()[bidx], yb) for n in arms}
    point = {n: float(auroc(tab[n])) for n in arms}

    # Each arm's own interval, not only its difference from the
    # reference. On an arm where everything comes out near chance the
    # paired difference says nothing, and what the reader needs is how
    # far above chance the best arm can be ruled out to be.
    absol = []
    for n in arms:
        b = boot[n][np.isfinite(boot[n])]
        q = np.percentile(b, [2.5, 97.5])
        absol.append(dict(arm=n, auroc=round(point[n], 4),
                          lo=round(float(q[0]), 4),
                          hi=round(float(q[1]), 4),
                          n_clones=int(clones.size)))
    A = pd.DataFrame(absol).sort_values("auroc", ascending=False)
    A.to_csv(a.out_prefix + "_absolute.csv", index=False)
    say("each arm's own interval:")
    print(A.to_string(index=False), flush=True)

    ref = "K_" + a.ref
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
        out.append(dict(arm=n, ref=ref,
                        auroc=round(point[n], 4),
                        auroc_ref=round(point[ref], 4),
                        diff=round(point[n] - point[ref], 4),
                        lo=round(float(q[0]), 4), hi=round(float(q[1]), 4),
                        width=round(w, 4),
                        floor_implied=round(w * FLOOR_PER_WIDTH, 4),
                        corr_scores=round(float(np.corrcoef(pi, pr)[0, 1]), 4),
                        corr_residuals=round(float(np.corrcoef(res_i, res_r)[0, 1]), 4),
                        excludes_zero=int(q[0] > 0 or q[1] < 0)))
    P = pd.DataFrame(out).sort_values("diff", ascending=False)
    P.to_csv(a.out_prefix + "_vs_velocity.csv", index=False)
    say("\n" + P.to_string(index=False))

    conn = [n for n in arms if n.startswith("K_cr_conn")]
    if conn:
        best = max(conn, key=lambda n: point[n])
        dflt = "K_" + DEFAULT_KERNEL
        note = []
        for label, n in (("default expression kernel", dflt),
                         ("best expression kernel (selected on the metric)", best)):
            if n in point:
                row = P[P.arm == n].iloc[0]
                note.append(dict(which=label, arm=n, auroc=point[n],
                                 velocity_kernel=point[ref],
                                 diff=row["diff"], lo=row["lo"], hi=row["hi"],
                                 width=row["width"],
                                 floor_implied=row["floor_implied"]))
        pd.DataFrame(note).to_csv(a.out_prefix + "_headline.csv", index=False)
        say("\n" + pd.DataFrame(note).to_string(index=False))
    say("wrote %s_{perrepeat,summary,vs_velocity,headline}.csv" % a.out_prefix)


if __name__ == "__main__":
    main()
