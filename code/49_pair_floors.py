"""Minimum detectable difference for chosen pairs of predictors, and the number
of clones it would take to detect a given difference.

30_kernel_eval.py pairs every predictor with one reference. The question of how
large an experiment would have to be depends on which two predictors are being
compared, because the paired interval narrows as the two predictors' errors
become more correlated: two near-identical pipelines can be told apart at a
difference that two different methods cannot. This script takes the per-clone
table that 30_kernel_eval.py writes with --dump-clones, resamples clones exactly
as that script does (2,000 resamples, seed 13, the same rank-based AUROC), and
for each requested pair reports the difference, its 95% interval, the minimum
detectable difference at 80% power, and the correlation of the two predictors'
per-clone residuals.

The clones needed for a target difference d are extrapolated as
n * (floor / d)^2, the 1/sqrt(n) scaling of a paired bootstrap interval. That
is an extrapolation, not a measurement, and it assumes the two predictors'
error correlation stays what it is at the current size.

Usage:
  python 49_pair_floors.py <cloneval.csv> <out.csv> \
      --pairs K_snap_statot:K_raw_matrix,K_snap_wot:K_snap_moscot \
      --targets 0.005,0.0087,0.02
"""
import argparse

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 13
FLOOR_PER_WIDTH = 2.80 / (2 * 1.96)


def auroc_rows(score, y):
    B, n = score.shape
    n1 = y.sum(1).astype(np.float64)
    n0 = n - n1
    # average ranks for tied scores (as roc_auc_score does), so that ties are
    # not broken by the order of the resampled rows
    from scipy.stats import rankdata
    ranks = rankdata(score, axis=1, method="average")
    out = ((ranks * y).sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = np.nan
    return out


def residuals(p, y):
    return p - np.where(y, p[y].mean(), p[~y].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloneval")
    ap.add_argument("out")
    ap.add_argument("--pairs", required=True,
                    help="comma-separated A:B column pairs; the difference is A - B")
    ap.add_argument("--targets", default="0.005",
                    help="differences to compute the clones needed for")
    a = ap.parse_args()

    T = pd.read_csv(a.cloneval)
    y = T["y"].to_numpy().astype(bool)
    n = y.size
    rng = np.random.default_rng(SEED)
    bidx = rng.integers(0, n, size=(N_BOOT, n))
    yb = y[bidx]
    targets = [float(x) for x in a.targets.split(",") if x.strip()]

    rows = []
    for pair in a.pairs.split(","):
        A, B = pair.split(":")
        pa, pb = T[A].to_numpy(dtype=np.float64), T[B].to_numpy(dtype=np.float64)
        da = auroc_rows(pa[bidx], yb)
        db = auroc_rows(pb[bidx], yb)
        d = (da - db)
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        w = float(hi - lo)
        floor = w * FLOOR_PER_WIDTH
        point = float(auroc_rows(pa[None, :], y[None, :])[0]
                      - auroc_rows(pb[None, :], y[None, :])[0])
        rc = float(np.corrcoef(residuals(pa, y), residuals(pb, y))[0, 1])
        row = dict(a=A, b=B, diff=round(point, 4), lo=round(float(lo), 4),
                   hi=round(float(hi), 4), floor=round(floor, 4),
                   corr_residuals=round(rc, 3), n_clones=n)
        for t in targets:
            row["clones_for_%g" % t] = int(round(n * (floor / t) ** 2))
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
