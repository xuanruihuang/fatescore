"""Is a method's clonal-fate score decided by how much it resembles the
expression baseline -- or is that relation arithmetic?

Across the arms of one evaluation, AUROC rises with the arm's per-clone
residual correlation against the tuned expression baseline. The relation is
suggestive but it may be nothing: residual correlation and AUROC are computed
from the SAME out-of-fold predictions, and two predictors that are both
accurate must resemble each other. This script separates the two readings
with a null that makes the arithmetic explicit.

The null. Take the baseline's own per-clone out-of-fold score, standardise it
and add independent noise:

    p(lambda) = z(p_base) + lambda * N(0, 1)

Every such predictor carries the baseline's signal and nothing else, so the
curve it traces in the (residual correlation, AUROC) plane is what "both
accurate, therefore similar" looks like on its own. Each arm is then placed
against that curve by its vertical deviation

    delta = AUROC(arm) - AUROC_null(residual correlation of the arm)

with a clone-level paired bootstrap, the same resampling the evaluation uses.
An arm that reads the baseline's geometry sits on the curve; an arm whose
signal comes from somewhere else does not have to.

The reading was registered before this ran (notes section 17.2): if the real
methods AND the control arms all cover zero, the relation is arithmetic and
must not be written as a mechanism.

Usage:
  python 46_geometry_null.py <cloneval.csv> <out_prefix> [--ref K_expr_hvg]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 13
# A 95% interval of width W implies sigma = W / (2 * 1.96); 80% power under
# the "bound excludes zero" rule needs delta >= 2.80 sigma. Same constant as
# 28_arm_correlation.py and 30_kernel_eval.py.
FLOOR_PER_WIDTH = 2.80 / (2 * 1.96)


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def auroc_rows(score, y):
    """AUROC of every row of `score` against the shared label vector `y`."""
    score = np.atleast_2d(score)
    B, n = score.shape
    n1 = float(y.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return np.full(B, np.nan)
    order = np.argsort(score, axis=1, kind="stable")
    ranks = np.empty((B, n), dtype=np.float64)
    seq = np.broadcast_to(np.arange(1, n + 1, dtype=np.float64), (B, n))
    np.put_along_axis(ranks, order, seq, axis=1)
    return ((ranks * y).sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def residuals(score, y):
    """Each row minus its own class means -- the scorer's definition."""
    score = np.atleast_2d(score).astype(np.float64)
    out = np.empty_like(score)
    m1 = score[:, y].mean(1, keepdims=True)
    m0 = score[:, ~y].mean(1, keepdims=True)
    out[:, y] = score[:, y] - m1
    out[:, ~y] = score[:, ~y] - m0
    return out


def corr_rows(A, b):
    """Correlation of every row of A with the single vector b."""
    A = np.atleast_2d(A).astype(np.float64)
    b = np.asarray(b, dtype=np.float64)
    A = A - A.mean(1, keepdims=True)
    b = b - b.mean()
    num = A @ b
    den = np.sqrt((A * A).sum(1) * (b * b).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def curve_point(scores, ref, y):
    """(residual correlation with ref, AUROC) for every row of `scores`."""
    R = residuals(scores, y)
    r0 = residuals(ref[None, :], y)[0]
    return corr_rows(R, r0), auroc_rows(scores, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloneval", help="<prefix>_cloneval.csv from 30_kernel_eval.py")
    ap.add_argument("out_prefix")
    ap.add_argument("--ref", default="K_expr_hvg",
                    help="the expression baseline column")
    ap.add_argument("--n-lambda", type=int, default=14)
    ap.add_argument("--n-draws", type=int, default=3,
                    help="independent noise draws per lambda; the null curve "
                         "averages over them")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--plot", default="",
                    help="optional path for a PNG of the null curve and the arms")
    a = ap.parse_args()

    T = pd.read_csv(a.cloneval)
    y = T["y"].to_numpy().astype(bool)
    arms = [c for c in T.columns if c not in ("clone", "y")]
    assert a.ref in arms, "baseline column %s not in %s" % (a.ref, arms)
    P = {c: T[c].to_numpy(dtype=np.float64) for c in arms}
    n = y.size
    say("%d clones, prevalence %.3f, %d arms, baseline %s"
        % (n, y.mean(), len(arms), a.ref))

    # The null family: the baseline's own signal plus independent noise.
    # The noise is drawn ONCE so that every bootstrap resample sees the same
    # null predictors and the deviation stays a paired quantity.
    base = P[a.ref]
    z = (base - base.mean()) / base.std()
    rng = np.random.default_rng(SEED)
    lam = np.concatenate([[0.0], np.geomspace(0.1, 12.0, a.n_lambda - 1)])
    eps = rng.standard_normal((a.n_draws, n))
    NULL = np.concatenate([z[None, :] + L * eps for L in lam], axis=0)
    lam_of_row = np.repeat(lam, a.n_draws)
    say("null family: %d lambda x %d draws = %d predictors"
        % (len(lam), a.n_draws, NULL.shape[0]))

    ARM = np.stack([P[c] for c in arms], axis=0)
    ref_i = arms.index(a.ref)

    def deviations(idx):
        """Delta for every arm on the clone subset `idx` (or all clones)."""
        yy = y[idx]
        if yy.all() or not yy.any():
            return None
        nc, na = curve_point(NULL[:, idx], base[idx], yy)
        ac, aa = curve_point(ARM[:, idx], base[idx], yy)
        # average the null over its noise draws at each lambda, then read the
        # curve as AUROC against residual correlation
        nc = nc.reshape(len(lam), a.n_draws).mean(1)
        na = na.reshape(len(lam), a.n_draws).mean(1)
        ok = np.isfinite(nc) & np.isfinite(na)
        nc, na = nc[ok], na[ok]
        o = np.argsort(nc)
        nc, na = nc[o], na[o]
        pred = np.interp(ac, nc, na)
        inside = (ac >= nc[0]) & (ac <= nc[-1])
        return aa - pred, ac, aa, pred, inside, nc, na

    full = deviations(np.arange(n))
    assert full is not None, "the full clone set must have both classes"
    d_obs, c_obs, a_obs, p_obs, inside, nc0, na0 = full

    rng_b = np.random.default_rng(SEED)
    B = rng_b.integers(0, n, size=(a.n_boot, n))
    D = np.full((a.n_boot, len(arms)), np.nan)
    kept = 0
    for b in range(a.n_boot):
        r = deviations(B[b])
        if r is None:
            continue
        D[kept] = r[0]
        kept += 1
    D = D[:kept]
    say("%d of %d bootstrap resamples usable" % (kept, a.n_boot))

    rows = []
    for i, c in enumerate(arms):
        d = D[:, i]
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        w = float(hi - lo)
        rows.append(dict(arm=c,
                         auroc=round(float(a_obs[i]), 4),
                         corr_residuals=round(float(c_obs[i]), 4),
                         auroc_null_at_that_corr=round(float(p_obs[i]), 4),
                         delta=round(float(d_obs[i]), 4),
                         lo=round(float(lo), 4), hi=round(float(hi), 4),
                         width=round(w, 4),
                         floor_implied=round(w * FLOOR_PER_WIDTH, 4),
                         excludes_zero=int(lo > 0 or hi < 0),
                         corr_inside_null_range=int(inside[i])))
    R = pd.DataFrame(rows).sort_values("delta", ascending=False)
    R.to_csv(a.out_prefix + "_deviation.csv", index=False)
    say("\n" + R.to_string(index=False))

    C = pd.DataFrame(dict(lam=lam, corr_residuals=np.round(nc0, 4),
                          auroc=np.round(na0, 4)))
    C.to_csv(a.out_prefix + "_nullcurve.csv", index=False)
    json.dump(dict(cloneval=a.cloneval, ref=a.ref, n_clones=int(n),
                   prevalence=round(float(y.mean()), 4),
                   n_lambda=int(len(lam)), n_draws=int(a.n_draws),
                   n_boot=int(a.n_boot), usable_boot=int(kept), seed=SEED),
              open(a.out_prefix + "_nulllog.json", "w"), indent=1)

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=160)
        ax.plot(nc0, na0, "-", color="0.4", lw=1.4,
                label="baseline + noise (null)")
        ax.scatter(c_obs, a_obs, s=22, color="C3", zorder=3)
        for i, c in enumerate(arms):
            ax.annotate(c.replace("K_", ""), (c_obs[i], a_obs[i]),
                        fontsize=5.5, xytext=(3, 2),
                        textcoords="offset points")
        ax.set_xlabel("per-clone residual correlation with the expression baseline")
        ax.set_ylabel("clonal-fate AUROC")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(a.plot)
        say("wrote %s" % a.plot)
    say("wrote %s_{deviation,nullcurve,nulllog}.*" % a.out_prefix)


if __name__ == "__main__":
    main()
