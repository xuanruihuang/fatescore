"""Core scoring. All functions take per-cell scores (probability of the positive fate) for the task
cells, the clone of each cell, and a per-clone binary label (1 = clone dominated by the positive
fate)."""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mdd_factor(alpha=0.05, power=0.8):
    """Minimum detectable difference per unit width of a two-sided (1 - alpha) interval."""
    za = norm.ppf(1 - alpha / 2)
    return (za + norm.ppf(power)) / (2 * za)


def clone_means(cell_scores, cell_clones, clones):
    """Average cell scores within each clone, in the order of `clones`."""
    idx = {c: i for i, c in enumerate(clones)}
    inv = np.array([idx[c] for c in cell_clones])
    s = np.asarray(cell_scores, dtype=float)
    return np.bincount(inv, weights=s, minlength=len(clones)) / np.bincount(inv, minlength=len(clones))


def clone_auroc(clone_scores, labels):
    return roc_auc_score(labels, clone_scores)


def _boot(n, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def gain_over_kernel(method, kernel, labels, n_boot=2000, seed=97, alpha=0.05, power=0.8):
    """AUROC(method) - AUROC(kernel) on the same clones, paired bootstrap over clones."""
    y, m, k = np.asarray(labels), np.asarray(method, float), np.asarray(kernel, float)
    d = [roc_auc_score(y[b], m[b]) - roc_auc_score(y[b], k[b]) for b in _boot(len(y), n_boot, seed)
         if y[b].min() != y[b].max()]
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    g = roc_auc_score(y, m) - roc_auc_score(y, k)
    return dict(gain=g, lo=lo, hi=hi, mdd=mdd_factor(alpha, power) * (hi - lo),
                verdict="above" if lo > 0 else ("below" if hi < 0 else "not separated"))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _oof(X, y, reps, seed):
    out = np.zeros(len(y))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], y[tr])
            out[te] += mdl.predict_proba(X[te])[:, 1]
    return out / reps


def increment_over_kernel(method, kernel, labels, reps=20, n_boot=2000, seed=97, alpha=0.05):
    """Out-of-fold AUROC of kernel + method minus kernel alone (cross-fitted logistic models over
    clones), with a paired bootstrap over clones of the out-of-fold predictions."""
    y = np.asarray(labels)
    pa = _oof(_logit(kernel)[:, None], y, reps, 1000)
    pb = _oof(np.c_[_logit(kernel), _logit(method)], y, reps, 1000)
    d = [roc_auc_score(y[b], pb[b]) - roc_auc_score(y[b], pa[b]) for b in _boot(len(y), n_boot, seed)
         if y[b].min() != y[b].max()]
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(increment=roc_auc_score(y, pb) - roc_auc_score(y, pa), lo=lo, hi=hi,
                adds=bool(lo > 0))


def score_table(methods, kernel, labels, kernel_settings=None, **kw):
    """methods: {name: clone scores}; kernel: clone scores of the reference kernel;
    kernel_settings: optional {name: clone scores} of the kernel at other graph settings."""
    import pandas as pd
    rows = []
    kb = [clone_auroc(v, labels) for v in (kernel_settings or {}).values()]
    for name, s in methods.items():
        g = gain_over_kernel(s, kernel, labels, **kw)
        inc = increment_over_kernel(s, kernel, labels)
        rows.append(dict(method=name, auroc=clone_auroc(s, labels), kernel_auroc=clone_auroc(kernel, labels),
                         kernel_min=min(kb) if kb else np.nan, kernel_max=max(kb) if kb else np.nan,
                         **g, increment=inc["increment"], inc_lo=inc["lo"], inc_hi=inc["hi"],
                         adds=inc["adds"]))
    return pd.DataFrame(rows)
