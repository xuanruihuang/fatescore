"""Shared clone-grouped cross-validation machinery.

One place defines the folds, the tuning grids and the fold-internal fitting of
PCA, scaling and hyperparameters, so every arm in every script is evaluated the
same way.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

GRIDS = {
    "logreg": [dict(C=c) for c in (1e-3, 1e-2, 1e-1, 1, 10, 100)],
    "knn": [dict(n_neighbors=k) for k in (5, 10, 15, 25, 50, 100)],
    "hgb": [dict(max_leaf_nodes=n, learning_rate=lr)
            for n in (4, 8, 16) for lr in (0.03, 0.1, 0.3)],
}


def make(kind, p, n_train=None):
    if kind == "logreg":
        return LogisticRegression(max_iter=4000, **p)
    if kind == "knn":
        # a small training fold can hold fewer cells than the grid's largest k
        q = dict(p)
        if n_train is not None:
            q["n_neighbors"] = max(1, min(q["n_neighbors"], n_train - 1))
        return KNeighborsClassifier(**q)
    return HistGradientBoostingClassifier(random_state=0, max_iter=200,
                                          early_stopping=False, **p)


def cp10k_log(M):
    tot = M.sum(1, keepdims=True)
    tot = np.where(tot == 0, 1, tot)
    return np.log1p(M / tot * 1e4)


def folds_of(groups, seed, k):
    """Split so that no group is in two folds; `seed` reshuffles the groups."""
    u = np.unique(groups)
    r = np.random.default_rng(seed)
    r.shuffle(u)
    assign = {c: i % k for i, c in enumerate(u)}
    f = np.array([assign[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def build(blocks, pca_blocks, raw_blocks, tr, te, pcs, pcs_by_block=None):
    """Fit PCA and scaling on the training rows only; return train/test X.

    `pcs_by_block` overrides the number of components for named blocks, which
    is how an arm can carry, say, ten velocity components instead of fifty.
    """
    Ptr, Pte = [], []
    for b in blocks:
        if b in raw_blocks:
            M = raw_blocks[b]
            Ptr.append(M[tr])
            Pte.append(M[te])
        else:
            M = pca_blocks[b]
            want = pcs if pcs_by_block is None else pcs_by_block.get(b, pcs)
            k = min(want, len(tr) - 1, M.shape[1])
            p = PCA(n_components=k, random_state=0).fit(M[tr])
            Ptr.append(p.transform(M[tr]))
            Pte.append(p.transform(M[te]))
    Xtr = np.hstack(Ptr) if Ptr else np.zeros((len(tr), 1))
    Xte = np.hstack(Pte) if Pte else np.zeros((len(te), 1))
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def run_arm(arm, blocks, pca_blocks, raw_blocks, y, groups, seed, k, pcs,
            pcs_by_block=None):
    """Out-of-fold predictions; classifier and hyperparameters chosen inside
    the training folds by a 3-fold clone-grouped inner split."""
    oof = np.full(len(y), np.nan)
    picked = []
    for tr, te in folds_of(groups, seed, k):
        if len(blocks) == 0:
            oof[te] = y[tr].mean()
            picked.append("prior")
            continue
        inner = folds_of(groups[tr], seed + 1000, 3)
        best, bestscore = None, -np.inf
        for kind, grid in GRIDS.items():
            for p in grid:
                sc = []
                for itr, ite in inner:
                    a, b = tr[itr], tr[ite]
                    if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
                        continue
                    Xa, Xb = build(blocks, pca_blocks, raw_blocks, a, b, pcs,
                                   pcs_by_block)
                    m = make(kind, p, len(a)).fit(Xa, y[a])
                    sc.append(roc_auc_score(y[b], m.predict_proba(Xb)[:, 1]))
                if sc and np.mean(sc) > bestscore:
                    best, bestscore = (kind, p), float(np.mean(sc))
        Xtr, Xte = build(blocks, pca_blocks, raw_blocks, tr, te, pcs,
                         pcs_by_block)
        m = make(best[0], best[1], len(tr)).fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
        picked.append("%s:%s" % (best[0], best[1]))
    return arm, seed, oof, picked


def clone_level(oof, y, groups):
    return pd.DataFrame(dict(p=oof, y=y, g=groups)).groupby("g").agg(
        p=("p", "mean"), y=("y", "first"))


def auroc(d):
    return np.nan if d["y"].nunique() < 2 else roc_auc_score(d["y"], d["p"])


def thin_to(counts, target, seed):
    """Thin each row of `counts` without replacement to `target[i]` reads."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(counts)
    for i in range(counts.shape[0]):
        colors = counts[i].astype(np.int64)
        n = int(min(target[i], colors.sum()))
        if n > 0:
            out[i] = rng.multivariate_hypergeometric(colors, n)
    return out


# ---- multiclass and regression variants, same folds and tuning budget ------
from sklearn.linear_model import Ridge                       # noqa: E402
from sklearn.neighbors import KNeighborsRegressor            # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor    # noqa: E402

GRIDS_REG = {
    "ridge": [dict(alpha=a) for a in (1e-2, 1e-1, 1, 10, 100, 1000)],
    "knn": [dict(n_neighbors=k) for k in (5, 10, 15, 25, 50, 100)],
    "hgb": [dict(max_leaf_nodes=n, learning_rate=lr)
            for n in (4, 8, 16) for lr in (0.03, 0.1, 0.3)],
}


def make_reg(kind, p):
    if kind == "ridge":
        return Ridge(**p)
    if kind == "knn":
        return KNeighborsRegressor(**p)
    return HistGradientBoostingRegressor(random_state=0, max_iter=200,
                                         early_stopping=False, **p)


def run_arm_multi(arm, blocks, pca_blocks, raw_blocks, y, groups, seed, k, pcs,
                  classes):
    """Out-of-fold class probabilities for a multiclass label."""
    from sklearn.metrics import roc_auc_score as _auc
    P = np.full((len(y), len(classes)), np.nan)
    picked = []
    for tr, te in folds_of(groups, seed, k):
        if len(blocks) == 0:
            for j, c in enumerate(classes):
                P[te, j] = (y[tr] == c).mean()
            picked.append("prior")
            continue
        inner = folds_of(groups[tr], seed + 1000, 3)
        best, bestscore = None, -np.inf
        for kind, grid in GRIDS.items():
            for p in grid:
                sc = []
                for itr, ite in inner:
                    a, b = tr[itr], tr[ite]
                    if len(np.unique(y[a])) < 2:
                        continue
                    Xa, Xb = build(blocks, pca_blocks, raw_blocks, a, b, pcs)
                    m = make(kind, p).fit(Xa, y[a])
                    pr = _align(m, Xb, classes)
                    keep = np.isin(y[b], np.unique(y[a]))
                    if keep.sum() == 0 or len(np.unique(y[b][keep])) < 2:
                        continue
                    sc.append(_macro_auc(y[b][keep], pr[keep], classes))
                if sc and np.mean(sc) > bestscore:
                    best, bestscore = (kind, p), float(np.mean(sc))
        Xtr, Xte = build(blocks, pca_blocks, raw_blocks, tr, te, pcs)
        m = make(*best).fit(Xtr, y[tr])
        P[te] = _align(m, Xte, classes)
        picked.append("%s:%s" % (best[0], best[1]))
    return arm, seed, P, picked


def _align(model, X, classes):
    pr = model.predict_proba(X)
    out = np.zeros((X.shape[0], len(classes)))
    for j, c in enumerate(classes):
        w = np.where(model.classes_ == c)[0]
        if len(w):
            out[:, j] = pr[:, w[0]]
    return out


def _macro_auc(y, P, classes):
    from sklearn.metrics import roc_auc_score as _auc
    vals = []
    for j, c in enumerate(classes):
        lab = (y == c).astype(int)
        if lab.min() == lab.max():
            continue
        vals.append(_auc(lab, P[:, j]))
    return float(np.mean(vals)) if vals else np.nan


def run_arm_reg(arm, blocks, pca_blocks, raw_blocks, y, groups, seed, k, pcs):
    """Out-of-fold predictions for a continuous target."""
    from scipy.stats import spearmanr
    oof = np.full(len(y), np.nan)
    picked = []
    for tr, te in folds_of(groups, seed, k):
        if len(blocks) == 0:
            oof[te] = y[tr].mean()
            picked.append("prior")
            continue
        inner = folds_of(groups[tr], seed + 1000, 3)
        best, bestscore = None, -np.inf
        for kind, grid in GRIDS_REG.items():
            for p in grid:
                sc = []
                for itr, ite in inner:
                    a, b = tr[itr], tr[ite]
                    Xa, Xb = build(blocks, pca_blocks, raw_blocks, a, b, pcs)
                    m = make_reg(kind, p).fit(Xa, y[a])
                    r = spearmanr(y[b], m.predict(Xb)).statistic
                    if np.isfinite(r):
                        sc.append(r)
                if sc and np.mean(sc) > bestscore:
                    best, bestscore = (kind, p), float(np.mean(sc))
        Xtr, Xte = build(blocks, pca_blocks, raw_blocks, tr, te, pcs)
        m = make_reg(*best).fit(Xtr, y[tr])
        oof[te] = m.predict(Xte)
        picked.append("%s:%s" % (best[0], best[1]))
    return arm, seed, oof, picked
