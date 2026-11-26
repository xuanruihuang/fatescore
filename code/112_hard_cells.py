"""Hard cells: is there room for dynamics in early cells whose present state does not yet decide fate,
and does any method beat the connectivity kernel there?

Stratifier, independent of the kernel and of every method: an out-of-fold prediction of the clone's
fate from day-2 (early) expression alone. Expression: total counts normalised to 1e4 per cell, log1p,
2,000 highly variable genes chosen on the early cells without labels; inside each training fold,
scaling and a 50-component PCA are fitted, then a logistic regression (C = 1) on the components (plus
the sorted-population label on LARRY). Folds are clone-grouped and stratified by the clone label
(5 folds, 10 repetitions); cell predictions are averaged over repetitions and then within clones.
Decidedness = |p - 0.5|; clones are split into tertiles (hard = least decided).

Within each tertile: clone AUROC of the expression classifier, of the connectivity kernel (reference
setting) and of every method; gain of each method over the kernel with a paired bootstrap over the
tertile's clones (2,000 resamples, seed 97). Sanity: the kernel at another graph setting.

LARRY task as 99 (clones with >= 3 sisters dominated by neutrophils or monocytes, cell shares from
feat_all/nm_*.npy); CellTag-multi tasks as 97 (clones with >= 3 mature cells, share of --pos).

Usage: 112_hard_cells.py larry <h5ad> <featdir_all> <d2_labels.tsv> <out.csv>
       112_hard_cells.py ct <h5ad> <featdir> <pos> <out.csv>
"""
import ast
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

MODE = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "99_calibrated_table.py"), encoding="utf-8").read()
NAMES = ast.literal_eval(src[src.index("NAMES = ") + 8: src.index("}", src.index("NAMES = ")) + 1])
NAMES.update({"snap_cospar_state": "CoSpar (state only)"})


def expression(h5, cells):
    A = ad.read_h5ad(h5, backed="r")
    idx = pd.Index(A.obs_names).get_indexer(cells)
    assert (idx >= 0).all(), "early cells missing from the object"
    B = A[np.sort(idx)].to_memory()
    B = B[pd.Index(B.obs_names).get_indexer(cells)].copy()
    sc.pp.normalize_total(B, target_sum=1e4)
    sc.pp.log1p(B)
    sc.pp.highly_variable_genes(B, n_top_genes=2000)
    X = B[:, B.var.highly_variable.values].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return X.astype(np.float32), B.obs


def oof_expression(X, extra, cell_clone, ucl, ydom, reps=10):
    inv = pd.Index(ucl).get_indexer(cell_clone)
    ycell = ydom[inv]
    out = np.zeros(len(cell_clone))
    for r in range(reps):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=2000 + r)
        for tr, te in cv.split(X, ycell, groups=inv):
            sc_ = StandardScaler().fit(X[tr])
            pca = PCA(50, random_state=r).fit(sc_.transform(X[tr]))
            Z = lambda i: np.c_[pca.transform(sc_.transform(X[i])), extra[i]] if extra is not None else pca.transform(sc_.transform(X[i]))  # noqa: E731
            m = LogisticRegression(C=1.0, max_iter=3000).fit(Z(tr), ycell[tr])
            out[te] += m.predict_proba(Z(te))[:, 1]
    out /= reps
    return np.bincount(inv, weights=out, minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))


def tertile_table(ds, ydom, expr, kern, sanity, methods):
    dec = np.abs(expr - 0.5)
    q = np.quantile(dec, [1 / 3, 2 / 3])
    tert = np.where(dec <= q[0], "hard", np.where(dec <= q[1], "middle", "easy"))
    rows = []
    for t in ["hard", "middle", "easy"]:
        m = tert == t
        y = ydom[m]
        rng = np.random.default_rng(97)
        B = rng.integers(0, m.sum(), size=(2000, m.sum()))
        k = kern[m]
        base = dict(dataset=ds, tertile=t, n_clones=int(m.sum()), n_pos=int(y.sum()),
                    decided_range="%.3f-%.3f" % (dec[m].min(), dec[m].max()),
                    auroc_expression=roc_auc_score(y, expr[m]), auroc_kernel=roc_auc_score(y, k))
        for lab, c in [("kernel, other setting", sanity)] + list(methods.items()):
            c = c[m]
            d = [roc_auc_score(y[b], c[b]) - roc_auc_score(y[b], k[b]) for b in B if y[b].min() != y[b].max()]
            lo, hi = np.percentile(d, [2.5, 97.5])
            g = roc_auc_score(y, c) - roc_auc_score(y, k)
            rows.append(dict(base, method=lab, auroc=roc_auc_score(y, c), gain=g, lo=lo, hi=hi,
                             verdict="above" if lo > 0 else ("below" if hi < 0 else "not separated")))
    return pd.DataFrame(rows)


if MODE == "larry":
    H5, FD, LAB, OUT = sys.argv[2:6]
    T = pd.read_csv(LAB, sep="\t")
    msk = (T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])
    im = np.where(msk.values)[0]
    cells = T.loc[msk, "cell"].values
    cl = T.loc[msk, "clone"].values
    ucl, inv = np.unique(cl, return_inverse=True)
    ydom = (T.loc[msk].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)

    def share(name):
        M = np.load(os.path.join(FD, "nm_" + name + ".npy"))
        if M.ndim == 2 and M.shape[1] > 1:
            s = M[:, 0] + M[:, 1]
            M = np.where(s > 0, M[:, 0] / np.where(s > 0, s, 1), 0.5)
        s = M.ravel()[im]
        return np.bincount(inv, weights=s, minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))

    X, obs = expression(H5, cells)
    pop = pd.get_dummies(T.loc[msk, "pop"].astype(str)).values.astype(float)
    expr = oof_expression(X, pop, cl, ucl, ydom)
    kern = share("cr_conn_n30_p30_d1")
    sanity = share("cr_conn_n15_p50_d0")
    methods = {lab: share(k) for k, lab in NAMES.items() if os.path.exists(os.path.join(FD, "nm_" + k + ".npy"))}
    R = tertile_table("LARRY", ydom, expr, kern, sanity, methods)
else:
    H5, FD, POS, OUT = sys.argv[2:6]
    L = pd.read_csv(os.path.join(FD, "labels.tsv"), sep="\t")
    early = json.load(open(os.path.join(FD, "d2_names.json")))
    assert list(L["cell"]) == early
    task = (L["n_mature"] >= 3).values
    it = np.where(task)[0]
    cells = L.loc[task, "cell"].values
    cl = L.loc[task, "clone"].values
    ucl, inv = np.unique(cl, return_inverse=True)
    ydom = (L.loc[task].groupby("clone")["dom"].first().reindex(ucl).values == POS).astype(int)

    def share(name):
        M = np.load(os.path.join(FD, name + ".npy"))
        pops = json.load(open(os.path.join(FD, name + "_log.json")))["populations"]
        tot = M.sum(1)
        s = np.where(tot > 0, M[:, pops.index(POS)] / np.where(tot > 0, tot, 1), 0.5)[it]
        return np.bincount(inv, weights=s, minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))

    X, obs = expression(H5, cells)
    expr = oof_expression(X, None, cl, ucl, ydom)
    kern = share("cr_conn_all_n30_p30_d1")
    sanity = share("cr_conn_all_n15_p50_d0")
    methods = {lab: share(k) for k, lab in NAMES.items()
               if os.path.exists(os.path.join(FD, k + ".npy")) and os.path.exists(os.path.join(FD, k + "_log.json"))}
    R = tertile_table(os.path.basename(FD.rstrip("/")), ydom, expr, kern, sanity, methods)

R.round(4).to_csv(OUT, index=False)
pd.set_option("display.width", 220)
print(R.drop_duplicates("tertile")[["dataset", "tertile", "n_clones", "n_pos", "decided_range", "auroc_expression", "auroc_kernel"]].round(3).to_string(index=False))
print(R[["tertile", "method", "auroc", "gain", "lo", "hi", "verdict"]].round(3).to_string(index=False))
