"""Control arms whose fate signal does not come from the expression manifold.

46_geometry_null.py asks whether an arm's clonal-fate AUROC is decided by how
much its errors resemble the expression baseline's, or whether that relation is
just two accurate predictors having to agree. The null curve answers half of
it; the other half needs arms that score above chance for a different reason.

This writes those arms into a feature directory as `ctl_*.npy`, one row per
early cell in the directory's own `d2_names.json` order, so 30_kernel_eval.py
picks them up with `--prefixes ctl_` and scores them on the same folds, with
the same classifier and the same tuning, as every real arm.

Two families:

  technical / non-manifold -- library size, detected genes, mitochondrial
    fraction, clone size, the starting population on its own, and a pure
    random feature. None of these is the expression neighbourhood structure
    that every kernel and every transport method is built from.

  capacity controls -- random Gaussian projections of the SAME expression
    block the baseline uses, down to 1, 2, 5 and 10 columns. These are
    expression geometry, only less of it, so they say whether a point's
    position on the null curve is about capacity rather than about where the
    signal comes from.

Usage:
  python 47_control_arms.py <featdir> <labels.tsv> --h5ad <file> \
      --expr-block raw_spliced
"""
import argparse
import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 11


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def col(x):
    """A single feature column, float64, as (n, 1)."""
    return np.asarray(x, dtype=np.float64).reshape(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("featdir")
    ap.add_argument("labels")
    ap.add_argument("--h5ad", required=True,
                    help="the arm's object, for the raw counts per cell")
    ap.add_argument("--expr-block", default="raw_spliced",
                    help="the expression block the baseline is built from; "
                         "the capacity controls are random projections of it")
    ap.add_argument("--mito-prefix", default="mt-",
                    help="gene-name prefix for the mitochondrial fraction")
    a = ap.parse_args()

    names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
    T = pd.read_csv(a.labels, sep="\t")
    assert list(T["cell"]) == names, "label table and feature rows disagree"
    n = len(names)
    say("%d early cells in %s" % (n, a.featdir))

    A = ad.read_h5ad(a.h5ad)
    A.obs_names_make_unique()
    miss = [c for c in names if c not in set(A.obs_names)]
    assert not miss, "%d early cells are not in the object, e.g. %s" % (
        len(miss), miss[:3])
    B = A[names]
    X = B.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()

    total = np.asarray(X.sum(1)).ravel()
    ngene = np.asarray((X > 0).sum(1)).ravel()
    gl = np.asarray(B.var_names, dtype=object)
    mt = np.array([str(g).lower().startswith(a.mito_prefix.lower())
                   for g in gl])
    say("%d mitochondrial genes by prefix %r" % (int(mt.sum()), a.mito_prefix))
    if mt.any():
        mito = np.asarray(X[:, mt].sum(1)).ravel() / np.maximum(total, 1.0)
    else:
        mito = None

    rng = np.random.default_rng(SEED)
    out = {}
    out["ctl_libsize"] = col(np.log1p(total))
    out["ctl_ngenes"] = col(ngene)
    if mito is not None:
        out["ctl_mito"] = col(mito)
    out["ctl_nsis"] = col(T["n_sis"].to_numpy())
    # The starting population is already a covariate in every arm, so an arm
    # built on a copy of it is the "population alone" floor.
    pops = sorted(set(T["pop"].astype(str)))
    out["ctl_pop"] = np.stack(
        [(T["pop"].astype(str).to_numpy() == p).astype(np.float64)
         for p in pops], axis=1)
    out["ctl_rand"] = col(rng.standard_normal(n))

    E = np.load(os.path.join(a.featdir, a.expr_block + ".npy"))
    assert E.shape[0] == n, "%s has %d rows, expected %d" % (
        a.expr_block, E.shape[0], n)
    Ez = (E - E.mean(0)) / np.maximum(E.std(0), 1e-12)
    for k in (1, 2, 5, 10):
        W = rng.standard_normal((E.shape[1], k)) / np.sqrt(E.shape[1])
        out["ctl_rp%d" % k] = Ez @ W

    log = {}
    for k, v in sorted(out.items()):
        v = np.ascontiguousarray(np.asarray(v, dtype=np.float32))
        assert v.shape[0] == n, "%s has %d rows" % (k, v.shape[0])
        assert np.isfinite(v).all(), "%s has non-finite entries" % k
        np.save(os.path.join(a.featdir, k + ".npy"), v)
        log[k] = list(v.shape)
        say("wrote %s %s" % (k, v.shape))
    json.dump(dict(seed=SEED, n_cells=n, expr_block=a.expr_block,
                   mito_prefix=a.mito_prefix,
                   n_mito_genes=int(mt.sum()), pops=pops, blocks=log),
              open(os.path.join(a.featdir, "control_arms_log.json"), "w"),
              indent=1)
    say("wrote control_arms_log.json")


if __name__ == "__main__":
    main()
