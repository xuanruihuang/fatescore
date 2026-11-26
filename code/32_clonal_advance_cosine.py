"""The published cosine against the clonal advance, with the controls it lacks.

A positive claim in the literature is that a velocity field agrees with the
direction a clone actually advances in: the cosine between the two, reported at
0.38 to 0.63 on this dataset, against another velocity method only -- no
expression baseline, no split, and no statement of how reproducible the target
itself is.

This computes the same statistic and adds the three things it is missing:

  * an expression-only field, the mean displacement from a cell to its own
    neighbours, which contains no splicing and no dynamics;
  * a permuted-velocity null, velocities shuffled between early cells of the
    same annotated state, twenty seeds;
  * a ceiling: the clone's late cells are split in half and each half gives its
    own advance vector, so the cosine between the two halves says how much of
    the target direction is reproducible at all. A field cannot be expected to
    beat it.

Everything is aggregated to the clone before averaging, and every interval is a
bootstrap over clones, because the clone is the independent unit.

Usage:
  python 32_clonal_advance_cosine.py <h5ad> <featdir> <labels.tsv> <out_prefix>
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import scvelo as scv

N_PCS, N_NEIGHBORS = 30, 30
N_BOOT = 2000
SEED = 13
N_PERM = 20


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def cosine_rows(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def clone_mean(values, clones, keys):
    """Mean within clone, then one row per clone in `keys` order."""
    s = pd.Series(values, index=clones)
    return s.groupby(level=0).mean().reindex(keys).to_numpy()


def boot_mean(table, rng, n_boot=N_BOOT):
    """Bootstrap the clone-level mean of every column of `table`."""
    n = table.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.nanmean(table[idx], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("featdir")
    ap.add_argument("labels")
    ap.add_argument("out_prefix")
    ap.add_argument("--state-col", default="state_info")
    ap.add_argument("--time-col", default="time_info")
    ap.add_argument("--early", default="2.0")
    ap.add_argument("--min-sis", type=int, default=3)
    a = ap.parse_args()

    hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
    d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
    A = ad.read_h5ad(a.h5ad)
    clone_of = np.asarray(sp.csr_matrix(A.obsm["X_clone"]).argmax(1)).ravel()
    has_clone = np.asarray(sp.csr_matrix(A.obsm["X_clone"]).sum(1)).ravel() > 0
    tcol = A.obs[a.time_col].astype(str)
    is_early = (tcol == a.early).values
    state_all = A.obs[a.state_col].astype(str).values
    say("cells %d, with a clone %d, early %d"
        % (A.n_obs, has_clone.sum(), is_early.sum()))

    B = A.copy()
    scv.pp.filter_genes(B, min_shared_counts=20)
    scv.pp.normalize_per_cell(B)
    sc.pp.log1p(B)
    keep = [g for g in hvg if g in B.var_names]
    assert len(keep) == len(hvg), (len(keep), len(hvg))
    B = B[:, keep].copy()
    sc.tl.pca(B, n_comps=N_PCS, svd_solver="arpack")
    X = np.asarray(B.obsm["X_pca"][:, :N_PCS], dtype=np.float64)
    load = np.asarray(B.varm["PCs"][:, :N_PCS], dtype=np.float64)
    say("pca done, loadings %s" % (load.shape,))

    d2 = np.array([B.obs_names.get_loc(n) for n in d2_names])
    assert is_early[d2].all()

    # the advance vector of each early cell's clone, and the two half-advances
    late = (~is_early) & has_clone
    by_clone = {}
    for i in np.where(late)[0]:
        by_clone.setdefault(clone_of[i], []).append(i)
    adv = np.full((d2.size, N_PCS), np.nan)
    adv_a = np.full((d2.size, N_PCS), np.nan)
    adv_b = np.full((d2.size, N_PCS), np.nan)
    n_late = np.zeros(d2.size, dtype=int)
    for k, i in enumerate(d2):
        if not has_clone[i]:
            continue
        ls = by_clone.get(clone_of[i])
        if not ls:
            continue
        ls = np.array(sorted(ls))
        n_late[k] = ls.size
        adv[k] = X[ls].mean(0) - X[i]
        if ls.size >= 2:
            adv_a[k] = X[ls[0::2]].mean(0) - X[i]
            adv_b[k] = X[ls[1::2]].mean(0) - X[i]
    say("early cells with a clonal advance: %d of %d (median %d late sisters)"
        % (int((n_late > 0).sum()), d2.size,
           int(np.median(n_late[n_late > 0]))))

    # fields, all in the same principal-component space
    fields = {}
    for tag, fn in (("V1_scvelo_steady", "vel_det"),
                    ("V2_scvelo_dynamical", "vel_dyn"),
                    ("V3_velovi", "vel_velovi")):
        p = os.path.join(a.featdir, fn + ".npy")
        if os.path.exists(p):
            fields[tag] = np.nan_to_num(np.load(p).astype(np.float64)) @ load
    # the expression-only field, exact neighbours in the same space
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, algorithm="brute").fit(X)
    _, ind = nn.kneighbors(X[d2])
    fields["E0_neighbour_drift"] = X[ind[:, 1:]].mean(1) - X[d2]
    say("fields: %s" % list(fields))

    T = pd.read_csv(a.labels, sep="\t")
    assert list(T["cell"]) == d2_names
    taskA = ((T["n_sis"] >= a.min_sis)
             & T["dom"].isin(["Neutrophil", "Monocyte"])).values

    rows = []
    for subset, mask in (("all early cells with an advance", n_late > 0),
                         ("task A clones", (n_late > 0) & taskA)):
        sel = np.where(mask)[0]
        clones = clone_of[d2[sel]]
        keys = np.unique(clones)
        cols, names = [], []
        for tag, V in fields.items():
            cols.append(clone_mean(cosine_rows(V[sel], adv[sel]), clones, keys))
            names.append(tag)
        # the ceiling: how reproducible the target direction is
        ok = np.isfinite(adv_a[sel]).all(1) & np.isfinite(adv_b[sel]).all(1)
        ceil = np.full(sel.size, np.nan)
        ceil[ok] = cosine_rows(adv_a[sel][ok], adv_b[sel][ok])
        cols.append(clone_mean(ceil, clones, keys))
        names.append("CEILING_half_vs_half")
        # the permuted-velocity null, on the dynamical field
        rng0 = np.random.default_rng(SEED)
        if "V2_scvelo_dynamical" in fields:
            V = fields["V2_scvelo_dynamical"][sel]
            st = state_all[d2[sel]]
            perms = []
            for s in range(N_PERM):
                r = np.random.default_rng(SEED + 100 + s)
                Vp = V.copy()
                for u in np.unique(st):
                    w = np.where(st == u)[0]
                    Vp[w] = V[w[r.permutation(w.size)]]
                perms.append(clone_mean(cosine_rows(Vp, adv[sel]), clones, keys))
            P = np.stack(perms)
            cols.append(P.mean(0))
            names.append("N1_permuted_dynamical_mean_of_%d" % N_PERM)
        tab = np.column_stack(cols)
        bm = boot_mean(tab, rng0)
        for j, nm in enumerate(names):
            lo, hi = np.nanpercentile(bm[:, j], [2.5, 97.5])
            rows.append(dict(subset=subset, field=nm, clones=len(keys),
                             cosine=round(float(np.nanmean(tab[:, j])), 4),
                             lo=round(float(lo), 4), hi=round(float(hi), 4)))
        # paired differences against the best velocity field
        vel_names = [n for n in names if n.startswith("V")]
        ref = max(vel_names, key=lambda n: np.nanmean(tab[:, names.index(n)]))
        jr = names.index(ref)
        for j, nm in enumerate(names):
            if nm == ref:
                continue
            d = bm[:, j] - bm[:, jr]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rows.append(dict(subset=subset, field="%s minus %s" % (nm, ref),
                             clones=len(keys),
                             cosine=round(float(np.nanmean(tab[:, j])
                                               - np.nanmean(tab[:, jr])), 4),
                             lo=round(float(lo), 4), hi=round(float(hi), 4)))
    R = pd.DataFrame(rows)
    R.to_csv(a.out_prefix + "_cosine.csv", index=False)
    say("\n" + R.to_string(index=False))
    say("wrote %s_cosine.csv" % a.out_prefix)


if __name__ == "__main__":
    main()
