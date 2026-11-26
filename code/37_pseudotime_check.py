"""Is the pseudotime field's backward sign the pseudotime's fault or the
projection's?

36_pseudotime_field.py found the pseudotime kernel's projected field pointing
the wrong way: negative on the forward boundaries, positive on the reverse ones,
and anti-correlated with the clonal advance. Before that is written up as
"velocity's direction carries more than a maturation gradient", two mundane
explanations have to be ruled out.

  1. The pseudotime may be locally mis-ordered even though it separates mature
     from immature cells on average. Checked directly: among the boundary pairs
     the metric uses, how often does the mature neighbour have the higher
     pseudotime?
  2. The projection may be at fault. The embedding projection weights each
     neighbour by its transition probability minus the uniform 1/n, so for a
     transition matrix that is nearly uniform over its neighbours the residual
     is small and its sign is set by the tail. Checked by asking whether the
     kernel moves mass up the pseudotime at all: the probability-weighted mean
     pseudotime change per cell.

Prints a small table; writes it as csv. Nothing here uses a fate label.

Usage:
  python 37_pseudotime_check.py <h5ad> <featdir> <out.csv>
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import numpy.testing as npt
import anndata as ad
import scanpy as sc
import scvelo as scv

_orig = npt.assert_array_equal


def _compat(*a, **k):
    if "x" in k:
        k["actual"] = k.pop("x")
    if "y" in k:
        k["desired"] = k.pop("y")
    return _orig(*a, **k)


npt.assert_array_equal = _compat
for _cls in (sp.csr_array, sp.csc_array):
    if not hasattr(_cls, "getnnz"):
        _cls.getnnz = _cls._getnnz
import cellrank as cr  # noqa: E402

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]
UNDIFF = "Undifferentiated"
N_PCS, N_NEIGHBORS = 30, 30


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("featdir")
    ap.add_argument("out")
    ap.add_argument("--state-col", default="state_info")
    ap.add_argument("--time-col", default="time_info")
    ap.add_argument("--basis", default="emb")
    a = ap.parse_args()

    hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
    B = ad.read_h5ad(a.h5ad)
    state_all = B.obs[a.state_col].astype(str).values
    tvals = pd.to_numeric(B.obs[a.time_col], errors="coerce").values
    scv.pp.filter_genes(B, min_shared_counts=20)
    scv.pp.normalize_per_cell(B)
    sc.pp.log1p(B)
    keep = [g for g in hvg if g in B.var_names]
    B = B[:, keep].copy()
    B.obs["state"] = state_all
    scv.pp.neighbors(B, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS, method="sklearn")
    scv.pp.moments(B)
    state = B.obs["state"].values
    Xe = np.asarray(B.obsm["X_" + a.basis], dtype=np.float64)
    D = sp.csr_matrix(B.obsp["distances"])
    nbr = [D.indices[D.indptr[i]:D.indptr[i + 1]] for i in range(B.n_obs)]

    present = [s for s in MATURE if (state == s).sum() >= 30]
    is_term = np.zeros(B.n_obs, bool)
    for s in present:
        is_term |= (state == s)
    sc.tl.diffmap(B)
    dm = B.obsm["X_diffmap"][:, 1:11]
    cand = (state == UNDIFF)
    if len(np.unique(tvals)) > 1:
        earliest = cand & (tvals == np.nanmin(tvals))
        if earliest.sum() > 0:
            cand = earliest
    centre = dm[is_term].mean(0)
    dist = np.linalg.norm(dm - centre, axis=1)
    B.uns["iroot"] = int(np.where(cand)[0][np.argmax(dist[cand])])
    sc.tl.dpt(B)
    pt = B.obs["dpt_pseudotime"].values.astype(float)
    if np.nanmean(pt[is_term]) < np.nanmean(pt[~is_term]):
        pt = np.nanmax(pt) - pt
    say("pseudotime: mature minus other = %.3f" % float(
        np.nanmean(pt[is_term]) - np.nanmean(pt[~is_term])))
    B.obs["dpt_pseudotime"] = pt

    rows = []

    # 1. local ordering on exactly the pairs the boundary metric uses
    src = np.where(state == UNDIFF)[0]
    higher, total = 0, 0
    diffs = []
    for i in src:
        js = nbr[i]
        js = js[is_term[js]]
        if js.size == 0:
            continue
        higher += int((pt[js] > pt[i]).sum())
        total += js.size
        diffs.append(float(np.mean(pt[js] - pt[i])))
    rows.append(dict(quantity="boundary pairs (undifferentiated, mature neighbour)",
                     value=total))
    rows.append(dict(quantity="fraction where the mature neighbour has the higher pseudotime",
                     value=round(higher / total, 4) if total else np.nan))
    rows.append(dict(quantity="mean pseudotime change to a mature neighbour",
                     value=round(float(np.mean(diffs)), 5) if diffs else np.nan))

    # 2. does the kernel move mass up the pseudotime?
    pk = cr.kernels.PseudotimeKernel(B, time_key="dpt_pseudotime")
    pk.compute_transition_matrix()
    T = sp.csr_matrix(pk.transition_matrix)
    up = np.zeros(B.n_obs)
    nz = np.zeros(B.n_obs, dtype=int)
    for i in range(B.n_obs):
        s, e = T.indptr[i], T.indptr[i + 1]
        js, pi = T.indices[s:e], T.data[s:e]
        nz[i] = js.size
        if js.size:
            up[i] = float((pi * (pt[js] - pt[i])).sum())
    rows.append(dict(quantity="kernel rows with at least one neighbour",
                     value=int((nz > 0).sum())))
    rows.append(dict(quantity="median neighbours per kernel row",
                     value=float(np.median(nz))))
    rows.append(dict(quantity="probability-weighted mean pseudotime change, all cells",
                     value=round(float(np.mean(up)), 6)))
    rows.append(dict(quantity="same, undifferentiated cells only",
                     value=round(float(np.mean(up[state == UNDIFF])), 6)))
    rows.append(dict(quantity="fraction of cells whose weighted change is positive",
                     value=round(float((up > 0).mean()), 4)))

    # 3. is the pseudotime gradient expressible in the two-dimensional basis?
    for k in range(Xe.shape[1]):
        rows.append(dict(quantity="corr(pseudotime, embedding axis %d)" % (k + 1),
                         value=round(float(np.corrcoef(pt, Xe[:, k])[0, 1]), 4)))
    # and how well the embedding's own geometry carries maturity
    mat = is_term.astype(float)
    for k in range(Xe.shape[1]):
        rows.append(dict(quantity="corr(is mature, embedding axis %d)" % (k + 1),
                         value=round(float(np.corrcoef(mat, Xe[:, k])[0, 1]), 4)))

    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    say("\n" + R.to_string(index=False))
    say("wrote %s" % a.out)


if __name__ == "__main__":
    main()
