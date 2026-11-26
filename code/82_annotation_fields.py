"""Can a field built from the annotation alone pass the reverse CBDir score?

71_direction_embedding.py scores a field pointing from each cell towards the
centroid of all mature cells. A stricter annotation-only field points each cell
towards the centroid of its nearest mature population (nearest in the space
where the metric is computed). Neither uses unspliced counts, sampling time or
dynamics. Same graph, same UMAP (seed 0) and same scoring as 71, with intervals
that resample clones.

Usage: 82_annotation_fields.py <h5ad> <featdir> <out.csv>
"""
import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import scvelo as scv

if not hasattr(sp.csr_array, "getnnz"):
    sp.csr_array.getnnz = sp.csr_array._getnnz

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]
H5, FD, OUT = sys.argv[1:4]
hvg = json.load(open(FD + "/hvg.json"))
B = ad.read_h5ad(H5)
state = B.obs["state_info"].astype(str).values
scv.pp.filter_genes(B, min_shared_counts=20)
scv.pp.normalize_per_cell(B)
sc.pp.log1p(B)
B = B[:, hvg].copy()
scv.pp.neighbors(B, n_pcs=30, n_neighbors=30, method="sklearn")
X = np.asarray(B.obsm["X_pca"][:, :30], dtype=np.float64)
D = sp.csr_matrix(B.obsp["distances"])
nbr = [D.indices[D.indptr[i]:D.indptr[i + 1]] for i in range(B.n_obs)]
sc.tl.umap(B, random_state=0)
U = np.asarray(B.obsm["X_umap"], dtype=np.float64)
Cm = sp.csr_matrix(B.obsm["X_clone"])
has = np.asarray(Cm.sum(1)).ravel() > 0
clone = np.where(has, np.asarray(Cm.argmax(1)).ravel(), -1 - np.arange(B.n_obs))
targets = [t for t in MATURE if (state == t).sum() >= 30]
tmask = [state == t for t in targets]
und = state == "Undifferentiated"
rng = np.random.default_rng(82)


def cos_rows(A, Bm):
    num = (A * Bm).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(Bm, axis=1)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def scores(V, P, src, tgts):
    vals, cells = [], []
    for tgt in tgts:
        for i in np.where(src)[0]:
            js = nbr[i][tgt[nbr[i]]]
            if js.size:
                c = cos_rows(np.repeat(V[i][None], js.size, 0), P[js] - P[i])
                c = c[np.isfinite(c)]
                if c.size:
                    vals.append(c.mean()); cells.append(i)
    return np.array(vals), np.array(cells)


def boot(vals, cells):
    g = clone[cells]
    ug, inv = np.unique(g, return_inverse=True)
    out = np.empty(2000)
    for b in range(2000):
        w = np.bincount(rng.integers(0, ug.size, ug.size), minlength=ug.size)[inv]
        out[b] = (vals * w).sum() / max(w.sum(), 1)
    return out


rows = []
for space, P in (("umap", U), ("pca30", X)):
    cents = np.stack([P[m].mean(0) for m in tmask])
    dist = np.linalg.norm(P[:, None, :] - cents[None], axis=2)
    nearest = cents[dist.argmin(1)] - P
    glob = P[np.isin(state, MATURE)].mean(0)[None, :] - P
    for name, V in (("nearest_mature_centroid", nearest), ("all_mature_centroid", glob)):
        fw, fc = scores(V, P, und, tmask)
        parts = [scores(V, P, tm, [und]) for tm in tmask]
        rv = np.concatenate([p[0] for p in parts]); rc = np.concatenate([p[1] for p in parts])
        d = boot(fw, fc) - boot(rv, rc)
        rows.append(dict(space=space, field=name, cbdir_fwd=fw.mean(), cbdir_rev=rv.mean(),
                         fwd_minus_rev=fw.mean() - rv.mean(), diff_lo=np.percentile(d, 2.5),
                         diff_hi=np.percentile(d, 97.5)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).round(4).to_csv(OUT, index=False)
