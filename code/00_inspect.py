"""Inspect the LARRY in-vitro h5ad (figshare 20780344) before any analysis.

Prints the structures the preregistration needs: obs columns and their value
counts, layer names and sparsity, clone matrix shape and occupancy.
"""
import sys
import numpy as np
import scipy.sparse as sp
import anndata as ad

path = sys.argv[1]
A = ad.read_h5ad(path)
print("shape", A.shape)
print("layers", list(A.layers.keys()))
print("obsm", list(A.obsm.keys()))
print("obsp", list(A.obsp.keys()))
print("uns", list(A.uns.keys()))
print("var columns", list(A.var.columns))
print()
print("obs columns", list(A.obs.columns))
for c in A.obs.columns:
    v = A.obs[c]
    n = v.nunique(dropna=True)
    print(f"--- {c}: dtype={v.dtype} nunique={n}")
    if n <= 30:
        print(v.value_counts(dropna=False).to_string())
    else:
        print(v.head(5).to_string())
print()
for k in A.layers:
    L = A.layers[k]
    if sp.issparse(L):
        print(k, "sparse", L.dtype, "nnz", L.nnz, "sum", float(L.sum()))
    else:
        print(k, "dense", L.dtype, "sum", float(np.asarray(L).sum()))
print("X", type(A.X), A.X.dtype if hasattr(A.X, "dtype") else "?",
      float(A.X.sum()) if A.X is not None else None)
print()
for k in A.obsm:
    M = A.obsm[k]
    print("obsm", k, type(M).__name__, getattr(M, "shape", None), getattr(M, "dtype", None))
if "X_clone" in A.obsm:
    C = A.obsm["X_clone"]
    if not sp.issparse(C):
        C = sp.csr_matrix(C)
    per_cell = np.asarray(C.sum(1)).ravel()
    per_clone = np.asarray(C.sum(0)).ravel()
    print("X_clone shape", C.shape, "nnz", C.nnz)
    print("cells with 0 clone", int((per_cell == 0).sum()),
          "with 1", int((per_cell == 1).sum()),
          "with >1", int((per_cell > 1).sum()))
    print("clones with >=1 cell", int((per_clone > 0).sum()))
    print("clone size quantiles",
          np.percentile(per_clone[per_clone > 0], [50, 75, 90, 99, 100]))
print()
print(A.var.head(10).to_string())
print(A.obs.head(5).to_string())
