"""Inspect the CellTag reprogramming object and count its state-fate clones.

Usage: 11_celltag_inspect.py <h5ad>

The second system is Biddy et al.'s CellTag time course of fibroblast to
induced endoderm progenitor reprogramming, in the form released with CellRank
(figshare 13105481), which carries spliced and unspliced layers alongside the
CellTag barcode calls from three tagging rounds. This script only reports what
is in the file and how many clones are observed both early and late; no model
is fitted.
"""
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

A = ad.read_h5ad(sys.argv[1])
print("shape", A.shape)
print("layers", list(A.layers.keys()))
print("obsm", list(A.obsm.keys()))
print("var columns", list(A.var.columns))
print()
for c in A.obs.columns:
    v = A.obs[c]
    n = v.nunique(dropna=True)
    print("--- %s dtype=%s nunique=%d nmissing=%d"
          % (c, v.dtype, n, int(v.isna().sum())))
    if n <= 20:
        print(v.value_counts(dropna=False).to_string())
    else:
        print(v.dropna().head(4).to_string())
print()
for k in A.layers:
    L = A.layers[k]
    tot = float(L.sum())
    print(k, "sum %.0f" % tot,
          "per-cell median %.0f" % np.median(np.asarray(L.sum(1)).ravel()))

# clone x time structure for each tagging round
day_col = [c for c in A.obs.columns
           if c.lower() in ("reprogramming_day", "day", "timecourse")]
print("\nday-like columns:", day_col)
for dc in day_col:
    print(A.obs[dc].value_counts(dropna=False).to_string())

tag_cols = [c for c in A.obs.columns if "celltag" in c.lower()]
print("\nCellTag columns:", tag_cols)
if day_col and tag_cols:
    dc = day_col[0]
    d = A.obs[dc].astype(str)
    for tc in tag_cols:
        t = A.obs[tc].astype(str)
        ok = (~A.obs[tc].isna()) & (t != "nan") & (t != "")
        print("\n=== %s: %d tagged cells, %d distinct clones"
              % (tc, int(ok.sum()), t[ok].nunique()))
        tab = pd.crosstab(t[ok], d[ok])
        print("cells per day among tagged:")
        print(d[ok].value_counts().to_string())
        days = list(tab.columns)
        print("clones present on each day:")
        print((tab > 0).sum(0).to_string())
        print("clones spanning >=2 days: %d" % int(((tab > 0).sum(1) >= 2).sum()))
