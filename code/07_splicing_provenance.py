"""Checks on the unspliced layer itself, before it is asked to predict anything.

Usage: 07_splicing_provenance.py <h5ad> <out.txt>

The deposit that carries the spliced/unspliced split does not say which
pipeline produced it. Four properties are checked here, all of which a standard
intron-aware assignment must have and an arbitrary matrix need not:

  1. spliced + unspliced + ambiguous reproduces the total count layer;
  2. the unspliced fraction rises with the gene's intronic span, and is near
     zero for genes with no intron to read;
  3. unspliced counts are sparser than spliced counts at the same total, which
     fixes how little per-cell, per-gene ratio information exists;
  4. the day-2 cells' unspliced fraction does not separate the fate classes on
     its own at the level of a single summary statistic.
"""
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

H5AD, OUT = sys.argv[1], sys.argv[2]
A = ad.read_h5ad(H5AD)
out = []


def p(s):
    print(s, flush=True)
    out.append(str(s))


S, U, AMB, M = (A.layers["spliced"], A.layers["unspliced"],
                A.layers["ambiguous"], A.layers["matrix"])
tot = S.sum() + U.sum() + AMB.sum()
p("1. layer arithmetic")
p("   spliced %.0f  unspliced %.0f  ambiguous %.0f  sum %.0f"
  % (S.sum(), U.sum(), AMB.sum(), tot))
p("   matrix layer %.0f   ratio sum/matrix %.6f" % (M.sum(), tot / M.sum()))
d = (S + U + AMB) - M
d = np.abs(d.data) if sp.issparse(d) else np.abs(d)
p("   max |spliced+unspliced+ambiguous - matrix| over all entries: %.4g"
  % (d.max() if d.size else 0.0))

p("\n2. unspliced fraction against genomic span")
span = (A.var["End"].astype(float) - A.var["Start"].astype(float)).values
s_g = np.asarray(S.sum(0)).ravel()
u_g = np.asarray(U.sum(0)).ravel()
keep = (s_g + u_g) >= 100
fr = u_g[keep] / (s_g[keep] + u_g[keep])
sp_kb = span[keep] / 1e3
q = pd.qcut(sp_kb, 6, duplicates="drop")
T = pd.DataFrame(dict(frac=fr, span=sp_kb, q=q)).groupby("q", observed=True).agg(
    n_genes=("frac", "size"), median_span_kb=("span", "median"),
    median_unspliced_frac=("frac", "median"))
p(T.to_string())
from scipy.stats import spearmanr
p("   spearman(unspliced fraction, genomic span) = %.3f over %d genes"
  % (spearmanr(fr, sp_kb).statistic, keep.sum()))

p("\n3. how much ratio information a single cell carries")
u_c = np.asarray(U.sum(1)).ravel()
s_c = np.asarray(S.sum(1)).ravel()
nz_u = np.asarray((U > 0).sum(1)).ravel()
nz_s = np.asarray((S > 0).sum(1)).ravel()
for name, v in [("unspliced UMIs per cell", u_c),
                ("spliced UMIs per cell", s_c),
                ("genes with >=1 unspliced UMI", nz_u),
                ("genes with >=1 spliced UMI", nz_s),
                ("unspliced fraction per cell", u_c / (u_c + s_c))]:
    p("   %-32s median %8.4g  q10 %8.4g  q90 %8.4g"
      % (name, np.median(v), np.percentile(v, 10), np.percentile(v, 90)))
both = np.asarray(((U > 0).multiply(S > 0)).sum(1)).ravel()
p("   genes with both a spliced and an unspliced UMI in the same cell: "
  "median %.0f" % np.median(both))

p("\n4. the per-cell unspliced fraction on its own, day-2 cells by clonal fate")
MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]
C = A.obsm["X_clone"]
C = C if sp.issparse(C) else sp.csr_matrix(C)
clone = np.asarray(C.argmax(1)).ravel()
obs = A.obs.assign(clone=clone, time=A.obs["time_info"].astype(int))
late = obs[(obs["time"] > 2) & obs["state_info"].isin(MATURE)]
cnt = late.groupby(["clone", "state_info"], observed=True).size().unstack(
    fill_value=0)
dom = {}
for c, row in cnt.iterrows():
    best, bestn = None, -1
    for s in MATURE:
        n = row.get(s, 0)
        if n > bestn:
            best, bestn = s, n
    if bestn > 0:
        dom[c] = best
e = obs[obs["time"] == 2].copy()
e["dom"] = e["clone"].map(dom)
e["ufrac"] = (u_c / (u_c + s_c))[(obs["time"] == 2).values]
e["nsis"] = e["clone"].map(late.groupby("clone").size())
sel = e[(e["nsis"] >= 3) & e["dom"].isin(["Neutrophil", "Monocyte"])]
g = sel.groupby("dom")["ufrac"].agg(["size", "mean", "std", "median"])
p(g.to_string())
from scipy.stats import mannwhitneyu
a = sel[sel.dom == "Neutrophil"]["ufrac"].values
b = sel[sel.dom == "Monocyte"]["ufrac"].values
p("   Mann-Whitney p = %.3g; difference in means %.5f"
  % (mannwhitneyu(a, b).pvalue, a.mean() - b.mean()))

open(OUT, "w").write("\n".join(out) + "\n")
