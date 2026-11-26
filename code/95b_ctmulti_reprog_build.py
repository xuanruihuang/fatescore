"""CellTag-multi direct reprogramming (Jindal et al. 2024) as a third clonal-fate data set.

Mouse fibroblasts reprogrammed towards induced endoderm progenitors were CellTag-
barcoded and sampled on day 3 (state) and days 12 and 21 (fate). The authors' clone
table (clone_tables/b4.rna&atac.r1&2_master_rev1_annotated.csv in
github.com/morris-lab/CellTag-multi-2023) gives each cell its clone, day, assay, cell
type (early / reprogramming / transition / dead-end) and the clone's most dominant
fate. This script joins the RNA matrices of GEO GSE216521 (B4D3, B4D12, B4D21 RNA
libraries) to the RNA cells of that table and writes, as 95_ctmulti_build.py does:

  <data>/ctmulti/reprog_rna.h5ad, <featdir>/{hvg.json, d2_names.json (day-3 cells),
  cellrank_log.json (terminal states: the fate cell types on days 12 and 21), labels.tsv}

Task (fixed before any method is run): day-3 cells whose clone has at least three RNA
cells on days 12/21; predict whether the clone's most dominant fate is reprogramming
(as annotated by the authors) or not (transition or dead-end); a method's score is its
probability of the reprogramming state divided by its total terminal probability.

Usage: 95b_ctmulti_reprog_build.py <tar> <b4_master.csv> <outdir_data> <featdir>
"""
import json
import os
import re
import sys
import tarfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

TAR, MASTER, OUTD, FD = sys.argv[1:5]
os.makedirs(OUTD, exist_ok=True)
os.makedirs(FD, exist_ok=True)
FATES = ["reprogramming", "transition", "dead-end"]

M = pd.read_csv(MASTER)
M = M[M["assay"] == "RNA"].drop_duplicates("cell.bc").set_index("cell.bc")
print("master RNA cells", M.shape[0], M["day"].value_counts().to_dict())

parts = []
with tarfile.open(TAR) as tf:
    for m in tf.getmembers():
        mm = re.match(r"GSM\d+_(B4D\d+-RNA-r\d-\d)_filtered_feature_bc_matrix\.h5$", m.name)
        if not mm:
            continue
        p = os.path.join(OUTD, m.name)
        if not os.path.exists(p):
            tf.extract(m, OUTD)
        A = sc.read_10x_h5(p)
        A.var_names_make_unique()
        A.obs_names = ["%s-%s" % (mm.group(1), b) for b in A.obs_names]
        keep = A.obs_names.isin(M.index)
        print(m.name, A.shape, "in clone table", int(keep.sum()))
        parts.append(A[keep].copy())
X = ad.concat(parts, join="inner")
X.obs["clone"] = M.loc[X.obs_names, "clone_id"].astype(str).values
X.obs["day"] = M.loc[X.obs_names, "day"].str.replace("D", "").astype(float).values
X.obs["time_info"] = X.obs["day"]
X.obs["state_info"] = M.loc[X.obs_names, "cell_type"].astype(str).values
X.obs["clone_fate"] = M.loc[X.obs_names, "most_dominant_fate"].astype(str).values
X.X = sp.csr_matrix(X.X).astype(np.float32)
sc.pp.filter_genes(X, min_cells=10)
X.write_h5ad(os.path.join(OUTD, "reprog_rna.h5ad"), compression="gzip")
print("object", X.shape, X.obs["day"].value_counts().to_dict())

B = X.copy()
sc.pp.normalize_total(B)
sc.pp.log1p(B)
sc.pp.highly_variable_genes(B, n_top_genes=2000, flavor="seurat")
json.dump(B.var_names[B.var["highly_variable"].values].tolist(), open(os.path.join(FD, "hvg.json"), "w"))
early = X.obs_names[X.obs["day"] == 3.0].tolist()
json.dump(early, open(os.path.join(FD, "d2_names.json"), "w"))
late = X.obs[X.obs["day"] > 3.0]
term = {s: int((late["state_info"] == s).sum()) for s in FATES}
json.dump({"terminal_states": term}, open(os.path.join(FD, "cellrank_log.json"), "w"), indent=1)
print("terminal states", term)

nlate = late.groupby("clone").size()
rows = []
for n in early:
    c = X.obs.at[n, "clone"]
    k = int(nlate.get(c, 0))
    f = X.obs.at[n, "clone_fate"]
    rows.append(dict(cell=n, clone=c, n_mature=k, dom=f, f_reprogramming=float(f == "reprogramming")))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(FD, "labels.tsv"), sep="\t", index=False)
task = T[T["n_mature"] >= 3]
cl = task.groupby("clone")["dom"].first()
print("task: %d day-3 cells, %d clones; by dominant fate %s"
      % (len(task), len(cl), cl.value_counts().to_dict()))
