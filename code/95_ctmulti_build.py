"""CellTag-multi haematopoiesis (Jindal et al. 2024) as a second clonal-fate data set.

Mouse haematopoietic progenitors were CellTag-barcoded, sampled on day 2.5 (state)
and day 5 (fate). The authors' clone table (clone_tables/hsc.rna&atac.r1&2_master_v2.csv
in github.com/morris-lab/CellTag-multi-2023) gives each barcoded cell its clone,
sampling day, assay and annotated cell type (cell_type2). This script joins the RNA
count matrices of GEO GSE216521 (GSM6681127 day 2.5; GSM6681128-35 day 5) to that
table, keeps the RNA cells of the table, and writes, in the layout the LARRY scripts
read:

  <data>/ctmulti/hsc_rna.h5ad    raw counts; obs time_info (2.5 / 5.0), state_info
                                 (cell_type2), clone
  <featdir>/hvg.json             2,000 highly variable genes (seurat flavour)
  <featdir>/d2_names.json        the day-2.5 cells
  <featdir>/cellrank_log.json    terminal states: the mature day-5 types
  <featdir>/labels.tsv           per day-2.5 cell: clone, number of mature day-5 sisters,
                                 their composition, dominant mature fate, monocyte share

Task (fixed before any method is run): for day-2.5 cells whose clone has at least three
mature day-5 cells, predict whether the clone's mature descendants are dominated by
monocytes or by another mature type; a method's score for a cell is its probability of
the monocyte state divided by its total probability of the mature states.

Usage: 95_ctmulti_build.py <tar> <master.csv> <outdir_data> <featdir>
"""
import json
import os
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
MATURE = ["Mono", "Neutro", "Ery/Meg", "Baso/Eos/Mast", "Lym"]
SAMPLES = {"GSM6681127": "d2_5"}
SAMPLES.update({"GSM66811%d" % (28 + i): "d5_%d" % (i + 1) for i in range(8)})

M = pd.read_csv(MASTER)
M = M[M["assay"] == "rna"].drop_duplicates("cell.bc").set_index("cell.bc")
print("master RNA cells", M.shape[0], M["day"].value_counts().to_dict())

parts = []
with tarfile.open(TAR) as tf:
    for m in tf.getmembers():
        gsm = m.name.split("_")[0]
        if gsm not in SAMPLES or "feature_bc_matrix" not in m.name:
            continue
        p = os.path.join(OUTD, m.name)
        if not os.path.exists(p):
            tf.extract(m, OUTD)
        A = sc.read_10x_h5(p)
        A.var_names_make_unique()
        bc = [b.rsplit("-", 1)[0] for b in A.obs_names]
        if SAMPLES[gsm] == "d2_5":
            cands = {"d2_5": ["d2_5-rna-%s-1" % b for b in bc]}
        else:
            # the clone table names day-5 cells by replicate and library index
            # (d5r<r>-rna-<barcode>-<k>); find the (r, k) whose names overlap this file
            cands = {"d5r%d_%d" % (r, k): ["d5r%d-rna-%s-%d" % (r, b, k) for b in bc]
                     for r in (1, 2) for k in (1, 2, 3, 4, 5)}
        hits = {key: int(pd.Index(v).isin(M.index).sum()) for key, v in cands.items()}
        best = max(hits, key=hits.get)
        A.obs_names = cands[best]
        keep = A.obs_names.isin(M.index)
        print(m.name, A.shape, "matched as", best, "in clone table", int(keep.sum()),
              "(next best %d)" % sorted(hits.values())[-2] if len(hits) > 1 else "")
        parts.append(A[keep].copy())
X = ad.concat(parts, join="inner")
X.obs["clone"] = M.loc[X.obs_names, "clone.id"].astype(int).values
X.obs["day"] = np.where(M.loc[X.obs_names, "day"].values == "d2", 2.5, 5.0)
X.obs["time_info"] = X.obs["day"]
X.obs["state_info"] = M.loc[X.obs_names, "cell_type2"].astype(str).values
X.X = sp.csr_matrix(X.X).astype(np.float32)
sc.pp.filter_genes(X, min_cells=10)
X.write_h5ad(os.path.join(OUTD, "hsc_rna.h5ad"), compression="gzip")
print("object", X.shape, X.obs["day"].value_counts().to_dict())

B = X.copy()
sc.pp.normalize_total(B)
sc.pp.log1p(B)
sc.pp.highly_variable_genes(B, n_top_genes=2000, flavor="seurat")
json.dump(B.var_names[B.var["highly_variable"].values].tolist(), open(os.path.join(FD, "hvg.json"), "w"))
d2 = X.obs_names[X.obs["day"] == 2.5].tolist()
json.dump(d2, open(os.path.join(FD, "d2_names.json"), "w"))
d5 = X.obs[X.obs["day"] == 5.0]
term = {s: int((d5["state_info"] == s).sum()) for s in MATURE}
json.dump({"terminal_states": term}, open(os.path.join(FD, "cellrank_log.json"), "w"), indent=1)
print("terminal states", term)

comp = (d5[d5["state_info"].isin(MATURE)].groupby(["clone", "state_info"]).size()
        .unstack(fill_value=0).reindex(columns=MATURE, fill_value=0))
rows = []
for n in d2:
    c = int(X.obs.at[n, "clone"])
    v = comp.loc[c] if c in comp.index else pd.Series(0, index=MATURE)
    tot = int(v.sum())
    rows.append(dict(cell=n, clone=c, n_mature=tot, **{"n_" + k.replace("/", "_"): int(v[k]) for k in MATURE},
                     dom=(v.idxmax() if tot > 0 else ""), f_mono=(v["Mono"] / tot if tot > 0 else np.nan)))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(FD, "labels.tsv"), sep="\t", index=False)
task = T[T["n_mature"] >= 3]
cl = task.groupby("clone").agg(dom=("dom", "first"))
print("task: %d day-2.5 cells, %d clones; clones by dominant mature fate %s"
      % (len(task), len(cl), cl["dom"].value_counts().to_dict()))
print("binary: monocyte-dominated %d clones, other %d clones"
      % (int((cl["dom"] == "Mono").sum()), int((cl["dom"] != "Mono").sum())))
