"""Which later cells would leak a held-out clone's own fate into a snapshot method.

StationaryOT, Waddington-OT, moscot and the all-cell graph couple every cell of
the experiment, including the day-4 and day-6 descendants of the clones being
scored; those descendants are the cells that define the clone's label. The
methods never see the barcodes, so this is not label leakage in the usual
sense, but a day-2 cell's transport mass can land on its own sisters, which
resemble it, and carry their fate back. For the null results this only helps
the methods; for a positive increment it is the first alternative to rule out.

For each fold of repeat `--seed` (the same folds as cvlib.folds_of on the clones
of the main task, which 56_stacked_increment.py uses), this writes the names of
all cells sampled after day 2 whose barcode belongs to a test clone of that
fold. 41_ot_fate.py --drop-cells then removes them from the coupling (after the
PCA, so the representation is unchanged), and 56_stacked_increment.py
--per-fold scores fold f with the features computed without fold f's cells.

Usage: 59_leakage_cells.py <h5ad> <d2_labels.tsv> <out_prefix> [--seed 0]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvlib import folds_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("labels")
ap.add_argument("out_prefix")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--min-sis", type=int, default=3)
a = ap.parse_args()

T = pd.read_csv(a.labels, sep="\t")
sel = (T["n_sis"] >= a.min_sis) & T["dom"].isin(["Neutrophil", "Monocyte"])
groups = T.loc[sel, "clone"].values

A = ad.read_h5ad(a.h5ad, backed="r")
C = A.obsm["X_clone"]
C = C if sp.issparse(C) else sp.csr_matrix(C)
C = sp.csr_matrix(C)
has = np.asarray(C.sum(1)).ravel() > 0
clone = np.asarray(C.argmax(1)).ravel()
day = pd.to_numeric(A.obs["time_info"], errors="coerce").to_numpy()
names = np.asarray(A.obs_names)

# the label table's clone ids must be the h5ad's own
pos = pd.Series(np.arange(names.size), index=names)
chk = T[["cell", "clone"]].copy()
chk["h5"] = clone[pos[chk["cell"]].to_numpy()]
assert (chk["h5"] == chk["clone"]).all(), "clone ids disagree with the h5ad"

for f, (tr, te) in enumerate(folds_of(groups, a.seed, a.folds)):
    test_clones = set(np.unique(groups[te]))
    drop = has & (day > 2) & np.isin(clone, list(test_clones))
    out = "%s_s%d_f%d.txt" % (a.out_prefix, a.seed, f)
    with open(out, "w") as fh:
        fh.write("\n".join(names[drop]) + "\n")
    print("fold %d: %d test clones, %d later cells dropped -> %s"
          % (f, len(test_clones), int(drop.sum()), out), flush=True)
