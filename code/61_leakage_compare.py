"""Paired comparison for the leakage test: a snapshot method's arm with and
without the held-out clones' later cells in the coupling (59_leakage_cells.py,
41_ot_fate.py --drop-cells, 56_stacked_increment.py --per-fold).

Reads <prefix>_oof.npz from a 56 run on repeat 0 that has both arms, and
writes <prefix>_leakdiff.csv: for the first stage alone (S1_) and the stacked
score (ST_), the clone-level AUROC with and without the exclusion and their
paired difference with a 2,000-resample clone bootstrap (seed 13, as 56).

Usage: 61_leakage_compare.py <prefix> [--arm snap_statot] [--lo snap_statot_lo]
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--arm", default="snap_statot")
ap.add_argument("--lo", default="snap_statot_lo")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "s56", os.path.join(here, "56_stacked_increment.py"))
s56 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s56)

Z = np.load(a.prefix + "_oof.npz", allow_pickle=True)
keys = list(Z["keys"])
y, g = Z["y"], Z["groups"]


def tab(k):
    return s56.clone_level(Z["oof"][keys.index(k), 0], y, g)


rows = []
rng = np.random.default_rng(13)
first = tab("ST_base")
clones = first.index.values
yv = first["y"].to_numpy().astype(bool)
idx = rng.integers(0, clones.size, size=(2000, clones.size))
for stage in ("S1_", "ST_"):
    A = tab(stage + a.arm).reindex(clones)["p"].to_numpy()
    B = tab(stage + a.lo).reindex(clones)["p"].to_numpy()
    d = s56.auroc_rows(B[idx], yv[idx]) - s56.auroc_rows(A[idx], yv[idx])
    d = d[np.isfinite(d)]
    q = np.percentile(d, [2.5, 97.5])
    rows.append(dict(stage=stage.rstrip("_"), auroc_with=roc_auc_score(yv, A),
                     auroc_without=roc_auc_score(yv, B),
                     diff=roc_auc_score(yv, B) - roc_auc_score(yv, A),
                     lo=q[0], hi=q[1]))
R = pd.DataFrame(rows).round(4)
R.to_csv(a.prefix + "_leakdiff.csv", index=False)
print(R.to_string(index=False))
