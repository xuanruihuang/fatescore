"""Figure: where the score comes from (LARRY main task, 567 clones, clone-level AUROC).

Rows: the sorted population alone and a classifier on day-2 expression (total counts) trained and
tested on disjoint clones (main_table.csv, graph d2, arms A1_pop and A3_state_total; 10 splits,
interval as stored); the walk without direction with the graph built from day-2 cells only
(feat_d2/nm_cr_conn_*.npy, 18 settings) and from all cells (feat_all, 18 settings): dot at the
reference setting n30 p30 d1, bar over the settings; every method in calibrated_table.csv (99) as one
dot each.

Usage: 115_fig_source.py <main_table.csv | ev_d2 prefix> <feat_d2> <feat_all> <d2_labels.tsv> <calibrated_table.csv> <out.pdf>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MAIN, FD2, FALL, LAB, TAB, OUT = sys.argv[1:7]
plt.rcParams.update({
    "font.size": 7.2, "axes.linewidth": 0.5, "pdf.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"]})
INK, MUTED, BAND = "#2b2b29", "#8a8984", "#e4e3df"

# task clones as in 99
T = pd.read_csv(LAB, sep="\t")
m = (T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])
im = np.where(m.values)[0]
ucl, inv = np.unique(T.loc[m, "clone"].values, return_inverse=True)
y = (T.loc[m].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)


def names(fd):
    d = json.load(open(os.path.join(fd, "d2_names.json")))
    return d if isinstance(d, list) else d.get("names", d)


assert names(FD2) == names(FALL), "day-2 cell order differs between the two feature directories"


def auc(fd, key):
    M = np.load(os.path.join(fd, "nm_" + key + ".npy"))
    if M.ndim == 2 and M.shape[1] > 1:
        s = M[:, 0] + M[:, 1]
        M = np.where(s > 0, M[:, 0] / np.where(s > 0, s, 1), 0.5)
    sh = M.ravel()
    c = np.bincount(inv, weights=sh[im], minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))
    return roc_auc_score(y, c)


def kernel(fd):
    keys = sorted(os.path.basename(p)[3:-4] for p in glob.glob(os.path.join(fd, "nm_cr_conn_n*_p*_d*.npy")))
    vals = {k: auc(fd, k) for k in keys}
    return vals["cr_conn_n30_p30_d1"], min(vals.values()), max(vals.values()), len(vals)


if MAIN.endswith(".csv"):  # main_table.csv of the first pipeline (09_figures.py)
    MT = pd.read_csv(MAIN)
    MT = MT[MT.graph == "d2"].set_index("arm")
else:  # the same numbers straight from 03_evaluate.py (<prefix>_summary.csv, <prefix>_bootstrap.csv)
    MT = pd.read_csv(MAIN + "_summary.csv").set_index("arm").rename(columns={"auroc_mean": "auroc"})
    MT = MT.join(pd.read_csv(MAIN + "_bootstrap.csv").set_index("arm")[["lo", "hi"]])
k2 = kernel(FD2)
ka = kernel(FALL)
C = pd.read_csv(TAB)
meth = C[C.dataset == "LARRY"].auroc_clone.values

import figstyle as fs  # noqa: E402
fs.apply()
L = C[C.dataset == "LARRY"]
rows = [("sorted population only", ("pt", MT.loc["A1_pop", "auroc"], MT.loc["A1_pop", "lo"], MT.loc["A1_pop", "hi"])),
        ("day-2 expression classifier", ("pt", MT.loc["A3_state_total", "auroc"],
                                          MT.loc["A3_state_total", "lo"], MT.loc["A3_state_total", "hi"])),
        ("walk, day-2 cells only", ("band", *k2[:3])),
        ("walk, all cells", ("band", *ka[:3])),
        ("the %d methods" % len(L), ("strip", L))]
fig, ax = plt.subplots(figsize=(fs.COL, 1.85))
n = len(rows)
ax.axvline(0.5, color=fs.AXIS, lw=0.5, zorder=0)
ax.text(0.503, n - 0.45, "chance", fontsize=6.5, color=fs.MUTED, va="bottom")
for i, (lab, spec) in enumerate(rows):
    yy = n - 1 - i
    if spec[0] == "pt":
        _, v, lo, hi = spec
        fs.interval(ax, lo, hi, yy, fs.INK)
        fs.dot(ax, v, yy, fs.INK)
        ax.text(hi + 0.01, yy, "%.2f" % v, va="center", fontsize=6.5, color=fs.INK2)
    elif spec[0] == "band":
        _, v, lo, hi = spec
        ax.plot([lo, hi], [yy, yy], color=fs.BAND, lw=5, solid_capstyle="butt", zorder=1)
        fs.dot(ax, v, yy, fs.INK)
        ax.text(hi + 0.01, yy, "%.2f" % v, va="center", fontsize=6.5, color=fs.INK2)
    else:
        g = spec[1].sort_values("auroc_clone")
        jit = 0.2 * np.where(np.arange(len(g)) % 2 == 0, 1, -1) * (np.arange(len(g)) % 3 > 0)
        for (m, a), j in zip(zip(g.method, g.auroc_clone), jit):
            fs.dot(ax, a, yy + j, fs.colour(m), ms=3.6)
ax.set_yticks(range(n - 1, -1, -1))
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0.48, 0.93)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_ylim(-0.6, n - 0.2)
ax.set_xlabel("clone-level AUROC, LARRY")
fs.clean(ax, left=False)
fs.save(fig, OUT)
print("walk day-2 only: ref %.3f, settings %.3f-%.3f (%d); all cells: ref %.3f, %.3f-%.3f (%d); methods %.3f-%.3f"
      % (k2 + ka + (meth.min(), meth.max())))
print("pop %.3f, expression %.3f" % (MT.loc["A1_pop", "auroc"], MT.loc["A3_state_total", "auroc"]))
