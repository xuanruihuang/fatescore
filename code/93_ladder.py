"""The deconstruction ladder: every validation, scored for the full method and for
the method with its dynamics removed one component at a time.

Columns (the ladder):
  method              velocity (the best of steady-state, stochastic/dynamical and
                      veloVI on each data set) or the optimal-transport method
  shuffled unspliced  the velocity model with the cell's own unspliced counts
                      replaced by another cell's (each gene's gamma kept)
  degradation only    the velocity model without unspliced counts, -gamma * s
  no direction        the neighbour average (velocity rows), the connectivity
                      kernel (fate rows), or the best snapshot reference of the
                      held-out time point (independent coupling, either end day)

Rows: validations scored within the snapshot (agreement with clonal fate, in-cluster
coherence, cross-boundary direction), and validations against an independent
measurement of change (a held-out sampling day, velocity from labelled RNA, the
measured cell-cycle direction). Scores are the paper's: Pearson r with the clone's
observed neutrophil share; ICCoh on gene-space vectors; CBDir in the embedding,
averaged within transitions; the share of the gap between the independent coupling
and the batch-to-batch distance closed by the interpolation; mean cosine with the
labelling velocity (gene space) or with the measured cell-cycle direction
(fluorescence plane).

Writes out/ladder.csv and the figure fig_ladder.{pdf,png}.

Usage: 93_ladder.py <figure_dir>
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
R = "../out/"
MM = 1 / 25.4
plt.rcParams.update({
    "font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"]})
COLS = ["method", "shuffled unspliced", "degradation only", "no direction"]
VELS = ["steady", "dynamical", "velovi"]

rows = []


def add(group, validation, data, vals, note=""):
    rows.append(dict(group=group, validation=validation, data=data, note=note,
                     **{c: v for c, v in zip(COLS, vals)}))


# ---- agreement with clonal fate (LARRY), CellRank ladder and optimal transport ----
F = pd.read_csv(R + "ladder_fate.csv").set_index("predictor")["r_cell"]
U = pd.read_csv(R + "unsupervised_reference.csv").set_index("predictor")["r_cell"]
add("snapshot", "Clonal fate, CellRank", "LARRY",
    [max(F["nm_cr_lad_velocity_dyn"], F["nm_cr_lad_velocity_ss"]), F["nm_cr_lad_shuffled"],
     F["nm_cr_lad_degradation"], F["nm_cr_lad_connectivity"]])
ot = ["nm_snap_statot", "nm_snap_wot", "nm_snap_moscot", "nm_snap_prescient"]
conn = U[U.index.str.startswith("nm_cr_conn_")]
add("snapshot", "Clonal fate, optimal transport", "LARRY",
    [U[ot].max(), np.nan, np.nan, conn.loc["nm_cr_conn_n30_p30_d1"]])

# ---- coherence and cross-boundary direction (benchmark convention) ----
E = pd.read_csv(R + "direction_embedding_bench.csv")
E = E[E["space"] == "umap"].set_index("field")
B = pd.read_csv(R + "benchmark_direction_bench.csv")
B = B[B["space"] == "embedding"]
sets = [("LARRY", E)] + [(d, g.set_index("field")) for d, g in B.groupby("dataset", sort=False)]
for d, T in sets:
    add("snapshot", "In-cluster coherence", d,
        [T.loc[VELS, "iccoh_gene"].max(), T.loc["perm_control", "iccoh_gene"],
         T.loc["minus_gamma_s", "iccoh_gene"], T.loc["knn_mean", "iccoh_gene"]])
for d, T in sets:
    add("snapshot", "Cross-boundary direction", d,
        [T.loc[VELS, "cbdir_fwd_t"].max(), T.loc["perm_control", "cbdir_fwd_t"],
         T.loc["minus_gamma_s", "cbdir_fwd_t"], T.loc["knn_mean", "cbdir_fwd_t"]])

# ---- independent measurements ----
O = pd.read_csv(R + "ot_interpolation.csv")
full = O["full"].astype(str) == "True"
m = O[full].groupby("name")["distance"].mean()
# the batch-to-batch distance (two day-4 wells) is computed only on the covariate split
m["P"] = O.loc[O["name"] == "P", "distance"].mean()
gap = lambda x: (m["R"] - x) / (m["R"] - m["P"])  # noqa: E731
add("independent", "Held-out sampling day", "LARRY",
    [gap(m["I"]), np.nan, np.nan, max(gap(m["R"]), gap(m["F"]), gap(m["L"]))],
    note="method: Waddington-OT")


def lab(f, ref, space):
    L = pd.read_csv(R + f)
    L = L[(L["reference"] == ref) & (L["space"] == space)].set_index("field")["mean"]
    return [L[["steady_state", "stochastic", "dynamical"]].max(), L["shuffled_unspliced"],
            L["degradation_only"], L["neighbour_average"]]


add("independent", "Velocity from labelled RNA", "haematopoiesis", lab("labelling_hsc.csv", "labelling", "gene"))
add("independent", "Velocity from labelled RNA", "RPE1", lab("labelling_rpe1.csv", "labelling", "gene"))
add("independent", "Measured cell-cycle direction", "RPE1", lab("labelling_rpe1.csv", "cycle", "umap"))

T = pd.DataFrame(rows)
best_base = T[COLS[1:]].max(axis=1)
T["method_minus_best_baseline"] = T["method"] - best_base
T.to_csv(R + "ladder.csv", index=False)
print(T.round(3).to_string())

# the figure keeps clonal fate and the independent measurements; the direction rows
# (coherence, CBDir) stay in ladder.csv
T = T[~T["validation"].isin(["In-cluster coherence", "Cross-boundary direction"])].reset_index(drop=True)

# ---- figure ----
# the label on the right: above every rung without dynamics, by the paired tests
# where the paper has them (clonal fate: Extended Data Table 2, no method
# significantly above the kernel), otherwise by the point estimate (labelling and
# cell-cycle rows: every difference excluded zero)
override = {("Clonal fate, optimal transport", "LARRY"): "not separated"}
cmap = LinearSegmentedColormap.from_list("lad", ["#f4f3ef", "#9cc3e4", "#1f5f9e"])
lines, y = [], 0.0
for i, r in T.iterrows():
    if i == 0 or T.loc[i - 1, "group"] != r["group"]:
        if i > 0:
            y += 0.5
        lines.append(("group", r["group"], y)); y += 1.15
    if i == 0 or T.loc[i - 1, "validation"] != r["validation"]:
        lines.append(("validation", r["validation"], y)); y += 0.95
    lines.append(("data", i, y)); y += 1.0
H = y
fig = plt.figure(figsize=(160 * MM, (H + 2.5) * 3.6 * MM))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
x0 = 3.2  # first cell column
for kind, v, yy in lines:
    if kind == "group":
        ax.text(0.0, yy, {"snapshot": "Scored against clonal fate on LARRY",
                          "independent": "Scored against an independent measurement of change"}[v],
                ha="left", va="center", fontsize=6.8, color="#52514e", style="italic")
        continue
    if kind == "validation":
        ax.text(0.15, yy, v, ha="left", va="center", fontsize=6.4, fontweight="bold")
        continue
    r = T.loc[v]
    ax.text(0.45, yy, r["data"], ha="left", va="center", fontsize=6.2)
    vals = r[COLS].values.astype(float)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    for j in range(len(COLS)):
        xc = x0 + j
        if not np.isfinite(vals[j]):
            ax.text(xc, yy, "–", ha="center", va="center", color="#9a9a96", fontsize=6.2)
            continue
        c = (vals[j] - lo) / (hi - lo) if hi > lo else 1.0
        ax.add_patch(plt.Rectangle((xc - 0.47, yy - 0.44), 0.94, 0.88, color=cmap(c), lw=0))
        if vals[j] == hi:
            ax.add_patch(plt.Rectangle((xc - 0.47, yy - 0.44), 0.94, 0.88, fill=False,
                                       ec="#1a1a1a", lw=0.8))
        ax.text(xc, yy, "%.2f" % vals[j], ha="center", va="center", fontsize=6.2,
                color="white" if c > 0.62 else "#1a1a1a")
    sep = r["method"] > np.nanmax(vals[1:])
    lab_ = override.get((r["validation"], r["data"]),
                        "separates" if sep else "passed by a baseline")
    col = {"separates": "#1f5f9e", "not separated": "#7a7873"}.get(lab_, "#b04a1c")
    ax.text(x0 + len(COLS) - 0.35, yy, lab_, ha="left", va="center", fontsize=6.2, color=col)
heads = ["method", "shuffled\nunspliced", "degradation\nonly", "no\ndirection"]
for j, h in enumerate(heads):
    ax.text(x0 + j, -1.2, h, ha="center", va="bottom", fontsize=6.4, linespacing=1.05)
ax.annotate("", xy=(x0 + len(COLS) - 0.55, -2.55), xytext=(x0 + 0.45, -2.55),
            arrowprops=dict(arrowstyle="-|>", lw=0.5, color="#52514e"))
ax.text(x0 + 2.0, -2.75, "information about dynamics removed", ha="center", va="bottom",
        fontsize=6.0, color="#52514e")
ax.set_xlim(-0.05, x0 + len(COLS) + 1.35)
ax.set_ylim(H - 0.3, -3.4)
ax.axis("off")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "fig_ladder.%s" % ext), dpi=300, bbox_inches="tight")
print("wrote fig_ladder")
