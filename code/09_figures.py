"""Figures and the main number table.

Usage: 09_figures.py <outdir> <figdir> [prefix]

Reads only the result files written by scripts 02b-15; nothing is recomputed,
so every number in a figure traces to a csv in <outdir>. The optional prefix
selects a dataset ("" for LARRY, "ct" for the CellTag time course).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT, FIG = sys.argv[1], sys.argv[2]
PFX = sys.argv[3] if len(sys.argv) > 3 else ""
LAB = os.path.join(OUT, ("ct_labels.tsv" if PFX else "d2_labels.tsv"))
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2,
    "ytick.major.size": 2, "pdf.fonttype": 42, "font.family": "sans-serif",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
MM = 1 / 25.4
GREY, DARK, BLUE, ORANGE = "#9e9e9e", "#303030", "#1f6fb4", "#d1670a"
POSNAME = "iEP" if PFX else "neutrophil"
NEGNAME = "fibroblast" if PFX else "monocyte"


def letter(ax, s, dx=-0.16, dy=1.08):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="top", ha="left")


def read(name):
    p = os.path.join(OUT, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def P(stem):
    return ("%s%s" % (PFX, stem)) if PFX else stem


LABELS = {
    "A0_prior": "majority class",
    "A1_pop": "sorting gate only" if not PFX else "time course only",
    "A2_state_spliced": "spliced counts",
    "A3_state_total": "total counts",
    "A4_state_smoothed": "spliced counts, graph-smoothed",
    "A4b_state_smoothed_10pc": "graph-smoothed, 10 components",
    "A5_unspliced_only": "unspliced counts only",
    "A6_state_plus_unspliced": "spliced + unspliced counts",
    "A7_smoothed_plus_ratio": "smoothed + splicing ratio",
    "N2_state_plus_depthmatched": "spliced + thinned copy of itself",
    "V1_smoothed_plus_vel_det": "smoothed + steady-state velocity",
    "V2_smoothed_plus_vel_dyn": "smoothed + dynamical velocity",
    "V2b_smoothed_plus_vel_dyn_10pc": "smoothed + dynamical velocity, 10 comp.",
    "V3_smoothed_plus_vel_velovi": "smoothed + veloVI velocity",
    "V4_smoothed_plus_cr_velo": "smoothed + CellRank velocity kernel",
    "V5_smoothed_plus_cr_pseudo": "smoothed + CellRank pseudotime kernel",
    "V4a_cr_velo_only": "CellRank velocity kernel alone",
    "V5a_cr_pseudo_only": "CellRank pseudotime kernel alone",
    "V1a_vel_det_only": "steady-state velocity alone",
    "V2a_vel_dyn_only": "dynamical velocity alone",
    "V3a_vel_velovi_only": "veloVI velocity alone",
}
ORDER = ["A1_pop", "A2_state_spliced", "A3_state_total", "A4_state_smoothed",
         "A5_unspliced_only", "A6_state_plus_unspliced",
         "A7_smoothed_plus_ratio", "V1_smoothed_plus_vel_det",
         "V2_smoothed_plus_vel_dyn", "V3_smoothed_plus_vel_velovi",
         "V4_smoothed_plus_cr_velo", "V5_smoothed_plus_cr_pseudo",
         "V1a_vel_det_only", "V2a_vel_dyn_only", "V3a_vel_velovi_only",
         "V4a_cr_velo_only", "V5a_cr_pseudo_only"]
FAMILY = {a: ("trivial" if a.startswith("A1") else
              "state" if a.startswith(("A2", "A3", "A4")) else
              "splicing" if a.startswith(("A5", "A6", "A7")) else "velocity")
          for a in LABELS}
FCOL = {"trivial": GREY, "state": BLUE, "splicing": ORANGE, "velocity": DARK}

# --------------------------------------------------------------- figure 1
T = pd.read_csv(LAB, sep="\t")
cl = (T.groupby("clone")
      .agg(n_neu_w1=("n_neu_w1", "first"), n_mono_w1=("n_mono_w1", "first"),
           n_neu_w2=("n_neu_w2", "first"), n_mono_w2=("n_mono_w2", "first"),
           n_sis=("n_sis", "first")))
cl["nm1"] = cl.n_neu_w1 + cl.n_mono_w1
cl["nm2"] = cl.n_neu_w2 + cl.n_mono_w2
both = cl[(cl.nm1 >= 1) & (cl.nm2 >= 1)]
f1 = (both.n_neu_w1 / both.nm1).values
f2 = (both.n_neu_w2 / both.nm2).values
agree = float(((f1 > 0.5) == (f2 > 0.5)).mean())

fig = plt.figure(figsize=(183 * MM, 54 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 0.95], wspace=0.45)

ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.6)
ax.axis("off")
letter(ax, "a", dx=-0.02, dy=1.02)
if PFX:
    boxes = [(0.2, 2.9, "days 3-9\n%d cells" % int((T["n_sis"] >= 0).sum()),
              BLUE),
             (5.1, 4.7, "day 21", GREY), (5.1, 1.2, "day 28", GREY)]
else:
    boxes = [(0.2, 2.9, "day 2, well 0\n4,638 cells", BLUE),
             (5.1, 4.7, "days 4 and 6, well 1\n19,059 cells", GREY),
             (5.1, 1.2, "days 4 and 6, well 2\n25,605 cells", GREY)]
for x, y, txt, col in boxes:
    ax.add_patch(plt.Rectangle((x, y), 3.5, 1.5, fill=False, lw=0.7,
                               edgecolor=col))
    ax.text(x + 1.75, y + 0.75, txt, ha="center", va="center", fontsize=5.5)
ax.annotate("", xy=(5.0, 5.2), xytext=(3.8, 3.9),
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color=DARK))
ax.annotate("", xy=(5.0, 2.1), xytext=(3.8, 3.5),
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color=DARK))
ax.text(4.4, 3.75, "same clonal barcode", fontsize=5, ha="center",
        va="bottom", color=DARK, rotation=0)
ax.text(1.95, 2.5, "predict from one cell", fontsize=5.5, color=BLUE,
        ha="center", va="top")
ax.text(6.85, 0.9, "two independent read-outs\nof the same clone",
        fontsize=5.5, color=DARK, ha="center", va="top")

ax = fig.add_subplot(gs[0, 1])
letter(ax, "b")
rng = np.random.default_rng(0)
j = 0.018
ax.scatter(f1 + rng.normal(0, j, len(f1)), f2 + rng.normal(0, j, len(f2)),
           s=2.4, c=DARK, alpha=0.3, linewidths=0)
ax.plot([0, 1], [0, 1], lw=0.5, color=GREY, ls="--")
ax.set_xlabel("%s fraction, read-out 1" % POSNAME)
ax.set_ylabel("%s fraction, read-out 2" % POSNAME)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.set_title("%d clones, %.0f%% agree on the majority"
             % (len(both), 100 * agree), pad=2)

ax = fig.add_subplot(gs[0, 2])
letter(ax, "c")
nsis = T[T.n_sis > 0].groupby("clone")["n_sis"].first()
ax.hist(np.clip(nsis, 0, 60), bins=np.arange(0, 62, 2), color=GREY, lw=0)
ax.axvline(3, color=DARK, lw=0.6)
ax.set_xlabel("fated clonal sisters")
ax.set_ylabel("clones")
fig.savefig(os.path.join(FIG, P("fig1_design.pdf")))
plt.close(fig)


# --------------------------------------------------------------- figure 2
def arm_frame(mode):
    s = read(P("ev_%s_summary.csv" % mode))
    b = read(P("ev_%s_bootstrap.csv" % mode))
    if s is None:
        return None
    s = s.rename(columns={s.columns[0]: "arm"})
    if b is not None:
        s = s.merge(b, on="arm", how="left")
    s = s[s["arm"].isin(ORDER)]
    s["label"] = s["arm"].map(LABELS)
    s["family"] = s["arm"].map(FAMILY)
    s["rank"] = s["arm"].apply(ORDER.index)
    return s.sort_values("rank")


ceil, ceil_lo, ceil_hi = {}, {}, {}
for mode in ("d2", "all"):
    c = read(P("st_%s_ceiling_mean.csv" % mode))
    if c is not None:
        c = c.rename(columns={c.columns[0]: "who"})
        row = c[c.who == "sister_oracle"]
        if len(row):
            ceil[mode] = float(row["auroc"].iloc[0])

frames = {m: arm_frame(m) for m in ("d2", "all")}
avail = [m for m in ("d2", "all") if frames[m] is not None]
if avail:
    fig, axes = plt.subplots(1, len(avail), figsize=(183 * MM, 72 * MM),
                             sharey=True)
    axes = np.atleast_1d(axes)
    title = {"d2": "graph built on the early cells only",
             "all": "graph built on all cells, the standard workflow"}
    for k, (mode, ax) in enumerate(zip(avail, axes)):
        S = frames[mode]
        yy = np.arange(len(S))[::-1]
        if "lo" in S:
            ax.errorbar(S["auroc_mean"], yy,
                        xerr=[S["auroc_mean"] - S["lo"],
                              S["hi"] - S["auroc_mean"]],
                        fmt="none", ecolor=GREY, elinewidth=0.5, capsize=0)
        ax.scatter(S["auroc_mean"], yy, s=10,
                   c=[FCOL[f] for f in S["family"]], zorder=3, linewidths=0)
        if mode in ceil:
            ax.axvline(ceil[mode], color=DARK, lw=0.7)
            ax.text(ceil[mode] - 0.002, len(S) - 0.6,
                    "precision of the label ", fontsize=5, color=DARK,
                    va="top", ha="right")
        ax.axvline(0.5, color=GREY, lw=0.5, ls=":")
        ax.set_yticks(yy)
        if k == 0:
            ax.set_yticklabels(S["label"])
        ax.set_xlabel("clone-level AUROC, %s versus %s" % (POSNAME, NEGNAME))
        ax.set_title(title[mode], pad=3)
        ax.set_xlim(0.47, max(0.9, ceil.get(mode, 0.87) + 0.04))
    h = [Line2D([], [], marker="o", ls="", color=FCOL[f], ms=3,
                label={"trivial": "no transcriptome", "state": "abundance",
                       "splicing": "splicing layer",
                       "velocity": "velocity"}[f])
         for f in ("trivial", "state", "splicing", "velocity")]
    axes[0].legend(handles=h, loc="lower right", frameon=False,
                   handletextpad=0.3)
    fig.savefig(os.path.join(FIG, P("fig2_arms.pdf")))
    plt.close(fig)

# --------------------------------------------------------------- figure 3
for mode in ("d2", "all"):
    D = read(P("dn_%s_summary.csv" % mode))
    RS = read(P("rs_%s.csv" % mode))
    if D is None:
        continue
    dep = read(P("dn_%s_depths.csv" % mode))
    med_spliced = (float(dep["median_spliced_on_selected_genes"].iloc[0])
                   if dep is not None else np.nan)
    med_unspliced = (float(dep["median_unspliced_on_selected_genes"].iloc[0])
                     if dep is not None else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 56 * MM),
                             gridspec_kw=dict(width_ratios=[1, 1.15],
                                              wspace=0.62))
    fr = [1.0, 0.5, 0.25, 0.125, 0.0625]
    xs, ys, es = [], [], []
    for f in fr:
        r = D[D.family == "D_frac%g" % f]
        if len(r):
            xs.append((med_spliced if np.isfinite(med_spliced) else 1.0) * f)
            ys.append(float(r.auroc_mean.iloc[0]))
            sd = float(r.auroc_sd.iloc[0])
            es.append(sd if np.isfinite(sd) else 0.0)
    ax = axes[0]
    letter(ax, "a")
    ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=3, lw=0.8, color=BLUE,
                ecolor=GREY, elinewidth=0.5, capsize=0,
                label="spliced counts, thinned")
    uns = None
    r = D[D.family == "A5_unspliced_only"]
    if len(r):
        uns = float(r.auroc_mean.iloc[0])
    else:
        ev = read(P("ev_%s_summary.csv" % mode))
        if ev is not None:
            ev = ev.rename(columns={ev.columns[0]: "arm"})
            q = ev[ev.arm == "A5_unspliced_only"]
            if len(q):
                uns = float(q.auroc_mean.iloc[0])
    if uns is not None and np.isfinite(med_unspliced):
        ax.scatter([med_unspliced], [uns], s=20, marker="D", color=ORANGE,
                   zorder=4, linewidths=0, label="unspliced counts")
    ax.set_xscale("log")
    if np.isfinite(med_spliced):
        ticks = [med_spliced * f for f in (1 / 16, 1 / 4, 1)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["%.0f" % t for t in ticks])
        ax.minorticks_off()
    ax.set_xlabel("molecules per cell on the selected genes (median)")
    ax.set_ylabel("clone-level AUROC")
    ax.set_title("what a layer is worth at its own depth", pad=2)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3)

    ax = axes[1]
    letter(ax, "b", dx=-0.1)
    if RS is not None:
        want = [
            ("independent equal-depth spliced copy added",
             "equal-depth spliced copy", BLUE),
            ("unspliced layer added in its place",
             "unspliced reads instead", ORANGE),
            ("unspliced minus independent spliced copy",
             "difference", DARK),
            ("dynamical velocity added to the smoothed state",
             "dynamical velocity", DARK),
            ("velocity shuffled within a state",
             "velocity, shuffled", GREY),
        ]
        rows = [(lab, col, RS[RS.quantity == q].iloc[0])
                for q, lab, col in want if (RS.quantity == q).any()]
        yy = np.arange(len(rows))[::-1]
        for (lab, col, r), y in zip(rows, yy):
            ax.errorbar([r["value"]], [y],
                        xerr=[[r["value"] - r["lo"]], [r["hi"] - r["value"]]],
                        fmt="o", ms=3.5, color=col, ecolor=GREY,
                        elinewidth=0.6, capsize=0)
        ax.axvline(0, color=GREY, lw=0.5, ls=":")
        ax.set_yticks(yy)
        ax.set_yticklabels([lab for lab, _, _ in rows])
        ax.set_xlabel("change in clone-level AUROC")
        ax.set_title("adding a layer to a model that already has abundance",
                     pad=2)
    fig.savefig(os.path.join(FIG, P("fig3_mechanism_%s.pdf" % mode)))
    plt.close(fig)

# --------------------------------------------------------------- figure 4
for mode in ("d2", "all"):
    TA = read(P("ta_%s.csv" % mode))
    if TA is None:
        continue
    name = {"state_itself": "the cell's own expression state",
            "det": "steady-state velocity", "dyn": "dynamical velocity",
            "velovi": "veloVI velocity"}
    keep = [k for k in ("state_itself", "det", "dyn", "velovi")
            if (TA.vector == k).any()]
    axes_show = [("maturation", "toward the mature cells of the same day"),
                 ("clonal fate", "toward the clone's observed fate")]
    fig, axs = plt.subplots(1, 2, figsize=(183 * MM, 48 * MM), sharey=True)
    for ax, (aname, title) in zip(axs, axes_show):
        sub = TA[TA.axis == aname]
        yy = np.arange(len(keep))[::-1]
        for k, y in zip(keep, yy):
            r = sub[sub.vector == k]
            if not len(r):
                continue
            r = r.iloc[0]
            col = BLUE if k == "state_itself" else DARK
            ax.errorbar([r["direction_correct"]], [y],
                        xerr=[r["sd"] if np.isfinite(r["sd"]) else 0.0],
                        fmt="o", ms=3.5, color=col, ecolor=GREY,
                        elinewidth=0.6, capsize=0)
            rl = sub[sub.vector == k + "_logscale"]
            if len(rl):
                ax.scatter([float(rl["direction_correct"].iloc[0])], [y],
                           s=12, marker="x", color=GREY, zorder=2,
                           linewidths=0.7)
        ax.axvline(0.5, color=GREY, lw=0.6, ls=":")
        ax.set_xlim(0, 1)
        ax.set_yticks(yy)
        ax.set_yticklabels([name[k] for k in keep])
        ax.set_xlabel("fraction of cells pointing the right way")
        ax.set_title(title, pad=3)
        ax.set_ylim(-0.7, len(keep) - 0.3)
        ax.text(0.51, -0.6, "chance", fontsize=5, color=GREY, va="bottom",
                ha="left")
    fig.savefig(os.path.join(FIG, P("fig4_fateaxis_%s.pdf" % mode)))
    plt.close(fig)

# --------------------------------------------------------------- figure 5
for mode in ("d2", "all"):
    I = read(P("st_%s_increments.csv" % mode))
    if I is None:
        continue
    I = I[I.ref == "A4_state_smoothed"]
    keep = ["A7_smoothed_plus_ratio", "V1_smoothed_plus_vel_det",
            "V2_smoothed_plus_vel_dyn", "V3_smoothed_plus_vel_velovi",
            "V4_smoothed_plus_cr_velo", "V5_smoothed_plus_cr_pseudo"]
    I = I[I.arm.isin(keep)]
    axes_order = ["pop", "sis_bin", "sis_time", "margin_tercile"]
    I = I[I.axis.isin(axes_order)]
    if not len(I):
        continue
    Q = I.pivot_table(index=["axis", "level"], columns="arm",
                      values="increment")
    Q = Q.reindex(sorted(Q.index, key=lambda t: axes_order.index(t[0])))
    Q = Q[[c for c in keep if c in Q.columns]]
    fig, ax = plt.subplots(figsize=(120 * MM, (16 + 5.2 * len(Q)) * MM))
    vmax = max(0.02, float(np.nanmax(np.abs(Q.values))))
    im = ax.imshow(Q.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(Q.shape[1]))
    ax.set_xticklabels([LABELS.get(c, c) for c in Q.columns], rotation=35,
                       ha="right")
    ax.set_yticks(range(Q.shape[0]))
    ax.set_yticklabels(["%s" % b for a, b in Q.index])
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if np.isfinite(Q.values[i, j]):
                ax.text(j, i, "%+.3f" % Q.values[i, j], ha="center",
                        va="center", fontsize=4.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("AUROC minus the smoothed abundance model")
    cb.ax.tick_params(labelsize=5)
    fig.savefig(os.path.join(FIG, P("fig5_strata_%s.pdf" % mode)))
    plt.close(fig)

# ------------------------------------------------------------- main table
rows = []
for mode in ("d2", "all"):
    S = frames.get(mode)
    if S is None:
        continue
    for _, r in S.iterrows():
        rows.append(dict(graph=mode, arm=r["arm"], label=r["label"],
                         auroc=r["auroc_mean"], auroc_sd=r["auroc_sd"],
                         acc=r["acc_mean"], lo=r.get("lo"), hi=r.get("hi"),
                         n_clones=r["n_clones"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, P("main_table.csv")), index=False)
print("figures written to", FIG)
print("sister-oracle AUROC:", ceil)
print("cross-read-out majority agreement on %d clones: %.3f"
      % (len(both), agree))
