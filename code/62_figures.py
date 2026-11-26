"""Main figures, built from the result files only (no number is typed in).

Usage: 62_figures.py <outdir>   (run from code/; reads ../out/*, writes
<outdir>/fig{1..5}.{pdf,png})

One colour means one thing in every figure:
  expression references  dark grey   #3b3b3b
  RNA velocity family     blue        #2a78d6
  snapshot OT / potential orange      #eb6834
  dynamics-free controls  aqua        #1baf7a
  synthetic calibration   mid grey    #9a9a96
Type is 6-7 pt at 183 mm width; panel letters 8 pt bold.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
R = "../out/"
EXPR, VEL, OT, CTRL, SYN = "#3b3b3b", "#2a78d6", "#eb6834", "#1baf7a", "#9a9a96"
MM = 1 / 25.4
plt.rcParams.update({
    "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})


def letter(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="right", va="bottom")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


def hdot(ax, y, x, lo, hi, color, hollow=False, size=3.2):
    ax.plot([lo, hi], [y, y], color=color, lw=0.9, solid_capstyle="round")
    ax.plot([x], [y], "o", ms=size, mfc="white" if hollow else color,
            mec=color, mew=0.9)


# ---------------------------------------------------------------- shared panels
def design(ax):
    """One barcoded clone on day 2: one cell is sequenced, its sisters are split
    between two wells, and each well's mature descendants on days 4 and 6 give an
    observation of the clone's fate."""
    from matplotlib.patches import Circle, FancyBboxPatch
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
    grey, neu, mono = "#bdbbb6", "#cf5a2a", "#2f6aa3"

    def cell(x, y, c, r=0.32, ec=None, lw=0.6):
        ax.add_patch(Circle((x, y), r, fc=c, ec=ec or c, lw=lw))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#6b6b68",
                                    mutation_scale=7, shrinkA=0, shrinkB=0))
    for x, y in ((1.75, 4.55), (2.45, 4.0), (1.75, 3.45), (2.45, 5.1), (2.45, 2.9)):
        cell(x, y, grey)
    cell(0.6, 4.0, "white", ec="#2b2b2b", lw=0.9)
    ax.text(0.6, 3.35, "sequenced\non day 2", ha="center", va="top", fontsize=5.4)
    ax.text(2.1, 5.75, "one clone", ha="center", va="bottom", fontsize=5.4)
    rng = np.random.default_rng(3)
    for k, (yc, frac) in enumerate(((6.15, 0.7), (1.85, 0.6))):
        ax.add_patch(FancyBboxPatch((4.3, yc - 1.05), 3.6, 2.1, boxstyle="round,pad=0.12",
                                    fc="#f4f4f2", ec="#9a9a96", lw=0.5))
        for i in range(12):
            x = 4.7 + (i % 6) * 0.56; y = yc - 0.45 + (i // 6) * 0.9
            cell(x, y, neu if rng.random() < frac else mono, r=0.2)
        ax.text(6.1, yc + 1.2 if k == 0 else yc - 1.25, "well %d, days 4 and 6" % (k + 1),
                ha="center", va="bottom" if k == 0 else "top", fontsize=5.4, color="#52514e")
        arrow(2.9, 4.6 if k == 0 else 3.4, 4.2, yc)
        arrow(8.05, yc, 8.95, 4.0 + (0.35 if k == 0 else -0.35))
    ax.text(10.4, 4.0, "clonal fate:\nshare of\nneutrophils", ha="center", va="center", fontsize=5.6)
    ax.set_title("A clone's fate, observed twice", loc="left", fontsize=7)


# ---------------------------------------------------------------- figure 1
def fig1():
    """Agreement with clonal fate, shown on the data. Top: design and the
    correlation of each predictor with observed fate bias. Bottom: the day-2
    cells on the SPRING layout, coloured by the rank of their clone's observed
    neutrophil fate bias and of the fate bias predicted by the connectivity
    kernel, Waddington-OT and CellRank's velocity kernel (ranks, because the
    correlation compares patterns, not scales)."""
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import rankdata
    S = pd.read_csv(R + "fig_spring.csv", index_col=0)
    U = pd.read_csv(R + "unsupervised_reference.csv").set_index("predictor")
    D2 = pd.read_csv(R + "refint_d2_pairs.csv").drop_duplicates("reference").set_index("reference")
    CR = pd.read_csv(R + "cell_r_continuous.csv").set_index("predictor")
    cmap = LinearSegmentedColormap.from_list("fate", ["#2f6aa3", "#e9e7e2", "#cf5a2a"])
    D = S[S["observed"].notna()]
    fig = plt.figure(figsize=(183 * MM, 136 * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.0], hspace=0.34)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.2, 1.25], wspace=0.5)
    ax = fig.add_subplot(top[0]); letter(ax, "a"); design(ax)
    # b: correlation with observed fate bias, sorted
    ax = fig.add_subplot(top[1]); letter(ax, "b")
    meth = [("Waddington-OT", "nm_snap_wot", OT), ("Real-time kernel", "nm_snap_rtk_ub", OT),
            ("moscot", "nm_snap_moscot", OT), ("PRESCIENT", "nm_snap_prescient", OT),
            ("StationaryOT", "nm_snap_statot", OT),
            ("Velocity + connectivity kernel", "nm_cr_vkck", VEL),
            ("Velocity kernel", "nm_cr_velo", VEL), ("Pseudotime kernel", "nm_cr_pseudo", VEL)]
    meth = sorted(meth, key=lambda m: -U.loc[m[1], "r_cell"])
    conn = U.loc[U.index.str.startswith("nm_cr_conn_"), "r_cell"]
    c2 = D2.loc[D2.index.str.contains("conn"), "r_ref"]
    ax.axvspan(conn.min(), conn.max(), color=CTRL, alpha=0.3, lw=0)
    ax.axvspan(c2.min(), c2.max(), color=CTRL, alpha=0.13, lw=0)
    for i, (n, k, c) in enumerate(meth):
        ax.barh(i, U.loc[k, "r_cell"], height=0.6, color=c, lw=0)
    ax.set_yticks(range(len(meth))); ax.set_yticklabels([m[0] for m in meth])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.7); ax.set_ylim(len(meth) - 0.4, -1.3)
    ax.set_xlabel("Correlation with observed neutrophil fate bias (r)")
    for x, t, c in ((c2.mean(), "kernel,\nday-2 cells", "#127a55"),
                    (conn.mean(), "connectivity\nkernel", "#127a55")):
        ax.text(x, -0.75, t, ha="center", va="bottom", fontsize=5.4, color=c)
    # c-f: maps
    bot = outer[1].subgridspec(1, 4, wspace=0.04)
    x0, x1 = np.percentile(D["x"], [0.5, 99.5]); y0, y1 = np.percentile(D["y"], [0.5, 99.5])
    pad = 0.12 * max(x1 - x0, y1 - y0)
    panels = [("observed", "Observed clonal fate"), ("connectivity", "Connectivity kernel"),
              ("wot", "Waddington-OT"), ("velocity_kernel", "CellRank velocity kernel")]
    for j, (k, title) in enumerate(panels):
        ax = fig.add_subplot(bot[j]); letter(ax, "cdef"[j])
        ax.scatter(S["x"], S["y"], s=0.5, c="#dddcd8", lw=0, rasterized=True)
        v = rankdata(D[k].values) / len(D)
        o = np.argsort(np.abs(v - 0.5))
        pts = ax.scatter(D["x"].values[o], D["y"].values[o], s=5, c=v[o], cmap=cmap,
                         vmin=0, vmax=1, lw=0, rasterized=True)
        ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, loc="left", fontsize=6.5)
    cax = fig.add_axes([0.39, 0.035, 0.22, 0.012])
    cb = fig.colorbar(pts, cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(["monocyte", "neutrophil"], fontsize=5.6)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0)
    cax.set_title("neutrophil fate bias (rank within panel)", fontsize=5.6, pad=2)
    save(fig, "fig1")


def ed11():
    """Agreement with clonal fate: methods against non-directional baselines,
    scored as published rankings score them, with paired intervals."""
    CR = pd.read_csv(R + "cell_r_continuous.csv").set_index("predictor")
    U = pd.read_csv(R + "unsupervised_reference.csv").set_index("predictor")
    D2 = pd.read_csv(R + "refint_d2_pairs.csv").drop_duplicates("reference").set_index("reference")
    P = pd.read_csv(R + "refint_invitro2_pairs.csv")
    P = P[P.reference == "nm_cr_conn_n30_p30_d1"].set_index("method")
    Q = pd.read_csv(R + "refint_vs_d2_pairs.csv")
    Q = Q[Q.reference == "nm_cr_conn_n30_p30_d1"].set_index("method")
    meth = [("Waddington-OT", "nm_snap_wot", OT), ("moscot", "nm_snap_moscot", OT),
            ("PRESCIENT", "nm_snap_prescient", OT), ("StationaryOT", "nm_snap_statot", OT),
            ("Real-time kernel (moscot)", "nm_snap_rtk_ub", OT),
            ("Velocity kernel", "nm_cr_velo", VEL),
            ("Pseudotime kernel", "nm_cr_pseudo", VEL),
            ("Velocity + connectivity kernel", "nm_cr_vkck", VEL)]
    conn = U.loc[U.index.str.startswith("nm_cr_conn_")]
    idw = U.loc[U.index.str.startswith("snap_idw_")]
    c2 = D2.loc[D2.index.str.contains("conn")]
    i2 = D2.loc[D2.index.str.contains("idw")]
    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.62, 1.0], height_ratios=[1.0, 0.72],
                          wspace=0.62, hspace=0.42)
    ax = fig.add_subplot(gs[0, 0]); letter(ax, "a"); design(ax)
    ax = fig.add_subplot(gs[0, 1]); letter(ax, "b")
    rows = [(n, "pt", U.loc[k, "r_cell"], c) for n, k, c in meth]
    rows += [("Connectivity kernel (18 settings)", "rng", conn["r_cell"].values, CTRL),
             ("Inverse-distance weighting (5)", "rng", idw["r_cell"].values, CTRL),
             ("Connectivity kernel, day-2 cells only", "rng", c2["r_ref"].values, CTRL),
             ("Inverse-distance, day-2 cells only", "rng", i2["r_ref"].values, CTRL)]
    for i, (n, kind, v, c) in enumerate(rows):
        if kind == "rng":
            ax.plot([v.min(), v.max()], [i, i], color=c, lw=3.0, solid_capstyle="butt",
                    alpha=1.0 if "only" not in n else 0.45)
            ax.plot(v, np.full(len(v), i), "|", color="#3b3b3b", ms=3.5, mew=0.6)
        else:
            ax.plot([v], [i], "o", ms=3.4, mfc=c, mec=c, mew=0.9)
    sup = [("Sorted population", "gate only"), ("+ day-2 spliced counts", "gate + expression (spliced)"),
           ("+ day-2 total counts", "gate + expression (total)")]
    base = len(rows) + 1.0
    ax.axhline(len(rows) - 0.3, color="#c8c8c4", lw=0.5)
    ax.text(0.715, len(rows) + 0.15, "fitted to clonal labels", ha="right",
            fontsize=5.5, color="#6b6b68", va="center")
    for i, (n, k) in enumerate(sup):
        v = CR.loc[k, "r_cell"]; sd = CR.loc[k, "r_cell_sd"]
        ax.plot([v - sd, v + sd], [base + i, base + i], color=EXPR, lw=2.0)
        ax.plot([v], [base + i], "o", ms=3.2, color="#9a9a96" if i == 0 else EXPR)
    ax.set_yticks(list(range(len(rows))) + [base + i for i in range(3)])
    ax.set_yticklabels([r[0] for r in rows] + [s[0] for s in sup])
    lo_r = min(0.0, float(np.nanmin(i2["r_ref"].values)) - 0.02)
    ax.set_ylim(base + 2.6, -0.7); ax.set_xlim(lo_r, 0.72)
    ax.set_xlabel("Pearson r with observed fate bias (868 day-2 cells)")
    ax.set_title("Correlation used in published rankings", loc="left", fontsize=7)
    sub = gs[1, :].subgridspec(1, 2, wspace=0.08)
    for j, (pre, xl, xlim) in enumerate((("r", "Method minus connectivity kernel, Pearson r",
                                          (-0.42, 0.32)),
                                         ("auroc", "Method minus connectivity kernel, AUROC",
                                          (-0.2, 0.09)))):
        ax = fig.add_subplot(sub[j]); letter(ax, "cd"[j])
        for i, (n, k, c) in enumerate(meth):
            r = P.loc[k]
            # minimum detectable difference (80% power): 1.43 x the margin of error
            mdd = 1.43 * (r[pre + "_hi"] - r[pre + "_lo"]) / 2
            ax.add_patch(plt.Rectangle((-mdd, i - 0.42), 2 * mdd, 0.84, color="#ececea",
                                       lw=0, zorder=0))
            hdot(ax, i - 0.15, r[pre + "_diff"], r[pre + "_lo"], r[pre + "_hi"], c, False)
            q = Q.loc[k]
            ax.plot([q[pre + "_lo"], q[pre + "_hi"]], [i + 0.2, i + 0.2], color=c, lw=0.7, alpha=0.5)
            ax.plot([q[pre + "_diff"]], [i + 0.2], "o", ms=3, mfc="white", mec=c, mew=0.8, alpha=0.8)
        ax.axvline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
        ax.set_yticks(range(len(meth)))
        ax.set_yticklabels([m[0] for m in meth] if j == 0 else [])
        ax.set_ylim(len(meth) - 0.45, -0.6); ax.set_xlim(*xlim)
        ax.set_xlabel(xl)
        ax.set_title(("Paired difference, r" if j == 0 else "Paired difference, AUROC"),
                     loc="left", fontsize=7)
    fig.text(0.5, 0.035, "Filled: connectivity kernel on all cells;  open: on day-2 cells only;  "
             "shaded: minimum detectable difference (80% power) against the all-cell kernel",
             ha="center", va="top", fontsize=5.6, color="#52514e")
    save(fig, "ed11")


# ---------------------------------------------------------------- figure 4 (state and increments)
def ed13():
    """What the measured state carries, what labels buy, and what the methods add.
    a: learning curve of the supervised expression baseline (with and without the
    sorted population) against the methods' direct AUROC, the noise ceilings and
    the extrapolations; b: stacked increments over total counts."""
    fig = plt.figure(figsize=(183 * MM, 92 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.8], wspace=0.95)
    ax = fig.add_subplot(gs[0]); letter(ax, "a")
    LC = pd.read_csv(R + "learning_curve_lowend_summary.csv")
    U = pd.read_csv(R + "unsupervised_reference.csv").set_index("predictor")
    C = pd.read_csv(R + "label_ceiling2.csv")
    two = C[C.fit == "two_level"].set_index("quantity")
    LP = pd.read_csv(R + "labelparam_summary.csv", index_col=0)
    LF = pd.read_csv(R + "learning_fit.csv")
    lf = LF[(LF.arm == "total")].set_index("form")["asymptote"]
    XINF = 540
    # noise ceilings (band over refits) and the gap
    cl, ch = LP.loc["ceiling_cell", "boot_lo"], LP.loc["ceiling_clone", "boot_hi"]
    ax.axhspan(cl, ch, color="#e6e6e3", lw=0, zorder=0)
    for q, lab in (("ceiling_main", "noise ceiling, clone propensity"),
                   ("ceiling_main_sister", "noise ceiling, cell's own propensity")):
        v = two.loc[q, "sim_mean"]
        ax.axhline(v, color=SYN, lw=0.6, zorder=1)
        ax.text(8, v + 0.003, lab, fontsize=5.2, color="#52514e", va="bottom")
    # connectivity kernel over its 18 settings
    conn = U.loc[U.index.str.startswith("nm_cr_conn_"), "auroc_clone"]
    ax.fill_between([0, 470], conn.min(), conn.max(), color=CTRL, alpha=0.22, lw=0, zorder=1)
    # methods' direct AUROC as short ticks at the right edge, labelled
    meth = [("Real-time kernel", "nm_snap_rtk_ub", OT), ("StationaryOT", "nm_snap_statot", OT),
            ("Velocity kernel", "nm_cr_velo", VEL), ("Pseudotime kernel", "nm_cr_pseudo", VEL),
            ("Waddington-OT", "nm_snap_wot", OT), ("moscot", "nm_snap_moscot", OT),
            ("PRESCIENT", "nm_snap_prescient", OT)]
    for n, k, c in meth:
        v = U.loc[k, "auroc_clone"]
        ax.plot([0, 470], [v, v], color=c, lw=0.5, ls=(0, (3, 2)), zorder=2)
    # label positions spread so that they do not collide
    ys = sorted([(U.loc[k, "auroc_clone"], n, c) for n, k, c in meth], reverse=True)
    pos, last = [], 10.0
    for v, n, c in ys:
        p = min(v, last - 0.0135); pos.append((p, v, n, c)); last = p
    for p, v, n, c in pos:
        ax.annotate(n, (470, v), (478, p), fontsize=5.2, color=c, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.3, color=c, shrinkA=0, shrinkB=0))
    ax.text(478, conn.max() + 0.012, "Connectivity kernel\n(18 settings)", fontsize=5.2,
            color="#127a55", va="bottom")
    # learning curves
    for arm, name, mfc in (("total", "Day-2 expression + sorted population", EXPR),
                           ("total_nopop", "Day-2 expression alone", "white")):
        L = LC[LC.arm == arm].sort_values("n_train_clones")
        ax.errorbar(L.n_train_clones, L.auroc_mean, yerr=L.auroc_sd, color=EXPR, lw=0.8,
                    elinewidth=0.5, capsize=0, marker="o", ms=3.0, mfc=mfc, mec=EXPR, mew=0.8,
                    zorder=4, label=name)
    # extrapolation to unlimited training clones
    # (the extrapolation to unlimited clones is uninformative and is given in the text only)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlim(0, XINF + 25); ax.set_ylim(0.55, 0.985)
    ax.set_xlabel("Labelled training clones")
    ax.set_ylabel("AUROC, 567 clones")
    ax.legend(frameon=False, loc="lower right", fontsize=5.4, handlelength=1.8,
              bbox_to_anchor=(0.86, 0.0))
    ax.set_title("What labels buy", loc="left", fontsize=7)
    # b: stacked increments over total counts
    S = pd.read_csv(R + "stk3nn_all_tot_summary.csv").set_index("arm")["auroc_mean"]
    P = pd.read_csv(R + "stk3nn_all_tot_pairs.csv").set_index("arm")
    Pd = pd.read_csv(R + "stk3nn_d2_tot_pairs.csv").set_index("arm")
    Pg = pd.read_csv(R + "stk3nn_configs_pairs.csv").set_index("arm")
    arms = [
        ("Unspliced counts", Pd.loc["raw_unspliced"], VEL, False),
        ("Steady-state velocity", Pd.loc["vel_det"], VEL, False),
        ("Dynamical velocity", Pd.loc["vel_dyn"], VEL, False),
        ("veloVI", Pd.loc["vel_velovi"], VEL, False),
        ("Velocity kernel (day-2 graph)", Pd.loc["cr_velo"], VEL, False),
        ("Pseudotime kernel (day-2 graph)", Pd.loc["cr_pseudo"], VEL, False),
        ("StationaryOT, 10 fates", P.loc["snap_statot"], OT, False),
        ("Waddington-OT, 10 fates", P.loc["snap_wot"], OT, False),
        ("moscot, 10 fates", P.loc["snap_moscot"], OT, False),
        ("PRESCIENT, 10 fates", P.loc["snap_prescient"], OT, False),
        ("Real-time kernel, 10 fates", Pg.loc["snap_rtk_ub"], OT, False),
        ("Connectivity kernel, 10 fates",
         pd.read_csv(R + "stk3nn_conn2_pairs.csv").set_index("arm").loc["cr_conn_n30_p30_d1"],
         CTRL, False),
    ]
    ax = fig.add_subplot(gs[1]); letter(ax, "b")
    for i, (n, r, c, h) in enumerate(arms):
        y = i + (0.6 if i >= 6 else 0) + (0.6 if i >= 11 else 0)
        hdot(ax, y, r["mean10"], r["lo_avg_part"], r["hi_avg_part"], c, h)
    ys_ = [i + (0.6 if i >= 6 else 0) + (0.6 if i >= 11 else 0) for i in range(len(arms))]
    ax.axvline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(ys_); ax.set_yticklabels([a[0] for a in arms])
    ax.set_ylim(ys_[-1] + 0.7, -0.7)
    ax.set_xlim(-0.012, 0.024); ax.set_xticks([-0.01, 0, 0.01, 0.02])
    ax.set_xlabel("AUROC increment over day-2 expression\n(total counts + sorted population)")
    ax.set_title("Added to the measured state", loc="left", fontsize=7)
    save(fig, "ed13")


def ed10():
    """The two state analyses moved out of Fig. 4: the day-2 transcriptome
    against the other well (212 clones), and the power analysis with synthetic
    features."""
    fig = plt.figure(figsize=(160 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.0], wspace=0.75)
    ax = fig.add_subplot(gs[0]); letter(ax, "a")
    M = pd.read_csv(R + "mc_d2.csv")
    lab = {"A2_state_spliced": "Spliced", "A3_state_total": "Total",
           "A4_state_smoothed": "Smoothed"}
    y = 0
    ticks, names = [], []
    for d, dn in (("well1_predicts_well2", "well 1 → 2"),
                  ("well2_predicts_well1", "well 2 → 1")):
        for k, n in lab.items():
            r = M[(M.direction == d) & (M.who == k)].iloc[0]
            hdot(ax, y, r["diff_vs_oracle"], r["lo"], r["hi"], EXPR)
            ticks.append(y); names.append("%s (%s)" % (n, dn)); y += 1
        y += 0.6
    ax.axvline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(ticks); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("AUROC, day-2 expression minus other well"); ax.set_xticks([-0.05, 0, 0.05])
    ax.set_title("Day-2 transcriptome vs the other well", loc="left", fontsize=7)
    S = pd.read_csv(R + "stk3nn_all_tot_summary.csv").set_index("arm")["auroc_mean"]
    Sd = pd.read_csv(R + "stk3nn_d2_tot_summary.csv").set_index("arm")["auroc_mean"]
    Sg = pd.read_csv(R + "stk3nn_configs_summary.csv").set_index("arm")["auroc_mean"]
    P = pd.read_csv(R + "stk3nn_all_tot_pairs.csv").set_index("arm")
    Pd = pd.read_csv(R + "stk3nn_d2_tot_pairs.csv").set_index("arm")
    Pg = pd.read_csv(R + "stk3nn_configs_pairs.csv").set_index("arm")
    outs = [(Sd["S1_" + k], Pd.loc[k], VEL) for k in
            ("raw_unspliced", "vel_det", "vel_dyn", "vel_velovi", "cr_velo", "cr_pseudo")]
    outs += [(S["S1_" + k], P.loc[k], OT) for k in
             ("nm_snap_statot", "snap_statot", "nm_snap_wot", "snap_wot", "nm_snap_moscot",
              "snap_moscot", "nm_snap_prescient", "snap_prescient")]
    outs += [(Sg["S1_snap_rtk_ub"], Pg.loc["snap_rtk_ub"], OT)]
    ax = fig.add_subplot(gs[1]); letter(ax, "b")
    for pre, mk, name in (("cal_a", "s", "synthetic, 10 columns"),
                          ("cal1b_a", "^", "synthetic, 1 column")):
        pts = [(S["S1_" + k], P.loc[k]) for k in P.index if k.startswith(pre)]
        xs = np.array([p[0] for p in pts]); o = np.argsort(xs)
        ys = np.array([p[1]["mean10"] for p in pts])
        for x, (_, r) in zip(xs, pts):
            ax.plot([x, x], [r["lo_avg_part"], r["hi_avg_part"]], color=SYN, lw=0.7)
        ax.plot(xs[o], ys[o], mk + "-", color=SYN, ms=3, lw=0.6, label=name)
    Y = pd.read_csv(R + "yardstick_sim.csv").pivot(index="target", columns="built",
                                                  values="increment")
    ratio = (Y["from_f"] / Y["from_pc"]).values
    pts = sorted([(S["S1_" + k], P.loc[k, "mean10"]) for k in P.index if k.startswith("cal_a")])
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    ax.fill_between(xs, ys / ratio.max(), ys / ratio.min(), color=SYN, alpha=0.35,
                    lw=0, label="10 columns, rescaled to an\nindependent day-2 signal")
    for a, r, c in outs:
        ax.plot([a, a], [r["lo_avg_part"], r["hi_avg_part"]], color=c, lw=0.5, alpha=0.6)
        ax.plot([a], [r["mean10"]], "o", ms=3.0, mfc=c, mec=c, mew=0.8, zorder=5)
    ax.axhline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(0.62, 0.96)
    ax.set_xlabel("AUROC alone (with sorted population)")
    ax.set_ylabel("AUROC increment")
    ax.plot([], [], "o", ms=3.0, color=VEL, label="splicing-based outputs")
    ax.plot([], [], "o", ms=3.0, color=OT, label="optimal-transport and potential-based\nmethods, real-time kernel")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, fontsize=5.4)
    ax.set_title("What an independent signal would add", loc="left", fontsize=7)
    save(fig, "ed10")


# ---------------------------------------------------------------- figure 3
def ed6():
    """Fig. 2a-c in 30 principal components, same encoding (fields keep colour,
    marker and row), with clone-resampled intervals for forward minus reverse."""
    E = pd.read_csv(R + "direction_embedding_bench.csv")
    E = E[E.space == "pca30"].set_index("field")
    ANN, ANN2 = "#8a8a86", "#52514e"
    fields = [("steady", "Steady-state velocity", VEL, "o", True),
              ("dynamical", "Dynamical velocity", VEL, "s", True),
              ("velovi", "veloVI", VEL, "D", True),
              ("perm_control", "Shuffled unspliced", CTRL, "o", False),
              ("minus_gamma_s", "Degradation only (−γs)", CTRL, "v", True),
              ("knn_mean_in_pca", "Neighbour average", EXPR, "^", False),
              ("knn_mean", "Neighbour average, gene space", EXPR, "^", True),
              ("centroid_in_pca", "Towards all mature cells", ANN, "P", True),
              ("nearest_mature_centroid", "Into nearest mature population", ANN2, "*", True),
              ("random", "Random vectors", SYN, "x", True)]
    fig = plt.figure(figsize=(183 * MM, 66 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.08)
    panels = [("cbdir_fwd", "Forward CBDir", "Cross-boundary direction (30 PCs)", (-0.1, 0.7)),
              ("fwd_minus_rev", "Forward − reverse CBDir", "Reverse score (30 PCs)",
               (-0.16, 0.1)),
              ("iccoh", "ICCoh", "In-cluster coherence (30 PCs)", (-0.05, 1.02))]
    n = len(fields)
    for j, (col, xl, title, xlim) in enumerate(panels):
        ax = fig.add_subplot(gs[j]); letter(ax, "abc"[j])
        for i in range(0, n, 2):
            ax.axhspan(i - 0.45, i + 0.45, color="#f4f4f2", lw=0, zorder=0)
        for i, (k, lab, c, m, filled) in enumerate(fields):
            if k not in E.index or not np.isfinite(E.loc[k, col]):
                continue
            if col == "fwd_minus_rev":
                ax.plot([E.loc[k, "diff_lo"], E.loc[k, "diff_hi"]], [i, i], color=c, lw=0.8, zorder=2)
            ax.plot([E.loc[k, col]], [i], m, ms=(4.2 if m == "*" else 3.2),
                    mfc=c if filled else "white", mec=c, mew=0.8, zorder=3)
        ax.axvline(0, color="#6b6b68", lw=0.5, zorder=1)
        ax.set_xlim(*xlim)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f[1] for f in fields] if j == 0 else [])
        ax.set_ylim(n - 0.5, -0.6)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xl); ax.set_title(title, loc="left", fontsize=7)
    save(fig, "ed6")


def ed7():
    fig = plt.figure(figsize=(120 * MM, 52 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.9)
    ax = fig.add_subplot(gs[0]); letter(ax, "a")
    Q = pd.read_csv(R + "rsscale_d2.csv").set_index("quantity")
    rows = [("Second subsample\nover the first", "second repeat on top of the first (R2-R1)", SYN),
            ("Unspliced layer\nover one subsample", "unspliced minus one repeat (R3-R1)", VEL),
            ("Unspliced layer\nover two subsamples", "unspliced minus two repeats (R3-R2)", VEL)]
    for i, (n, k, c) in enumerate(rows):
        r = Q.loc[k]
        hdot(ax, i, r["value"], r["lo"], r["hi"], c)
    ax.axvline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("AUROC difference on a fixed base")
    ax.set_title("Depth-matched calibration", loc="left", fontsize=7)

    ax = fig.add_subplot(gs[1]); letter(ax, "b")
    D = pd.read_csv(R + "dn_d2_summary.csv").set_index("family")
    dep = pd.read_csv(R + "dn_d2_depths.csv").iloc[0]
    fr = [1, 0.5, 0.25, 0.125, 0.0625]
    xs = [dep["median_spliced_on_selected_genes"] * f for f in fr]
    ys = [D.loc["D_frac%s" % (("%g" % f) if f != 1 else "1"), "auroc_mean"] for f in fr]
    es = [0 if f == 1 else D.loc["D_frac%g" % f, "auroc_sd"] for f in fr]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=EXPR, ms=3, lw=0.8, elinewidth=0.6,
                label="spliced counts, downsampled")
    per = pd.read_csv(R + "ev_d2_perrepeat.csv")
    u0 = per[(per.arm == "A5_unspliced_only") & (per.seed == 0)]["auroc"].iloc[0]
    ux = dep["median_unspliced_on_selected_genes"]
    ax.plot([ux], [u0], "D", color=VEL, ms=3.6, label="unspliced counts")
    ax.plot([ux], [D.loc["D_udepth", "auroc_mean"]], "o", mfc="white", mec=EXPR,
            ms=3.2, mew=0.9, label="spliced at unspliced depth")
    ax.set_xscale("log"); ax.set_xlabel("Median molecules per cell (selected genes)")
    ax.set_xticks([50, 100, 200, 400, 800]); ax.set_xticklabels(["50", "100", "200", "400", "800"])
    ax.minorticks_off()
    ax.set_ylabel("AUROC (first repetition)")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.4)
    ax.set_title("Depth series", loc="left", fontsize=7)
    save(fig, "ed7")


# ---------------------------------------------------------------- figure 2 (direction)
def fig2():
    """Direction, shown on the data: streamlines of steady-state velocity and of
    two fields without information about dynamics on the same UMAP; then the
    benchmarks' metrics for every field (CBDir averaged within transitions,
    forward minus reverse with clone-resampled intervals, ICCoh on gene-space
    vectors as the benchmarks compute it)."""
    from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid
    Z = np.load(R + "fig_stream.npz", allow_pickle=False)
    Uemb, state = Z["umap"].astype(float), Z["state"]
    path = R + "direction_embedding3.csv"
    E = pd.read_csv(path if os.path.exists(path) else R + "direction_embedding_bench.csv")
    E = E[E.space == "umap"].set_index("field")
    t = "_t" if "cbdir_fwd_t" in E.columns else ""
    pop = {"Neutrophil": "#cf5a2a", "Monocyte": "#2f6aa3", "Baso": "#8c6bb1",
           "Mast": "#c9a227", "Meg": "#5f9a3a", "Erythroid": "#b5446e"}
    fig = plt.figure(figsize=(183 * MM, 128 * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.28)
    top = outer[0].subgridspec(1, 3, wspace=0.04)
    for j, (k, title) in enumerate((("steady", "Steady-state velocity"),
                                    ("knn_mean", "Neighbour average"),
                                    ("minus_gamma_s", "Degradation only ($-\\gamma s$)"))):
        ax = fig.add_subplot(top[j]); letter(ax, "abc"[j])
        col = np.array([pop.get(s, "#dcdad5") for s in state])
        o = np.argsort([s in pop for s in state])
        ax.scatter(Uemb[o, 0], Uemb[o, 1], s=0.6, c=col[o], lw=0, rasterized=True)
        Xg, Vg = compute_velocity_on_grid(Uemb, Z[k].astype(float), density=1.0, smooth=0.6,
                                          autoscale=False, adjust_for_stream=True)
        lw = 0.7 * np.sqrt((Vg ** 2).sum(0)); lw = np.nan_to_num(lw / np.nanmax(lw)) * 0.9 + 0.2
        ax.streamplot(Xg[0], Xg[1], Vg[0], Vg[1], color="#2b2b2b", linewidth=lw, density=1.1,
                      arrowsize=0.5, arrowstyle="-|>")
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, loc="left", fontsize=7)

    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker="o", ms=3.5, mfc=c, mec=c, label=s) for s, c in pop.items()]
    hs.append(Line2D([], [], ls="", marker="o", ms=3.5, mfc="#dcdad5", mec="#dcdad5",
                     label="Undifferentiated and other"))
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.465), ncol=7, frameon=False,
               fontsize=5.6, handletextpad=0.2, columnspacing=1.0)
    fields = [("steady", "Steady-state velocity", VEL), ("dynamical", "Dynamical velocity", VEL),
              ("velovi", "veloVI", VEL), ("perm_control", "Shuffled unspliced", CTRL),
              ("minus_gamma_s", "Degradation only", CTRL), ("knn_mean", "Neighbour average", EXPR)]
    bot = outer[1].subgridspec(1, 3, wspace=0.12)
    specs = [("cbdir_fwd" + t, None, "CBDir (forward)"),
             ("fwd_minus_rev" + t, ("diff%s_lo" % t, "diff%s_hi" % t), "CBDir, forward minus reverse"),
             ("iccoh_gene" if "iccoh_gene" in E.columns else "iccoh", None, "ICCoh")]
    for j, (col, ci, xl) in enumerate(specs):
        ax = fig.add_subplot(bot[j]); letter(ax, "def"[j])
        for i, (k, n, c) in enumerate(fields):
            v = E.loc[k, col]
            ax.barh(i, v, height=0.62, color=c, lw=0)
            if ci is not None:
                ax.plot([E.loc[k, ci[0]], E.loc[k, ci[1]]], [i, i], color="#2b2b2b", lw=0.7)
        ax.axvline(0, color="#6b6b68", lw=0.5)
        ax.set_yticks(range(len(fields)))
        ax.set_yticklabels([f[1] for f in fields] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis(); ax.set_xlabel(xl)
    save(fig, "fig2")


def ed12():
    """Direction metrics with null fields on LARRY. One encoding throughout: a
    field keeps its colour, marker and row in every panel."""
    T = pd.read_csv(R + "ta_d2_controls.csv")
    E = pd.read_csv(R + "direction_embedding_bench.csv")
    E = E[E.space == "umap"].set_index("field")
    ANN, ANN2 = "#8a8a86", "#52514e"
    # key in direction_embedding2 / annotation_fields, key in ta_d2_controls,
    # label, colour, marker, filled
    fields = [("steady", "det", "Steady-state velocity", VEL, "o", True),
              ("dynamical", "dyn", "Dynamical velocity", VEL, "s", True),
              ("velovi", "velovi", "veloVI", VEL, "D", True),
              ("perm_control", "det_shuffled_u", "Shuffled unspliced", CTRL, "o", False),
              ("minus_gamma_s", "minus_gamma_s", "Degradation only (−γs)", CTRL, "v", True),
              ("knn_mean_in_umap", None, "Neighbour average", EXPR, "^", False),
              ("knn_mean", None, "Neighbour average, gene space", EXPR, "^", True),
              ("centroid_in_umap", None, "Towards all mature cells", ANN, "P", True),
              ("nearest_mature_centroid", None, "Into nearest mature population", ANN2, "*", True),
              ("random", None, "Random vectors", SYN, "x", True),
              (None, "state_itself", "Cell's own position (reference)", "#b0b0ac", "h", True)]
    fig = plt.figure(figsize=(183 * MM, 128 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.42, height_ratios=[1.0, 0.68])

    def mark(ax, x, y, f, lo=None, hi=None):
        _, _, _, c, m, filled = f
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=c, lw=0.8, zorder=2)
        ax.plot([x], [y], m, ms=(4.2 if m == "*" else 3.2), mfc=c if filled else "white",
                mec=c, mew=0.8, zorder=3)

    def frame(ax, j, xl, title, n, rows=None):
        rows = fields[:n] if rows is None else rows
        ax.set_yticks(range(n))
        ax.set_yticklabels([f[2] for f in rows] if j == 0 else [])
        ax.set_ylim(n - 0.5, -0.6)
        ax.tick_params(axis="y", length=0)
        for i in range(0, n, 2):
            ax.axhspan(i - 0.45, i + 0.45, color="#f4f4f2", lw=0, zorder=0)
        ax.set_xlabel(xl); ax.set_title(title, loc="left", fontsize=7)

    n_emb = 10
    panels = [("cbdir_fwd", "Forward CBDir", "Cross-boundary direction (UMAP)", (-0.45, 0.25)),
              ("fwd_minus_rev", "Forward − reverse CBDir", "Reverse score removes geometry",
               (-0.8, 0.45)),
              ("iccoh", "ICCoh", "In-cluster coherence (gene space)", (-0.05, 1.02))]
    for j, (col, xl, title, xlim) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j]); letter(ax, "abc"[j])
        for i, f in enumerate(fields[:n_emb]):
            k = f[0]
            if k not in E.index or not np.isfinite(E.loc[k, col]):
                continue
            if col == "fwd_minus_rev":
                mark(ax, E.loc[k, col], i, f, E.loc[k, "diff_lo"], E.loc[k, "diff_hi"])
            else:
                mark(ax, E.loc[k, col], i, f)
        ax.axvline(0, color="#6b6b68", lw=0.5, zorder=1)
        ax.set_xlim(*xlim)
        frame(ax, j, xl, title, n_emb)
    for j, (axis, title) in enumerate((("maturation", "Sign along the maturation axis"),
                                       ("clonal fate", "Sign along the fate axis"))):
        ax = fig.add_subplot(gs[1, j]); letter(ax, "de"[j])
        sub = [f for f in fields if f[1] is not None]
        for i, f in enumerate(sub):
            r = T[(T.axis == axis) & (T.vector == f[1])].iloc[0]
            v, sd = r["direction_correct"], r["sd"]
            mark(ax, v, i, f, v - sd, v + sd)
        ax.axvline(0.5, color="#6b6b68", lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(0.25, 1.0)
        frame(ax, j, "Fraction of day-2 cells with correct sign", title, len(sub), sub)
    # legend panel
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker=f[4], ms=(4.6 if f[4] == "*" else 3.6),
                 mfc=f[3] if f[5] else "white", mec=f[3], mew=0.8, label=f[2]) for f in fields]
    ax.legend(handles=hs, frameon=False, loc="center left", fontsize=5.6, handletextpad=0.4,
              labelspacing=0.7, borderaxespad=0)
    save(fig, "ed12")


# ---------------------------------------------------------------- figure 3 (ranking)
def fig3():
    """Rankings: (a) clones needed to resolve an AUROC difference, as a curve;
    (b) the values in a published ranking against the margin of error of a
    comparison scored on different cells."""
    U = pd.read_csv(R + "unpaired_margin.csv").set_index("n2")
    Pf = pd.read_csv(R + "pair_floors_unified.csv")
    fig = plt.figure(figsize=(165 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.55)
    ax = fig.add_subplot(gs[0]); letter(ax, "a")
    d = np.geomspace(0.003, 0.1, 200)
    diff = Pf.iloc[2:]["floor"]; near = Pf.iloc[0]["floor"]
    ax.fill_between(d, 567 * (diff.min() / d) ** 2, 567 * (diff.max() / d) ** 2, color=OT, alpha=0.35,
                    lw=0, label="two different methods")
    ax.plot(d, 567 * (near / d) ** 2, color=EXPR, lw=1.0, label="two nearly identical pipelines")
    ax.axhline(567, color="#6b6b68", lw=0.6, ls=(0, (3, 2)))
    ax.text(0.0032, 500, "LARRY in vitro (567 clones)", fontsize=5.5, va="top", color="#52514e")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.003, 0.1); ax.set_ylim(100, 3e5)
    ax.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1]); ax.set_xticklabels(["0.005", "0.01", "0.02", "0.05", "0.1"])
    ax.set_xlabel("AUROC difference to resolve")
    ax.set_ylabel("Clones needed")
    ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    ax.set_title("Resolution of a lineage-tracing benchmark", loc="left", fontsize=7)
    ax = fig.add_subplot(gs[1]); letter(ax, "b")
    rows = [("Chreode", 0.468, 183), ("scDiffEq", 0.468 - 0.0053, 296),
            ("Waddington-OT", 0.468 - 0.0087, 265), ("Static control", 0.468 - 0.0855, 335)]
    top = rows[0][1]
    ax.axvline(top, color=OT, lw=0.6, ls=(0, (2, 2)))
    for i, (n, r, n2) in enumerate(rows):
        if i:
            m = U.loc[n2, "margin_unpaired"]
            ax.plot([top - m, top + m], [i, i], color="#cfcdc8", lw=5, solid_capstyle="butt")
        ax.plot([r], [i], "o", ms=4.2, color=OT if i == 0 else EXPR, zorder=5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.5, -0.6); ax.set_xlim(0.30, 0.62)
    ax.set_xlabel("Reported Pearson r with observed fate bias")
    ax.set_title("A published ranking (Chreode)", loc="left", fontsize=7)
    save(fig, "fig3")


def old_fig3():
    F = pd.read_csv(R + "rfloor2_invitro_summary.csv")
    F = F[F.scenario == "n183"].sort_values("rho")
    U = pd.read_csv(R + "unpaired_margin.csv").set_index("n2")
    fig = plt.figure(figsize=(165 * MM, 55 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=1.25)
    # a: the general result -- clones needed to resolve small differences
    ax = fig.add_subplot(gs[0]); letter(ax, "a")
    Pf = pd.read_csv(R + "pair_floors_unified.csv")
    nm = pd.read_csv(R + "nm_incr_raw_matrix_vs_velocity.csv")
    main = nm[nm.arm.isin(["K_nm_snap_statot", "K_nm_snap_wot", "K_nm_snap_moscot",
                           "K_nm_snap_prescient"])]
    inc = 567 * ((0.714 * main["width"]) / 0.005) ** 2
    groups = [("Increment of a method\nover expression (stacked)", inc.min(), inc.max(), OT),
              ("Two near-identical\npipelines", Pf.iloc[0]["clones_for_0.005"],
               Pf.iloc[0]["clones_for_0.005"], EXPR),
              ("Two different methods, or a\nmethod vs expression, alone",
               Pf.iloc[2:]["clones_for_0.005"].min(), Pf.iloc[2:]["clones_for_0.005"].max(), OT)]
    for i, (n, lo, hi, c) in enumerate(groups):
        ax.plot([lo, hi], [i, i], color=c, lw=3.0, solid_capstyle="round")
        ax.plot([lo, hi], [i, i], "|", color=c, ms=6, mew=1.2)
    ax.axvline(567, color=EXPR, lw=0.8)
    ax.text(600, -0.45, "LARRY in vitro, 567", fontsize=5.5, ha="left", va="center")
    ax.set_xscale("log"); ax.set_xlim(300, 1e5); ax.set_ylim(2.5, -0.8)
    ax.set_yticks(range(3)); ax.set_yticklabels([g[0] for g in groups])
    ax.set_xlabel("Clones meeting the criteria\nto resolve 0.005 AUROC")
    ax.set_title("Clones needed", loc="left", fontsize=7)
    # b: Chreode's three differences against the margins of error
    ax = fig.add_subplot(gs[1]); letter(ax, "b")
    m_lo = np.interp(0.75, F.rho, F.critical_r); m_hi = np.interp(0.49, F.rho, F.critical_r)
    h_lo = np.interp(0.995, F.rho, F.critical_r); h_hi = np.interp(0.95, F.rho, F.critical_r)
    rows = [("vs next best (scDiffEq)", 0.0053, 296), ("vs Waddington-OT", 0.0087, 265),
            ("vs static control", 0.0855, 335)]
    for i, (n, gap, n2) in enumerate(rows):
        ax.plot([h_lo, h_hi], [i - 0.18, i - 0.18], color="#c8c8c4", lw=3.0, solid_capstyle="butt")
        ax.plot([m_lo, m_hi], [i, i], color="#9a9a96", lw=3.0, solid_capstyle="butt")
        ax.plot([U.loc[n2, "margin_unpaired"]], [i + 0.18], "|", color=EXPR, ms=7, mew=1.4)
        ax.plot([gap], [i], "o", ms=4, color=OT, zorder=5)
    ax.set_xscale("log"); ax.set_xlim(0.004, 0.2)
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(2.6, -0.9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=OT, marker="o", ls="", ms=4, label="Chreode's difference in r"),
                       Line2D([], [], color="#9a9a96", lw=3,
                              label="margin, paired, ρ$_\\mathrm{noise}$ implied by\nmeasured residual correlations"),
                       Line2D([], [], color="#c8c8c4", lw=3, label="margin, paired, ρ$_\\mathrm{noise}$ 0.95–0.995"),
                       Line2D([], [], color=EXPR, marker="|", ls="", ms=7, mew=1.4, label="margin, unpaired")],
              frameon=False, fontsize=5.2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_xlabel("Difference in Pearson r (183 clones)")
    ax.set_title("A published ranking", loc="left", fontsize=7)
    save(fig, "old_fig3")


def ed8():
    F = pd.read_csv(R + "rfloor2_invitro_summary.csv")
    F = F[F.scenario == "n183"]
    fig = plt.figure(figsize=(100 * MM, 62 * MM))
    ax = fig.add_subplot(111)
    ax.plot(F.rho, F.critical_r, "o-", color=EXPR, ms=2.6, lw=0.9,
            label="margin of error (half-width)")
    ax.plot(F.rho, F.mde_r, "o--", color=EXPR, ms=2.6, lw=0.7, mfc="white",
            label="minimum detectable difference")
    ax.axvspan(0.49, 0.75, color="#dfe9f7", lw=0, zorder=0)
    ax.text(0.62, 0.035, "range implied by measured\nresidual correlations", ha="center",
            va="center", fontsize=5.5, color="#52514e")
    for v, t in ((0.0855, "vs static control"), (0.0087, "vs Waddington-OT"),
                 (0.0053, "vs next best")):
        ax.axhline(v, color=OT, lw=0.8)
        ax.text(1.005, v, "%s\n%.4f" % (t, v), fontsize=5.5, va="center", ha="left",
                transform=ax.get_yaxis_transform())
    ax.set_yscale("log"); ax.set_xlim(0.48, 1.0); ax.set_ylim(0.004, 0.2)
    ax.set_xlabel("Noise correlation between the two predictors, ρ$_\\mathrm{noise}$")
    ax.set_ylabel("Difference in Pearson r, 183 clones")
    ax.legend(frameon=False, loc="upper right", fontsize=5.5)
    save(fig, "ed8")



def xed1():
    """Extended Data Fig. 1: each method minus the connectivity kernel."""
    P = pd.read_csv(R + "refint_invitro2_pairs.csv")
    P = P[P.reference == "nm_cr_conn_n30_p30_d1"].set_index("method")
    Q = pd.read_csv(R + "refint_vs_d2_pairs.csv")
    Q = Q[Q.reference == "nm_cr_conn_n30_p30_d1"].set_index("method")
    meth = [("Waddington-OT", "nm_snap_wot", OT), ("moscot", "nm_snap_moscot", OT),
            ("PRESCIENT", "nm_snap_prescient", OT), ("StationaryOT", "nm_snap_statot", OT),
            ("Real-time kernel", "nm_snap_rtk_ub", OT), ("Velocity kernel", "nm_cr_velo", VEL),
            ("Pseudotime kernel", "nm_cr_pseudo", VEL),
            ("Velocity + connectivity kernel", "nm_cr_vkck", VEL)]
    fig = plt.figure(figsize=(160 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    for j, (pre, xl, xlim) in enumerate((("r", "Method minus kernel, Pearson r", (-0.42, 0.32)),
                                         ("auroc", "Method minus kernel, AUROC", (-0.2, 0.09)))):
        ax = fig.add_subplot(gs[j]); letter(ax, "ab"[j])
        for i, (n, k, c) in enumerate(meth):
            r = P.loc[k]
            ax.plot([r[pre + "_lo"], r[pre + "_hi"]], [i - 0.14, i - 0.14], color=c, lw=1.0)
            ax.plot([r[pre + "_diff"]], [i - 0.14], "o", ms=3.4, color=c)
            q = Q.loc[k]
            ax.plot([q[pre + "_lo"], q[pre + "_hi"]], [i + 0.18, i + 0.18], color=c, lw=0.7, alpha=0.45)
            ax.plot([q[pre + "_diff"]], [i + 0.18], "o", ms=3.0, mfc="white", mec=c, mew=0.8)
        ax.axvline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
        ax.set_yticks(range(len(meth)))
        ax.set_yticklabels([m[0] for m in meth] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(meth) - 0.45, -0.6); ax.set_xlim(*xlim); ax.set_xlabel(xl)
    save(fig, "ed1")


def xed2():
    """Extended Data Fig. 2: learning curve of the classifier, the methods'
    direct AUROC and the noise ceiling."""
    LC = pd.read_csv(R + "learning_curve_lowend_summary.csv")
    U = pd.read_csv(R + "unsupervised_reference.csv").set_index("predictor")
    LP = pd.read_csv(R + "labelparam_summary.csv", index_col=0)
    fig = plt.figure(figsize=(110 * MM, 78 * MM))
    ax = fig.add_subplot(111)
    cl, ch = LP.loc["ceiling_cell", "boot_lo"], LP.loc["ceiling_clone", "boot_hi"]
    ax.axhspan(cl, ch, color="#e6e6e3", lw=0, zorder=0)
    ax.text(10, ch - 0.004, "noise ceiling", fontsize=5.6, color="#52514e", va="top")
    conn = U.loc[U.index.str.startswith("nm_cr_conn_"), "auroc_clone"]
    ax.fill_between([0, 470], conn.min(), conn.max(), color=CTRL, alpha=0.25, lw=0, zorder=1)
    meth = [("Real-time kernel", "nm_snap_rtk_ub", OT), ("StationaryOT", "nm_snap_statot", OT),
            ("Velocity kernel", "nm_cr_velo", VEL), ("Waddington-OT", "nm_snap_wot", OT),
            ("moscot", "nm_snap_moscot", OT), ("PRESCIENT", "nm_snap_prescient", OT)]
    for n, k, c in meth:
        ax.plot([0, 470], [U.loc[k, "auroc_clone"]] * 2, color=c, lw=0.5, ls=(0, (3, 2)), zorder=2)
    ys = sorted([(U.loc[k, "auroc_clone"], n, c) for n, k, c in meth], reverse=True)
    last = 10.0
    for v, n, c in ys:
        p = min(v, last - 0.016); last = p
        ax.annotate(n, (470, v), (480, p), fontsize=5.4, color=c, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.3, color=c, shrinkA=0, shrinkB=0))
    ax.text(480, conn.max() + 0.012, "connectivity kernel", fontsize=5.4, color="#127a55",
            va="bottom")
    for arm, name, mfc in (("total", "with sorted population", EXPR),
                           ("total_nopop", "expression alone", "white")):
        L = LC[LC.arm == arm].sort_values("n_train_clones")
        ax.errorbar(L.n_train_clones, L.auroc_mean, yerr=L.auroc_sd, color=EXPR, lw=0.8,
                    elinewidth=0.5, capsize=0, marker="o", ms=3.0, mfc=mfc, mec=EXPR, mew=0.8,
                    zorder=4, label="classifier on day-2 " + name)
    ax.set_xlim(0, 600); ax.set_ylim(0.55, 0.985)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("Labelled training clones"); ax.set_ylabel("AUROC, 567 clones")
    ax.legend(frameon=False, loc="lower right", fontsize=5.6, bbox_to_anchor=(0.8, 0.0))
    save(fig, "ed2")


def xed3():
    """Extended Data Fig. 3: the benchmarks' metrics on LARRY and four benchmark
    data sets, for the velocity estimates and three baselines."""
    B = pd.read_csv(R + "benchmark_direction_bench.csv")
    L = pd.read_csv(R + "direction_embedding_bench.csv")
    L["dataset"] = "LARRY"
    L["space"] = L["space"].replace({"umap": "embedding"})
    L["field"] = L["field"].replace({"knn_mean_in_umap": "knn_mean_in_embedding"})
    D = pd.concat([L, B], ignore_index=True)
    sets = ["LARRY", "pancreas", "dentate gyrus", "gastrulation erythroid", "bone marrow"]
    labels = ["LARRY", "Pancreas", "Dentate gyrus", "Gastrulation erythroid", "Bone marrow"]
    panels = [("embedding", "iccoh", "ICCoh (gene space)", {"nbr": "knn_mean"}),
              ("embedding", "cbdir_fwd", "CBDir, embedding", {"nbr": "knn_mean"}),
              ("pca30", "cbdir_fwd", "CBDir, 30 PCs", {"nbr": "knn_mean"})]
    base = [("perm_control", "shuffled unspliced", CTRL, "o", False),
            ("minus_gamma_s", "degradation only", CTRL, "o", True),
            ("nbr", "neighbour average", EXPR, "s", True)]
    fig = plt.figure(figsize=(170 * MM, 58 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.08)
    for j, (space, col, title, alias) in enumerate(panels):
        ax = fig.add_subplot(gs[j]); letter(ax, "abc"[j])
        for i, ds in enumerate(sets):
            if i % 2 == 0:
                ax.axhspan(i - 0.45, i + 0.45, color="#f4f4f2", lw=0, zorder=0)
            S = D[(D.dataset == ds) & (D.space == space)].set_index("field")
            v = [S.loc[k, col] for k in ("steady", "dynamical", "velovi") if k in S.index]
            ax.plot([min(v), max(v)], [i, i], color=VEL, lw=2.2, alpha=0.35, solid_capstyle="round")
            ax.plot(v, [i] * len(v), "|", color=VEL, ms=6, mew=1.2)
            for k, n, c, m, filled in base:
                kk = alias.get(k, k)
                if kk in S.index and np.isfinite(S.loc[kk, col]):
                    ax.plot([S.loc[kk, col]], [i], m, ms=4, mfc=c if filled else "white", mec=c,
                            mew=0.9, zorder=3)
        ax.axvline(0, color="#6b6b68", lw=0.5)
        ax.set_yticks(range(len(sets))); ax.set_yticklabels(labels if j == 0 else [])
        ax.tick_params(axis="y", length=0); ax.set_ylim(len(sets) - 0.5, -0.5)
        ax.set_xlabel(title)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=VEL, marker="|", ls="-", lw=2.2, alpha=0.6, ms=6, mew=1.2,
                 label="velocity estimates (steady-state, dynamical, veloVI)")]
    hs += [Line2D([], [], ls="", marker=m, ms=4, mfc=c if f else "white", mec=c, mew=0.9, label=n)
           for k, n, c, m, f in base]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, fontsize=5.6,
               bbox_to_anchor=(0.5, -0.2))
    save(fig, "ed3")


def xed4():
    """Extended Data Fig. 4: the sign of velocity along the maturation and fate axes."""
    T = pd.read_csv(R + "ta_d2_controls.csv")
    rows = [("det", "Steady-state velocity", VEL, "o", True), ("dyn", "Dynamical velocity", VEL, "s", True),
            ("velovi", "veloVI", VEL, "D", True), ("det_shuffled_u", "Shuffled unspliced", CTRL, "o", False),
            ("minus_gamma_s", "Degradation only", CTRL, "o", True),
            ("state_itself", "Cell's own position", "#9a9a96", "s", True)]
    fig = plt.figure(figsize=(130 * MM, 52 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    for j, (axis, title) in enumerate((("maturation", "Maturation axis"), ("clonal fate", "Fate axis"))):
        ax = fig.add_subplot(gs[j]); letter(ax, "ab"[j])
        for i, (k, n, c, m, f) in enumerate(rows):
            r = T[(T.axis == axis) & (T.vector == k)].iloc[0]
            v, sd = r["direction_correct"], r["sd"]
            ax.plot([v - sd, v + sd], [i, i], color=c, lw=0.8)
            ax.plot([v], [i], m, ms=3.6, mfc=c if f else "white", mec=c, mew=0.9)
        ax.axvline(0.5, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
        ax.set_xlim(0.25, 1.0); ax.set_ylim(len(rows) - 0.5, -0.6)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Fraction of day-2 cells with correct sign"); ax.set_title(title, loc="left", fontsize=7)
    save(fig, "ed4")


for f in [globals()[n] for n in (sys.argv[2].split(",") if len(sys.argv) > 2 else
                                 ["fig1", "fig2", "fig3", "xed1", "xed2", "xed3", "xed4"])]:
    try:
        f()
    except Exception as e:  # report and keep going, so one broken panel does not hide the rest
        import traceback
        traceback.print_exc()
