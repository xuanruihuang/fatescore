"""Extended Data figures, from result files only.

Usage: 65_extended_figures.py <outdir>   (run from code/; reads ../out/*)
Colours as in 62_figures.py.
  ED1  the pre-specified concatenation design: increments over the smoothed
       base (RNA-velocity family) and over total counts (snapshot methods,
       10 fates and the 1-column share), repeat-0 paired-bootstrap intervals.
  ED2  StationaryOT's 10-fate increment: with and without the held-out clones'
       later cells in the coupling; its columns split; expression-graph
       (connectivity) kernels at three graph settings.
  ED3  the two graph constructions (day-2 cells only vs all time points) for the
       RNA-velocity family, and the comparisons within each sorting gate.
  ED4  increments within the half of clones the reference is least / most sure
       about, synthetic features beside the methods.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
R = "../out/"
EXPR, VEL, OT, CTRL, SYN = "#3b3b3b", "#2a78d6", "#eb6834", "#1baf7a", "#9a9a96"
MM = 1 / 25.4
plt.rcParams.update({
    "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})


def letter(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="right", va="bottom")


def hdot(ax, y, x, lo, hi, c, hollow=False):
    ax.plot([lo, hi], [y, y], color=c, lw=0.9)
    ax.plot([x], [y], "o", ms=3.2, mfc="white" if hollow else c, mec=c, mew=0.9)


def rows_plot(ax, rows, xlabel, title):
    for i, (n, x, lo, hi, c, h) in enumerate(rows):
        hdot(ax, i, x, lo, hi, c, h)
    ax.axvline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.set_title(title, loc="left", fontsize=7)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", name, flush=True)


def ed1():
    E = pd.read_csv(R + "ev_d2_pairs.csv")
    E = E[E.ref == "A4_state_smoothed"].set_index("arm")
    vel = [("Unspliced counts (on spliced)", None),
           ("Smoothed splicing ratio", "A7_smoothed_plus_ratio"),
           ("Steady-state velocity", "V1_smoothed_plus_vel_det"),
           ("Dynamical velocity", "V2_smoothed_plus_vel_dyn"),
           ("veloVI", "V3_smoothed_plus_vel_velovi"),
           ("CellRank velocity kernel", "V4_smoothed_plus_cr_velo"),
           ("CellRank pseudotime kernel", "V5_smoothed_plus_cr_pseudo")]
    Es = pd.read_csv(R + "ev_d2_pairs.csv")
    Es = Es[Es.ref == "A2_state_spliced"].set_index("arm")
    rows = []
    for n, k in vel:
        r = Es.loc["A6_state_plus_unspliced"] if k is None else E.loc[k]
        rows.append((n, r["diff"], r["lo"], r["hi"], VEL, False))
    I10 = pd.read_csv(R + "incr_all_raw_matrix_vs_velocity.csv").set_index("arm")
    I1 = pd.read_csv(R + "nm_incr_raw_matrix_vs_velocity.csv").set_index("arm")
    snap = []
    for n, k in (("StationaryOT", "statot"), ("Waddington-OT", "wot"),
                 ("moscot", "moscot"), ("PRESCIENT", "prescient")):
        r = I10.loc["K_snap_" + k]; snap.append((n + ", 10 fates", r["diff"], r["lo"], r["hi"], OT, False))
        r = I1.loc["K_nm_snap_" + k]; snap.append((n + ", fate bias", r["diff"], r["lo"], r["hi"], OT, True))
    fig = plt.figure(figsize=(160 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 2, wspace=1.0)
    ax = fig.add_subplot(gs[0]); letter(ax, "a")
    rows_plot(ax, rows, "AUROC change, concatenated", "RNA velocity family (base: smoothed counts)")
    ax = fig.add_subplot(gs[1]); letter(ax, "b")
    rows_plot(ax, snap, "AUROC change, concatenated", "OT and potential-based methods (base: total counts)")
    save(fig, "ed1")


def ed2():
    L = pd.read_csv(R + "leak/stk_leak_tot_leakdiff.csv").set_index("stage")
    Pp = pd.read_csv(R + "stk3nn_parts_pairs.csv").set_index("arm")
    Pc = pd.read_csv(R + "stk3nn_conn_pairs.csv").set_index("arm")
    fig = plt.figure(figsize=(110 * MM, 125 * MM))
    gs = fig.add_gridspec(3, 1, hspace=0.9, height_ratios=[0.6, 1, 1])
    ax = fig.add_subplot(gs[0]); letter(ax, "a")
    rows = [("Alone", L.loc["S1", "diff"], L.loc["S1", "lo"], L.loc["S1", "hi"], OT, False),
            ("Added to total counts", L.loc["ST", "diff"], L.loc["ST", "lo"], L.loc["ST", "hi"], OT, False)]
    rows_plot(ax, rows, "AUROC, without minus with\nheld-out clones' later cells", "Held-out descendants removed")
    ax = fig.add_subplot(gs[1]); letter(ax, "b")
    rows = [("All 10 fates", *Pp.loc["snap_statot", ["mean10", "lo_avg_part", "hi_avg_part"]], OT, False),
            ("Neutrophil + monocyte columns", *Pp.loc["snap_statot_nm2", ["mean10", "lo_avg_part", "hi_avg_part"]], OT, True),
            ("Other 8 fates", *Pp.loc["snap_statot_other8", ["mean10", "lo_avg_part", "hi_avg_part"]], OT, True),
            ("Fate bias (one column)", *Pp.loc["nm_snap_statot", ["mean10", "lo_avg_part", "hi_avg_part"]], OT, True)]
    rows_plot(ax, rows, "AUROC increment over expression model", "StationaryOT, columns split")
    ax = fig.add_subplot(gs[2]); letter(ax, "c")
    rows = [("StationaryOT, 10 fates", *Pp.loc["snap_statot", ["mean10", "lo_avg_part", "hi_avg_part"]], OT, False)]
    for k, n in (("cr_conn_n30_p15_d1", "Connectivity kernel, best alone"),
                 ("cr_conn_n60_p50_d0", "Connectivity kernel, best on first repetition"),
                 ("cr_conn_n60_p30_d0", "Connectivity kernel, third setting")):
        rows.append((n, *Pc.loc[k, ["mean10", "lo_avg_part", "hi_avg_part"]], CTRL, False))
    rows_plot(ax, rows, "AUROC increment over expression model", "Against connectivity kernels")
    save(fig, "ed2")


def ed3():
    G = pd.read_csv(R + "cross_graph.csv").set_index("arm")
    W = pd.read_csv(R + "wp_d2_pairs.csv")
    lab = {"A5_unspliced_only": "Unspliced counts alone", "V4a_cr_velo_only": "CellRank velocity kernel alone",
           "V1_smoothed_plus_vel_det": "Smoothed + steady-state", "V2_smoothed_plus_vel_dyn": "Smoothed + dynamical",
           "V3_smoothed_plus_vel_velovi": "Smoothed + veloVI", "A4_state_smoothed": "Smoothed counts",
           "A2_state_spliced": "Spliced counts (no graph)"}
    rows = [(lab[k], G.loc[k, "diff"], G.loc[k, "lo"], G.loc[k, "hi"], VEL if k not in ("A4_state_smoothed", "A2_state_spliced") else EXPR, False)
            for k in lab if k in G.index]
    fig = plt.figure(figsize=(170 * MM, 60 * MM))
    gs = fig.add_gridspec(1, 2, wspace=1.0)
    ax = fig.add_subplot(gs[0]); letter(ax, "a")
    rows_plot(ax, rows, "AUROC, all-time-point graph minus day-2 graph", "Graph construction")
    ax = fig.add_subplot(gs[1]); letter(ax, "b")
    nm = {"A4_state_smoothed": "Smoothed counts", "A5_unspliced_only": "Unspliced alone",
          "A6_state_plus_unspliced": "Spliced + unspliced", "A7_smoothed_plus_ratio": "Smoothed + ratio",
          "V1_smoothed_plus_vel_det": "Smoothed + steady-state", "V2_smoothed_plus_vel_dyn": "Smoothed + dynamical",
          "V3_smoothed_plus_vel_velovi": "Smoothed + veloVI", "V4_smoothed_plus_cr_velo": "Smoothed + velocity kernel",
          "V5_smoothed_plus_cr_pseudo": "Smoothed + pseudotime kernel"}
    rows = []
    for pop in ("LSK", "LK"):
        for k, n in nm.items():
            r = W[(W["pop"] == pop) & (W.arm == k) & (W.ref == "A2_state_spliced")]
            if len(r):
                r = r.iloc[0]
                rows.append(("%s, %s" % (pop, n), r["diff"], r["lo"], r["hi"], VEL if k.startswith("V") or "unspliced" in k else EXPR, pop == "LK"))
    rows_plot(ax, rows, "AUROC vs spliced counts, within population", "Within each sorted population")
    save(fig, "ed3")


def ed4():
    A = pd.read_csv(R + "stk3nn_all_tot_strata.csv")
    D = pd.read_csv(R + "stk3nn_d2_tot_strata.csv")
    show = [("veloVI", D, "vel_velovi", VEL), ("Dynamical velocity", D, "vel_dyn", VEL),
            ("CellRank pseudotime kernel", D, "cr_pseudo", VEL),
            ("StationaryOT, 10 fates", A, "snap_statot", OT), ("Waddington-OT, 10 fates", A, "snap_wot", OT),
            ("Spliced counts", D, "raw_spliced", EXPR), ("Unspliced counts", D, "raw_unspliced", VEL),
            ("Synthetic, alone 0.85", A, "cal_a0.5_s0", SYN), ("Synthetic, alone 0.87", A, "cal_a0.5_s1", SYN),
            ("Random number", A, "ctl_rand", SYN)]
    fig = plt.figure(figsize=(150 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.15)
    for j, (st, title) in enumerate((("less_sure_half", "Expression model less confident (284 clones)"),
                                     ("surer_half", "Expression model more confident (283 clones)"))):
        ax = fig.add_subplot(gs[j]); letter(ax, "ab"[j])
        rows = []
        for n, T, k, c in show:
            r = T[(T.stratum == st) & (T.arm == k)].iloc[0]
            rows.append((n, r["mean10"], r["lo"], r["hi"], c, False))
        XM = 0.06
        for i, (n, x, lo, hi, c, h) in enumerate(rows):
            if x > XM:
                ax.plot([lo, XM], [i, i], color=c, lw=0.9)
                ax.plot([XM], [i], ">", ms=3.2, color=c)
                ax.text(XM - 0.002, i - 0.32, "%.2f [%.2f, %.2f]" % (x, lo, hi), fontsize=4.8,
                        ha="right", va="bottom", color="#52514e")
            else:
                hdot(ax, i, x, lo, min(hi, XM), c, h)
        ax.axvline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows] if j == 0 else [])
        ax.invert_yaxis(); ax.set_xlabel("AUROC increment over expression model")
        ax.set_title(title, loc="left", fontsize=7)
        ax.set_xlim(-0.03, XM + 0.004)
    save(fig, "ed4")


def ed5():
    """Positive control (74_positive_control.py): with fate-carrying dynamics the
    velocity kernel beats the connectivity kernel and adds to expression, and
    forward-minus-reverse CBDir separates the true velocity from the kNN-mean
    displacement, which forward CBDir alone ranks the wrong way round."""
    P = pd.read_csv(R + "positive_control.csv")
    G = P.groupby("delta")
    d = np.array(sorted(P["delta"].unique()))
    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.75, hspace=0.7, width_ratios=[1, 1, 0.9])
    ax = fig.add_subplot(gs[0, 0]); letter(ax, "a")
    for col, name, c, mk in (("auroc_velocity_kernel", "velocity kernel", VEL, "o"),
                             ("auroc_connectivity_kernel", "connectivity kernel", CTRL, "s"),
                             ("auroc_expression", "expression (20 PCs)", EXPR, "^")):
        ax.plot(P["delta"], P[col], mk, color=c, ms=2.2, alpha=0.5)
        ax.plot(d, G[col].mean().loc[d], "-", color=c, lw=0.9, label=name)
    ax.axhline(0.5, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("Fate component of the true velocity, δ")
    ax.set_ylabel("AUROC for the true fate (progenitors)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0.47, 0.76)
    ax.set_title("Stand-alone", loc="left", fontsize=7)
    ax = fig.add_subplot(gs[0, 1]); letter(ax, "b")
    for col, lo, hi, name, c in (("vk_minus_ck", "vk_minus_ck_lo", "vk_minus_ck_hi",
                                  "velocity minus connectivity\nkernel", VEL),
                                 ("increment", "increment_lo", "increment_hi",
                                  "velocity kernel added\nto expression", OT)):
        for _, r in P.iterrows():
            ax.plot([r["delta"]] * 2, [r[lo], r[hi]], color=c, lw=0.6, alpha=0.6)
        ax.plot(d, G[col].mean().loc[d], "o-", color=c, ms=2.6, lw=0.9, label=name)
    ax.axhline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("Fate component of the true velocity, δ")
    ax.set_ylabel("AUROC difference")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5)
    ax.set_title("Reference and increment", loc="left", fontsize=7)
    ax = fig.add_subplot(gs[0, 2]); letter(ax, "c")
    rows = [("True velocity", "true_velocity", VEL), ("Neighbour\naverage", "knn_mean", CTRL),
            ("Random", "random", SYN)]
    for i, (n, k, c) in enumerate(rows):
        f, rv = P["cbdir_fwd_" + k].mean(), P["cbdir_rev_" + k].mean()
        ax.plot([f, rv], [i, i], color=c, lw=0.7)
        ax.plot([f], [i], "o", ms=3.4, color=c)
        ax.plot([rv], [i], "o", ms=3.4, mfc="white", mec=c, mew=0.9)
        ax.text(1.0, i - 0.28, "%+.2f" % (f - rv), fontsize=5.5, va="center", ha="right")
    ax.text(1.0, -0.75, "fwd − rev", fontsize=5.5, ha="right", va="center")
    ax.set_ylim(2.5, -1.1)
    ax.axvline(0, color="#6b6b68", lw=0.5)
    ax.set_xlim(-0.25, 1.0)
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("CBDir (filled, forward; open, reverse)")
    ax.set_title("Direction", loc="left", fontsize=7)
    # d, e: count-level control (78_kinetic_control.py): fate genes switched on just
    # before sampling, counts at LARRY's depth or deeper, the full velocity pipeline.
    K = pd.concat([pd.read_csv(R + f) for f in ("kinetic_control.csv", "kinetic_control_primed16.csv")
                   if os.path.exists(R + f)])
    K = K[K.priming.isin([0.0, 0.16])]
    for j, (cols, title, yl) in enumerate((
            ((("auroc_velocity_kernel", "velocity kernel", VEL), ("auroc_connectivity_kernel", "connectivity kernel", CTRL),
              ("auroc_expression", "expression", EXPR)), "Counts: stand-alone", "AUROC for the fate (scored cells)"),
            ((("increment", "velocity kernel added", VEL), ("increment_unspliced", "unspliced layer added", OT)),
             "Counts: added to expression", "AUROC increment over expression"))):
        ax = fig.add_subplot(gs[1, j]); letter(ax, "de"[j])
        for pr, ls in ((0.0, "-"), (0.16, "--")):
            Kp = K[K.priming == pr]
            if Kp.empty:
                continue
            g = Kp.groupby("depth").mean(numeric_only=True)
            for col, name, c in cols:
                ax.plot(Kp["depth"], Kp[col], "o", ms=2, color=c, alpha=0.4)
                ax.plot(g.index, g[col], ls, color=c, lw=0.9,
                        label=name if pr == 0.0 else None)
        if j == 1:
            ax.axhline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
        ax.set_xscale("log"); ax.set_xticks([1, 5, 20]); ax.set_xticklabels(["1×", "5×", "20×"])
        ax.minorticks_off()
        ax.set_xlabel("Depth (1× = LARRY: 650 spliced,\n120 unspliced molecules per cell)")
        ax.set_ylabel(yl)
        ax.legend(frameon=False, fontsize=5.2, loc="center right" if j == 0 else "upper left")
        ax.set_title(title + " (solid, no priming;\ndashed, primed state)", loc="left", fontsize=6.5)
    # f: forward minus reverse CBDir in the count-level simulation
    ax = fig.add_subplot(gs[1, 2]); letter(ax, "f")
    for pr, ls in ((0.0, "-"), (0.16, "--")):
        Kp = K[K.priming == pr]
        if Kp.empty:
            continue
        for fld, name, c in (("velocity", "estimated velocity", VEL), ("knn_mean", "neighbour average", CTRL)):
            v = Kp["cbdir_fwd_" + fld] - Kp["cbdir_rev_" + fld]
            g = v.groupby(Kp["depth"]).mean()
            ax.plot(Kp["depth"], v, "o", ms=2, color=c, alpha=0.4)
            ax.plot(g.index, g.values, ls, color=c, lw=0.9, label=name if pr == 0.0 else None)
    ax.axhline(0, color="#6b6b68", lw=0.5, ls=(0, (2, 2)))
    ax.set_xscale("log"); ax.set_xticks([1, 5, 20]); ax.set_xticklabels(["1×", "5×", "20×"])
    ax.minorticks_off()
    ax.set_xlabel("Depth")
    ax.set_ylabel("CBDir, forward minus reverse")
    ax.legend(frameon=False, fontsize=5.2, loc="center right")
    ax.set_title("Counts: direction", loc="left", fontsize=6.5)
    save(fig, "ed5")


def ed6():
    """CBDir forward/reverse and ICCoh in 30 principal components
    (71_direction_embedding.py, second run), with clone-resampled intervals."""
    E = pd.read_csv(R + "direction_embedding_bench.csv")
    E = E[E.space == "pca30"].set_index("field")
    fields = [("steady", "Steady-state velocity", VEL, False),
              ("dynamical", "Dynamical velocity", VEL, False),
              ("velovi", "veloVI", VEL, False),
              ("perm_control", "Shuffled unspliced", CTRL, False),
              ("minus_gamma_s", "Degradation only (−γs)", CTRL, True),
              ("knn_mean_in_pca", "Neighbour average", CTRL, False),
              ("knn_mean", "Neighbour average, gene space", CTRL, True),
              ("centroid_in_pca", "Towards all mature cells", "#8a8a86", False),
              ("nearest_mature_centroid", "Into nearest mature population", "#5a5a57", False),
              ("random", "Random vectors", SYN, True)]
    fig = plt.figure(figsize=(160 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.12)
    for j, (col, title, xl) in enumerate((("cbdir_fwd", "Cross-boundary direction (30 PCs)", "CBDir"),
                                          ("iccoh", "In-cluster coherence (30 PCs)", "ICCoh"))):
        ax = fig.add_subplot(gs[j]); letter(ax, "ab"[j])
        for i, (k, n, c, h) in enumerate(fields):
            v = E.loc[k, col]
            if not np.isfinite(v):
                continue
            if col == "cbdir_fwd":
                rv = E.loc[k, "cbdir_rev"]
                ax.plot([v, rv], [i, i], color=c, lw=0.7)
                ax.plot([v], [i], "o", ms=3.4, mfc=c, mec=c, mew=0.9)
                ax.plot([rv], [i], "o", ms=3.4, mfc="white", mec=c, mew=0.9)
                ax.text(1.28, i, "%+.3f [%+.3f, %+.3f]" % (E.loc[k, "fwd_minus_rev"], E.loc[k, "diff_lo"],
                                                          E.loc[k, "diff_hi"]),
                        fontsize=4.8, va="center", ha="right")
            else:
                ax.plot([0, v], [i, i], color=c, lw=0.9)
                ax.plot([v], [i], "o", ms=3.2, mfc="white" if h else c, mec=c, mew=0.9)
        if col == "cbdir_fwd":
            ax.text(1.28, -0.9, "fwd − rev [clone-resampled 95% interval]", fontsize=4.8,
                    ha="right", va="center")
            ax.set_xlim(-0.15, 1.3); ax.set_ylim(len(fields) - 0.4, -1.3)
        else:
            ax.invert_yaxis()
        ax.axvline(0, color="#6b6b68", lw=0.5)
        ax.set_yticks(range(len(fields)))
        ax.set_yticklabels([f[1] for f in fields] if j == 0 else [])
        ax.set_xlabel(xl + (" (filled, forward; open, reverse)" if j == 0 else ""))
        ax.set_title(title, loc="left", fontsize=7)
    save(fig, "ed6")


def ed9():
    """Direction metrics with null fields on LARRY and on four standard velocity
    benchmarks (71_direction_embedding.py, 84_benchmark_direction.py). Rows: data
    sets; columns: forward CBDir, forward minus reverse (with 95% intervals) and
    ICCoh; top row in the data set's own embedding, bottom row in 30 PCs."""
    B = pd.read_csv(R + "benchmark_direction_bench.csv")
    L = pd.read_csv(R + "direction_embedding_bench.csv")
    L = L.rename(columns={"fwd_lo": "_fl", "fwd_hi": "_fh"})
    L["dataset"] = "LARRY (in vitro)"
    L["space"] = L["space"].replace({"umap": "embedding"})
    L["field"] = L["field"].replace({"knn_mean_in_umap": "knn_mean_in_embedding"})
    D = pd.concat([L, B], ignore_index=True)
    D.loc[D.field == "knn_mean_in_pca", "field"] = "knn_mean_in_embedding_pca"
    sets = ["LARRY (in vitro)", "pancreas", "dentate gyrus", "gastrulation erythroid", "bone marrow"]
    labels = ["LARRY", "Pancreas", "Dentate gyrus", "Gastrulation\nerythroid", "Bone marrow"]
    fields = [("steady", "Steady-state velocity", VEL, "o", True),
              ("dynamical", "Dynamical velocity", VEL, "s", True),
              ("velovi", "veloVI", VEL, "D", True),
              ("perm_control", "Shuffled unspliced", CTRL, "o", False),
              ("minus_gamma_s", "Degradation only (−γs)", CTRL, "v", True),
              ("knn", "Neighbour average", EXPR, "^", False),
              ("random", "Random vectors", SYN, "x", True)]
    off = np.linspace(-0.3, 0.3, len(fields))
    cols = [("cbdir_fwd", "Forward CBDir"), ("fwd_minus_rev", "Forward − reverse CBDir"),
            ("iccoh", "ICCoh")]
    fig = plt.figure(figsize=(180 * MM, 118 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.32)
    for r, (space, sname) in enumerate((("embedding", "Embedding"),
                                         ("pca30", "30 principal components"))):
        for c, (col, cname) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c]); letter(ax, "abcdef"[3 * r + c])
            for i, ds in enumerate(sets):
                if i % 2 == 0:
                    ax.axhspan(i - 0.45, i + 0.45, color="#f2f2f0", lw=0, zorder=0)
                S = D[(D.dataset == ds) & (D.space == space)].set_index("field")
                knn = "knn_mean_in_embedding" if space == "embedding" else "knn_mean_in_embedding_pca"
                for (k, n, colr, m, filled), o in zip(fields, off):
                    kk = (("knn_mean" if (col == "iccoh" and space == "embedding") else knn) if k == "knn" else k)
                    if kk not in S.index:
                        continue
                    v = S.loc[kk, col]
                    y = i + o
                    if col == "fwd_minus_rev":
                        ax.plot([S.loc[kk, "diff_lo"], S.loc[kk, "diff_hi"]], [y, y], color=colr,
                                lw=1.0, zorder=2)
                    ax.plot([v], [y], m, ms=3.0 if m != "x" else 3.2,
                            mfc=colr if filled else "white", mec=colr, mew=0.7, zorder=3,
                            label=n if (i == 0 and r == 0 and c == 0) else None)
            ax.axvline(0, color="#6b6b68", lw=0.5, zorder=1)
            ax.set_yticks(range(len(sets)))
            ax.set_yticklabels(labels if c == 0 else [])
            ax.set_ylim(len(sets) - 0.5, -0.5)
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel(cname)
            ax.set_title("%s, %s" % (["Forward CBDir", "Reverse score", "Coherence"][c],
                                     "30 PCs" if space == "pca30" else "embedding"),
                         loc="left", fontsize=7)
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower center", ncol=7,
               frameon=False, fontsize=5.6, bbox_to_anchor=(0.5, -0.035), handletextpad=0.3,
               columnspacing=1.0)
    save(fig, "ed9")
    # counts quoted in the text, written beside the figure
    rows = []
    for space in ("embedding", "pca30"):
        knn = "knn_mean_in_embedding" if space == "embedding" else "knn_mean_in_embedding_pca"
        for ds in sets:
            S = D[(D.dataset == ds) & (D.space == space)].set_index("field")
            vel = S.loc[["steady", "dynamical", "velovi"]]
            rows.append(dict(dataset=ds, space=space,
                             best_velocity_fwd=vel.cbdir_fwd.max(), knn_fwd=S.loc[knn, "cbdir_fwd"],
                             knn_fwd_above_all_velocities=S.loc[knn, "cbdir_fwd"] > vel.cbdir_fwd.max(),
                             best_velocity_fmr=vel.fwd_minus_rev.max(),
                             degr_fmr=S.loc["minus_gamma_s", "fwd_minus_rev"],
                             knn_fmr=S.loc[knn, "fwd_minus_rev"],
                             degr_fmr_above_all_velocities=S.loc["minus_gamma_s", "fwd_minus_rev"]
                             > vel.fwd_minus_rev.max(),
                             best_velocity_iccoh=vel.iccoh.max(), degr_iccoh=S.loc["minus_gamma_s", "iccoh"],
                             degr_iccoh_above_all_velocities=S.loc["minus_gamma_s", "iccoh"] > vel.iccoh.max()))
    T = pd.DataFrame(rows).round(3)
    T.to_csv(R + "benchmark_direction_summary.csv", index=False)
    print(T.to_string(index=False))


for f in [globals()[n] for n in (sys.argv[2].split(",") if len(sys.argv) > 2 else
                                 ["ed5"])]:  # the other Extended Data figures are in 62_figures.py
    try:
        f()
    except Exception:
        import traceback; traceback.print_exc()
