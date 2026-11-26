"""Extended Data Table 4: every stacked increment, written from the result files.

One row per output, one column per expression model; each cell is the mean
increment over ten repetitions with its 95% interval (paired bootstrap averaged
over repetitions, widened by the partition term), exactly as
56_stacked_increment.py writes them (mean10, lo_avg_part, hi_avg_part). Nothing
is typed in.

Usage: 75_tables.py <outdir_of_results> <out.tex>
"""
import sys

import pandas as pd

R, OUT = sys.argv[1], sys.argv[2]


def load(prefix):
    return pd.read_csv("%s/%s_pairs.csv" % (R, prefix)).set_index("arm")


def cell(P, arm):
    if arm not in P.index:
        return "--"
    r = P.loc[arm]

    def f(x):
        s = "%+.3f" % x
        if s in ("+0.000", "-0.000"):
            s = "0.000"
        return "$%s$" % s
    out = "%s [%s, %s]" % (f(r["mean10"]), f(r["lo_avg_part"]), f(r["hi_avg_part"]))
    # intervals excluding zero (judged on unrounded values) in bold
    if r["lo_avg_part"] > 0 or r["hi_avg_part"] < 0:
        out = "\\textbf{%s}" % out.replace("$", "")
        out = "$\\mathbf{%s}$" % out[len("\\textbf{"):-1]
    return out


blocks = [
    ("Splicing-based outputs, day-2 graph",
     [("stk3nn_d2_tot", "Total"), ("stk3nn_d2_ens", "Stacked"),
      ("stk3nn_d2_Ms", "Smoothed"), ("stk3nn_d2_spl", "Spliced")],
     [("raw_unspliced", "Unspliced counts"), ("vel_det", "Steady-state velocity"),
      ("vel_dyn", "Dynamical velocity"), ("vel_velovi", "veloVI"),
      ("cr_velo", "CellRank velocity kernel, 3 fates"),
      ("cr_pseudo", "CellRank pseudotime kernel, 3 fates")]),
    ("Outputs on the graph of all cells",
     [("stk3nn_all_tot", "Total"), ("stk3nn_all_ens", "Stacked"),
      ("stk3nn_all_spl", "Spliced")],
     [("nm_snap_statot", "StationaryOT, fate bias"), ("snap_statot", "StationaryOT, 10 fates"),
      ("nm_snap_wot", "Waddington-OT, fate bias"), ("snap_wot", "Waddington-OT, 10 fates"),
      ("nm_snap_moscot", "moscot, fate bias"), ("snap_moscot", "moscot, 10 fates"),
      ("nm_snap_prescient", "PRESCIENT, fate bias"), ("snap_prescient", "PRESCIENT, 10 fates"),
      ("nm_cr_velo", "CellRank velocity kernel, fate bias"),
      ("cr_velo", "CellRank velocity kernel, 10 fates"),
      ("nm_cr_pseudo", "CellRank pseudotime kernel, fate bias"),
      ("cr_pseudo", "CellRank pseudotime kernel, 10 fates"),
      ("nm_cr_conn_n60_p30_d0", "Connectivity kernel, fate bias"),
      ("cr_conn_n60_p30_d0", "Connectivity kernel, 10 fates"),
      ("ctl_rand", "Negative control: random column"),
      ("ctl_libsize", "Negative control: library size")]),
    ("Method variants (graph of all cells)",
     [("stk3nn_configs", "Total")],
     [("nm_snap_rtk_ub", "Real-time kernel, fate bias"), ("snap_rtk_ub", "Real-time kernel, 10 fates"),
      ("nm_cr_vkck", "Velocity + connectivity kernel, fate bias"),
      ("cr_vkck", "Velocity + connectivity kernel, 10 fates"),
      ("nm_snap_moscot_ub", "moscot unbalanced, fate bias"), ("snap_moscot_ub", "moscot unbalanced, 10 fates")]),
    ("Connectivity kernel, further settings",
     [("stk3nn_conn", "Total")],
     [("cr_conn_n30_p15_d1", "30 neighbours, 15 PCs, normalized, 10 fates"),
      ("nm_cr_conn_n30_p15_d1", "30 neighbours, 15 PCs, normalized, fate bias"),
      ("cr_conn_n60_p50_d0", "60 neighbours, 50 PCs, 10 fates"),
      ("nm_cr_conn_n60_p50_d0", "60 neighbours, 50 PCs, fate bias")]),
    ("Cytokine experiment (total counts)",
     [("stk3nn_cyto", "Total")],
     [("nm_snap_statot", "StationaryOT, fate bias"), ("snap_statot", "StationaryOT, 10 fates"),
      ("nm_snap_wot", "Waddington-OT, fate bias"), ("snap_wot", "Waddington-OT, 10 fates"),
      ("nm_snap_moscot", "moscot, fate bias"), ("snap_moscot", "moscot, 10 fates"),
      ("nm_snap_prescient", "PRESCIENT, fate bias"), ("snap_prescient", "PRESCIENT, 10 fates")]),
]

def render(sel):
    """One tabular for the blocks with the given indices."""
    L = ["\\setlength{\\tabcolsep}{1.5pt}", "\\begin{tabular}{p{2.0cm}llll}", "\\toprule"]
    for title, runs, arms in [blocks[k] for k in sel]:
        P = {k: load(k) for k, _ in runs}
        head = " & ".join(["\\textit{%s}" % title] + [h for _, h in runs]
                          + [""] * (4 - len(runs)))
        L.append(head + " \\\\")
        L.append("\\midrule")
        for arm, name in arms:
            L.append(" & ".join([name] + [cell(P[k], arm) for k, _ in runs]
                                + [""] * (4 - len(runs))) + " \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    return "\n".join(L) + "\n"


# part a: splicing-based outputs, method variants, connectivity-kernel settings;
# part b: outputs on the graph of all cells, and the cytokine experiment
open(OUT, "w").write(render([0, 2, 3]))
open(OUT.replace("ed_table4", "ed_table4b"), "w").write(render([1, 4]))
print(render([0, 2, 3]))

# ---------------------------------------------------------------- Extended Data Table 5
# Each output's stand-alone AUROC through the classifier (ten-repetition mean of the
# stacking run's first stage), its increment over total counts, and what an
# independent signal of the same stand-alone AUROC would add under the same stacking:
# the ten-column synthetic curve interpolated at that AUROC and divided by the
# rescaling factor of 60_yardstick_sim.py (largest and smallest ratio over targets).
OUT5 = OUT.replace("ed_table4", "ed_table5")
runs5 = [("stk3nn_d2_tot", [("raw_unspliced", "Unspliced counts"), ("vel_det", "Steady-state velocity"),
                            ("vel_dyn", "Dynamical velocity"), ("vel_velovi", "veloVI"),
                            ("cr_velo", "Velocity kernel, day-2 graph, 3 fates"),
                            ("cr_pseudo", "Pseudotime kernel, day-2 graph, 3 fates")]),
         ("stk3nn_all_tot", [("nm_snap_statot", "StationaryOT, fate bias"), ("snap_statot", "StationaryOT, 10 fates"),
                             ("nm_snap_wot", "Waddington-OT, fate bias"), ("snap_wot", "Waddington-OT, 10 fates"),
                             ("nm_snap_moscot", "moscot, fate bias"), ("snap_moscot", "moscot, 10 fates"),
                             ("nm_snap_prescient", "PRESCIENT, fate bias"), ("snap_prescient", "PRESCIENT, 10 fates"),
                             ("nm_cr_velo", "Velocity kernel, fate bias"), ("cr_velo", "Velocity kernel, 10 fates"),
                             ("nm_cr_pseudo", "Pseudotime kernel, fate bias"), ("cr_pseudo", "Pseudotime kernel, 10 fates")]),
         ("stk3nn_configs", [("nm_snap_rtk_ub", "Real-time kernel, fate bias"), ("snap_rtk_ub", "Real-time kernel, 10 fates"),
                             ("nm_cr_vkck", "Velocity + connectivity, fate bias"), ("cr_vkck", "Velocity + connectivity, 10 fates"),
                             ("nm_snap_moscot_ub", "moscot unbalanced, fate bias"), ("snap_moscot_ub", "moscot unbalanced, 10 fates")])]
Sall = pd.read_csv("%s/stk3nn_all_tot_summary.csv" % R).set_index("arm")["auroc_mean"]
Pall = load("stk3nn_all_tot")
cal = sorted((Sall["S1_" + k], Pall.loc[k, "mean10"]) for k in Pall.index if k.startswith("cal_a"))
cx = [c[0] for c in cal]; cy = [c[1] for c in cal]
Y = pd.read_csv("%s/yardstick_sim.csv" % R).pivot(index="target", columns="built", values="increment")
ratio = (Y["from_f"] / Y["from_pc"]).values
import numpy as np
rows5 = []
for run, arms in runs5:
    S = pd.read_csv("%s/%s_summary.csv" % (R, run)).set_index("arm")["auroc_mean"]
    P = load(run)
    for arm, name in arms:
        a = S["S1_" + arm]
        e = max(0.0, float(np.interp(a, cx, cy)))
        rows5.append(dict(name=name, alone=a, inc=P.loc[arm, "mean10"], lo=P.loc[arm, "lo_avg_part"],
                          hi=P.loc[arm, "hi_avg_part"], exp_lo=e / ratio.max(), exp_hi=e / ratio.min()))
T5 = pd.DataFrame(rows5).sort_values("alone")
# one criterion for every output: does the upper end of its interval lie below the
# smallest increment an independent signal of the same strength would give?
T5["excludes_independent"] = T5["hi"] < T5["exp_lo"]
# The comparison is informative only where that increment exceeds what the
# output's own interval could detect: the minimum detectable increment, 1.43
# times the half-width of the interval (80% power at a two-sided 5% level).
T5["mdi"] = 1.43 * (T5["hi"] - T5["lo"]) / 2
T5["informative"] = T5["exp_lo"] > T5["mdi"]
T5.to_csv(OUT5.replace(".tex", ".csv"), index=False)
L = ["\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{p{3.3cm}rrrrc}", "\\toprule",
     "Output & AUROC alone & Increment [95\\% interval] & Detectable & Independent signal & Excluded \\\\",
     "\\midrule"]
for _, r in T5.iterrows():
    f = lambda x: ("$%+.3f$" % x).replace("+0.000", "0.000").replace("-0.000", "0.000")
    L.append("%s & %.3f & %s [%s, %s] & %.3f & %.3f--%.3f & %s \\\\" % (
        r["name"], r["alone"], f(r["inc"]), f(r["lo"]), f(r["hi"]), r["mdi"], r["exp_lo"], r["exp_hi"],
        ("yes" if r["excludes_independent"] else "no") if r["informative"] else "--"))
L += ["\\bottomrule", "\\end{tabular}"]
open(OUT5, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("independent-signal increment excluded for %d of %d informative outputs (%d outputs)"
      % ((T5.excludes_independent & T5.informative).sum(), T5.informative.sum(), len(T5)))
