"""Extended Data Table 6: every method against every connectivity-kernel setting.

For each method's fate bias (direct scoring, no classifier), the number of the 18
connectivity-kernel settings at which the method is significantly above / below the
kernel (paired clone bootstrap, 95%), by Pearson r, Spearman correlation and AUROC,
against the kernel on the graph of all cells (67_reference_intervals.py,
refint_invitro2) and on day-2 cells only (refint_vs_d2), with the method's own scores.

Usage: 81_paired_table.py <outdir> <out.tex>
"""
import sys

import pandas as pd

R, OUT = sys.argv[1], sys.argv[2]
A = pd.read_csv(R + "/refint_invitro2_pairs.csv")
D = pd.read_csv(R + "/refint_vs_d2_pairs.csv")
names = [("nm_snap_wot", "Waddington-OT"), ("nm_snap_moscot", "moscot"),
         ("nm_snap_prescient", "PRESCIENT"), ("nm_snap_statot", "StationaryOT"),
         ("nm_snap_rtk_ub", "Real-time kernel"), ("nm_cr_velo", "Velocity kernel"),
         ("nm_cr_pseudo", "Pseudotime kernel"), ("nm_cr_vkck", "Velocity + connectivity")]


def counts(P, m, k):
    x = P[(P.method == m) & P.reference.str.startswith("nm_cr_conn_")]
    return "%d/%d" % ((x[k + "_lo"] > 0).sum(), (x[k + "_hi"] < 0).sum())


L = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lcccccccc}", "\\toprule",
     " & \\multicolumn{3}{c}{Method's score} & \\multicolumn{3}{c}{vs kernel, all cells}"
     " & \\multicolumn{2}{c}{vs kernel, day-2 cells} \\\\",
     "Method & $r$ & Spearman & AUROC & $r$ & Spearman & AUROC & $r$ & AUROC \\\\", "\\midrule"]
for m, n in names:
    x = A[A.method == m].iloc[0]
    L.append("%s & %.3f & %.3f & %.3f & %s & %s & %s & %s & %s \\\\" % (
        n, x["r_method"], x["rho_method"], x["auroc_method"],
        counts(A, m, "r"), counts(A, m, "rho"), counts(A, m, "auroc"), counts(D, m, "r"), counts(D, m, "auroc")))
c = A[A.reference.str.startswith("nm_cr_conn_")].drop_duplicates("reference")
d = D[D.reference.str.startswith("nm_cr_conn_")].drop_duplicates("reference")
L.append("\\midrule")
L.append("Kernel, all cells (18) & %.3f--%.3f & %.3f--%.3f & %.3f--%.3f & & & & & \\\\" % (
    c.r_ref.min(), c.r_ref.max(), c.rho_ref.min(), c.rho_ref.max(), c.auroc_ref.min(), c.auroc_ref.max()))
L.append("Kernel, day-2 cells (18) & %.3f--%.3f & %.3f--%.3f & %.3f--%.3f & & & & & \\\\" % (
    d.r_ref.min(), d.r_ref.max(), d.rho_ref.min(), d.rho_ref.max(), d.auroc_ref.min(), d.auroc_ref.max()))
L += ["\\bottomrule", "\\end{tabular}"]
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
sp = A[A.reference.str.startswith("nm_cr_conn_")]
print("Spearman above/below:", {m: counts(A, m, "rho") for m, _ in names})
