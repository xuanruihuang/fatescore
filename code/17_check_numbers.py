"""Check every number quoted in the manuscript against the file it comes from.

Usage: 17_check_numbers.py <outdir> <paper_dir>

Each entry names a quantity, where it lives, and the value the text gives. The
script fails if any quoted value differs from the generated one by more than
the rounding the text uses, and also reports any entry whose source file is
missing. It does not parse the LaTeX; the point is that the table below is
maintained by hand and verified mechanically, so a number can only drift if
someone edits both places.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

OUT, PAPER = sys.argv[1], sys.argv[2]
fails, missing = [], []


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        missing.append(name)
        return None
    return pd.read_csv(p, sep="\t" if name.endswith(".tsv") else ",")


def check(label, got, want, tol):
    if got is None:
        return
    if not np.isfinite(got):
        fails.append("%s: generated value is not finite" % label)
    elif abs(got - want) > tol:
        fails.append("%s: text says %.4g, files say %.4g" % (label, want, got))
    else:
        print("ok   %-58s %.4g" % (label, got))


def arm(df, a, col):
    if df is None:
        return None
    s = df[df.iloc[:, 0] == a]
    return float(s[col].iloc[0]) if len(s) else None


def pair(df, a, ref, col="diff"):
    if df is None:
        return None
    s = df[(df["arm"] == a) & (df["ref"] == ref)]
    return float(s[col].iloc[0]) if len(s) else None


def quant(df, q, col="value"):
    if df is None:
        return None
    s = df[df["quantity"] == q]
    return float(s[col].iloc[0]) if len(s) else None


dg = load("01_design.csv")
pc = load("pc_d2_calibration.csv")
sm = load("sm_d2.csv")
ev = load("ev_d2_summary.csv")
pr = load("ev_d2_pairs.csv")
rs = load("rs_d2.csv")
dn = load("dn_d2_summary.csv")
fa = load("fa_d2_summary.csv")
ta = load("ta_d2.csv")
uf = load("ufrac_d2.csv")
wp = load("wp_d2_pairs.csv")
bb = load("bc_d2_taskB_summary.csv")
ctev = load("ctev_d2_summary.csv")
ctrs20 = load("ctrs20_d2.csv")
ct1 = load("ctrs_tc1.csv")
ct2 = load("ctrs_tc2.csv")
ctsm = load("ctsm_d2.csv")
ctdep = load("ctdn_d2_depths.csv")
ctce = load("ctceilci_d2.csv")
ctc1 = load("ctceilci_tc1.csv")
ctc2 = load("ctceilci_tc2.csv")
ctm = load("ctmulti_clones.csv")
ctta = load("cta_d2.csv")
ctb = load("ctbatch_d2.csv")
ctv1 = load("ctev_tc1_summary.csv")
ctv2 = load("ctev_tc2_summary.csv")
cta1 = load("cta_tc1.csv")
cta2 = load("cta_tc2.csv")
cg = load("cross_graph.csv")
cc = load("ceilci_d2.csv")
mc = load("mc_d2.csv")
bc = load("bc_d2_taskC_summary.csv")

# ---- design


def design(q):
    if dg is None:
        return None
    r = dg[dg["quantity"] == q]
    return float(r["value"].iloc[0]) if len(r) else None


check("task A cells", design("task A cells"), 862, 0.5)
check("task A clones", design("task A clones"), 567, 0.5)
check("task A clone-level majority", design("task A clone-level majority"),
      0.517, 0.0006)
check("task A cell-level majority", design("task A cell-level majority"),
      0.546, 0.0006)
check("undifferentiated at day 2",
      design("task A cells annotated undifferentiated at day 2"), 847, 0.5)
# the ceiling paragraph of the methods, every threshold it quotes
check("clones with mature cells in both wells",
      design("clones, >=1 mature per well"), 788, 0.5)
check("cross-well agreement, >=1 per well",
      design("cross-well dominant-fate agreement, >=1 mature per well"),
      0.810, 0.0006)
check("chance agreement, >=1 per well",
      design("chance agreement, >=1 mature per well"), 0.274, 0.0006)
check("two-way clones, >=1 per well",
      design("clones, neutrophil or monocyte, >=1 per well"), 509, 0.5)
check("two-way agreement, >=1 per well",
      design("cross-well agreement, neutrophil or monocyte, >=1 per well"),
      0.843, 0.0006)
check("clones, >=3 mature per well", design("clones, >=3 mature per well"),
      307, 0.5)
check("clones sampled on both later days",
      design("clones, day-4 and day-6, >=1 per time point"), 1064, 0.5)
check("day-4 against day-6 agreement",
      design("day-4 vs day-6 agreement, >=1 per time point"), 0.852, 0.0006)
check("cross-well agreement, >=3 per well",
      design("cross-well dominant-fate agreement, >=3 mature per well"),
      0.824, 0.0006)
check("two-way clones, >=3 per well",
      design("clones, neutrophil or monocyte, >=3 per well"), 185, 0.5)
check("two-way agreement, >=3 per well",
      design("cross-well agreement, neutrophil or monocyte, >=3 per well"),
      0.832, 0.0006)

check("accuracy of the gate majority fate",
      design("accuracy of the gate majority fate"), 0.64, 0.005)
check("cross-well agreement, all fates",
      design("cross-well dominant-fate agreement, >=1 mature per well"),
      0.810, 0.0006)
check("clones with both wells", design("clones, >=1 mature per well"), 788,
      0.5)
check("chance agreement",
      design("chance agreement, >=1 mature per well"), 0.274, 0.0006)
check("cross-well agreement, neutrophil or monocyte",
      design("cross-well agreement, neutrophil or monocyte, >=1 per well"),
      0.843, 0.0006)
check("clones, neutrophil or monocyte, both wells",
      design("clones, neutrophil or monocyte, >=1 per well"), 509, 0.5)


# ---- the calibration of the increment assay
def calib(kind, level, col="increment"):
    if pc is None:
        return None
    r = pc[(pc["kind"] == kind) & (pc["level"] == level)]
    return float(r[col].iloc[0]) if len(r) else None


check("noise block costs", calib("synthetic block", 0.0), -0.022, 0.001)
check("block worth 0.78 alone gains", calib("synthetic block", 0.2), 0.009,
      0.001)
check("block worth 0.93 alone gains", calib("synthetic block", 0.4), 0.088,
      0.001)
check("block alone at level 0.2",
      calib("synthetic block", 0.2, "block_alone_auroc"), 0.784, 0.002)
check("block alone at level 0.4",
      calib("synthetic block", 0.4, "block_alone_auroc"), 0.928, 0.002)
check("complement at one unspliced depth",
      calib("read-split complement", 1.0), -0.024, 0.001)
check("complement at two", calib("read-split complement", 2.0), 0.001, 0.001)
check("complement at four", calib("read-split complement", 4.0), 0.066, 0.001)
check("complement at eight", calib("read-split complement", 8.0), 0.202,
      0.001)


# ---- the splicing layers themselves
def smq(q):
    if sm is None:
        return None
    r = sm[sm["quantity"] == q]
    return float(r["value"].iloc[0]) if len(r) else None


check("selected genes with a spliced molecule",
      smq("selected genes with a spliced molecule, per cell (median)"), 326,
      0.5)
check("selected genes with an unspliced molecule",
      smq("selected genes with an unspliced molecule, per cell (median)"), 86,
      0.5)
check("selected genes with both",
      smq("selected genes with both, per cell (median)"), 17, 0.5)
check("R2 of smoothed unspliced on smoothed spliced",
      smq("R2 of smoothed unspliced counts on smoothed spliced counts "
          "(median)"), 0.01, 0.002)
check("R2 of the smoothed ratio on smoothed spliced",
      smq("R2 of the smoothed splicing ratio on smoothed spliced counts "
          "(median)"), 0.08, 0.002)

# ---- section 1
check("best abundance model, AUROC", arm(ev, "A3_state_total", "auroc_mean"),
      0.870, 0.0006)
check("best abundance model, accuracy", arm(ev, "A3_state_total", "acc_mean"),
      0.789, 0.0006)
check("sorting gate alone, AUROC", arm(ev, "A1_pop", "auroc_mean"), 0.625,
      0.0006)
if mc is not None:
    def m(direction, who, col):
        r = mc[(mc["direction"] == direction) & (mc["who"] == who)]
        return float(r[col].iloc[0]) if len(r) else None

    d12 = "well1_predicts_well2"
    d21 = "well2_predicts_well1"
    check("matched clones", m(d12, "sister_oracle", "n_clones"), 212, 0.5)
    check("oracle AUROC, well 1 to well 2",
          m(d12, "sister_oracle", "auroc"), 0.868, 0.0006)
    check("oracle accuracy, well 1 to well 2",
          m(d12, "sister_oracle", "acc"), 0.821, 0.0006)
    check("model AUROC, well 1 to well 2",
          m(d12, "A4_state_smoothed", "auroc"), 0.887, 0.0006)
    check("model accuracy, well 1 to well 2",
          m(d12, "A4_state_smoothed", "acc"), 0.811, 0.0006)
    check("difference, well 1 to well 2",
          m(d12, "A4_state_smoothed", "diff_vs_oracle"), 0.019, 0.0006)
    check("oracle AUROC, well 2 to well 1",
          m(d21, "sister_oracle", "auroc"), 0.882, 0.0006)
    check("oracle accuracy, well 2 to well 1",
          m(d21, "sister_oracle", "acc"), 0.816, 0.0006)
    check("model AUROC, well 2 to well 1",
          m(d21, "A4_state_smoothed", "auroc"), 0.871, 0.0006)
    check("model accuracy, well 2 to well 1",
          m(d21, "A4_state_smoothed", "acc"), 0.783, 0.0006)
    check("difference, well 2 to well 1",
          m(d21, "A4_state_smoothed", "diff_vs_oracle"), -0.010, 0.0006)
check("neutrophil-bias correlation, model",
      arm(bc, "A2_state_spliced", "pearson"), 0.677, 0.0006)

# ---- section 2
check("unspliced alone minus spliced", pair(pr, "A5_unspliced_only",
                                            "A2_state_spliced"), -0.088, 0.001)
check("unspliced added to spliced", pair(pr, "A6_state_plus_unspliced",
                                         "A2_state_spliced"), -0.006, 0.001)
check("veloVI arm minus spliced", pair(pr, "V3_smoothed_plus_vel_velovi",
                                       "A2_state_spliced"), -0.003, 0.001)
check("dynamical arm minus spliced", pair(pr, "V2_smoothed_plus_vel_dyn",
                                          "A2_state_spliced"), -0.032, 0.001)
check("CellRank velocity kernel minus smoothed",
      pair(pr, "V4_smoothed_plus_cr_velo", "A4_state_smoothed"), 0.007, 0.001)
check("CellRank pseudotime kernel minus smoothed",
      pair(pr, "V5_smoothed_plus_cr_pseudo", "A4_state_smoothed"), -0.016,
      0.001)
check("ten-component smoothed baseline",
      arm(ev, "A4b_state_smoothed_10pc", "auroc_mean"), 0.863, 0.0006)
check("ten-component velocity arm",
      arm(ev, "V2b_smoothed_plus_vel_dyn_10pc", "auroc_mean"), 0.859, 0.0006)

# ---- section 3
check("equal-depth spliced copy added",
      quant(rs, "independent equal-depth spliced copy added"), -0.013, 0.001)
check("unspliced layer in its place",
      quant(rs, "unspliced layer added in its place"), -0.008, 0.001)
check("difference between the two",
      quant(rs, "unspliced minus independent spliced copy"), 0.005, 0.001)
check("depth series, full depth", arm(dn, "D_frac1", "auroc_mean"), 0.858,
      0.0006)
check("depth series, one sixteenth", arm(dn, "D_frac0.0625", "auroc_mean"),
      0.670, 0.0006)
check("unspliced layer alone", arm(dn, "A5_unspliced_only", "auroc_mean")
      or arm(ev, "A5_unspliced_only", "auroc_mean"), 0.778, 0.0006)
check("spliced thinned to the unspliced depth",
      arm(dn, "D_udepth", "auroc_mean"), 0.764, 0.0006)

# ---- section 4
if fa is not None:
    f = fa[fa.basis == "smoothed"]
    k = lambda fam, c: (float(f[f.family == fam][c].iloc[0])
                        if (f.family == fam).any() else None)
    check("velocity against state projection, veloVI",
          k("velovi", "corr_with_state_projection"), -0.52, 0.006)
    check("velocity against state projection, steady state",
          k("det", "corr_with_state_projection"), -0.35, 0.006)


# ---- the direction test and its positive control


def tav(axis, vector, col="direction_correct"):
    if ta is None:
        return None
    r = ta[(ta["axis"] == axis) & (ta["vector"] == vector)]
    return float(r[col].iloc[0]) if len(r) else None


check("maturation, steady state", tav("maturation", "det"), 0.942, 0.0006)
check("maturation, veloVI", tav("maturation", "velovi"), 0.878, 0.0006)
check("maturation, dynamical", tav("maturation", "dyn"), 0.800, 0.0006)
check("maturation, the cell's own state", tav("maturation", "state_itself"),
      0.417, 0.0006)
check("maturation, steady state on the log scale",
      tav("maturation", "det_logscale"), 0.921, 0.0006)
check("maturation, veloVI on the log scale",
      tav("maturation", "velovi_logscale"), 0.885, 0.0006)
check("maturation, dynamical on the log scale",
      tav("maturation", "dyn_logscale"), 0.732, 0.0006)
check("sampling day, veloVI", tav("sampling day", "velovi"), 0.998, 0.0006)
check("sampling day, steady state", tav("sampling day", "det"), 0.101, 0.0006)
check("sampling day, dynamical", tav("sampling day", "dyn"), 0.119, 0.0006)
check("fate, dynamical", tav("clonal fate", "dyn"), 0.520, 0.0006)
check("fate, steady state", tav("clonal fate", "det"), 0.431, 0.0006)
check("fate, veloVI", tav("clonal fate", "velovi"), 0.368, 0.0006)
check("fate, the cell's own state", tav("clonal fate", "state_itself"), 0.766,
      0.0006)
check("fate, steady state reversed", tav("clonal fate", "det", "complement"),
      0.569, 0.0006)
check("fate, veloVI reversed", tav("clonal fate", "velovi", "complement"),
      0.632, 0.0006)
check("fate, steady state on the log scale",
      tav("clonal fate", "det_logscale"), 0.426, 0.0006)
check("fate, veloVI on the log scale", tav("clonal fate", "velovi_logscale"),
      0.311, 0.0006)
check("fate, dynamical on the log scale", tav("clonal fate", "dyn_logscale"),
      0.498, 0.0006)


# ---- the within-gate comparison and the unspliced fraction at clone level


def ufq(q):
    if uf is None:
        return None
    r = uf[uf["quantity"] == q]
    return float(r["value"].iloc[0]) if len(r) else None


check("unspliced fraction, clone-level AUROC",
      ufq("clone-level AUROC of the unspliced fraction alone"), 0.453, 0.0006)
check("unspliced fraction, permutation p",
      ufq("two-sided permutation p over clones"), 0.054, 0.001)
if wp is not None:
    vel = wp[wp.arm.str.startswith("V")]
    check("largest within-gate velocity difference", float(vel["diff"].max()),
          0.043, 0.001)
    check("within-gate velocity differences whose interval excludes zero",
          float(((vel["lo"] > 0) | (vel["hi"] < 0)).sum()), 0, 0.5)
    a6 = wp[wp.arm == "A6_state_plus_unspliced"].sort_values("diff")
    check("spliced plus unspliced inside a gate, worse of the two",
          float(a6["diff"].iloc[0]), -0.080, 0.001)
    check("spliced plus unspliced inside a gate, better of the two",
          float(a6["diff"].iloc[1]), -0.050, 0.001)
if bb is not None:
    bb2 = bb.rename(columns={bb.columns[0]: "arm"})
    tr = bb2[~bb2.arm.isin(["A0_prior", "A1_pop", "A5_unspliced_only"])]
    check("six-way, lowest transcriptome arm", float(tr["macro_auroc"].min()),
          0.898, 0.001)
    check("six-way, highest arm", float(tr["macro_auroc"].max()), 0.927, 0.001)
    check("six-way, unspliced counts alone",
          float(bb2[bb2.arm == "A5_unspliced_only"]["macro_auroc"].iloc[0]),
          0.855, 0.001)

# ---- section 5
if cg is not None:
    c = lambda a, col: float(cg[cg.arm == a][col].iloc[0])
    check("cross-graph, spliced counts", c("A2_state_spliced", "diff"), 0.002,
          0.001)
    check("cross-graph, smoothed counts", c("A4_state_smoothed", "diff"),
          0.006, 0.001)
    check("cross-graph, CellRank velocity kernel alone",
          c("V4a_cr_velo_only", "diff"), 0.043, 0.001)
    check("CellRank velocity kernel, all-cell graph",
          c("V4a_cr_velo_only", "auroc_all_graph"), 0.844, 0.0006)
    check("spliced counts, early graph",
          c("A2_state_spliced", "auroc_early_graph"), 0.866, 0.0006)



# ---- the second dataset


def q(df, name, col="value", key="quantity"):
    if df is None:
        return None
    r = df[df[key] == name]
    return float(r[col].iloc[0]) if len(r) else None


if ctdep is not None:
    check("CellTag unspliced depth on the selected genes",
          float(ctdep["median_unspliced_on_selected_genes"].iloc[0]), 758, 0.5)
check("CellTag genes with both, per cell",
      q(ctsm, "selected genes with both, per cell (median)"), 126, 0.5)
check("CellTag, equal-depth spliced complement added",
      q(ctrs20, "independent equal-depth spliced copy added"), -0.013, 0.001)
check("CellTag, unspliced layer in its place",
      q(ctrs20, "unspliced layer added in its place"), 0.058, 0.001)
check("CellTag, difference, pooled",
      q(ctrs20, "unspliced minus independent spliced copy"), 0.071, 0.001)
check("CellTag, difference inside time course 1",
      q(ct1, "unspliced minus independent spliced copy"), -0.001, 0.001)
check("CellTag, difference inside time course 2",
      q(ct2, "unspliced minus independent spliced copy"), -0.001, 0.001)
if ctev is not None:
    cte = ctev.rename(columns={ctev.columns[0]: "arm"})
    g = lambda a: float(cte[cte.arm == a]["auroc_mean"].iloc[0])
    check("CellTag, best arm", g("V5a_cr_pseudo_only"), 0.640, 0.0006)
    check("CellTag, velocity kernel alone", g("V4a_cr_velo_only"), 0.624,
          0.0006)

if ctce is not None:
    r1 = ctce[(ctce.direction == "readout1_predicts_readout2")]
    check("CellTag ceiling clones", float(r1["n_clones"].iloc[0]), 34, 0.5)
    check("CellTag oracle, first direction",
          float(r1[r1.arm == "A2_state_spliced"]["oracle_auroc"].iloc[0]),
          0.928, 0.0006)
    check("CellTag best state model, first direction",
          float(r1[r1.arm == "A2_state_spliced"]["auroc"].iloc[0]), 0.659,
          0.0006)
    r2 = ctce[(ctce.direction == "readout2_predicts_readout1")]
    check("CellTag oracle, second direction",
          float(r2[r2.arm == "A2_state_spliced"]["oracle_auroc"].iloc[0]),
          0.871, 0.0006)
    check("CellTag best abundance model, second direction",
          float(r2[r2.arm == "A4b_state_smoothed_10pc"]["auroc"].iloc[0]),
          0.686, 0.0006)
    check("CellTag best arm at the ceiling, first direction",
          float(r1[r1.arm == "V5a_cr_pseudo_only"]["auroc"].iloc[0]), 0.742,
          0.0006)
    check("CellTag best arm at the ceiling, second direction",
          float(r2[r2.arm == "V5a_cr_pseudo_only"]["auroc"].iloc[0]), 0.758,
          0.0006)

# the same comparison inside one time course, where the oracle can no longer
# be carried by the batch: the four values the text quotes, and the widths
for tag, df, want in (("first", ctc1,
                       dict(n=20, orc1=0.880, orc2=0.841, best1="A3_state_total",
                            b1=0.613, best2="A3_state_total", b2=0.692,
                            d1=-0.267, lo1=-0.506, hi1=-0.021,
                            d2=-0.148, lo2=-0.448, hi2=0.161)),
                      ("second", ctc2,
                       dict(n=14, orc1=1.000, orc2=0.944,
                            best1="V5a_cr_pseudo_only", b1=0.857,
                            best2="V2a_vel_dyn_only", b2=0.900,
                            d1=-0.143, lo1=-0.408, hi1=0.000,
                            d2=-0.044, lo2=-0.290, hi2=0.138))):
    if df is None:
        continue
    for j, (src, tgt) in enumerate(((1, 2), (2, 1)), start=1):
        d = df[df.direction == "readout%d_predicts_readout%d" % (src, tgt)]
        check("CellTag %s time course, ceiling clones" % tag,
              float(d["n_clones"].iloc[0]), want["n"], 0.5)
        check("CellTag %s time course, oracle, direction %d" % (tag, j),
              float(d["oracle_auroc"].iloc[0]), want["orc%d" % j], 0.0006)
        row = d[d.arm == want["best%d" % j]]
        check("CellTag %s time course, best arm, direction %d" % (tag, j),
              float(row["auroc"].iloc[0]), want["b%d" % j], 0.0006)
        check("CellTag %s time course, difference, direction %d" % (tag, j),
              float(row["diff"].iloc[0]), want["d%d" % j], 0.0006)
        check("CellTag %s time course, lower bound, direction %d" % (tag, j),
              float(row["lo"].iloc[0]), want["lo%d" % j], 0.0006)
        check("CellTag %s time course, upper bound, direction %d" % (tag, j),
              float(row["hi"].iloc[0]), want["hi%d" % j], 0.0006)
        check("CellTag %s time course, interval width, direction %d"
              % (tag, j), float(row["hi"].iloc[0] - row["lo"].iloc[0]),
              want["hi%d" % j] - want["lo%d" % j], 0.0016)
        if not 0.40 <= float(row["hi"].iloc[0] - row["lo"].iloc[0]) <= 0.61:
            fails.append("CellTag %s time course, direction %d: the interval "
                         "is outside the 0.40 to 0.61 the text quotes"
                         % (tag, j))
        # the arm the text calls best must really be the best one scored
        scored = d[~d.arm.isin(["A0_prior", "A1_pop"])]
        if scored["auroc"].idxmax() != row.index[0]:
            fails.append("CellTag %s time course, direction %d: the arm the "
                         "text calls best is not the highest scored" %
                         (tag, j))

if ctm is not None:
    m = lambda name: q(ctm, name)
    check("CellTag-multi, clones spanning state and fate",
          m("clones spanning state and fate, either assay"), 2227, 0.5)
    check("CellTag-multi, of those with a state transcriptome",
          m("of those, with a transcriptome in the state sample"), 1489, 0.5)
    check("CellTag-multi, raw libraries in the archive", m("sra runs"), 68,
          0.5)
    check("CellTag-multi, raw data in gigabytes", m("sra total gb"), 636, 0.5)
if ctta is not None:
    t = lambda a, v: float(ctta[(ctta.axis == a) & (ctta.vector == v)]
                           ["direction_correct"].iloc[0])
    check("CellTag sampling day, steady state", t("sampling day", "det"),
          0.800, 0.0006)
    check("CellTag sampling day, veloVI", t("sampling day", "velovi"), 1.000,
          0.0006)
    check("CellTag sampling day, dynamical", t("sampling day", "dyn"), 0.88,
          0.005)
    check("CellTag maturation, steady state", t("maturation", "det"), 0.32,
          0.005)
    check("CellTag maturation, dynamical", t("maturation", "dyn"), 0.46,
          0.005)
    check("CellTag maturation, veloVI", t("maturation", "velovi"), 0.004,
          0.0006)
    check("CellTag fate, largest estimate",
          t("clonal fate", "velovi_logscale"), 0.617, 0.0006)
    sdv = float(ctta[(ctta.axis == "clonal fate")
                     & (ctta.vector == "velovi_logscale")]["sd"].iloc[0])
    check("CellTag fate, that estimate in standard deviations",
          (t("clonal fate", "velovi_logscale") - 0.5) / sdv, 1.1, 0.05)

if ctb is not None:
    b = lambda n: float(ctb[ctb.block == n]["auroc"].iloc[0])
    check("CellTag batch, unspliced layer", b("unspliced layer"), 0.533,
          0.0006)
    check("CellTag batch, equal-depth spliced complement",
          b("equal-depth spliced complement"), 0.525, 0.0006)
    check("CellTag batch, all spliced counts", b("all spliced counts"), 0.548,
          0.0006)
    check("CellTag clones in time course 1", b("clones in tc1"), 65, 0.5)
    check("CellTag endoderm clones in time course 1",
          b("of those with fate iEP in tc1"), 45, 0.5)
    check("CellTag clones in time course 2", b("clones in tc2"), 63, 0.5)
    check("CellTag endoderm clones in time course 2",
          b("of those with fate iEP in tc2"), 33, 0.5)

ABUND = ["A2_state_spliced", "A3_state_total", "A4_state_smoothed",
         "A4b_state_smoothed_10pc"]


def armtab(df):
    d = df.rename(columns={df.columns[0]: "arm"})
    return d.set_index("arm")["auroc_mean"]


if ctev is not None:
    e = armtab(ctev)
    check("CellTag arms in the table", len(e), 21, 0.5)
    check("CellTag, best abundance-alone arm, pooled", e[ABUND].max(), 0.555,
          0.0006)
for name, d in (("time course 1", ctv1), ("time course 2", ctv2)):
    if d is None:
        continue
    e = armtab(d)
    best, ab = (0.550, 0.476) if name.endswith("1") else (0.554, 0.496)
    check("CellTag best arm inside %s" % name, e.max(), best, 0.0006)
    check("CellTag best abundance-alone arm inside %s" % name, e[ABUND].max(),
          ab, 0.0006)
fa, sd_lo, sd_hi = [], [], []
for d in (cta1, cta2):
    if d is None:
        continue
    fa += list(d[d.axis == "clonal fate"]["direction_correct"].values)
    g = d[d.axis == "sampling day"].set_index("vector")["direction_correct"]
    sd_lo.append(float(g["dyn"]))
    sd_hi.append(float(g["velovi"]))
if fa:
    check("CellTag fate axis inside a time course, lowest", min(fa), 0.330,
          0.0006)
    check("CellTag fate axis inside a time course, highest", max(fa), 0.577,
          0.0006)
    check("CellTag sampling day inside time course 1, dynamical", sd_lo[0],
          0.861, 0.0006)
    check("CellTag sampling day inside time course 2, dynamical", sd_lo[1],
          0.837, 0.0006)
    check("CellTag sampling day inside a time course, veloVI", min(sd_hi),
          1.000, 0.0006)

tie = load("tie_rules.csv")
if tie is not None:
    t = tie.set_index("rule")
    check("tie rule, preregistered, agreement",
          float(t.loc["preregistered fixed order", "agreement"]), 0.810,
          0.0006)
    check("tie rule, preregistered, two-way clones",
          float(t.loc["preregistered fixed order", "nm_clones"]), 509, 0.5)
    check("tie rule, alphabetical, agreement",
          float(t.loc["alphabetical", "agreement"]), 0.793, 0.0006)
    check("tie rule, alphabetical, two-way clones",
          float(t.loc["alphabetical", "nm_clones"]), 500, 0.5)
    check("tie rule, alphabetical, two-way agreement",
          float(t.loc["alphabetical", "nm_agreement"]), 0.832, 0.0006)
    check("tie rule, spread in the agreement",
          float(t.loc["spread across the three rules", "agreement"]), 0.016,
          0.0006)
    check("tie rule, spread in the two-way agreement",
          float(t.loc["spread across the three rules", "nm_agreement"]),
          0.011, 0.0006)

# ---- the manuscript source itself
import glob
# any control byte except the newline. A tab is the one that bites: a patch
# applied through a shell turns the backslash-t of \times or \texttt into one,
# which LaTeX swallows as whitespace and typesets the rest of the command name
CTRL = set(range(32)) - {10}
stray = []
for f in sorted(glob.glob(os.path.join(PAPER, "*.tex"))):
    nbad = sum(1 for b in open(f, "rb").read() if b in CTRL)
    if nbad:
        stray.append((os.path.basename(f), nbad))
if stray:
    fails.append("control bytes inside the LaTeX source: %s" % stray)
else:
    print("ok   %-58s %s" % ("no stray control bytes in the LaTeX source",
                             "-"))

# the same accident with the backslash dropped instead of replaced leaves no
# control byte at all, only the tail of the command name sitting in the text
EATEN = ["imes", "exttt{", "extbf{", "extit{", "extsc{", "ext{", "abel{",
         "ef{", "ite{", "egin{", "nd{", "rac{", "extsuperscript{"]
eaten = []
for f in sorted(glob.glob(os.path.join(PAPER, "*.tex"))):
    body = open(f, encoding="utf-8").read()
    for tail in EATEN:
        for m in re.finditer("(?<![A-Za-z])" + re.escape(tail), body):
            eaten.append((os.path.basename(f), tail,
                          body[max(0, m.start() - 20):m.start() + 10]))
if eaten:
    fails.append("LaTeX command names with the backslash missing: %s" % eaten)
else:
    print("ok   %-58s %s" % ("no LaTeX command left without its backslash",
                             "-"))

# ---- typesetting rules that a reader would see
B = chr(92)  # a literal backslash, kept out of the patterns below
TEX_ORDER = ["intro.tex", "results.tex", "celltag.tex", "discussion.tex",
             "methods.tex"]
pat = re.compile(B + B + "(label|ref)" + B + "{"
                 + "((?:fig|tab):[^}]+)" + B + "}")
labels, first_ref = [], []
for name in TEX_ORDER:
    f = os.path.join(PAPER, name)
    if not os.path.exists(f):
        continue
    for m in pat.finditer(open(f, encoding="utf-8").read()):
        if m.group(1) == "label":
            labels.append(m.group(2))
        elif m.group(2) not in first_ref:
            first_ref.append(m.group(2))
if labels:
    uncited = [x for x in labels if x not in first_ref]
    if uncited:
        fails.append("floats never cited in the text: %s" % uncited)
    elif labels != first_ref:
        fails.append("float citation order differs from label order: cited %s, "
                     "defined %s" % (first_ref, labels))
    else:
        print("ok   %-58s %d floats" % ("floats cited in the order they are "
                                        "defined", len(labels)))

# no table may break across pages, so no environment that can break one
breakable = []
for f in sorted(glob.glob(os.path.join(PAPER, "*.tex"))):
    body = open(f, encoding="utf-8").read()
    for env in ("longtable", "supertabular", "xtabular"):
        if (B + "begin{" + env) in body:
            breakable.append((os.path.basename(f), env))
if breakable:
    fails.append("environments that can break a table across pages: %s"
                 % breakable)
else:
    print("ok   %-58s %s" % ("no table can break across pages", "-"))

# the build log, when the caller kept it: any box the reader would see
log = os.path.join(PAPER, "build.log")
if os.path.exists(log):
    text = open(log, encoding="utf-8", errors="replace").read()
    hb = ("Overfull " + B + B + "hbox " + B + "(" + "([0-9.]+)pt too wide"
          + B + ")")
    wide = [float(x) for x in re.findall(hb, text)]
    tall = text.count("Overfull " + B + B + "vbox")
    bad = [w for w in wide if w > 5.0]
    if bad or tall:
        fails.append("boxes out of the text block: %d overfull vbox, "
                     "overfull hbox over 5pt %s" % (tall, bad))
    else:
        print("ok   %-58s %s" % ("no overfull box a reader would see",
                                 "%d hbox under 5pt" % len(wide)))
else:
    print("note %-58s %s" % ("build.log not present, box check skipped", "-"))

print()
if missing:
    print("MISSING result files: %s" % ", ".join(missing))
if fails:
    print("MISMATCHES (%d):" % len(fails))
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("all quoted numbers agree with the generated files")
