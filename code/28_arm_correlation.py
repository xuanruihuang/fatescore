"""How sharp a paired comparison between two similar methods really is.

The power budget in 27_power_budget.py has one free quantity: how correlated the
two arms' per-clone noise is. It was fixed at 0.5 with 1.0 and 0.0 as
sensitivities, and that choice turned out to drive the answer -- the detection
floor is proportional to sqrt(2(1 - rho)). Two methods of the same family,
scored on the same clones, may be far more correlated than 0.5, and if they are,
the floor for comparing them is much lower than the budget reported.

This measures it instead of assuming it, from results the manuscript already
produced: the out-of-fold predictions of all 21 arms on the same clone-grouped
folds. For every pair of arms it reports the correlation of their clone-level
scores, the correlation of their within-class residuals, and the width of the
paired bootstrap interval on the difference -- in AUROC and in the Pearson r
against the clone's fate bias, which is the unit most of the literature reports.
From the width it reads back the rho that the simulation would need, and the
detection floor that width implies.

This was added after the pre-registered budget was computed and read, in answer
to that objection; it is a measurement, not a change to any decision rule.

Usage:
  python 28_arm_correlation.py <task.json> <oof.npy> <labels.tsv> <out_prefix>
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 13
# The simulated width at rho = 0.5 on this task, from out/power_budget.csv
# (T1, A0 = 0.65, delta = 0). Used only to express a measured width as the rho
# the simulation would need: width(rho) / width(0.5) = sqrt(2(1 - rho)).
SIM_WIDTH_AT_HALF = 0.0904
# A 95% interval of width W has sigma = W / (2 * 1.96); detecting an effect with
# 80% power under the "lower bound above zero" rule needs delta >= 2.80 sigma.
FLOOR_PER_WIDTH = 2.80 / (2 * 1.96)


def auroc_rows(score, y):
    B, n = score.shape
    n1 = y.sum(1).astype(np.float64)
    n0 = n - n1
    order = np.argsort(score, axis=1, kind="stable")
    ranks = np.empty((B, n), dtype=np.float64)
    seq = np.broadcast_to(np.arange(1, n + 1, dtype=np.float64), (B, n))
    np.put_along_axis(ranks, order, seq, axis=1)
    out = ((ranks * y).sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = np.nan
    return out


def pearson_rows(a, b):
    x = a - a.mean(1, keepdims=True)
    z = b - b.mean(1, keepdims=True)
    num = (x * z).sum(1)
    den = np.sqrt((x * x).sum(1) * (z * z).sum(1))
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("oof")
    ap.add_argument("labels")
    ap.add_argument("out_prefix")
    ap.add_argument("--repeat", type=int, default=0,
                    help="which repeat's out-of-fold predictions to use; the "
                         "manuscript's paired bootstrap uses the first")
    a = ap.parse_args()

    task = json.load(open(a.task, encoding="utf-8"))
    arms = task["arms"]
    oof = np.load(a.oof)                      # (arm, repeat, cell)
    y_cell = np.asarray(task["y"], dtype=int)
    clone_cell = np.asarray([int(c) for c in task["clones"]])

    lab = pd.read_csv(a.labels, sep="\t")
    f_of_clone = (lab.drop_duplicates("clone").set_index("clone")["f_neu"]
                  .to_dict())

    clones = np.unique(clone_cell)
    # clone-level score per arm, as cvlib.clone_level does it: mean over cells
    P = np.empty((len(arms), clones.size))
    for i in range(len(arms)):
        s = pd.Series(oof[i, a.repeat], index=clone_cell)
        P[i] = s.groupby(level=0).mean().reindex(clones).to_numpy()
    y = pd.Series(y_cell, index=clone_cell).groupby(level=0).first() \
        .reindex(clones).to_numpy().astype(bool)
    f = np.array([f_of_clone.get(int(c), np.nan) for c in clones])
    ok_f = np.isfinite(f)

    print("arms %d, clones %d, positives %d, clones with a fate bias %d"
          % (len(arms), clones.size, int(y.sum()), int(ok_f.sum())))

    # one set of bootstrap resamples, shared by every arm, so the pairing is kept
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, clones.size, size=(N_BOOT, clones.size))
    yb = y[idx]
    boot_auc = np.empty((len(arms), N_BOOT))
    for i in range(len(arms)):
        boot_auc[i] = auroc_rows(P[i][idx], yb)

    # the same in the unit the literature reports: r against the fate bias
    fi = np.where(ok_f)[0]
    idx_f = rng.integers(0, fi.size, size=(N_BOOT, fi.size))
    fb = f[fi][idx_f]
    boot_r = np.empty((len(arms), N_BOOT))
    for i in range(len(arms)):
        boot_r[i] = pearson_rows(P[i][fi][idx_f], fb)

    point_auc = auroc_rows(P, np.broadcast_to(y, P.shape))
    point_r = pearson_rows(P[:, fi], np.broadcast_to(f[fi], (len(arms), fi.size)))

    def family(name):
        if name.startswith(("A0", "A1")):
            return "trivial"
        if name.startswith(("A2", "A3", "A4", "A4b")):
            return "abundance"
        if name.startswith(("A5", "A6", "A7", "N2")):
            return "splicing"
        return "velocity"

    rows = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            d = boot_auc[i] - boot_auc[j]
            d = d[np.isfinite(d)]
            dr = boot_r[i] - boot_r[j]
            dr = dr[np.isfinite(dr)]
            q = np.percentile(d, [2.5, 97.5])
            qr = np.percentile(dr, [2.5, 97.5])
            w = float(q[1] - q[0])
            wr = float(qr[1] - qr[0])
            # within-class residuals: take out what the label explains
            res_i = P[i] - np.where(y, P[i][y].mean(), P[i][~y].mean())
            res_j = P[j] - np.where(y, P[j][y].mean(), P[j][~y].mean())
            rho_w = 1.0 - (w / SIM_WIDTH_AT_HALF) ** 2 / 2.0
            rows.append(dict(
                arm_a=arms[i], arm_b=arms[j],
                family_a=family(arms[i]), family_b=family(arms[j]),
                same_family=int(family(arms[i]) == family(arms[j])),
                auroc_a=round(float(point_auc[i]), 4),
                auroc_b=round(float(point_auc[j]), 4),
                diff_auroc=round(float(point_auc[i] - point_auc[j]), 4),
                width_auroc=round(w, 4),
                floor_auroc_implied=round(w * FLOOR_PER_WIDTH, 4),
                rho_implied_from_width=round(float(rho_w), 4),
                r_a=round(float(point_r[i]), 4), r_b=round(float(point_r[j]), 4),
                diff_r=round(float(point_r[i] - point_r[j]), 4),
                width_r=round(wr, 4),
                floor_r_implied=round(wr * FLOOR_PER_WIDTH, 4),
                corr_scores=round(float(np.corrcoef(P[i], P[j])[0, 1]), 4),
                corr_residuals=round(float(np.corrcoef(res_i, res_j)[0, 1]), 4),
            ))

    R = pd.DataFrame(rows)
    R.to_csv(a.out_prefix + "_pairs.csv", index=False)

    sel = R[R.same_family == 1].sort_values("width_auroc")
    print("\nsame-family pairs, sharpest first:")
    print(sel[["arm_a", "arm_b", "diff_auroc", "width_auroc",
               "floor_auroc_implied", "rho_implied_from_width",
               "corr_residuals", "width_r", "floor_r_implied"]]
          .head(18).to_string(index=False))

    named = [("A2_state_spliced", "A4_state_smoothed"),
             ("A2_state_spliced", "A6_state_plus_unspliced"),
             ("A2_state_spliced", "A3_state_total"),
             ("V1_smoothed_plus_vel_det", "V2_smoothed_plus_vel_dyn"),
             ("V1a_vel_det_only", "V2a_vel_dyn_only"),
             ("A4_state_smoothed", "V2_smoothed_plus_vel_dyn")]
    out = []
    for x, z in named:
        m = R[((R.arm_a == x) & (R.arm_b == z)) | ((R.arm_a == z) & (R.arm_b == x))]
        if len(m):
            out.append(m.iloc[0])
    if out:
        print("\nthe pairs named in the objection:")
        print(pd.DataFrame(out)[["arm_a", "arm_b", "diff_auroc", "width_auroc",
                                 "floor_auroc_implied", "rho_implied_from_width",
                                 "corr_scores", "corr_residuals",
                                 "width_r", "floor_r_implied"]]
              .to_string(index=False))

    summ = []
    for key, g in R.groupby("same_family"):
        summ.append(dict(group="same family" if key else "different family",
                         pairs=len(g),
                         median_width_auroc=round(float(g.width_auroc.median()), 4),
                         min_width_auroc=round(float(g.width_auroc.min()), 4),
                         median_rho=round(float(g.rho_implied_from_width.median()), 4),
                         max_rho=round(float(g.rho_implied_from_width.max()), 4),
                         median_width_r=round(float(g.width_r.median()), 4),
                         min_width_r=round(float(g.width_r.min()), 4)))
    S = pd.DataFrame(summ)
    S.to_csv(a.out_prefix + "_summary.csv", index=False)
    print("\n" + S.to_string(index=False))
    print("\nwrote %s_{pairs,summary}.csv" % a.out_prefix)


if __name__ == "__main__":
    main()
