"""Two derived quantities, from result files only.

1. Label efficiency. An unsupervised method offers fate probabilities without
   clonal labels. How many labelled clones does the supervised expression
   baseline (total counts, with and without the sorted population) need to
   match each method's direct AUROC? First crossing on the learning curve of
   76_learning_curve.py (training clones per fold against AUROC, 5-100%,
   learning_curve_lowend_summary.csv); values below the smallest training size
   are reported as below it.
2. Clones needed to resolve a difference of 0.005 in Pearson r (rather than in
   AUROC), from the label-only simulation of 53_r_floor.py on the 567 clones:
   567 x (minimum detectable difference in r / 0.005)^2, at the noise
   correlations implied by the measured residual correlations (0.49-0.75) and
   for near-identical pipelines (0.95).

Usage: 80_label_efficiency.py <outdir> <out_prefix>
"""
import sys

import numpy as np
import pandas as pd

R, OUT = sys.argv[1], sys.argv[2]
# The learning curve with the finer low end (5-100% of the training clones),
# with the sorted population (as every fitted model) and without it, which is
# like-for-like with a method's direct AUROC (the methods do not use it).
LC = pd.read_csv(R + "/learning_curve_lowend_summary.csv")
U = pd.read_csv(R + "/unsupervised_reference.csv").set_index("predictor")


def first_crossing(L, a):
    """Training clones at which the mean curve first reaches a, linearly
    interpolated between that point and the one before; the curve without the
    sorted population is not monotone, so the first crossing is used."""
    n, y = L.n_train_clones.values.astype(float), L.auroc_mean.values
    if a <= y[0]:
        return n[0], "below"
    for i in range(1, len(y)):
        if y[i] >= a:
            return n[i - 1] + (a - y[i - 1]) / (y[i] - y[i - 1]) * (n[i] - n[i - 1]), ""
    return n[-1], "above"


rows = []
for k in U.index:
    a = U.loc[k, "auroc_clone"]
    r = dict(predictor=k, kind=U.loc[k, "kind"], direct_auroc=a)
    for arm, tag in (("total", "with_pop"), ("total_nopop", "without_pop")):
        L = LC[LC.arm == arm].sort_values("n_train_clones")
        n, note = first_crossing(L, a)
        r["clones_%s" % tag] = round(float(n))
        r["note_%s" % tag] = note
    rows.append(r)
E = pd.DataFrame(rows)
E.to_csv(OUT + "_label_efficiency.csv", index=False)
print(E.to_string(index=False))

F = pd.read_csv(R + "/rfloor2_invitro_summary.csv")
F = F[F.scenario == "n567"].sort_values("rho")
out = []
for rho in (0.49, 0.75, 0.95):
    mde = float(np.interp(rho, F.rho, F.mde_r))
    out.append(dict(rho_noise=rho, mde_r_567=mde, clones_for_r_0005=567 * (mde / 0.005) ** 2,
                    clones_for_r_001=567 * (mde / 0.01) ** 2))
O = pd.DataFrame(out).round(4)
O.to_csv(OUT + "_clones_for_r.csv", index=False)
print(O.to_string(index=False))
