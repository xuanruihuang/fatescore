"""Two quantities the mechanism argument asserts and should measure.

Usage: 18_smoothing_check.py <featdir> <out.csv>

First, co-detection on the genes the models actually use: in how many of the
selected genes does one cell carry both a spliced and an unspliced molecule.
The whole-transcriptome figure is not the relevant one, because most genes are
zero in both layers and never enter the analysis.

Second, how much of the smoothed unspliced counts, of the smoothed splicing
ratio and of the inferred velocity is a function of the smoothed spliced counts
alone. For each selected gene we regress the quantity on the smoothed spliced
count of the same gene across cells and report the distribution of R^2. If the
neighbour averaging leaves a function of neighbourhood mean abundance, these
should be high.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

FD, OUT = sys.argv[1], sys.argv[2]


def load(n):
    p = os.path.join(FD, n + ".npy")
    return np.load(p) if os.path.exists(p) else None


raw_s, raw_u = load("raw_spliced"), load("raw_unspliced")
Ms, Mu = load("Ms"), load("Mu")
rows = []

both = ((raw_s > 0) & (raw_u > 0)).sum(1)
any_s = (raw_s > 0).sum(1)
any_u = (raw_u > 0).sum(1)
n_genes = raw_s.shape[1]
rows.append(dict(quantity="selected genes", value=n_genes))
for name, v in (("selected genes with a spliced molecule, per cell", any_s),
                ("selected genes with an unspliced molecule, per cell", any_u),
                ("selected genes with both, per cell", both),
                ("fraction of selected genes with both, per cell",
                 both / n_genes)):
    rows.append(dict(quantity=name + " (median)", value=float(np.median(v))))
    rows.append(dict(quantity=name + " (10th pct)",
                     value=float(np.percentile(v, 10))))
    rows.append(dict(quantity=name + " (90th pct)",
                     value=float(np.percentile(v, 90))))


def r2_against_ms(M, label):
    """Per gene, the squared correlation with the smoothed spliced count."""
    x = Ms - Ms.mean(0)
    y = M - M.mean(0)
    sx = np.sqrt((x ** 2).sum(0))
    sy = np.sqrt((y ** 2).sum(0))
    ok = (sx > 0) & (sy > 0)
    r = np.zeros(M.shape[1])
    r[ok] = (x[:, ok] * y[:, ok]).sum(0) / (sx[ok] * sy[ok])
    r2 = r[ok] ** 2
    for q, v in (("median", np.median(r2)), ("25th pct", np.percentile(r2, 25)),
                 ("75th pct", np.percentile(r2, 75)),
                 ("fraction above 0.5", float((r2 > 0.5).mean())),
                 ("fraction above 0.9", float((r2 > 0.9).mean()))):
        rows.append(dict(quantity="R2 of %s on smoothed spliced counts (%s)"
                                  % (label, q), value=float(v)))


r2_against_ms(Mu, "smoothed unspliced counts")
r2_against_ms(Mu / (Ms + Mu + 1e-8), "the smoothed splicing ratio")
for k in ("det", "dyn", "velovi"):
    V = load("vel_" + k)
    if V is not None:
        r2_against_ms(V, "the %s velocity" % k)

R = pd.DataFrame(rows)
R.to_csv(OUT, index=False)
print(R.to_string(index=False))
