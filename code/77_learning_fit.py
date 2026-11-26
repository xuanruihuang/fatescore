"""Extrapolate the learning curve of 76_learning_curve.py.

Two saturating forms are fitted to the mean AUROC against the number of
training clones: a power law a - b n^-c and an exponential a - b exp(-n/t).
Each is also fitted without the largest training size and asked to predict it,
so the form that extrapolates better can be told from the one that only fits
better. Reports the asymptote a and the value at twice the full training size.

Usage: 77_learning_fit.py <learning_curve_summary.csv> <out.csv>
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

S = pd.read_csv(sys.argv[1])
FORMS = {"power": (lambda n, a, b, c: a - b * n ** (-c), [0.9, 2.0, 0.7]),
         "exponential": (lambda n, a, b, t: a - b * np.exp(-n / t), [0.88, 0.2, 150.0])}
rows = []
for arm, G in S.groupby("arm"):
    n = G["n_train_clones"].to_numpy(float)
    y = G["auroc_mean"].to_numpy(float)
    for name, (f, p0) in FORMS.items():
        p, _ = curve_fit(f, n, y, p0=p0, maxfev=20000)
        q, _ = curve_fit(f, n[:-1], y[:-1], p0=p0, maxfev=20000)
        rows.append(dict(arm=arm, form=name, asymptote=p[0],
                         at_twice_n=f(2 * n[-1], *p),
                         heldout_pred=f(n[-1], *q), heldout_obs=y[-1],
                         heldout_error=f(n[-1], *q) - y[-1]))
R = pd.DataFrame(rows).round(4)
R.to_csv(sys.argv[2], index=False)
print(R.to_string(index=False))
