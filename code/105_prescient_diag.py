"""Why PRESCIENT scores low on the published LARRY test cells: cheap checks.

Rebuilds the 298 matched test cells exactly as 101_weinreb_test_set.py, then for each named predictor
reads its raw per-population probabilities (<featdir>/<name>.npy with populations in <name>_log.json)
and reports: test cells with zero neutrophil+monocyte mass (share filled with 0.5), Pearson r of the
share on all cells, on cells with mass only, with the two columns swapped, and of the raw neutrophil
probability; plus the nm_ file agreement with the raw columns and the log's landing fraction.

Usage: 105_prescient_diag.py <fig5_dir> <h5ad> <featdir> <d2_labels.tsv> <name> [<name> ...]
"""
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

F5, H5, FD, LAB = sys.argv[1:5]
NAMES = sys.argv[5:]
ld = lambda n: np.load(os.path.join(F5, n + ".npy"), allow_pickle=True)  # noqa: E731
tp, nm, early, held = ld("timepoints"), ld("neutrophil_monocyte_trajectory_mask"), ld("early_cells"), ld("heldout_mask")
cfm = ld("clonal_fate_matrix")
sx, sy = ld("coordinates_x"), ld("coordinates_y")
nm2 = nm[tp == 2]; e_nm = early[nm2]; keep = ~held
has_fate = (cfm[:, 5:7].sum(1) > 0) & nm2
has_fate_eh = has_fate[nm2 & early][keep[e_nm]]
test_orig = np.where(tp == 2)[0][nm2 & early][keep[e_nm]][has_fate_eh]
y = (cfm[has_fate, 5] / cfm[has_fate, 5:7].sum(1))[(e_nm & keep)[has_fate[nm2]]]
obs = ad.read_h5ad(H5, backed="r").obs
key = lambda x, y_: np.round(x, 2).astype(str).astype(object) + "_" + np.round(y_, 2).astype(str).astype(object)  # noqa: E731
okey = pd.Series(obs.index.values, index=key(obs["SPRING-x"].values.astype(float), obs["SPRING-y"].values.astype(float)))
okey = okey[~okey.index.duplicated(keep=False)]
cell = np.array([okey.get(k, None) for k in key(sx[test_orig].astype(float), sy[test_orig].astype(float))], dtype=object)
d2 = json.load(open(os.path.join(FD, "d2_names.json")))
pos = {n: i for i, n in enumerate(d2)}
ok = np.array([c is not None and c in pos for c in cell])
idx = np.array([pos[c] for c in cell[ok]]); yt = y[ok]
print("test cells %d, matched %d" % (len(y), ok.sum()))
for n in NAMES:
    P = np.load(os.path.join(FD, n + ".npy"))
    L = json.load(open(os.path.join(FD, n + "_log.json")))
    pops = L["populations"]
    neu, mo = P[idx, pops.index("Neutrophil")], P[idx, pops.index("Monocyte")]
    tot = neu + mo
    sh = np.where(tot > 0, neu / np.where(tot > 0, tot, 1), 0.5)
    m = tot > 0
    nmf = os.path.join(FD, "nm_" + n + ".npy")
    agree = np.nan
    if os.path.exists(nmf):
        M = np.load(nmf).reshape(len(d2), -1)
        if M.shape[1] > 1:
            t2 = M[:, 0] + M[:, 1]
            s2 = np.where(t2 > 0, M[:, 0] / np.where(t2 > 0, t2, 1), 0.5)
        else:
            s2 = M[:, 0]
        agree = float(np.nanmax(np.abs(s2[idx] - sh)))
    other = 1 - P[idx].sum(1) if P.shape[1] else np.nan
    print("%-24s sims %s land %.3f | no Neu+Mo mass %3d/%d | r all %.3f | r with mass %.3f (n %d) | swapped %.3f | "
          "raw P(neu) %.3f | mean Neu+Mo mass %.3f | max|nm-raw| %s"
          % (n, L.get("num_sims"), L.get("fraction_landing_mature", np.nan), (~m).sum(), len(m), pearsonr(sh, yt)[0],
             pearsonr(sh[m], yt[m])[0] if m.sum() > 2 else np.nan, m.sum(), pearsonr(1 - sh, yt)[0],
             pearsonr(neu, yt)[0], tot.mean(), agree))
