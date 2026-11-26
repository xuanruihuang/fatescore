"""One table across the lineage data sets: each method's clone-level AUROC as a gain
over the matched predictor without dynamics.

The matched predictor is CellRank's connectivity kernel (n30 p30, density
normalised) on the same cells, to the same terminal states. For each method:
the AUROC, its difference from the kernel with a paired bootstrap over clones
(2,000 resamples, seed 97, as 97_fate_score.py), and the minimum detectable
difference of that comparison (0.714 x interval width). The kernel's own range
over its 18 graph settings is reported alongside.

LARRY is scored here from the shares written by 51_fate_share.py (main task of
66_unsupervised_reference.py: clones with at least three sister cells dominated by
neutrophils or monocytes); the CellTag-multi data sets are read from the outputs of
97_fate_score.py.

Usage: 99_calibrated_table.py <larry_featdir_all> <larry_labels.tsv> <ctm_scores.csv>
       <ctr_scores.csv> <out.csv>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FALL, LAB, CTM, CTR, OUT = sys.argv[1:6]
NAMES = {"snap_statot": "StationaryOT", "snap_wot": "Waddington-OT", "snap_moscot": "moscot",
         "snap_prescient": "PRESCIENT", "snap_otcfm_g50k": "OT-CFM", "snap_sbcfm_g50k": "SB-CFM",
         "snap_rtk_ub": "CellRank real-time kernel",
         "snap_scnode": "scNODE", "snap_scimf": "scIMF", "snap_pisde": "PI-SDE", "snap_mioflow": "MIOFlow",
         "snap_cospar_state": "CoSpar", "snap_usb": "USB", "snap_pace_fit": "PACE",
         "cr_velo": "CellRank velocity kernel (dynamical)",
         "cr_lad_velocity_ss": "CellRank velocity kernel (steady state)",
         "cr_vkck": "CellRank velocity + connectivity kernel",
         "cr_pseudo": "CellRank pseudotime kernel"}
REF = "cr_conn_n30_p30_d1"

# LARRY
T = pd.read_csv(LAB, sep="\t")
m = (T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])
im = np.where(m.values)[0]
ucl, inv = np.unique(T.loc[m, "clone"].values, return_inverse=True)
ydom = (T.loc[m].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)


def share(name):
    M = np.load(os.path.join(FALL, name + ".npy"))
    if M.ndim == 2 and M.shape[1] > 1:
        s = M[:, 0] + M[:, 1]
        return np.where(s > 0, M[:, 0] / np.where(s > 0, s, 1), 0.5)
    return M.ravel()


def clone_mean(s):
    return np.bincount(inv, weights=s[im], minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))


rng = np.random.default_rng(97)
B = rng.integers(0, len(ucl), size=(2000, len(ucl)))
ref = clone_mean(share("nm_" + REF))
rows = []
kern = [roc_auc_score(ydom, clone_mean(share(os.path.basename(f)[:-4])))
        for f in sorted(glob.glob(os.path.join(FALL, "nm_cr_conn_n*.npy")))]
for k, lab in NAMES.items():
    if not os.path.exists(os.path.join(FALL, "nm_" + k + ".npy")):
        continue
    c = clone_mean(share("nm_" + k))
    d = [roc_auc_score(ydom[b], c[b]) - roc_auc_score(ydom[b], ref[b]) for b in B
         if ydom[b].min() != ydom[b].max()]
    lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append(dict(dataset="LARRY", method=lab, auroc_clone=roc_auc_score(ydom, c),
                     kernel_auroc=roc_auc_score(ydom, ref), kernel_min=min(kern), kernel_max=max(kern),
                     gain=roc_auc_score(ydom, c) - roc_auc_score(ydom, ref), lo=lo, hi=hi,
                     mdd=0.714 * (hi - lo), n_clones=len(ucl), n_pos=int(ydom.sum())))

# CellTag-multi
for ds, fn in (("CellTag-multi haematopoiesis", CTM), ("CellTag-multi reprogramming", CTR)):
    S = pd.read_csv(fn).set_index("predictor")
    kr = S.loc[[i for i in S.index if i.startswith("cr_conn_all_")], "auroc_clone"]
    for k, lab in NAMES.items():
        if k not in S.index:
            continue
        r = S.loc[k]
        rows.append(dict(dataset=ds, method=lab, auroc_clone=r.auroc_clone,
                         kernel_auroc=S.loc["cr_conn_all_n30_p30_d1", "auroc_clone"],
                         kernel_min=kr.min(), kernel_max=kr.max(), gain=r.diff_vs_ref,
                         lo=r.diff_lo, hi=r.diff_hi, mdd=r.mdd, n_clones=int(r.n_clones),
                         n_pos=int(r.n_pos_clones)))
R = pd.DataFrame(rows)
R["separated"] = np.where(R.lo > 0, "above", np.where(R.hi < 0, "below", "not separated"))
R.round(4).to_csv(OUT, index=False)
print(R.round(3).to_string(index=False))
