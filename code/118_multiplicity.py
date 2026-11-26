"""Multiplicity check for the main table: a two-sided bootstrap p-value for each of the 48
method-data set gains over the walk, and Holm's correction over all 48.

The gains and their paired bootstrap over clones are recomputed exactly as in the main table
(LARRY as in 99_calibrated_table.py, from the neutrophil-versus-monocyte shares of feat_all/nm_*;
CellTag-multi as in 97_fate_score.py, positive population over all terminal populations), with the
same resamples (2,000, seed 97). The script checks that its intervals reproduce calibrated_table.csv
and then reports, for each pair, p = 2 min(P(d <= 0), P(d >= 0)) over the resamples (at least
1/2,000), the Holm-adjusted p, and the pair's status at a familywise level of 0.05.

Usage: 118_multiplicity.py <larry_featdir_all> <larry_labels.tsv> <featdir_ctm> <featdir_ctr>
       <calibrated_table.csv> <out.csv>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FALL, LAB, FCTM, FCTR, TAB, OUT = sys.argv[1:7]
NAMES = {"snap_statot": "StationaryOT", "snap_wot": "Waddington-OT", "snap_moscot": "moscot",
         "snap_prescient": "PRESCIENT", "snap_otcfm_g50k": "OT-CFM", "snap_sbcfm_g50k": "SB-CFM",
         "snap_rtk_ub": "CellRank real-time kernel",
         "snap_scnode": "scNODE", "snap_scimf": "scIMF", "snap_pisde": "PI-SDE", "snap_mioflow": "MIOFlow",
         "snap_cospar_state": "CoSpar", "snap_usb": "USB", "snap_pace_fit": "PACE",
         "cr_velo": "CellRank velocity kernel (dynamical)",
         "cr_lad_velocity_ss": "CellRank velocity kernel (steady state)",
         "cr_vkck": "CellRank velocity + connectivity kernel",
         "cr_pseudo": "CellRank pseudotime kernel"}


def boot_diffs(ydom, c, ref):
    rng = np.random.default_rng(97)
    B = rng.integers(0, len(ydom), size=(2000, len(ydom)))
    return np.array([roc_auc_score(ydom[b], c[b]) - roc_auc_score(ydom[b], ref[b]) for b in B
                     if ydom[b].min() != ydom[b].max()])


rows = []

# LARRY, as 99_calibrated_table.py
T = pd.read_csv(LAB, sep="\t")
m = (T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])
im = np.where(m.values)[0]
ucl, inv = np.unique(T.loc[m, "clone"].values, return_inverse=True)
ydom = (T.loc[m].groupby("clone")["dom"].first().reindex(ucl).values == "Neutrophil").astype(int)


def share_larry(name):
    M = np.load(os.path.join(FALL, name + ".npy"))
    if M.ndim == 2 and M.shape[1] > 1:
        s = M[:, 0] + M[:, 1]
        return np.where(s > 0, M[:, 0] / np.where(s > 0, s, 1), 0.5)
    return M.ravel()


def cm_larry(s):
    return np.bincount(inv, weights=s[im], minlength=len(ucl)) / np.bincount(inv, minlength=len(ucl))


ref = cm_larry(share_larry("nm_cr_conn_n30_p30_d1"))
for k, lab in NAMES.items():
    if os.path.exists(os.path.join(FALL, "nm_" + k + ".npy")):
        c = cm_larry(share_larry("nm_" + k))
        rows.append(("LARRY", lab, roc_auc_score(ydom, c) - roc_auc_score(ydom, ref), boot_diffs(ydom, c, ref)))

# CellTag-multi, as 97_fate_score.py
for ds, fd, pos in (("CellTag-multi haematopoiesis", FCTM, "Mono"),
                    ("CellTag-multi reprogramming", FCTR, "reprogramming")):
    L = pd.read_csv(os.path.join(fd, "labels.tsv"), sep="\t")
    assert list(L["cell"]) == json.load(open(os.path.join(fd, "d2_names.json")))
    task = (L["n_mature"] >= 3).values
    it = np.where(task)[0]
    u, iv = np.unique(L.loc[task, "clone"].values, return_inverse=True)
    yd = (L.loc[task].groupby("clone")["dom"].first().reindex(u).values == pos).astype(int)

    def share(name):
        P = np.load(os.path.join(fd, name + ".npy"))
        pops = json.load(open(os.path.join(fd, name + "_log.json")))["populations"]
        tot = P.sum(1)
        return np.where(tot > 0, P[:, pops.index(pos)] / np.where(tot > 0, tot, 1), 0.5)

    def cm(s):
        return np.bincount(iv, weights=s[it], minlength=len(u)) / np.bincount(iv, minlength=len(u))

    r = cm(share("cr_conn_all_n30_p30_d1"))
    for k, lab in NAMES.items():
        if os.path.exists(os.path.join(fd, k + ".npy")) and os.path.exists(os.path.join(fd, k + "_log.json")):
            c = cm(share(k))
            rows.append((ds, lab, roc_auc_score(yd, c) - roc_auc_score(yd, r), boot_diffs(yd, c, r)))

C = pd.read_csv(TAB).set_index(["dataset", "method"])
out = []
for ds, lab, g, d in rows:
    if (ds, lab) not in C.index:
        continue
    lo, hi = np.percentile(d, [2.5, 97.5])
    t = C.loc[(ds, lab)]
    assert abs(lo - t.lo) < 1e-3 and abs(hi - t.hi) < 1e-3 and abs(g - t.gain) < 1e-3, (ds, lab, lo, t.lo)
    p = max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1.0 / len(d))
    out.append(dict(dataset=ds, method=lab, gain=g, lo=lo, hi=hi, p=p, separated=t.separated))
R = pd.DataFrame(out).sort_values("p").reset_index(drop=True)
k, run, adj = len(R), 0.0, []
for i, p in enumerate(R.p):
    run = max(run, min(1.0, (k - i) * p))
    adj.append(run)
R["p_holm"] = adj
R["separated_holm"] = np.where(R.p_holm < 0.05, np.where(R.gain > 0, "above", "below"), "not separated")
R.round(4).to_csv(OUT, index=False)
print(R.round(4).to_string(index=False))
print("pairs: %d; per-pair 95%%: %s; Holm over all pairs: %s"
      % (k, R.separated.value_counts().to_dict(), R.separated_holm.value_counts().to_dict()))
