"""Six-fate version of the LARRY task (a reader asked whether the neutrophil-versus-monocyte
decision is especially favourable to the walk, and whether the result holds with more fates).

Every method's fate probabilities over LARRY's ten terminal populations (feat_all/<block>.npy,
columns in the order of cellrank_log.json; the learned models' and couplings' logs list the same
order) are restricted to the six fates with enough clones (neutrophil, monocyte, basophil,
megakaryocyte, mast cell, erythroid), renormalised (uniform where a cell puts no mass on any of
them), averaged over a clone's day-2 cells, and scored by the macro-averaged one-vs-rest AUROC over
the clones whose dominant fate among >= 3 mature sisters is one of the six. No classifier is fitted.

Gain over the walk (connectivity kernel, n30 p30 d1) with a paired bootstrap over clones (2,000
resamples, seed 97), the minimum detectable difference (0.714 x interval width), a two-sided
bootstrap p-value with Holm's correction over the methods, and, over the 18 walk settings, the number
whose paired interval lies entirely above or below the method. Also, from the same resamples: each
method's gain per fate, and its macro gain over the fates with at least 30 clones.

Usage: 117_multifate.py <featdir> <labels.tsv> <out.csv>   (also writes <out>_perfate.csv)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FD, LAB, OUT = sys.argv[1:4]
CLASSES = ["Neutrophil", "Monocyte", "Baso", "Meg", "Mast", "Erythroid"]
REF = "cr_conn_n30_p30_d1"
METHODS = {"snap_wot": "Waddington-OT", "snap_moscot": "moscot", "snap_statot": "StationaryOT",
           "snap_cospar_state": "CoSpar", "snap_rtk_ub": "CellRank real-time kernel",
           "snap_prescient": "PRESCIENT", "snap_otcfm_g50k": "OT-CFM", "snap_sbcfm_g50k": "SB-CFM",
           "snap_scnode": "scNODE", "snap_scimf": "scIMF", "snap_pisde": "PI-SDE",
           "snap_mioflow": "MIOFlow", "snap_usb": "USB", "snap_pace_fit": "PACE",
           "cr_velo": "CellRank velocity kernel (dynamical)",
           "cr_lad_velocity_ss": "CellRank velocity kernel (steady state)",
           "cr_vkck": "CellRank velocity + connectivity kernel",
           "cr_pseudo": "CellRank pseudotime kernel"}
CONN = ["cr_conn_n%d_p%d_d%d" % (n, p, d) for n in (15, 30, 60) for p in (15, 30, 50) for d in (0, 1)]

order = list(json.load(open(os.path.join(FD, "cellrank_log.json")))["terminal_states"].keys())
cols = [order.index(c) for c in CLASSES]
T = pd.read_csv(LAB, sep="\t")
assert list(T["cell"]) == json.load(open(os.path.join(FD, "d2_names.json")))
sel = ((T["n_sis"] >= 3) & T["dom"].isin(CLASSES)).values
clone = T["clone"].values[sel]
uc, inv = np.unique(clone, return_inverse=True)
yc = pd.Series(T["dom"].values[sel]).groupby(inv).first().values
Y = np.array([[c == k for k in CLASSES] for c in yc], dtype=int)
BIG = [j for j in range(len(CLASSES)) if Y[:, j].sum() >= 30]


def clone_probs(block):
    log = os.path.join(FD, block + "_log.json")
    if os.path.exists(log):
        pops = json.load(open(log)).get("populations")
        if pops is not None:
            assert list(pops) == order, block
    P = np.load(os.path.join(FD, block + ".npy"))[sel][:, cols].astype(float)
    s = P.sum(1, keepdims=True)
    P = np.where(s > 0, P / np.where(s > 0, s, 1), 1.0 / len(CLASSES))
    return pd.DataFrame(P).groupby(inv).mean().values


def per_fate(Yb, Pb):
    """One-vs-rest AUROC per fate; NaN where the resample lacks a fate or has only that fate."""
    return np.array([roc_auc_score(Yb[:, j], Pb[:, j]) if 0 < Yb[:, j].sum() < len(Yb) else np.nan
                     for j in range(len(CLASSES))])


def macro(a, js=None):
    a = a if js is None else a[..., js]
    return np.nanmean(a, axis=-1)


present = [b for b in METHODS if os.path.exists(os.path.join(FD, b + ".npy"))]
for b in METHODS:
    if b not in present:  # e.g. the steady-state velocity kernel: its ladder run kept two-fate shares only
        print("not scored (no %s.npy with all terminal states): %s" % (b, METHODS[b]))
P = {b: clone_probs(b) for b in present + CONN}
rng = np.random.default_rng(97)
n = len(yc)
B = [rng.integers(0, n, n) for _ in range(2000)]
full = {k: per_fate(Y, v) for k, v in P.items()}
boot = {k: np.array([per_fate(Y[i], v[i]) for i in B]) for k, v in P.items()}  # (2000, 6)

rows, frows = [], []
for b in present:
    g = macro(full[b]) - macro(full[REF])
    d = macro(boot[b]) - macro(boot[REF])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1.0 / len(B))
    above = sum(np.percentile(macro(boot[b]) - macro(boot[c]), 2.5) > 0 for c in CONN)
    below = sum(np.percentile(macro(boot[b]) - macro(boot[c]), 97.5) < 0 for c in CONN)
    db = macro(boot[b], BIG) - macro(boot[REF], BIG)
    rows.append(dict(method=METHODS[b], block=b, macro_auroc=macro(full[b]), gain=g, lo=lo, hi=hi,
                     mdd=0.714 * (hi - lo), p=p, n_settings_above=above, n_settings_below=below,
                     gain_big=macro(full[b], BIG) - macro(full[REF], BIG),
                     lo_big=np.percentile(db, 2.5), hi_big=np.percentile(db, 97.5)))
    for j, c in enumerate(CLASSES):
        dj = boot[b][:, j] - boot[REF][:, j]
        dj = dj[np.isfinite(dj)]
        frows.append(dict(method=METHODS[b], fate=c, n_clones=int(Y[:, j].sum()),
                          auroc=full[b][j], walk=full[REF][j], gain=full[b][j] - full[REF][j],
                          lo=np.percentile(dj, 2.5), hi=np.percentile(dj, 97.5)))
R = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
k, run, holm = len(R), 0.0, []
for i, p in enumerate(R.p):
    run = max(run, min(1.0, (k - i) * p))
    holm.append(run)
R["p_holm"] = holm
cv = [macro(full[c]) for c in CONN]
R = pd.concat([R, pd.DataFrame([dict(method="Walk (18 settings)", block="cr_conn_*",
                                     macro_auroc=macro(full[REF]), gain=0.0, lo=min(cv), hi=max(cv))])])
R.round(4).to_csv(OUT, index=False)
F = pd.DataFrame(frows).round(4)
F.to_csv(OUT.replace(".csv", "") + "_perfate.csv", index=False)

pd.set_option("display.width", 250)
print("clones %d; per fate %s; fates with >= 30 clones: %s"
      % (n, dict(zip(CLASSES, Y.sum(0).tolist())), [CLASSES[j] for j in BIG]))
print("walk: reference %.3f, 18 settings %.3f-%.3f" % (macro(full[REF]), min(cv), max(cv)))
print(R.round(4).to_string(index=False))
v = R.iloc[:-1]
print("above the walk: %d; below: %d; not separated: %d of %d; above after Holm: %d"
      % ((v.lo > 0).sum(), (v.hi < 0).sum(), ((v.lo <= 0) & (v.hi >= 0)).sum(), len(v),
         ((v.lo > 0) & (v.p_holm < 0.05)).sum()))
print(F[F.method.isin(["StationaryOT", "CellRank velocity kernel (dynamical)",
                       "CellRank velocity + connectivity kernel"])].to_string(index=False))
