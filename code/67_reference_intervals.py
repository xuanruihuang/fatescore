"""Paired clone-bootstrap intervals: methods against unsupervised references.

66_unsupervised_reference.py gives point values. Here every method's share,
used as it is (no classifier), is compared with each unsupervised reference
without dynamics on the same cells, with a paired bootstrap over clones
(2,000 resamples; a clone's cells are drawn together):
  - the cell-level Pearson r with the clone's observed neutrophil share on the
    continuous task (the metric published LARRY rankings use), when the label
    file has the neutrophil and monocyte counts;
  - the clone-level AUROC for the dominant fate on the main task (the share
    averaged within clones).
The reference settings are not chosen on the result: the comparison is made
against every one of the 18 connectivity-kernel settings and five
inverse-distance exponents, and the output keeps each pair, so a summary can
report the default setting (30 neighbours, 30 principal components, density
normalized) and the range over settings.

Usage: 67_reference_intervals.py <featdir> <labels.tsv> <out_prefix>
         [--neu Neutrophil --mono Monocyte] [--methods a,b,...] [--seed 67]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--neu", default="Neutrophil")
ap.add_argument("--mono", default="Monocyte")
ap.add_argument("--methods", default="nm_snap_statot,nm_snap_wot,nm_snap_moscot,"
                "nm_snap_prescient,nm_snap_prescient_g,nm_cr_velo,nm_cr_pseudo")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=67)
ap.add_argument("--ref-dir", default="",
                help="take the references from this feature directory instead "
                     "(same day-2 cells in the same order), e.g. day-2-only ones")
ap.add_argument("--exclude-mature", action="store_true",
                help="drop day-2 cells annotated as mature on day 2")
a = ap.parse_args()
RD = a.ref_dir or a.featdir

T = pd.read_csv(a.labels, sep="\t")
if a.ref_dir:
    import json
    assert json.load(open(os.path.join(a.featdir, "d2_names.json"))) == \
        json.load(open(os.path.join(RD, "d2_names.json"))), "day-2 cells differ"
KEEP = (T["state_d2"].astype(str).values == "Undifferentiated") if a.exclude_mature \
    else np.ones(len(T), bool)


def share(name, d=None):
    M = np.load(os.path.join(d or a.featdir, name + ".npy"))
    if M.ndim == 2 and M.shape[1] > 1:
        s = M[:, 0] + M[:, 1]
        return np.where(s > 0, M[:, 0] / np.where(s > 0, s, 1), 0.5)
    return M.ravel().astype(float)


methods = [m for m in a.methods.split(",")
           if os.path.exists(os.path.join(a.featdir, m + ".npy"))]
refs = sorted(os.path.basename(f)[:-4] for f in
              glob.glob(os.path.join(RD, "nm_cr_conn_*.npy")))
refs += sorted(os.path.basename(f)[:-4] for f in
               glob.glob(os.path.join(RD, "snap_idw_g*.npy")))
names = methods + refs
S = {n: share(n) for n in methods}
S.update({n: share(n, RD) for n in refs})
rng = np.random.default_rng(a.seed)

# main task: clone-level AUROC
m = (T["n_sis"] >= 3) & T["dom"].isin([a.neu, a.mono]) & KEEP
im = np.where(m.values)[0]
D = pd.DataFrame({n: S[n][im] for n in names})
D["g"] = T.loc[m, "clone"].values
D["y"] = (T.loc[m, "dom"].values == a.neu).astype(int)
C = D.groupby("g").mean()
y = C["y"].values.astype(int)
nC = len(C)
Bm = rng.integers(0, nC, size=(a.boot, nC))
auc = {n: roc_auc_score(y, C[n].values) for n in names}
aucb = {}
for n in names:
    v = C[n].values
    aucb[n] = np.array([roc_auc_score(y[b], v[b]) for b in Bm])

rows = []
# continuous task: cell-level r, resampling clones
has_r = "f_neu" in T.columns and "n_neu_mono" in T.columns
if has_r:
    c = (T["n_neu_mono"] >= 3) & T["f_neu"].notna() & KEEP
    ic = np.where(c.values)[0]
    yc = T.loc[c, "f_neu"].values.astype(float)
    gc = T.loc[c, "clone"].values
    ug, inv = np.unique(gc, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    Bc = rng.integers(0, len(ug), size=(a.boot, len(ug)))
    idx_b = [np.concatenate([members[k] for k in b]) for b in Bc]
    from scipy.stats import rankdata
    rr = {n: np.corrcoef(S[n][ic], yc)[0, 1] for n in names}
    sr = {n: np.corrcoef(rankdata(S[n][ic]), rankdata(yc))[0, 1] for n in names}
    sb = {}
    for n in names:
        x = S[n][ic]
        sb[n] = np.array([np.corrcoef(rankdata(x[i]), rankdata(yc[i]))[0, 1] for i in idx_b])
    rb = {}
    for n in names:
        x = S[n][ic]
        rb[n] = np.array([np.corrcoef(x[i], yc[i])[0, 1] for i in idx_b])

for mth in methods:
    for ref in refs:
        rec = dict(method=mth, reference=ref,
                   auroc_method=auc[mth], auroc_ref=auc[ref],
                   auroc_diff=auc[mth] - auc[ref])
        d = aucb[mth] - aucb[ref]
        rec["auroc_lo"], rec["auroc_hi"] = np.percentile(d, [2.5, 97.5])
        if has_r:
            rec.update(r_method=rr[mth], r_ref=rr[ref], r_diff=rr[mth] - rr[ref])
            d = rb[mth] - rb[ref]
            rec["r_lo"], rec["r_hi"] = np.percentile(d, [2.5, 97.5])
            rec.update(rho_method=sr[mth], rho_ref=sr[ref], rho_diff=sr[mth] - sr[ref])
            d = sb[mth] - sb[ref]
            rec["rho_lo"], rec["rho_hi"] = np.percentile(d, [2.5, 97.5])
        rows.append(rec)
R = pd.DataFrame(rows).round(4)
R.to_csv(a.out + "_pairs.csv", index=False)
print("clones (main task) %d; continuous task %s" % (nC, has_r and len(ug)))
dflt = "nm_cr_conn_n30_p30_d1"
cols = ["method", "auroc_method", "auroc_ref", "auroc_diff", "auroc_lo", "auroc_hi"]
if has_r:
    cols += ["r_method", "r_ref", "r_diff", "r_lo", "r_hi"]
print("against the default connectivity kernel (%s):" % dflt)
print(R[R.reference == dflt][cols].to_string(index=False))
conn = R[R.reference.str.startswith("nm_cr_conn_")]
agg = {"auroc_diff": ["min", "max"], "auroc_lo": "min", "auroc_hi": "max"}
if has_r:
    agg.update({"r_diff": ["min", "max"], "r_lo": "min", "r_hi": "max"})
G = conn.groupby("method").agg(agg).round(4)
print("over the 18 connectivity-kernel settings:")
print(G.to_string())
# how many settings have an interval excluding zero, in each direction
for k in (["auroc"] + (["r"] if has_r else [])):
    pos = conn.groupby("method").apply(lambda x: int((x[k + "_lo"] > 0).sum()))
    neg = conn.groupby("method").apply(lambda x: int((x[k + "_hi"] < 0).sum()))
    print("%s: settings where the method is significantly above / below the kernel" % k)
    print(pd.DataFrame({"above": pos, "below": neg}).to_string())
