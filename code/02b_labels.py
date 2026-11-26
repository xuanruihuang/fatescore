"""Build the day-2 cell table: covariates, clonal fate labels, per-well reads.

Usage: 02b_labels.py <h5ad> <out.tsv>

One row per day-2 cell, in the order the day-2 cells appear in the input file
(the same order as the arrays written by 02_features.py). No model is fitted.
"""
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

H5AD, OUT = sys.argv[1], sys.argv[2]
MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]

A = ad.read_h5ad(H5AD)
obs = A.obs.copy()
C = A.obsm["X_clone"]
C = C if sp.issparse(C) else sp.csr_matrix(C)
obs["clone"] = np.asarray(C.argmax(1)).ravel()
obs["time"] = obs["time_info"].astype(int)
obs["well"] = obs["Well"].astype(int)
obs["pop"] = obs["Starting population"].map(
    {"Lin-Kit+Sca1+": "LSK", "Lin-Kit+Sca1-": "LK"})
obs["mature"] = obs["state_info"].isin(MATURE)

late = obs[(obs["time"] > 2) & obs["mature"]]


def dominant(counts):
    """Most frequent mature fate; ties broken by the fixed MATURE order."""
    best, bestn = None, -1
    for s in MATURE:
        n = counts.get(s, 0)
        if n > bestn:
            best, bestn = s, n
    return best if bestn > 0 else None


def fate_table(df, key):
    g = df.groupby(["clone", "state_info"], observed=True).size().unstack(
        fill_value=0)
    out = {}
    for c, row in g.iterrows():
        d = row.to_dict()
        out[c] = dict(dom=dominant(d), n=int(row.sum()),
                      n_neu=int(d.get("Neutrophil", 0)),
                      n_mono=int(d.get("Monocyte", 0)))
    return out


pooled = fate_table(late, "all")
by_well = {w: fate_table(late[late.well == w], f"w{w}") for w in (1, 2)}
by_day = {t: fate_table(late[late.time == t], f"d{t}") for t in (4, 6)}

e = obs[obs["time"] == 2].copy()
rows = []
for name, r in e.iterrows():
    c = r["clone"]
    p = pooled.get(c, dict(dom=None, n=0, n_neu=0, n_mono=0))
    rec = dict(cell=name, clone=int(c), pop=r["pop"], library=str(r["Library"]),
               state_d2=str(r["state_info"]),
               n_sis=p["n"], dom=p["dom"], n_neu=p["n_neu"], n_mono=p["n_mono"])
    for w in (1, 2):
        q = by_well[w].get(c, dict(dom=None, n=0, n_neu=0, n_mono=0))
        rec[f"n_sis_w{w}"] = q["n"]
        rec[f"dom_w{w}"] = q["dom"]
        rec[f"n_neu_w{w}"] = q["n_neu"]
        rec[f"n_mono_w{w}"] = q["n_mono"]
    for t in (4, 6):
        q = by_day[t].get(c, dict(dom=None, n=0))
        rec[f"n_sis_d{t}"] = q["n"]
        rec[f"dom_d{t}"] = q["dom"]
    nm = rec["n_neu"] + rec["n_mono"]
    rec["f_neu"] = rec["n_neu"] / nm if nm > 0 else np.nan
    rec["n_neu_mono"] = nm
    rows.append(rec)

T = pd.DataFrame(rows)
T.to_csv(OUT, sep="\t", index=False)
print("rows", len(T))
print("task A (n_sis>=3, dom in Neu/Mono):",
      int(((T.n_sis >= 3) & T.dom.isin(["Neutrophil", "Monocyte"])).sum()))
print("clones:", T[(T.n_sis >= 3) & T.dom.isin(["Neutrophil", "Monocyte"])]
      ["clone"].nunique())
