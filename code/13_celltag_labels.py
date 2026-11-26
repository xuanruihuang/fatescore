"""Build the early-cell table for the CellTag reprogramming time course.

Usage: 13_celltag_labels.py <h5ad> <out.tsv> [tag_column]

The output has the same columns as the LARRY table written by 02b_labels.py, so
the same evaluation scripts run on it unchanged. The mapping is:

  clone      the CellTag integer, prefixed with its time course, because the
             integers are assigned within a time course
  pop        the time course, "tc1" or "tc2"
  dom        the clone's dominant late fate, "iEP" or "Fibroblast"
  n_neu      number of iEP sisters;  n_mono  number of Fibroblast sisters
  _w1 / _w2  the two independent late read-outs, day 21 and day 28, which take
             the place of the two wells in the LARRY experiment
  state_d2   the early cell's own annotation, which is usually missing here

Early cells are days 3, 6 and 9; late cells are days 21 and 28 annotated iEP or
Fibroblast. Nothing is fitted here.
"""
import sys

import numpy as np
import pandas as pd
import anndata as ad

H5AD, OUT = sys.argv[1], sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else "CellTagD0_85k"
EARLY = [3.0, 6.0, 9.0]
LATE = [21.0, 28.0]
POS, NEG = "iEP", "Fibroblast"

A = ad.read_h5ad(H5AD)
obs = A.obs.copy()
obs["day"] = pd.to_numeric(obs["reprogramming_day"].astype(str),
                           errors="coerce")
obs["tc"] = "tc" + obs["timecourse"].astype(str)
obs["ct"] = obs["cell_type"].astype(str)
v = pd.to_numeric(obs[TAG], errors="coerce")
obs["clone_key"] = np.where(v.notna(),
                            obs["tc"].astype(str) + ":" + v.astype(str), None)

d = obs[obs["clone_key"].notna() & obs["day"].notna()]
late = d[d["day"].isin(LATE) & d["ct"].isin([POS, NEG])]
# the row order must match the arrays written by 02_features.py, which cover
# every early cell, tagged or not; untagged cells get empty fate columns and
# are dropped later by the sister-count threshold


def counts(df):
    g = df.groupby(["clone_key", "ct"], observed=True).size().unstack(
        fill_value=0)
    for f in (POS, NEG):
        if f not in g.columns:
            g[f] = 0
    g["n"] = g[POS] + g[NEG]
    g["dom"] = np.where(g[POS] >= g[NEG], POS, NEG)
    return g


pooled = counts(late)
by_day = {t: counts(late[late["day"] == t]) for t in LATE}

early = obs[obs["day"].isin(EARLY)].copy()
rows = []
for name, r in early.iterrows():
    c = r["clone_key"]
    if c is None or (isinstance(c, float) and np.isnan(c)) or c not in pooled.index:
        p = dict(n=0, dom=None, pos=0, neg=0)
    else:
        q = pooled.loc[c]
        p = dict(n=int(q["n"]), dom=q["dom"], pos=int(q[POS]), neg=int(q[NEG]))
    rec = dict(cell=name, clone=c if c is not None else "", pop=r["tc"],
               library=str(r["day"]),
               state_d2=r["ct"], n_sis=p["n"], dom=p["dom"],
               n_neu=p["pos"], n_mono=p["neg"], day=r["day"])
    for w, t in ((1, 21.0), (2, 28.0)):
        g = by_day[t]
        if c is not None and c in g.index:
            q = g.loc[c]
            rec["n_sis_w%d" % w] = int(q["n"])
            rec["dom_w%d" % w] = q["dom"]
            rec["n_neu_w%d" % w] = int(q[POS])
            rec["n_mono_w%d" % w] = int(q[NEG])
        else:
            rec["n_sis_w%d" % w] = 0
            rec["dom_w%d" % w] = None
            rec["n_neu_w%d" % w] = 0
            rec["n_mono_w%d" % w] = 0
    for w, t in (("d4", 21.0), ("d6", 28.0)):
        g = by_day[t]
        has = c is not None and c in g.index
        rec["n_sis_%s" % w] = int(g.loc[c, "n"]) if has else 0
        rec["dom_%s" % w] = g.loc[c, "dom"] if has else None
    nm = rec["n_neu"] + rec["n_mono"]
    rec["n_neu_mono"] = nm
    rec["f_neu"] = rec["n_neu"] / nm if nm > 0 else np.nan
    rows.append(rec)

T = pd.DataFrame(rows)
T.to_csv(OUT, sep="\t", index=False)
print("tag column", TAG)
print("early cells", len(T), "clones", T["clone"].nunique())
for k in (1, 3):
    s = T[(T.n_sis >= k) & T.dom.isin([POS, NEG])]
    print(">=%d late sisters: %d cells, %d clones, %s"
          % (k, len(s), s["clone"].nunique(), s["dom"].value_counts().to_dict()))
print("by time course:", T["pop"].value_counts().to_dict())
print("by day:", T["day"].value_counts().sort_index().to_dict())
