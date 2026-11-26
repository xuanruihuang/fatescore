"""Count the state-fate design in the CellTag reprogramming time course.

Usage: 12_celltag_counts.py <h5ad>

Clone identifiers are integers assigned within each of the two independent
time courses, so a clone key has to carry the time course as well as the
tagging round. Early cells are the ones sampled on days 3 to 9, late cells
those on days 21 and 28, and a late cell has a fate if it is annotated iEP or
Fibroblast. Nothing is fitted here.
"""
import sys

import numpy as np
import pandas as pd
import anndata as ad

A = ad.read_h5ad(sys.argv[1])
obs = A.obs.copy()
obs["day"] = pd.to_numeric(obs["reprogramming_day"].astype(str),
                           errors="coerce")
obs["tc"] = obs["timecourse"].astype(str)
obs["ct"] = obs["cell_type"].astype(str)

print("cells with a day:", int(obs["day"].notna().sum()))
print(pd.crosstab(obs["day"], obs["tc"], dropna=False).to_string())
print("\ncell_type by day")
print(pd.crosstab(obs["day"], obs["ct"], dropna=False).to_string())
print("\nreprogramming flag by day")
print(pd.crosstab(obs["day"], obs["reprogramming"].astype(str),
                  dropna=False).to_string())

u = np.asarray(A.layers["unspliced"].sum(1)).ravel()
s = np.asarray(A.layers["spliced"].sum(1)).ravel()
print("\nper-cell depth: spliced median %.0f, unspliced median %.0f, "
      "unspliced fraction median %.3f"
      % (np.median(s), np.median(u), np.median(u / (u + s))))

TAGS = ["CellTagD0_48k", "CellTagD3_48k", "CellTagD13_48k",
        "CellTagD0_85k", "CellTagD3_85k", "CellTagD13_85k"]
EARLY = [3, 6, 9]
LATE = [21, 28]
FATES = ["iEP", "Fibroblast"]

for tag in TAGS:
    v = pd.to_numeric(obs[tag], errors="coerce")
    ok = v.notna()
    key = obs["tc"].astype(str) + ":" + v.astype("Int64").astype(str)
    key = key.where(ok)
    d = obs.assign(clone=key)
    d = d[d["clone"].notna() & d["day"].notna()]
    early = d[d["day"].isin(EARLY)]
    late = d[d["day"].isin(LATE) & d["ct"].isin(FATES)]
    cnt = late.groupby(["clone", "ct"], observed=True).size().unstack(
        fill_value=0)
    for f in FATES:
        if f not in cnt.columns:
            cnt[f] = 0
    cnt["n"] = cnt[FATES].sum(1)
    cnt["dom"] = np.where(cnt["iEP"] >= cnt["Fibroblast"], "iEP", "Fibroblast")
    shared = sorted(set(early["clone"]) & set(cnt.index))
    e = early[early["clone"].isin(shared)].copy()
    e["n_sis"] = e["clone"].map(cnt["n"])
    e["dom"] = e["clone"].map(cnt["dom"])
    print("\n=== %s" % tag)
    print("  tagged cells %d, clones %d; early cells %d; "
          "late cells with a fate %d" % (int(ok.sum()), int(key.nunique()),
                                         len(early), len(late)))
    print("  clones with an early cell and a fated late cell: %d" % len(shared))
    for k in (1, 2, 3, 5):
        ee = e[e["n_sis"] >= k]
        if not len(ee):
            continue
        comp = ee["dom"].value_counts()
        print("    >=%d late sisters: %d early cells, %d clones, dominant %s"
              % (k, len(ee), ee["clone"].nunique(), comp.to_dict()))
    if len(e):
        print("  early cells by day:",
              e["day"].value_counts().sort_index().to_dict())
        print("  early cell_type:", e["ct"].value_counts().to_dict())
    # ceiling: the two time courses are separate experiments, so the usable
    # internal replicate is day 21 against day 28 within a clone
    w = (d[d["day"].isin(LATE) & d["ct"].isin(FATES)]
         .groupby(["clone", "day", "ct"], observed=True).size()
         .unstack(fill_value=0))
    for f in FATES:
        if f not in w.columns:
            w[f] = 0
    idx = w.index.get_level_values(0)
    both = [c for c in idx.unique()
            if (c, 21.0) in w.index and (c, 28.0) in w.index]
    if len(both) >= 10:
        a = w.loc[[(c, 21.0) for c in both]]
        b = w.loc[[(c, 28.0) for c in both]]
        da = np.where(a["iEP"].values >= a["Fibroblast"].values, 1, 0)
        db = np.where(b["iEP"].values >= b["Fibroblast"].values, 1, 0)
        print("  day-21 vs day-28 dominant-fate agreement: %.3f over %d clones"
              % float((da == db).mean()) if False else
              "  day-21 vs day-28 dominant-fate agreement: %.3f over %d clones"
              % ((da == db).mean(), len(both)))
