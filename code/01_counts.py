"""Count the design before any model is fitted.

Everything printed here is a property of the experiment, not a result: how many
day-2 cells have clonal sisters at day 4/6, how many sisters each has, how the
sisters' fates are distributed, and how often sisters in the two independently
sampled wells agree. These numbers fix the sample sizes and the ceiling quoted
in the preregistration.
"""
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

path = sys.argv[1]
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else None
design_rows = []


def record(name, value):
    design_rows.append(dict(quantity=name, value=value))
A = ad.read_h5ad(path)
obs = A.obs.copy()
C = A.obsm["X_clone"]
if not sp.issparse(C):
    C = sp.csr_matrix(C)
clone = np.asarray(C.argmax(1)).ravel()
obs["clone"] = clone
obs["time"] = obs["time_info"].astype(int)
obs["well"] = obs["Well"].astype(int)
obs["pop"] = obs["Starting population"].map(
    {"Lin-Kit+Sca1+": "LSK", "Lin-Kit+Sca1-": "LK"})

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]
obs["mature"] = obs["state_info"].isin(MATURE)

print("=== time x well")
print(pd.crosstab(obs["time"], obs["well"]).to_string())
print("\n=== time x starting population")
print(pd.crosstab(obs["time"], obs["pop"]).to_string())
print("\n=== state_info x time")
print(pd.crosstab(obs["state_info"], obs["time"]).to_string())
print("\n=== library x time x well (first 40)")
print(obs.groupby(["Library", "time", "well"], observed=True)
         .size().to_string()[:4000])

print("\n=== unspliced depth per cell")
u = np.asarray(A.layers["unspliced"].sum(1)).ravel()
s = np.asarray(A.layers["spliced"].sum(1)).ravel()
for name, v in [("unspliced", u), ("spliced", s), ("u/(u+s)", u / (u + s))]:
    print(f"{name}: median {np.median(v):.4g} q10 {np.percentile(v,10):.4g} "
          f"q90 {np.percentile(v,90):.4g}")
ng = (np.asarray((A.layers["unspliced"] > 0).sum(1)).ravel())
print(f"genes with >=1 unspliced UMI per cell: median {np.median(ng):.0f}")

# --- clone-level bookkeeping -------------------------------------------------
early = obs[obs["time"] == 2]
late = obs[obs["time"] > 2]
late_m = late[late["mature"]]

print("\n=== clones")
print("clones with >=1 day-2 cell:", early["clone"].nunique())
print("clones with >=1 late cell:", late["clone"].nunique())
print("clones with >=1 mature late cell:", late_m["clone"].nunique())
shared = set(early["clone"]) & set(late_m["clone"])
print("clones spanning day-2 and a mature late cell:", len(shared))

nsis = late_m.groupby("clone").size()
nsis_d4 = late_m[late_m["time"] == 4].groupby("clone").size()
nsis_d6 = late_m[late_m["time"] == 6].groupby("clone").size()
early["n_sis"] = early["clone"].map(nsis).fillna(0).astype(int)
early["n_sis_d4"] = early["clone"].map(nsis_d4).fillna(0).astype(int)
early["n_sis_d6"] = early["clone"].map(nsis_d6).fillna(0).astype(int)

print("\n=== day-2 cells by number of mature clonal sisters at day 4/6")
for k in [1, 2, 3, 5, 7, 10, 20]:
    e = early[early["n_sis"] >= k]
    print(f">= {k:3d} sisters: {len(e):5d} day-2 cells, "
          f"{e['clone'].nunique():4d} clones")
print("day-2 cells with >=1 sister, split by day:",
      f"d4 only {int(((early.n_sis_d4>0)&(early.n_sis_d6==0)).sum())}",
      f"d6 only {int(((early.n_sis_d6>0)&(early.n_sis_d4==0)).sum())}",
      f"both {int(((early.n_sis_d4>0)&(early.n_sis_d6>0)).sum())}")

# fate composition per clone. Ties are broken by the fixed MATURE order, the
# same rule 02b_labels.py uses, so that the counts and the labels agree.
fate = (late_m.groupby(["clone", "state_info"], observed=True).size()
        .unstack(fill_value=0))
fate = fate.loc[:, fate.sum(0) > 0]


def dominant(row):
    best, bestn = None, -1
    for st in MATURE:
        n = row.get(st, 0)
        if n > bestn:
            best, bestn = st, n
    return best if bestn > 0 else None


dom = fate.apply(dominant, axis=1)
print("\n=== dominant mature fate of clones (>=1 mature sister)")
print(dom.value_counts().to_string())

print("\n=== dominant fate of clones with a day-2 cell and >=3 mature sisters")
sel = sorted(set(early[early["n_sis"] >= 3]["clone"]))
print(dom.loc[sel].value_counts().to_string())

print("\n=== day-2 cells by dominant fate (clone >=3 mature sisters)")
e3 = early[early["n_sis"] >= 3].copy()
e3["dom"] = e3["clone"].map(dom)
print(e3["dom"].value_counts().to_string())
print("by starting population:")
print(pd.crosstab(e3["dom"], e3["pop"]).to_string())

nm = ["Neutrophil", "Monocyte"]
e3nm = e3[e3["dom"].isin(nm)]
print(f"\nNeu-vs-Mono task: {len(e3nm)} day-2 cells, "
      f"{e3nm['clone'].nunique()} clones, "
      f"majority class {e3nm['dom'].value_counts(normalize=True).max():.3f}")
print(pd.crosstab(e3nm["dom"], e3nm["pop"]).to_string())
print("day-2 state_info of those cells:")
print(e3nm["state_info"].value_counts().to_string())

record("task A cells", len(e3nm))
record("task A clones", int(e3nm["clone"].nunique()))
record("task A cell-level majority",
       float(e3nm["dom"].value_counts(normalize=True).max()))
_cl = e3nm.groupby("clone")["dom"].first()
record("task A clone-level majority",
       float(_cl.value_counts(normalize=True).max()))
record("task A cells annotated undifferentiated at day 2",
       int((e3nm["state_info"] == "Undifferentiated").sum()))
_gate = pd.crosstab(e3nm["dom"], e3nm["pop"])
record("accuracy of the gate majority fate",
       float(_gate.max(0).sum() / len(e3nm)))

# --- sister ceiling: two independently sampled wells -------------------------
print("\n=== sister-based ceiling (wells 1 vs 2, mature late cells)")
w = (late_m.groupby(["clone", "well", "state_info"], observed=True).size()
     .unstack(fill_value=0))
w = w.loc[:, w.sum(0) > 0]
tot = w.sum(1)
both = [c for c in w.index.get_level_values(0).unique()
        if (c, 1) in w.index and (c, 2) in w.index]
rows = []
for c in both:
    a, b = w.loc[(c, 1)], w.loc[(c, 2)]
    rows.append((c, int(a.sum()), int(b.sum()), dominant(a), dominant(b)))
D = pd.DataFrame(rows, columns=["clone", "n1", "n2", "dom1", "dom2"])
print("clones with mature cells in both wells:", len(D))
for k in [1, 2, 3, 5]:
    d = D[(D.n1 >= k) & (D.n2 >= k)]
    if len(d) == 0:
        continue
    agree = (d.dom1 == d.dom2).mean()
    # chance from the marginal frequencies of dom1/dom2
    p1 = d.dom1.value_counts(normalize=True)
    p2 = d.dom2.value_counts(normalize=True)
    chance = sum(p1.get(k2, 0) * p2.get(k2, 0) for k2 in set(p1.index) | set(p2.index))
    print(f"  >= {k} mature cells per well: n={len(d):4d} "
          f"agreement {agree:.3f} chance {chance:.3f}")
    record(f"cross-well dominant-fate agreement, >={k} mature per well", agree)
    record(f"clones, >={k} mature per well", len(d))
    record(f"chance agreement, >={k} mature per well", chance)
    dnm = d[d.dom1.isin(nm) & d.dom2.isin(nm)]
    if len(dnm):
        print(f"      Neu/Mono only: n={len(dnm):4d} "
              f"agreement {(dnm.dom1==dnm.dom2).mean():.3f}")
        record(f"cross-well agreement, neutrophil or monocyte, >={k} per well",
               float((dnm.dom1 == dnm.dom2).mean()))
        record(f"clones, neutrophil or monocyte, >={k} per well", len(dnm))
print("\n=== also: day-4 vs day-6 sisters of the same clone")
wt = (late_m.groupby(["clone", "time", "state_info"], observed=True).size()
      .unstack(fill_value=0))
wt = wt.loc[:, wt.sum(0) > 0]
bt = [c for c in wt.index.get_level_values(0).unique()
      if (c, 4) in wt.index and (c, 6) in wt.index]
rows = []
for c in bt:
    a, b = wt.loc[(c, 4)], wt.loc[(c, 6)]
    rows.append((int(a.sum()), int(b.sum()), dominant(a), dominant(b)))
T = pd.DataFrame(rows, columns=["n4", "n6", "dom4", "dom6"])
for k in [1, 3]:
    t = T[(T.n4 >= k) & (T.n6 >= k)]
    print(f"  >= {k} per time point: n={len(t):4d} "
          f"agreement {(t.dom4 == t.dom6).mean():.3f}")
    record(f"day-4 vs day-6 agreement, >={k} per time point",
           float((t.dom4 == t.dom6).mean()))
    record(f"clones, day-4 and day-6, >={k} per time point", float(len(t)))

if OUT_CSV:
    pd.DataFrame(design_rows).to_csv(OUT_CSV, index=False)
    print("\nwrote", OUT_CSV)
