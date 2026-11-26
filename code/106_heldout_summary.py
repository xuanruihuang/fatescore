"""Summary of the held-out time point test: share of the gap from independent pairing to the floor
that each method closes, (independent - method) / (independent - floor), per seed against the same
seed's references (same evaluation cells), then mean and range over seeds.

Usage: 106_heldout_summary.py <heldout_dir> <out.csv>
"""
import glob
import os
import sys

import pandas as pd

D, OUT = sys.argv[1:3]
rows = []
for refs in sorted(glob.glob(os.path.join(D, "*_refs.csv"))):
    ds = os.path.basename(refs)[:-len("_refs.csv")]
    R = pd.read_csv(refs).pivot(index="seed", columns="method", values="w2")
    parts = [pd.read_csv(f) for f in glob.glob(os.path.join(D, ds + "_*.csv")) if not f.endswith("_refs.csv")]
    M = pd.concat(parts) if parts else pd.DataFrame(columns=["method", "seed", "w2"])
    allm = pd.concat([M[["method", "seed", "w2"]],
                      pd.read_csv(refs)[["method", "seed", "w2"]]])
    for meth, g in allm.groupby("method"):
        g = g.set_index("seed")["w2"].dropna()
        s = [s_ for s_ in g.index if s_ in R.index]
        gap = (R.loc[s, "independent"] - g.loc[s]) / (R.loc[s, "independent"] - R.loc[s, "floor"])
        rows.append(dict(dataset=ds, method=meth, n_seeds=len(s), w2_mean=g.loc[s].mean(),
                         w2_min=g.loc[s].min(), w2_max=g.loc[s].max(), gap_closed_mean=gap.mean(),
                         gap_closed_min=gap.min(), gap_closed_max=gap.max()))
T = pd.DataFrame(rows).sort_values(["dataset", "gap_closed_mean"], ascending=[True, False])
T.round(4).to_csv(OUT, index=False)
pd.set_option("display.width", 200)
print(T.round(3).to_string(index=False))
