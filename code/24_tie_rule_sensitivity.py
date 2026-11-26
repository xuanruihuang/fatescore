"""How much of the label's measured precision is the tie rule.

Usage: 24_tie_rule_sensitivity.py <h5ad> <out.csv>

A clone's dominant fate is ambiguous whenever two mature states are tied, and
the preregistration fixed one rule for breaking those ties. The rule is a
convention, not a property of the data, so the cross-well agreement it produces
has a component that another convention would have changed. This measures that
component by recomputing the same agreement under three rules on the same
clones: the preregistered fixed order of mature states, alphabetical order
(what a plain idxmax gives), and the order the states appear in the object.

The clone, well and mature-state definitions are copied from 01_counts.py so
that the preregistered row here reproduces the number quoted in the paper.
Nothing is fitted and no model is involved.
"""
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

H5AD, OUT = sys.argv[1], sys.argv[2]
MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]
NM = ["Neutrophil", "Monocyte"]

A = ad.read_h5ad(H5AD)
obs = A.obs.copy()
C = A.obsm["X_clone"]
if not sp.issparse(C):
    C = sp.csr_matrix(C)
obs["clone"] = np.asarray(C.argmax(1)).ravel()
obs["time"] = obs["time_info"].astype(int)
obs["well"] = obs["Well"].astype(int)
obs["mature"] = obs["state_info"].isin(MATURE)
late_m = obs[(obs["time"] > 2) & obs["mature"]]

w = (late_m.groupby(["clone", "well", "state_info"], observed=True).size()
     .unstack(fill_value=0))
w = w.loc[:, w.sum(0) > 0]
ORDERS = {
    "preregistered fixed order": [m for m in MATURE if m in w.columns],
    "alphabetical": sorted(w.columns),
    "order of appearance in the object": list(w.columns),
}


def dominant(row, order):
    best, bestn = None, -1
    for st in order:
        n = row.get(st, 0)
        if n > bestn:
            best, bestn = st, n
    return best if bestn > 0 else None


both = [c for c in w.index.get_level_values(0).unique()
        if (c, 1) in w.index and (c, 2) in w.index]
rows = []
for name, order in ORDERS.items():
    rec = [(dominant(w.loc[(c, 1)], order), dominant(w.loc[(c, 2)], order))
           for c in both]
    D = pd.DataFrame(rec, columns=["dom1", "dom2"])
    dnm = D[D.dom1.isin(NM) & D.dom2.isin(NM)]
    rows.append(dict(rule=name, clones=len(D),
                     agreement=float((D.dom1 == D.dom2).mean()),
                     nm_clones=len(dnm),
                     nm_agreement=float((dnm.dom1 == dnm.dom2).mean())))
O = pd.DataFrame(rows)
O.loc[len(O)] = dict(
    rule="spread across the three rules", clones=len(both),
    agreement=float(O["agreement"].max() - O["agreement"].min()),
    nm_clones=float(O["nm_clones"].max() - O["nm_clones"].min()),
    nm_agreement=float(O["nm_agreement"].max() - O["nm_agreement"].min()))
O.to_csv(OUT, index=False)
print(O.to_string(index=False))
