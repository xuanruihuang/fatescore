"""How much each arm gains when the neighbour graph is allowed to span time.

Usage: 14_cross_graph.py <ev_prefix_A> <ev_prefix_B> <out.csv>

Prefix A is the graph built on early cells only, prefix B the graph built on all
cells. The two runs score the same clones with the same labels, so the
difference per arm can be bootstrapped over clones as a paired quantity. A
positive difference means that averaging a cell's counts over neighbours
sampled at later time points makes its clone's fate easier to predict.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

A, B, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ta = json.load(open(A + "_task.json"))
tb = json.load(open(B + "_task.json"))
assert ta["clones"] == tb["clones"] and ta["y"] == tb["y"], \
    "the two runs did not score the same clones"
y = np.array(ta["y"])
groups = np.array(ta["clones"])
OA = np.load(A + "_oof.npy")
OB = np.load(B + "_oof.npy")
arms = [a for a in ta["arms"] if a in tb["arms"]]
ia = {a: i for i, a in enumerate(ta["arms"])}
ib = {a: i for i, a in enumerate(tb["arms"])}

clones = np.unique(groups)


def clone_tab(p):
    return (pd.DataFrame(dict(p=p, y=y, g=groups)).groupby("g")
            .agg(p=("p", "mean"), y=("y", "first")).reindex(clones))


def auroc(d):
    return np.nan if d["y"].nunique() < 2 else roc_auc_score(d["y"], d["p"])


tabs = {}
for a in arms:
    tabs[(a, "early")] = clone_tab(OA[ia[a]].mean(0))
    tabs[(a, "all")] = clone_tab(OB[ib[a]].mean(0))

NB = 2000
rng = np.random.default_rng(7)
boot = {k: np.empty(NB) for k in tabs}
for b in range(NB):
    take = rng.integers(0, len(clones), len(clones))
    for k, t in tabs.items():
        boot[k][b] = auroc(t.iloc[take])

rows = []
for a in arms:
    d = boot[(a, "all")] - boot[(a, "early")]
    rows.append(dict(arm=a,
                     auroc_early_graph=auroc(tabs[(a, "early")]),
                     auroc_all_graph=auroc(tabs[(a, "all")]),
                     diff=auroc(tabs[(a, "all")]) - auroc(tabs[(a, "early")]),
                     lo=np.nanpercentile(d, 2.5),
                     hi=np.nanpercentile(d, 97.5)))
R = pd.DataFrame(rows).sort_values("diff", ascending=False)
R.to_csv(OUT, index=False)
print(R.to_string(index=False))
