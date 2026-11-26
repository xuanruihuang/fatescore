"""Held-out time point: the references, in the shared space of 106_heldout_common.

floor        two disjoint batches of real middle-day cells
independent  random pairs of first- and last-day cells, interpolated linearly at the middle day's
             position in time (the coupling without information, as Waddington-OT's 'R')
ot           exact balanced optimal-transport coupling of the first- and last-day cells (squared
             Euclidean cost in the shared PCA), pairs drawn from it and interpolated the same way
first, last  the first or the last day itself (moving nothing)

Usage: 106_heldout_refs.py <h5ad> <featdir> <dataset> <out.csv> [--seeds 3 --n 2000]
"""
import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("hc", os.path.join(HERE, "106_heldout_common.py"))
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)

ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("dataset")
ap.add_argument("out")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


import ot  # noqa: E402

B, X, day, days = hc.build(a.h5ad, a.featdir)
pca, Z = hc.space(X, day, days)
f = hc.frac(days)
say("%s: days %s, middle at fraction %.3f; cells per day %s" % (a.dataset, days, f,
    [int((day == d).sum()) for d in days]))
rows = []
for seed in range(a.seeds):
    S = hc.draw(day, days, seed, a.n)
    rng = np.random.default_rng(2060 + seed)
    Z0, Z2, E = Z[S["first"]], Z[S["last"]], Z[S["eval"]]
    r = {"floor": hc.w2(Z[S["floor"]], E), "first": hc.w2(Z0, E), "last": hc.w2(Z2, E)}
    perm = rng.permutation(len(Z2))[:len(Z0)] if len(Z2) >= len(Z0) else rng.integers(0, len(Z2), len(Z0))
    r["independent"] = hc.w2((1 - f) * Z0 + f * Z2[perm], E)
    G = ot.emd(np.full(len(Z0), 1 / len(Z0)), np.full(len(Z2), 1 / len(Z2)),
               ot.dist(Z0.astype(np.float64), Z2.astype(np.float64)), numItermax=5_000_000)
    flat = G.ravel() / G.sum()
    k = rng.choice(flat.size, size=len(Z0), p=flat)
    i, j = np.unravel_index(k, G.shape)
    r["ot"] = hc.w2((1 - f) * Z0[i] + f * Z2[j], E)
    for name, v in r.items():
        rows.append(dict(dataset=a.dataset, method=name, seed=seed, w2=v, n_eval=len(E)))
    say("seed %d: %s" % (seed, {k_: round(v_, 3) for k_, v_ in r.items()}))
    pd.DataFrame(rows).to_csv(a.out, index=False)
say("done")
