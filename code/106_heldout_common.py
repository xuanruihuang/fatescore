"""Shared pieces of the held-out time point test (106_heldout_*.py).

The middle sampling day is held out: every method is trained on the first and last days only and
asked for the middle day. All distances are computed in one space, a 30-component PCA fitted on
the cells of the first and last days only (so the held-out cells shape neither the training data
nor the representation), on the same object every other analysis uses (41_ot_fate.build). For each
seed the same cells are drawn for everyone: n start cells of the first day, n cells of the last
day, and two disjoint batches of n cells of the middle day (evaluation batch and floor batch).
Distance: exact earth mover's distance with squared Euclidean cost, square root (W2), PCA units.
"""
import importlib.util
import os

import numpy as np
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
N_PCS = 30


def load_otfate():
    spec = importlib.util.spec_from_file_location("otfate", os.path.join(HERE, "41_ot_fate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(h5ad, featdir):
    """Returns the object B, dense log-normalised genes X, sampling day per cell, the three days."""
    otfate = load_otfate()
    B, _ = otfate.build(h5ad, featdir, "time_info", "state_info")
    X = B.X.toarray() if hasattr(B.X, "toarray") else np.asarray(B.X)
    X = X.astype(np.float32)
    day = B.obs["day"].values.astype(float)
    days = sorted(set(day[~np.isnan(day)].tolist()))
    assert len(days) == 3, "the held-out test needs three sampling days: %s" % days
    return B, X, day, days


def space(X, day, days):
    """PCA fitted on the first and last days only; returns (pca, Z for all cells)."""
    m = np.isin(day, [days[0], days[2]])
    pca = PCA(n_components=N_PCS, svd_solver="randomized", random_state=0).fit(X[m])
    return pca, pca.transform(X).astype(np.float32)


def draw(day, days, seed, n):
    rng = np.random.default_rng(1060 + seed)
    first = np.where(day == days[0])[0]
    last = np.where(day == days[2])[0]
    mid = rng.permutation(np.where(day == days[1])[0])
    n_mid = min(n, len(mid) // 2)
    return dict(first=rng.choice(first, size=min(n, len(first)), replace=False),
                last=rng.choice(last, size=min(n, len(last)), replace=False),
                eval=mid[:n_mid], floor=mid[n_mid:2 * n_mid])


def frac(days):
    return (days[1] - days[0]) / (days[2] - days[0])


def w2(P, Q):
    import ot
    M = ot.dist(P.astype(np.float64), Q.astype(np.float64))
    return float(np.sqrt(ot.emd2(np.full(len(P), 1 / len(P)), np.full(len(Q), 1 / len(Q)), M,
                                 numItermax=5_000_000)))
