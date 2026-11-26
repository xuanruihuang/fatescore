"""Shared pieces for USB and PACE under this analysis' protocol (110_usb.py, 110_pace.py).

Fate task (as 100_sctimebench_fate.py): the shared object of 41_ot_fate.build (this directory's
highly variable genes, log-normalised, 30-component PCA), all sampling days; each early cell is
carried forward by the trained model to the last day, and the endpoint is classified by the
fraction of its k = 20 nearest annotated last-day cells in each terminal population. Training is
checked without clonal labels: first-day cells carried to each later day, W2 against real cells,
next to the same-day floor and the earlier day itself (the same references and cells as 100).

Held-out time point (as 106_heldout_*.py): imported from 106_heldout_common.py.
"""
import importlib.util
import json
import os
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def heldout_common():
    return _load("hc", "106_heldout_common.py")


class FateData:
    """Shared object in its 30-component PCA, early cells, terminal populations, classifier."""

    def __init__(self, h5ad, featdir, k=20, fit_n=2000):
        otfate = _load("otfate", "41_ot_fate.py")
        B, d2 = otfate.build(h5ad, featdir, "time_info", "state_info")
        self.pops = otfate.populations(featdir, B.obs["state"].values)
        self.Z = np.asarray(B.obsm["X_pca"], dtype=np.float32)
        self.day = B.obs["day"].values.astype(float)
        self.days = sorted(set(self.day[~np.isnan(self.day)].tolist()))
        self.state = B.obs["state"].values
        self.d2i = np.array([B.obs_names.get_loc(n) for n in d2]) if isinstance(d2[0], str) else np.asarray(d2)
        pop_idx = {p: j for j, p in enumerate(self.pops)}
        last = np.where(self.day == self.days[-1])[0]
        self.nn = NearestNeighbors(n_neighbors=k).fit(self.Z[last])
        self.lab = np.array([pop_idx.get(x, -1) for x in self.state[last]])
        rng = np.random.default_rng(940)  # the same fit cells as 100
        self.fit_idx = {}
        for d in self.days:
            pool = np.where(self.day == d)[0]
            self.fit_idx[d] = rng.choice(pool, size=min(2 * fit_n, len(pool)), replace=False)
        self.half = {d: len(self.fit_idx[d]) // 2 for d in self.days}

    def frames(self):
        """Cells of each sampling day, in day order (PCA coordinates)."""
        return [self.Z[self.day == d] for d in self.days]

    def classify(self, E):
        """E: n x 30 endpoints -> n x populations (fraction of the k nearest last-day cells)."""
        _, nb = self.nn.kneighbors(E)
        L = self.lab[nb]
        return np.stack([(L == j).mean(1) for j in range(len(self.pops))], 1)

    def fit_reference(self, w2):
        ref, days = {}, self.days
        Z, fi, h = self.Z, self.fit_idx, self.half
        for d0, d1 in zip(days[:-1], days[1:]):
            ref["day%g_vs_day%g" % (d0, d1)] = w2(Z[fi[d0][:h[d0]]], Z[fi[d1][:h[d1]]])
        for d in days[1:]:
            ref["day%g_vs_day%g" % (d, d)] = w2(Z[fi[d][:h[d]]], Z[fi[d][h[d]:2 * h[d]]])
        return ref

    def fit_start(self):
        return self.Z[self.fit_idx[self.days[0]][:self.half[self.days[0]]]]

    def fit_target(self, d):
        return self.Z[self.fit_idx[d][:self.half[d]]]

    def save(self, featdir, name, P, log):
        np.save(os.path.join(featdir, name + ".npy"), P.astype(np.float32))
        json.dump(dict(populations=self.pops, method=name), open(os.path.join(featdir, name + "_log.json"), "w"))
        json.dump(log, open(os.path.join(featdir, name + "_run_log.json"), "w"), indent=1)


def w2(P, Q, cap=4000, seed=0):
    import ot
    rng = np.random.default_rng(seed)
    if len(P) > cap:
        P = P[rng.choice(len(P), cap, replace=False)]
    if len(Q) > cap:
        Q = Q[rng.choice(len(Q), cap, replace=False)]
    M = ot.dist(np.asarray(P, np.float64), np.asarray(Q, np.float64))
    return float(np.sqrt(ot.emd2(np.full(len(P), 1 / len(P)), np.full(len(Q), 1 / len(Q)), M,
                                 numItermax=5_000_000)))
