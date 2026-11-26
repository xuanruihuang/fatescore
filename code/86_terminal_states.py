"""Sensitivity of the CellRank kernels to the definition of terminal states (referee).

02c_cellrank.py and 29_expression_kernel.py make every cell of each annotated
mature population (>= 30 cells) absorbing. CellRank usually absorbs into a small
set of the most terminal cells per state. Here, on the all-cell pipeline rebuilt
as in 70_combined_kernel.py and 85_solver_check.py, the velocity kernel and the
connectivity kernel are recomputed with four terminal-state definitions:
  population: every cell of each mature population (the paper's definition);
  day6: only the day-6 cells of each mature population;
  core30: per population, the 30 day-6 cells closest to the population's day-6
      centroid in the 30 principal components;
  gpcca: CellRank's own estimator (GPCCA macrostates on each kernel, named by
      the majority annotation, mature ones kept, 30 cells each), if it runs.
Each is scored on the day-2 cells as in the paper: Pearson r between the
predicted and observed neutrophil fate bias over cells whose clone has >= 3
neutrophil or monocyte descendants, clone-level AUROC on the main task, the
share of those cells with bias in [0.93, 1], the median neutrophil+monocyte
absorption mass, and the velocity-minus-connectivity AUROC difference with a
paired bootstrap over clones (2,000 resamples).

Usage: 86_terminal_states.py <raw h5ad> <featdir> <labels.tsv> <out.csv> [--only-gpcca]
(GPCCA needs PETSc/SLEPc for the Krylov Schur decomposition on ~50k cells: env_gpcca)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import numpy.testing as npt
import anndata as ad
from sklearn.metrics import roc_auc_score

_orig = npt.assert_array_equal


def _compat(*x, **k):
    if "x" in k:
        k["actual"] = k.pop("x")
    if "y" in k:
        k["desired"] = k.pop("y")
    return _orig(*x, **k)


npt.assert_array_equal = _compat
for cls in (sp.csr_array, sp.csc_array):
    if not hasattr(cls, "getnnz"):
        cls.getnnz = cls._getnnz
import scvelo as scv  # noqa: E402
import scanpy as sc  # noqa: E402
import cellrank as cr  # noqa: E402
from sklearn.neighbors import KNeighborsTransformer  # noqa: E402

RAW, FD, LAB, OUT = sys.argv[1:5]
# --only-gpcca: skip the hand-made definitions (already computed) and run CellRank's estimator
ONLY_GPCCA = "--only-gpcca" in sys.argv


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


B = ad.read_h5ad(RAW)
hvg = json.load(open(os.path.join(FD, "hvg.json")))
scv.pp.filter_genes(B, min_shared_counts=20)
scv.pp.normalize_per_cell(B)
sc.pp.log1p(B)
B = B[:, hvg].copy()
sc.tl.pca(B, n_comps=30, svd_solver="arpack")
sc.pp.neighbors(B, n_neighbors=30, n_pcs=30, use_rep="X_pca",
                transformer=KNeighborsTransformer(n_neighbors=30, metric="euclidean",
                                                  algorithm="brute", n_jobs=8))
scv.pp.moments(B, n_pcs=30, n_neighbors=30)
scv.tl.recover_dynamics(B, n_jobs=16, var_names="all")
scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
scv.tl.velocity_graph(B, vkey="vel_dyn", n_jobs=8, show_progress_bar=False)
say("pipeline rebuilt %s" % (B.shape,))

d2_names = json.load(open(os.path.join(FD, "d2_names.json")))
d2 = np.array([B.obs_names.get_loc(n) for n in d2_names])
T = pd.read_csv(LAB, sep="\t")
assert list(T["cell"]) == d2_names
present = list(json.load(open(os.path.join(FD, "cellrank_log.json")))["terminal_states"].keys())
state = B.obs["state_info"].astype(str).values
tday = B.obs["time_info"].astype(str).str.extract(r"(\d+)")[0].astype(int).values
X = B.obsm["X_pca"]

cont = (T["n_neu_mono"] >= 3).values
fobs = T["f_neu"].values
main = ((T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])).values
clone = T["clone"].values
ymain = (T["dom"].values == "Neutrophil").astype(int)


def absorb(Tm, masks):
    Tm = sp.csr_matrix(Tm)
    is_term = np.any(masks, axis=0)
    trans = ~is_term
    Q = Tm[trans][:, trans].tocsr()
    R = np.column_stack([np.asarray(Tm[trans][:, m].sum(1)).ravel() for m in masks])
    Xa = np.zeros_like(R)
    for it in range(200000):
        Xa = Q @ Xa + R
        if it % 50 == 0 and float(np.abs(1 - Xa.sum(1)).max()) < 1e-8:
            break
    F = np.zeros((Tm.shape[0], len(masks)))
    F[trans] = Xa
    for j, m in enumerate(masks):
        F[m, j] = 1.0
    return F


def clone_scores(fb):
    d = pd.DataFrame(dict(c=clone[main], s=fb[main], y=ymain[main])).groupby("c").agg(
        s=("s", "mean"), y=("y", "first"))
    return d


def score(F, names, tag, kernel):
    neu = F[:, [j for j, n in enumerate(names) if n.startswith("Neutrophil")]].sum(1)
    mono = F[:, [j for j, n in enumerate(names) if n.startswith("Monocyte")]].sum(1)
    neu, mono = neu[d2], mono[d2]
    mass = neu + mono
    fb = np.where(mass > 0, neu / np.where(mass > 0, mass, 1), 0.5)
    cs = clone_scores(fb)
    r = dict(terminal=tag, kernel=kernel, n_terminal_cells=None,
             r_cell=float(np.corrcoef(fb[cont], fobs[cont])[0, 1]),
             auroc_clone=float(roc_auc_score(cs.y, cs.s)),
             share_093_100=float(np.mean(fb[cont] >= 0.93)),
             mass_nm_median=float(np.median(mass[cont])))
    return r, cs


def variants():
    pops = [np.asarray(state == s) for s in present]
    yield "population", present, pops
    yield "day6", present, [m & (tday == 6) for m in pops]
    core = []
    for m in pops:
        mm = m & (tday == 6)
        idx = np.where(mm)[0]
        if idx.size == 0:
            idx = np.where(m)[0]
        c = X[idx].mean(0)
        keep = idx[np.argsort(np.linalg.norm(X[idx] - c, axis=1))[:30]]
        z = np.zeros(B.n_obs, bool); z[keep] = True
        core.append(z)
    yield "core30", present, core


rows = []
kernels = {}
vk = cr.kernels.VelocityKernel(B, vkey="vel_dyn"); vk.compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(B); ck.compute_transition_matrix()
kernels = {"velocity": vk, "connectivity": ck}
rng = np.random.default_rng(86)
for tag, names, masks in ([] if ONLY_GPCCA else variants()):
    res = {}
    for kn, k in kernels.items():
        r, cs = score(absorb(k.transition_matrix, masks), names, tag, kn)
        r["n_terminal_cells"] = int(np.any(masks, axis=0).sum())
        res[kn] = (r, cs)
        say(json.dumps(r))
    a, b = res["velocity"][1], res["connectivity"][1]
    assert (a.index == b.index).all()
    n = len(a)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        if a.y.values[i].min() == a.y.values[i].max():
            continue
        bs.append(roc_auc_score(a.y.values[i], a.s.values[i]) - roc_auc_score(b.y.values[i], b.s.values[i]))
    for kn in kernels:
        rr = res[kn][0]
        rr["vk_minus_ck_auroc"] = res["velocity"][0]["auroc_clone"] - res["connectivity"][0]["auroc_clone"]
        rr["vk_minus_ck_lo"] = float(np.percentile(bs, 2.5))
        rr["vk_minus_ck_hi"] = float(np.percentile(bs, 97.5))
        rows.append(rr)
    pd.DataFrame(rows).to_csv(OUT, index=False)

# CellRank's own terminal states: GPCCA macrostates on each kernel (Krylov Schur
# decomposition), named by the majority annotation, mature ones kept, 30 cells each
for kn, k in kernels.items():
    g = cr.estimators.GPCCA(k)
    try:
        g.compute_schur(n_components=25, method="krylov")
    except Exception as e:
        rows.append(dict(terminal="gpcca", kernel=kn, gpcca_states="schur failed: %s" % str(e)[:200]))
        say("schur %s failed: %s" % (kn, e)); continue
    for ns in (10, 12, 15):
        try:
            g.compute_macrostates(n_states=ns, cluster_key="state_info")
            ms = list(g.macrostates.cat.categories)
            keep = [m for m in ms if any(m.startswith(p) for p in present)]
            g.set_terminal_states(states=keep, n_cells=30)
            ts = g.terminal_states
            names = list(ts.cat.categories)
            masks = [np.asarray(ts == n) for n in names]
            r, _ = score(absorb(k.transition_matrix, masks), names, "gpcca_%d" % ns, kn)
            r["n_terminal_cells"] = int(np.any(masks, axis=0).sum())
            r["gpcca_states"] = ";".join(names)
            r["all_macrostates"] = ";".join(ms)
            rows.append(r)
            say(json.dumps(r))
        except Exception as e:
            rows.append(dict(terminal="gpcca_%d" % ns, kernel=kn, gpcca_states="failed: %s" % str(e)[:200]))
            say("gpcca %s %d failed: %s" % (kn, ns, e))
        pd.DataFrame(rows).to_csv(OUT, index=False)
say("done")
