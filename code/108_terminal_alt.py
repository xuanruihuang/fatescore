"""Terminal-state sensitivity of the clonal-fate comparison, definitions not covered by
86_terminal_states.py (which has: every cell of each annotated mature population, day-6
cells only, 30 core cells, and GPCCA macrostates at fixed 10/12/15 states).

On the same all-cell pipeline (as 86 / 92: 2,000 hvg, 30 PCs, exact 30-NN graph, scVelo
dynamical velocity), the velocity kernel and the connectivity kernel are recomputed with:
  nm_only       only the two scored populations (neutrophil, monocyte) absorbing; every other
                mature cell is transient;
  nm_only_day6  the same, day-6 cells only;
  population    the paper's definition (sanity: must reproduce 86's 0.784 / 0.809);
  gpcca_auto    CellRank's own estimator on each kernel with its documented automatic choices:
                GPCCA.fit(n_states=None) (eigengap of the real Schur decomposition) and
                predict_terminal_states(method="stability") (default threshold); macrostates
                named by majority annotation. Needs PETSc/SLEPc (env_gpcca).
Scored as in 86: clone-level AUROC on the 567-clone main task (neutrophil- vs monocyte-
dominated clones, cell scores averaged within clones), cell r on cells with >= 3 Neu/Mo
sisters, and velocity-minus-connectivity AUROC with a paired bootstrap over clones (2,000).
No fate label is used to build any kernel or terminal set.

Usage: 108_terminal_alt.py <raw h5ad> <featdir> <labels.tsv> <out.csv>
"""
import json
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
import os  # noqa: E402
import scvelo as scv  # noqa: E402
import scanpy as sc  # noqa: E402
import cellrank as cr  # noqa: E402
from sklearn.neighbors import KNeighborsTransformer  # noqa: E402

RAW, FD, LAB, OUT = sys.argv[1:5]


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


def score(F, names, tag, kernel):
    neu = F[:, [j for j, n in enumerate(names) if n.startswith("Neutrophil")]].sum(1)[d2]
    mono = F[:, [j for j, n in enumerate(names) if n.startswith("Monocyte")]].sum(1)[d2]
    mass = neu + mono
    fb = np.where(mass > 0, neu / np.where(mass > 0, mass, 1), 0.5)
    cs = pd.DataFrame(dict(c=clone[main], s=fb[main], y=ymain[main])).groupby("c").agg(
        s=("s", "mean"), y=("y", "first"))
    return dict(terminal=tag, kernel=kernel,
                r_cell=float(np.corrcoef(fb[cont], fobs[cont])[0, 1]) if fb[cont].std() > 0 else np.nan,
                auroc_clone=float(roc_auc_score(cs.y, cs.s)),
                mass_nm_median=float(np.median(mass[cont]))), cs


vk = cr.kernels.VelocityKernel(B, vkey="vel_dyn")
vk.compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(B)
ck.compute_transition_matrix()
kernels = {"velocity": vk, "connectivity": ck}
rng = np.random.default_rng(108)
rows = []


def paired(res, tag):
    a, b = res["velocity"][1], res["connectivity"][1]
    assert (a.index == b.index).all()
    n = len(a)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        if a.y.values[i].min() == a.y.values[i].max():
            continue
        bs.append(roc_auc_score(a.y.values[i], a.s.values[i]) - roc_auc_score(b.y.values[i], b.s.values[i]))
    for kn in res:
        rr = res[kn][0]
        rr["vk_minus_ck_auroc"] = res["velocity"][0]["auroc_clone"] - res["connectivity"][0]["auroc_clone"]
        rr["vk_minus_ck_lo"] = float(np.percentile(bs, 2.5))
        rr["vk_minus_ck_hi"] = float(np.percentile(bs, 97.5))
        rows.append(rr)
        say(json.dumps(rr))
    pd.DataFrame(rows).to_csv(OUT, index=False)


pops = {s: np.asarray(state == s) for s in present}
defs = [("population", present, [pops[s] for s in present]),
        ("nm_only", ["Neutrophil", "Monocyte"], [pops["Neutrophil"], pops["Monocyte"]]),
        ("nm_only_day6", ["Neutrophil", "Monocyte"],
         [pops["Neutrophil"] & (tday == 6), pops["Monocyte"] & (tday == 6)])]
for tag, names, masks in defs:
    res = {}
    for kn, k in kernels.items():
        r, cs = score(absorb(k.transition_matrix, masks), names, tag, kn)
        r["n_terminal_cells"] = int(np.any(masks, axis=0).sum())
        res[kn] = (r, cs)
    paired(res, tag)

# CellRank's estimator with its automatic choices
for kn, k in kernels.items():
    rec = dict(terminal="gpcca_auto", kernel=kn)
    try:
        g = cr.estimators.GPCCA(k)
        g.fit(n_states=None, cluster_key="state_info", n_cells=30)
        ms = list(g.macrostates.cat.categories)
        g.predict_terminal_states(method="stability")
        ts = g.terminal_states
        names = list(ts.cat.categories)
        rec.update(n_macrostates=len(ms), macrostates=";".join(ms), terminal_states=";".join(names))
        masks = [np.asarray(ts == n) for n in names]
        if not any(n.startswith("Neutrophil") for n in names) or not any(n.startswith("Monocyte") for n in names):
            rec["note"] = "no neutrophil or no monocyte terminal state"
        else:
            r, _ = score(absorb(k.transition_matrix, masks), names, "gpcca_auto", kn)
            rec.update(r)
            rec["n_terminal_cells"] = int(np.any(masks, axis=0).sum())
    except Exception as e:  # report, do not hide
        rec["note"] = "failed: %s" % str(e)[:300]
    rows.append(rec)
    say(json.dumps(rec, default=str))
    pd.DataFrame(rows).to_csv(OUT, index=False)
say("done")
