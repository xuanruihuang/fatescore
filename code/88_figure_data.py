"""Data for the redrawn main figures: pictures of the data rather than statistics.

1. Fig. 1: the LARRY SPRING layout (as deposited) with each day-2 cell's observed
   clonal neutrophil fate bias and the predicted fate bias of the connectivity
   kernel (reference setting), Waddington-OT, StationaryOT, the real-time kernel
   and CellRank's velocity kernel (out/fig_spring.csv; all cells, predictions on
   day-2 cells only).
2. Fig. 2: steady-state velocity and two baseline fields (neighbour average in
   gene space, degradation only) projected into the UMAP of 71_direction_embedding.py
   with scVelo's velocity_graph and velocity_embedding, for streamlines
   (out/fig_stream.npz). Same preprocessing, graph and UMAP seed as 71.

Usage: 88_figure_data.py <raw h5ad> <featdir> <labels.tsv> <outdir>
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
import scanpy as sc
import scvelo as scv

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
scv.settings.verbosity = 0
RAW, FD, LAB, OUT = sys.argv[1:5]


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


# ---- 1. SPRING layout with observed and predicted fate bias
B = ad.read_h5ad(RAW)
obs = B.obs
d2 = json.load(open(os.path.join(FD, "d2_names.json")))
T = pd.read_csv(LAB, sep="\t")
assert list(T["cell"]) == d2
S = pd.DataFrame({"x": obs["SPRING-x"].values, "y": obs["SPRING-y"].values,
                  "state": obs["state_info"].astype(str).values,
                  "day": obs["time_info"].astype(float).astype(int).values},
                 index=obs.index)
pred = {"observed": np.where(T["n_neu_mono"].values >= 3, T["f_neu"].values, np.nan)}
for k, f in (("connectivity", "nm_cr_conn_n30_p30_d1"), ("wot", "nm_snap_wot"),
             ("statot", "nm_snap_statot"), ("realtime", "nm_snap_rtk_ub"),
             ("velocity_kernel", "nm_cr_velo")):
    pred[k] = np.load(os.path.join(FD, f + ".npy")).astype(float).ravel()
for k, v in pred.items():
    S[k] = np.nan
    S.loc[d2, k] = v
S.to_csv(os.path.join(OUT, "fig_spring.csv"))
say("wrote fig_spring.csv")

# ---- 2. fields for streamlines, pipeline as in 71_direction_embedding.py
hvg = json.load(open(os.path.join(FD, "hvg.json")))
state = obs["state_info"].astype(str).values
scv.pp.filter_genes(B, min_shared_counts=20)
scv.pp.normalize_per_cell(B)
sc.pp.log1p(B)
B = B[:, hvg].copy()
scv.pp.neighbors(B, n_pcs=30, n_neighbors=30, method="sklearn")
scv.pp.moments(B)
D = sp.csr_matrix(B.obsp["distances"])
nbr = [D.indices[D.indptr[i]:D.indptr[i + 1]] for i in range(B.n_obs)]
Ms = np.asarray(B.layers["Ms"], dtype=np.float64)
scv.tl.velocity(B, mode="deterministic", vkey="vel_det")
gam = np.nan_to_num(np.asarray(B.var["vel_det_gamma"], dtype=np.float64))
vg = np.asarray(B.var["vel_det_genes"], dtype=bool)
fields = {"steady": np.nan_to_num(np.asarray(B.layers["vel_det"], dtype=np.float64)),
          "minus_gamma_s": np.where(vg, -gam * Ms, 0.0),
          "knn_mean": np.array([Ms[nbr[i]].mean(0) - Ms[i] for i in range(B.n_obs)])}
sc.tl.umap(B, random_state=0)
U = np.asarray(B.obsm["X_umap"], dtype=np.float64)
emb = {}
for name, F in fields.items():
    key = "f_" + name
    B.layers[key] = F
    B.var[key + "_genes"] = np.abs(F).sum(0) > 0
    scv.tl.velocity_graph(B, vkey=key, xkey="Ms", n_jobs=16, show_progress_bar=False)
    scv.tl.velocity_embedding(B, basis="umap", vkey=key)
    emb[name] = np.asarray(B.obsm[key + "_umap"], dtype=np.float32)
    say("projected %s" % name)
np.savez_compressed(os.path.join(OUT, "fig_stream.npz"), umap=U.astype(np.float32),
                    state=state.astype(str), **emb)
say("wrote fig_stream.npz")
