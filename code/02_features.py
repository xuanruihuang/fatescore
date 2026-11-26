"""Build every feature matrix the evaluation needs, for one kNN-graph mode.

Usage: 02_features.py <h5ad> <outdir> <graph_mode>   graph_mode in {all, d2}

"all" is the standard workflow: normalize, select HVGs and compute first-order
moments on all 49,302 cells, so a day-2 cell's smoothed counts average over
neighbours that include day-4 and day-6 cells. "d2" restricts the object to the
4,638 day-2 cells before the same steps, so the graph connects day-2 cells only.

Saved arrays all have one row per day-2 cell, in the order of the day-2 cells in
the input file. Nothing here uses a fate label. The processed object is written
to <outdir>/pipeline.h5ad so that 02c_cellrank.py can add fate probabilities
without recomputing the dynamics.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import scvelo as scv

H5AD, OUTDIR, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
assert MODE in ("all", "d2")
# optional, for a second dataset: which column holds the sampling time, which
# of its values are the early cells, and which column holds the state
# annotation. The defaults are the LARRY file's.
TIME_COL = sys.argv[4] if len(sys.argv) > 4 else "time_info"
EARLY = (sys.argv[5].split(",") if len(sys.argv) > 5 else ["2"])
STATE_COL = sys.argv[6] if len(sys.argv) > 6 else "state_info"
os.makedirs(OUTDIR, exist_ok=True)
N_TOP_GENES = 2000
N_PCS, N_NEIGHBORS = 30, 30
log = {}


def say(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


say("mode=%s reading %s" % (MODE, H5AD))
A = ad.read_h5ad(H5AD)
tcol = A.obs[TIME_COL].astype(str)
A.obs["time"] = pd.to_numeric(tcol, errors="coerce")
is_d2 = tcol.isin(EARLY).values
d2_names = A.obs_names[is_d2].tolist()
log["n_cells_total"] = int(A.n_obs)
log["n_d2"] = int(is_d2.sum())

B = A[is_d2].copy() if MODE == "d2" else A.copy()
say("object for pipeline: %s" % (B.shape,))
# scvelo 0.3.4 dropped n_top_genes from filter_and_normalize and removed
# filter_genes_dispersion, so the same four steps are spelled out with scanpy:
# gene filter, per-cell normalisation of X and the spliced/unspliced layers,
# log1p of X, then the 2000 most variable genes.
scv.pp.filter_genes(B, min_shared_counts=20)
scv.pp.normalize_per_cell(B)
sc.pp.log1p(B)
sc.pp.highly_variable_genes(B, n_top_genes=N_TOP_GENES, flavor="seurat")
B = B[:, B.var["highly_variable"].values].copy()
hvg = B.var_names.tolist()
log["n_hvg"] = len(hvg)
say("after gene selection: %s" % (B.shape,))
scv.pp.moments(B, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS)
say("moments done")

kept = [n for n in d2_names if n in B.obs_names]
assert len(kept) == len(d2_names), (len(kept), len(d2_names))
d2_idx_in_B = np.array([B.obs_names.get_loc(n) for n in d2_names])


def save(name, M):
    M = np.asarray(M, dtype=np.float32)
    nan = int(np.isnan(M).sum())
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(os.path.join(OUTDIR, name + ".npy"), M[d2_idx_in_B])
    log[name + "_nan_frac"] = nan / M.size
    say("saved %s %s nan_frac=%.4g" % (name, M[d2_idx_in_B].shape, nan / M.size))


save("Ms", B.layers["Ms"])
save("Mu", B.layers["Mu"])

def layer_of(obj, layer):
    """The named layer, or the sum of the splicing layers when a deposit does
    not carry a total-count layer of its own. Sparsity is preserved."""
    if layer in obj.layers:
        return obj.layers[layer]
    parts = [obj.layers[k] for k in ("spliced", "unspliced", "ambiguous")
             if k in obj.layers]
    M = parts[0]
    for q in parts[1:]:
        M = M + q
    return M


def dense(obj, layer):
    M = layer_of(obj, layer)
    return M.toarray() if sp.issparse(M) else np.asarray(M)


raw = A[is_d2, hvg]
for layer in ("spliced", "unspliced", "matrix"):
    np.save(os.path.join(OUTDIR, "raw_%s.npy" % layer),
            dense(raw, layer).astype(np.float32))
allraw = A[is_d2]
for layer in ("spliced", "unspliced", "matrix"):
    t = layer_of(allraw, layer).sum(1)
    np.save(os.path.join(OUTDIR, "total_%s.npy" % layer),
            np.asarray(t, dtype=np.float32).ravel())
log["has_matrix_layer"] = "matrix" in A.layers
say("raw day-2 counts saved (total-count layer present: %s)"
    % log["has_matrix_layer"])

say("scvelo deterministic")
scv.tl.velocity(B, mode="deterministic", vkey="vel_det")
save("vel_det", B.layers["vel_det"])

say("scvelo dynamical: recover_dynamics")
scv.tl.recover_dynamics(B, n_jobs=16, var_names="all")
scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
save("vel_dyn", B.layers["vel_dyn"])
log["n_genes_fit"] = int(np.isfinite(B.var["fit_alpha"].values).sum())

try:
    from velovi import VELOVI, preprocess_data
    say("veloVI")
    V = B.copy()
    V = preprocess_data(V, spliced_layer="Ms", unspliced_layer="Mu")
    VELOVI.setup_anndata(V, spliced_layer="Ms", unspliced_layer="Mu")
    m = VELOVI(V)
    m.train(max_epochs=500, early_stopping=True, enable_progress_bar=False)
    vv = np.asarray(m.get_velocity(n_samples=25, velo_statistic="mean"),
                    dtype=np.float32)
    full = np.zeros((B.n_obs, B.n_vars), dtype=np.float32)
    full[:, [B.var_names.get_loc(g) for g in V.var_names]] = vv
    save("vel_velovi", full)
    log["velovi_genes"] = int(V.n_vars)
except Exception as e:  # recorded, never silently skipped
    log["velovi_error"] = "%s: %s" % (type(e).__name__, e)
    say("veloVI FAILED: %s" % e)

B.obs["state_info"] = A.obs[STATE_COL].reindex(B.obs_names).astype(str).values
B.write_h5ad(os.path.join(OUTDIR, "pipeline.h5ad"), compression="gzip")
say("pipeline object written")

json.dump(hvg, open(os.path.join(OUTDIR, "hvg.json"), "w"))
json.dump(d2_names, open(os.path.join(OUTDIR, "d2_names.json"), "w"))
json.dump(log, open(os.path.join(OUTDIR, "log.json"), "w"), indent=1,
          default=str)
say("done")
