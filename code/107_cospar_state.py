"""CoSpar (Wang et al. 2022, Nat Biotechnol) from state information alone, on our fate tasks.

CoSpar infers a transition map between sampling days and enforces coherence by smoothing over the
cell-cell similarity graph. Its state-only mode (cs.tmap.infer_Tmap_from_state_info_alone) gives
every cell a unique barcode, so no clonal label enters. Settings are the function's documented
defaults (OT initialisation with shortest-path cost, smooth_array [15, 10, 5], sparsity 0.2,
CoSpar_KNN 20, full similarity matrix); --init HighVar gives the package's other state-only
initialisation. Nothing is tuned on clonal scores.

Object: the same as 41_ot_fate.py builds for this feature directory (its genes, median-scaled
total-count normalisation, log1p, 30 principal components; days as strings, state_info as given);
X_emb is SPRING for LARRY and the first two components otherwise (CoSpar uses it only for plots).
Map: all earlier days -> the last day. Read-out: CoSpar's own fate map (cs.tl.fate_map,
map_backward=True) towards this directory's terminal populations (cellrank_log.json order),
method 'norm-sum' (the default of its fate_bias, written as snap_cospar_state[_tag]) and 'sum'
(the transition probability into each population, written as snap_cospar_state_sum[_tag]).
Rows: the directory's early cells (d2_names.json). For LARRY also nm_<name>.npy, the neutrophil
share among neutrophil + monocyte as 51_fate_share.py writes it (0.5 where both are zero).

Usage: 107_cospar_state.py <h5ad> <featdir> [--init OT|HighVar] [--larry]
"""
import argparse
import json
import os
import time

import anndata as ad
import cospar as cs
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

N_PCS = 30

# scanpy 1.11 calls .getnnz(axis=1) on the k-nearest-neighbour graph, which scikit-learn now returns
# as a scipy sparse *array*; sparse arrays dropped getnnz. Restore it with the same meaning.
if not hasattr(sp.csr_array, "getnnz"):
    def _getnnz(self, axis=None):
        if axis is None:
            return self.nnz
        c = sp.csr_matrix(self)
        return np.diff(c.indptr) if axis in (1, -1) else np.bincount(c.indices, minlength=c.shape[1])
    sp.csr_array.getnnz = _getnnz


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("--init", default="OT", choices=["OT", "HighVar"])
ap.add_argument("--larry", action="store_true")
a = ap.parse_args()
tag = "" if a.init == "OT" else "_highvar"

hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
d2_names = d2_names if isinstance(d2_names, list) else d2_names.get("names", d2_names)
pops = list(json.load(open(os.path.join(a.featdir, "cellrank_log.json")))["terminal_states"].keys())
B = ad.read_h5ad(a.h5ad)
assert B.obs_names.is_unique
keep = [g for g in hvg if g in B.var_names]
assert len(keep) == len(hvg)
B = B[:, keep].copy()
day = pd.to_numeric(B.obs["time_info"], errors="coerce").astype(float)
B = B[day.notna().values].copy()
day = day[day.notna()]
sc.pp.normalize_total(B, target_sum=None)
sc.pp.log1p(B)
sc.tl.pca(B, n_comps=N_PCS, svd_solver="arpack")
days = sorted(set(day.tolist()))
tstr = np.array(["%g" % d for d in day.values])
emb = B.obsm["X_pca"][:, :2] if not a.larry else B.obs[["SPRING-x", "SPRING-y"]].values.astype(float)
say("object %s, days %s, populations %s, early cells %d" % (B.shape, days, pops, len(d2_names)))

adata = cs.pp.initialize_adata_object(
    X_state=B.X, cell_names=np.asarray(B.obs_names), gene_names=np.asarray(B.var_names),
    time_info=tstr, X_clone=None, X_pca=B.obsm["X_pca"], X_emb=emb,
    state_info=B.obs["state_info"].astype(str).values, data_des="velfate_%s%s" % (os.path.basename(a.featdir.rstrip("/")), tag))
init_t = ["%g" % d for d in days[:-1]]
later_t = "%g" % days[-1]
cs.settings.data_path = os.path.join(a.featdir, "cospar_data")
cs.settings.figure_path = os.path.join(a.featdir, "cospar_figure")
os.makedirs(cs.settings.data_path, exist_ok=True)
os.makedirs(cs.settings.figure_path, exist_ok=True)
t0 = time.time()
adata = cs.tmap.infer_Tmap_from_state_info_alone(
    adata, initial_time_points=init_t, later_time_point=later_t, initialize_method=a.init, compute_new=True)
say("transition map %s in %.0f s; maps %s" % (adata.uns["transition_map"].shape, time.time() - t0,
                                                adata.uns.get("available_map")))

names = np.asarray(adata.obs_names)
pos = {n: i for i, n in enumerate(names)}
rows = np.array([pos[n] for n in d2_names])
t1 = set(np.asarray(adata.uns["Tmap_cell_id_t1"]).tolist())
assert all(r in t1 for r in rows), "an early cell is not an initial cell of the map"
present = [p for p in pops if (adata.obs["state_info"].values[np.asarray(adata.uns["Tmap_cell_id_t2"])] == p).any()]
say("populations among the later cells of the map: %s" % present)
out = {}
for method in ("norm-sum", "sum"):
    cs.tl.fate_map(adata, selected_fates=present, source="transition_map", map_backward=True, method=method, force_run=True)
    cols = []
    for p in pops:
        k = "fate_map_transition_map_%s" % p
        cols.append(np.nan_to_num(adata.obs[k].values[rows].astype(float)) if (p in present and k in adata.obs) else np.zeros(len(rows)))
    P = np.column_stack(cols)
    name = "snap_cospar_state%s%s" % ("" if method == "norm-sum" else "_sum", tag)
    np.save(os.path.join(a.featdir, name + ".npy"), P.astype(np.float32))
    json.dump(dict(method="cospar_state_only", cospar_version=cs.__version__, initialize_method=a.init,
                   fate_map_method=method, initial_time_points=init_t, later_time_point=later_t,
                   defaults="infer_Tmap_from_state_info_alone defaults", populations=pops,
                   populations_present=present, n_cells_in_object=int(B.n_obs),
                   rows_zero_mass=int((P.sum(1) == 0).sum())),
              open(os.path.join(a.featdir, name + "_log.json"), "w"), indent=1)
    say("saved %s %s; rows with zero mass %d" % (name, P.shape, int((P.sum(1) == 0).sum())))
    if a.larry:
        i, j = pops.index("Neutrophil"), pops.index("Monocyte")
        s = P[:, i] + P[:, j]
        share = np.where(s > 0, P[:, i] / np.where(s > 0, s, 1), 0.5)
        np.save(os.path.join(a.featdir, "nm_" + name + ".npy"), share)
        say("saved nm_%s" % name)
    out[method] = P
print("exit-ok")
