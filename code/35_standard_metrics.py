"""The same comparisons again, in the protocol the field actually uses.

Our first pass computed cross-boundary direction correctness, in-cluster
coherence and the clonal-advance cosine in a 30-dimensional principal-component
space, with velocities projected linearly. That is not how these metrics are
normally computed, and it could depress a velocity field relative to a
displacement field. The standard route, and the one used here, is:

  * velocities projected into a two-dimensional embedding with scVelo's
    velocity_embedding, which uses the velocity graph's transition
    probabilities rather than a linear map;
  * the embedding is the one published with this dataset, adata.obsm["X_emb"],
    the authors' own SPRING coordinates, so no new embedding is invented;
  * cross-boundary direction correctness and in-cluster coherence computed as
    in a published evaluation's implementation (UniTVelo, unitvelo/eval_utils.py,
    functions cross_boundary_correctness and inner_cluster_coh), including its
    aggregation: the mean over cells within a cluster edge, then the mean over
    edges, and -- as in that implementation -- coherence on the gene-space
    velocity rather than the embedded one;
  * the clonal advance vector taken in the same embedding.

Two additions of our own, both declared: the reverse edges (each mature
population back to the undifferentiated compartment), which tell apart a field
that points at its neighbours from a field that points at the future; and an
expression-only control field, the mean displacement to a cell's own
neighbours, in the embedding for the boundary score and in gene space for the
coherence score, matching whichever space the metric uses.

Usage:
  python 35_standard_metrics.py <h5ad> <featdir> <labels.tsv> <out_prefix>
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import numpy.testing as npt
import anndata as ad
import scanpy as sc
import scvelo as scv

_orig = npt.assert_array_equal


def _compat(*a, **k):
    if "x" in k:
        k["actual"] = k.pop("x")
    if "y" in k:
        k["desired"] = k.pop("y")
    return _orig(*a, **k)


npt.assert_array_equal = _compat
for _cls in (sp.csr_array, sp.csc_array):
    if not hasattr(_cls, "getnnz"):
        _cls.getnnz = _cls._getnnz

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]
UNDIFF = "Undifferentiated"
N_PCS, N_NEIGHBORS = 30, 30
N_BOOT = 2000
SEED = 13


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def cos_to(points, vec):
    """Cosine between a single vector and each row of `points`."""
    num = points @ vec
    den = np.linalg.norm(points, axis=1) * np.linalg.norm(vec)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def cross_boundary_correctness(nbr, state, Xe, Ve, edges):
    """As in UniTVelo's eval_utils.cross_boundary_correctness: per cell, the
    mean cosine between its embedded velocity and the displacement to its
    neighbours in the target cluster; then the mean per edge; then over edges."""
    per_edge, per_cell_all = {}, []
    for u, v in edges:
        sel = np.where(state == u)[0]
        vals = []
        for i in sel:
            nodes = nbr[i]
            nodes = nodes[state[nodes] == v]
            if nodes.size == 0:
                continue
            c = cos_to(Xe[nodes] - Xe[i], Ve[i])
            c = c[np.isfinite(c)]
            if c.size:
                vals.append(float(c.mean()))
        per_edge[(u, v)] = float(np.mean(vals)) if vals else np.nan
        per_cell_all += vals
    finite = [s for s in per_edge.values() if np.isfinite(s)]
    return (float(np.mean(finite)) if finite else np.nan, per_edge,
            float(np.mean(per_cell_all)) if per_cell_all else np.nan,
            len(per_cell_all))


def inner_cluster_coh(nbr, state, V):
    """As in UniTVelo's eval_utils.inner_cluster_coh: on the gene-space
    velocity, the mean cosine with same-cluster neighbours, per cluster, then
    the mean over clusters."""
    out = {}
    for cat in np.unique(state):
        sel = np.where(state == cat)[0]
        vals = []
        for i in sel:
            nodes = nbr[i]
            nodes = nodes[state[nodes] == cat]
            if nodes.size == 0:
                continue
            c = cos_to(V[nodes], V[i])
            c = c[np.isfinite(c)]
            if c.size:
                vals.append(float(c.mean()))
        out[cat] = float(np.mean(vals)) if vals else np.nan
    finite = [s for s in out.values() if np.isfinite(s)]
    return (float(np.mean(finite)) if finite else np.nan), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("featdir")
    ap.add_argument("labels")
    ap.add_argument("out_prefix")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--state-col", default="state_info")
    ap.add_argument("--time-col", default="time_info")
    ap.add_argument("--early", default="2.0")
    ap.add_argument("--basis", default="emb")
    a = ap.parse_args()

    hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
    d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
    A = ad.read_h5ad(a.h5ad)
    clone_of = np.asarray(sp.csr_matrix(A.obsm["X_clone"]).argmax(1)).ravel()
    has_clone = np.asarray(sp.csr_matrix(A.obsm["X_clone"]).sum(1)).ravel() > 0
    is_early = (A.obs[a.time_col].astype(str) == a.early).values
    state_all = A.obs[a.state_col].astype(str).values

    B = A.copy()
    scv.pp.filter_genes(B, min_shared_counts=20)
    scv.pp.normalize_per_cell(B)
    sc.pp.log1p(B)
    keep = [g for g in hvg if g in B.var_names]
    assert len(keep) == len(hvg), (len(keep), len(hvg))
    B = B[:, keep].copy()
    B.obs["state"] = state_all
    scv.pp.neighbors(B, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS, method="sklearn")
    scv.pp.moments(B)
    say("object %s, moments done" % (B.shape,))

    D = sp.csr_matrix(B.obsp["distances"])
    nbr = [D.indices[D.indptr[i]:D.indptr[i + 1]] for i in range(B.n_obs)]
    state = B.obs["state"].values
    Xe = np.asarray(B.obsm["X_" + a.basis], dtype=np.float64)
    say("embedding X_%s %s, median neighbours %d"
        % (a.basis, Xe.shape, int(np.median([len(x) for x in nbr]))))

    # gene-space fields, then their embedded versions through velocity_embedding
    gene_fields, emb_fields = {}, {}

    scv.tl.velocity(B, mode="deterministic", vkey="vel_det")
    gene_fields["V1_scvelo_steady"] = np.nan_to_num(
        np.asarray(B.layers["vel_det"], dtype=np.float64))
    say("steady state done")

    scv.tl.recover_dynamics(B, n_jobs=a.jobs, var_names="all")
    scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
    gene_fields["V2_scvelo_dynamical"] = np.nan_to_num(
        np.asarray(B.layers["vel_dyn"], dtype=np.float64))
    say("dynamical done")

    try:
        from velovi import VELOVI, preprocess_data
        V = B.copy()
        V = preprocess_data(V, spliced_layer="Ms", unspliced_layer="Mu")
        VELOVI.setup_anndata(V, spliced_layer="Ms", unspliced_layer="Mu")
        m = VELOVI(V)
        m.train(max_epochs=500, early_stopping=True, enable_progress_bar=False)
        vv = np.asarray(m.get_velocity(n_samples=25, velo_statistic="mean"),
                        dtype=np.float64)
        full = np.zeros((B.n_obs, B.n_vars))
        full[:, [B.var_names.get_loc(g) for g in V.var_names]] = vv
        B.layers["vel_velovi"] = full
        gene_fields["V3_velovi"] = full
        say("veloVI done on %d genes" % V.n_vars)
    except Exception as e:
        say("veloVI FAILED: %s: %s" % (type(e).__name__, e))

    for tag, vkey in (("V1_scvelo_steady", "vel_det"),
                      ("V2_scvelo_dynamical", "vel_dyn"),
                      ("V3_velovi", "vel_velovi")):
        if tag not in gene_fields:
            continue
        if vkey not in B.layers:
            B.layers[vkey] = gene_fields[tag]
        scv.tl.velocity_graph(B, vkey=vkey, n_jobs=a.jobs)
        scv.tl.velocity_embedding(B, vkey=vkey, basis=a.basis)
        emb_fields[tag] = np.asarray(B.obsm["%s_%s" % (vkey, a.basis)],
                                     dtype=np.float64)
        say("embedded %s" % tag)

    # the expression-only control, in whichever space the metric uses
    Xg = np.asarray(B.layers["Ms"], dtype=np.float64)
    drift_g = np.zeros_like(Xg)
    drift_e = np.zeros_like(Xe)
    for i in range(B.n_obs):
        js = nbr[i]
        if js.size:
            drift_g[i] = Xg[js].mean(0) - Xg[i]
            drift_e[i] = Xe[js].mean(0) - Xe[i]
    gene_fields["E0_neighbour_drift"] = drift_g
    emb_fields["E0_neighbour_drift"] = drift_e
    say("expression-only drift built in both spaces")

    targets = [t for t in MATURE if (state == t).sum() >= 30]
    fwd = [(UNDIFF, t) for t in targets]
    rev = [(t, UNDIFF) for t in targets]

    rows = []
    for tag in emb_fields:
        rec = {"field": tag}
        for name, edges in (("forward", fwd), ("reverse", rev)):
            sc_edge, per_edge, pooled, n = cross_boundary_correctness(
                nbr, state, Xe, emb_fields[tag], edges)
            rec["cbdir_%s_edgemean" % name] = round(sc_edge, 4)
            rec["cbdir_%s_cellpooled" % name] = round(pooled, 4)
            rec["cbdir_%s_n_cells" % name] = n
        ic, _ = inner_cluster_coh(nbr, state, gene_fields[tag])
        rec["iccoh_genespace"] = round(ic, 4)
        rows.append(rec)
        say("%-22s cbdir fwd %.3f rev %.3f  iccoh %.3f"
            % (tag, rec["cbdir_forward_edgemean"],
               rec["cbdir_reverse_edgemean"], rec["iccoh_genespace"]))

    # the clonal advance cosine, in the same embedding
    d2 = np.array([B.obs_names.get_loc(n) for n in d2_names])
    late = (~is_early) & has_clone
    by_clone = {}
    for i in np.where(late)[0]:
        by_clone.setdefault(clone_of[i], []).append(i)
    adv = np.full((d2.size, Xe.shape[1]), np.nan)
    adv_a = np.full_like(adv, np.nan)
    adv_b = np.full_like(adv, np.nan)
    n_late = np.zeros(d2.size, dtype=int)
    for k, i in enumerate(d2):
        ls = by_clone.get(clone_of[i]) if has_clone[i] else None
        if not ls:
            continue
        ls = np.array(sorted(ls))
        n_late[k] = ls.size
        adv[k] = Xe[ls].mean(0) - Xe[i]
        if ls.size >= 2:
            adv_a[k] = Xe[ls[0::2]].mean(0) - Xe[i]
            adv_b[k] = Xe[ls[1::2]].mean(0) - Xe[i]

    T = pd.read_csv(a.labels, sep="\t")
    assert list(T["cell"]) == d2_names
    taskA = ((T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])).values

    def cos_pairs(Avec, Bvec):
        num = (Avec * Bvec).sum(1)
        den = np.linalg.norm(Avec, axis=1) * np.linalg.norm(Bvec, axis=1)
        return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)

    cos_rows = []
    rng = np.random.default_rng(SEED)
    for subset, mask in (("all early cells with an advance", n_late > 0),
                         ("task A clones", (n_late > 0) & taskA)):
        sel = np.where(mask)[0]
        clones = clone_of[d2[sel]]
        keys = np.unique(clones)
        cols, names = [], []
        for tag, Vf in emb_fields.items():
            c = cos_pairs(Vf[d2[sel]], adv[sel])
            cols.append(pd.Series(c, index=clones).groupby(level=0).mean()
                        .reindex(keys).to_numpy())
            names.append(tag)
        ok = np.isfinite(adv_a[sel]).all(1) & np.isfinite(adv_b[sel]).all(1)
        ceil = np.full(sel.size, np.nan)
        ceil[ok] = cos_pairs(adv_a[sel][ok], adv_b[sel][ok])
        cols.append(pd.Series(ceil, index=clones).groupby(level=0).mean()
                    .reindex(keys).to_numpy())
        names.append("CEILING_half_vs_half")
        tab = np.column_stack(cols)
        idx = rng.integers(0, tab.shape[0], size=(N_BOOT, tab.shape[0]))
        bm = np.nanmean(tab[idx], axis=1)
        best = max([n for n in names if n.startswith("V")],
                   key=lambda n: np.nanmean(tab[:, names.index(n)]))
        jb = names.index(best)
        for j, nm in enumerate(names):
            lo, hi = np.nanpercentile(bm[:, j], [2.5, 97.5])
            cos_rows.append(dict(subset=subset, field=nm, clones=len(keys),
                                 cosine=round(float(np.nanmean(tab[:, j])), 4),
                                 lo=round(float(lo), 4), hi=round(float(hi), 4)))
        for j, nm in enumerate(names):
            if nm == best:
                continue
            d = bm[:, j] - bm[:, jb]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            cos_rows.append(dict(subset=subset,
                                 field="%s minus %s" % (nm, best),
                                 clones=len(keys),
                                 cosine=round(float(np.nanmean(tab[:, j])
                                                    - np.nanmean(tab[:, jb])), 4),
                                 lo=round(float(lo), 4), hi=round(float(hi), 4)))

    pd.DataFrame(rows).to_csv(a.out_prefix + "_direction.csv", index=False)
    pd.DataFrame(cos_rows).to_csv(a.out_prefix + "_cosine.csv", index=False)
    say("\n" + pd.DataFrame(cos_rows).to_string(index=False))
    say("wrote %s_{direction,cosine}.csv" % a.out_prefix)


if __name__ == "__main__":
    main()
