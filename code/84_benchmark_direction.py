"""Direction metrics with null fields on the standard RNA-velocity benchmarks.

On LARRY, forward cross-boundary direction correctness (CBDir) scores the
kNN-mean displacement above velocity, and in-cluster coherence (ICCoh) scores the
degradation-only null velocity highest (71_direction_embedding.py). Do these hold
on the data sets the velocity benchmarks use? For four of them, with the
transitions Luo et al. (2026) score:
  pancreas (endocrinogenesis day 15): Ngn3 high EP -> Pre-endocrine;
      Pre-endocrine -> Alpha, Beta, Delta, Epsilon
  dentate gyrus (10X43_1): OPC -> OL; nIPC -> Neuroblast; Neuroblast -> Granule
      immature; Granule immature -> Granule mature; Radial Glia-like -> Astrocytes
  gastrulation erythroid: Blood progenitors 1 -> 2 -> Erythroid1 -> 2 -> 3
  human bone marrow: HSC_1 -> Ery_1; HSC_1 -> HSC_2; Ery_1 -> Ery_2
Pipeline as scVelo's tutorials: gene filter (20 shared counts), per-cell
normalization, log, 2,000 highly variable genes, 30 principal components and 30
neighbours, first moments; steady-state and dynamical velocity, veloVI; the
permuted-unspliced and degradation-only null velocities; the kNN-mean
displacement (in the embedding, and in gene space projected); random vectors.
Every gene-space field is projected into the data set's own embedding (UMAP,
or t-SNE for bone marrow) with scVelo's velocity_graph and velocity_embedding,
as the benchmarks project velocity, and also onto the 30 principal components.
CBDir forward: source cells adjacent to target cells, cosine between the vector
and the displacements to those neighbours, averaged over transitions; reverse:
target cells adjacent to source cells. ICCoh within annotated clusters.
Intervals resample cells (no clones here), 2,000 times, separately for the
forward and the reverse sets.

Usage: 84_benchmark_direction.py <bench_dir> <out.csv> [--jobs 16]
"""
import argparse
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
if not hasattr(sp.csr_array, "getnnz"):
    sp.csr_array.getnnz = sp.csr_array._getnnz
scv.settings.verbosity = 0

SETS = {
    "pancreas": ("pancreas.h5ad", "clusters", "X_umap",
                 [("Ngn3 high EP", "Pre-endocrine"), ("Pre-endocrine", "Alpha"),
                  ("Pre-endocrine", "Beta"), ("Pre-endocrine", "Delta"),
                  ("Pre-endocrine", "Epsilon")]),
    "dentate gyrus": ("dentategyrus.h5ad", "clusters", "X_umap",
                      [("OPC", "OL"), ("nIPC", "Neuroblast"), ("Neuroblast", "Granule immature"),
                       ("Granule immature", "Granule mature"), ("Radial Glia-like", "Astrocytes")]),
    "gastrulation erythroid": ("gastrulation_erythroid.h5ad", "celltype", "X_umap",
                               [("Blood progenitors 1", "Blood progenitors 2"),
                                ("Blood progenitors 2", "Erythroid1"),
                                ("Erythroid1", "Erythroid2"), ("Erythroid2", "Erythroid3")]),
    "bone marrow": ("bonemarrow.h5ad", "clusters", "X_tsne",
                    [("HSC_1", "Ery_1"), ("HSC_1", "HSC_2"), ("Ery_1", "Ery_2")]),
}


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def cos_rows(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def boundary(V, P, nbr, lab, src, tgt):
    vals = []
    for i in np.where(lab == src)[0]:
        js = nbr[i][lab[nbr[i]] == tgt]
        if js.size:
            c = cos_rows(np.repeat(V[i][None], js.size, 0), P[js] - P[i])
            c = c[np.isfinite(c)]
            if c.size:
                vals.append(c.mean())
    return np.array(vals)


def iccoh_bench(V, nbr, lab):
    """ICCoh as the benchmarks compute it (VeloAE inner_cluster_coh): on the
    gene-space vectors, neighbour list including the cell itself, averaged
    within clusters and then over clusters."""
    per = {}
    for i in range(V.shape[0]):
        js = np.append(nbr[i], i)
        js = js[lab[js] == lab[i]]
        c = cos_rows(np.repeat(V[i][None], js.size, 0), V[js])
        c = c[np.isfinite(c)]
        if c.size:
            per.setdefault(lab[i], []).append(c.mean())
    return float(np.mean([np.mean(v) for v in per.values()]))


def iccoh(V, nbr, lab):
    vals = []
    for i in range(V.shape[0]):
        js = nbr[i][lab[nbr[i]] == lab[i]]
        if js.size:
            c = cos_rows(np.repeat(V[i][None], js.size, 0), V[js])
            c = c[np.isfinite(c)]
            if c.size:
                vals.append(c.mean())
    return float(np.mean(vals))


ap = argparse.ArgumentParser()
ap.add_argument("bench")
ap.add_argument("out")
ap.add_argument("--jobs", type=int, default=16)
a = ap.parse_args()
rng = np.random.default_rng(84)
rows = []
for name, (fn, key, embkey, trans) in SETS.items():
    B = ad.read_h5ad("%s/%s" % (a.bench, fn))
    B.var_names_make_unique()
    lab = B.obs[key].astype(str).values
    scv.pp.filter_genes(B, min_shared_counts=20)
    scv.pp.normalize_per_cell(B)
    sc.pp.log1p(B)
    sc.pp.highly_variable_genes(B, n_top_genes=2000, flavor="seurat")
    B = B[:, B.var["highly_variable"].values].copy()
    sc.pp.pca(B, n_comps=30)
    sc.pp.neighbors(B, n_pcs=30, n_neighbors=30, use_rep="X_pca", transformer="sklearn")
    scv.pp.moments(B)
    X = np.asarray(B.obsm["X_pca"][:, :30], dtype=float)
    pcs = B.varm["PCs"][:, :30]
    D = sp.csr_matrix(B.obsp["distances"])
    nbr = [D.indices[D.indptr[i]:D.indptr[i + 1]] for i in range(B.n_obs)]
    E = np.asarray(B.obsm[embkey], dtype=float)[:, :2]
    Ms = np.asarray(B.layers["Ms"], dtype=float)
    Mu = np.asarray(B.layers["Mu"], dtype=float)
    scv.tl.velocity(B, mode="deterministic", vkey="vel_det")
    gam = np.nan_to_num(np.asarray(B.var["vel_det_gamma"], dtype=float))
    vg = np.asarray(B.var["vel_det_genes"], dtype=bool)
    perm = rng.permutation(B.n_obs)
    gf = {"steady": np.nan_to_num(np.asarray(B.layers["vel_det"], dtype=float)),
          "perm_control": np.where(vg, Mu[perm] - gam * Ms, 0.0),
          "minus_gamma_s": np.where(vg, -gam * Ms, 0.0),
          "knn_mean": np.array([Ms[nbr[i]].mean(0) - Ms[i] for i in range(B.n_obs)])}
    scv.tl.recover_dynamics(B, n_jobs=a.jobs, var_names="all")
    scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
    gf["dynamical"] = np.nan_to_num(np.asarray(B.layers["vel_dyn"], dtype=float))
    try:
        from velovi import VELOVI, preprocess_data
        V = preprocess_data(B.copy(), spliced_layer="Ms", unspliced_layer="Mu")
        VELOVI.setup_anndata(V, spliced_layer="Ms", unspliced_layer="Mu")
        mdl = VELOVI(V)
        mdl.train(max_epochs=500, early_stopping=True, enable_progress_bar=False)
        vv = np.asarray(mdl.get_velocity(n_samples=25, velo_statistic="mean"))
        full = np.zeros((B.n_obs, B.n_vars))
        gi = [B.var_names.get_loc(g) for g in V.var_names]
        # veloVI's velocity is in min-max scaled units of Ms; back to Ms units
        full[:, gi] = vv * (Ms.max(0) - Ms.min(0))[gi]
        gf["velovi"] = full
    except Exception as e:
        say("%s: veloVI failed: %s" % (name, e))
    basis = embkey.replace("X_", "")
    emb, pca = {}, {}
    for f, F in gf.items():
        k = "f_" + f
        B.layers[k] = F
        B.var[k + "_genes"] = np.abs(F).sum(0) > 0
        scv.tl.velocity_graph(B, vkey=k, xkey="Ms", n_jobs=a.jobs, show_progress_bar=False)
        scv.tl.velocity_embedding(B, basis=basis, vkey=k)
        emb[f] = np.asarray(B.obsm["%s_%s" % (k, basis)], dtype=float)
        pca[f] = F @ pcs
    emb["knn_mean_in_embedding"] = np.array([E[nbr[i]].mean(0) - E[i] for i in range(B.n_obs)])
    pca["knn_mean_in_pca"] = np.array([X[nbr[i]].mean(0) - X[i] for i in range(B.n_obs)])
    emb["random"] = rng.standard_normal(E.shape)
    pca["random"] = rng.standard_normal(X.shape)
    for space, P, fields in (("embedding", E, emb), ("pca30", X, pca)):
        for f, Vf in fields.items():
            fw = np.concatenate([boundary(Vf, P, nbr, lab, s, t) for s, t in trans])
            rv = np.concatenate([boundary(Vf, P, nbr, lab, t, s) for s, t in trans])
            bf = np.array([fw[rng.integers(0, fw.size, fw.size)].mean() for _ in range(2000)])
            br = np.array([rv[rng.integers(0, rv.size, rv.size)].mean() for _ in range(2000)])
            d = bf - br
            # benchmark convention: mean within each transition, then over transitions;
            # intervals resample cells within each transition
            fwl = [boundary(Vf, P, nbr, lab, s, t) for s, t in trans]
            rvl = [boundary(Vf, P, nbr, lab, t, s) for s, t in trans]
            fwl = [x for x in fwl if x.size]; rvl = [x for x in rvl if x.size]
            ft = np.mean([x.mean() for x in fwl]); rt = np.mean([x.mean() for x in rvl])
            dt = np.array([np.mean([x[rng.integers(0, x.size, x.size)].mean() for x in fwl])
                           - np.mean([x[rng.integers(0, x.size, x.size)].mean() for x in rvl])
                           for _ in range(2000)])
            rows.append(dict(dataset=name, space=space, field=f, cbdir_fwd=fw.mean(), n_fwd=fw.size,
                             cbdir_rev=rv.mean(), n_rev=rv.size, fwd_minus_rev=fw.mean() - rv.mean(),
                             diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5),
                             iccoh=iccoh(Vf, nbr, lab),
                             cbdir_fwd_t=ft, cbdir_rev_t=rt, fwd_minus_rev_t=ft - rt,
                             diff_t_lo=np.percentile(dt, 2.5), diff_t_hi=np.percentile(dt, 97.5),
                             iccoh_gene=(iccoh_bench(gf[f], nbr, lab)
                                         if space == "embedding" and f in gf else np.nan)))
            say("%s %s %-22s fwd %.3f rev %.3f iccoh %.3f" % (name, space, f, fw.mean(), rv.mean(),
                                                              rows[-1]["iccoh"]))
    pd.DataFrame(rows).round(4).to_csv(a.out, index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
