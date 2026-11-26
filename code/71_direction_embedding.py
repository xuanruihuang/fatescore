"""Direction metrics as the velocity benchmarks compute them: in a UMAP embedding.

31_direction_metrics.py computes cross-boundary direction correctness (CBDir)
and in-cluster coherence (ICCoh) in 30 principal components. The benchmarks
that use these metrics compute them on velocities projected into a
low-dimensional embedding (Wu et al. 2026: UMAP-projected velocity vectors;
Luo et al. 2026: positions and velocities in a low-dimensional space), and the
standard projection is scVelo's velocity_embedding, which turns each cell's
velocity into a weighted average of displacements to its neighbours. This
script recomputes every field of 31 on the same pipeline, projects each one
into UMAP with scVelo's own velocity_graph / velocity_embedding, and scores
CBDir forward, CBDir reverse and ICCoh there, and again in the 30 principal
components for reference.

Fields: steady-state, dynamical and veloVI velocity; the permutation control
(each gene's gamma kept, the cell's smoothed unspliced counts replaced by
another cell's); the -gamma*s control; the kNN-mean displacement in gene space
(mean smoothed spliced counts of a cell's neighbours minus its own), projected
like a velocity; the kNN-mean displacement taken directly in UMAP; random
vectors.

Intervals: clones are resampled (2,000 times; cells without a barcode count as
their own clone) separately within the forward and the reverse boundary sets, so
forward minus reverse has an interval.

Also scored: a field directed by the annotation alone (from each cell towards the
centroid of all annotated mature cells, in gene space and projected, and directly
in UMAP and in principal components), and the kNN-mean displacement taken directly
in principal components.

Usage: 71_direction_embedding.py <h5ad> <featdir> <out_prefix> [--jobs 16]
"""
import argparse
import json
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

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]
UNDIFF = "Undifferentiated"


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def cos_rows(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def boundary_scores(V, X, nbr, state, src_mask, tgt_masks):
    """Per-cell CBDir contributions: for a cell in src with neighbours in a
    target population, the mean cosine between its vector and the
    displacements to those neighbours; one value per (cell, target)."""
    vals, cells = [], []
    for tgt in tgt_masks:
        for i in np.where(src_mask)[0]:
            js = nbr[i]
            js = js[tgt[js]]
            if js.size == 0:
                continue
            c = cos_rows(np.repeat(V[i][None, :], js.size, axis=0), X[js] - X[i])
            c = c[np.isfinite(c)]
            if c.size:
                vals.append(c.mean())
                cells.append(i)
    return np.array(vals), np.array(cells, dtype=int)


def boundary_scores_t(V, X, nbr, src_mask, tgt_masks):
    """As boundary_scores, but also returns the index of the transition of each
    value, so that scores can be averaged within transitions first, as the
    benchmarks do (VeloAE's cross_boundary_correctness)."""
    vals, cells, tix = [], [], []
    for t, tgt in enumerate(tgt_masks):
        v, c = boundary_scores(V, X, nbr, None, src_mask, [tgt])
        vals.append(v); cells.append(c); tix.append(np.full(v.size, t))
    return np.concatenate(vals), np.concatenate(cells), np.concatenate(tix)


def pertrans_mean(vals, tix):
    return float(np.mean([vals[tix == t].mean() for t in np.unique(tix)]))


def iccoh_bench(V, nbr, state):
    """In-cluster coherence as the benchmarks compute it (VeloAE's
    inner_cluster_coh): on the vectors given (gene space), each cell's mean
    cosine with its neighbours in the same cluster, the neighbour list
    including the cell itself; averaged within clusters, then over clusters."""
    per = {}
    for i in range(V.shape[0]):
        js = np.append(nbr[i], i)
        js = js[state[js] == state[i]]
        c = cos_rows(np.repeat(V[i][None, :], js.size, axis=0), V[js])
        c = c[np.isfinite(c)]
        if c.size:
            per.setdefault(state[i], []).append(c.mean())
    return float(np.mean([np.mean(v) for v in per.values()]))


def iccoh(V, nbr, state):
    vals = []
    for i in range(V.shape[0]):
        js = nbr[i]
        js = js[state[js] == state[i]]
        if js.size == 0:
            continue
        c = cos_rows(np.repeat(V[i][None, :], js.size, axis=0), V[js])
        c = c[np.isfinite(c)]
        if c.size:
            vals.append(c.mean())
    return float(np.mean(vals))


ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("out")
ap.add_argument("--jobs", type=int, default=16)
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

hvg = json.load(open(a.featdir + "/hvg.json"))
B = ad.read_h5ad(a.h5ad)
state = B.obs["state_info"].astype(str).values
scv.pp.filter_genes(B, min_shared_counts=20)
scv.pp.normalize_per_cell(B)
sc.pp.log1p(B)
B = B[:, hvg].copy()
B.obs["state"] = state
scv.pp.neighbors(B, n_pcs=30, n_neighbors=30, method="sklearn")
scv.pp.moments(B)
X = np.asarray(B.obsm["X_pca"][:, :30], dtype=np.float64)
pcs = B.varm["PCs"][:, :30]
D = sp.csr_matrix(B.obsp["distances"])
nbr = [D.indices[D.indptr[i]:D.indptr[i + 1]] for i in range(B.n_obs)]
say("moments done; object %s" % (B.shape,))

Ms = np.asarray(B.layers["Ms"], dtype=np.float64)
Mu = np.asarray(B.layers["Mu"], dtype=np.float64)
scv.tl.velocity(B, mode="deterministic", vkey="vel_det")
gam = np.nan_to_num(np.asarray(B.var["vel_det_gamma"], dtype=np.float64))
vg = np.asarray(B.var["vel_det_genes"], dtype=bool)
perm = np.random.default_rng(31).permutation(B.n_obs)
gene_fields = {
    "steady": np.nan_to_num(np.asarray(B.layers["vel_det"], dtype=np.float64)),
    "perm_control": np.where(vg, Mu[perm] - gam * Ms, 0.0),
    "minus_gamma_s": np.where(vg, -gam * Ms, 0.0),
}
knn = np.zeros_like(Ms)
for i in range(B.n_obs):
    knn[i] = Ms[nbr[i]].mean(0) - Ms[i]
gene_fields["knn_mean"] = knn
# a field directed by the annotation alone: towards the centroid of the mature cells
mature_mask = np.isin(state, MATURE)
gene_fields["centroid_mature"] = Ms[mature_mask].mean(0)[None, :] - Ms
scv.tl.recover_dynamics(B, n_jobs=a.jobs, var_names="all")
scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
gene_fields["dynamical"] = np.nan_to_num(np.asarray(B.layers["vel_dyn"], dtype=np.float64))
say("dynamical done")
try:
    from velovi import VELOVI, preprocess_data
    V = B.copy()
    V = preprocess_data(V, spliced_layer="Ms", unspliced_layer="Mu")
    VELOVI.setup_anndata(V, spliced_layer="Ms", unspliced_layer="Mu")
    mdl = VELOVI(V)
    mdl.train(max_epochs=500, early_stopping=True, enable_progress_bar=False)
    vv = np.asarray(mdl.get_velocity(n_samples=25, velo_statistic="mean"))
    full = np.zeros((B.n_obs, B.n_vars))
    gi = [B.var_names.get_loc(g) for g in V.var_names]
    # veloVI works on min-max scaled Ms (preprocess_data); its velocity is in
    # scaled units per gene. Back to the units of Ms, against which every field
    # is projected (velocity_graph with xkey="Ms") and on whose PCs it is scored.
    full[:, gi] = vv * (Ms.max(0) - Ms.min(0))[gi]
    gene_fields["velovi"] = full
    say("veloVI done")
except Exception as e:
    say("veloVI FAILED: %s" % e)

sc.tl.umap(B, random_state=0)
U = np.asarray(B.obsm["X_umap"], dtype=np.float64)
emb = {}
for name, F in gene_fields.items():
    key = "f_" + name
    B.layers[key] = F
    B.var[key + "_genes"] = np.abs(F).sum(0) > 0
    scv.tl.velocity_graph(B, vkey=key, xkey="Ms", n_jobs=a.jobs, show_progress_bar=False)
    scv.tl.velocity_embedding(B, basis="umap", vkey=key)
    emb[name] = np.asarray(B.obsm[key + "_umap"], dtype=np.float64)
    say("projected %s" % name)
du = np.zeros_like(U)
for i in range(B.n_obs):
    du[i] = U[nbr[i]].mean(0) - U[i]
emb["knn_mean_in_umap"] = du
emb["centroid_in_umap"] = U[np.isin(state, MATURE)].mean(0)[None, :] - U
rng = np.random.default_rng(71)
emb["random"] = rng.standard_normal(U.shape)
pca_fields = {k: F @ pcs for k, F in gene_fields.items()}
pca_fields["random"] = rng.standard_normal(X.shape)
dp = np.zeros_like(X)
for i in range(B.n_obs):
    dp[i] = X[nbr[i]].mean(0) - X[i]
pca_fields["knn_mean_in_pca"] = dp
pca_fields["centroid_in_pca"] = X[np.isin(state, MATURE)].mean(0)[None, :] - X
# clone of each cell, for intervals that resample clones (cells without a barcode
# form their own block)
Cm = B.obsm["X_clone"] if "X_clone" in B.obsm else None
if Cm is not None:
    Cm = sp.csr_matrix(Cm)
    has = np.asarray(Cm.sum(1)).ravel() > 0
    clone = np.where(has, np.asarray(Cm.argmax(1)).ravel(), -1 - np.arange(B.n_obs))
else:
    clone = -1 - np.arange(B.n_obs)


def block_boot_t(vals, cells, tix, n, rng):
    """Clone-resampled per-transition means, averaged over transitions."""
    g = clone[cells]
    ug, inv = np.unique(g, return_inverse=True)
    T = np.unique(tix)
    out = np.empty(n)
    for b in range(n):
        w = np.bincount(rng.integers(0, ug.size, ug.size), minlength=ug.size)[inv]
        m = [(vals[tix == t] * w[tix == t]).sum() / w[tix == t].sum()
             for t in T if w[tix == t].sum() > 0]
        out[b] = np.mean(m)
    return out


def block_boot(vals, cells, n, rng):
    """Means over clone-resampled data: clones drawn with replacement, every
    value of a drawn clone counted as often as the clone is drawn."""
    g = clone[cells]
    ug, inv = np.unique(g, return_inverse=True)
    out = np.empty(n)
    for b in range(n):
        cnt = np.bincount(rng.integers(0, ug.size, ug.size), minlength=ug.size)
        w = cnt[inv]
        out[b] = (vals * w).sum() / max(w.sum(), 1)
    return out

targets = [t for t in MATURE if (state == t).sum() >= 30]
tmask = [state == t for t in targets]
und = state == UNDIFF
# the annotation-only field into the nearest mature population (as in
# 82_annotation_fields.py), computed here so that it gets every score
for nm, P, fields in (("umap", U, emb), ("pca30", X, pca_fields)):
    cents = np.stack([P[m].mean(0) for m in tmask])
    dist = np.linalg.norm(P[:, None, :] - cents[None], axis=2)
    fields["nearest_mature_centroid"] = cents[dist.argmin(1)] - P
rows = []
for space, P, fields in (("umap", U, emb), ("pca30", X, pca_fields)):
    for name, Vf in fields.items():
        fw, fc = boundary_scores(Vf, P, nbr, state, und, tmask)
        parts = [boundary_scores(Vf, P, nbr, state, tm, [und]) for tm in tmask]
        rv = np.concatenate([q[0] for q in parts])
        rc = np.concatenate([q[1] for q in parts])
        bf = block_boot(fw, fc, a.boot, rng)
        br = block_boot(rv, rc, a.boot, rng)
        d = bf - br
        # benchmark convention: mean within each transition, then over transitions
        fwt, fct, ft = boundary_scores_t(Vf, P, nbr, und, tmask)
        rvt = [boundary_scores_t(Vf, P, nbr, tm, [und]) for tm in tmask]
        rvv = np.concatenate([q[0] for q in rvt]); rvc = np.concatenate([q[1] for q in rvt])
        rvi = np.concatenate([np.full(q[0].size, k) for k, q in enumerate(rvt)])
        bft = block_boot_t(fwt, fct, ft, a.boot, rng)
        brt = block_boot_t(rvv, rvc, rvi, a.boot, rng)
        dt = bft - brt
        rec = dict(space=space, field=name, cbdir_fwd=fw.mean(), n_fwd=fw.size,
                   cbdir_rev=rv.mean(), n_rev=rv.size,
                   fwd_lo=np.percentile(bf, 2.5), fwd_hi=np.percentile(bf, 97.5),
                   fwd_minus_rev=fw.mean() - rv.mean(),
                   diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5),
                   iccoh=iccoh(Vf, nbr, state),
                   cbdir_fwd_t=pertrans_mean(fwt, ft), cbdir_rev_t=pertrans_mean(rvv, rvi),
                   fwd_minus_rev_t=pertrans_mean(fwt, ft) - pertrans_mean(rvv, rvi),
                   fwd_t_lo=np.percentile(bft, 2.5), fwd_t_hi=np.percentile(bft, 97.5),
                   diff_t_lo=np.percentile(dt, 2.5), diff_t_hi=np.percentile(dt, 97.5),
                   iccoh_gene=(iccoh_bench(gene_fields[name], nbr, state)
                               if space == "umap" and name in gene_fields else np.nan))
        rows.append(rec)
        say("%s %-18s fwd %.3f rev %.3f diff %.3f [%.3f, %.3f] iccoh %.3f"
            % (space, name, rec["cbdir_fwd"], rec["cbdir_rev"], rec["fwd_minus_rev"],
               rec["diff_lo"], rec["diff_hi"], rec["iccoh"]))
R = pd.DataFrame(rows).round(4)
R.to_csv(a.out + ".csv", index=False)
print(R.to_string(index=False))
