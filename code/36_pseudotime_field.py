"""A field that carries maturation direction and no splicing, for comparison.

The manuscript's mechanism statement is that velocity encodes the maturation
direction and not which fate. The expression drift field of
35_standard_metrics.py does not test that, because it carries no direction at
all, so losing to a velocity field in the standard protocol tells us nothing.
The direct test is a field that carries direction but no splicing: CellRank's
pseudotime kernel, whose transition matrix comes from a diffusion pseudotime
computed on expression alone, projected into the same embedding by the same
route as the velocity field.

Two controls, not one. The pseudotime kernel is directed; the connectivity
kernel is not, and is projected the same way, which says how much direction the
projection itself manufactures.

  * if the pseudotime field matches or beats the dynamical velocity field, the
    mechanism statement is directly supported: the direction information the
    splicing layer supplies is no more than a maturation gradient;
  * if it is clearly lower, velocity's direction carries something beyond
    maturation, and the mechanism statement has to narrow.

The pseudotime, its rooting, the terminal states and the kernels are built
exactly as in 02c_cellrank.py. The metrics are the ones in
35_standard_metrics.py, which follow UniTVelo's eval_utils.py.

Usage:
  python 36_pseudotime_field.py <h5ad> <featdir> <labels.tsv> <out_prefix>
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
import cellrank as cr  # noqa: E402

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]
UNDIFF = "Undifferentiated"
N_PCS, N_NEIGHBORS = 30, 30
N_BOOT = 2000
SEED = 13


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def cos_to(points, vec):
    num = points @ vec
    den = np.linalg.norm(points, axis=1) * np.linalg.norm(vec)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def cross_boundary_correctness(nbr, state, Xe, Ve, edges):
    per_edge, pooled = {}, []
    for u, v in edges:
        vals = []
        for i in np.where(state == u)[0]:
            nodes = nbr[i]
            nodes = nodes[state[nodes] == v]
            if nodes.size == 0:
                continue
            c = cos_to(Xe[nodes] - Xe[i], Ve[i])
            c = c[np.isfinite(c)]
            if c.size:
                vals.append(float(c.mean()))
        per_edge[(u, v)] = float(np.mean(vals)) if vals else np.nan
        pooled += vals
    finite = [s for s in per_edge.values() if np.isfinite(s)]
    return (float(np.mean(finite)) if finite else np.nan,
            float(np.mean(pooled)) if pooled else np.nan, len(pooled))


def project_transition(T, Xe, n_neighbors_mask=None):
    """scVelo's embedding projection, applied to any transition matrix.

    v_emb[i] = sum_j (pi_ij - 1/n_i) * (x_j - x_i) / ||x_j - x_i||, over the
    neighbours j that the matrix gives cell i. This is the formula
    scv.tl.velocity_embedding uses on a velocity graph's transition matrix.
    """
    T = sp.csr_matrix(T)
    out = np.zeros_like(Xe)
    for i in range(T.shape[0]):
        s, e = T.indptr[i], T.indptr[i + 1]
        js = T.indices[s:e]
        if js.size == 0:
            continue
        pi = T.data[s:e]
        d = Xe[js] - Xe[i]
        nrm = np.linalg.norm(d, axis=1)
        keep = nrm > 0
        if not keep.any():
            continue
        u = d[keep] / nrm[keep][:, None]
        w = pi[keep] - 1.0 / keep.sum()
        out[i] = (w[:, None] * u).sum(0)
    return out


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
    tvals = pd.to_numeric(A.obs[a.time_col], errors="coerce").values

    B = A.copy()
    scv.pp.filter_genes(B, min_shared_counts=20)
    scv.pp.normalize_per_cell(B)
    sc.pp.log1p(B)
    keep = [g for g in hvg if g in B.var_names]
    assert len(keep) == len(hvg)
    B = B[:, keep].copy()
    B.obs["state"] = state_all
    scv.pp.neighbors(B, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS, method="sklearn")
    scv.pp.moments(B)
    Xe = np.asarray(B.obsm["X_" + a.basis], dtype=np.float64)
    D = sp.csr_matrix(B.obsp["distances"])
    nbr = [D.indices[D.indptr[i]:D.indptr[i + 1]] for i in range(B.n_obs)]
    state = B.obs["state"].values
    say("object %s, embedding %s" % (B.shape, Xe.shape))

    emb_fields = {}

    # the reference: the dynamical velocity field, embedded the standard way
    scv.tl.recover_dynamics(B, n_jobs=a.jobs, var_names="all")
    scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
    scv.tl.velocity_graph(B, vkey="vel_dyn", n_jobs=a.jobs)
    scv.tl.velocity_embedding(B, vkey="vel_dyn", basis=a.basis)
    emb_fields["V2_scvelo_dynamical"] = np.asarray(
        B.obsm["vel_dyn_" + a.basis], dtype=np.float64)
    say("dynamical velocity field embedded")

    # the pseudotime kernel, exactly as 02c_cellrank.py builds it
    present = [s for s in MATURE if (state == s).sum() >= 30]
    is_term = np.zeros(B.n_obs, bool)
    for s in present:
        is_term |= (state == s)
    sc.tl.diffmap(B)
    dm = B.obsm["X_diffmap"][:, 1:11]
    cand = (state == UNDIFF)
    if cand.sum() == 0:
        cand = ~is_term
    if len(np.unique(tvals)) > 1:
        earliest = cand & (tvals == np.nanmin(tvals))
        if earliest.sum() > 0:
            cand = earliest
    centre = dm[is_term].mean(0)
    dist = np.linalg.norm(dm - centre, axis=1)
    B.uns["iroot"] = int(np.where(cand)[0][np.argmax(dist[cand])])
    sc.tl.dpt(B)
    pt = B.obs["dpt_pseudotime"].values
    if np.nanmean(pt[is_term]) < np.nanmean(pt[~is_term]):
        B.obs["dpt_pseudotime"] = np.nanmax(pt) - pt
    say("pseudotime: mature minus other = %.3f"
        % float(np.nanmean(B.obs["dpt_pseudotime"].values[is_term])
                - np.nanmean(B.obs["dpt_pseudotime"].values[~is_term])))

    pk = cr.kernels.PseudotimeKernel(B, time_key="dpt_pseudotime")
    pk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(B)
    ck.compute_transition_matrix()
    for tag, kern in (("P0_pseudotime_kernel", pk),
                      ("C0_connectivity_kernel", ck)):
        done = False
        try:
            kern.compute_projection(basis=a.basis, key_added=tag, copy=False)
            if tag in B.obsm:
                emb_fields[tag] = np.asarray(B.obsm[tag], dtype=np.float64)
                done = True
                say("%s projected by cellrank's compute_projection" % tag)
        except Exception as e:
            say("%s compute_projection unavailable (%s: %s), using the formula"
                % (tag, type(e).__name__, e))
        if not done:
            emb_fields[tag] = project_transition(kern.transition_matrix, Xe)
            say("%s projected with the velocity_embedding formula" % tag)

    # the undirected expression drift, for continuity with the previous run
    drift_e = np.zeros_like(Xe)
    for i in range(B.n_obs):
        js = nbr[i]
        if js.size:
            drift_e[i] = Xe[js].mean(0) - Xe[i]
    emb_fields["E0_neighbour_drift"] = drift_e

    targets = [t for t in MATURE if (state == t).sum() >= 30]
    fwd = [(UNDIFF, t) for t in targets]
    rev = [(t, UNDIFF) for t in targets]
    rows = []
    for tag, Ve in emb_fields.items():
        rec = {"field": tag}
        for nm, edges in (("forward", fwd), ("reverse", rev)):
            em, pooled, n = cross_boundary_correctness(nbr, state, Xe, Ve, edges)
            rec["cbdir_%s_edgemean" % nm] = round(em, 4)
            rec["cbdir_%s_cellpooled" % nm] = round(pooled, 4)
            rec["cbdir_%s_n" % nm] = n
        rec["cbdir_forward_minus_reverse"] = round(
            rec["cbdir_forward_edgemean"] - rec["cbdir_reverse_edgemean"], 4)
        rows.append(rec)
        say("%-24s fwd %.3f rev %.3f  fwd-rev %.3f"
            % (tag, rec["cbdir_forward_edgemean"],
               rec["cbdir_reverse_edgemean"],
               rec["cbdir_forward_minus_reverse"]))
    pd.DataFrame(rows).to_csv(a.out_prefix + "_direction.csv", index=False)

    # the clonal advance cosine in the same embedding
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

    def cos_pairs(P, Q):
        num = (P * Q).sum(1)
        den = np.linalg.norm(P, axis=1) * np.linalg.norm(Q, axis=1)
        return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)

    rng = np.random.default_rng(SEED)
    cos_rows = []
    ref = "V2_scvelo_dynamical"
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
        jr = names.index(ref)
        for j, nm in enumerate(names):
            lo, hi = np.nanpercentile(bm[:, j], [2.5, 97.5])
            cos_rows.append(dict(subset=subset, field=nm, clones=len(keys),
                                 cosine=round(float(np.nanmean(tab[:, j])), 4),
                                 lo=round(float(lo), 4), hi=round(float(hi), 4)))
        for j, nm in enumerate(names):
            if nm == ref:
                continue
            d = bm[:, j] - bm[:, jr]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            cos_rows.append(dict(subset=subset, field="%s minus %s" % (nm, ref),
                                 clones=len(keys),
                                 cosine=round(float(np.nanmean(tab[:, j])
                                                    - np.nanmean(tab[:, jr])), 4),
                                 lo=round(float(lo), 4), hi=round(float(hi), 4)))
    C = pd.DataFrame(cos_rows)
    C.to_csv(a.out_prefix + "_cosine.csv", index=False)
    say("\n" + C.to_string(index=False))
    say("wrote %s_{direction,cosine}.csv" % a.out_prefix)


if __name__ == "__main__":
    main()
