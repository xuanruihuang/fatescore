"""A field whose direction the experiment supplies, not the splicing layer.

The manuscript's mechanism statement is that velocity encodes the maturation
direction and not which fate. Testing it needs a field that carries maturation
direction with no splicing in it and no fate annotation: this one points each
cell at the mean position of its nearest cells from the NEXT sampling day. The
direction comes from the experiment's own time course.

Scored against the same measured target as the velocity field -- the direction a
clone actually advances in -- in the same embedding, with the same metrics, plus
the forward and reverse boundary scores. The clonal advance mixes maturation
with branching, so each field's agreement with it is also decomposed onto three
axes fitted in the embedding the way the manuscript fits them in feature space:
undifferentiated against mature among day-2 cells, day-2 against the later days,
and neutrophil-fated against monocyte-fated clones.

The decision rules and the k values were registered before this was run; see
notes/velocity_fate_extension_log.md section 12.4.

Usage:
  python 39_sampling_day_field.py <h5ad> <featdir> <labels.tsv> <out_prefix>
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

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
K_GRID = (15, 30, 100)


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def cos_to(points, vec):
    num = points @ vec
    den = np.linalg.norm(points, axis=1) * np.linalg.norm(vec)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def cos_pairs(P, Q):
    num = (P * Q).sum(1)
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(Q, axis=1)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def cbdir(nbr, state, Xe, Ve, edges):
    per_edge, pooled, used = {}, [], 0
    for u, v in edges:
        vals = []
        for i in np.where(state == u)[0]:
            if not np.isfinite(Ve[i]).all():
                continue
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


def axis_from(Xe, y):
    """A unit direction in the embedding, fitted the way the manuscript fits
    its axes in feature space: a logistic separation of two known groups."""
    m = LogisticRegression(max_iter=4000).fit(Xe, y)
    w = m.coef_.ravel()
    return w / np.linalg.norm(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("featdir")
    ap.add_argument("labels")
    ap.add_argument("out_prefix")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--state-col", default="state_info")
    ap.add_argument("--time-col", default="time_info")
    ap.add_argument("--basis", default="emb")
    a = ap.parse_args()

    hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
    d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
    A = ad.read_h5ad(a.h5ad)
    clone_of = np.asarray(sp.csr_matrix(A.obsm["X_clone"]).argmax(1)).ravel()
    has_clone = np.asarray(sp.csr_matrix(A.obsm["X_clone"]).sum(1)).ravel() > 0
    tnum = pd.to_numeric(A.obs[a.time_col], errors="coerce").values
    state_all = A.obs[a.state_col].astype(str).values

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
    X = np.asarray(B.obsm["X_pca"][:, :N_PCS], dtype=np.float64)
    Xe = np.asarray(B.obsm["X_" + a.basis], dtype=np.float64)
    D = sp.csr_matrix(B.obsp["distances"])
    nbr = [D.indices[D.indptr[i]:D.indptr[i + 1]] for i in range(B.n_obs)]
    state = B.obs["state"].values
    days = sorted(set(np.unique(tnum[~np.isnan(tnum)]).tolist()))
    say("object %s, sampling days %s" % (B.shape, days))

    fields = {}
    scv.tl.recover_dynamics(B, n_jobs=a.jobs, var_names="all")
    scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
    scv.tl.velocity_graph(B, vkey="vel_dyn", n_jobs=a.jobs)
    scv.tl.velocity_embedding(B, vkey="vel_dyn", basis=a.basis)
    fields["V2_scvelo_dynamical"] = np.asarray(B.obsm["vel_dyn_" + a.basis],
                                               dtype=np.float64)
    say("reference velocity field embedded")

    # the sampling-day fields
    for k in K_GRID:
        F = np.full_like(Xe, np.nan)
        for t, t_next in zip(days[:-1], days[1:]):
            src = np.where(tnum == t)[0]
            tgt = np.where(tnum == t_next)[0]
            if src.size == 0 or tgt.size == 0:
                continue
            kk = min(k, tgt.size)
            nn = NearestNeighbors(n_neighbors=kk, algorithm="brute").fit(X[tgt])
            _, ind = nn.kneighbors(X[src])
            F[src] = Xe[tgt][ind].mean(1) - Xe[src]
            say("  day %s -> %s: %d cells, k=%d" % (t, t_next, src.size, kk))
        fields["S%d_next_day_k%d" % (k, k)] = F
    say("sampling-day fields built for k in %s" % (K_GRID,))

    targets = [t for t in MATURE if (state == t).sum() >= 30]
    fwd = [(UNDIFF, t) for t in targets]
    rev = [(t, UNDIFF) for t in targets]
    rows = []
    for tag, Ve in fields.items():
        rec = {"field": tag,
               "cells_with_an_arrow": int(np.isfinite(Ve).all(1).sum())}
        for nm, edges in (("forward", fwd), ("reverse", rev)):
            em, pooled, n = cbdir(nbr, state, Xe, Ve, edges)
            rec["cbdir_%s_edgemean" % nm] = round(em, 4)
            rec["cbdir_%s_n" % nm] = n
        rec["cbdir_forward_minus_reverse"] = round(
            rec["cbdir_forward_edgemean"] - rec["cbdir_reverse_edgemean"], 4)
        rows.append(rec)
        say("%-22s arrows %6d  fwd %.3f rev %.3f  fwd-rev %.3f"
            % (tag, rec["cells_with_an_arrow"], rec["cbdir_forward_edgemean"],
               rec["cbdir_reverse_edgemean"], rec["cbdir_forward_minus_reverse"]))
    pd.DataFrame(rows).to_csv(a.out_prefix + "_direction.csv", index=False)

    # the clonal advance, and its decomposition onto three fitted axes
    d2 = np.array([B.obs_names.get_loc(n) for n in d2_names])
    is_early = tnum == float(2)
    late = (~is_early) & has_clone
    by_clone = {}
    for i in np.where(late)[0]:
        by_clone.setdefault(clone_of[i], []).append(i)
    adv = np.full((d2.size, Xe.shape[1]), np.nan)
    adv_a = np.full_like(adv, np.nan)
    adv_b = np.full_like(adv, np.nan)
    n_late = np.zeros(d2.size, dtype=int)
    for q, i in enumerate(d2):
        ls = by_clone.get(clone_of[i]) if has_clone[i] else None
        if not ls:
            continue
        ls = np.array(sorted(ls))
        n_late[q] = ls.size
        adv[q] = Xe[ls].mean(0) - Xe[i]
        if ls.size >= 2:
            adv_a[q] = Xe[ls[0::2]].mean(0) - Xe[i]
            adv_b[q] = Xe[ls[1::2]].mean(0) - Xe[i]

    T = pd.read_csv(a.labels, sep="\t")
    assert list(T["cell"]) == d2_names
    taskA = ((T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])).values

    mature_mask = np.isin(state, targets)
    ax = {}
    sel = is_early & (mature_mask | (state == UNDIFF))
    ax["maturation"] = axis_from(Xe[sel], mature_mask[sel].astype(int))
    ax["sampling_day"] = axis_from(Xe, (~is_early).astype(int))
    fa = taskA & (n_late > 0)
    ax["fate"] = axis_from(Xe[d2[fa]],
                           (T.loc[fa, "dom"].values == "Neutrophil").astype(int))
    for nm, v in ax.items():
        say("axis %-13s = (%.3f, %.3f)" % (nm, v[0], v[1]))

    rng = np.random.default_rng(SEED)
    cos_rows, dec_rows = [], []
    ref = "V2_scvelo_dynamical"
    for subset, mask in (("all early cells with an advance", n_late > 0),
                         ("task A clones", (n_late > 0) & taskA)):
        sel = np.where(mask)[0]
        clones = clone_of[d2[sel]]
        keys = np.unique(clones)

        def by_clone_mean(v):
            return (pd.Series(v, index=clones).groupby(level=0).mean()
                    .reindex(keys).to_numpy())

        cols, names = [], []
        for tag, Vf in fields.items():
            cols.append(by_clone_mean(cos_pairs(Vf[d2[sel]], adv[sel])))
            names.append(tag)
        ok = np.isfinite(adv_a[sel]).all(1) & np.isfinite(adv_b[sel]).all(1)
        ceil = np.full(sel.size, np.nan)
        ceil[ok] = cos_pairs(adv_a[sel][ok], adv_b[sel][ok])
        cols.append(by_clone_mean(ceil))
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

        # decomposition: along each axis, do the field's and the advance's
        # components agree across clones?
        for axis_name, u in ax.items():
            av = by_clone_mean(adv[sel] @ u)
            for tag, Vf in fields.items():
                fv = by_clone_mean(Vf[d2[sel]] @ u)
                good = np.isfinite(av) & np.isfinite(fv)
                r = (float(np.corrcoef(fv[good], av[good])[0, 1])
                     if good.sum() > 2 else np.nan)
                dec_rows.append(dict(
                    subset=subset, axis=axis_name, field=tag,
                    corr_of_components=round(r, 4),
                    mean_field_component=round(float(np.nanmean(fv)), 4),
                    mean_advance_component=round(float(np.nanmean(av)), 4),
                    sign_agreement=round(float(np.nanmean(
                        np.sign(fv[good]) == np.sign(av[good]))), 4)))

    pd.DataFrame(cos_rows).to_csv(a.out_prefix + "_cosine.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(a.out_prefix + "_decomposition.csv", index=False)
    say("\n" + pd.DataFrame(cos_rows).to_string(index=False))
    say("\n" + pd.DataFrame(dec_rows).to_string(index=False))
    say("wrote %s_{direction,cosine,decomposition}.csv" % a.out_prefix)


if __name__ == "__main__":
    main()
