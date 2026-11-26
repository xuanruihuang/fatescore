"""What sign agreement is worth, and a coarse maturation proxy worth comparing to.

Two things this answers, both registered before it ran
(notes/velocity_fate_extension_log.md section 14).

Sign agreement between a field's component along an axis and the clonal
advance's component along the same axis has a chance level set by the margin,
not by one half: almost every clone advances toward maturity, so a field that
always points that way agrees most of the time. So each axis is reported with
two nulls -- a constant field at the majority direction of the advance, which is
that axis's chance level, and the velocity arrows permuted between early cells
of the same annotated state -- and with the velocity field's excess over both.
The fate axis also gets the sign distribution of the velocity component itself,
since a field that leans one way on that axis can score below one half without
predicting anything in reverse.

And the first maturation proxy was poor in a way I could name -- nearest
next-day cells are pulled by local density -- so three coarser ones are built
from sampling day and state annotation alone, with no clones and no splicing:
per-state displacement to the same state's next-day centroid, the same thing in
a 500-cell neighbourhood, and a single global vector from the day-2 centroid to
the later cells', which is the floor.

Usage:
  python 40_chance_and_coarse.py <h5ad> <featdir> <labels.tsv> <out_prefix>
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
N_PERM = 20
K_COARSE = 500


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def cos_pairs(P, Q):
    num = (P * Q).sum(1)
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(Q, axis=1)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def axis_from(Xe, y):
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
    state = B.obs["state"].values
    days = sorted(set(np.unique(tnum[~np.isnan(tnum)]).tolist()))
    say("object %s, days %s" % (B.shape, days))

    fields = {}
    scv.tl.recover_dynamics(B, n_jobs=a.jobs, var_names="all")
    scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
    scv.tl.velocity_graph(B, vkey="vel_dyn", n_jobs=a.jobs)
    scv.tl.velocity_embedding(B, vkey="vel_dyn", basis=a.basis)
    fields["V2_scvelo_dynamical"] = np.asarray(B.obsm["vel_dyn_" + a.basis],
                                               dtype=np.float64)
    say("reference velocity field embedded")

    # C1: per annotated state, the displacement to that state's next-day centroid
    C1 = np.full_like(Xe, np.nan)
    for s in np.unique(state):
        for t, t1 in zip(days[:-1], days[1:]):
            here = (state == s) & (tnum == t)
            there = (state == s) & (tnum == t1)
            if here.sum() == 0 or there.sum() < 5:
                continue
            C1[here] = Xe[there].mean(0) - Xe[here].mean(0)
    fields["C1_state_coarse"] = C1
    say("C1: %d cells have an arrow" % int(np.isfinite(C1).all(1).sum()))

    # C2: the same idea in a 500-cell neighbourhood, with no state annotation
    nn = NearestNeighbors(n_neighbors=K_COARSE, algorithm="brute").fit(X)
    C2 = np.full_like(Xe, np.nan)
    nxt = {t: t1 for t, t1 in zip(days[:-1], days[1:])}
    for t in days[:-1]:
        idx = np.where(tnum == t)[0]
        if idx.size == 0:
            continue
        _, ind = nn.kneighbors(X[idx])
        for r, i in enumerate(idx):
            nb = ind[r]
            same = nb[tnum[nb] == t]
            later = nb[tnum[nb] == nxt[t]]
            if same.size >= 5 and later.size >= 5:
                C2[i] = Xe[later].mean(0) - Xe[same].mean(0)
        say("  C2 day %s: %d cells" % (t, idx.size))
    fields["C2_nbhd%d" % K_COARSE] = C2
    say("C2: %d cells have an arrow" % int(np.isfinite(C2).all(1).sum()))

    # C3: one global vector, the floor
    g = Xe[tnum > days[0]].mean(0) - Xe[tnum == days[0]].mean(0)
    fields["C3_global_constant"] = np.broadcast_to(g, Xe.shape).copy()
    say("C3 global vector = (%.1f, %.1f)" % (g[0], g[1]))

    # the clonal advance
    d2 = np.array([B.obs_names.get_loc(n) for n in d2_names])
    is_early = tnum == days[0]
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

    mature_mask = np.isin(state, [t for t in MATURE if (state == t).sum() >= 30])
    ax = {}
    sel = is_early & (mature_mask | (state == UNDIFF))
    ax["maturation"] = axis_from(Xe[sel], mature_mask[sel].astype(int))
    ax["sampling_day"] = axis_from(Xe, (~is_early).astype(int))
    fa = taskA & (n_late > 0)
    ax["fate"] = axis_from(Xe[d2[fa]],
                           (T.loc[fa, "dom"].values == "Neutrophil").astype(int))
    for nm, v in ax.items():
        say("axis %-13s = (%.3f, %.3f)" % (nm, v[0], v[1]))

    # the permutation null: velocity arrows shuffled within annotated state
    perms = []
    V = fields["V2_scvelo_dynamical"]
    st_d2 = state[d2]
    for s in range(N_PERM):
        r = np.random.default_rng(SEED + 500 + s)
        P = V[d2].copy()
        for u in np.unique(st_d2):
            w = np.where(st_d2 == u)[0]
            P[w] = V[d2][w[r.permutation(w.size)]]
        perms.append(P)
    say("permutation null: %d draws, within %d annotated states"
        % (N_PERM, len(np.unique(st_d2))))

    rng = np.random.default_rng(SEED)
    cos_rows, sign_rows = [], []
    ref = "V2_scvelo_dynamical"
    for subset, mask in (("all early cells with an advance", n_late > 0),
                         ("task A clones", (n_late > 0) & taskA)):
        sel = np.where(mask)[0]
        clones = clone_of[d2[sel]]
        keys = np.unique(clones)

        def cm(v):
            return (pd.Series(v, index=clones).groupby(level=0).mean()
                    .reindex(keys).to_numpy())

        cols, names = [], []
        for tag, Vf in fields.items():
            cols.append(cm(cos_pairs(Vf[d2[sel]], adv[sel])))
            names.append(tag)
        ok = np.isfinite(adv_a[sel]).all(1) & np.isfinite(adv_b[sel]).all(1)
        ceil = np.full(sel.size, np.nan)
        ceil[ok] = cos_pairs(adv_a[sel][ok], adv_b[sel][ok])
        cols.append(cm(ceil))
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
            if nm != ref:
                d = bm[:, j] - bm[:, jr]
                lo2, hi2 = np.nanpercentile(d, [2.5, 97.5])
                cos_rows.append(dict(
                    subset=subset, field="%s minus %s" % (nm, ref),
                    clones=len(keys),
                    cosine=round(float(np.nanmean(tab[:, j])
                                       - np.nanmean(tab[:, jr])), 4),
                    lo=round(float(lo2), 4), hi=round(float(hi2), 4)))

        # sign agreement with its chance level, per axis
        for axis_name, u in ax.items():
            av = cm(adv[sel] @ u)
            pos = float(np.nanmean(av > 0))
            chance = max(pos, 1.0 - pos)
            sign_rows.append(dict(subset=subset, axis=axis_name,
                                  field="CHANCE_constant_majority",
                                  sign_agreement=round(chance, 4),
                                  excess_over_chance=0.0, lo="", hi="",
                                  note="advance component positive in %.3f of clones"
                                       % pos))
            agree = {}
            for tag, Vf in fields.items():
                fv = cm(Vf[d2[sel]] @ u)
                good = np.isfinite(av) & np.isfinite(fv)
                s_i = (np.sign(fv) == np.sign(av)).astype(float)
                s_i[~good] = np.nan
                agree[tag] = s_i
                bs = np.nanmean(s_i[idx], axis=1)
                lo, hi = np.nanpercentile(bs - chance, [2.5, 97.5])
                sign_rows.append(dict(
                    subset=subset, axis=axis_name, field=tag,
                    sign_agreement=round(float(np.nanmean(s_i)), 4),
                    excess_over_chance=round(float(np.nanmean(s_i)) - chance, 4),
                    lo=round(float(lo), 4), hi=round(float(hi), 4),
                    note="field component positive in %.3f of clones"
                         % float(np.nanmean(cm(Vf[d2[sel]] @ u) > 0))))
            # the permutation null
            pv = []
            for P in perms:
                fv = cm(P[sel] @ u)
                good = np.isfinite(av) & np.isfinite(fv)
                pv.append(float(np.nanmean((np.sign(fv) == np.sign(av))[good])))
            sign_rows.append(dict(
                subset=subset, axis=axis_name, field="NULL_permuted_within_state",
                sign_agreement=round(float(np.mean(pv)), 4),
                excess_over_chance=round(float(np.mean(pv)) - chance, 4),
                lo=round(float(np.min(pv)), 4), hi=round(float(np.max(pv)), 4),
                note="%d permutations, min and max shown" % N_PERM))
            # velocity's excess over the permutation null, paired over clones
            s_v = agree[ref]
            d_perm = []
            for P in perms:
                fv = cm(P[sel] @ u)
                s_p = (np.sign(fv) == np.sign(av)).astype(float)
                s_p[~(np.isfinite(av) & np.isfinite(fv))] = np.nan
                d_perm.append(np.nanmean((s_v - s_p)[idx], axis=1))
            dp = np.stack(d_perm).mean(0)
            lo, hi = np.nanpercentile(dp, [2.5, 97.5])
            sign_rows.append(dict(
                subset=subset, axis=axis_name,
                field="%s minus NULL_permuted" % ref,
                sign_agreement="", excess_over_chance=round(float(dp.mean()), 4),
                lo=round(float(lo), 4), hi=round(float(hi), 4),
                note="paired over clones, averaged over permutations"))

    pd.DataFrame(cos_rows).to_csv(a.out_prefix + "_cosine.csv", index=False)
    pd.DataFrame(sign_rows).to_csv(a.out_prefix + "_sign.csv", index=False)
    say("\n" + pd.DataFrame(cos_rows).to_string(index=False))
    say("\n" + pd.DataFrame(sign_rows).to_string(index=False))
    say("wrote %s_{cosine,sign}.csv" % a.out_prefix)


if __name__ == "__main__":
    main()
