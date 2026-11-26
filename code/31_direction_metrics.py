"""The field's own direction metrics, on the same data and the same methods.

Velocity methods are ranked in the literature mostly by expert direction
metrics: whether the arrow at a cell on a cluster boundary points at the
cluster that is known to come next (cross-boundary direction correctness), and
whether neighbouring arrows agree (in-cluster coherence). This computes those
two on the first dataset for every velocity field we can build, so that they
can be put beside the clone-level fate increment the same arms get. The
question the section asks is whether a method that scores well on the field's
metrics carries more information about measured clonal fate.

Two things make the comparison interpretable. The boundaries are the ones the
annotation already fixes -- undifferentiated cells to each mature population --
so no new reference is invented. And one of the fields contains no splicing and
no dynamics at all: the mean displacement to a cell's own neighbours in
expression space. If the direction metrics cannot tell that field from a
velocity field, they are not measuring what they are taken to measure.

Everything upstream is the same as 02_features.py mode "all", except that the
neighbour graph is built with scikit-learn's exact search, because scanpy's
default backend does not compile against the pinned numpy in the rebuilt
environment.

Usage:
  python 31_direction_metrics.py <h5ad> <featdir> <out.csv> [--jobs 16]
"""
import argparse
import json
import os
import time

import numpy as np
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
# scipy 1.16 dropped .getnnz from the sparse array classes; scanpy 1.11 still
# calls it. Same restoration as 29_expression_kernel.py.
for _cls in (sp.csr_array, sp.csc_array):
    if not hasattr(_cls, "getnnz"):
        _cls.getnnz = _cls._getnnz

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]
UNDIFF = "Undifferentiated"
N_PCS, N_NEIGHBORS = 30, 30


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def cosine_rows(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def cbdir(V, X, nbr, state, targets):
    """Cross-boundary direction correctness: at an undifferentiated cell with a
    neighbour in a mature population, does the arrow point at that neighbour?
    Returned per target population and pooled over them."""
    out = {}
    src = state == UNDIFF
    pooled = []
    for t in targets:
        tgt = state == t
        cos = []
        for i in np.where(src)[0]:
            js = nbr[i]
            js = js[tgt[js]]
            if js.size == 0:
                continue
            d = X[js] - X[i]
            v = np.repeat(V[i][None, :], js.size, axis=0)
            c = cosine_rows(v, d)
            c = c[np.isfinite(c)]
            if c.size:
                cos.append(float(c.mean()))
        out["cbdir_" + t] = float(np.mean(cos)) if cos else np.nan
        out["cbdir_n_" + t] = len(cos)
        pooled += cos
    out["cbdir_pooled"] = float(np.mean(pooled)) if pooled else np.nan
    out["cbdir_n_pooled"] = len(pooled)
    return out


def cbdir_reverse(V, X, nbr, state, targets):
    """The same score with the boundary taken the wrong way round: at a cell of
    a mature population with an undifferentiated neighbour, does the arrow
    point at that neighbour? A field that follows the annotated order should
    score low here; a field that only follows neighbourhood geometry scores
    about as well as forward."""
    und = state == UNDIFF
    pooled = []
    for t in targets:
        for i in np.where(state == t)[0]:
            js = nbr[i]
            js = js[und[js]]
            if js.size == 0:
                continue
            d = X[js] - X[i]
            c = cosine_rows(np.repeat(V[i][None, :], js.size, axis=0), d)
            c = c[np.isfinite(c)]
            if c.size:
                pooled.append(float(c.mean()))
    return {"cbdir_reverse_pooled": float(np.mean(pooled)) if pooled else np.nan,
            "cbdir_reverse_n": len(pooled)}


def iccoh(V, nbr, state):
    """In-cluster coherence: agreement between an arrow and its same-cluster
    neighbours' arrows."""
    vals = []
    for i in range(V.shape[0]):
        js = nbr[i]
        js = js[state[js] == state[i]]
        if js.size == 0:
            continue
        c = cosine_rows(np.repeat(V[i][None, :], js.size, axis=0), V[js])
        c = c[np.isfinite(c)]
        if c.size:
            vals.append(float(c.mean()))
    return float(np.mean(vals)) if vals else np.nan


def save_fields(fields, B, featdir):
    """Each field's day-2 rows, named so 30_kernel_eval.py can pick them up."""
    d2_names = json.load(open(os.path.join(featdir, "d2_names.json")))
    idx = np.array([B.obs_names.get_loc(n) for n in d2_names])
    for name, Vf in fields.items():
        p = os.path.join(featdir, "field_%s.npy" % name)
        np.save(p, np.asarray(Vf[idx], dtype=np.float32))
        say("saved %s %s" % (p, Vf[idx].shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("featdir")
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--state-col", default="state_info")
    ap.add_argument("--save-fields", action="store_true",
                    help="also save each field's day-2 rows into the feature "
                         "directory, so a field can be scored as an arm")
    ap.add_argument("--drift-only", action="store_true",
                    help="skip the velocity stages; builds the expression-only "
                         "drift field alone, which needs no dynamics")
    ap.add_argument("--controls-only", action="store_true",
                    help="with --structure-controls: stop after the steady-state "
                         "field and its controls (skip the dynamical model and "
                         "veloVI)")
    ap.add_argument("--structure-controls", action="store_true",
                    help="score only the steady-state field and two fields "
                         "that keep its construction but carry none of the "
                         "cell's own unspliced information (Mu taken from a "
                         "random other cell, minus gamma*Ms; and -gamma*Ms "
                         "alone), plus the drift field; skips the dynamical "
                         "model and veloVI")
    a = ap.parse_args()

    hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
    B = ad.read_h5ad(a.h5ad)
    state_all = B.obs[a.state_col].astype(str).values
    scv.pp.filter_genes(B, min_shared_counts=20)
    scv.pp.normalize_per_cell(B)
    sc.pp.log1p(B)
    keep = [g for g in hvg if g in B.var_names]
    assert len(keep) == len(hvg), (len(keep), len(hvg))
    B = B[:, keep].copy()
    B.obs["state"] = state_all
    say("object %s" % (B.shape,))

    scv.pp.neighbors(B, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS, method="sklearn")
    scv.pp.moments(B)
    say("moments done")

    pcs = B.varm["PCs"][:, :N_PCS] if "PCs" in B.varm else None
    if pcs is None:
        sc.tl.pca(B, n_comps=N_PCS, svd_solver="arpack")
        pcs = B.varm["PCs"][:, :N_PCS]
    X = np.asarray(B.obsm["X_pca"][:, :N_PCS], dtype=np.float64)

    # neighbour index lists from the graph the moments used
    D = sp.csr_matrix(B.obsp["distances"])
    nbr = [D.indices[D.indptr[i]:D.indptr[i + 1]] for i in range(B.n_obs)]
    say("median neighbours %d" % int(np.median([len(x) for x in nbr])))

    fields = {}

    # a field with no splicing and no dynamics in it at all
    drift = np.zeros_like(X)
    for i in range(B.n_obs):
        js = nbr[i]
        if js.size:
            drift[i] = X[js].mean(0) - X[i]
    fields["E0_neighbour_drift"] = drift
    say("expression-only drift field built")

    if a.drift_only:
        if a.save_fields:
            save_fields(fields, B, a.featdir)
        say("drift field only, as asked; no velocity stage")
        rows = []
        state = B.obs["state"].values
        targets = [t for t in MATURE if (state == t).sum() >= 30]
        for name, Vf in fields.items():
            rec = {"field": name}
            rec.update(cbdir(Vf, X, nbr, state, targets))
            rec.update(cbdir_reverse(Vf, X, nbr, state, targets))
            rec["iccoh"] = iccoh(Vf, nbr, state)
            rows.append(rec)
        import pandas as pd
        pd.DataFrame(rows).to_csv(a.out, index=False)
        say("wrote %s" % a.out)
        return

    scv.tl.velocity(B, mode="deterministic", vkey="vel_det")
    fields["V1_scvelo_steady"] = np.asarray(
        np.nan_to_num(B.layers["vel_det"]), dtype=np.float64) @ pcs
    say("scvelo steady-state done")

    if a.structure_controls:
        # The steady-state velocity is u - gamma*s on the moments. Keep each
        # gene's gamma and the set of velocity genes, and replace the cell's
        # own unspliced moments by another cell's (one fixed permutation over
        # all cells), or drop u altogether. Neither field contains anything the
        # cell's own unspliced reads say; whatever they score on a direction
        # metric comes from the construction.
        Ms = np.asarray(B.layers["Ms"], dtype=np.float64)
        Mu = np.asarray(B.layers["Mu"], dtype=np.float64)
        gam = np.nan_to_num(np.asarray(B.var["vel_det_gamma"], dtype=np.float64))
        vg = np.asarray(B.var["vel_det_genes"], dtype=bool)
        chk = np.where(vg, Mu - gam * Ms, 0.0)
        ref = np.nan_to_num(B.layers["vel_det"])
        say("steady-state reconstruction max |u - gamma s - vel_det| on "
            "velocity genes: %.3g" % float(np.abs(chk[:, vg] - ref[:, vg]).max()))
        perm = np.random.default_rng(31).permutation(B.n_obs)
        fields["V1c_steady_shuffled_u"] = np.where(vg, Mu[perm] - gam * Ms, 0.0) @ pcs
        fields["V1g_minus_gamma_s"] = np.where(vg, -gam * Ms, 0.0) @ pcs
        rng = np.random.default_rng(32)
        fields["R0_random"] = rng.standard_normal(X.shape)
    if a.structure_controls and a.controls_only:
        state = B.obs["state"].values
        targets = [t for t in MATURE if (state == t).sum() >= 30]
        rows = []
        for name, Vf in fields.items():
            rec = {"field": name}
            rec.update(cbdir(Vf, X, nbr, state, targets))
            rec.update(cbdir_reverse(Vf, X, nbr, state, targets))
            rec["iccoh"] = iccoh(Vf, nbr, state)
            rows.append(rec)
            say("%-24s cbdir_pooled %.3f  iccoh %.3f"
                % (name, rec["cbdir_pooled"], rec["iccoh"]))
        import pandas as pd
        pd.DataFrame(rows).to_csv(a.out, index=False)
        say("wrote %s" % a.out)
        return

    scv.tl.recover_dynamics(B, n_jobs=a.jobs, var_names="all")
    scv.tl.velocity(B, mode="dynamical", vkey="vel_dyn")
    fields["V2_scvelo_dynamical"] = np.asarray(
        np.nan_to_num(B.layers["vel_dyn"]), dtype=np.float64) @ pcs
    say("scvelo dynamical done")

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
        fields["V3_velovi"] = full @ pcs
        say("veloVI done on %d genes" % V.n_vars)
    except Exception as e:
        say("veloVI FAILED: %s: %s" % (type(e).__name__, e))

    state = B.obs["state"].values
    targets = [t for t in MATURE if (state == t).sum() >= 30]
    say("boundary targets: %s" % targets)

    rows = []
    for name, Vf in fields.items():
        t0 = time.time()
        rec = {"field": name}
        rec.update(cbdir(Vf, X, nbr, state, targets))
        rec.update(cbdir_reverse(Vf, X, nbr, state, targets))
        rec["iccoh"] = iccoh(Vf, nbr, state)
        rows.append(rec)
        say("%-22s cbdir_pooled %.3f  iccoh %.3f  (%.0f s)"
            % (name, rec["cbdir_pooled"], rec["iccoh"], time.time() - t0))

    if a.save_fields:
        save_fields(fields, B, a.featdir)

    import pandas as pd
    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    say("wrote %s" % a.out)


if __name__ == "__main__":
    main()
