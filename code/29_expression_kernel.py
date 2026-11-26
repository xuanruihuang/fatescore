"""Fate probabilities from CellRank's expression kernel, over a tuning grid.

The published comparison this answers reports a velocity kernel beating an
expression kernel on clone-held-out LARRY, with the expression kernel left at
its defaults. Our own evaluation has the velocity kernel's fate probabilities
(cr_velo) and a pseudotime kernel's (cr_pseudo) but no connectivity kernel, so
there is nothing to compare an expression kernel against. This builds them, for
a grid of graph parameters, so that the comparison can be made both at defaults
and at the best the expression kernel can do.

Everything about the pipeline is the same as 02_features.py mode "all": the same
gene filter, normalisation, log, and the same 2,000 highly variable genes, read
back from the feature directory's hvg.json so the gene set is identical rather
than merely recomputed. The terminal states and the absorption-probability
solver are the same as 02c_cellrank.py, so the only thing that differs between
cr_velo and cr_conn_* is the transition matrix.

Nothing here uses a fate label.

Usage:
  python 29_expression_kernel.py <h5ad> <featdir> [--grid "15,30,60"] ...
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

# cellrank 2.0.7 calls np.testing.assert_array_equal(x=..., y=...); numpy 2
# renamed those parameters. Same shim as 02c_cellrank.py.
_orig = npt.assert_array_equal


def _compat(*a, **k):
    if "x" in k:
        k["actual"] = k.pop("x")
    if "y" in k:
        k["desired"] = k.pop("y")
    return _orig(*a, **k)


npt.assert_array_equal = _compat

# scanpy 1.11.5's neighbour code calls .getnnz(axis=1) on the distance matrix,
# and scipy 1.16 dropped that public method from the sparse *array* classes
# while keeping it on the sparse *matrix* classes and keeping ._getnnz on both.
# Which of the two types the neighbour search returns depends on versions that
# the published run inherited rather than pinned, so it worked there and fails
# here. Restoring the pre-removal alias is the smallest fix and changes no
# behaviour: it is the same method under its old name.
if not hasattr(sp.csr_array, "getnnz"):
    sp.csr_array.getnnz = sp.csr_array._getnnz
if not hasattr(sp.csc_array, "getnnz"):
    sp.csc_array.getnnz = sp.csc_array._getnnz

import cellrank as cr  # noqa: E402

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("featdir")
    ap.add_argument("--neighbors", default="15,30,60")
    ap.add_argument("--pcs", default="15,30,50")
    ap.add_argument("--density", default="1,0",
                    help="density_normalize settings for the kernel")
    ap.add_argument("--state-col", default="state_info")
    ap.add_argument("--jobs", type=int, default=8,
                    help="threads for the exact neighbour search")
    ap.add_argument("--d2-only", action="store_true",
                    help="restrict the object to the directory's day-2 cells "
                         "before every step: a reference that sees only the "
                         "present cells (terminal states: the mature day-2 cells)")
    ap.add_argument("--no-gene-filter", action="store_true",
                    help="arm without a splicing layer: skip the "
                         "splicing gene filter and the rescaling")
    a = ap.parse_args()

    hvg = json.load(open(os.path.join(a.featdir, "hvg.json")))
    d2_names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
    say("reusing %d highly variable genes and %d day-2 cells"
        % (len(hvg), len(d2_names)))

    B = ad.read_h5ad(a.h5ad)
    if a.d2_only:
        B = B[d2_names].copy()
        say("day-2 cells only: %s" % (B.shape,))
    # The two arms without a splicing layer are deposited as normalised
    # counts and their gene set was already chosen in 42_build_arm.py, so
    # the splicing-based gene filter has nothing to act on and the counts
    # must not be rescaled twice.
    if not a.no_gene_filter:
        scv.pp.filter_genes(B, min_shared_counts=20)
        scv.pp.normalize_per_cell(B)
    sc.pp.log1p(B)
    keep = [g for g in hvg if g in B.var_names]
    assert len(keep) == len(hvg), (len(keep), len(hvg))
    B = B[:, keep].copy()
    say("object %s" % (B.shape,))
    sc.tl.pca(B, n_comps=50, svd_solver="arpack")

    state = B.obs[a.state_col].values
    # The terminal states belong to the arm, not to this file. MATURE is the
    # in vitro deposit's naming; the two arms without a splicing layer label
    # the same populations with abbreviations, so a hard-coded list silently
    # matches one of them and every kernel comes out as a single column of
    # ones. The arm's own list is the one the rest of the analysis uses, in
    # the order the fate blocks are stored in, so read it from there and only
    # fall back to MATURE when a directory predates it.
    crl = os.path.join(a.featdir, "cellrank_log.json")
    declared = []
    if os.path.exists(crl):
        declared = list(json.load(open(crl)).get("terminal_states", {}).keys())
    wanted = declared or MATURE
    present = [s for s in wanted if (state == s).sum() >= 30]
    assert len(present) >= 2, \
        "only %d terminal states of %s are present with >=30 cells; a " \
        "one-column fate block is a naming mismatch, not a result" % (
            len(present), wanted)
    if declared:
        say("terminal states from %s: %s" % (crl, present))
    term_masks = [np.asarray(state == s) for s in present]
    is_term = np.zeros(B.n_obs, bool)
    for m in term_masks:
        is_term |= m
    trans = ~is_term
    say("terminal states %s, transient %d" % (present, trans.sum()))

    d2_idx = np.array([B.obs_names.get_loc(n) for n in d2_names])
    log = {"terminal_states": {s: int((state == s).sum()) for s in present},
           "n_transient": int(trans.sum()),
           "n_cells_in_object": int(B.n_obs), "grid": []}

    def fate(kernel, tag, tol=1e-8, max_iter=200000):
        T = sp.csr_matrix(kernel.transition_matrix)
        Q = T[trans][:, trans].tocsr()
        R = np.column_stack([np.asarray(T[trans][:, m].sum(1)).ravel()
                             for m in term_masks])
        X = np.zeros_like(R)
        it = 0
        while it < max_iter:
            X = Q @ X + R
            it += 1
            if it % 50 == 0 and float(np.abs(1.0 - X.sum(1)).max()) < tol:
                break
        gap = float(np.abs(1.0 - X.sum(1)).max())
        rs = X.sum(1, keepdims=True)
        X = X / np.where(rs == 0, 1, rs)
        F = np.zeros((B.n_obs, len(present)), dtype=np.float32)
        F[trans] = X
        for j, m in enumerate(term_masks):
            F[m, j] = 1.0
        np.save(os.path.join(a.featdir, "cr_%s.npy" % tag), F[d2_idx])
        say("  %s: %d iterations, row-sum gap %.3g, saved %s"
            % (tag, it, gap, F[d2_idx].shape))
        return dict(tag=tag, iterations=it, max_rowsum_gap=gap,
                    min_value=float(F.min()))

    # The neighbour search backend scanpy picks for an object this size
    # (pynndescent, compiled by numba) does not build against the pinned numpy
    # in the rebuilt environment, so the graph is built with scikit-learn's
    # exact k-nearest neighbours instead. Exact rather than approximate is the
    # better of the two; what it means is that these graphs are not bit-identical
    # to the approximate ones the published run used, which is recorded in the
    # log and in the notes.
    from sklearn.neighbors import KNeighborsTransformer

    def graph(nn, npc):
        sc.pp.neighbors(
            B, n_neighbors=nn, n_pcs=npc, use_rep="X_pca",
            transformer=KNeighborsTransformer(
                n_neighbors=nn, metric="euclidean", algorithm="brute",
                n_jobs=a.jobs))

    log["neighbour_backend"] = "sklearn KNeighborsTransformer, exact, brute"

    for nn in [int(x) for x in a.neighbors.split(",")]:
        for npc in [int(x) for x in a.pcs.split(",")]:
            graph(nn, npc)
            for dn in [bool(int(x)) for x in a.density.split(",")]:
                tag = "conn_n%d_p%d_d%d" % (nn, npc, int(dn))
                say("expression kernel %s" % tag)
                ck = cr.kernels.ConnectivityKernel(B)
                ck.compute_transition_matrix(density_normalize=dn)
                rec = fate(ck, tag)
                rec.update(n_neighbors=nn, n_pcs=npc, density_normalize=dn)
                log["grid"].append(rec)

    json.dump(log, open(os.path.join(a.featdir, "expression_kernel_log.json"),
                        "w"), indent=1, default=str)
    say("done, %d kernels" % len(log["grid"]))


if __name__ == "__main__":
    main()
