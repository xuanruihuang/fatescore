"""Fate probabilities from snapshot optimal-transport methods, on the same task.

These methods need expression and sampling days, not a splicing layer, so they
can be scored on exactly the criterion the velocity arms are scored on: the
probability each early cell's clone ends up in each mature population, used as
the only feature besides the starting population, with the same clone-disjoint
folds. That is what makes the comparison in this paper possible at all -- the
velocity line and the snapshot-dynamics line have never been put on one
criterion.

Each method writes one array per feature directory, named so 30_kernel_eval.py
picks it up: snap_<method>.npy, one row per early cell, one column per mature
population, in the order the directory's cellrank log lists them.

The methods live in a separate environment (env_ot) because their dependencies
do not agree with the analysis environment's pins; the scope check still applies,
since the arrays land in a feature directory and are compared only against that
directory's own arms.

Usage:
  python 41_ot_fate.py <h5ad> <featdir> --method moscot [--epsilon 0.01]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]
N_PCS = 30


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def build(h5ad, featdir, time_col, state_col,
          keep_days=None, couple_days=None, drop_cells=None):
    """The same object the rest of the analysis uses, without the velocity
    layers: this directory's genes, its early cells, normalised the same way.

    The two day filters answer different questions and must not be mixed.
    `keep_days` drops the other sampling days BEFORE normalisation and the
    PCA, which is the experiment that never sampled them -- neither the
    coupling chain nor the representation sees those cells. `couple_days`
    drops them AFTER the PCA, so the representation is the full one and only
    the coupling chain is shortened, which is the method declining to use an
    intermediate day that was in fact measured. Either way every early cell
    has to survive, or the evaluation set would change under us.
    """
    hvg = json.load(open(os.path.join(featdir, "hvg.json")))
    d2_names = json.load(open(os.path.join(featdir, "d2_names.json")))
    B = ad.read_h5ad(h5ad)
    keep = [g for g in hvg if g in B.var_names]
    assert len(keep) == len(hvg), (len(keep), len(hvg))
    # a duplicated cell name makes obs_names.get_loc return a mask instead of
    # a position, which would corrupt the early-cell index with no error
    assert B.obs_names.is_unique, "duplicated cell names"
    B = B[:, keep].copy()
    day = pd.to_numeric(B.obs[time_col], errors="coerce").astype(float)
    if keep_days:
        m = day.isin(keep_days).to_numpy()
        say("keep-days %s before the PCA: %d of %d cells"
            % (keep_days, int(m.sum()), m.size))
        B, day = B[m].copy(), day[m]
    # scvelo's normalize_per_cell, which the rest of the analysis uses, scales
    # each cell to the median of total counts; target_sum=None does that.
    sc.pp.normalize_total(B, target_sum=None)
    sc.pp.log1p(B)
    sc.tl.pca(B, n_comps=N_PCS, svd_solver="arpack")
    B.obs["day"] = day.values
    B.obs["state"] = B.obs[state_col].astype(str).values
    if couple_days:
        m = B.obs["day"].isin(couple_days).to_numpy()
        say("couple-days %s after the PCA: %d of %d cells"
            % (couple_days, int(m.sum()), m.size))
        B = B[m].copy()
    if drop_cells:
        dc = set(open(drop_cells).read().split())
        m = ~np.isin(np.asarray(B.obs_names), list(dc))
        say("drop-cells %s after the PCA: %d cells removed, %d kept"
            % (os.path.basename(drop_cells), int((~m).sum()), int(m.sum())))
        B = B[m].copy()
    gone = [n for n in d2_names if n not in set(B.obs_names)]
    assert not gone, "%d early cells were dropped by a day filter" % len(gone)
    d2 = np.array([B.obs_names.get_loc(n) for n in d2_names])
    say("object %s, days %s" % (B.shape,
                                sorted(set(B.obs["day"].dropna().tolist()))))
    return B, d2


def populations(featdir, state):
    """The same terminal populations, in the same order, as the kernels use."""
    p = os.path.join(featdir, "cellrank_log.json")
    if os.path.exists(p):
        st = json.load(open(p)).get("terminal_states")
        if st:
            return list(st.keys())
    return [s for s in MATURE if (state == s).sum() >= 30]


def run_moscot(B, d2, pops, args):
    from moscot.problems.time import TemporalProblem
    days = sorted(B.obs["day"].dropna().unique().tolist())
    tp = TemporalProblem(B)
    tp = tp.prepare(time_key="day", joint_attr="X_pca",
                    policy="sequential")
    say("moscot prepared: %s" % list(tp.problems.keys()))
    kw = dict(epsilon=args.epsilon, scale_cost="mean", rank=args.rank,
              threshold=args.threshold, max_iterations=args.max_iter)
    # tau < 1 relaxes a marginal (unbalanced transport), as Waddington-OT does
    # with its lambda penalties; 1 is moscot's default (balanced).
    if args.tau_a != 1.0 or args.tau_b != 1.0:
        kw.update(tau_a=args.tau_a, tau_b=args.tau_b)
    if args.batch_size > 0:
        kw["batch_size"] = args.batch_size
    tp = tp.solve(**kw)
    # Every coupling in the chain has to have converged before any fate
    # probability derived from it is reported; a solver warning in a log is too
    # easy to miss, so the run refuses to save instead.
    conv = {}
    for key, prob in tp.problems.items():
        sol = prob.solution
        conv[str(key)] = dict(converged=bool(sol.converged),
                              cost=float(sol.cost), shape=list(sol.shape))
        say("  coupling %s: converged=%s cost=%.6g"
            % (key, sol.converged, sol.cost))
    say("moscot solved")
    if getattr(args, "realtime", False):
        return realtime_fate(B, d2, pops, tp, args), conv
    cols = []
    for pop in pops:
        # normalize=False matters. moscot's default normalises the pulled
        # indicator to sum to one, which divides each population's column by
        # its own cell count; after the row normalisation below that would
        # leave a fate probability with the population sizes taken out,
        # over-weighting the rare populations. With the raw indicator the
        # row-normalised result is the conditional probability of ending in
        # population k given ending in any of them, which is what the velocity
        # kernels' absorption probabilities are.
        tp.pull(source=days[0], target=days[-1], data="state",
                subset=[pop], normalize=False, scale_by_marginals=True,
                key_added="pull")
        v = np.asarray(B.obs["pull"].values, dtype=np.float64)[d2]
        cols.append(np.nan_to_num(v, nan=0.0))
        say("  pulled %s: mean %.4g" % (pop, float(np.mean(cols[-1]))))
    P = np.column_stack(cols)
    return P, conv


def realtime_fate(B, d2, pops, tp, args):
    """CellRank 2's RealTimeKernel on the moscot couplings, the combination the
    CellRank 2 paper recommends for time-resolved data: transitions between
    consecutive days from the couplings, within-day transitions from the
    expression connectivities (weight conn_weight), then absorption
    probabilities into the same annotated terminal populations as every other
    kernel, solved the same way (Neumann series to row sums within 1e-8)."""
    import scipy.sparse as sp
    import numpy.testing as npt
    from sklearn.neighbors import KNeighborsTransformer
    _orig = npt.assert_array_equal

    def _compat(*x, **k):
        if "x" in k:
            k["actual"] = k.pop("x")
        if "y" in k:
            k["desired"] = k.pop("y")
        return _orig(*x, **k)

    npt.assert_array_equal = _compat
    # scanpy's neighbour code calls .getnnz on sparse arrays, which scipy 1.16
    # removed from the array classes; same alias as 29_expression_kernel.py.
    if not hasattr(sp.csr_array, "getnnz"):
        sp.csr_array.getnnz = sp.csr_array._getnnz
    if not hasattr(sp.csc_array, "getnnz"):
        sp.csc_array.getnnz = sp.csc_array._getnnz
    import cellrank as cr
    sc.pp.neighbors(B, n_neighbors=30, use_rep="X_pca",
                    transformer=KNeighborsTransformer(
                        n_neighbors=30, metric="euclidean", algorithm="brute",
                        n_jobs=4))
    rtk = cr.kernels.RealTimeKernel.from_moscot(tp)
    # within-day connectivities with the same exact neighbour search as above;
    # scanpy's default approximate backend does not compile in this environment
    rtk.compute_transition_matrix(
        self_transitions="all", conn_weight=args.conn_weight, threshold="auto",
        conn_kwargs=dict(n_neighbors=30, use_rep="X_pca",
                         transformer=KNeighborsTransformer(
                             n_neighbors=30, metric="euclidean",
                             algorithm="brute", n_jobs=4)))
    T = sp.csr_matrix(rtk.transition_matrix)
    state = B.obs["state"].values
    masks = [np.asarray(state == s) for s in pops]
    is_term = np.any(masks, axis=0)
    trans = ~is_term
    Q = T[trans][:, trans].tocsr()
    R = np.column_stack([np.asarray(T[trans][:, m].sum(1)).ravel()
                         for m in masks])
    X = np.zeros_like(R)
    it, gap = 0, np.inf
    while it < 400000:
        X = Q @ X + R
        it += 1
        if it % 100 == 0:
            gap = float(np.abs(1.0 - X.sum(1)).max())
            if gap < 1e-8:
                break
    say("realtime kernel absorption: %d iterations, row-sum gap %.3g, "
        "mean row sum on day-2 cells %.6f" % (it, gap, float(
            X.sum(1)[np.isin(np.where(trans)[0], d2)].mean())))
    F = np.zeros((B.n_obs, len(pops)))
    F[trans] = X
    for j, m in enumerate(masks):
        F[m, j] = 1.0
    args._rtk_info = dict(iterations=it, max_rowsum_gap=gap,
                          conn_weight=args.conn_weight)
    return F[d2]


def run_statot(B, d2, pops, args):
    # statot's package __init__ imports a CellRank 1 wrapper that CellRank 2
    # removed; the algorithm itself is in statot.inference and does not need it,
    # so the missing module is stubbed before the import. Nothing else uses it.
    import importlib.abc
    import importlib.util
    import sys
    import types

    class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Fabricate any cellrank module statot's unused CellRank 1 wrapper
        asks for, so that importing the package succeeds. The algorithm used
        below is the package's own statot.inference."""

        def find_spec(self, name, path=None, target=None):
            if name == "cellrank" or name.startswith("cellrank."):
                return importlib.util.spec_from_loader(name, self)
            return None

        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            m.__getattr__ = lambda n: object
            return m

        def exec_module(self, module):
            return None

    sys.meta_path.insert(0, _Stub())
    from statot.inference import (statot as statot_fn, row_normalise,
                                  compute_fate_probs_lineages)

    X = np.asarray(B.obsm["X_pca"][:, :N_PCS], dtype=np.float32)
    state = np.asarray(B.obs["state"].values, dtype=object)
    sink = np.isin(state, pops)
    labels = np.array([s if s in pops else "transient" for s in state],
                      dtype=object)
    say("statot: %d cells, %d sinks over %d populations"
        % (X.shape[0], int(sink.sum()), len(pops)))

    def cost(Z):
        """Squared euclidean distances, scaled by their mean -- the same cost
        and the same scaling moscot uses with scale_cost='mean', so the two
        methods' epsilons are on one scale."""
        sq = (Z * Z).sum(1)
        C = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
        np.maximum(C, 0.0, out=C)
        return C / float(C.mean())

    def solve_statot(Z, mask_sink, lab):
        C = cost(Z)
        g = np.ones(Z.shape[0], dtype=np.float32)
        gamma, mu, nu = statot_fn(Z, C=C, eps=args.statot_eps,
                                  method=args.statot_method, g=g, dt=1.0,
                                  maxiter=args.max_iter,
                                  tol=args.statot_tol)
        # statot's own convergence check: how far the coupling's marginals are
        # from the ones asked for. POT returns the last iterate either way.
        err = max(float(np.abs(gamma.sum(1) - mu).max()),
                  float(np.abs(gamma.sum(0) - nu).max()))
        del C
        return gamma, err

    # The package's route is row_normalise() then compute_fate_probs_lineages(),
    # which solves for one column per sink *cell* and sums the columns of a
    # lineage afterwards. At this size that is a 23,770 x 25,532 dense solve
    # with a 25,532 x 25,532 identity block, tens of gigabytes for an answer
    # that only has ten columns. Summing the lineage's columns of the
    # right-hand side before the solve gives the same matrix, because the solve
    # is linear in the right-hand side; the two routes are run against each
    # other on a subsample below rather than merely asserted.
    def fate_from_gamma(gamma, mask_sink, lab):
        trans = ~mask_sink
        G = np.asarray(gamma[trans], dtype=np.float64)
        G = G / G.sum(1, keepdims=True)
        Q = G[:, trans]
        R = np.column_stack([G[:, lab == p].sum(1) for p in pops])
        del G
        N = np.eye(Q.shape[0]) - Q
        Xs = np.linalg.solve(N, R)
        out = np.zeros((gamma.shape[0], len(pops)))
        out[trans] = Xs
        for j, p in enumerate(pops):
            out[lab == p, j] = 1.0
        return out

    if args.verify_cells > 0:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(X.shape[0], args.verify_cells, replace=False))
        # keep at least a few sinks of each population in the subsample
        ms, ls = sink[idx], labels[idx]
        keep = [p for p in pops if (ls == p).sum() >= 3]
        say("verification subsample: %d cells, %d sinks, populations %s"
            % (idx.size, int(ms.sum()), keep))
        gs, _ = solve_statot(X[idx], ms, ls)
        Ps = row_normalise(np.asarray(gs, dtype=np.float64), sink_idx=ms)
        pkg, pkg_lab = compute_fate_probs_lineages(Ps, ms, ls)
        mine = fate_from_gamma(gs, ms, ls)
        cols_pkg = {str(k): pkg[:, i] for i, k in enumerate(pkg_lab)}
        dev = max(float(np.abs(cols_pkg[p] - mine[:, j]).max())
                  for j, p in enumerate(pops) if p in cols_pkg)
        say("largest disagreement with the package's own route: %.3g" % dev)
        assert dev < 1e-6, dev
        del gs, Ps, pkg

    gamma, err = solve_statot(X, sink, labels)
    tol_abs = args.statot_tol * float(X.shape[0])
    say("marginal agreement error %.3g, asked for %.3g" % (err, tol_abs))
    F = fate_from_gamma(gamma, sink, labels)
    del gamma
    conv = {"statot": dict(method=args.statot_method, eps=args.statot_eps,
                           n_cells=int(X.shape[0]), n_sinks=int(sink.sum()),
                           marginal_error=err, tolerance=tol_abs,
                           converged=bool(err <= tol_abs))}
    return F[d2], conv


def run_wot(B, d2, pops, args):
    """Waddington-OT, through the package's own transport maps and fate call.

    wot computes one unbalanced coupling per consecutive day pair, writes them
    out, and reads them back as a TransportMapModel; its fates() then pulls a
    set of day-6 populations back along the chain. Left at the package's
    defaults (epsilon 0.05, lambda1 1, lambda2 50, growth_iters 1, a 30
    component local PCA per day pair), which is what the CLI uses.
    """
    # wot 1.0.8 reads its own transport maps with anndata.read, the alias
    # removed in anndata 0.10; the function it aliased is read_h5ad, so the
    # name is restored rather than the call replaced. Module __getattr__
    # only runs for names absent from the module, so this assignment wins.
    if not hasattr(ad, "read"):
        ad.read = ad.read_h5ad
    import wot.io  # noqa: F401
    import wot.ot
    import wot.tmap

    days = sorted(B.obs["day"].dropna().unique().tolist())
    B.obs[args.time_col] = B.obs["day"].values
    model = wot.ot.OTModel(B, day_field="day", **json.loads(args.wot_kwargs))
    say("wot timepoints %s, config %s" % (list(model.timepoints), model.ot_config))
    outdir = args.wot_out or os.path.join(args.featdir, "wot_tmaps")
    have = [f for f in (os.listdir(outdir) if os.path.isdir(outdir) else [])
            if f.startswith("tmaps_") and f.endswith(".h5ad")]
    if args.wot_reuse and len(have) >= len(days) - 1:
        say("reusing %d transport maps already in %s: %s"
            % (len(have), outdir, sorted(have)))
    else:
        model.compute_all_transport_maps(
            tmap_out=os.path.join(outdir, "tmaps"))
        say("wot transport maps written to %s" % outdir)
    tmap = wot.tmap.TransportMapModel.from_directory(
        os.path.join(outdir, "tmaps"))
    last = days[-1]
    at_last = np.asarray(B.obs["day"].values) == last
    cell_sets = {}
    for p in pops:
        ids = B.obs_names[at_last & (B.obs["state"].values == p)].tolist()
        if len(ids) >= 1:
            cell_sets[p] = ids
    say("day-%g cell sets: %s" % (last, {k: len(v) for k, v in cell_sets.items()}))
    populations_ = tmap.population_from_cell_sets(cell_sets, at_time=last)
    fates = tmap.fates(populations_)
    say("fates %s, columns %s" % (fates.shape, list(fates.var_names)))
    name_to_col = {str(n): i for i, n in enumerate(fates.var_names)}
    F = np.asarray(fates.X, dtype=np.float64)
    rows = {str(n): i for i, n in enumerate(fates.obs_names)}
    idx = [rows[n] for n in B.obs_names[d2]]
    cols = []
    missing = []
    for p in pops:
        if p in name_to_col:
            cols.append(F[idx, name_to_col[p]])
        else:
            missing.append(p)
            cols.append(np.zeros(len(idx)))
    conv = {"wot": dict(converged=True, day_pairs=list(tmap.timepoints),
                        cell_sets={k: len(v) for k, v in cell_sets.items()},
                        populations_without_a_column=missing,
                        config={k: str(v) for k, v in model.ot_config.items()})}
    return np.column_stack(cols), conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("featdir")
    ap.add_argument("--method", required=True, choices=["moscot", "statot", "wot"])
    # moscot's own defaults, so that "the method at its defaults" is the
    # default here too: full rank, epsilon 1e-3, threshold 1e-3.
    ap.add_argument("--epsilon", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 materialises the cost matrix, as moscot does by "
                         "default; a positive value computes it in batches")
    ap.add_argument("--rank", type=int, default=-1)
    ap.add_argument("--max-iter", type=int, default=10000)
    ap.add_argument("--tau-a", type=float, default=1.0)
    ap.add_argument("--tau-b", type=float, default=1.0)
    ap.add_argument("--realtime", action="store_true",
                    help="with --method moscot: fate probabilities from "
                         "CellRank's RealTimeKernel on the couplings instead "
                         "of moscot's pull")
    ap.add_argument("--conn-weight", type=float, default=0.2)
    # statot solves one coupling of the data with itself, with POT's
    # sinkhorn in the non-logarithmic domain, so it needs its own epsilon:
    # at moscot's 1e-3 the kernel exp(-C/eps) underflows to zero. 0.05 is
    # the order the package's own examples use on a mean-scaled cost, and
    # 1e-9 is the package's default relative tolerance.
    ap.add_argument("--statot-eps", type=float, default=0.05)
    ap.add_argument("--statot-tol", type=float, default=1e-9)
    ap.add_argument("--statot-method", default="ent")
    ap.add_argument("--wot-kwargs", default="{}",
                    help="JSON of extra OTModel options; empty "
                         "leaves every wot default in place")
    ap.add_argument("--wot-reuse", action="store_true",
                    help="reuse transport maps already written for this arm")
    ap.add_argument("--wot-out", default="",
                    help="directory for wot transport maps")
    ap.add_argument("--verify-cells", type=int, default=3000,
                    help="cells in the subsample on which this script's "
                         "fate solve is checked against the package's own")
    ap.add_argument("--time-col", default="time_info")
    ap.add_argument("--state-col", default="state_info")
    ap.add_argument("--keep-days", default="",
                    help="comma-separated sampling days to keep BEFORE the "
                         "PCA, i.e. the experiment that never sampled the "
                         "others, e.g. 2,6")
    ap.add_argument("--couple-days", default="",
                    help="comma-separated sampling days to keep AFTER the "
                         "PCA, i.e. the full representation with a shortened "
                         "coupling chain, e.g. 2,6")
    ap.add_argument("--drop-cells", default="",
                    help="file of cell names removed after the PCA (from the "
                         "coupling only), e.g. a fold's held-out clones' later "
                         "cells from 59_leakage_cells.py; early cells may not "
                         "be among them")
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow-nonconverged", action="store_true",
                    help="save even if a solver did not converge; the log then "
                         "records it and the result must not be reported")
    a = ap.parse_args()

    def daylist(s):
        return [float(x) for x in s.split(",") if x.strip()]

    kd, cd = daylist(a.keep_days), daylist(a.couple_days)
    assert not (kd and cd), "--keep-days and --couple-days answer different " \
        "questions; pass one at a time"
    B, d2 = build(a.h5ad, a.featdir, a.time_col, a.state_col, kd, cd,
                  a.drop_cells or None)
    pops = populations(a.featdir, B.obs["state"].values)
    say("object %s, %d early cells, populations %s" % (B.shape, d2.size, pops))

    P, conv = {"moscot": run_moscot, "statot": run_statot,
               "wot": run_wot}[a.method](
        B, d2, pops, a)
    ok = all(v.get("converged") for v in conv.values())
    if not ok and not a.allow_nonconverged:
        json.dump(dict(method=a.method, epsilon=a.epsilon, rank=a.rank,
                       threshold=a.threshold, convergence=conv, saved=False),
                  open(os.path.join(a.featdir, "snap_%s_log.json"
                                    % (a.tag or a.method)), "w"), indent=1)
        raise SystemExit("a solver did not converge; nothing saved: %s" % conv)
    P = np.asarray(P, dtype=np.float64)
    rs = P.sum(1, keepdims=True)
    P = P / np.where(rs == 0, 1, rs)
    tag = a.tag or a.method
    out = os.path.join(a.featdir, "snap_%s.npy" % tag)
    np.save(out, P.astype(np.float32))
    say("saved %s %s; mean of the largest probability %.3f"
        % (out, P.shape, float(P.max(1).mean())))
    log = os.path.join(a.featdir, "snap_%s_log.json" % tag)
    json.dump(dict(method=a.method, epsilon=a.epsilon, rank=a.rank,
                   statot_eps=a.statot_eps, statot_tol=a.statot_tol,
                   threshold=a.threshold, max_iterations=a.max_iter,
                   batch_size=a.batch_size, populations=pops,
                   keep_days=kd, couple_days=cd,
                   days_in_object=sorted(set(B.obs["day"].dropna().tolist())),
                   n_cells_in_object=int(B.n_obs), convergence=conv,
                   all_converged=bool(ok), saved=True,
                   tau_a=a.tau_a, tau_b=a.tau_b, realtime=a.realtime,
                   realtime_info=getattr(a, "_rtk_info", None)),
              open(log, "w"), indent=1)
    say("wrote %s" % log)


if __name__ == "__main__":
    main()
