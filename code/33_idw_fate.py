"""The inverse-distance baseline that one paper tuned on its own test cells.

MultistageOT's comparison includes a deliberately naive model: fate
probabilities from a cell's distance to each terminal population's centroid in
principal-component space, with prior weights from the annotation counts and one
exponent. Its exponent was chosen by maximising accuracy on the same cells the
comparison is scored on, which can only shrink the gap between the model and
that baseline.

Here the same baseline is built as a feature block over a grid of exponents, so
it can be scored the way every other arm in this evaluation is scored: the same
clone-disjoint folds, the classifier chosen inside the training folds, the
clone as the unit. Two readings, declared before the run: the exponent that
paper reports choosing, and the best of the grid, which is selected on the
scored metric and is therefore an optimistic bound rather than an estimate.

Nothing here uses a fate label; the terminal populations come from the same
annotation the velocity kernels' terminal states come from.

Usage:
  python 33_idw_fate.py <h5ad> <featdir> [--gammas 1,5,10,20,50] [--pcs 50]
"""
import argparse
import json
import os
import time

import numpy as np
import anndata as ad
import scanpy as sc
import scvelo as scv

MATURE = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid",
          "Lymphoid", "Eos", "Ccr7_DC", "pDC"]


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("featdir")
    ap.add_argument("--gammas", default="1,5,10,20,50")
    ap.add_argument("--pcs", type=int, default=50)
    ap.add_argument("--state-col", default="state_info")
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
    B = ad.read_h5ad(a.h5ad)
    if a.d2_only:
        B = B[json.load(open(os.path.join(a.featdir, "d2_names.json")))].copy()
        print("day-2 cells only:", B.shape, flush=True)
    state = B.obs[a.state_col].astype(str).values
    if not a.no_gene_filter:
        scv.pp.filter_genes(B, min_shared_counts=20)
        scv.pp.normalize_per_cell(B)
    sc.pp.log1p(B)
    keep = [g for g in hvg if g in B.var_names]
    assert len(keep) == len(hvg), (len(keep), len(hvg))
    B = B[:, keep].copy()
    sc.tl.pca(B, n_comps=a.pcs, svd_solver="arpack")
    X = np.asarray(B.obsm["X_pca"][:, :a.pcs], dtype=np.float64)
    d2 = np.array([B.obs_names.get_loc(n) for n in d2_names])

    present = [s for s in MATURE if (state == s).sum() >= 30]
    cent = np.stack([X[state == s].mean(0) for s in present])
    prior = np.array([(state == s).sum() for s in present], dtype=np.float64)
    prior = prior / prior.sum()
    say("terminal populations %s" % present)
    say("prior weights %s" % np.round(prior, 4).tolist())

    D = np.linalg.norm(X[d2][:, None, :] - cent[None, :, :], axis=2)
    D = np.maximum(D, 1e-12)
    log = {"populations": present, "prior": prior.tolist(),
           "n_pcs": a.pcs, "n_cells_in_object": int(B.n_obs), "gammas": []}
    for g in [float(x) for x in a.gammas.split(",")]:
        W = prior[None, :] * D ** (-g)
        P = W / W.sum(1, keepdims=True)
        tag = "snap_idw_g%g" % g
        np.save(os.path.join(a.featdir, "%s.npy" % tag),
                np.asarray(P, dtype=np.float32))
        log["gammas"].append(dict(gamma=g, tag=tag,
                                  max_prob_mean=float(P.max(1).mean())))
        say("%s: mean of the largest probability %.3f" % (tag, P.max(1).mean()))

    json.dump(log, open(os.path.join(a.featdir, "idw_log.json"), "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()
