"""Does the inferred velocity point along directions that are known to be right?

Usage: 21_time_axis.py <h5ad> <featdir> <labels.tsv> <out.csv> [--pos P --neg N]

The fate test says velocity points the wrong way along the axis that separates
neutrophil-fated from monocyte-fated clones. On its own that is ambiguous: a
field that is simply wrong would look the same. This script scores the same
velocity vectors against two directions whose sign is not in doubt, using the
same clone-disjoint folds and the same projection.

  maturation   undifferentiated day-2 cells against mature day-2 cells. Both
               sides are day-2 cells, so no later time point enters anywhere,
               and progenitors in a differentiating culture must on average be
               moving toward the mature side.
  sampling day day-2 cells against day-4 and day-6 cells. The later cells enter
               only through the direction that is fitted; the velocities being
               scored are the ones computed on the day-2 graph.

For reference the fate direction is scored again in the same run. Each velocity
is scored as inferred and after the chain-rule transfer to the log scale the
directions are fitted on. The complement of each score is reported as well,
since a value below one half means the field is anti-aligned rather than
uninformative.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from cvlib import folds_of, cp10k_log

ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--pcs", type=int, default=50)
ap.add_argument("--pos", default="Neutrophil")
ap.add_argument("--neg", default="Monocyte")
ap.add_argument("--late-sample", type=int, default=6000)
ap.add_argument("--time-col", default="time_info")
ap.add_argument("--early", default="2,2.0",
                help="values of the time column that are the early cells")
ap.add_argument("--late", default="",
                help="values that are the late cells; empty means all others")
ap.add_argument("--state-col", default="state_info")
ap.add_argument("--mature", default="Neutrophil,Monocyte,Baso,Mast,Meg,"
                                    "Erythroid,Lymphoid,Eos,Ccr7_DC,pDC")
ap.add_argument("--controls", action="store_true",
                help="also score two fields with the steady-state structure "
                     "and no unspliced information (Mu shuffled across cells "
                     "minus gamma*Ms; and -gamma*Ms alone, gamma recovered per "
                     "gene from vel_det), and the fate direction "
                     "orthogonalised to the maturation direction")
ap.add_argument("--restrict-pop", default="",
                help="keep only cells whose pop column equals this")
CFG = ap.parse_args()


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


T = pd.read_csv(CFG.labels, sep="\t")
d2_names = json.load(open(os.path.join(CFG.featdir, "d2_names.json")))
assert list(T["cell"]) == d2_names
hvg = json.load(open(os.path.join(CFG.featdir, "hvg.json")))


def load(n):
    p = os.path.join(CFG.featdir, n + ".npy")
    return np.load(p) if os.path.exists(p) else None


Ms = load("Ms")
VEL = {k: load("vel_" + k) for k in ("det", "dyn", "velovi")}
VEL = {k: v for k, v in VEL.items() if v is not None}
X_d2 = np.log1p(np.clip(Ms, 0, None))

A = ad.read_h5ad(CFG.h5ad)
tcol = A.obs[CFG.time_col].astype(str)
EARLY = [v for v in CFG.early.split(",") if v]
is_d2_all = tcol.isin(EARLY).values
state = A.obs[CFG.state_col].astype(str).values
MATURE = [m for m in CFG.mature.split(",") if m]

if CFG.late:
    late_idx = np.where(tcol.isin(CFG.late.split(",")).values)[0]
else:
    late_idx = np.where(~is_d2_all)[0]
rng = np.random.default_rng(3)
if len(late_idx) > CFG.late_sample:
    late_idx = rng.choice(late_idx, CFG.late_sample, replace=False)
sub = A[late_idx, hvg]
L = sub.layers["spliced"]
X_late = cp10k_log(L.toarray() if sp.issparse(L) else np.asarray(L))
say("late cells for the sampling-day direction: %d" % X_late.shape[0])

d2_state = state[is_d2_all]           # aligned with the saved day-2 arrays
d2_mature = np.isin(d2_state, MATURE)
mature_idx = np.where(d2_mature)[0]
undiff_idx = np.where(~d2_mature)[0]
say("day-2 cells annotated mature: %d of %d"
    % (len(mature_idx), len(d2_mature)))

sel = (T["n_sis"] >= CFG.min_sis) & T["dom"].isin([CFG.pos, CFG.neg])
if CFG.restrict_pop:
    sel = sel & (T["pop"] == CFG.restrict_pop)
idx = np.where(sel.values)[0]
y_fate = (T.loc[sel, "dom"].values == CFG.pos).astype(int)
groups = T.loc[sel, "clone"].values
say("task cells %d, clones %d" % (len(idx), len(np.unique(groups))))

if CFG.controls:
    # The steady-state velocity is v = u - gamma * s on the smoothed moments,
    # so along any axis its component is pulled against the cell's own
    # position by the -gamma*s term, whatever u says. Two fields keep that
    # term and drop the information in u: u taken from another day-2 cell
    # (a fixed random permutation), and -gamma*s alone. gamma is recovered
    # per gene from vel_det by least squares through the origin of
    # (Mu - vel_det) on Ms; genes scVelo left without a velocity get 0.
    Mu = load("Mu")
    vd = np.nan_to_num(VEL["det"])
    num = ((Mu - vd) * Ms).sum(0)
    den_g = (Ms * Ms).sum(0)
    gamma = np.where(den_g > 0, num / np.where(den_g > 0, den_g, 1), 0.0)
    gamma[np.abs(vd).sum(0) == 0] = 0.0
    fit = Mu - gamma * Ms
    act = np.abs(vd).sum(0) > 0
    resid = float(np.abs(fit[:, act] - vd[:, act]).max())
    say("gamma recovered for %d genes; max |(Mu - gamma Ms) - vel_det| %.3g"
        % (int(act.sum()), resid))
    perm = np.random.default_rng(21).permutation(Ms.shape[0])
    VEL["det_shuffled_u"] = np.where(act, Mu[perm] - gamma * Ms, 0.0)
    VEL["minus_gamma_s"] = np.where(act, -gamma * Ms, 0.0)
V = {k: v[idx] for k, v in VEL.items()}
den = 1.0 + np.clip(Ms[idx], 0, None)
for k in list(VEL):
    V[k + "_logscale"] = VEL[k][idx] / den
Xsel = X_d2[idx]


def axis_from(Xa, Xb):
    """Direction separating two groups of cells, fitted in the shared space."""
    X = np.vstack([Xa, Xb])
    lab = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    k = min(CFG.pcs, len(X) - 1, X.shape[1])
    pca = PCA(n_components=k, random_state=0).fit(X)
    Z = pca.transform(X)
    sc = StandardScaler().fit(Z)
    clf = LogisticRegression(max_iter=4000, C=1.0).fit(sc.transform(Z), lab)
    w = pca.components_.T @ (clf.coef_.ravel() / sc.scale_)
    return w / np.linalg.norm(w)


rows = []
for rep in range(CFG.repeats):
    for tr, te in folds_of(groups, rep, CFG.folds):
        # the three directions, each fitted without the held-out cells
        axes = {}
        axes["sampling day"] = axis_from(Xsel[tr], X_late)
        held = set(idx[te].tolist())
        und_keep = np.array([i for i in undiff_idx if i not in held])
        mat_keep = np.array([i for i in mature_idx if i not in held])
        axes["maturation"] = axis_from(X_d2[und_keep], X_d2[mat_keep])
        # the fate direction, fitted on training cells only
        Xa = Xsel[tr][y_fate[tr] == 0]
        Xb = Xsel[tr][y_fate[tr] == 1]
        axes["clonal fate"] = axis_from(Xa, Xb)

        truth = {"sampling day": np.ones(len(te)),
                 "maturation": np.ones(len(te)),
                 "clonal fate": np.where(y_fate[te] == 1, 1.0, -1.0)}
        if CFG.controls:
            wf, wm = axes["clonal fate"], axes["maturation"]
            wo = wf - (wf @ wm) * wm
            axes["clonal fate, orthogonal to maturation"] = wo / np.linalg.norm(wo)
            truth["clonal fate, orthogonal to maturation"] = truth["clonal fate"]
            rows.append(dict(axis="cosine fate-maturation", vector="axes",
                             rep=rep, direction_correct=float(wf @ wm),
                             mean_cosine=float(wf @ wm)))
        for aname, w in axes.items():
            for vname, M in V.items():
                d = M[te]
                nrm = np.linalg.norm(d, axis=1)
                proj = (d @ w) / np.where(nrm == 0, np.nan, nrm)
                ok = np.isfinite(proj)
                corr = float((np.sign(proj[ok]) == truth[aname][ok]).mean())
                rows.append(dict(axis=aname, vector=vname, rep=rep,
                                 direction_correct=corr,
                                 mean_cosine=float(np.nanmean(proj))))
            # the cell's own displacement from the training mean, as a check
            d = Xsel[te] - Xsel[tr].mean(0)
            nrm = np.linalg.norm(d, axis=1)
            proj = (d @ w) / np.where(nrm == 0, np.nan, nrm)
            ok = np.isfinite(proj)
            rows.append(dict(axis=aname, vector="state_itself", rep=rep,
                             direction_correct=float(
                                 (np.sign(proj[ok]) == truth[aname][ok]).mean()),
                             mean_cosine=float(np.nanmean(proj))))

R = pd.DataFrame(rows)
S = (R.groupby(["axis", "vector"])
     .agg(direction_correct=("direction_correct", "mean"),
          sd=("direction_correct", "std"),
          complement=("direction_correct", lambda v: 1 - np.mean(v)),
          mean_cosine=("mean_cosine", "mean"))
     .reset_index())
S.to_csv(CFG.out, index=False)
print(S.to_string(index=False))
say("done")
