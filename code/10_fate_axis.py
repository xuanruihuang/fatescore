"""Does the inferred velocity point toward the clone's observed fate?

Usage: 10_fate_axis.py <featdir> <labels.tsv> <outprefix>

Velocity benchmarks measure direction correctness against boundaries drawn
between annotated clusters. Here the boundary is the observed clonal fate. For
each clone-disjoint fold we learn, from the training cells' day-2 expression
alone, the direction in gene space that separates neutrophil-fated from
monocyte-fated clones, and then ask, for each held-out cell, whether its
inferred velocity vector has a positive component along that direction when
its clone became neutrophil and a negative one when it became monocyte.

Chance is one half. The same projection applied to the cell's own expression
state is the positive control: the axis was learned to separate exactly that,
so it must score high.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cvlib import folds_of, cp10k_log

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("outprefix")
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--pcs", type=int, default=50)
ap.add_argument("--perms", type=int, default=20)
ap.add_argument("--pos", default="Neutrophil")
ap.add_argument("--neg", default="Monocyte")
CFG = ap.parse_args()


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


T = pd.read_csv(CFG.labels, sep="\t")
assert list(T["cell"]) == json.load(
    open(os.path.join(CFG.featdir, "d2_names.json")))


def load(n):
    p = os.path.join(CFG.featdir, n + ".npy")
    return np.load(p) if os.path.exists(p) else None


Ms, Mu = load("Ms"), load("Mu")
raw_s = load("raw_spliced")
VEL = {k: load("vel_" + k) for k in ("det", "dyn", "velovi")}
VEL = {k: v for k, v in VEL.items() if v is not None}

sel = (T["n_sis"] >= CFG.min_sis) & T["dom"].isin([CFG.pos, CFG.neg])
idx = np.where(sel.values)[0]
y = (T.loc[sel, "dom"].values == CFG.pos).astype(int)
sgn = np.where(y == 1, 1.0, -1.0)
groups = T.loc[sel, "clone"].values
f_neu = T.loc[sel, "f_neu"].values.astype(float)
state_d2 = T.loc[sel, "state_d2"].astype(str).values
say("task A: %d cells, %d clones" % (len(idx), len(np.unique(groups))))

X_ms = np.log1p(np.clip(Ms[idx], 0, None))
X_s = cp10k_log(raw_s[idx])
V = {k: v[idx] for k, v in VEL.items()}
# the direction from the cell's current state toward the differentiated
# compartment, as the splicing model would have it
V["ratio"] = (Mu / (Ms + Mu + 1e-8))[idx]
# The separating direction is fitted on log-normalised abundance, while a
# velocity is a rate of change of untransformed abundance. The chain rule
# turns one into the other, so each velocity is also carried over as
# d/dt log(1 + s) = v / (1 + s); if the projection changed sign under this
# transformation the result would be an artefact of the scale.
_denom = 1.0 + np.clip(Ms[idx], 0, None)
for k in list(VEL):
    V[k + "_logscale"] = VEL[k][idx] / _denom

rng = np.random.default_rng(0)
PERM = []
for s in range(CFG.perms):
    r = np.random.default_rng(700 + s)
    p = np.arange(len(idx))
    for st in np.unique(state_d2):
        m = np.where(state_d2 == st)[0]
        p[m] = r.permutation(m)
    PERM.append(p)

rows = []
for basis_name, Xb in (("smoothed", X_ms), ("spliced", X_s)):
    for rep in range(CFG.repeats):
        proj = {k: np.full(len(y), np.nan) for k in V}
        proj["state_itself"] = np.full(len(y), np.nan)
        permproj = {s: np.full(len(y), np.nan) for s in range(CFG.perms)}
        for tr, te in folds_of(groups, rep, CFG.folds):
            pca = PCA(n_components=min(CFG.pcs, len(tr) - 1),
                      random_state=0).fit(Xb[tr])
            Ztr = pca.transform(Xb[tr])
            sc = StandardScaler().fit(Ztr)
            clf = LogisticRegression(max_iter=4000, C=1.0).fit(
                sc.transform(Ztr), y[tr])
            # the separating direction, carried back into gene space
            w_pc = clf.coef_.ravel() / sc.scale_
            w = pca.components_.T @ w_pc
            w = w / np.linalg.norm(w)
            for k, M in V.items():
                d = M[te]
                n = np.linalg.norm(d, axis=1)
                proj[k][te] = (d @ w) / np.where(n == 0, np.nan, n)
            d = Xb[te] - Xb[tr].mean(0)
            n = np.linalg.norm(d, axis=1)
            proj["state_itself"][te] = (d @ w) / np.where(n == 0, np.nan, n)
            for s, p in enumerate(PERM):
                d = V["dyn"][p][te] if "dyn" in V else V[list(V)[0]][p][te]
                n = np.linalg.norm(d, axis=1)
                permproj[s][te] = (d @ w) / np.where(n == 0, np.nan, n)

        def score(name, s, ref=None):
            ok = np.isfinite(s)
            cell_correct = float((np.sign(s[ok]) == sgn[ok]).mean())
            d = (pd.DataFrame(dict(s=s[ok], y=y[ok], g=groups[ok],
                                   f=f_neu[ok]))
                 .groupby("g").agg(s=("s", "mean"), y=("y", "first"),
                                   f=("f", "first")))
            rs = np.nan
            if ref is not None:
                m2 = ok & np.isfinite(ref)
                if m2.sum() > 10:
                    rs = float(np.corrcoef(s[m2], ref[m2])[0, 1])
            return dict(basis=basis_name, rep=rep, what=name,
                        corr_with_state_projection=rs,
                        cell_direction_correct=cell_correct,
                        clone_direction_correct=float(
                            (np.sign(d["s"]) == np.where(d["y"] == 1, 1, -1))
                            .mean()),
                        clone_auroc=float(roc_auc_score(d["y"], d["s"]))
                        if d["y"].nunique() > 1 else np.nan,
                        clone_spearman_fneu=float(
                            spearmanr(d["f"], d["s"]).statistic),
                        n_cells=int(ok.sum()), n_clones=len(d))

        for k in proj:
            rows.append(score(k, proj[k],
                              None if k == "state_itself"
                              else proj["state_itself"]))
        pr = [score("dyn_permuted_%d" % s, permproj[s])
              for s in range(CFG.perms)]
        for r in pr:
            rows.append(r)

R = pd.DataFrame(rows)
R.to_csv(CFG.outprefix + "_perrepeat.csv", index=False)
R["family"] = np.where(R["what"].str.startswith("dyn_permuted"),
                       "dyn_permuted", R["what"])
S = (R.groupby(["basis", "family"])
     .agg(cell_direction_correct=("cell_direction_correct", "mean"),
          corr_with_state_projection=("corr_with_state_projection", "mean"),
          cell_sd=("cell_direction_correct", "std"),
          clone_direction_correct=("clone_direction_correct", "mean"),
          clone_auroc=("clone_auroc", "mean"),
          clone_auroc_sd=("clone_auroc", "std"),
          clone_spearman_fneu=("clone_spearman_fneu", "mean"),
          n_clones=("n_clones", "first"))
     .reset_index())
S.to_csv(CFG.outprefix + "_summary.csv", index=False)
print(S.to_string(index=False))
say("done")
