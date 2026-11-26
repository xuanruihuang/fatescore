"""Parameter uncertainty of the two-level label model, by parametric bootstrap.

57_label_ceiling.py reports the ceilings of a day-2 predictor with the
model's parameters fixed at their maximum-likelihood values, and
60_yardstick_sim.py the factor by which the synthetic-feature calibration
overstates an independent day-2 signal, also at those values. Neither interval
includes the uncertainty of the fit. Here each bootstrap replicate simulates
the per-well neutrophil and monocyte counts of the 567 main-task clones from
the fitted model (well sizes fixed at the observed ones), refits (a, b, nu) by
maximum likelihood exactly as 57 does, and recomputes, at the refitted values:
  - the two ceilings (AUROC of the clone propensity, and of a cell's own
    propensity drawn around it, for the pooled dominant fate), averaged over
    100 simulated label sets;
  - the ratio of increments of a feature built from the observed share and one
    built from the clone propensity at equal stand-alone AUROC (0.80 and 0.86),
    as in 60_yardstick_sim.py with the clone table repeated 20 times.
The 2.5-97.5% range over replicates is the parameter part of the uncertainty.

Usage: 72_label_param_uncertainty.py <d2_labels.tsv> <out_prefix> [--boot 200]
         [--workers 20]
"""
import argparse
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import beta as beta_dist, betabinom  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--boot", type=int, default=200)
ap.add_argument("--workers", type=int, default=20)
a = ap.parse_args()

T = pd.read_csv(a.labels, sep="\t")
M = T[(T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])]
C = M.groupby("clone").agg(n_early=("cell", "size"),
                           nw1=("n_neu_w1", "first"), mw1=("n_mono_w1", "first"),
                           nw2=("n_neu_w2", "first"), mw2=("n_mono_w2", "first"))
NE = C["n_early"].to_numpy(float)
W = {1: (C["nw1"] + C["mw1"]).to_numpy(int), 2: (C["nw2"] + C["mw2"]).to_numpy(int)}
K = {1: C["nw1"].to_numpy(int), 2: C["nw2"].to_numpy(int)}
Q = 2000
U = (np.arange(Q) + 0.5) / Q


def fit(k):
    def nll(t):
        A, B, nu = np.exp(t)
        pc = np.clip(beta_dist.ppf(U, A, B), 1e-7, 1 - 1e-7)[None, :]
        ll = np.zeros((len(NE), Q))
        for w in (1, 2):
            ll += betabinom.logpmf(k[w][:, None], W[w][:, None], nu * pc, nu * (1 - pc))
        m = ll.max(1, keepdims=True)
        return -float((m[:, 0] + np.log(np.exp(ll - m).mean(1))).sum())
    best = None
    for nu0 in (1.0, 3.0, 10.0, 30.0):
        r = minimize(nll, x0=np.log([0.3, 0.3, nu0]), method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-6, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x)


def draw_wells(rng, A, B, NU, w1, w2):
    pc = rng.beta(A, B, size=w1.size)
    k = {}
    for w, n in ((1, w1), (2, w2)):
        pw = rng.beta(np.maximum(NU * pc, 1e-9), np.maximum(NU * (1 - pc), 1e-9))
        k[w] = rng.binomial(n, pw)
    return pc, k


def ceilings(rng, A, B, NU, sims=100):
    c1, c2 = [], []
    for _ in range(sims):
        pc, k = draw_wells(rng, A, B, NU, W[1], W[2])
        tot = W[1] + W[2]
        y = (2 * (k[1] + k[2]) >= tot).astype(int)
        if y.min() == y.max():
            continue
        c1.append(roc_auc_score(y, pc))
        ps = rng.beta(np.maximum(NU * pc, 1e-9), np.maximum(NU * (1 - pc), 1e-9))
        c2.append(roc_auc_score(y, ps))
    return float(np.mean(c1)), float(np.mean(c2))


def ratio(rng, A, B, NU, rep=20, targets=(0.80, 0.86), expr=0.87):
    ne = np.tile(NE, rep)
    w1, w2 = np.tile(W[1], rep), np.tile(W[2], rep)
    n = ne.size
    pc, k = draw_wells(rng, A, B, NU, w1, w2)
    f = (k[1] + k[2]) / (w1 + w2)
    y = (2 * (k[1] + k[2]) >= (w1 + w2)).astype(int)
    lp = np.log(np.clip(pc, 1e-6, 1 - 1e-6) / np.clip(1 - pc, 1e-6, 1))

    def at(make, target):
        lo, hi = 1e-3, 1e3
        z = rng.standard_normal(n) / np.sqrt(ne)
        for _ in range(40):
            mid = np.sqrt(lo * hi)
            if roc_auc_score(y, make(mid, z)) > target:
                lo = mid
            else:
                hi = mid
        return make(np.sqrt(lo * hi), z)

    E = at(lambda s, z: lp + s * z, expr)
    aE = roc_auc_score(y, E)
    out = {}
    for t in targets:
        inc = {}
        for name, sig in (("f", (f - f.mean()) / f.std()),
                          ("pc", (pc - pc.mean()) / pc.std())):
            S = at(lambda s, z: sig + s * z, t)
            X = np.column_stack([E, S])
            m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
            inc[name] = roc_auc_score(y, m.decision_function(X)) - aE
        out["ratio_%.2f" % t] = inc["f"] / inc["pc"]
        out["inc_pc_%.2f" % t] = inc["pc"]
    return out


A0, B0, NU0 = fit(K)
print("point fit a=%.4f b=%.4f nu=%.4f" % (A0, B0, NU0), flush=True)


def one(b):
    rng = np.random.default_rng(72000 + b)
    _, k = draw_wells(rng, A0, B0, NU0, W[1], W[2])
    A, B, NU = fit(k)
    c1, c2 = ceilings(rng, A, B, NU)
    rec = dict(boot=b, a=A, b=B, nu=NU, ceiling_clone=c1, ceiling_cell=c2)
    rec.update(ratio(rng, A, B, NU))
    return rec


if __name__ == "__main__":
    with Pool(a.workers) as p:
        rows = p.map(one, range(a.boot))
    R = pd.DataFrame(rows)
    R.to_csv(a.out + "_boot.csv", index=False)
    rng = np.random.default_rng(7200)
    pt = dict(a=A0, b=B0, nu=NU0)
    pt["ceiling_clone"], pt["ceiling_cell"] = ceilings(rng, A0, B0, NU0, sims=500)
    pt.update(ratio(rng, A0, B0, NU0, rep=60))
    S = pd.DataFrame({"point": pd.Series(pt),
                      "boot_lo": R.drop(columns="boot").quantile(0.025),
                      "boot_median": R.drop(columns="boot").median(),
                      "boot_hi": R.drop(columns="boot").quantile(0.975)}).round(4)
    S.to_csv(a.out + "_summary.csv")
    print(S.to_string())
