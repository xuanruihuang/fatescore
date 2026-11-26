"""How much room the label leaves above the day-2 transcriptome.

A reviewer's objection: the day-2 transcriptome already ties with the clone's
descendants in the other well (22_matched_ceiling.py), so perhaps nothing could
add to it and the increment design has no room to detect anything. The tie does
not settle this, because the other well is itself a noisy read-out: each well
holds only a few mature sisters.

This asks what AUROC a predictor that knew each clone's fate propensity exactly
would reach, under the generative model of 27_power_budget.py: each clone has a
propensity p ~ Beta(a, b), its mature Neutrophil/Monocyte sisters are Binomial
draws given p, and the label is the majority call (ties to Neutrophil, the
fixed order). The model is first checked against two things it was not fitted
to: the agreement of the two wells' calls, and the AUROC of one well's
Neutrophil fraction predicting the other well's call, on the clones that both
wells reach (the 212 clones of 22_matched_ceiling.py). Only if it reproduces
those is its ceiling worth reporting.

Two fits of the one-level model: the moment fit of 27_power_budget.py, and the
maximum-likelihood beta-binomial fit, on the 567 clones of the main task. The
one-level model fails the check (it predicts far more agreement between the
wells than is observed), so a two-level model is fitted as well: a clone
propensity p_c ~ Beta(a, b), shared by everything that descends from the clone
as it was at the split, and a well propensity p_cw ~ Beta(nu p_c, nu (1 - p_c))
around it, with the sisters in each well Binomial given p_cw. It is fitted by
maximum likelihood on the per-well counts of the 567 clones. A day-2 cell
precedes the split, so what it can know about the label is at most p_c: the
AUROC of p_c for the pooled majority call is the ceiling for any day-2
predictor under this model.

Usage: 57_label_ceiling.py <labels.tsv> <out.csv> [--sims 2000]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import betabinom
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--sims", type=int, default=2000)
ap.add_argument("--pos", default="Neutrophil")
ap.add_argument("--neg", default="Monocyte")
ap.add_argument("--min-sis", type=int, default=3)
a = ap.parse_args()

T = pd.read_csv(a.labels, sep="\t")
C = T.groupby("clone").first()

main = C[(C["n_sis"] >= a.min_sis) & C["dom"].isin([a.pos, a.neg])]
k = main["n_neu"].to_numpy(int)
n = (main["n_neu"] + main["n_mono"]).to_numpy(int)
y_main = (main["dom"] == a.pos).to_numpy(int)
print("main task: %d clones" % len(main))


def fit_moment(k, n):
    f = k / n
    mu, var, m1 = f.mean(), f.var(ddof=1), (1.0 / n).mean()
    rho = (var / (mu * (1 - mu)) - m1) / (1 - m1)
    s = 1 / rho - 1
    return mu * s, (1 - mu) * s


def fit_mle(k, n):
    nll = lambda t: -betabinom.logpmf(k, n, np.exp(t[0]), np.exp(t[1])).sum()
    r = minimize(nll, x0=np.log(fit_moment(k, n)), method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    return tuple(np.exp(r.x))


def majority(k, n):
    return ((2 * k > n) | (2 * k == n)).astype(int)


both = C[C["dom_w1"].isin([a.pos, a.neg]) & C["dom_w2"].isin([a.pos, a.neg])
         & ((C["n_neu_w1"] + C["n_mono_w1"]) >= 1)
         & ((C["n_neu_w2"] + C["n_mono_w2"]) >= 1)]
# the 212 clones of 22_matched_ceiling.py also need a day-2 cell; every clone
# in this table has one, since the table is indexed by day-2 cells
nw = {w: (both["n_neu_w%d" % w] + both["n_mono_w%d" % w]).to_numpy(int)
      for w in (1, 2)}
kw = {w: both["n_neu_w%d" % w].to_numpy(int) for w in (1, 2)}
yw = {w: (both["dom_w%d" % w] == a.pos).to_numpy(int) for w in (1, 2)}
obs = dict(
    n_pair=len(both),
    agree=float((yw[1] == yw[2]).mean()),
    oracle_1to2=roc_auc_score(yw[2], kw[1] / nw[1]),
    oracle_2to1=roc_auc_score(yw[1], kw[2] / nw[2]),
    median_sisters_per_well=float(np.median(np.r_[nw[1], nw[2]])),
    median_sisters_main=float(np.median(n)))
print("observed:", obs)

rows = []
for fit, (A, B) in (("moment", fit_moment(k, n)), ("mle", fit_mle(k, n))):
    rng = np.random.default_rng(57)
    sim = {q: [] for q in ("agree", "oracle_1to2", "oracle_2to1",
                           "ceiling_target_well", "ceiling_main")}
    for _ in range(a.sims):
        p = rng.beta(A, B, size=len(both))
        s = {w: rng.binomial(nw[w], p) for w in (1, 2)}
        l = {w: majority(s[w], nw[w]) for w in (1, 2)}
        if min(l[1].sum(), l[2].sum()) == 0 or max(l[1].mean(), l[2].mean()) == 1:
            continue
        sim["agree"].append((l[1] == l[2]).mean())
        sim["oracle_1to2"].append(roc_auc_score(l[2], s[1] / nw[1]))
        sim["oracle_2to1"].append(roc_auc_score(l[1], s[2] / nw[2]))
        sim["ceiling_target_well"].append(
            0.5 * (roc_auc_score(l[2], p) + roc_auc_score(l[1], p)))
        pm = rng.beta(A, B, size=len(n))
        lm = majority(rng.binomial(n, pm), n)
        sim["ceiling_main"].append(roc_auc_score(lm, pm))
    for q, v in sim.items():
        v = np.asarray(v)
        rows.append(dict(fit=fit, beta_a=round(A, 4), beta_b=round(B, 4),
                         quantity=q, observed=obs.get(q, np.nan),
                         sim_mean=round(float(v.mean()), 4),
                         sim_lo=round(float(np.percentile(v, 2.5)), 4),
                         sim_hi=round(float(np.percentile(v, 97.5)), 4),
                         n_sims=len(v)))
# ---------------------------------------------------------------- two levels
from scipy.stats import beta as beta_dist

kw_m = {w: main["n_neu_w%d" % w].to_numpy(int) for w in (1, 2)}
nw_m = {w: (main["n_neu_w%d" % w] + main["n_mono_w%d" % w]).to_numpy(int)
        for w in (1, 2)}
Q = 2000
U = (np.arange(Q) + 0.5) / Q


def nll2(t):
    A, B, nu = np.exp(t)
    pc = np.clip(beta_dist.ppf(U, A, B), 1e-7, 1 - 1e-7)[None, :]
    ll = np.zeros((len(main), Q))
    for w in (1, 2):
        ll += betabinom.logpmf(kw_m[w][:, None], nw_m[w][:, None],
                               nu * pc, nu * (1 - pc))
    m = ll.max(1, keepdims=True)
    return -float((m[:, 0] + np.log(np.exp(ll - m).mean(1))).sum())


best = None
for nu0 in (1.0, 3.0, 10.0, 30.0):
    r = minimize(nll2, x0=np.log([0.3, 0.3, nu0]), method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-6, maxiter=4000))
    if best is None or r.fun < best.fun:
        best = r
A2, B2, NU = np.exp(best.x)
ll1 = -betabinom.logpmf(k, n, *fit_mle(k, n)).sum()
print("two-level fit: a=%.4f b=%.4f nu=%.3f  nll %.2f (one-level, pooled "
      "counts: %.2f, not comparable)" % (A2, B2, NU, best.fun, ll1))

rng = np.random.default_rng(58)
sim = {q: [] for q in ("agree", "oracle_1to2", "oracle_2to1",
                       "ceiling_target_well", "ceiling_main",
                       "ceiling_main_sister")}
# a day-2 cell is itself one of the clone's cells at the time of the split; if
# the well-level spread is heterogeneity among those cells, the cell knows its
# own propensity, drawn around p_c like a third well, not p_c itself. A
# separate generator keeps the draws above identical to the earlier runs.
rng_s = np.random.default_rng(59)
for _ in range(a.sims):
    pc = rng.beta(A2, B2, size=len(both))
    s, l = {}, {}
    for w in (1, 2):
        pw = rng.beta(np.maximum(NU * pc, 1e-9), np.maximum(NU * (1 - pc), 1e-9))
        s[w] = rng.binomial(nw[w], pw)
        l[w] = majority(s[w], nw[w])
    if min(l[1].sum(), l[2].sum()) == 0 or max(l[1].mean(), l[2].mean()) == 1:
        continue
    sim["agree"].append((l[1] == l[2]).mean())
    sim["oracle_1to2"].append(roc_auc_score(l[2], s[1] / nw[1]))
    sim["oracle_2to1"].append(roc_auc_score(l[1], s[2] / nw[2]))
    sim["ceiling_target_well"].append(
        0.5 * (roc_auc_score(l[2], pc) + roc_auc_score(l[1], pc)))
    pcm = rng.beta(A2, B2, size=len(main))
    km = 0
    for w in (1, 2):
        pw = rng.beta(np.maximum(NU * pcm, 1e-9),
                      np.maximum(NU * (1 - pcm), 1e-9))
        km = km + rng.binomial(nw_m[w], pw)
    lm = majority(km, nw_m[1] + nw_m[2])
    sim["ceiling_main"].append(roc_auc_score(lm, pcm))
    ps = rng_s.beta(np.maximum(NU * pcm, 1e-9), np.maximum(NU * (1 - pcm), 1e-9))
    sim["ceiling_main_sister"].append(roc_auc_score(lm, ps))
for q, v in sim.items():
    v = np.asarray(v)
    rows.append(dict(fit="two_level", beta_a=round(A2, 4), beta_b=round(B2, 4),
                     nu=round(NU, 4), quantity=q, observed=obs.get(q, np.nan),
                     sim_mean=round(float(v.mean()), 4),
                     sim_lo=round(float(np.percentile(v, 2.5)), 4),
                     sim_hi=round(float(np.percentile(v, 97.5)), 4),
                     n_sims=len(v)))

R = pd.DataFrame(rows)
R.to_csv(a.out, index=False)
print(R.to_string(index=False))
print(obs)
