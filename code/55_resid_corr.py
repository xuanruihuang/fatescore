"""Map the simulation's noise correlation onto the residual correlation we measure.

Usage: 55_resid_corr.py <spec.json> <out.csv> [--datasets 500] [--workers 8]

53_r_floor.py states its critical values against rho, the correlation between
the two simulated predictors' score noise. What we measure on real predictors
(30_kernel_eval.py, 49_pair_floors.py) is the residual correlation: per clone,
each predictor's score minus the mean score of the clone's class, correlated
between the two predictors. The two are not the same quantity: two simulated
predictors share the clone's latent propensity, and its spread within a class
enters both residuals, so the residual correlation exceeds rho. This script
computes the residual correlation exactly as the scorer does, on simulated
predictors at each rho, so that a measured residual correlation can be read
as the rho it corresponds to.

Two versions of the score are used, because the real scores are classifier
probabilities while the simulated ones are on the logit scale: the raw logit
score, and that score passed through a logistic fit of the label (a monotone
recalibration onto the probability scale). Both on the 567 in-vitro clones with
their own early-cell counts, and on 183 clones with one early cell each, at
A0 = 0.75 and delta = 0, with the generative model of 27_power_budget.py.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import importlib.util
import json
from multiprocessing import Pool

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "power_budget", os.path.join(_here, "27_power_budget.py"))
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)
_spec2 = importlib.util.spec_from_file_location(
    "r_floor", os.path.join(_here, "53_r_floor.py"))
rf = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(rf)

A0 = 0.75
RHOS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)
TABLE = {}


def resid(s, y):
    yb = y.astype(bool)
    return s - np.where(yb, s[yb].mean(), s[~yb].mean())


def logistic_prob(s, y, iters=30):
    """Probability from a one-variable logistic fit of y on standardised s."""
    z = (s - s.mean()) / (s.std() + 1e-12)
    X = np.column_stack([np.ones_like(z), z])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        w = w + np.linalg.solve(H, g)
    return 1.0 / (1.0 + np.exp(-X @ w))


def run(job):
    scen, n, one, rho, sig, n_data, seed = job
    n_early_all, n_sib_all, a, b, tie = TABLE["t"]
    rng = np.random.default_rng(seed)
    r2 = np.sqrt(max(0.0, 1.0 - rho * rho))
    c_logit, c_prob = [], []
    for _ in range(n_data):
        sel = (np.arange(n) if n == n_sib_all.size
               else rng.choice(n_sib_all.size, size=n, replace=False))
        n_sib = n_sib_all[sel]
        ne = np.ones(n) if one else n_early_all[sel].astype(float)
        if "two" in rf.TABLE:
            y, _, lp = rf.gen(rng, sel)
            y = y.astype(float)
        else:
            p = rng.beta(a, b, size=n)
            y = pb.majority(rng.binomial(n_sib, p), n_sib, tie).astype(float)
            lp = pb.logit(p)
        if y.min() == y.max():
            continue
        z, z2 = rng.normal(size=n), rng.normal(size=n)
        s1 = lp + sig * z / np.sqrt(ne)
        s2 = lp + sig * (rho * z + r2 * z2) / np.sqrt(ne)
        c_logit.append(np.corrcoef(resid(s1, y), resid(s2, y))[0, 1])
        c_prob.append(np.corrcoef(resid(logistic_prob(s1, y), y),
                                  resid(logistic_prob(s2, y), y))[0, 1])
    return dict(scenario=scen, rho=rho,
                resid_corr_logit=round(float(np.median(c_logit)), 4),
                resid_corr_prob=round(float(np.median(c_prob)), 4),
                resid_corr_prob_q05=round(float(np.percentile(c_prob, 5)), 4),
                resid_corr_prob_q95=round(float(np.percentile(c_prob, 95)), 4),
                n_datasets=len(c_prob))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--datasets", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--labels", default="",
                    help="d2_labels.tsv; needed with --two-level")
    ap.add_argument("--two-level", default="",
                    help="a,b,nu of the two-level label model "
                         "(57_label_ceiling.py); labels are then drawn as in "
                         "53_r_floor.py --two-level")
    a_ = ap.parse_args()
    task = [t for t in json.load(open(a_.spec, encoding="utf-8"))
            if t["name"] == "T1_invitro_neu_vs_mono"][0]
    tie = bool(task.get("tie_to_positive", True))
    if a_.two_level:
        import pandas as pd
        T = pd.read_csv(a_.labels, sep="\t")
        T = T[(T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])]
        C = T.groupby("clone").agg(
            n_early=("cell", "size"), n_neu=("n_neu", "first"),
            n_mono=("n_mono", "first"), n_neu_w1=("n_neu_w1", "first"),
            n_mono_w1=("n_mono_w1", "first"), n_neu_w2=("n_neu_w2", "first"),
            n_mono_w2=("n_mono_w2", "first"))
        n_early = C["n_early"].to_numpy(np.int64)
        n_sib = (C["n_neu"] + C["n_mono"]).to_numpy(np.int64)
        nw1 = (C["n_neu_w1"] + C["n_mono_w1"]).to_numpy(np.int64)
        nw2 = (C["n_neu_w2"] + C["n_mono_w2"]).to_numpy(np.int64)
        a, b, nu = (float(x) for x in a_.two_level.split(","))
        rf.TABLE["two"] = (a, b, nu, nw1, nw2)
    else:
        n_early, n_sib, k = pb.clone_table(task)
        a, b, _ = pb.fit_beta(k, n_sib)
    TABLE["t"] = (n_early, n_sib, a, b, tie)
    rf.TABLE["t"] = TABLE["t"]
    jobs = []
    for si, (scen, n, one) in enumerate((("n567", n_sib.size, False),
                                         ("n183", 183, True))):
        ne = np.ones_like(n_early) if one else n_early
        if "two" in rf.TABLE:
            sig, _ = rf.calibrate(ne, A0, [pb.SEED, 900 + si, 0])
        else:
            sig, _ = pb.calibrate_sigma(ne, n_sib, tie, a, b, A0,
                                        [pb.SEED, 900 + si, 0])
        for ri, rho in enumerate(RHOS):
            jobs.append((scen, n, one, rho, sig, a_.datasets,
                         [pb.SEED, 55, si, ri]))
    with Pool(a_.workers) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    with open(a_.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r, flush=True)
    # the AUROC-to-r correspondence of the generative model, for the check
    # against PRESCIENT's reported pair (r = 0.399 with AUROC = 0.725): the
    # median Pearson r between a predictor's score and the observed fate bias
    # at baseline AUROC A0, 567 clones with their own early-cell counts
    rng = np.random.default_rng([pb.SEED, 55, 99])
    for a0 in (0.65, 0.70, 0.75):
        if "two" in rf.TABLE:
            sig, _ = rf.calibrate(n_early, a0, [pb.SEED, 955, int(a0 * 100)])
        else:
            sig, _ = pb.calibrate_sigma(n_early, n_sib, tie, a, b, a0,
                                        [pb.SEED, 955, int(a0 * 100)])
        rs = []
        sel = np.arange(n_sib.size)
        for _ in range(a_.datasets):
            if "two" in rf.TABLE:
                y, bias, lp = rf.gen(rng, sel)
            else:
                p = rng.beta(a, b, size=sel.size)
                kk = rng.binomial(n_sib, p)
                bias, lp = kk / n_sib, pb.logit(p)
            s_ = lp + sig * rng.normal(size=sel.size) / np.sqrt(n_early)
            rs.append(np.corrcoef(s_, bias)[0, 1])
        print("r_at_A0 %.2f median_r %.4f" % (a0, float(np.median(rs))),
              flush=True)


if __name__ == "__main__":
    main()
