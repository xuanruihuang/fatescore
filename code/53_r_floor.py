"""The power simulation of 27_power_budget.py, read out in Pearson r units.

Why this script exists. 27_power_budget.py calibrates the second predictor by
a target AUROC difference (its DELTA_GRID) and computes, alongside, the power
of the paired test on Pearson r. Its "floor_r" column then interpolates that
r-test power against DELTA_GRID, so the number it reports is the AUROC
difference at which the r test reaches 80% power, not an r difference. At the
working point used for Chreode (A0 = 0.75, baseline r about 0.46) an AUROC
difference corresponds to an r difference about 1.8 times as large, so every
"r floor" taken from that column was too small by about that factor. Found on
2026-11-23 while revising the text.

What this script reports instead, per noise correlation rho:
  - the median half-width of the 95% paired-bootstrap interval of the r
    difference when the two predictors are equally good (delta = 0): the
    critical value an observed r difference has to exceed to be significant;
  - the r difference detectable with 80% power, interpolated on the r axis
    (the median observed r difference of each grid cell), with an upper bound
    reported when the smallest grid cell already has power >= 0.8.

Two scenarios, both on the in-vitro neutrophil-versus-monocyte clone table:
  n567  the 567 clones as they are (one early cell or more per clone), which
        reproduces the AUROC floors of 27_power_budget.py as a check;
  n183  183 clones drawn without replacement from the 567 for every simulated
        dataset, each with exactly one early cell: Chreode scored 183 cells and
        we treat them as 183 independent clones in a paired comparison, the two
        assumptions that favour the published differences.

The generative model, the AUROC calibration of the noise, the paired bootstrap
(2,000 resamples) and the seed convention are those of 27_power_budget.py,
imported from it rather than copied.

Two-level option (added after 57_label_ceiling.py). The one-level model, one
propensity per clone, predicts far more agreement between the two wells than is
observed (0.92 against 0.83), so its labels are too clean. With
--labels d2_labels.tsv --two-level a,b,nu the labels come from the two-level
model of 57_label_ceiling.py instead: a clone propensity p_c ~ Beta(a, b), a
well propensity around it with concentration nu, and each well's sisters
Binomial given their well's propensity, with the per-well sister counts and
day-2 cell counts of the 567 clones taken from the label table. The day-2
predictors are built on logit(p_c), the part a day-2 cell can know. --a0 sets
the working point; the published r of about 0.46 is the target, and under the
two-level model it is reached at a different AUROC than under the one-level
model, so the working point is chosen by r, not carried over.

Usage:
  python 53_r_floor.py <spec.json> <out.csv> [--workers 8] [--datasets 1000]
      [--labels d2_labels.tsv --two-level a,b,nu --a0 0.75]
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

A0 = 0.75
DELTAS = (0.0, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040,
          0.050, 0.065, 0.080, 0.100)
RHOS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)
N_BOOT = 2000
POWER = 0.80

TABLE = {}


def gen(rng, sel):
    """Labels, the clonal fate bias, and logit of the propensity a day-2 cell
    can know, for the clones in `sel`."""
    n_early_all, n_sib_all, a, b, tie = TABLE["t"]
    n_sib = n_sib_all[sel]
    if "two" not in TABLE:
        p = rng.beta(a, b, size=sel.size)
        k_sim = rng.binomial(n_sib, p)
    else:
        A2, B2, NU, nw1, nw2 = TABLE["two"]
        p = rng.beta(A2, B2, size=sel.size)
        k_sim = 0
        for nw in (nw1[sel], nw2[sel]):
            pw = rng.beta(np.maximum(NU * p, 1e-9),
                          np.maximum(NU * (1 - p), 1e-9))
            k_sim = k_sim + rng.binomial(nw, pw)
    y = pb.majority(k_sim, n_sib, tie)
    return y, k_sim / n_sib, pb.logit(p)


def calibrate(n_early, target, seed):
    """pb.calibrate_sigma with the label model of gen()."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, TABLE["t"][1].size, size=pb.CALIB_N)
    y, _, lp = gen(rng, idx)
    z = rng.normal(0.0, 1.0, size=pb.CALIB_N) / np.sqrt(n_early[idx])
    at = lambda sig: pb.auroc(lp + sig * z, y)
    ceiling = at(0.0)
    lo, hi = 1e-3, 1e3
    if at(lo) < target:
        return None, ceiling
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if at(mid) > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi)), ceiling


def run(job):
    scen, n, one_early, delta, rho, sig_b, sig_f, n_data, seed = job
    n_early_all, n_sib_all, a, b, tie = TABLE["t"]
    rng = np.random.default_rng(seed)
    r2 = np.sqrt(max(0.0, 1.0 - rho * rho))
    lo_a, hw_a, lo_r, hw_r, dr_med, rb = [], [], [], [], [], []
    for _ in range(n_data):
        if n == n_sib_all.size:
            sel = np.arange(n)
        else:
            sel = rng.choice(n_sib_all.size, size=n, replace=False)
        n_early = np.ones(n) if one_early else n_early_all[sel].astype(float)
        y, bias, lp = gen(rng, sel)
        scale = 1.0 / np.sqrt(n_early)
        z = rng.normal(size=n)
        z2 = rng.normal(size=n)
        s_b = lp + sig_b * scale * z
        s_f = lp + sig_f * scale * (rho * z + r2 * z2)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        yb = y[idx]
        d = pb.auroc_rows(s_f[idx], yb) - pb.auroc_rows(s_b[idx], yb)
        d = d[np.isfinite(d)]
        bb = bias[idx]
        rbase = pb.pearson_rows(s_b[idx], bb)
        dr = pb.pearson_rows(s_f[idx], bb) - rbase
        dr = dr[np.isfinite(dr)]
        if d.size < N_BOOT // 2 or dr.size < N_BOOT // 2:
            continue
        qa = np.percentile(d, [2.5, 97.5])
        qr = np.percentile(dr, [2.5, 97.5])
        lo_a.append(qa[0]); hw_a.append((qa[1] - qa[0]) / 2)
        lo_r.append(qr[0]); hw_r.append((qr[1] - qr[0]) / 2)
        dr_med.append(float(np.median(dr)))
        rb.append(float(np.nanmedian(rbase)))
    lo_a, lo_r = np.array(lo_a), np.array(lo_r)
    return dict(scenario=scen, n=n, rho=rho, delta_auroc=delta,
                power_auroc=float((lo_a > 0).mean()),
                power_r=float((lo_r > 0).mean()),
                median_dr=float(np.median(dr_med)),
                median_halfwidth_auroc=float(np.median(hw_a)),
                median_halfwidth_r=float(np.median(hw_r)),
                median_r_base=float(np.median(rb)),
                n_datasets=int(lo_a.size))


def floor(xs, ps):
    """Smallest x with power >= POWER, linear between grid cells; the first
    nonzero cell is returned as an upper bound when it already passes."""
    pts = sorted(zip(xs, ps))
    for i, (x, p) in enumerate(pts):
        if p >= POWER:
            if i == 0:
                return x, "upper bound"
            x0, p0 = pts[i - 1]
            if x0 == 0.0 and i == 1:
                return x, "upper bound"
            return x0 + (POWER - p0) * (x - x0) / (p - p0), ""
    return float("nan"), "beyond grid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", type=int, default=1000)
    ap.add_argument("--labels", default="")
    ap.add_argument("--two-level", default="")
    ap.add_argument("--a0", type=float, default=A0)
    ap.add_argument("--min-sis", type=int, default=3,
                    help="two-level mode: keep clones with at least this many "
                         "mature descendants (the label noise falls as it rises)")
    ap.add_argument("--n183-only", action="store_true")
    a_ = ap.parse_args()
    task = [t for t in json.load(open(a_.spec, encoding="utf-8"))
            if t["name"] == "T1_invitro_neu_vs_mono"][0]
    tie = bool(task.get("tie_to_positive", True))
    if a_.two_level:
        import pandas as pd
        T = pd.read_csv(a_.labels, sep="\t")
        T = T[(T["n_sis"] >= a_.min_sis)
              & T["dom"].isin(["Neutrophil", "Monocyte"])]
        C = T.groupby("clone").agg(
            n_early=("cell", "size"), n_neu=("n_neu", "first"),
            n_mono=("n_mono", "first"), n_neu_w1=("n_neu_w1", "first"),
            n_mono_w1=("n_mono_w1", "first"), n_neu_w2=("n_neu_w2", "first"),
            n_mono_w2=("n_mono_w2", "first"))
        n_early = C["n_early"].to_numpy(np.int64)
        n_sib = (C["n_neu"] + C["n_mono"]).to_numpy(np.int64)
        nw1 = (C["n_neu_w1"] + C["n_mono_w1"]).to_numpy(np.int64)
        nw2 = (C["n_neu_w2"] + C["n_mono_w2"]).to_numpy(np.int64)
        A2, B2, NU = (float(x) for x in a_.two_level.split(","))
        a, b = A2, B2
        TABLE["two"] = (A2, B2, NU, nw1, nw2)
        print("two-level labels: clones %d  a=%.4f b=%.4f nu=%.4f"
              % (n_sib.size, A2, B2, NU), flush=True)
    else:
        n_early, n_sib, k = pb.clone_table(task)
        a, b, degen = pb.fit_beta(k, n_sib)
        print("clones %d  beta a=%.3f b=%.3f degenerate=%s"
              % (n_sib.size, a, b, degen), flush=True)
    TABLE["t"] = (n_early, n_sib, a, b, tie)
    A0w = a_.a0

    jobs = []
    scens = (("n567", n_sib.size, False), ("n183", 183, True))
    for si, (scen, n, one) in enumerate(scens):
        if a_.n183_only and scen != "n183":
            continue
        ne = np.ones_like(n_early) if one else n_early
        sig = {}
        for di, dl in enumerate(DELTAS):
            if "two" in TABLE:
                s, ceil = calibrate(ne, A0w + dl, [pb.SEED, 900 + si, di])
            else:
                s, ceil = pb.calibrate_sigma(ne, n_sib, tie, a, b, A0w + dl,
                                             [pb.SEED, 900 + si, di])
            sig[dl] = s
            print("%s A0+%.4f sigma %s (ceiling %.3f)" % (scen, dl, s, ceil),
                  flush=True)
        for ri, rho in enumerate(RHOS):
            for di, dl in enumerate(DELTAS):
                jobs.append((scen, n, one, dl, rho, sig[0.0], sig[dl],
                             a_.datasets, [pb.SEED, 53, si, ri, di]))
    # the clone table is set before the pool forks, so workers inherit it
    with Pool(a_.workers) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    keys = list(rows[0].keys())
    with open(a_.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    summ = a_.out.replace(".csv", "_summary.csv")
    with open(summ, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "rho", "median_r_base", "critical_r",
                    "mde_r", "mde_r_note", "critical_auroc", "mde_auroc",
                    "mde_auroc_note"])
        for scen in ("n567", "n183"):
            if a_.n183_only and scen != "n183":
                continue
            for rho in RHOS:
                cell = [r for r in rows if r["scenario"] == scen
                        and r["rho"] == rho]
                zero = [r for r in cell if r["delta_auroc"] == 0.0][0]
                fr, nr = floor([r["median_dr"] for r in cell],
                               [r["power_r"] for r in cell])
                fa, na = floor([r["delta_auroc"] for r in cell],
                               [r["power_auroc"] for r in cell])
                w.writerow([scen, rho, round(zero["median_r_base"], 4),
                            round(zero["median_halfwidth_r"], 4),
                            round(fr, 4), nr,
                            round(zero["median_halfwidth_auroc"], 4),
                            round(fa, 4), na])
    print(open(summ).read(), flush=True)


if __name__ == "__main__":
    main()
