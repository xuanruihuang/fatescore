"""What increment the design can resolve, from the clone labels alone.

The manuscript's comparison on the first dataset resolves differences of about
0.05 in clone-level AUROC, and the increment it measured was 0.005. Before
spending weeks of bandwidth on requantifying other arms of the same deposit,
ask what those arms could resolve. The answer depends only on the label
structure -- how many clones, how many mature siblings called each clone's
fate, how many early cells carry its state, how balanced the classes are --
so it is computable from the GEO metadata and clone matrix, with no splicing
layer and no expression data at all.

The generative model, the grid and the decision rules were fixed before any
number here was computed: notes/velocity_fate_power_prereg.md.

Usage:
  python 27_power_budget.py <spec.json> <out_prefix> [--workers 8]
"""
import os

# Set before numpy is imported: this is a shared machine and the work is
# parallelised over processes, so each process stays single-threaded.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

# From the pre-registration. Nothing here is tuned to a result.
A0_GRID = (0.55, 0.60, 0.65, 0.70, 0.75)
A0_MAIN = 0.65
DELTA_GRID = (0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080)
RHO_GRID = (1.0, 0.5, 0.0)
RHO_MAIN = 0.5
N_DATASETS = 1000
N_BOOT = 2000
SEED = 13
CALIB_N = 200000
MIN_SIBLINGS = 3
POWER_TARGET = 0.80
CLIP = 1e-6


# ---------------------------------------------------------------- input
# The two readers mirror 26_larry_arms_feasibility.py, including that the
# deposit writes the clone matrix clones x cells for some arms and
# cells x clones for others.

def opener(path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") \
        if path.endswith(".gz") else open(path, "rt", encoding="utf-8", newline="")


def read_meta(path):
    with opener(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_clone_cells(path, n_cells):
    cells_of = defaultdict(set)
    shape = None
    flip = None
    with opener(path) as fh:
        for line in fh:
            if line.startswith("%"):
                continue
            parts = line.split()
            if not parts:
                continue
            if shape is None:
                shape = (int(parts[0]), int(parts[1]))
                flip = shape[0] == n_cells
                if not flip and shape[1] != n_cells:
                    raise SystemExit("clone matrix %d x %d matches neither the "
                                     "%d cells" % (shape + (n_cells,)))
                continue
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
            if len(parts) > 2 and float(parts[2]) == 0.0:
                continue
            cells_of[j if flip else i].add(i if flip else j)
    return cells_of


def clone_table(task):
    """Per clone: early cells, the siblings that define the label, positives."""
    meta = read_meta(task["meta"])
    cells_of = read_clone_cells(task["clones"], len(meta))
    time = [r[task.get("time_col", "Time point")] for r in meta]
    ctype = [r[task.get("type_col", "Cell type annotation")] for r in meta]
    early = [t == task["early"] for t in time]
    rank = {t: i for i, t in enumerate(task["mature"])}
    pos, neg = task["positive"], task.get("negative")

    n_early, n_sib, k_pos = [], [], []
    for cells in cells_of.values():
        e = [i for i in cells if early[i]]
        if not e:
            continue
        lm = [i for i in cells if not early[i] and ctype[i] in rank]
        if len(lm) < MIN_SIBLINGS:
            continue
        cnt = Counter(ctype[i] for i in lm)
        top = max(cnt.values())
        called = min((t for t in cnt if cnt[t] == top), key=rank.get)
        if neg is not None:
            if called not in (pos, neg):
                continue
            tot = cnt[pos] + cnt[neg]
            n_sib.append(tot)
            k_pos.append(cnt[pos])
        else:
            n_sib.append(len(lm))
            k_pos.append(cnt[pos])
        n_early.append(len(e))
    return (np.array(n_early, dtype=np.int64),
            np.array(n_sib, dtype=np.int64),
            np.array(k_pos, dtype=np.int64))


# ---------------------------------------------------------------- model

def fit_beta(k, n):
    """Moment fit of the clone-propensity Beta with the binomial noise of a
    finite sibling count taken out. Returns (a, b, degenerate)."""
    f = k / n
    mu = float(f.mean())
    var = float(f.var(ddof=1))
    m1 = float((1.0 / n).mean())
    if not (0.0 < mu < 1.0):
        return 1.0, 1.0, True
    rho = (var / (mu * (1.0 - mu)) - m1) / (1.0 - m1)
    if not (0.0 < rho < 1.0):
        return 1.0, 1.0, True
    s = 1.0 / rho - 1.0
    a, b = mu * s, (1.0 - mu) * s
    if a <= 0 or b <= 0:
        return 1.0, 1.0, True
    return a, b, False


def majority(k, n, tie_to_positive):
    """The manuscript's rule: the majority class, ties by the fixed order."""
    return (k * 2 > n) | (bool(tie_to_positive) & (k * 2 == n))


def logit(p):
    pc = np.clip(p, CLIP, 1.0 - CLIP)
    return np.log(pc / (1.0 - pc))


def auroc(score, y):
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(score, kind="stable")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    return (ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def pearson_rows(score, target):
    """Pearson r for each row of a (B, n) score matrix against the matching row
    of a (B, n) target matrix."""
    s = score - score.mean(1, keepdims=True)
    t = target - target.mean(1, keepdims=True)
    num = (s * t).sum(1)
    den = np.sqrt((s * s).sum(1) * (t * t).sum(1))
    out = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    return out


def auroc_rows(score, y):
    """AUROC of each row of a (B, n) score matrix against a (B, n) label
    matrix."""
    B, n = score.shape
    n1 = y.sum(1).astype(np.float64)
    n0 = n - n1
    order = np.argsort(score, axis=1, kind="stable")
    ranks = np.empty((B, n), dtype=np.float64)
    seq = np.broadcast_to(np.arange(1, n + 1, dtype=np.float64), (B, n))
    np.put_along_axis(ranks, order, seq, axis=1)
    out = ((ranks * y).sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = np.nan
    return out


def calibrate_sigma(n_early, n_sib, tie, a, b, target, seed):
    """Noise scale at which the clone-level AUROC equals target; None when even
    a noiseless score cannot reach it, because the label itself is noisy."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_sib.size, size=CALIB_N)
    ns, ne = n_sib[idx], n_early[idx]
    p = rng.beta(a, b, size=CALIB_N)
    y = majority(rng.binomial(ns, p), ns, tie)
    lp = logit(p)
    z = rng.normal(0.0, 1.0, size=CALIB_N) / np.sqrt(ne)

    def at(sig):
        return auroc(lp + sig * z, y)

    ceiling = at(0.0)
    lo, hi = 1e-3, 1e3
    if at(lo) < target:
        return None, ceiling
    if at(hi) > target:
        return hi, ceiling
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if at(mid) > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi)), ceiling


# ---------------------------------------------------------------- experiment

def run_cell(job):
    """One (task, A0, delta, rho) cell: N_DATASETS simulated datasets, each
    with its own paired bootstrap over clones."""
    (key, n_early, n_sib, tie, a, b, sig_base, sig_full, rho, seed) = job
    n = n_sib.size
    rng = np.random.default_rng(seed)
    r2 = np.sqrt(max(0.0, 1.0 - rho * rho))
    lo_b, pts, wid, var = [], [], [], []
    rlo, rpts, rbase, rvar = [], [], [], []
    for _ in range(N_DATASETS):
        p = rng.beta(a, b, size=n)
        k_sim = rng.binomial(n_sib, p)
        y = majority(k_sim, n_sib, tie)
        bias = k_sim / n_sib        # the clonal fate bias the literature scores
        lp = logit(p)
        scale = 1.0 / np.sqrt(n_early)
        z = rng.normal(0.0, 1.0, size=n)
        z2 = rng.normal(0.0, 1.0, size=n)
        s_base = lp + sig_base * scale * z
        s_full = lp + sig_full * scale * (rho * z + r2 * z2)
        d_point = auroc(s_full, y) - auroc(s_base, y)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        yb = y[idx]
        d = auroc_rows(s_full[idx], yb) - auroc_rows(s_base[idx], yb)
        d = d[np.isfinite(d)]
        # the same comparison in the units the literature reports: Pearson r
        # against the observed clonal fate bias
        bb = bias[idx]
        r_base_rows = pearson_rows(s_base[idx], bb)
        dr = pearson_rows(s_full[idx], bb) - r_base_rows
        dr = dr[np.isfinite(dr)]
        if not np.isfinite(d_point) or d.size < N_BOOT // 2:
            continue
        q = np.percentile(d, [2.5, 97.5])
        lo_b.append(q[0])
        wid.append(q[1] - q[0])
        pts.append(d_point)
        var.append(float(np.var(d, ddof=1)))
        if dr.size >= N_BOOT // 2:
            rq = np.percentile(dr, [2.5, 97.5])
            rlo.append(rq[0])
            rpts.append(float(np.median(dr)))
            rbase.append(float(np.nanmedian(r_base_rows)))
            rvar.append(float(np.var(dr, ddof=1)))
    return key, (np.array(lo_b), np.array(pts), np.array(wid), np.array(var),
                 np.array(rlo), np.array(rpts), np.array(rbase),
                 np.array(rvar))


def floor_from_grid(deltas, powers):
    """Smallest delta reaching POWER_TARGET, linear between grid points."""
    for i, pw in enumerate(powers):
        if not np.isfinite(pw) or pw < POWER_TARGET:
            continue
        if i == 0:
            return float(deltas[0])
        d0, d1, p0, p1 = deltas[i - 1], deltas[i], powers[i - 1], powers[i]
        if not np.isfinite(p0) or p1 == p0:
            return float(d1)
        return round(d0 + (POWER_TARGET - p0) * (d1 - d0) / (p1 - p0), 3)
    return None


def main():
    global N_DATASETS, N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out_prefix")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", type=int, default=N_DATASETS,
                    help="only for a smoke test; the registered value is the "
                         "default and the reported run must use it")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help="only for a smoke test; see --datasets")
    ap.add_argument("--rho", default="",
                    help="comma-separated noise correlations to use instead of "
                         "the registered 1.0,0.5,0.0; the first becomes the "
                         "headline one. Added after 28_arm_correlation.py "
                         "measured what this correlation really is between "
                         "arms of the same family")
    args = ap.parse_args()
    N_DATASETS, N_BOOT = args.datasets, args.boot
    if args.rho:
        global RHO_GRID, RHO_MAIN
        RHO_GRID = tuple(float(x) for x in args.rho.split(","))
        RHO_MAIN = RHO_GRID[0]
        print("rho grid overridden: %s (headline %.3f)"
              % (RHO_GRID, RHO_MAIN), flush=True)
    if (N_DATASETS, N_BOOT) != (1000, 2000):
        print("WARNING: smoke test settings, not the registered %d x %d"
              % (1000, 2000), flush=True)
    tasks = json.load(open(args.spec, encoding="utf-8"))

    design_rows, grid_rows, floor_rows = [], [], []
    jobs = []
    meta_of = {}

    for ti, t in enumerate(tasks):
        name = t["name"]
        n_early, n_sib, k = clone_table(t)
        tie = bool(t.get("tie_to_positive", True))
        a, b, degen = fit_beta(k, n_sib)
        meta_of[name] = dict(ti=ti, n=n_sib.size, pool=bool(t.get("pool")))
        sig0, ceiling = calibrate_sigma(n_early, n_sib, tie, a, b, A0_MAIN,
                                        [SEED, ti, 0])
        design_rows += [
            (name, "clones", n_sib.size),
            (name, "early cells", int(n_early.sum())),
            (name, "median early cells per clone", float(np.median(n_early))),
            (name, "median siblings defining the label", float(np.median(n_sib))),
            (name, "observed positive fraction", round(float((k / n_sib).mean()), 4)),
            (name, "beta a", round(a, 4)),
            (name, "beta b", round(b, 4)),
            (name, "beta fit degenerate", int(degen)),
            (name, "label ceiling", round(float(ceiling), 4)),
        ]
        print("%-26s n=%4d early=%5d beta(%.2f,%.2f)%s label ceiling %.3f"
              % (name, n_sib.size, n_early.sum(), a, b,
                 " DEGENERATE" if degen else "", ceiling), flush=True)

        for ai, a0 in enumerate(A0_GRID):
            sb, _ = calibrate_sigma(n_early, n_sib, tie, a, b, a0, [SEED, ti, ai])
            if sb is None:
                grid_rows.append((name, a0, "", "", "", "", "",
                                  "A0 above the label ceiling"))
                continue
            rhos = RHO_GRID if a0 == A0_MAIN else (RHO_MAIN,)
            for ri, rho in enumerate(rhos):
                # delta = 0 for the validity check and for the achievable width
                jobs.append((("cell", name, a0, 0.0, rho), n_early, n_sib, tie,
                             a, b, sb, sb, rho, [SEED, ti, ai, 0, ri]))
                for di, dl in enumerate(DELTA_GRID):
                    sf, _ = calibrate_sigma(n_early, n_sib, tie, a, b, a0 + dl,
                                            [SEED, ti, ai, di + 1])
                    if sf is None:
                        grid_rows.append((name, a0, dl, rho, "", "", "",
                                          "A0+delta above the label ceiling"))
                        continue
                    jobs.append((("cell", name, a0, dl, rho), n_early, n_sib,
                                 tie, a, b, sb, sf, rho,
                                 [SEED, ti, ai, di + 1, ri]))

    print("cells to simulate: %d, workers %d" % (len(jobs), args.workers),
          flush=True)
    results = {}
    with Pool(args.workers) as pool:
        for i, (key, val) in enumerate(pool.imap_unordered(run_cell, jobs,
                                                           chunksize=1)):
            results[key] = val
            if (i + 1) % 10 == 0:
                print("  %d/%d" % (i + 1, len(jobs)), flush=True)

    for (_, name, a0, dl, rho), val in sorted(
            results.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][4], kv[0][3])):
        lo, pts, wid, var, rlo, rpts, rbase, rvar = val
        grid_rows.append((name, a0, dl, rho, round(float((lo > 0).mean()), 4),
                          round(float(np.median(pts)), 4),
                          round(float(np.median(wid)), 4),
                          round(float((rlo > 0).mean()), 4) if rlo.size else "",
                          round(float(np.median(rpts)), 4) if rpts.size else "",
                          round(float(np.median(rbase)), 4) if rbase.size else "",
                          "delta = 0" if dl == 0.0 else ""))

    for name in meta_of:
        for a0 in A0_GRID:
            for rho in (RHO_GRID if a0 == A0_MAIN else (RHO_MAIN,)):
                powers, powers_r = [], []
                for dl in DELTA_GRID:
                    r = results.get(("cell", name, a0, dl, rho))
                    powers.append(float((r[0] > 0).mean()) if r else np.nan)
                    powers_r.append(float((r[4] > 0).mean())
                                    if r is not None and r[4].size else np.nan)
                z = results.get(("cell", name, a0, 0.0, rho))
                if z is None:
                    continue
                fl = floor_from_grid(DELTA_GRID, powers)
                fl_r = floor_from_grid(DELTA_GRID, powers_r)
                floor_rows.append((name, a0, rho,
                                   fl if fl is not None else "",
                                   round(float(np.median(z[2])), 4),
                                   fl_r if fl_r is not None else "",
                                   round(float(np.median(z[6])), 4)
                                   if z[6].size else ""))
                print("%-26s A0=%.2f rho=%.1f  width@0 %.3f  floor %s  "
                      "r_base %s  floor_r %s"
                      % (name, a0, rho, float(np.median(z[2])),
                         "%.3f" % fl if fl is not None
                         else "> %.3f" % DELTA_GRID[-1],
                         "%.3f" % float(np.median(z[6])) if z[6].size else "NA",
                         "%.3f" % fl_r if fl_r is not None
                         else "> %.3f" % DELTA_GRID[-1]), flush=True)

    # Pooled over the realisations of the same task, inverse variance.
    pool_names = [n for n, m in meta_of.items() if m["pool"]]
    if len(pool_names) >= 2:
        for rho in RHO_GRID:
            powers, widths, powers_r = [], [], []

            def pooled_power(est_list, var_list):
                est = np.stack(est_list)
                v = np.stack(var_list)
                if not np.all(v > 0):
                    return None, None, None
                w = 1.0 / v
                mean = (est * w).sum(0) / w.sum(0)
                se = np.sqrt(1.0 / w.sum(0))
                return (float((mean - 1.96 * se > 0).mean()),
                        float(np.median(mean)), float(np.median(2 * 1.96 * se)))

            for dl in list(DELTA_GRID) + [0.0]:
                cells = [results.get(("cell", n, A0_MAIN, dl, rho))
                         for n in pool_names]
                if any(c is None for c in cells):
                    if dl != 0.0:
                        powers.append(np.nan)
                        powers_r.append(np.nan)
                    continue
                m = min(c[1].size for c in cells)
                pw, pt, wd = pooled_power([c[1][:m] for c in cells],
                                          [c[3][:m] for c in cells])
                mr = min(c[5].size for c in cells)
                if mr > 0:
                    pw_r, pt_r, wd_r = pooled_power([c[5][:mr] for c in cells],
                                                    [c[7][:mr] for c in cells])
                else:
                    pw_r = pt_r = wd_r = None
                nm = "+".join(pool_names)
                if pw is None:
                    # rho = 1 makes the paired difference deterministic, so the
                    # bootstrap variance is zero and the pooling is undefined.
                    grid_rows.append((nm, A0_MAIN, dl, rho, "", "", "",
                                      "", "", "",
                                      "degenerate: zero bootstrap variance"))
                    if dl != 0.0:
                        powers.append(np.nan)
                        powers_r.append(np.nan)
                    continue
                grid_rows.append((nm, A0_MAIN, dl, rho, round(pw, 4),
                                  round(pt, 4), round(wd, 4),
                                  round(pw_r, 4) if pw_r is not None else "",
                                  round(pt_r, 4) if pt_r is not None else "",
                                  "",
                                  "inverse-variance pooled"
                                  + (", delta = 0" if dl == 0.0 else "")))
                if dl == 0.0:
                    widths.append(wd)
                else:
                    powers.append(pw)
                    powers_r.append(pw_r if pw_r is not None else np.nan)
            fl = floor_from_grid(DELTA_GRID, powers)
            fl_r = floor_from_grid(DELTA_GRID, powers_r)
            floor_rows.append(("+".join(pool_names), A0_MAIN, rho,
                               fl if fl is not None else "",
                               round(widths[0], 4) if widths else "",
                               fl_r if fl_r is not None else "", ""))
            print("POOLED %-20s A0=%.2f rho=%.1f  width@0 %s  floor %s  floor_r %s"
                  % ("+".join(pool_names), A0_MAIN, rho,
                     "%.3f" % widths[0] if widths else "NA",
                     "%.3f" % fl if fl is not None
                     else "> %.3f" % DELTA_GRID[-1],
                     "%.3f" % fl_r if fl_r is not None
                     else "> %.3f" % DELTA_GRID[-1]), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".",
                exist_ok=True)
    with open(args.out_prefix + "_budget.csv", "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "A0", "delta", "rho_noise", "power",
                    "median_point_estimate", "median_interval_width",
                    "power_r", "median_point_estimate_r", "median_r_baseline",
                    "note"])
        w.writerows(grid_rows)
    with open(args.out_prefix + "_floor.csv", "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "A0", "rho_noise", "detection_floor",
                    "interval_width_at_zero", "detection_floor_r",
                    "r_of_the_baseline_arm"])
        w.writerows(floor_rows)
    with open(args.out_prefix + "_design.csv", "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "quantity", "value"])
        w.writerows(design_rows)
    print("wrote %s_{budget,floor,design}.csv" % args.out_prefix)


if __name__ == "__main__":
    main()
