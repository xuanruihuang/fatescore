"""What a second abundance measurement at the unspliced layer's depth is worth,
and what a permuted velocity field is worth.

Usage: 05_depth_and_null.py <featdir> <labels.tsv> <outprefix>

Three things are measured, all with the folds and tuning of 03_evaluate.py.

1. Read splitting. Each cell's spliced counts are split into two disjoint
   parts: the second part gets exactly that cell's unspliced depth, the first
   keeps the rest. The second part is therefore an independent measurement of
   the same abundances at the same depth as the unspliced layer. Adding it to a
   model built on the first part says what any such measurement is worth;
   adding the unspliced layer in its place says whether the splicing ratio is
   worth more than that.

   The arm preregistered as N2 added a thinned copy of the *same* reads. That
   copy is a deterministic function of the counts it came from, so by
   construction it can add no information; it is reported for completeness but
   the disjoint split is the informative control.

2. A depth curve: the state baseline on spliced counts thinned to a series of
   depths, which turns "AUROC" into "effective depth" and lets the unspliced
   layer's contribution be read off in reads.

3. The preregistered permutation null: the dynamical velocity vectors are
   shuffled among day-2 cells within the same annotated state, 20 times.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cvlib import cp10k_log, thin_to, run_arm, clone_level, auroc

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("outprefix")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--n-jobs", type=int, default=24)
ap.add_argument("--pcs", type=int, default=50)
ap.add_argument("--split-seeds", type=int, default=5)
ap.add_argument("--depth-seeds", type=int, default=3)
ap.add_argument("--perms", type=int, default=20)
ap.add_argument("--restrict-pop", default="",
                help="keep only cells whose pop column equals this, so that a "
                     "batch that differs in fate composition cannot carry the "
                     "comparison")
ap.add_argument("--pos", default="Neutrophil",
                help="dominant fate counted as the positive class")
ap.add_argument("--neg", default="Monocyte",
                help="dominant fate counted as the negative class")
ap.add_argument("--pop-ref", default="LSK",
                help="value of the pop column encoded as 1")
CFG = ap.parse_args()


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


T = pd.read_csv(CFG.labels, sep="\t")
d2_names = json.load(open(os.path.join(CFG.featdir, "d2_names.json")))
assert list(T["cell"]) == d2_names


def load(name):
    p = os.path.join(CFG.featdir, name + ".npy")
    return np.load(p) if os.path.exists(p) else None


raw_s, raw_u = load("raw_spliced"), load("raw_unspliced")
Ms, Mu, vel_dyn = load("Ms"), load("Mu"), load("vel_dyn")

sel = (T["n_sis"] >= CFG.min_sis) & T["dom"].isin([CFG.pos, CFG.neg])
if CFG.restrict_pop:
    sel = sel & (T["pop"] == CFG.restrict_pop)
idx = np.where(sel.values)[0]
y = (T.loc[sel, "dom"].values == CFG.pos).astype(int)
groups = T.loc[sel, "clone"].values
state_d2 = T.loc[sel, "state_d2"].astype(str).values
say("task A: %d cells, %d clones" % (len(idx), len(np.unique(groups))))

S = raw_s[idx]
U = raw_u[idx]
u_depth = U.sum(1)
s_depth = S.sum(1)
say("median spliced depth %.0f, median unspliced depth %.0f, ratio %.3f"
    % (np.median(s_depth), np.median(u_depth),
       float(np.median(u_depth / np.maximum(s_depth, 1)))))
pd.DataFrame([dict(median_spliced_on_selected_genes=float(np.median(s_depth)),
                   median_unspliced_on_selected_genes=float(np.median(u_depth)),
                   median_unspliced_fraction=float(
                       np.median(u_depth / np.maximum(s_depth + u_depth, 1))),
                   n_cells=len(idx))]
             ).to_csv(CFG.outprefix + "_depths.csv", index=False)

pca_blocks = {
    "state_s": cp10k_log(S),
    "unspl": cp10k_log(U),
    "Ms": np.log1p(np.clip(Ms[idx], 0, None)),
}
raw_blocks = {"pop": (T.loc[sel, "pop"].values == CFG.pop_ref).astype(float)[:, None]}
jobs = []

# ---- 1. read splitting -----------------------------------------------------
for s in range(CFG.split_seeds):
    part2 = thin_to(S, u_depth, seed=100 + s)
    part1 = S - part2
    assert (part1 >= 0).all()
    pca_blocks["part1_s%d" % s] = cp10k_log(part1)
    pca_blocks["part2_s%d" % s] = cp10k_log(part2)
    pca_blocks["thincopy_s%d" % s] = cp10k_log(thin_to(S, u_depth, seed=200 + s))
    jobs += [
        ("B0_part1_only_s%d" % s, ["pop", "part1_s%d" % s]),
        ("B1_part1_plus_part2_s%d" % s,
         ["pop", "part1_s%d" % s, "part2_s%d" % s]),
        ("B2_part1_plus_unspliced_s%d" % s,
         ["pop", "part1_s%d" % s, "unspl"]),
        ("N2_state_plus_thincopy_s%d" % s,
         ["pop", "state_s", "thincopy_s%d" % s]),
    ]
say("read-splitting blocks built")

# ---- 2. depth curve --------------------------------------------------------
FRACS = [1.0, 0.5, 0.25, 0.125, 0.0625]
for f in FRACS:
    for s in range(CFG.depth_seeds):
        key = "dep_f%g_s%d" % (f, s)
        pca_blocks[key] = (pca_blocks["state_s"] if f == 1.0
                           else cp10k_log(thin_to(S, s_depth * f, seed=300 + s)))
        jobs.append(("D_frac%g_s%d" % (f, s), ["pop", key]))
        if f == 1.0:
            break
for s in range(CFG.depth_seeds):
    key = "dep_udepth_s%d" % s
    pca_blocks[key] = cp10k_log(thin_to(S, u_depth, seed=400 + s))
    jobs.append(("D_udepth_s%d" % s, ["pop", key]))
say("depth-curve blocks built")

# ---- 3. permutation null for the velocity field ----------------------------
V = vel_dyn[idx]
for s in range(CFG.perms):
    r = np.random.default_rng(500 + s)
    perm = np.arange(len(idx))
    for st in np.unique(state_d2):
        m = np.where(state_d2 == st)[0]
        perm[m] = r.permutation(m)
    pca_blocks["velperm_s%d" % s] = V[perm]
    jobs.append(("N1_smoothed_plus_vel_perm_s%d" % s,
                 ["pop", "Ms", "velperm_s%d" % s]))
pca_blocks["vel_dyn"] = V
jobs.append(("V2_smoothed_plus_vel_dyn", ["pop", "Ms", "vel_dyn"]))
jobs.append(("A4_state_smoothed", ["pop", "Ms"]))
jobs.append(("A2_state_spliced", ["pop", "state_s"]))
jobs.append(("A6_state_plus_unspliced", ["pop", "state_s", "unspl"]))
say("%d jobs on %d workers" % (len(jobs), CFG.n_jobs))

res = Parallel(n_jobs=CFG.n_jobs, verbose=5)(
    delayed(run_arm)(a, b, pca_blocks, raw_blocks, y, groups, 0, CFG.folds,
                     CFG.pcs) for a, b in jobs)

rows = []
oof_store = {}
for arm, seed, oof, picked in res:
    d = clone_level(oof, y, groups)
    oof_store[arm] = oof
    rows.append(dict(arm=arm, auroc=auroc(d),
                     acc=float(((d["p"] > 0.5).astype(int) == d["y"]).mean()),
                     n_clones=len(d), picked=";".join(sorted(set(picked)))))
R = pd.DataFrame(rows)
R.to_csv(CFG.outprefix + "_arms.csv", index=False)
np.save(CFG.outprefix + "_oof.npy", np.vstack([oof_store[a] for a, _ in jobs]))
json.dump([a for a, _ in jobs], open(CFG.outprefix + "_armnames.json", "w"))


def family(prefix):
    v = R[R.arm.str.startswith(prefix)]["auroc"].values
    return (np.mean(v), np.std(v), len(v)) if len(v) else (np.nan, np.nan, 0)


summary = []
for name, pref in [
        ("B0_part1_only", "B0_part1_only"),
        ("B1_part1_plus_part2", "B1_part1_plus_part2"),
        ("B2_part1_plus_unspliced", "B2_part1_plus_unspliced"),
        ("N2_state_plus_thincopy", "N2_state_plus_thincopy"),
        ("N1_vel_permuted", "N1_smoothed_plus_vel_perm"),
        ("D_udepth", "D_udepth")]:
    m, s, n = family(pref)
    summary.append(dict(family=name, auroc_mean=m, auroc_sd=s, n_runs=n))
for f in FRACS:
    m, s, n = family("D_frac%g" % f)
    summary.append(dict(family="D_frac%g" % f, auroc_mean=m, auroc_sd=s,
                        n_runs=n))
for a in ("V2_smoothed_plus_vel_dyn", "A4_state_smoothed", "A2_state_spliced",
          "A6_state_plus_unspliced"):
    summary.append(dict(family=a, auroc_mean=float(R[R.arm == a].auroc.iloc[0]),
                        auroc_sd=np.nan, n_runs=1))
SU = pd.DataFrame(summary)
SU.to_csv(CFG.outprefix + "_summary.csv", index=False)
print(SU.to_string())

b0, b1, b2 = (family("B0_part1_only")[0], family("B1_part1_plus_part2")[0],
              family("B2_part1_plus_unspliced")[0])
print("\nincrement of an independent same-depth abundance copy: %+.4f"
      % (b1 - b0))
print("increment of the unspliced layer in its place:            %+.4f"
      % (b2 - b0))
print("difference:                                               %+.4f"
      % (b2 - b1))
nm, nsd, _ = family("N1_smoothed_plus_vel_perm")
v2 = float(R[R.arm == "V2_smoothed_plus_vel_dyn"].auroc.iloc[0])
a4 = float(R[R.arm == "A4_state_smoothed"].auroc.iloc[0])
print("\nvelocity increment over the smoothed state: %+.4f" % (v2 - a4))
print("permuted velocity increment: %+.4f (sd %.4f over %d permutations)"
      % (nm - a4, nsd, CFG.perms))
say("done")
