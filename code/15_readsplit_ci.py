"""Bootstrap interval for the read-splitting comparison.

Usage: 15_readsplit_ci.py <dn_prefix> <labels.tsv> <out.csv> [--pos P --neg N]

05_depth_and_null.py runs its arms over five independent splits of the spliced
reads and over twenty permutations of the velocity field. This script reads
back its out-of-fold predictions and bootstraps over clones the quantities the
mechanism claim rests on: what an independent equal-depth copy of the spliced
reads adds, what the unspliced layer adds in its place, and the difference
between the two. Within a bootstrap resample the AUROC is computed for each
seed separately and then averaged, so that several runs of the same arm are
never pooled into an ensemble, which would score better than any single run.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("dn_prefix")
ap.add_argument("labels")
ap.add_argument("out")
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--pos", default="Neutrophil")
ap.add_argument("--neg", default="Monocyte")
ap.add_argument("--restrict-pop", default="",
                help="must match the value given to 05_depth_and_null.py")
CFG = ap.parse_args()

names = json.load(open(CFG.dn_prefix + "_armnames.json"))
O = np.load(CFG.dn_prefix + "_oof.npy")
T = pd.read_csv(CFG.labels, sep="\t")
sel = (T["n_sis"] >= CFG.min_sis) & T["dom"].isin([CFG.pos, CFG.neg])
if CFG.restrict_pop:
    sel = sel & (T["pop"] == CFG.restrict_pop)
y = (T.loc[sel, "dom"].values == CFG.pos).astype(int)
groups = T.loc[sel, "clone"].values
clones = np.unique(groups)


def tab(p):
    return (pd.DataFrame(dict(p=p, y=y, g=groups)).groupby("g")
            .agg(p=("p", "mean"), y=("y", "first")).reindex(clones))


def auroc(d):
    return np.nan if d["y"].nunique() < 2 else roc_auc_score(d["y"], d["p"])


def family(prefix):
    """The out-of-fold predictions of every run whose name starts with
    `prefix`, one row per run."""
    idx = [i for i, n in enumerate(names) if n.startswith(prefix)]
    assert idx, prefix
    return O[idx]


FAM = {
    "B0_part1_only": "B0_part1_only",
    "B1_part1_plus_part2": "B1_part1_plus_part2",
    "B2_part1_plus_unspliced": "B2_part1_plus_unspliced",
    "N2_state_plus_thincopy": "N2_state_plus_thincopy",
    "N1_vel_permuted": "N1_smoothed_plus_vel_perm",
    "V2_smoothed_plus_vel_dyn": "V2_smoothed_plus_vel_dyn",
    "A4_state_smoothed": "A4_state_smoothed",
    "A2_state_spliced": "A2_state_spliced",
    "A6_state_plus_unspliced": "A6_state_plus_unspliced",
}
tabs = {k: [tab(p) for p in family(v)] for k, v in FAM.items()}
point = {k: float(np.mean([auroc(t) for t in ts])) for k, ts in tabs.items()}

NB = 2000
rng = np.random.default_rng(11)
boot = {k: np.empty(NB) for k in tabs}
for b in range(NB):
    take = rng.integers(0, len(clones), len(clones))
    for k, ts in tabs.items():
        boot[k][b] = np.mean([auroc(t.iloc[take]) for t in ts])


def row(label, a, b):
    d = boot[a] - boot[b]
    return dict(quantity=label, value=point[a] - point[b],
                lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5))


rows = [dict(quantity=k, value=point[k], lo=np.nanpercentile(boot[k], 2.5),
             hi=np.nanpercentile(boot[k], 97.5), n_runs=len(tabs[k]))
        for k in tabs]
rows += [
    row("independent equal-depth spliced copy added",
        "B1_part1_plus_part2", "B0_part1_only"),
    row("unspliced layer added in its place",
        "B2_part1_plus_unspliced", "B0_part1_only"),
    row("unspliced minus independent spliced copy",
        "B2_part1_plus_unspliced", "B1_part1_plus_part2"),
    row("unspliced added to the full spliced counts",
        "A6_state_plus_unspliced", "A2_state_spliced"),
    row("thinned copy of the same reads added",
        "N2_state_plus_thincopy", "A2_state_spliced"),
    row("dynamical velocity added to the smoothed state",
        "V2_smoothed_plus_vel_dyn", "A4_state_smoothed"),
    row("velocity shuffled within a state",
        "N1_vel_permuted", "A4_state_smoothed"),
    row("dynamical velocity minus the same velocities shuffled",
        "V2_smoothed_plus_vel_dyn", "N1_vel_permuted"),
]
R = pd.DataFrame(rows)
R.to_csv(CFG.out, index=False)
print(R.to_string(index=False))
