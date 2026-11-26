"""Each method scored the way the published LARRY benchmarks score it: no classifier.

Usage: 52_direct_score.py <featdir> <labels.tsv> <cloneval.csv> <out.csv>
       [--pos P --neg N --blocks a,b,...]

The published LARRY benchmarks (PRESCIENT, Chreode) score a method's predicted share
of the positive fate, p_pos / (p_pos + p_neg), directly against the clone's
observed fate. This does the same with the nm_<block>.npy columns written by
51_fate_share.py: the share is averaged over a clone's early cells and the
clone-level AUROC is computed with no fitting at all, so no gate and no
classifier is involved. Each is then compared, paired over clones, with the
expression classifiers' repeat-0 out-of-fold clone scores, read from a
<prefix>_cloneval.csv written by 30_kernel_eval.py --dump-clones. The
bootstrap uses the same 2000 resamples over clones and seed as the scorer.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("cloneval")
ap.add_argument("out")
ap.add_argument("--pos", default="Neutrophil")
ap.add_argument("--neg", default="Monocyte")
ap.add_argument("--min-sis", type=int, default=3)
ap.add_argument("--blocks", default="snap_statot,snap_wot,snap_moscot,"
                "snap_prescient,cr_velo,cr_pseudo")
ap.add_argument("--refs", default="K_raw_spliced,K_raw_matrix")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=13)
a = ap.parse_args()

T = pd.read_csv(a.labels, sep="\t")
sel = ((T["n_sis"] >= a.min_sis) & T["dom"].isin([a.pos, a.neg])).values
y = (T.loc[sel, "dom"].values == a.pos).astype(int)
g = T.loc[sel, "clone"].values
C = pd.read_csv(a.cloneval).set_index("clone")
yy = C["y"].values
rng = np.random.default_rng(a.seed)
idx = [rng.integers(0, len(yy), len(yy)) for _ in range(a.boot)]
idx = [j for j in idx if yy[j].min() != yy[j].max()]
rows = []
for k in a.blocks.split(","):
    x = np.load("%s/nm_%s.npy" % (a.featdir, k))[sel, 0]
    s = pd.DataFrame(dict(p=x, g=g)).groupby("g")["p"].mean()
    yk = pd.DataFrame(dict(y=y, g=g)).groupby("g")["y"].first()
    assert (yk.loc[C.index].values == yy).all(), "labels disagree with cloneval"
    sa = s.loc[C.index].values
    row = dict(block=k, direct_auroc=round(roc_auc_score(yy, sa), 4),
               n_clones=len(yy))
    for r in a.refs.split(","):
        sb = C[r].values
        d0 = roc_auc_score(yy, sa) - roc_auc_score(yy, sb)
        bs = [roc_auc_score(yy[j], sa[j]) - roc_auc_score(yy[j], sb[j])
              for j in idx]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        row.update({"diff_vs_" + r: round(d0, 4), "lo_vs_" + r: round(lo, 4),
                    "hi_vs_" + r: round(hi, 4)})
    rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(a.out, index=False)
print(R.to_string(index=False))
