"""Calibration of the increment design for blocks as narrow as a method's output.

Usage: 50_lowdim_calibration.py <featdir> <labels.tsv> [--ncol 10]

19_positive_control.py calibrated the increment design with 50-component
blocks, the width of the splicing and velocity blocks. The fate probabilities
of the snapshot methods (41_ot_fate.py, 43_prescient_fate.py, 02c_cellrank.py)
are much narrower: one column per terminal state, ten in the in vitro arm. How
large an increment the design recovers depends on that width, so the snapshot
methods' increments can only be read against a calibration of the same width.

This writes synthetic blocks of that width into <featdir> as
cal_a<level>_s<seed>.npy, built exactly like calibration 1 of
19_positive_control.py: the clone's own neutrophil fraction along one random
direction, times a level, plus independent unit noise in every column. Level 0
is pure noise. 30_kernel_eval.py then scores them on the same folds both alone
(replacement design) and on top of an expression block (--add-to), so the
increment can be read against the information each block actually carries.

The blocks contain the label by construction. They are calibration material
and nothing else; nothing scored on them is a result about the data.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("labels")
ap.add_argument("--ncol", type=int, default=10)
ap.add_argument("--levels", default="0,0.1,0.2,0.3,0.5,0.8,1.2")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--prefix", default="cal_",
                help="file prefix, so blocks of different widths can coexist")
a = ap.parse_args()

T = pd.read_csv(a.labels, sep="\t")
names = json.load(open(os.path.join(a.featdir, "d2_names.json")))
assert list(T["cell"]) == names, "label table and feature rows disagree"
f = T["f_neu"].to_numpy(dtype=float)
# cells outside the task carry no label; give them the task mean so the block
# is defined on every row (the scorer drops them before anything is fitted)
fin = np.isfinite(f)
f = np.where(fin, f, np.nanmean(f))
W = np.random.default_rng(21).normal(size=(1, a.ncol))
sig = (f - f.mean())[:, None] * W
levels = [float(x) for x in a.levels.split(",")]
log = dict(ncol=a.ncol, levels=levels, seeds=a.seeds,
           rows=len(f), rows_with_label=int(fin.sum()))
for lev in levels:
    for s in range(a.seeds):
        r = np.random.default_rng(900 + 17 * s)
        X = lev * sig + r.normal(size=(len(f), a.ncol))
        np.save(os.path.join(a.featdir, "%sa%g_s%d.npy" % (a.prefix, lev, s)),
                X.astype(np.float32))
json.dump(log, open(os.path.join(a.featdir, a.prefix + "log.json"), "w"), indent=1)
print(json.dumps(log))
