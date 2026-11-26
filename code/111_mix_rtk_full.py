"""The real-time kernel family of 111 on any data set: same object, couplings and absorption as
41_ot_fate.py --method moscot --realtime --tau-a 0.95, at conn_weight = 1 - w for
w in {0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8}. Writes, per w, the early cells' full fate-probability
matrix <outdir>/mix_RTK_w<w>.npy with <outdir>/mix_RTK_w<w>_log.json listing the populations, the
layout 97_fate_score.py reads.

Usage (env_ot): 111_mix_rtk_full.py <h5ad> <featdir> <outdir>
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

H5, FD, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "ot41", os.path.join(os.path.dirname(os.path.abspath(__file__)), "41_ot_fate.py"))
ot41 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ot41)
say = ot41.say

B, d2 = ot41.build(H5, FD, "time_info", "state_info")
pops = ot41.populations(FD, B.obs["state"].values)
from moscot.problems.time import TemporalProblem  # noqa: E402

tp = TemporalProblem(B).prepare(time_key="day", joint_attr="X_pca", policy="sequential")
tp = tp.solve(epsilon=1e-3, scale_cost="mean", rank=-1, threshold=1e-3, max_iterations=10000,
              tau_a=0.95, tau_b=1.0)
conv = {str(k): bool(p.solution.converged) for k, p in tp.problems.items()}
say("moscot solved: %s" % conv)
assert all(conv.values()), conv
for w in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
    args = argparse.Namespace(conn_weight=1.0 - w)
    F = ot41.realtime_fate(B, d2, pops, tp, args)
    rs = F.sum(1, keepdims=True)
    F = F / np.where(rs == 0, 1, rs)
    np.save(os.path.join(OUT, "mix_RTK_w%s.npy" % w), F.astype(np.float32))
    json.dump(dict(populations=pops, conn_weight=1.0 - w, convergence=conv,
                   realtime_info=args._rtk_info), open(os.path.join(OUT, "mix_RTK_w%s_log.json" % w), "w"),
              indent=1, default=str)
    say("  RTK w=%s written" % w)
say("done")
