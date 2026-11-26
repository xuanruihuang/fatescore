"""The real-time kernel family of 111: CellRank 2's RealTimeKernel on the moscot couplings, which is
itself a mixture of the within-day connectivity walk (weight conn_weight) and the between-day
couplings (the direction). conn_weight = 1 - w for w in {0.02, 0.05, 0.1, 0.2, 0.3, 0.5}, plus the
published default 0.2 (w = 0.8, the arm snap_rtk_ub) for reference. Object, couplings and
absorption exactly as 41_ot_fate.py --method moscot --realtime --tau-a 0.95 (the rtk_ub arm), whose
functions are imported. Writes <outdir>/nm_mix_RTK_w<w>.npy (neutrophil share, d2_names order).

Usage (env_ot): 111_mix_rtk.py <raw h5ad> <featdir> <outdir>
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

RAW, FD, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "ot41", os.path.join(os.path.dirname(os.path.abspath(__file__)), "41_ot_fate.py"))
ot41 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ot41)
say = ot41.say

B, d2 = ot41.build(RAW, FD, "time_info", "state_info")
pops = ot41.populations(FD, B.obs["state"].values)
ineu, imono = pops.index("Neutrophil"), pops.index("Monocyte")

from moscot.problems.time import TemporalProblem  # noqa: E402

tp = TemporalProblem(B).prepare(time_key="day", joint_attr="X_pca", policy="sequential")
tp = tp.solve(epsilon=1e-3, scale_cost="mean", rank=-1, threshold=1e-3, max_iterations=10000,
              tau_a=0.95, tau_b=1.0)
conv = {str(k): bool(p.solution.converged) for k, p in tp.problems.items()}
say("moscot solved: %s" % conv)
assert all(conv.values()), conv

log = dict(populations=pops, convergence=conv)
for w in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
    args = argparse.Namespace(conn_weight=1.0 - w)
    F = ot41.realtime_fate(B, d2, pops, tp, args)
    s = F[:, ineu] + F[:, imono]
    p = np.where(s > 0, F[:, ineu] / np.where(s > 0, s, 1.0), 0.5)
    np.save(os.path.join(OUT, "nm_mix_RTK_w%s.npy" % w), p[:, None].astype(np.float32))
    log["RTK_w%s" % w] = dict(args._rtk_info, mean_share=float(p.mean()))
    say("  RTK w=%s (conn_weight %.2f): mean share %.3f" % (w, 1.0 - w, p.mean()))
json.dump(log, open(os.path.join(OUT, "mix_rtk_log.json"), "w"), indent=1, default=str)
say("done")
