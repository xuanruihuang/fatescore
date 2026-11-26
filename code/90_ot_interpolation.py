"""Held-out time-point interpolation, the validation optimal-transport methods
report, against baselines without dynamics.

Waddington-OT validates a coupling by interpolating an intermediate sampling
day from the days around it and measuring the distance to the cells actually
sampled then (Schiebinger et al. 2019). Its own validation routine also scores
references that carry no information about how cells change: an independent
coupling of the two end points ('R', random pairs), the first day alone ('F'),
the last day alone ('L'), and, as a noise floor, the distance between two
batches of the intermediate day itself ('P'; here the two LARRY wells). On LARRY,
day 4 is interpolated from days 2 and 6, in the 30-component local PCA of each
day pair that Waddington-OT uses, at its default settings.

Earth mover's distance on all cells is costly, so each run subsamples cells per
day (default 3,000) and interpolates as many cells; three seeds.

Usage: 90_ot_interpolation.py <h5ad> <featdir> <out.csv> [--per-day 3000 --seeds 3]
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("otfate", os.path.join(HERE, "41_ot_fate.py"))
otfate = importlib.util.module_from_spec(spec)
sys.argv_saved = sys.argv
spec.loader.exec_module(otfate)


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("out")
ap.add_argument("--per-day", type=int, default=3000)
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()

if not hasattr(ad, "read"):
    ad.read = ad.read_h5ad
import wot.ot  # noqa: E402

B, _ = otfate.build(a.h5ad, a.featdir, "time_info", "state_info")
if "Well" in B.obs:
    well = B.obs["Well"].astype(str).values
else:
    # CellTag-multi: the replicate (r1 / r2) is in the cell name, e.g. B4D12-RNA-r1-1-<barcode>
    import re
    well = np.array([(re.search(r"-(r\d)-", n) or re.search(r"(d5r\d)", n) or ["", "0"])[1]
                     for n in B.obs_names])
days = sorted(set(B.obs["day"].dropna().tolist()))
assert len(days) == 3, "held-out interpolation needs three sampling days: %s" % days
TRIPLET = tuple(days)
rows = []
for seed in range(a.seeds):
    rng = np.random.default_rng(900 + seed)
    keep = []
    for d in TRIPLET:
        idx = np.where(B.obs["day"].values == d)[0]
        keep.append(rng.choice(idx, size=min(a.per_day, idx.size), replace=False))
    S = B[np.sort(np.concatenate(keep))].copy()
    S.obs["covariate"] = well[np.sort(np.concatenate(keep))]
    say("seed %d: %s, wells at day 4: %s" % (seed, S.shape,
        pd.Series(S.obs.loc[S.obs["day"] == TRIPLET[1], "covariate"]).value_counts().to_dict()))
    model = wot.ot.OTModel(S, day_field="day", covariate_field="covariate")
    summ = wot.ot.compute_validation_summary(model, day_triplets=[TRIPLET],
                                             interp_size=a.per_day, compute_full_distances=True)
    summ["seed"] = seed
    rows.append(summ)
    say(summ.groupby("name")["distance"].mean().round(3).to_dict())
    pd.concat(rows).to_csv(a.out, index=False)
R = pd.concat(rows)
print(R.groupby(["full", "name"])["distance"].agg(["mean", "std", "count"]).round(3).to_string())
