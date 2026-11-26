"""Each method's own answer to the two-fate question, as a single column.

Usage: 51_fate_share.py <featdir> [--pos Neutrophil --neg Monocyte]

The fate probabilities written by 41_ot_fate.py, 43_prescient_fate.py and
02c_cellrank.py have one column per terminal state, ten in the in vitro arm.
Scored on a two-fate task, eight of those columns are about other decisions,
and several are nearly constant: StandardScaler in cvlib.build then stretches a
handful of cells far out along them. Two of the methods lost more when their
ten columns were added to the expression block than ten columns of pure noise
cost (notes/velocity_fate_extension_log.md, section 17.18). This writes the
quantity each method actually predicts for the task -- the probability of the
positive fate among the two, p_pos / (p_pos + p_neg) -- as nm_<block>.npy, so
the increment can be rerun with the method's output in its most direct form.
Cells given zero probability of both fates get 0.5.

This is a sensitivity analysis added after the ten-column increment was seen;
it is reported as such.
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("featdir")
ap.add_argument("--pos", default="Neutrophil")
ap.add_argument("--neg", default="Monocyte")
a = ap.parse_args()


def states_of(name):
    lp = os.path.join(a.featdir, name + "_log.json")
    if os.path.exists(lp):
        pops = json.load(open(lp)).get("populations")
        if pops:
            return pops
    cr = os.path.join(a.featdir, "cellrank_log.json")
    if name.startswith("cr_") and os.path.exists(cr):
        L = json.load(open(cr))
        if name + "_states" in L:
            return L[name + "_states"]
        st = L.get("terminal_states")
        if isinstance(st, dict):
            return list(st)
    return None


done = {}
for p in sorted(glob.glob(os.path.join(a.featdir, "*.npy"))):
    name = os.path.basename(p)[:-4]
    if not (name.startswith("snap_") or name.startswith("cr_")):
        continue
    st = states_of(name)
    if st is None or a.pos not in st or a.neg not in st:
        print("skip %s: no state names" % name)
        continue
    X = np.load(p)
    assert X.shape[1] == len(st), (name, X.shape, len(st))
    pp, pn = X[:, st.index(a.pos)], X[:, st.index(a.neg)]
    s = pp + pn
    share = np.where(s > 0, pp / np.where(s > 0, s, 1.0), 0.5)
    np.save(os.path.join(a.featdir, "nm_" + name + ".npy"),
            share[:, None].astype(np.float32))
    done[name] = dict(states=st, both_zero=int((s <= 0).sum()),
                      mean_share=float(share.mean()))
json.dump(dict(pos=a.pos, neg=a.neg, blocks=done),
          open(os.path.join(a.featdir, "nm_log.json"), "w"), indent=1)
print(json.dumps({k: (v["both_zero"], round(v["mean_share"], 3))
                  for k, v in done.items()}))
