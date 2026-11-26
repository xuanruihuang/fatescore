"""Pick, per data set, the PACE preset with the better distribution fit (as reported by 110:
ipsc_liu on LARRY and reprogramming, op_cite on haematopoiesis) and write it as snap_pace_fit,
so that one key names PACE in every table.

Usage: 110b_pace_fit_pick.py <out_dir>
"""
import json
import os
import shutil
import sys

OUT = sys.argv[1]
PICK = {"feat_all": "snap_pace_ipsc", "feat_ctm": "snap_pace", "feat_ctr": "snap_pace_ipsc"}
for d, src in PICK.items():
    D = os.path.join(OUT, d)
    shutil.copyfile(os.path.join(D, src + ".npy"), os.path.join(D, "snap_pace_fit.npy"))
    j = json.load(open(os.path.join(D, src + "_log.json")))
    j["method"] = "snap_pace_fit"
    j["source"] = src
    j["note"] = "PACE preset chosen by distribution fit (110)"
    json.dump(j, open(os.path.join(D, "snap_pace_fit_log.json"), "w"))
    if os.path.exists(os.path.join(D, "nm_" + src + ".npy")):
        shutil.copyfile(os.path.join(D, "nm_" + src + ".npy"), os.path.join(D, "nm_snap_pace_fit.npy"))
    print(d, "<-", src)
