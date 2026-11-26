"""Figure: the published LARRY fate benchmark on its own test cells (Weinreb et al. 2020, Fig. 5;
the 335 cells also used by PRESCIENT), every method's Pearson r as a difference from the
connectivity kernel on the same cells.

Reads out/weinreb_test_scores.csv (101_weinreb_test_set.py). Filled markers: scored here on the
298 test cells present in this object, with a 95% bootstrap interval over clones of the
difference from the kernel. Open marker: PRESCIENT as published (all 335 cells, its own
pipeline; with and without proliferation weights; bar = +- SD over 5 training seeds), not paired.
Grey band: the kernel's 18 graph settings.

Usage: 101b_fig_published.py <weinreb_test_scores.csv> <out.pdf>
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TAB, OUT = sys.argv[1:3]
import figstyle as fs  # noqa: E402
fs.apply()
R = pd.read_csv(TAB).set_index("method")
REF = "cr_conn_n30_p30_d1"
k_r = R.loc[REF, "r"]
kset = R.loc[[i for i in R.index if i.startswith("cr_conn_n")], "r"] - k_r
PRESCIENT_PUB = {"growth": (0.399, 0.025), "none": (0.196, 0.020)}   # Yeo et al. 2021, mean +- SD over 5 seeds

# published block: (label, key, family-of or None)
PUB = [("upper bound (held-out clones)", "published: held-out clonal (upper bound)", None),
       ("PRESCIENT, weighted", "pub:growth", "PRESCIENT"),
       ("PRESCIENT, unweighted", "pub:none", "PRESCIENT"),
       ("FateID", "published: FateID", None),
       ("PBA", "published: PBA", None),
       ("Waddington-OT", "published: WOT", "Waddington-OT")]
OURS = {"coupling": [("StationaryOT", "snap_statot"), ("Waddington-OT", "snap_wot"), ("moscot", "snap_moscot"),
                     ("CoSpar", "snap_cospar_state")],
        "generative": [("PRESCIENT", "snap_prescient"), ("OT-CFM", "snap_otcfm_g50k"), ("SB-CFM", "snap_sbcfm_g50k"),
                       ("scNODE", "snap_scnode"), ("scIMF", "snap_scimf"), ("PI-SDE", "snap_pisde"),
                       ("MIOFlow", "snap_mioflow"), ("USB", "snap_usb"), ("PACE", "snap_pace_fit")],
        "cellrank": [("real-time kernel", "snap_rtk_ub"), ("pseudotime kernel", "cr_pseudo"),
                     ("velocity kernel", "cr_velo"), ("velocity, steady state", "cr_lad_velocity_ss"),
                     ("velocity + connectivity", "cr_vkck")]}
rows, y = [], 0.0
for lab, key, fam_of in PUB:
    c = fs.colour(fam_of) if fam_of else fs.INK2
    rows.append((y, lab, key, c, True))
    y -= 1
y_split = y + 0.35
y -= 0.9
for fam, items in OURS.items():
    items = [(a, b) for a, b in items if b in R.index]
    items.sort(key=lambda t: -R.loc[t[1], "r"])           # within a family, best first
    for lab, key in items:
        rows.append((y, lab, key, fs.FAM_COLOUR[fam], False))
        y -= 1
    y -= 0.55
fig, ax = plt.subplots(figsize=(fs.COL, 4.5))
ax.axvspan(kset.min(), kset.max(), color=fs.BAND, lw=0, zorder=0)
fs.walk_line(ax)
XR = 0.37
for yy, lab, key, c, hollow in rows:
    if key.startswith("pub:"):
        pm, ps = PRESCIENT_PUB[key[4:]]
        d = pm - k_r
        fs.interval(ax, d - ps, d + ps, yy, c)
        fs.dot(ax, d, yy, c, hollow=True)
        val = pm
    else:
        r = R.loc[key]
        fs.interval(ax, r.r_lo, r.r_hi, yy, c)
        if key.startswith("published: held-out"):
            fs.dot(ax, r.r_minus_kernel, yy, fs.INK, hollow=True, marker="D", ms=3.6)
        else:
            fs.dot(ax, r.r_minus_kernel, yy, c, hollow=hollow)
        val = r.r
    ax.text(XR, yy, "%.2f" % val, va="center", ha="right", fontsize=6.5, color=fs.INK2)
ax.text(XR, rows[0][0] + 0.9, "$r$", va="bottom", ha="right", fontsize=6.5, color=fs.MUTED)
ax.text(-0.5, rows[0][0] + 0.9, "as published", va="bottom", ha="left", fontsize=6.5, color=fs.MUTED)
ax.text(-0.5, y_split - 0.75, "run here on the same cells", va="bottom", ha="left", fontsize=6.5, color=fs.MUTED)
ax.text(0.008, y + 0.25, "walk, $r$ = %.2f" % k_r, ha="left", va="center", fontsize=6.5, color=fs.INK)
ax.set_yticks([r[0] for r in rows])
ax.set_yticklabels([r[1] for r in rows])
ax.set_ylim(y - 0.2, rows[0][0] + 1.8)
ax.set_xlim(-0.5, XR + 0.01)
ax.set_xticks([-0.4, -0.2, 0, 0.2])
ax.set_xticklabels(["−0.4", "−0.2", "0", "0.2"])
ax.set_xlabel("Pearson $r$ minus the walk")
fs.clean(ax, left=False)
fs.save(fig, OUT)
print("kernel r %.3f, settings %.3f to %.3f; wrote %s" % (k_r, k_r + kset.min(), k_r + kset.max(), OUT))
