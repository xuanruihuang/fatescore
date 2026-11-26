"""How fine a difference the calibrated scoring can resolve, per data set.

Reads out/calibrated_table.csv (99) and out/mechanism_sim.csv (98, for the band of gains that
true dynamics add in simulation at the real data sets' kernel level, as drawn in 99b). Writes one
row per data set: the minimum detectable difference (MDD = 0.714 x CI width of the paired gain)
over methods, the spread of the kernel over its 18 graph settings, and how many method pairs could
detect the low and the high end of the simulated true-dynamics gain.

Usage: 102_resolution.py <calibrated_table.csv> <band_lo> <band_hi> <out.csv>
(band_lo, band_hi as printed by 99b_fig_gain.py.)
"""
import sys

import pandas as pd

TAB, LO, HI, OUT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
t = pd.read_csv(TAB)
rows = []
for ds, g in t.groupby("dataset", sort=False):
    rows.append({"dataset": ds, "n_clones": int(g.n_clones.iloc[0]), "n_methods": len(g),
                 "kernel_auroc": g.kernel_auroc.iloc[0],
                 "kernel_settings_spread": g.kernel_max.iloc[0] - g.kernel_min.iloc[0],
                 "mdd_min": g.mdd.min(), "mdd_median": g.mdd.median(), "mdd_max": g.mdd.max(),
                 "detect_band_lo": int((g.mdd < LO).sum()), "detect_band_hi": int((g.mdd < HI).sum()),
                 "below": int((g.separated == "below").sum()), "above": int((g.separated == "above").sum())})
R = pd.DataFrame(rows)
R.to_csv(OUT, index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
print("all pairs: %d; MDD < %.3f in %d, < %.3f in %d; above the kernel %d"
      % (len(t), LO, (t.mdd < LO).sum(), HI, (t.mdd < HI).sum(), (t.separated == "above").sum()))
