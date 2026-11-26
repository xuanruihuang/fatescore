"""Figure: the same learned models on two tests. x: share of the gap from independent pairing (0%)
to the floor (100%) closed on a held-out time point (mean, range over seeds; 106/110 summary);
y: gain over the connectivity kernel on clonal fate (clone-level AUROC, 95% paired bootstrap; 99).
One panel per data set with both tests (LARRY, CellTag-multi reprogramming). Dashed line: exact
optimal-transport interpolation on the held-out day.

Usage: 114_fig_heldout_fate.py <calibrated_table.csv> <heldout_summary_110.csv> <out.pdf>
"""
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TAB, HO, OUT = sys.argv[1:4]
import figstyle as fs  # noqa: E402
fs.apply()
# calibrated-table name -> held-out summary name (PACE: the fit-chosen ipsc preset on both data sets)
MAP = {"scNODE": "scNODE", "OT-CFM": "otcfm", "SB-CFM": "sbcfm", "PI-SDE": "PISDE", "scIMF": "scIMF",
       "MIOFlow": "MIOFlow", "USB": "usb", "PACE": "pace_ipsc"}
PANELS = [("LARRY", "larry", "a", "LARRY"), ("CellTag-multi reprogramming", "ctr", "b", "Reprogramming")]
OFF = {("LARRY", "SB-CFM"): (-5, 3, "right"), ("LARRY", "OT-CFM"): (5, 4, "left"),
       ("LARRY", "scNODE"): (6, -3, "left"),
       ("CellTag-multi reprogramming", "OT-CFM"): (6, -11, "left"),
       ("CellTag-multi reprogramming", "SB-CFM"): (3, 5, "left")}
T = pd.read_csv(TAB)
H = pd.read_csv(HO)
GEN = fs.FAM_COLOUR["generative"]

fig, axs = plt.subplots(1, 2, figsize=(fs.TEXT, 2.4))
for ax, (ds, hk, letter, title) in zip(axs, PANELS):
    g = T[T.dataset == ds].set_index("method")
    h = H[H.dataset == hk].set_index("method")
    ot = 100 * h.loc["ot", "gap_closed_mean"]
    lo_x = min(100 * h.loc[list(MAP.values()), "gap_closed_min"].min(), ot) - 8
    hi_x = max(100 * h.loc[list(MAP.values()), "gap_closed_max"].max(), ot) + 16
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-0.27, 0.07)
    ax.fill_between([0, hi_x], 0, 0.07, color=fs.TRUE_WASH, lw=0, zorder=0)
    fs.walk_line(ax, 0, horizontal=True)
    ax.axvline(0, color=fs.INK, lw=0.5, zorder=1)
    ax.axvline(ot, color=fs.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for m, k in MAP.items():
        x, y = 100 * h.loc[k, "gap_closed_mean"], g.loc[m, "gain"]
        fs.interval(ax, 100 * h.loc[k, "gap_closed_min"], 100 * h.loc[k, "gap_closed_max"], y, GEN)
        fs.interval(ax, g.loc[m, "lo"], g.loc[m, "hi"], x, GEN, horizontal=False)
        fs.dot(ax, x, y, GEN, ms=5)
        dx, dy, ha = OFF.get((ds, m), (5, 3, "left"))
        ax.annotate(m, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=fs.INK, ha=ha)
    ax.text(hi_x - 1, 0.035, "passes both tests", ha="right", va="center", fontsize=6.5, color=fs.TRUE_TEXT)
    ax.text(lo_x + 1, 0.005, "walk", ha="left", va="bottom", fontsize=6.5, color=fs.INK)
    ax.text(1.0, -0.266, "independent\npairing", ha="left", va="bottom", fontsize=6.0, color=fs.INK2,
            linespacing=0.95)
    ax.text(ot + 1.0, 0.065, "exact OT", ha="left", va="top", fontsize=6.0, color=fs.MUTED)
    ax.set_xlabel("held-out time point: gap closed (%)")
    fs.panel(ax, letter, title)
    fs.clean(ax)
axs[0].set_ylabel("clonal fate: gain over the walk")
fs.save(fig, OUT)
print("wrote", OUT)
