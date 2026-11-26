"""Figure: every method's gain over the walk without direction (CellRank's connectivity kernel) on the
three lineage data sets, against what the simulation says true dynamics would add at the walk's level.

Reads out/calibrated_table.csv (99_calibrated_table.py) and out/mechanism_sim.csv
(98_mechanism_sim.py). The gold wash is the simulated gain of the velocity kernel fed the true
velocity, at the priming where the walk scores as it does on the real data (priming 0.2, AUROC
about 0.83), for a fate signal in the dynamics of 0.25 to 1 (mean over seeds). Methods are grouped
and coloured by family; missing methods (no splicing layer) are left blank. Drawn at the printed
size (figstyle.py).

Usage: 99b_fig_gain.py <calibrated_table.csv> <mechanism_sim.csv> <out.pdf>
"""
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import figstyle as fs  # noqa: E402

TAB, SIM, OUT = sys.argv[1:4]
fs.apply()
R = pd.read_csv(TAB)
S = pd.read_csv(SIM)
band = S[(S.priming == 0.2) & (S.delta > 0)].groupby("delta")["vk_minus_ck"].mean()
lo_b, hi_b = band.min(), band.max()
GROUPS = [("coupling", ["StationaryOT", "Waddington-OT", "CoSpar", "moscot"]),
          ("generative", ["PI-SDE", "PRESCIENT", "OT-CFM", "SB-CFM", "scNODE", "MIOFlow", "PACE", "scIMF", "USB"]),
          ("cellrank", ["CellRank real-time kernel", "CellRank velocity kernel (dynamical)",
                        "CellRank velocity + connectivity kernel", "CellRank pseudotime kernel",
                        "CellRank velocity kernel (steady state)"])]
SHORT = {"CellRank real-time kernel": "real-time kernel", "CellRank velocity kernel (dynamical)": "velocity kernel",
         "CellRank velocity + connectivity kernel": "velocity + connectivity",
         "CellRank pseudotime kernel": "pseudotime kernel",
         "CellRank velocity kernel (steady state)": "velocity, steady state"}
present = set(R.method)
rows, y = [], 0.0
for g, ms in GROUPS:
    for m in [m for m in ms if m in present]:
        rows.append((y, m))
        y -= 1
    y -= 0.7
ymin = y + 0.7
DS = [("LARRY", "LARRY"), ("CellTag-multi haematopoiesis", "Haematopoiesis"),
      ("CellTag-multi reprogramming", "Reprogramming")]
fig, axes = plt.subplots(1, 3, figsize=(fs.TEXT, 4.0), sharey=True, sharex=True)
for i, (ax, (ds, short)) in enumerate(zip(axes, DS)):
    D = R[R.dataset == ds].set_index("method")
    ax.axvspan(lo_b, hi_b, color=fs.TRUE_WASH, lw=0, zorder=0)
    fs.walk_line(ax)
    for yy, m in rows:
        if m not in D.index:
            continue
        r = D.loc[m]
        c = fs.colour(m)
        fs.interval(ax, r.lo, r.hi, yy, c)
        fs.dot(ax, r.gain, yy, c)
    k = D.iloc[0]
    fs.panel(ax, "abc"[i], short)
    ax.text(1.0, 1.005, "%d clones, walk %.2f" % (k.n_clones, k.kernel_auroc), transform=ax.transAxes,
            fontsize=6.5, color=fs.MUTED, va="bottom", ha="right")
    ax.set_xlim(-0.26, 0.12)
    ax.set_xticks([-0.2, -0.1, 0, 0.1])
    ax.set_xticklabels(["−0.2", "−0.1", "0", "0.1"])
    fs.clean(ax, left=False)
axes[0].text((lo_b + hi_b) / 2, rows[0][0] + 1.0, "true\ndynamics", ha="center", va="bottom", fontsize=6.5,
             color=fs.TRUE_TEXT, linespacing=0.95)
axes[0].text(-0.005, rows[0][0] + 1.0, "walk", ha="right", va="bottom", fontsize=6.5, color=fs.INK)
axes[1].set_xlabel("AUROC gain over the walk")
axes[0].set_yticks([r[0] for r in rows])
axes[0].set_yticklabels([SHORT.get(r[1], r[1]) for r in rows])
axes[0].set_ylim(ymin - 0.6, rows[0][0] + 2.2)
h, lab = fs.family_handles()
fig.legend(h, lab, loc="outside lower center", ncol=3, handlelength=0.8, columnspacing=1.8)
fs.save(fig, OUT)
print("band %.3f-%.3f; wrote %s" % (lo_b, hi_b, OUT))
