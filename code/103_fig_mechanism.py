"""Figure: the simulated positive control and its error axes (98_mechanism_sim.py outputs).

(a) Gain of CellRank's velocity kernel driven by the true velocity over the connectivity kernel on
    the same graph, against how much fate the dynamics carry (delta), one line per priming (how
    much fate the progenitors' present state carries; the kernel's AUROC in the legend). Shaded:
    the real data sets' minimum detectable differences (min to max over method pairs).
(b) Priming 0.2 (kernel ~0.83, the real data sets' level): the same gain as the velocity is mixed
    with another cell's velocity (noise 0 to 1), one line per delta.
(c) Priming 0.2: a uniform degradation-like term added (decay, in units of the velocity's norm),
    one line per delta, including dynamics that favour the wrong fate (delta < 0).
Means over seeds; bars = range of the per-seed 95% intervals.

Usage: 103_fig_mechanism.py <mechanism_sim.csv> <mechanism_sim_noise.csv> <mechanism_sim_wrong.csv>
       <resolution.csv> <out.pdf>
"""
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIM, NOISE, WRONG, RES, OUT = sys.argv[1:6]
import figstyle as fs  # noqa: E402
fs.apply()
DIV = {-0.5: fs.DIV_NEG, 0.0: fs.DIV_ZERO, 0.5: fs.DIV_POS_LIGHT, 1.0: fs.DIV_POS}
S, N, W = pd.read_csv(SIM), pd.read_csv(NOISE), pd.read_csv(WRONG)
R = pd.read_csv(RES)
mdd_lo, mdd_hi = R.mdd_min.min(), R.mdd_max.max()


def agg(d, by):
    return d.groupby(by).agg(g=("vk_minus_ck", "mean"), lo=("lo", "min"), hi=("hi", "max"),
                             k=("auroc_connectivity_kernel", "mean")).reset_index()


def line(ax, x, d, color, label, lw=1.2, dy=0.0, at_start=False):
    ax.fill_between(d[x], d.lo, d.hi, color=color, alpha=0.18, lw=0, zorder=2)
    ax.plot(d[x], d.g, "-", color=color, lw=lw, zorder=3, solid_capstyle="round")
    ax.plot(d[x], d.g, "o", ms=3.2 if lw < 1.5 else 3.8, mfc=color, mec="white", mew=0.5, zorder=4)
    if at_start:
        xe, ye = d[x].iloc[0], d.g.iloc[0]
        ax.text(xe + 0.04, ye + dy + 0.006, label, va="bottom", ha="left", fontsize=6.5, color=fs.INK2)
    else:
        xe, ye = d[x].iloc[-1], d.g.iloc[-1]
        ax.text(xe + 0.03 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ye + dy, label, va="center", ha="left",
                fontsize=6.5, color=fs.INK if lw >= 1.5 else fs.INK2)


fig, axs = plt.subplots(1, 3, figsize=(fs.TEXT, 2.15), sharey=True)
for ax in axs:
    ax.axhspan(mdd_lo, mdd_hi, color=fs.BAND, lw=0, zorder=0)
    fs.walk_line(ax, 0, horizontal=True)
    ax.set_ylim(-0.07, 0.165)
axs[0].set_xlim(-0.04, 1.32)
axs[1].set_xlim(-0.04, 1.32)
axs[2].set_xlim(-0.08, 2.65)
a = agg(S, ["priming", "delta"])
DY0 = {0.0: 0.009, 0.1: -0.003, 0.2: 0.0, 0.4: 0.0}
for i, (p, d) in enumerate(a.groupby("priming")):
    line(axs[0], "delta", d, fs.RAMP[i], "priming %.1f" % p, lw=1.8 if abs(p - 0.2) < 1e-9 else 1.0,
         dy=DY0.get(round(p, 1), 0))
axs[0].set_xlabel("fate carried by the dynamics, $\\delta$")
fs.panel(axs[0], "a", "true dynamics")
n0 = S[(S.priming == 0.2) & S.delta.isin(N.delta.unique())].assign(noise=0.0)
b = agg(pd.concat([n0, N[N.priming == 0.2]]), ["delta", "noise"])
DY1 = {0.0: 0.004, 0.5: 0.0, 1.0: -0.006}
for dl, d in b.groupby("delta"):
    line(axs[1], "noise", d, DIV.get(dl, fs.INK), "$\\delta$ = %.1f" % dl, at_start=True)
axs[1].set_xlabel("random error (share of velocity replaced)")
fs.panel(axs[1], "b", "random error")
w = W[(W.priming == 0.2) & (W.noise == 0)]
c = agg(w, ["delta", "decay"])
for dl, d in c.groupby("delta"):
    line(axs[2], "decay", d, DIV.get(dl, fs.INK), ("$\\delta$ = %+.1f" % dl) if dl else "$\\delta$ = 0")
axs[2].set_xlabel("uniform decay term ($\\times$ velocity norm)")
fs.panel(axs[2], "c", "decay term, wrong-fate dynamics")
for ax in axs:
    fs.clean(ax)
axs[1].text(1.3, mdd_hi - 0.003, "resolution\nof the data", ha="right", va="top", fontsize=6.0, color=fs.MUTED,
            linespacing=0.95)
axs[2].text(-0.04, -0.066, "below the walk", ha="left", va="bottom", fontsize=6.5, color=fs.DIV_NEG)
axs[1].text(-0.02, -0.004, "walk", ha="left", va="top", fontsize=6.5, color=fs.INK)
axs[0].set_ylabel("AUROC gain over the walk")
fs.save(fig, OUT)
print(a.round(3).to_string(index=False)); print(b.round(3).to_string(index=False)); print(c.round(3).to_string(index=False))
