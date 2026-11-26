"""Figure 1 (page one): what is compared and what comes out.

(a) A dynamical method: early cells and two mature populations on a neighbour graph; each cell
    carries an inferred direction, and an early cell's fate is predicted by following it.
(b) The opponent: the same cells and graph without direction; a random walk from the early cell is
    absorbed in the mature populations, which gives its fate prediction.
    Panels (a) and (b) are a schematic drawn from a fixed random layout, not data.
(c) Data: every method's gain over the walk (clone-level AUROC minus the connectivity kernel) on
    the three lineage-traced data sets, one dot per method coloured by family, from
    calibrated_table.csv (99); the wash marks the gain that true dynamics carrying fate add in
    simulation at the walk's level on these data (as in 99b). Drawn at the printed size (figstyle.py).

Usage: 113_fig_concept.py <calibrated_table.csv> <band_lo> <band_hi> <out.pdf>
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import figstyle as fs  # noqa: E402

TAB, LO, HI, OUT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
fs.apply()
EARLY, FATE_A, FATE_B, EDGE = "#c9c7c1", fs.INK, "#8f8c86", "#e6e4df"

# ---------- schematic layout (fixed seed) ----------
rng = np.random.default_rng(3)
n_e, n_m = 70, 45
t = rng.uniform(0, 1, n_e)
early = np.c_[t * 1.6, rng.normal(0, 0.28, n_e)]
s = rng.uniform(0, 1, n_m)
neu = np.c_[1.7 + s * 1.3, 0.35 + s * 0.9 + rng.normal(0, 0.12, n_m)]
s = rng.uniform(0, 1, n_m)
mono = np.c_[1.7 + s * 1.3, -0.35 - s * 0.9 + rng.normal(0, 0.12, n_m)]
X = np.r_[early, neu, mono]
kind = np.r_[np.zeros(n_e), np.ones(n_m), 2 * np.ones(n_m)].astype(int)
tree = cKDTree(X)
_, nb = tree.query(X, k=6)
edges = {(min(i, j), max(i, j)) for i in range(len(X)) for j in nb[i, 1:]}
start = int(np.argmin(np.abs(early[:, 0] - 0.9) + np.abs(early[:, 1] - 0.05)))
col = np.array([EARLY, FATE_A, FATE_B])[kind]


def base(ax):
    for i, j in edges:
        ax.plot(X[[i, j], 0], X[[i, j], 1], color=EDGE, lw=0.35, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], s=6, c=col, edgecolors="white", linewidths=0.25, zorder=3)
    ax.scatter(*X[start], s=30, c="white", edgecolors=fs.INK, linewidths=0.9, zorder=6)
    ax.set_xlim(-0.25, 3.25)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(3.15, 1.38, "neutrophil", color=FATE_A, ha="right", fontsize=6.5)
    ax.text(3.15, -1.5, "monocyte", color=FATE_B, ha="right", fontsize=6.5)


def bar(ax, p_a, label, x=-0.15, y=-1.42, w=0.95, h=0.11):
    ax.add_patch(plt.Rectangle((x, y), w * p_a, h, color=FATE_A, lw=0, zorder=6))
    ax.add_patch(plt.Rectangle((x + w * p_a, y), w * (1 - p_a), h, color=FATE_B, lw=0, zorder=6))
    ax.text(x, y + h + 0.06, label, fontsize=6.0, color=fs.INK2, ha="left", va="bottom")


fig = plt.figure(figsize=(fs.TEXT, 1.95))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])

# (a) method with inferred directions
ax = fig.add_subplot(gs[0])
base(ax)
v = np.zeros_like(X)
v[kind == 0] = np.c_[np.full(n_e, 0.2), 0.05 * rng.normal(size=n_e)]
v[kind == 1] = np.c_[np.full(n_m, 0.15), np.full(n_m, 0.1)]
v[kind == 2] = np.c_[np.full(n_m, 0.15), np.full(n_m, -0.1)]
for i in rng.choice(len(X), 55, replace=False):
    ax.add_patch(FancyArrowPatch(X[i], X[i] + v[i], arrowstyle="-|>", mutation_scale=3.5, lw=0.45,
                                 color=fs.MUTED, zorder=4))
path = [X[start]]
p = X[start].copy()
for k in range(9):
    p = p + np.array([0.24, 0.05 + 0.012 * k])
    path.append(p.copy())
path = np.array(path)
ax.plot(path[:, 0], path[:, 1], color=fs.INK, lw=1.0, zorder=5)
bar(ax, 0.82, "predicted fate")
fs.panel(ax, "a", "a method follows its direction")

# (b) walk without direction on the same graph
ax = fig.add_subplot(gs[1])
base(ax)
adj = {i: [] for i in range(len(X))}
for i, j in edges:
    adj[i].append(j)
    adj[j].append(i)
wr = np.random.default_rng(11)
for r in range(3):
    cur, walk = start, [start]
    for _ in range(120):
        cur = int(wr.choice(adj[cur]))
        walk.append(cur)
        if kind[cur] != 0 and (X[cur, 0] > 2.5):
            break
    W = X[walk]
    ax.plot(W[:, 0], W[:, 1], color=fs.INK, lw=0.5, alpha=0.4, zorder=4)
bar(ax, 0.64, "absorption probability")
fs.panel(ax, "b", "the walk ignores direction")

# (c) data: gain over the walk, 48 pairs, coloured by family
ax = fig.add_subplot(gs[2])
T = pd.read_csv(TAB)
order = ["LARRY", "CellTag-multi haematopoiesis", "CellTag-multi reprogramming"]
short = {"LARRY": "LARRY", "CellTag-multi haematopoiesis": "haematopoiesis",
         "CellTag-multi reprogramming": "reprogramming"}
ax.axvspan(LO, HI, color=fs.TRUE_WASH, lw=0, zorder=0)
fs.walk_line(ax)
for k, ds in enumerate(order):
    g = T[T.dataset == ds].sort_values("gain").reset_index(drop=True)
    yc = 2 - k
    # simple beeswarm: offset points that sit close on x
    ys = np.zeros(len(g))
    for i in range(len(g)):
        near = [ys[j] for j in range(i) if abs(g.gain[i] - g.gain[j]) < 0.012]
        for cand in (0, 0.17, -0.17, 0.34, -0.34):
            if all(abs(cand - q) > 0.1 for q in near):
                ys[i] = cand
                break
    for i, r in g.iterrows():
        fs.dot(ax, r.gain, yc + ys[i], fs.colour(r.method), ms=4)
    ax.text(-0.235, yc + 0.43, short[ds], fontsize=6.5, color=fs.INK2, va="center")
ax.set_yticks([])
ax.set_ylim(-0.6, 2.75)
ax.set_xlim(-0.24, 0.12)
ax.set_xticks([-0.2, -0.1, 0, 0.1])
ax.set_xticklabels(["−0.2", "−0.1", "0", "0.1"])
ax.set_xlabel("gain over the walk (clone-level AUROC)")
ax.text((LO + HI) / 2, 2.72, "true\ndynamics", ha="center", va="top", fontsize=6.5, color=fs.TRUE_TEXT,
        linespacing=0.95)
n_above = int((T.lo > 0).sum())
ax.text((LO + HI) / 2, -0.5, "%d of %d\nabove" % (n_above, len(T)), ha="center", va="bottom", fontsize=6.5,
        color=fs.INK)
fs.clean(ax, left=False)
fs.panel(ax, "c", "every method against the walk")
fs.save(fig, OUT)
print("wrote", OUT, "pairs", len(T), "above", n_above)
