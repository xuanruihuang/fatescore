"""One visual system for every figure of the ICML paper (notes/velocity_fate_figure_guide.md).

Colour means method family only, in three muted hues (slate, clay, celadon) chosen for colour-blind
distance; everything else is ink and greys. The walk without direction is the reference, drawn as a
thin ink line; where true dynamics would appear is a pale gold wash. Published values are open
markers, our runs filled. Figures are drawn at their final printed size (3.25 in column, 6.75 in
text width) with a constrained layout and saved without cropping, so text prints at 7 pt.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COL, TEXT = 3.25, 6.75
INK = "#3b3a37"
INK2 = "#5f5d58"
MUTED = "#93928e"
AXIS = "#b9b7b1"
BAND = "#e3e1dc"           # the walk over graph settings; detectable-difference band
TRUE_WASH = "#f6edd3"      # where true dynamics would appear
TRUE_TEXT = "#95753c"
FAM_COLOUR = {"coupling": "#4a6297", "generative": "#b06d47", "cellrank": "#68ada8"}
FAM_LABEL = {"coupling": "couplings", "generative": "learned generative models",
             "cellrank": "CellRank kernels"}
RAMP = ["#b1bfd1", "#8094ae", "#536b89", "#2e445f"]
DIV_NEG, DIV_ZERO, DIV_POS, DIV_POS_LIGHT = "#6b588d", "#93928e", "#95753c", "#c5ad86"
FAMILY = {
    "StationaryOT": "coupling", "Waddington-OT": "coupling", "moscot": "coupling", "CoSpar": "coupling",
    "PRESCIENT": "generative", "OT-CFM": "generative", "SB-CFM": "generative", "scNODE": "generative",
    "scIMF": "generative", "PI-SDE": "generative", "MIOFlow": "generative", "USB": "generative",
    "PACE": "generative",
    "CellRank real-time kernel": "cellrank", "CellRank velocity kernel (dynamical)": "cellrank",
    "CellRank velocity kernel (steady state)": "cellrank", "CellRank velocity + connectivity kernel": "cellrank",
    "CellRank pseudotime kernel": "cellrank", "Real-time kernel": "cellrank",
    "Velocity kernel (dynamical)": "cellrank", "Velocity kernel (steady state)": "cellrank",
    "Velocity + connectivity kernel": "cellrank", "Pseudotime kernel": "cellrank",
}


def colour(method):
    return FAM_COLOUR[FAMILY[method]]


def apply():
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "legend.frameon": False, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    })


def clean(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)


def panel(ax, letter, title=None):
    """Lowercase bold letter at the top left of the axes area, with an optional short title."""
    t = ax.set_title(("$\\bf{%s}$   " % letter if letter else "") + (title or ""), loc="left", pad=4,
                     fontsize=7, color=INK)
    return t


def dot(ax, x, y, c, ms=4, hollow=False, z=4, marker="o"):
    if hollow:
        ax.plot(x, y, marker, ms=ms, mfc="white", mec=c, mew=0.9, zorder=z)
    else:
        ax.plot(x, y, marker, ms=ms, mfc=c, mec="white", mew=0.5, zorder=z)


def interval(ax, a, b, at, c, horizontal=True, lw=1.0, alpha=0.55, z=3):
    xs, ys = ([a, b], [at, at]) if horizontal else ([at, at], [a, b])
    ax.plot(xs, ys, color=c, lw=lw, alpha=alpha, solid_capstyle="butt", zorder=z)


def walk_line(ax, x=0, horizontal=False):
    (ax.axhline if horizontal else ax.axvline)(x, color=INK, lw=0.8, zorder=2)


def family_handles(fams=("coupling", "generative", "cellrank")):
    return ([Line2D([], [], marker="o", ls="", ms=4, mfc=FAM_COLOUR[f], mec="white", mew=0.5) for f in fams],
            [FAM_LABEL[f] for f in fams])


def save(fig, out):
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)
