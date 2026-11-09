"""One figure style for every notebook.

Colours follow the job they do:
* sample identity (categorical): fixed colour per sample type, never re-assigned
* magnitude (sequential): one blue ramp, light -> dark
Text is always neutral ink, never the series colour.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots 1-3 of a colour-vision-deficiency-checked palette.
SAMPLE_COLORS = {"Plasma": "#2a78d6", "Serum": "#eb6834", "tQC": "#1baf7a"}
SAMPLE_ORDER = ["Plasma", "Serum", "tQC"]

INK = "#0b0b0b"        # primary text
INK_MUTED = "#52514e"  # secondary text, annotations
GRID = "#e1e0d9"       # hairline gridlines
AXIS = "#c3c2b7"       # axis lines / baselines
SURFACE = "#fcfcfb"    # figure background
FLAG = "#d03b3b"       # reserved for QC flags only (status "critical")

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def set_style():
    """Apply the shared matplotlib style."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 5,
    })


def sequential_cmap():
    """Single-hue blue colour map for magnitudes (e.g. a correlation heatmap)."""
    return mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)


def scatter_by_type(ax, x, y, sample_type, labels=None, size=14, alpha=0.85):
    """Scatter coloured by sample type, drawn in a fixed order with a legend."""
    labels = labels or {"Plasma": "Plasma", "Serum": "Serum", "tQC": "Pooled QC"}
    for stype in SAMPLE_ORDER:
        sel = (sample_type == stype).to_numpy()
        if sel.any():
            ax.scatter(x[sel], y[sel], s=size, alpha=alpha, color=SAMPLE_COLORS[stype],
                       edgecolor=SURFACE, linewidth=0.5, label=labels[stype])
    return ax


def save_figure(fig, name, folder="../figures"):
    """Save to /figures as PNG (for the README) and return the path."""
    path = Path(folder) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path
