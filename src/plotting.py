"""One figure style for every notebook.

Colours follow the job they do:
* sample identity (categorical): fixed colour per sample type, never re-assigned
* magnitude (sequential): one blue ramp, light -> dark
Text is always neutral ink, never the series colour.
"""
from pathlib import Path

import numpy as np

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


def dot_scoreboard(ax, table, columns, colors, labels=None, xlabel="", highlight=None):
    """Dot plot comparing methods (rows of ``table``) on several metrics (``columns``).

    One row per method, one coloured dot per metric. Easier to read than a
    grouped bar chart when the question is "which method is lowest".

    highlight : optional method name drawn in bold (the chosen method)
    """
    labels = labels or {c: c for c in columns}
    y = np.arange(len(table))[::-1]                     # first method at the top
    for col in columns:
        ax.scatter(table[col], y, s=46, color=colors[col], edgecolor=SURFACE, linewidth=0.8,
                   label=labels[col], zorder=3)
    for yi in y:                                        # faint guide line per method
        ax.axhline(yi, color=GRID, lw=0.6, zorder=1)
    ax.set_yticks(y, table.index)
    if highlight is not None and highlight in table.index:
        for tick in ax.get_yticklabels():
            if tick.get_text() == highlight:
                tick.set_fontweight("bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    return ax


def volcano(ax, result, label=None, alpha=0.05, title=None):
    """Volcano plot of a ``stats.differential`` result: log2 fold change vs -log10 q.

    Points are coloured by call; on/off proteins are drawn as hollow markers so
    they are never confused with proteins tested without imputation.
    label : optional dict {protein: text} to annotate selected proteins
    """
    colors = {"up": "#eb6834", "down": "#2a78d6", "not significant": "#c3c2b7"}
    r = result[result["p"].notna()].copy()
    r["y"] = -np.log10(r["q"].clip(lower=1e-300))
    for call, color in colors.items():
        for tier, filled in [("tested", True), ("on/off", False)]:
            sel = (r["call"] == call) & (r["tier"] == tier)
            if sel.any():
                ax.scatter(r.loc[sel, "log2fc"], r.loc[sel, "y"], s=22 if filled else 30,
                           facecolor=color if filled else "none", edgecolor=color if not filled else SURFACE,
                           linewidth=1.2 if not filled else 0.5,
                           label=f"{call}{' (on/off)' if not filled else ''}")
    ax.axhline(-np.log10(alpha), color=INK_MUTED, lw=1, ls=":")
    ax.axvline(0, color=AXIS, lw=1)
    if label:
        # Alternate label offsets (above / below the point) so neighbouring labels do not overlap
        offsets = [(6, 4), (6, -11), (-38, 6)]
        for i, (prot, text) in enumerate(label.items()):
            if prot in r.index:
                ax.annotate(text, (r.loc[prot, "log2fc"], r.loc[prot, "y"]), xytext=offsets[i % 3],
                            textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 q (BH)")
    if title:
        ax.set_title(title)
    return ax
