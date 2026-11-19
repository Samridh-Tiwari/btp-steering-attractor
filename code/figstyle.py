"""Shared figure style. Colours are fixed by ROLE and never change between figures."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8985"
GRID = "#e8e7e3"
SURF = "#ffffff"

OPT = "#2a78d6"      # optimal channels                (categorical slot 1, blue)
TEST = "#eb6834"     # channels under test: student fit / teacher value (slot 2, orange)
GO = "#1baf7a"       # teacher go channel              (slot 3, aqua)
RAND = "#9b9a95"     # random reference                (neutral, not a categorical slot)

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.edgecolor": INK3, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    })


def panel_label(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=INK, ha="right", va="bottom")
