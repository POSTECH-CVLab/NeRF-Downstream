"""
Relevant constants and configurations
"""
import os

import matplotlib.style

# Data
DATA_FORMAT = os.environ.get("DATA_FORMAT", "channels_first")
DATA_NUM_WORKERS = int(os.environ.get("DATA_NUM_WORKERS", "2"))

# Opinionated default plotting styles
import matplotlib as mpl

DEFAULT_FIGSIZE = 8
MARKERS = "oxP.X"  # TODO: Update
DEFAULT_LINEWIDTH = 3
DEFAULT_FONTSIZE = 22
mpl.style.use("seaborn-colorblind")
mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams.update(
    {
        "font.size": 14,
        "lines.linewidth": 4,
        "figure.figsize": (DEFAULT_FIGSIZE, DEFAULT_FIGSIZE / 1.61),
    }
)
mpl.rcParams["grid.color"] = "k"
mpl.rcParams["grid.linestyle"] = ":"
mpl.rcParams["errorbar.capsize"] = 2
mpl.rcParams["image.cmap"] = "cividis"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["lines.markersize"] = 6
mpl.rcParams["lines.marker"] = None
mpl.rcParams["axes.grid"] = True
COLORS = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
mpl.rcParams.update(
    {
        "font.size": DEFAULT_FONTSIZE,
        "lines.linewidth": DEFAULT_LINEWIDTH,
        "legend.fontsize": DEFAULT_FONTSIZE,
        "axes.labelsize": DEFAULT_FONTSIZE,
        "xtick.labelsize": DEFAULT_FONTSIZE,
        "ytick.labelsize": DEFAULT_FONTSIZE,
        "figure.figsize": (7, 7.0 / 1.4),
    }
)
