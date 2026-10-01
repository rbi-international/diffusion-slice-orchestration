"""Shared figure style for the manuscript (Scientific Reports figure guidelines).

Sans-serif type (Arial, or the metrically identical Liberation Sans), lettering in
sentence case, lines at least 1 pt wide, white background, bold lowercase panel
letters. Each figure is written as vector PDF and EPS and as 600 dpi PNG and
LZW-compressed TIFF.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
MM = 1 / 25.4
INK, INK2, GREY, LIGHT, GRID = "#0b0b0b", "#52514e", "#9a9893", "#f3f2ef", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.linewidth": 1.0, "xtick.major.width": 1.0,
        "ytick.major.width": 1.0, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
        "lines.markersize": 4, "savefig.facecolor": "white", "figure.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_all(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.eps", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", f"results/figures/{name}.pdf/.eps/.png/.tif")
