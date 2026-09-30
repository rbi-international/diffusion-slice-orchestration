"""Figures for study B and the manuscript schematic.

Usage:
    python scripts/make_figures_b.py [results/study_b]
Reads frontier_b.csv and tables/cell_tests.csv; writes results/figures/fig_b_*.pdf/.png
and fig_pipeline.pdf/.png. Colours follow scripts/make_figures.py: hue = forecaster,
line style and marker fill = reservation rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_figures import DOUBLE, GRID, INK, INK2, MM, save, setup  # noqa: E402
from dsorch.experiments_b import frontier_points  # noqa: E402

HUE = {"Diffusion": "#2a78d6", "GAN": "#eb6834", "QuantileMLP": "#1baf7a"}
MARK = {"Diffusion": "o", "GAN": "s", "QuantileMLP": "^"}
CONDS = [("heavy", 1.0), ("heavy", 0.82), ("extreme", 1.0), ("extreme", 0.82)]
def panel_letter(ax, letter: str) -> None:
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="bottom", ha="left")


BURST = {"heavy": "heavy bursts", "extreme": "extreme bursts"}


def fig_frontiers(frontier: pd.DataFrame) -> None:
    grid = list(np.arange(24, 62, 2.0))
    pts = frontier_points(frontier, grid, ["workload", "seed", "capacity", "method"])
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 52 * MM), sharey=False)
    for ax, (w, cap), letter in zip(axes, CONDS, "abcd"):
        sub = pts[(pts.workload == w) & (pts.capacity == cap)]
        ax.axvspan(30, 50, color=GRID, alpha=0.45, lw=0, zorder=0)
        for model in ["Diffusion", "GAN", "QuantileMLP"]:
            for rule, ls, fill in [("RB", "-", True), ("JCSO", "--", False)]:
                m = f"{model}-{rule}"
                g = sub[sub.method == m].groupby("budget")["miss_rate"]
                mean, n = g.mean(), g.count()
                mean = mean[n >= 72]                      # same availability rule as the analysis
                ax.plot(mean.index, mean.values, ls=ls, color=HUE[model], lw=1.2,
                        marker=MARK[model], markevery=3, ms=3.5,
                        mfc=HUE[model] if fill else "white", mec=HUE[model], mew=0.8,
                        label=f"{model}-{rule}")
        ax.set_title(f"{BURST[w].capitalize()}, capacity {cap:.2f}", pad=6, fontsize=6.8)
        ax.set_xlabel("Over-reservation (%)")
        ax.set_xlim(24, 60)
        panel_letter(ax, letter)
    axes[0].set_ylabel("Deadline-miss rate (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_b_frontiers")


LABEL = {"H1": "H1: Diffusion-RB\nminus GAN-RB",
         "H2": "H2: Diffusion-RB\nminus QuantileMLP-RB",
         "H3": "H3: Diffusion-RB\nminus Diffusion-JCSO",
         "H4": "H4: difference in\ndifferences"}


def fig_forest(tests: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 70 * MM), sharey=True)
    order = [(w, c, b) for (w, c) in CONDS for b in (30, 40, 50)]
    ylab = [f"{w}, {c:.2f}, {b}%" for w, c, b in order]
    y = np.arange(len(order))[::-1]
    for ax, hyp, letter in zip(axes, ["H1", "H2", "H3", "H4"], "abcd"):
        t = tests[tests.hypothesis == hyp].set_index(["workload", "capacity", "budget"])
        for yi, key in zip(y, order):
            r = t.loc[key]
            half = 1.96 * r["sd_diff"] / np.sqrt(r["n"])
            sig = r["p_holm"] < 0.05
            col = INK if sig else INK2
            ax.plot([r["mean_diff"] - half, r["mean_diff"] + half], [yi, yi], color=col, lw=1.0)
            ax.plot(r["mean_diff"], yi, "o", ms=4, mfc=col if sig else "white", mec=col, mew=0.8)
        ax.axvline(0, color=INK2, lw=0.7)
        for yy in (2.5, 5.5, 8.5):
            ax.axhline(yy, color=GRID, lw=0.6)
        ax.set_title(LABEL[hyp], pad=4, loc="left", fontsize=6.8)
        ax.set_xlabel("Miss-rate difference (pp)")
        panel_letter(ax, letter)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ylab)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_b_cell_differences")


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE, 80 * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, title, body):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc="white", ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=6.6, fontweight="bold")
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=5.8, color=INK2, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                     color=INK2, shrinkA=0, shrinkB=0))

    box(0.5, 23, 16, 17, "Demand history", "five slots of\nbandwidth and CPU\ndemand per slice,\nplus a slice indicator")
    box(21, 31, 21, 14, "Generative samplers", "conditional DDPM or\nconditional GAN,\n200 samples per decision")
    box(21, 12, 21, 14, "Quantile regression", "QuantileMLP trained\nwith the pinball loss,\n15 quantile levels")
    box(48, 31, 23, 14, "Eq. (20) rule (JCSO)", "0.86 quantile blended\nwith an LWMA anchor,\nfixed slice margins")
    box(48, 12, 23, 14, "Risk-budgeted rule (RB)", "joint-coverage reservation\nat a recalibrated\nper-slice risk level")
    box(77, 23, 22.5, 22, "Joint placement", "queue correction, then\nbandwidth and CPU on one\nedge node per slice with\nisolation-aware ranking;\nRB relaxes the slice with\nthe least added shortfall\nwhen capacity is short")
    box(77, 2, 22.5, 16, "Outcome", "deadline-miss rate at\nmatched over-reservation\nof 30, 40 and 50 percent")
    arrow(17.3, 34, 20.4, 38)
    arrow(17.3, 29, 20.4, 19)
    for y0 in (38, 19):
        for y1 in (38, 19):
            arrow(42.7, y0, 47.4, y1)
    arrow(71.7, 38, 76.4, 38)
    arrow(71.7, 19, 76.4, 28)
    arrow(88.25, 22.4, 88.25, 18.6)
    ax.text(1, 6, "Every forecaster is paired with both rules (six controllers).\n"
            "Each controller's target is scaled over 21 values from 0.6 to 1.8\n"
            "to trace a miss-rate versus over-reservation frontier.",
            ha="left", va="center", fontsize=5.8, color=INK2, linespacing=1.4)
    save(fig, "fig_pipeline")


def main() -> None:
    setup()
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "study_b"
    frontier = pd.read_csv(res / "frontier_b.csv")
    tests = pd.read_csv(res / "tables" / "cell_tests.csv")
    fig_frontiers(frontier)
    fig_forest(tests)
    fig_pipeline()


if __name__ == "__main__":
    main()
