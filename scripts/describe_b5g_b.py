"""Descriptive (post hoc) view of the public-trace replay beyond the pre-specified budgets,
and Figure 5 of the manuscript.

The pre-specified budgets (30, 40, 50 percent over-reservation) lie in a region
where every controller misses most deadlines on the public trace. This script
reports mean miss rates at larger over-reservation (60 to 200 percent) and draws
the full mean frontiers. These budgets were chosen after the results were seen,
so no tests are attached to them; they are descriptive only.

Usage:
    python scripts/describe_b5g_b.py [results/b5g_b]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dsorch.experiments_b import frontier_points  # noqa: E402
from figure_style import BLUE, GREEN, GRID, MM, ORANGE, apply_style, save_all  # noqa: E402

HUE = {"Diffusion": BLUE, "GAN": ORANGE, "QuantileMLP": GREEN}
MARK = {"Diffusion": "o", "GAN": "s", "QuantileMLP": "^"}
POST_HOC = [60, 80, 100, 150, 200]


def main() -> None:
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "b5g_b"
    f = pd.read_csv(res / "frontier_b5g_b.csv")
    keys = ["workload", "seed", "capacity", "method"]
    n_pairs = f["seed"].nunique()

    pts = frontier_points(f, POST_HOC, keys)
    t = pts.groupby(["capacity", "budget", "method"])["miss_rate"].agg(["mean", "std", "count"]).reset_index()
    t.to_csv(res / "tables" / "descriptive_budgets.csv", index=False)
    print(t.pivot_table(index=["capacity", "budget"], columns="method", values="mean").round(1).to_string())

    apply_style()
    grid = list(np.arange(20, 305, 5.0))
    fp = frontier_points(f, grid, keys)
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM), sharey=True)
    for ax, cap, letter in zip(axes, [1.0, 0.82], "ab"):
        sub = fp[fp.capacity == cap]
        ax.axvspan(30, 50, color=GRID, alpha=0.6, lw=0, zorder=0)
        for model in ["Diffusion", "GAN", "QuantileMLP"]:
            for rule, ls, fill in [("RB", "-", True), ("JCSO", "--", False)]:
                g = sub[sub.method == f"{model}-{rule}"].groupby("budget")["miss_rate"]
                mean = g.mean()[g.count() >= int(np.ceil(0.9 * n_pairs))]
                ax.plot(mean.index, mean.values, ls=ls, color=HUE[model], lw=1.5, marker=MARK[model], markevery=6,
                        ms=3.5, mfc=HUE[model] if fill else "white", mec=HUE[model], mew=1.0, label=f"{model}-{rule}")
        ax.set_title(f"Public trace, capacity {cap:.2f}", pad=5)
        ax.set_xlabel("Over-reservation (%)")
        ax.set_xlim(20, 300)
        ax.set_ylim(0, 100)
        ax.text(-0.1, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=8.5)
        ax.annotate("pre-specified budgets\n(30 to 50%)", xy=(50, 8), xytext=(58, 4), fontsize=5.8, color="#52514e")
    axes[0].set_ylabel("Deadline-miss rate (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save_all(fig, "fig_b5g_b_frontiers")


if __name__ == "__main__":
    main()
