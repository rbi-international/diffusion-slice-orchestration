"""Figure 1 of the manuscript: what the study does, in four panels.

a  The orchestration problem: three slices with deadlines send bandwidth and CPU
   demand every slot (real traces from the workload generator, heavy bursts,
   test seed 1001); the controller forecasts, reserves and places each slice on
   one of four edge nodes (capacities from configs/base.yaml).
b  The six controllers: three demand models x two reservation rules.
c  How the two rules turn the same predictive samples into a reservation
   (illustrative samples; rule parameters as in the study).
d  How controllers are compared: mean frontiers of two controllers from the
   pre-registered test (results/study_b), read at 30, 40 and 50 percent
   over-reservation.

Usage:
    python scripts/make_figure1.py
Writes results/figures/fig1_overview.{pdf,eps,png,tif} (600 dpi bitmaps).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from figure_style import BLUE, GREEN, GREY, INK, INK2, LIGHT, MM, ORANGE, apply_style, save_all  # noqa: E402
from dsorch.config import load_config  # noqa: E402
from dsorch.experiments_b import frontier_points  # noqa: E402
from dsorch.system import SystemModel  # noqa: E402
from dsorch.workload import controlled_trace  # noqa: E402

SLICE_COL = {"eMBB": "#5b5b5b", "URLLC": "#5b5b5b", "mMTC": "#5b5b5b"}


def arrow(ax, x0, y0, x1, y1, lw=1.2, col=INK2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, lw=lw, color=col,
                                 shrinkA=0, shrinkB=0, transform=ax.transAxes))


def box(ax, x, y, w, h, fc="white", ec=INK2, lw=1.0, r=0.015):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                transform=ax.transAxes))


def letter(fig, ax, s):
    bb = ax.get_position()
    dx = 0.06 if s in "cd" else 0.012
    fig.text(bb.x0 - dx, bb.y1 + 0.012, s, fontsize=9, fontweight="bold", va="bottom", ha="left")


def panel_a(fig, ax, cfg):
    ax.axis("off")
    sm = SystemModel.from_config(cfg, 1.0)
    trace = controlled_trace(sm.slices, 260, cfg["workload"]["burst_factors"]["heavy"], 1001)
    t = np.arange(100, 160)
    ax.text(0.0, 1.0, "In every time slot, each slice requests bandwidth and edge CPU", fontsize=7, fontweight="bold",
            va="top", transform=ax.transAxes)
    # slices with real demand traces
    ys = [0.70, 0.43, 0.16]
    for s, y in enumerate(ys):
        spec = sm.slices[s]
        box(ax, 0.0, y - 0.02, 0.27, 0.22)
        ax.text(0.015, y + 0.17, f"{spec.name}", fontsize=7, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.26, y + 0.17, f"deadline {spec.deadline_ms:g} ms", fontsize=6, va="top", ha="right",
                color=INK2, transform=ax.transAxes)
        ins = ax.inset_axes([0.02, y + 0.005, 0.23, 0.12])
        ins.plot(t, trace[100:160, s, 0], color=INK, lw=1.0, label="bandwidth")
        ins.plot(t, trace[100:160, s, 1], color=GREY, lw=1.0, label="CPU")
        ins.set_xticks([])
        ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_visible(False)
        ins.grid(False)
        arrow(ax, 0.275, y + 0.09, 0.335, 0.47)
    ax.text(0.135, 0.075, "black: bandwidth, grey: CPU demand", fontsize=5.6, color=INK2, ha="center",
            transform=ax.transAxes)
    # controller
    box(ax, 0.335, 0.12, 0.33, 0.76, fc=LIGHT)
    ax.text(0.50, 0.855, "Controller, once per slot", fontsize=6.6, fontweight="bold", ha="center",
            va="top", transform=ax.transAxes)
    steps = [("1", "Forecast", "predictive distribution of\nnext-slot demand from the\nlast five slots"),
             ("2", "Reserve", "turn the forecast into a\nbandwidth and CPU target\nper slice (reservation rule)"),
             ("3", "Place", "assign each slice to one\nedge node, avoiding strongly\ncoupled slice pairs")]
    for i, (n, head, body) in enumerate(steps):
        y = 0.66 - i * 0.205
        ax.add_patch(plt.Circle((0.37, y + 0.06), 0.017, transform=ax.transAxes, fc=INK, ec=INK))
        ax.text(0.37, y + 0.06, n, color="white", fontsize=6, ha="center", va="center", fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.395, y + 0.09, head, fontsize=6.6, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.395, y + 0.055, body, fontsize=5.8, va="top", color=INK2, linespacing=1.25, transform=ax.transAxes)
    # edge nodes with capacities
    ax.text(0.83, 0.86, "Four edge nodes", fontsize=6.6, fontweight="bold", ha="center", va="top",
            transform=ax.transAxes)
    bw = [n.bw for n in sm.nodes]
    cpu = [n.cpu for n in sm.nodes]
    for j in range(4):
        y = 0.66 - j * 0.155
        box(ax, 0.705, y, 0.25, 0.12)
        ax.text(0.715, y + 0.06, f"node {j + 1}", fontsize=5.8, va="center", transform=ax.transAxes)
        ax.add_patch(Rectangle((0.80, y + 0.07), 0.14 * bw[j] / 132, 0.03, transform=ax.transAxes, fc=INK, lw=0))
        ax.add_patch(Rectangle((0.80, y + 0.025), 0.14 * cpu[j] / 132, 0.03, transform=ax.transAxes, fc=GREY, lw=0))
        arrow(ax, 0.655, 0.47, 0.703, y + 0.06, lw=1.0)
    ax.text(0.83, 0.03, "bars: bandwidth (black) and CPU\n(grey) capacity of each node", fontsize=5.6,
            color=INK2, ha="center", transform=ax.transAxes)
    ax.text(0.50, 0.035, "A request misses its deadline if it is\nrejected or served too late",
            fontsize=5.8, ha="center", color=INK2, transform=ax.transAxes)


def panel_b(fig, ax):
    ax.axis("off")
    ax.text(0.0, 1.0, "Six controllers: every demand model\nwith both reservation rules", fontsize=7,
            fontweight="bold", va="top", transform=ax.transAxes)
    cols = ["Published rule\n(JCSO)", "Risk-budgeted\nrule (RB, new)"]
    rows = [("Diffusion model\n(DDPM)", BLUE), ("Adversarial\nnetwork (GAN)", ORANGE), ("Quantile\nregression", GREEN)]
    x0, y0, cw, rh = 0.30, 0.30, 0.35, 0.16
    for c, name in enumerate(cols):
        ax.text(x0 + cw * (c + 0.5), y0 + rh * 3 + 0.03, name, fontsize=6.2, ha="center", va="bottom",
                fontweight="bold", transform=ax.transAxes)
    short = {0: "Diffusion", 1: "GAN", 2: "QuantileMLP"}
    for r, (name, col) in enumerate(rows):
        y = y0 + rh * (2 - r)
        ax.text(x0 - 0.02, y + rh / 2, name, fontsize=6.2, ha="right", va="center", transform=ax.transAxes)
        for c, rule in enumerate(["JCSO", "RB"]):
            ax.add_patch(Rectangle((x0 + cw * c + 0.008, y + 0.008), cw - 0.016, rh - 0.016, transform=ax.transAxes,
                                   fc="white", ec=col, lw=1.6 if rule == "RB" else 1.0,
                                   ls="-" if rule == "RB" else (0, (3, 2))))
            ax.text(x0 + cw * (c + 0.5), y + rh / 2, f"{short[r]}-{rule}", fontsize=5.6, ha="center", va="center",
                    color=INK, transform=ax.transAxes)


def panel_b_note(ax):
    ax.text(0.0, 0.17, "Experiment I: the three models with the\npublished rule, 10 seeds.\n"
            "Experiment II: all six controllers, 80 new\nseeds, hypotheses fixed in advance.",
            fontsize=6, va="top", color=INK, linespacing=1.35, transform=ax.transAxes)


def panel_c(fig, ax):
    rng = np.random.default_rng(7)
    m = 200
    bw = 40 * rng.lognormal(0, 0.18, m)
    cpu = 0.55 * bw + 12 * rng.lognormal(0, 0.25, m)
    ax.scatter(bw, cpu, s=5, color=BLUE, alpha=0.45, lw=0, label="forecast samples")
    med = np.array([np.median(bw), np.median(cpu)])
    h = np.array([np.quantile(bw, 0.95), np.quantile(cpu, 0.95)]) - med
    lam = 0.0
    while ((bw <= med[0] + lam * h[0]) & (cpu <= med[1] + lam * h[1])).mean() < 0.80:
        lam += 0.01
    r = med + lam * h
    ax.add_patch(Rectangle((bw.min() - 5, cpu.min() - 5), r[0] - bw.min() + 5, r[1] - cpu.min() + 5, fc="none",
                           ec=INK, lw=1.4))
    ax.plot(*r, "s", color=INK, ms=5)
    ax.annotate("RB: smallest target that\ncovers 80% of samples\nin both resources", xy=r, xytext=(r[0] + 3, r[1] - 16),
                fontsize=5.8, va="bottom", arrowprops=dict(arrowstyle="-", lw=1.0, color=INK))
    q86 = np.array([np.quantile(bw, 0.86), np.quantile(cpu, 0.86)])
    jcso = 1.3 * q86
    ax.plot(*jcso, "D", color=INK2, ms=4.5, mfc="white", mew=1.2)
    ax.annotate("JCSO: 0.86 quantile of each\nresource times a fixed\nslice margin", xy=jcso, xytext=(bw.min() - 3, jcso[1] + 2),
                fontsize=5.8, va="bottom", arrowprops=dict(arrowstyle="-", lw=1.0, color=INK2))
    ax.set_xlabel("Next-slot bandwidth demand")
    ax.set_ylabel("Next-slot CPU demand")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(bw.min() - 6, jcso[0] + 12)
    ax.set_ylim(cpu.min() - 6, max(jcso[1], r[1]) + 16)
    ax.set_title("Same forecast, two reservation rules", loc="left", fontsize=7, fontweight="bold", pad=4)


def panel_d(fig, ax):
    f = pd.read_csv(ROOT / "results" / "study_b" / "frontier_b.csv")
    f = f[(f.workload == "heavy") & (f.capacity == 1.0) & f.method.isin(["Diffusion-RB", "QuantileMLP-RB"])]
    grid = list(np.arange(22, 62, 1.0))
    pts = frontier_points(f, grid, ["workload", "seed", "capacity", "method"])
    for m, col, mk in [("Diffusion-RB", BLUE, "o"), ("QuantileMLP-RB", GREEN, "^")]:
        g = pts[pts.method == m].groupby("budget")["miss_rate"]
        mean = g.mean()[g.count() >= 72]
        ax.plot(mean.index, mean.values, color=col, lw=1.6, label=m)
        for b in (30, 40, 50):
            ax.plot(b, mean.loc[b], mk, color=col, ms=4.5, mec="white", mew=0.6, zorder=3)
    for b in (30, 40, 50):
        ax.axvline(b, color=GREY, lw=1.0, ls=(0, (2, 2)), zorder=0)
    ax.text(40, ax.get_ylim()[1] * 0.97, "matched budgets", fontsize=5.8, ha="center", va="top", color=INK2,
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_xlabel("Over-reservation (% of demand)")
    ax.set_ylabel("Deadline-miss rate (%)")
    ax.legend(loc="upper right", frameon=False, fontsize=5.8, bbox_to_anchor=(1.0, 0.86))
    ax.set_title("Compare miss rates at equal reserved capacity", loc="left", fontsize=7, fontweight="bold", pad=4)
    ax.text(0.98, 0.60, "heavy bursts,\nfull capacity,\n80 test seeds", transform=ax.transAxes, fontsize=5.6,
            color=INK2, ha="right", va="top")


def main() -> None:
    apply_style()
    cfg = load_config(ROOT / "configs" / "base.yaml")
    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    ax_a = fig.add_axes([0.03, 0.53, 0.60, 0.44])
    ax_b = fig.add_axes([0.66, 0.53, 0.33, 0.44])
    ax_c = fig.add_axes([0.07, 0.07, 0.38, 0.36])
    ax_d = fig.add_axes([0.58, 0.07, 0.40, 0.36])
    panel_a(fig, ax_a, cfg)
    panel_b(fig, ax_b)
    panel_b_note(ax_b)
    panel_c(fig, ax_c)
    panel_d(fig, ax_d)
    for ax, s in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        letter(fig, ax, s)
    save_all(fig, "fig1_overview")


if __name__ == "__main__":
    main()
