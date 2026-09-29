"""Generate manuscript figures from the result CSVs (PDF vector + 600-dpi PNG).

Colours follow a validated categorical palette assigned per controller, fixed
across every figure; each series also carries a distinct marker so identity
never depends on colour alone.
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
RES = ROOT / "results"
FIG = RES / "figures"

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # method: (colour, marker)
    "Diffusion-JCSO": ("#2a78d6", "o"),
    "GAN-JCSO": ("#eb6834", "s"),
    "QuantileMLP-JCSO": ("#1baf7a", "^"),
    "MovingAverage-JCSO": ("#eda100", "D"),
    "Persistence-JCSO": ("#e87ba4", "v"),
    "JointHeuristic": ("#008300", "P"),
    "StaticReserve": ("#4a3aa7", "X"),
    "Independent": ("#e34948", "*"),
}
FORECASTER_METHOD = {"DDPM": "Diffusion-JCSO", "GAN": "GAN-JCSO", "QuantileMLP": "QuantileMLP-JCSO",
                     "MovingAverage": "MovingAverage-JCSO", "Persistence": "Persistence-JCSO"}
LABEL = {"DDPM": "Diffusion", "GAN": "GAN", "QuantileMLP": "Quantile MLP", "MovingAverage": "Moving average",
         "Persistence": "Persistence"}
WORKLOADS = ["normal", "medium", "heavy", "extreme"]
INFO_LABEL = {"request_observed": "Request observed (Eq. 20)", "proactive": "Proactive reservation"}
MM = 1 / 25.4
DOUBLE = 183 * MM


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
        "lines.markersize": 4, "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42,
    })


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"results/figures/{name}.pdf/.png")


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="bottom", ha="right")


def fig_miss_vs_capacity() -> None:
    path = RES / "orchestration" / "orchestration.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    methods = ["Diffusion-JCSO", "GAN-JCSO", "QuantileMLP-JCSO", "MovingAverage-JCSO", "Persistence-JCSO",
               "JointHeuristic"]
    infos = [i for i in INFO_LABEL if i in set(df["information"])]
    fig, axes = plt.subplots(len(infos), 4, figsize=(DOUBLE, 2.0 * len(infos) + 0.4), sharex=True, squeeze=False)
    letters = iter("abcdefgh")
    for r, info in enumerate(infos):
        for c, w in enumerate(WORKLOADS):
            ax = axes[r, c]
            sub = df[(df["information"] == info) & (df["workload"] == w)]
            for m in methods:
                g = sub[sub["method"] == m].groupby("capacity")["miss_rate"]
                mu, sd = g.mean(), g.std(ddof=1)
                col, mk = STYLE[m]
                ax.errorbar(mu.index, mu.values, yerr=sd.values, color=col, marker=mk, capsize=1.5,
                            elinewidth=0.6, label=m, markeredgecolor="white", markeredgewidth=0.4)
            ax.set_title(f"{w.capitalize()} bursts", color=INK)
            if c == 0:
                ax.set_ylabel(f"{INFO_LABEL[info]}\nmiss rate (%)")
            if r == len(infos) - 1:
                ax.set_xlabel("Capacity multiplier")
            ax.set_xticks([0.82, 0.92, 1.0])
            ax.set_ylim(bottom=0)
            panel_letter(ax, next(letters))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig_miss_vs_capacity")


def fig_forecast_quality() -> None:
    path = RES / "forecast" / "forecast.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    order = ["DDPM", "GAN", "QuantileMLP", "MovingAverage", "Persistence"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.3))
    x = np.arange(len(WORKLOADS))
    specs = [("crps_grid", "CRPS, quantile grid (lower is better)", order, None),
             ("quantile_coverage", "Coverage of the 0.86 quantile", ["DDPM", "GAN", "QuantileMLP"], 0.86),
             ("quantile_pinball", "Pinball loss at 0.86 (lower is better)", ["DDPM", "GAN", "QuantileMLP"], None)]
    for ax, (metric, ylabel, fcs, ref), letter in zip(axes, specs, "abc"):
        width = 0.8 / len(fcs)
        for i, f in enumerate(fcs):
            g = df[df["forecaster"] == f].groupby("workload")[metric]
            mu = g.mean().reindex(WORKLOADS)
            sd = g.std(ddof=1).reindex(WORKLOADS)
            col, _ = STYLE[FORECASTER_METHOD[f]]
            ax.bar(x + (i - (len(fcs) - 1) / 2) * width, mu.values, width * 0.9, yerr=sd.values, color=col,
                   label=LABEL[f], error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": INK2})
        if ref is not None:
            ax.axhline(ref, color=INK, lw=0.8, ls="--")
            ax.text(len(WORKLOADS) - 0.5, ref, " nominal 0.86", va="bottom", ha="right", fontsize=6, color=INK2)
            ax.set_ylim(0, 1.0)
        ax.set_xticks(x, [w.capitalize() for w in WORKLOADS])
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        panel_letter(ax, letter)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig_forecast_quality")


def fig_frontier() -> None:
    path = RES / "frontier" / "frontier.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[(df["workload"] == "heavy") & (df["capacity"] == 1.0)]
    infos = [i for i in INFO_LABEL if i in set(df["information"])]
    levers = [("scale", "Envelope scale"), ("quantile", "Reservation quantile")]
    fig, axes = plt.subplots(len(levers), len(infos), figsize=(DOUBLE * 0.75, 4.4), sharey="row", squeeze=False)
    letters = iter("abcd")
    for r, (lever, lever_label) in enumerate(levers):
        for c, info in enumerate(infos):
            ax = axes[r, c]
            sub = df[(df["information"] == info) & (df["lever"] == lever)]
            for m in ["Diffusion-JCSO", "GAN-JCSO", "QuantileMLP-JCSO", "MovingAverage-JCSO"]:
                g = sub[sub["method"] == m].groupby("level")[["over_reservation", "miss_rate"]].mean()
                if g.empty:
                    continue
                col, mk = STYLE[m]
                ax.plot(g["over_reservation"], g["miss_rate"], color=col, marker=mk, label=m,
                        markeredgecolor="white", markeredgewidth=0.4)
            ax.set_title(f"{INFO_LABEL[info]}; lever: {lever_label.lower()}", color=INK)
            ax.set_xlabel("Over-reservation (% of demand)")
            if c == 0:
                ax.set_ylabel("Miss rate (%)")
            ax.set_ylim(bottom=0)
            panel_letter(ax, next(letters))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_matched_budget_frontier")


def fig_training() -> None:
    path = RES / "orchestration" / "training_loss.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["workload"] == "heavy"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.75, 2.1))
    for ax, model, cols, letter in [(axes[0], "GAN", [("d_loss", "Discriminator"), ("g_loss", "Generator")], "a"),
                                    (axes[1], "DDPM", [("loss", "Denoising MSE")], "b")]:
        sub = df[df["model"] == model]
        for (col, lab), colour in zip(cols, ["#eb6834", "#2a78d6"] if model == "GAN" else ["#2a78d6"]):
            g = sub.groupby("epoch")[col]
            mu = g.mean().rolling(7, min_periods=1).mean()
            sd = g.std(ddof=1).rolling(7, min_periods=1).mean()
            ax.plot(mu.index, mu.values, color=colour, label=lab)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=colour, alpha=0.18, lw=0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title("Conditional GAN" if model == "GAN" else "Conditional DDPM", color=INK)
        if model == "GAN":
            ax.legend(frameon=False)
        panel_letter(ax, letter)
    fig.tight_layout()
    save(fig, "fig_training_loss")


def fig_ablation() -> None:
    path = RES / "ablation" / "ablation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["information"] == "proactive"]
    variants = list(dict.fromkeys(df["variant"]))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.6), sharey=True)
    y = np.arange(len(variants))[::-1]
    for ax, (metric, label), letter in zip(axes, [("crps", "CRPS"), ("miss_rate", "Miss rate (%)"),
                                                   ("over_reservation", "Over-reservation (%)")], "abc"):
        g = df.groupby("variant")[metric]
        mu, sd = g.mean().reindex(variants), g.std(ddof=1).reindex(variants)
        colours = ["#2a78d6" if v.startswith("default") else "#86b6ef" for v in variants]
        ax.barh(y, mu.values, xerr=sd.values, color=colours, height=0.7,
                error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": INK2})
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        panel_letter(ax, letter)
    axes[0].set_yticks(y, variants)
    fig.tight_layout()
    save(fig, "fig_ablation")


def fig_b5g() -> None:
    path = RES / "b5g" / "b5g.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    seg = df.groupby(["information", "capacity", "method", "segment"])["miss_rate"].mean().reset_index()
    methods = [m for m in STYLE if m in set(seg["method"])]
    combos = [(i, c) for i in INFO_LABEL for c in sorted(seg["capacity"].unique(), reverse=True)
              if ((seg["information"] == i) & (seg["capacity"] == c)).any()]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.3))
    x = np.arange(len(combos))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        mu, sd = [], []
        for info, cap in combos:
            v = seg[(seg["information"] == info) & (seg["capacity"] == cap) & (seg["method"] == m)]["miss_rate"]
            mu.append(v.mean())
            sd.append(v.std(ddof=1))
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, mu, width * 0.9, yerr=sd, color=STYLE[m][0], label=m,
               error_kw={"elinewidth": 0.6, "capsize": 1.0, "ecolor": INK2})
    ax.set_xticks(x, [f"{INFO_LABEL[i]}\ncapacity {c:g}" for i, c in combos])
    ax.set_ylabel("Miss rate (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_b5g_replay")


def main() -> None:
    setup()
    for fn in (fig_miss_vs_capacity, fig_forecast_quality, fig_frontier, fig_training, fig_ablation, fig_b5g):
        fn()


if __name__ == "__main__":
    sys.exit(main())
