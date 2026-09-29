"""Aggregate raw experiment CSVs into summary tables and paired statistical tests.

Reads results/<experiment>/*.csv and writes results/tables/*.csv plus a
human-readable results/SUMMARY.md. Every number in the manuscript is taken
from these tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsorch.stats import compare_against, holm, paired_comparison  # noqa: E402

RES = ROOT / "results"
TAB = RES / "tables"
FOCAL = "Diffusion-JCSO"
JCSO_COMPARATORS = ["GAN-JCSO", "QuantileMLP-JCSO", "MovingAverage-JCSO", "Persistence-JCSO"]
ALL_COMPARATORS = JCSO_COMPARATORS + ["JointHeuristic", "StaticReserve", "Independent"]
METHOD_ORDER = ["Diffusion-JCSO", "GAN-JCSO", "QuantileMLP-JCSO", "MovingAverage-JCSO", "Persistence-JCSO",
                "JointHeuristic", "StaticReserve", "Independent"]
WORKLOAD_ORDER = ["normal", "medium", "heavy", "extreme"]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def mean_sd(df, by, metrics):
    g = df.groupby(by)[metrics]
    m = g.mean().add_suffix("_mean")
    s = g.std(ddof=1).add_suffix("_sd")
    return pd.concat([m, s], axis=1).reset_index()


def order_methods(df):
    df = df.copy()
    if "method" in df:
        df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    if "workload" in df:
        df["workload"] = pd.Categorical(df["workload"], WORKLOAD_ORDER, ordered=True)
    return df


def analyze_orchestration(md: list) -> None:
    path = RES / "orchestration" / "orchestration.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metrics = ["miss_rate", "acceptance", "mean_latency_ms", "p95_latency_ms", "p99_latency_ms",
               "isolation", "utilization", "fragmentation", "over_reservation"]
    summ = order_methods(mean_sd(df, ["information", "workload", "capacity", "method"], metrics))
    summ = summ.sort_values(["information", "workload", "capacity", "method"])
    summ.to_csv(TAB / "orchestration_summary.csv", index=False)
    tests = []
    for metric in ["miss_rate", "acceptance", "p95_latency_ms"]:
        tests.append(compare_against(df, FOCAL, ALL_COMPARATORS, metric, ["information", "workload", "capacity"]))
    tests = pd.concat(tests)
    tests.to_csv(TAB / "orchestration_tests.csv", index=False)

    md.append("## Main orchestration grid (10 seeds, held-out slots 100-259)\n")
    for info in ["request_observed", "proactive"]:
        sub = summ[(summ["information"] == info)]
        if sub.empty:
            continue
        piv = sub.pivot_table(index=["workload", "capacity"], columns="method", values="miss_rate_mean",
                              observed=True).reset_index()
        md.append(f"### Miss rate (%), information = {info}\n")
        md.append(md_table(piv) + "\n")
        piv = sub.pivot_table(index=["workload", "capacity"], columns="method", values="over_reservation_mean",
                              observed=True).reset_index()
        md.append(f"### Over-reservation (%), information = {info}\n")
        md.append(md_table(piv, "{:.1f}") + "\n")
    t = tests[(tests["metric"] == "miss_rate") & tests["comparator"].isin(JCSO_COMPARATORS)]
    t = t.assign(p_holm=t["p_holm"].map(fmt_p), p_t=t["p_t"].map(fmt_p))
    md.append("### Diffusion-JCSO minus comparator, miss rate (percentage points), Holm over the family\n")
    md.append(md_table(t[["information", "workload", "capacity", "comparator", "mean_diff", "ci_low",
                          "ci_high", "p_t", "p_holm"]]) + "\n")


def analyze_forecast(md: list) -> None:
    path = RES / "forecast" / "forecast.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metrics = ["crps", "crps_grid", "energy_score", "nmae", "rmse", "quantile_pinball", "quantile_coverage",
               "quantile_joint_coverage", "estimate_pinball", "estimate_coverage", "sample_spread",
               "ms_per_prediction", "fit_seconds", "params_online", "params_training_only"]
    summ = mean_sd(df, ["workload", "forecaster"], metrics)
    summ["workload"] = pd.Categorical(summ["workload"], WORKLOAD_ORDER, ordered=True)
    summ = summ.sort_values(["workload", "forecaster"])
    summ.to_csv(TAB / "forecast_summary.csv", index=False)
    # paired tests DDPM vs GAN and QuantileMLP on CRPS and pinball
    rows = []
    for metric in ["crps", "crps_grid", "quantile_pinball", "quantile_coverage"]:
        for w, g in df.groupby("workload"):
            piv = g.pivot_table(index="seed", columns="forecaster", values=metric)
            for comp in ["GAN", "QuantileMLP", "MovingAverage", "Persistence"]:
                r = paired_comparison(piv["DDPM"].to_numpy(), piv[comp].to_numpy())
                rows.append({"metric": metric, "workload": w, "comparator": comp, **r})
    tests = pd.DataFrame(rows)
    tests["p_holm"] = np.nan
    for metric, idx in tests.groupby("metric").groups.items():
        tests.loc[idx, "p_holm"] = holm(tests.loc[idx, "p_t"].fillna(1.0))
    tests.to_csv(TAB / "forecast_tests.csv", index=False)
    md.append("## Forecast quality on held-out slots\n")
    show = summ[["workload", "forecaster", "crps_mean", "crps_grid_mean", "quantile_pinball_mean", "quantile_coverage_mean",
                 "quantile_joint_coverage_mean", "nmae_mean", "sample_spread_mean", "ms_per_prediction_mean",
                 "params_online_mean"]]
    md.append(md_table(show, "{:.3f}") + "\n")


def analyze_frontier(md: list) -> None:
    path = RES / "frontier" / "matched_budget.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    summ = mean_sd(df, ["information", "workload", "capacity", "lever", "budget", "method"],
                   ["miss_rate", "acceptance", "p95_latency_ms", "p99_latency_ms"])
    summ.to_csv(TAB / "matched_budget_summary.csv", index=False)
    tests = []
    for lever in ["scale", "quantile"]:
        sub = df[df["lever"] == lever]
        comps = ["GAN-JCSO", "QuantileMLP-JCSO"] + (["MovingAverage-JCSO"] if lever == "scale" else [])
        t = compare_against(sub, FOCAL, comps, "miss_rate", ["information", "workload", "capacity", "budget"])
        t["lever"] = lever
        tests.append(t)
    tests = pd.concat(tests)
    tests.to_csv(TAB / "matched_budget_tests.csv", index=False)
    md.append("## Matched over-reservation budgets (miss rate %, mean over 10 seeds)\n")
    for lever in ["scale", "quantile"]:
        sub = summ[summ["lever"] == lever]
        piv = sub.pivot_table(index=["information", "workload", "capacity", "budget"], columns="method",
                              values="miss_rate_mean").reset_index()
        md.append(f"### Lever = {lever}\n")
        md.append(md_table(piv) + "\n")
    t = tests.assign(p_holm=tests["p_holm"].map(fmt_p))
    md.append("### Diffusion-JCSO minus comparator at matched budget (miss rate, pp)\n")
    md.append(md_table(t[["lever", "information", "workload", "capacity", "budget", "comparator", "n",
                          "mean_diff", "ci_low", "ci_high", "p_holm"]]) + "\n")


def analyze_ablation(md: list) -> None:
    path = RES / "ablation" / "ablation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metrics = ["crps", "quantile_coverage", "quantile_pinball", "miss_rate", "acceptance", "p95_latency_ms",
               "over_reservation", "ms_per_prediction", "fit_seconds", "terminal_alpha_bar"]
    summ = mean_sd(df, ["group", "variant", "information", "capacity"], metrics)
    summ.to_csv(TAB / "ablation_summary.csv", index=False)
    md.append("## Diffusion ablations (heavy bursts, nominal capacity)\n")
    show = summ[["variant", "information", "terminal_alpha_bar_mean", "crps_mean", "quantile_coverage_mean",
                 "miss_rate_mean", "over_reservation_mean", "ms_per_prediction_mean"]]
    md.append(md_table(show, "{:.3f}") + "\n")


def analyze_b5g(md: list) -> None:
    path = RES / "b5g" / "b5g.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metrics = ["miss_rate", "acceptance", "p95_latency_ms", "isolation", "over_reservation"]
    seg = df.groupby(["information", "capacity", "method", "segment"])[metrics].mean().reset_index()
    summ = order_methods(mean_sd(seg, ["information", "capacity", "method"], metrics))
    summ = summ.sort_values(["information", "capacity", "method"])
    summ.to_csv(TAB / "b5g_summary.csv", index=False)
    seg = seg.rename(columns={"segment": "seed"})
    tests = compare_against(seg, FOCAL, ALL_COMPARATORS, "miss_rate", ["information", "capacity"])
    tests.to_csv(TAB / "b5g_tests.csv", index=False)
    md.append("## Public B5G trace replay (3 segments x 5 model seeds; tests over segments, n = 3, exploratory)\n")
    piv = summ.pivot_table(index=["information", "capacity"], columns="method", values="miss_rate_mean",
                           observed=True).reset_index()
    md.append(md_table(piv) + "\n")


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    md = ["# Results summary\n", "Generated by `scripts/analyze.py` from the CSV files in `results/`. "
          "Do not edit by hand.\n"]
    sel = RES / "validation" / "selected.yaml"
    if sel.exists():
        md.append("## Validation-selected training lengths\n")
        md.append("```\n" + sel.read_text() + "```\n")
    for fn in (analyze_orchestration, analyze_forecast, analyze_frontier, analyze_ablation, analyze_b5g):
        fn(md)
    (RES / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {TAB.relative_to(ROOT)}/ and results/SUMMARY.md")


if __name__ == "__main__":
    main()
