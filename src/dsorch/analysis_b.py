"""Pre-specified analysis of study B (docs/PREREGISTRATION_B.md v3.1 sections 5 and 6,
with Amendment 1). Written and tested before any test-seed result existed.

Input: the frontier table written by `exp_test_b` (one row per workload, seed,
controller, lever value and capacity). Output: per-cell tests, hypothesis
verdicts and secondary tables.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from .experiments_b import frontier_points
from .stats import holm

BUDGETS = [30, 40, 50]
CELL_KEYS = ["workload", "capacity", "budget"]

# hypothesis -> (controllers involved, how the per-seed difference is formed)
HYPOTHESES = {
    "H1": {"label": "Diffusion-RB < GAN-RB", "controllers": ["Diffusion-RB", "GAN-RB"]},
    "H2": {"label": "Diffusion-RB < QuantileMLP-RB", "controllers": ["Diffusion-RB", "QuantileMLP-RB"]},
    "H3": {"label": "Diffusion-RB < Diffusion-JCSO", "controllers": ["Diffusion-RB", "Diffusion-JCSO"]},
    "H4": {"label": "(Diffusion-RB - Diffusion-JCSO) - (GAN-RB - GAN-JCSO) < 0",
           "controllers": ["Diffusion-RB", "Diffusion-JCSO", "GAN-RB", "GAN-JCSO"]},
}


def seed_differences(piv: pd.DataFrame, hyp: str) -> pd.Series:
    """Per-seed difference whose predicted sign is negative, on seeds that have every controller."""
    need = HYPOTHESES[hyp]["controllers"]
    sub = piv.reindex(columns=need).dropna()
    if hyp == "H4":
        return (sub["Diffusion-RB"] - sub["Diffusion-JCSO"]) - (sub["GAN-RB"] - sub["GAN-JCSO"])
    return sub[need[0]] - sub[need[1]]


def cell_tests(points: pd.DataFrame, n_seeds: int, min_frac: float = 0.9, metric: str = "miss_rate") -> pd.DataFrame:
    """One row per hypothesis x cell: n, mean difference, two-sided p, analysed flag."""
    min_n = int(np.ceil(min_frac * n_seeds))
    rows = []
    for cell, g in points.groupby(CELL_KEYS):
        piv = g.pivot_table(index="seed", columns="method", values=metric)
        for hyp in HYPOTHESES:
            d = seed_differences(piv, hyp)
            n = len(d)
            analysed = n >= min_n
            if analysed and n > 1 and d.std(ddof=1) > 0:
                p = float(stats.ttest_1samp(d, 0.0).pvalue)     # paired test == one-sample test on differences
            elif analysed:
                p = 1.0 if (d == 0).all() else 0.0
            else:
                p = np.nan
            rows.append({"hypothesis": hyp, **dict(zip(CELL_KEYS, cell)), "n": n, "min_n": min_n,
                         "analysed": analysed, "mean_diff": float(d.mean()) if n else np.nan,
                         "sd_diff": float(d.std(ddof=1)) if n > 1 else np.nan, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for hyp, idx in out.groupby("hypothesis").groups.items():
        sel = out.loc[idx]
        ok = sel["analysed"]
        if ok.any():
            out.loc[sel.index[ok], "p_holm"] = holm(sel.loc[ok, "p"].to_numpy())
    out["sig_predicted"] = out["analysed"] & (out["p_holm"] < 0.05) & (out["mean_diff"] < 0)
    out["sig_opposite"] = out["analysed"] & (out["p_holm"] < 0.05) & (out["mean_diff"] > 0)
    return out


def verdicts(tests: pd.DataFrame, points: pd.DataFrame, n_boot: int = 10_000, seed: int = 20260930,
             metric: str = "miss_rate") -> pd.DataFrame:
    """Section 6 decision rule plus the seed-level bootstrap interval of the mean difference."""
    rng = np.random.default_rng(seed)
    rows = []
    for hyp, t in tests.groupby("hypothesis"):
        n_pred, n_opp = int(t["sig_predicted"].sum()), int(t["sig_opposite"].sum())
        if n_pred >= 6 and n_opp == 0:
            verdict = "supported"
        elif n_opp >= 6:
            verdict = "contradicted"
        else:
            verdict = "not supported"
        # seed-level bootstrap over analysed cells: per-seed mean of the available cell differences
        per_seed = []
        for _, r in t[t["analysed"]].iterrows():
            g = points[(points["workload"] == r["workload"]) & (points["capacity"] == r["capacity"])
                       & (points["budget"] == r["budget"])]
            d = seed_differences(g.pivot_table(index="seed", columns="method", values=metric), hyp)
            per_seed.append(d.rename(f"{r['workload']}|{r['capacity']}|{r['budget']}"))
        if per_seed:
            mat = pd.concat(per_seed, axis=1)
            seed_means = mat.mean(axis=1).to_numpy()
            boots = np.array([rng.choice(seed_means, len(seed_means)).mean() for _ in range(n_boot)])
            mean, lo, hi = float(seed_means.mean()), *np.percentile(boots, [2.5, 97.5])
        else:
            mean = lo = hi = np.nan
        rows.append({"hypothesis": hyp, "claim": HYPOTHESES[hyp]["label"], "cells_analysed": int(t["analysed"].sum()),
                     "cells_excluded": int((~t["analysed"]).sum()), "cells_sig_predicted": n_pred,
                     "cells_sig_opposite": n_opp, "verdict": verdict,
                     "mean_diff_pp": mean, "boot_ci_low": float(lo), "boot_ci_high": float(hi)})
    return pd.DataFrame(rows)


SECONDARY = ["acceptance", "p95_latency_ms", "p99_latency_ms", "utilization", "realized_reservation_ratio"]


def analyse(frontier: pd.DataFrame, n_seeds: int) -> Dict[str, pd.DataFrame]:
    points = frontier_points(frontier, BUDGETS, ["workload", "seed", "capacity", "method"])
    tests = cell_tests(points, n_seeds)
    ver = verdicts(tests, points)
    cell_means = points.groupby(CELL_KEYS + ["method"])[["miss_rate"] + SECONDARY].mean().reset_index()
    reach = points.assign(reached=points["miss_rate"].notna()).groupby(CELL_KEYS + ["method"])["reached"].sum()
    diag_cols = [c for c in ["g", "frac_clip_low", "frac_clip_high", "frac_lambda_capped", "mean_relax_steps",
                             "realized_joint_coverage", "ms_per_prediction"] if c in frontier]
    diag = frontier.groupby(["workload", "capacity", "method"])[diag_cols].mean().reset_index() if diag_cols else None
    return {"points": points, "cell_tests": tests, "verdicts": ver, "cell_means": cell_means,
            "reach": reach.reset_index(), "diagnostics": diag}
