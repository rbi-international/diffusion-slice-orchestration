"""Paired statistical comparisons with Holm correction."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvalues: Iterable[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    p = np.asarray(list(pvalues), dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided paired t-test and Wilcoxon signed-rank test of a - b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1) if n > 1 else np.nan
    if n > 1 and sd > 0:
        t = stats.ttest_rel(a, b)
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        p_t = float(t.pvalue)
    else:
        half, p_t = np.nan, (1.0 if n > 1 else np.nan)
    try:
        p_w = float(stats.wilcoxon(a, b).pvalue) if n > 1 and np.any(d != 0) else 1.0
    except ValueError:
        p_w = np.nan
    return {
        "n": n,
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "mean_diff": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "cohen_dz": mean / sd if sd and sd > 0 else np.nan,
        "p_t": p_t,
        "p_wilcoxon": p_w,
    }


def compare_against(
    df: pd.DataFrame,
    focal: str,
    comparators: List[str],
    metric: str,
    by: List[str],
    pair_on: str = "seed",
) -> pd.DataFrame:
    """Compare ``focal`` with each comparator per scenario, Holm-adjusted over the whole family."""
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        piv = g.pivot_table(index=pair_on, columns="method", values=metric)
        for comp in comparators:
            if focal not in piv or comp not in piv:
                continue
            sub = piv[[focal, comp]].dropna()
            res = paired_comparison(sub[focal].to_numpy(), sub[comp].to_numpy())
            rows.append({**dict(zip(by, key)), "metric": metric, "focal": focal, "comparator": comp, **res})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_t"].fillna(1.0))
    return out
