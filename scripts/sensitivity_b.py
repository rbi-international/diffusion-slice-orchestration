"""Sensitivity analysis (not pre-registered): Wilcoxon signed-rank tests in place of
the pre-registered paired t-tests, same cells, same seeds, Holm within hypothesis,
same decision rule. Also writes the per-cell table used as Supplementary Table S1.

Usage:
    python scripts/sensitivity_b.py [results/study_b]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsorch.analysis_b import CELL_KEYS, HYPOTHESES, seed_differences  # noqa: E402
from dsorch.stats import holm  # noqa: E402


def main() -> None:
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "study_b"
    pts = pd.read_csv(res / "tables" / "points.csv")
    tests = pd.read_csv(res / "tables" / "cell_tests.csv")
    rows = []
    for cell, g in pts.groupby(CELL_KEYS):
        piv = g.pivot_table(index="seed", columns="method", values="miss_rate")
        for hyp in HYPOTHESES:
            d = seed_differences(piv, hyp)
            t = stats.ttest_1samp(d, 0.0)
            w = stats.wilcoxon(d)
            rows.append({"hypothesis": hyp, **dict(zip(CELL_KEYS, cell)), "n": len(d),
                         "mean_diff": d.mean(), "sd_diff": d.std(ddof=1), "t": t.statistic,
                         "p_t": t.pvalue, "p_wilcoxon": w.pvalue})
    out = pd.DataFrame(rows)
    out["p_holm_t"] = np.nan
    out["p_holm_wilcoxon"] = np.nan
    for hyp, idx in out.groupby("hypothesis").groups.items():
        out.loc[idx, "p_holm_t"] = holm(out.loc[idx, "p_t"].to_numpy())
        out.loc[idx, "p_holm_wilcoxon"] = holm(out.loc[idx, "p_wilcoxon"].to_numpy())
    chk = out.merge(tests[["hypothesis"] + CELL_KEYS + ["p_holm"]], on=["hypothesis"] + CELL_KEYS)
    assert np.allclose(chk["p_holm"], chk["p_holm_t"]), "t-test reproduction differs from analyze_b"
    out.to_csv(res / "tables" / "sensitivity_wilcoxon.csv", index=False)
    for hyp, g in out.groupby("hypothesis"):
        for col in ("p_holm_t", "p_holm_wilcoxon"):
            pred = int(((g[col] < 0.05) & (g.mean_diff < 0)).sum())
            opp = int(((g[col] < 0.05) & (g.mean_diff > 0)).sum())
            verdict = "supported" if pred >= 6 and opp == 0 else ("contradicted" if opp >= 6 else "not supported")
            print(f"{hyp} {col:16s} predicted {pred:2d} opposite {opp:2d} -> {verdict}")


if __name__ == "__main__":
    main()
