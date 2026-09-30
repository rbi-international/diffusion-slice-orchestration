"""Exploratory (not pre-registered) study-B contrasts, reported as such in the manuscript.

Usage:
    python scripts/exploratory_b.py [results/study_b]
Reads tables/points.csv (written by analyze_b.py) and writes
tables/exploratory_contrasts.csv: per cell paired t-tests over seeds with Holm
correction within each contrast (12 cells), same missing-seed handling as the
pre-registered analysis (seeds with both controllers at the budget).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsorch.stats import holm  # noqa: E402

CONTRASTS = [
    ("QuantileMLP-RB", "QuantileMLP-JCSO"),
    ("GAN-RB", "GAN-JCSO"),
    ("QuantileMLP-RB", "GAN-RB"),
    ("QuantileMLP-JCSO", "Diffusion-JCSO"),
    ("GAN-JCSO", "Diffusion-JCSO"),
]


def main() -> None:
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "study_b"
    pts = pd.read_csv(res / "tables" / "points.csv")
    rows = []
    for a, b in CONTRASTS:
        block = []
        for (w, cap, bud), g in pts.groupby(["workload", "capacity", "budget"]):
            piv = g.pivot_table(index="seed", columns="method", values="miss_rate")[[a, b]].dropna()
            d = piv[a] - piv[b]
            block.append({"contrast": f"{a} minus {b}", "workload": w, "capacity": cap, "budget": bud,
                          "n": len(d), "mean_diff": d.mean(), "sd_diff": d.std(ddof=1),
                          "p": stats.ttest_1samp(d, 0.0).pvalue})
        ph = holm(np.array([r["p"] for r in block]))
        for r, q in zip(block, ph):
            r["p_holm"] = q
        rows += block
    out = pd.DataFrame(rows)
    out.to_csv(res / "tables" / "exploratory_contrasts.csv", index=False)
    summ = out.groupby("contrast").agg(mean_diff=("mean_diff", "mean"), min_diff=("mean_diff", "min"),
                                       max_diff=("mean_diff", "max"),
                                       sig_negative=("p_holm", lambda s: int(((s < 0.05) & (out.loc[s.index, "mean_diff"] < 0)).sum())),
                                       sig_positive=("p_holm", lambda s: int(((s < 0.05) & (out.loc[s.index, "mean_diff"] > 0)).sum())))
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
