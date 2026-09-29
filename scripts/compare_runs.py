"""Compare two runs of the orchestration experiment (for example Linux vs Windows).

Usage:
    python scripts/compare_runs.py results/orchestration results/verify/orchestration

Reports (1) the rows that differ most, (2) how far the per-scenario means move,
and (3) whether the paper's comparative statements agree between the runs:
the sign of the Diffusion-JCSO minus comparator mean difference in miss rate
and in over-reservation for every scenario.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["workload", "seed", "information", "capacity", "method"]
SCEN = ["information", "workload", "capacity"]
FOCAL = "Diffusion-JCSO"
COMPS = ["GAN-JCSO", "QuantileMLP-JCSO", "MovingAverage-JCSO", "Persistence-JCSO", "JointHeuristic"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--csv", default="orchestration.csv")
    args = ap.parse_args()
    a = pd.read_csv(args.run_a / args.csv)
    b = pd.read_csv(args.run_b / args.csv)
    m = a.merge(b, on=KEYS, suffixes=("_a", "_b"))
    print(f"rows: A={len(a)}  B={len(b)}  matched={len(m)}")

    metrics = ["miss_rate", "acceptance", "p95_latency_ms", "over_reservation"]
    m["d_miss"] = (m["miss_rate_a"] - m["miss_rate_b"]).abs()
    m["d_or"] = (m["over_reservation_a"] - m["over_reservation_b"]).abs()
    identical = (m["d_miss"] == 0) & (m["d_or"] == 0)
    print(f"rows with identical miss rate and over-reservation: {identical.sum()} of {len(m)}")
    print("\nrows that differ, by method:")
    print(m.assign(differs=~identical).groupby("method")["differs"].sum().to_string())

    print("\nlargest row-level differences (miss rate):")
    top = m.sort_values("d_miss", ascending=False).head(10)
    print(top[KEYS + ["miss_rate_a", "miss_rate_b", "over_reservation_a", "over_reservation_b"]]
          .round(2).to_string(index=False))

    print("\nper-scenario means over seeds, largest absolute change:")
    ga = a.groupby(SCEN + ["method"])[metrics].mean()
    gb = b.groupby(SCEN + ["method"])[metrics].mean()
    diff = (ga - gb).abs()
    print(diff.max().round(3).to_string())

    print("\nsign agreement of Diffusion-JCSO minus comparator (scenario means):")
    rows = []
    for metric in ["miss_rate", "over_reservation"]:
        for comp in COMPS:
            da = ga[metric].xs(FOCAL, level="method") - ga[metric].xs(comp, level="method")
            db = gb[metric].xs(FOCAL, level="method") - gb[metric].xs(comp, level="method")
            agree = (np.sign(da.round(6)) == np.sign(db.round(6))).mean()
            rows.append({"metric": metric, "comparator": comp, "scenarios": len(da),
                         "sign_agreement": round(float(agree), 3),
                         "mean_diff_A": round(float(da.mean()), 3), "mean_diff_B": round(float(db.mean()), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
