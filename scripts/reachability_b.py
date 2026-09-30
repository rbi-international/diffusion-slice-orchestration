"""Reachability check for study-B Amendment 1 (conditions A1 and A3).

Computes, on the validation seeds only, the requested over-reservation of
every controller at the extremes of the shared scale lever (s = 0.6 and
s = 1.8) for all 30 RB tuning candidates and the three Eq. (20) controllers,
and checks on a subset that over-reservation is monotone in s so that the
extremes bound each frontier.

Only over-reservation is written. The episodes compute outcomes internally,
but every outcome column (miss rate, acceptance, latency and so on) is
dropped by `outcome=False` before anything is stored or printed.

Usage:
    python scripts/reachability_b.py --jobs 2
Outputs (results/validation_b/):
    reachability_endpoints.csv   over-reservation at s = 0.6 and s = 1.8
    reachability_fullgrid.csv    all 21 values of s, seed 1, heavy bursts
    reachability_summary.csv     seed-conditions (of 20) reaching each budget
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsorch.config import load_config  # noqa: E402
from dsorch.experiments import apply_selected, run_parallel  # noqa: E402
from dsorch.experiments_b import (  # noqa: E402
    MODELS, REACH_COLUMNS, jcso_frontier, prepare, rb_candidates, rb_frontier, scale_grid, study_b_config,
)
from dsorch.repro import write_manifest  # noqa: E402

OUT = ROOT / "results" / "validation_b"


def _job(cfg, workload: str, seed: int, full: bool):
    prep = prepare(cfg, workload, seed)
    prep["seed"] = seed
    sb = cfg["study_b"]
    grid = scale_grid(sb)
    scales = grid if full else [grid[0], grid[-1]]
    rows = []
    for name in MODELS:
        for c, shape in rb_candidates(sb):
            rows += rb_frontier(cfg, prep, name, shape, c, scales, sb["capacity"], outcome=False)
    jc = jcso_frontier(cfg, prep, sb["capacity"], outcome=False)
    rows += [r for r in jc if full or r["lever"] in (grid[0], grid[-1])]
    for r in rows:
        assert set(r) == set(REACH_COLUMNS), "outcome columns must not be stored"
        r.update(workload=workload, seed=seed)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    cfg = study_b_config(apply_selected(load_config(ROOT / "configs" / "validation_b.yaml"), ROOT))
    sb = cfg["study_b"]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(cfg, w, s, False) for w in sb["workloads"] for s in sb["validation_seeds"]]
    jobs.append((cfg, "heavy", sb["validation_seeds"][0], True))
    res = run_parallel(_job, jobs, args.jobs)
    ends = pd.DataFrame(sum(res[:-1], []))
    full = pd.DataFrame(res[-1])
    ends.to_csv(OUT / "reachability_endpoints.csv", index=False)
    full.to_csv(OUT / "reachability_fullgrid.csv", index=False)

    keys = ["method", "candidate"]
    rng = ends.groupby(["workload", "seed", "capacity"] + keys, dropna=False)["over_reservation"].agg(["min", "max"])
    rows = []
    for (method, cand), g in rng.groupby(level=keys, dropna=False):
        row = {"method": method, "candidate": cand, "seed_conditions": len(g)}
        for b in sb["budgets"]:
            row[f"reach_{b}"] = int(((g["min"] <= b) & (g["max"] >= b)).sum())
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "reachability_summary.csv", index=False)

    # monotonicity of over-reservation in s on the full grid
    mono = []
    for k, g in full.groupby(["capacity"] + keys, dropna=False):
        g = g.sort_values("lever")
        mono.append(bool((g["over_reservation"].diff().dropna() >= -1e-9).all()))
    budgets = [f"reach_{b}" for b in sb["budgets"]]
    per_method = summ.groupby("method")[budgets].min()
    print("minimum over candidates of seed-conditions (of 20) reaching each budget:")
    print(per_method.to_string())
    print(f"frontiers monotone in s on the full grid: {sum(mono)} of {len(mono)}")
    distinct = {}
    for c, b in rb_candidates(sb):
        import numpy as np
        from dsorch.reservation import clip_eps
        distinct[tuple(np.round(clip_eps(c * np.array(b)), 12))] = True
    print(f"distinct clipped base-level vectors among {len(rb_candidates(sb))} candidates: {len(distinct)}")
    write_manifest(OUT / "reachability_manifest", cfg, {
        "experiment": "reachability_b", "outputs": ["over_reservation only"],
        "stored_columns": REACH_COLUMNS + ["workload", "seed"],
    })


if __name__ == "__main__":
    main()
